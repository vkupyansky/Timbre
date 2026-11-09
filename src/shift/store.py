"""SQLite queue. Call these synchronous methods with asyncio.to_thread in async code."""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from unicodedata import category

from .protocol import (
    JOB_TIMEOUT_SECONDS,
    LEASE_SECONDS,
    LEGACY_DELIVERY_TAGS,
    MAX_PROFILE_NAME,
    MAX_TEXT,
    MAX_WAITING,
    PREVIEW_TEXT,
    RETENTION_SECONDS,
    FailureReason,
    JobKind,
    JobStatus,
    Profile,
    ReferenceSample,
    VoiceDelivery,
    VoiceSettingKey,
    WorkerJob,
)


class QueueFull(ValueError):
    pass


class NoProfile(ValueError):
    pass


class InsufficientBalance(ValueError):
    pass


TRIAL_MS = 600_000
MAX_OUTPUT_MS = 180_000
MAX_PREVIEW_MS = 30_000


@dataclass(frozen=True)
class Account:
    user_id: int
    is_admin: bool
    balance_ms: int
    reserved_ms: int
    username: str | None = field(default=None, repr=False)
    display_name: str | None = field(default=None, repr=False)

    @property
    def available_ms(self) -> int:
        return max(0, self.balance_ms - self.reserved_ms)


class VoiceTagLimit(ValueError):
    pass


class StaleLease(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    id: str
    chat_id: int
    status: JobStatus
    kind: JobKind
    created: float
    updated: float
    failure: FailureReason | None
    cancel_requested: bool
    profile_id: str | None


DraftStage = Literal["upload", "name", "rename", "delete", "setting", "reference"]


@dataclass(frozen=True)
class ProfileDraft:
    id: str
    stage: DraftStage
    profile_id: str | None
    updated: float
    setting_key: VoiceSettingKey | None = None


@dataclass(frozen=True)
class ReferenceChange:
    id: str
    profile_id: str
    stage: Literal["upload", "processing", "ready", "failed"]
    job_id: str | None
    candidate_count: int
    updated: float
    failure: FailureReason | None = None


class Store:
    def __init__(
        self,
        db: Path,
        data_dir: Path,
        owner_id: int,
        *,
        admin_user_ids: frozenset[int] = frozenset(),
    ) -> None:
        self._validate_panel_id(owner_id)
        for user_id in admin_user_ids:
            self._validate_panel_id(user_id)
        self.db = db
        self.data_dir = data_dir.resolve()
        self.owner_id = owner_id
        self.admin_user_ids = admin_user_ids or frozenset({owner_id})
        self._lock = threading.RLock()
        self._pending_profiles: dict[sqlite3.Connection, list[tuple[Path, Path]]] = {}
        self._profile_writes: dict[sqlite3.Connection, dict[Path, bytes]] = {}

    def allowed_user(self, user_id: int) -> bool:
        return type(user_id) is int and user_id in self.admin_user_ids

    def cancel_unauthorized_jobs(self) -> int:
        placeholders = ",".join("?" for _ in self.admin_user_ids)
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE jobs SET status='cancelled',cancel_requested=1,text=NULL,"
                "lease=NULL,lease_until=NULL,updated=? "
                "WHERE status IN ('queued','running','ready') "
                f"AND chat_id NOT IN ({placeholders})",
                (time.time(), *sorted(self.admin_user_ids)),
            ).rowcount
            self._release_reservations(connection)
            return changed

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, closing(sqlite3.connect(self.db, timeout=10)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            created: list[tuple[Path, Path]] = []
            self._pending_profiles[connection] = created
            originals: dict[Path, bytes] = {}
            self._profile_writes[connection] = originals
            try:
                yield connection
                connection.commit()
            except BaseException:
                try:
                    if originals:
                        from .config import atomic_private

                        for path, content in originals.items():
                            atomic_private(path, content)
                    for directory, marker in reversed(created):
                        if directory.exists():
                            shutil.rmtree(directory)
                        marker.unlink(missing_ok=True)
                finally:
                    connection.rollback()
                raise
            else:
                for _, marker in created:
                    marker.unlink(missing_ok=True)
            finally:
                del self._pending_profiles[connection]
                del self._profile_writes[connection]

    def initialize(self) -> None:
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(mode=0o700, exist_ok=True)
        (self.data_dir / "profiles").mkdir(mode=0o700, exist_ok=True)
        (self.data_dir / "drafts").mkdir(mode=0o700, exist_ok=True)
        (self.data_dir / "profile-trash").mkdir(mode=0o700, exist_ok=True)
        (self.data_dir / "reference-changes").mkdir(mode=0o700, exist_ok=True)
        with self._transaction() as connection:
            self._recover_default_voice_copies(connection)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > 9:
                raise ValueError("Unsupported database schema")
            if version >= 8:
                original = connection.execute(
                    "SELECT value FROM settings WHERE key='initial_owner_id'"
                ).fetchone()
                if original is None:
                    raise ValueError("Initial ownership metadata is missing")
                self.owner_id = int(original[0])
                self._validate_panel_id(self.owner_id)
            if version == 0:
                connection.execute(
                    "CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE updates(update_id INTEGER PRIMARY KEY, job_id TEXT NOT NULL)"
                )
                connection.execute(
                    """CREATE TABLE jobs(
                        id TEXT PRIMARY KEY, chat_id INTEGER NOT NULL,
                        kind TEXT NOT NULL CHECK(kind IN ('tts','vc')),
                        text TEXT, profile TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN
                         ('queued','running','ready','sending','sent','uncertain','failed','cancelled')),
                        created REAL NOT NULL, updated REAL NOT NULL,
                        lease TEXT, lease_until REAL, started REAL,
                        cancel_requested INTEGER NOT NULL DEFAULT 0
                         CHECK(cancel_requested IN (0,1)),
                        failure TEXT, delivered_message_id INTEGER
                    )"""
                )
                connection.execute("CREATE INDEX jobs_status_created ON jobs(status,created)")
                connection.execute(
                    "CREATE UNIQUE INDEX one_running_job ON jobs((1)) WHERE status='running'"
                )
                connection.execute("PRAGMA user_version=1")
            if version < 2:
                connection.execute("ALTER TABLE jobs ADD COLUMN status_message_id INTEGER")
                connection.execute("ALTER TABLE jobs ADD COLUMN status_notified TEXT")
                connection.execute("PRAGMA user_version=2")
            if version < 3:
                connection.execute(
                    "CREATE TABLE delivery_callbacks("
                    "callback_id TEXT PRIMARY KEY, job_id TEXT NOT NULL)"
                )
                connection.execute("PRAGMA user_version=3")
            if version < 4:
                connection.execute(
                    """CREATE TABLE jobs_v4(
                        id TEXT PRIMARY KEY, chat_id INTEGER NOT NULL,
                        kind TEXT NOT NULL CHECK(kind IN ('tts','vc','profile')),
                        text TEXT, profile TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN
                         ('queued','running','ready','sending','sent','uncertain','failed','cancelled')),
                        created REAL NOT NULL, updated REAL NOT NULL,
                        lease TEXT, lease_until REAL, started REAL,
                        cancel_requested INTEGER NOT NULL DEFAULT 0
                         CHECK(cancel_requested IN (0,1)),
                        failure TEXT, delivered_message_id INTEGER,
                        status_message_id INTEGER, status_notified TEXT,
                        result_profile_id TEXT
                    )"""
                )
                connection.execute("INSERT INTO jobs_v4 SELECT *,NULL FROM jobs")
                connection.execute("DROP TABLE jobs")
                connection.execute("ALTER TABLE jobs_v4 RENAME TO jobs")
                connection.execute("CREATE INDEX jobs_status_created ON jobs(status,created)")
                connection.execute(
                    "CREATE UNIQUE INDEX one_running_job ON jobs((1)) WHERE status='running'"
                )
                connection.execute(
                    "CREATE TABLE profile_drafts(chat_id INTEGER PRIMARY KEY,id TEXT NOT NULL,"
                    "stage TEXT NOT NULL CHECK(stage IN ('upload','name','rename','delete')),"
                    "profile_id TEXT,updated REAL NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE ui_callbacks(callback_id TEXT PRIMARY KEY,created REAL NOT NULL)"
                )
                connection.execute("PRAGMA user_version=4")
            if version < 5:
                connection.execute(
                    "CREATE TABLE profile_drafts_v5(chat_id INTEGER PRIMARY KEY,id TEXT NOT NULL,"
                    "stage TEXT NOT NULL "
                    "CHECK(stage IN ('upload','name','rename','delete','setting')),"
                    "profile_id TEXT,updated REAL NOT NULL,setting_key TEXT,"
                    "CHECK((stage='setting' AND setting_key IS NOT NULL "
                    "AND profile_id IS NOT NULL) "
                    "OR (stage!='setting' AND setting_key IS NULL)))"
                )
                connection.execute(
                    "INSERT INTO profile_drafts_v5 SELECT *,NULL FROM profile_drafts"
                )
                connection.execute("DROP TABLE profile_drafts")
                connection.execute("ALTER TABLE profile_drafts_v5 RENAME TO profile_drafts")
                connection.execute("PRAGMA user_version=5")
            if version < 6:
                connection.execute(
                    "CREATE TABLE ui_panels("
                    "chat_id INTEGER PRIMARY KEY CHECK(chat_id>0),"
                    "message_id INTEGER NOT NULL CHECK(message_id>0),"
                    "text TEXT,keyboard_json TEXT,"
                    "CHECK((text IS NULL AND keyboard_json IS NULL) OR "
                    "(text IS NOT NULL AND keyboard_json IS NOT NULL "
                    "AND length(text) BETWEEN 1 AND 3500 "
                    "AND length(keyboard_json) BETWEEN 1 AND 20000)))"
                )
                connection.execute("PRAGMA user_version=6")
            if version < 7:
                connection.execute(
                    "CREATE TABLE jobs_v7("
                    "id TEXT PRIMARY KEY,chat_id INTEGER NOT NULL,"
                    "kind TEXT NOT NULL CHECK(kind IN ('tts','vc','profile','reference')),"
                    "text TEXT,profile TEXT NOT NULL,"
                    "status TEXT NOT NULL CHECK(status IN "
                    "('queued','running','ready','sending','sent','uncertain','failed','cancelled')),"
                    "created REAL NOT NULL,updated REAL NOT NULL,lease TEXT,lease_until REAL,"
                    "started REAL,cancel_requested INTEGER NOT NULL DEFAULT 0 "
                    "CHECK(cancel_requested IN (0,1)),failure TEXT,delivered_message_id INTEGER,"
                    "status_message_id INTEGER,status_notified TEXT,result_profile_id TEXT)"
                )
                connection.execute("INSERT INTO jobs_v7 SELECT * FROM jobs")
                connection.execute("DROP TABLE jobs")
                connection.execute("ALTER TABLE jobs_v7 RENAME TO jobs")
                connection.execute("CREATE INDEX jobs_status_created ON jobs(status,created)")
                connection.execute(
                    "CREATE UNIQUE INDEX one_running_job ON jobs((1)) WHERE status='running'"
                )
                connection.execute(
                    "CREATE TABLE profile_drafts_v7(chat_id INTEGER PRIMARY KEY,id TEXT NOT NULL,"
                    "stage TEXT NOT NULL CHECK(stage IN "
                    "('upload','name','rename','delete','setting','reference')),"
                    "profile_id TEXT,updated REAL NOT NULL,setting_key TEXT,"
                    "CHECK((stage='setting' AND setting_key IS NOT NULL AND profile_id IS NOT NULL)"
                    " OR (stage!='setting' AND setting_key IS NULL)))"
                )
                connection.execute("INSERT INTO profile_drafts_v7 SELECT * FROM profile_drafts")
                connection.execute("DROP TABLE profile_drafts")
                connection.execute("ALTER TABLE profile_drafts_v7 RENAME TO profile_drafts")
                connection.execute(
                    "CREATE TABLE reference_changes("
                    "id TEXT PRIMARY KEY,chat_id INTEGER NOT NULL,profile_id TEXT NOT NULL UNIQUE,"
                    "stage TEXT NOT NULL CHECK(stage IN ('upload','processing','ready')),"
                    "operation TEXT NOT NULL CHECK(operation IN ('add','remove')),"
                    "sample_index INTEGER,baseline TEXT NOT NULL,job_id TEXT UNIQUE,"
                    "candidate_count INTEGER NOT NULL CHECK(candidate_count BETWEEN 1 AND 3),"
                    "updated REAL NOT NULL)"
                )
                connection.execute("PRAGMA user_version=7")
            self._recover_reference_swaps(connection)
            self._recover_profile_deletions(connection)
            if version < 8:
                connection.execute(
                    "INSERT OR IGNORE INTO settings(key,value) VALUES('initial_owner_id',?)",
                    (str(self.owner_id),),
                )
                original = connection.execute(
                    "SELECT value FROM settings WHERE key='initial_owner_id'"
                ).fetchone()
                self.owner_id = int(original[0])
                self._validate_panel_id(self.owner_id)
                connection.execute(
                    "CREATE TABLE accounts("
                    "user_id INTEGER PRIMARY KEY CHECK(user_id>0),"
                    "balance_ms INTEGER NOT NULL CHECK(balance_ms>=0),"
                    "selected_profile_id TEXT,created REAL NOT NULL,"
                    "username TEXT,display_name TEXT,"
                    "default_voice_added INTEGER NOT NULL DEFAULT 0 "
                    "CHECK(default_voice_added IN (0,1)))"
                )
                connection.execute(
                    "CREATE TABLE profile_owners("
                    "profile_id TEXT PRIMARY KEY,user_id INTEGER NOT NULL "
                    "REFERENCES accounts(user_id),is_default INTEGER NOT NULL DEFAULT 0 "
                    "CHECK(is_default IN (0,1)))"
                )
                connection.execute("CREATE INDEX profile_owners_user ON profile_owners(user_id)")
                connection.execute(
                    "CREATE TABLE job_billing("
                    "job_id TEXT PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES accounts(user_id),"
                    "exempt INTEGER NOT NULL CHECK(exempt IN (0,1)),"
                    "preview INTEGER NOT NULL CHECK(preview IN (0,1)),"
                    "reserved_ms INTEGER NOT NULL CHECK(reserved_ms>=0),"
                    "charged_ms INTEGER NOT NULL DEFAULT 0 CHECK(charged_ms>=0),"
                    "state TEXT NOT NULL CHECK(state IN ('reserved','settled','released')))"
                )
                connection.execute("CREATE INDEX job_billing_user ON job_billing(user_id,state)")
                self._ensure_account(connection, self.owner_id, add_default=False)
                for profile in self._profiles():
                    connection.execute(
                        "INSERT INTO profile_owners(profile_id,user_id) VALUES(?,?)",
                        (profile.id, self.owner_id),
                    )
                selected = connection.execute(
                    "SELECT value FROM settings WHERE key='profile'"
                ).fetchone()
                if selected is not None:
                    connection.execute(
                        "UPDATE accounts SET selected_profile_id=? WHERE user_id=? "
                        "AND EXISTS(SELECT 1 FROM profile_owners WHERE profile_id=? AND user_id=?)",
                        (selected[0], self.owner_id, selected[0], self.owner_id),
                    )
                connection.execute("DELETE FROM settings WHERE key='profile'")
                connection.execute(
                    "UPDATE ui_panels SET text=NULL,keyboard_json=NULL WHERE chat_id!=?",
                    (self.owner_id,),
                )
                # Existing work began before billing. Preserve its result without a new charge.
                for row in connection.execute(
                    "SELECT id,chat_id,status FROM jobs WHERE kind IN ('tts','vc')"
                ).fetchall():
                    self._ensure_account(connection, int(row["chat_id"]), add_default=False)
                    connection.execute(
                        "INSERT INTO job_billing(job_id,user_id,exempt,preview,reserved_ms,state) "
                        "VALUES(?,?,1,0,0,?)",
                        (
                            row["id"],
                            row["chat_id"],
                            "reserved" if row["status"] in ("queued", "running") else "released",
                        ),
                    )
                # Old guests could not edit profiles. Discard obsolete cross-account input.
                connection.execute(
                    "DELETE FROM profile_drafts WHERE profile_id IS NOT NULL AND NOT EXISTS("
                    "SELECT 1 FROM profile_owners "
                    "WHERE profile_owners.profile_id=profile_drafts.profile_id "
                    "AND profile_owners.user_id=profile_drafts.chat_id)"
                )
                connection.execute("PRAGMA user_version=8")
            if version < 9:
                self._migrate_history(connection)
                connection.execute("PRAGMA user_version=9")
            self._reconcile_voice_metadata(connection)
            self._default_voice_template(connection)
            for user_id in self.admin_user_ids:
                self._ensure_account(connection, user_id)
                connection.execute("UPDATE accounts SET balance_ms=0 WHERE user_id=?", (user_id,))
            for row in connection.execute(
                "SELECT user_id FROM accounts WHERE default_voice_added=0"
            ).fetchall():
                self._add_default_voice(connection, int(row[0]))
            connection.execute("UPDATE accounts SET is_admin=0")
            connection.executemany(
                "UPDATE accounts SET is_admin=1 WHERE user_id=?",
                ((user_id,) for user_id in self.admin_user_ids),
            )
            if (
                connection.execute(
                    "SELECT 1 FROM settings WHERE key='separate_job_status_messages'"
                ).fetchone()
                is None
            ):
                # All older anchors referred to menu panels, including panels already replaced.
                connection.execute(
                    "UPDATE jobs SET status_message_id=NULL,status_notified=CASE "
                    "WHEN status IN ('queued','running','ready','sending') THEN NULL "
                    "ELSE status || ':' || cancel_requested END"
                )
                connection.execute(
                    "UPDATE ui_panels SET text=NULL,keyboard_json=NULL "
                    "WHERE text IN ('<b>Озвучка</b>','<b>Новый голос</b>')"
                )
                connection.execute(
                    "INSERT INTO settings(key,value) VALUES('separate_job_status_messages','1')"
                )

    @staticmethod
    def _voice_metadata_json(profile: Profile) -> str:
        metadata = profile.model_dump(mode="json")
        metadata["effective_native_tags"] = list(profile.effective_native_tags)
        return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _sync_voice_metadata(self, connection: sqlite3.Connection, profile: Profile) -> None:
        connection.execute(
            "INSERT INTO voice_metadata(profile_id,profile_json,updated) VALUES(?,?,?) "
            "ON CONFLICT(profile_id) DO UPDATE SET "
            "profile_json=excluded.profile_json,updated=excluded.updated",
            (profile.id, self._voice_metadata_json(profile), time.time()),
        )

    def _reconcile_voice_metadata(self, connection: sqlite3.Connection) -> None:
        """Restore a metadata mirror interrupted before its SQLite transaction committed."""
        from .config import atomic_private

        for row in connection.execute("SELECT profile_id,profile_json FROM voice_metadata"):
            values = json.loads(str(row["profile_json"]))
            values.pop("effective_native_tags", None)
            committed = Profile.model_validate(values)
            if committed.id != row["profile_id"]:
                raise ValueError("Database voice identifier does not match its metadata")
            self._profile_reference_paths(committed)
            path = self.data_dir / "profiles" / committed.id / "profile.json"
            if path.is_symlink():
                raise ValueError("Invalid profile metadata path")
            try:
                unchanged = (
                    path.is_file() and Profile.model_validate_json(path.read_text()) == committed
                )
            except ValueError:
                unchanged = False
            if not unchanged:
                # Equal profiles keep their original bytes, including historical formatting.
                atomic_private(path, committed.model_dump_json().encode())

    def _migrate_history(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "ALTER TABLE accounts ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0 "
            "CHECK(is_admin IN (0,1))"
        )
        connection.execute(
            "CREATE TABLE voice_metadata(profile_id TEXT PRIMARY KEY "
            "REFERENCES profile_owners(profile_id) ON DELETE CASCADE,"
            "profile_json TEXT NOT NULL CHECK(json_valid(profile_json)),updated REAL NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE generation_history("
            "job_id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,kind TEXT NOT NULL,"
            "created REAL NOT NULL,updated REAL NOT NULL,"
            "historical INTEGER NOT NULL CHECK(historical IN (0,1)),"
            "profile_id TEXT,profile_name TEXT,voice_snapshot_json TEXT,reference_fingerprint TEXT,"
            "input_text TEXT,transcript TEXT,preview INTEGER,exempt INTEGER,"
            "status TEXT NOT NULL,generation_status TEXT NOT NULL,delivery_status TEXT,"
            "failure TEXT,cancel_requested INTEGER NOT NULL DEFAULT 0,"
            "duration_ms INTEGER,charged_ms INTEGER,billing_state TEXT,"
            "delivered_message_id INTEGER,result_profile_id TEXT,result_voice_snapshot_json TEXT,"
            "generated_at REAL,"
            "CHECK(voice_snapshot_json IS NULL OR json_valid(voice_snapshot_json)),"
            "CHECK(result_voice_snapshot_json IS NULL OR json_valid(result_voice_snapshot_json)),"
            "CHECK(duration_ms IS NULL OR duration_ms>0),"
            "CHECK(charged_ms IS NULL OR charged_ms>=0))"
        )
        connection.execute(
            "CREATE INDEX generation_history_user_created ON generation_history(user_id,created)"
        )
        owners = {
            str(row[0]) for row in connection.execute("SELECT profile_id FROM profile_owners")
        }
        for profile in self._profiles():
            if profile.id in owners:
                self._sync_voice_metadata(connection, profile)
        # A surviving job already contains the voice controls used when it was accepted.
        # Completed text and deleted files cannot be reconstructed from current voice settings.
        for row in connection.execute("SELECT * FROM jobs").fetchall():
            raw = json.loads(str(row["profile"]))
            snapshot = Profile.model_validate(raw) if raw is not None else None
            self._record_generation(connection, str(row["id"]), snapshot, historical=True)
        connection.execute(
            "CREATE TRIGGER generation_job_state AFTER UPDATE ON jobs BEGIN "
            "UPDATE generation_history SET updated=NEW.updated,status=NEW.status,"
            "generation_status=CASE "
            "WHEN NEW.status IN ('ready','sending','sent','uncertain') "
            "OR (NEW.status='failed' AND NEW.failure='delivery_error') THEN 'ready' "
            "ELSE NEW.status END,"
            "delivery_status=CASE "
            "WHEN NEW.status IN ('ready','sending','sent','uncertain') THEN NEW.status "
            "WHEN NEW.status='failed' AND NEW.failure='delivery_error' THEN 'failed' ELSE NULL END,"
            "failure=NEW.failure,cancel_requested=NEW.cancel_requested,"
            "delivered_message_id=NEW.delivered_message_id,result_profile_id=NEW.result_profile_id,"
            "generated_at=CASE WHEN generated_at IS NULL AND NEW.status='ready' "
            "AND OLD.status='running' THEN NEW.updated ELSE generated_at END "
            "WHERE job_id=NEW.id; END"
        )
        connection.execute(
            "CREATE TRIGGER generation_billing_state AFTER UPDATE ON job_billing BEGIN "
            "UPDATE generation_history SET charged_ms=NEW.charged_ms,billing_state=NEW.state "
            "WHERE job_id=NEW.job_id; END"
        )
        immutable = (
            "job_id",
            "user_id",
            "kind",
            "created",
            "historical",
            "profile_id",
            "profile_name",
            "voice_snapshot_json",
            "reference_fingerprint",
            "input_text",
            "preview",
            "exempt",
        )
        changed = " OR ".join(f"OLD.{field} IS NOT NEW.{field}" for field in immutable)
        connection.execute(
            "CREATE TRIGGER generation_snapshot_immutable BEFORE UPDATE ON generation_history "
            f"WHEN {changed} BEGIN SELECT RAISE(ABORT,'Generation snapshot is immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER generation_transcript_immutable BEFORE UPDATE OF transcript "
            "ON generation_history WHEN OLD.transcript IS NOT NULL "
            "AND OLD.transcript IS NOT NEW.transcript "
            "BEGIN SELECT RAISE(ABORT,'Recognized text is immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER generation_result_snapshot_immutable "
            "BEFORE UPDATE OF result_voice_snapshot_json ON generation_history "
            "WHEN OLD.result_voice_snapshot_json IS NOT NULL "
            "AND OLD.result_voice_snapshot_json IS NOT NEW.result_voice_snapshot_json "
            "BEGIN SELECT RAISE(ABORT,'Prepared voice snapshot is immutable'); END"
        )
        connection.execute(
            "CREATE VIEW report_users AS SELECT a.user_id,a.username,a.display_name,a.is_admin,"
            "a.balance_ms,COALESCE((SELECT SUM(b.reserved_ms) FROM job_billing b "
            "WHERE b.user_id=a.user_id AND b.state='reserved'),0) AS reserved_ms,"
            "a.selected_profile_id,json_extract(v.profile_json,'$.name') AS selected_voice_name,"
            "a.created FROM accounts a LEFT JOIN voice_metadata v "
            "ON v.profile_id=a.selected_profile_id AND EXISTS(SELECT 1 FROM profile_owners p "
            "WHERE p.profile_id=v.profile_id AND p.user_id=a.user_id)"
        )
        connection.execute(
            "CREATE VIEW report_voices AS SELECT p.user_id,p.profile_id,p.is_default,"
            "COALESCE(a.selected_profile_id=p.profile_id,0) AS is_selected,"
            "json_extract(v.profile_json,'$.name') AS name,"
            "json_extract(v.profile_json,'$.speed') AS speed,"
            "json_extract(v.profile_json,'$.effective_native_tags') AS native_tags_json,"
            "json_extract(v.profile_json,'$.style_instruction') AS style_instruction,"
            "json_extract(v.profile_json,'$.temperature') AS temperature,"
            "json_extract(v.profile_json,'$.top_p') AS top_p,"
            "json_extract(v.profile_json,'$.top_k') AS top_k,"
            "json_extract(v.profile_json,'$.seed') AS seed,"
            "1+json_array_length(v.profile_json,'$.extra_references') AS reference_count,"
            "v.profile_json,v.updated FROM profile_owners p JOIN voice_metadata v "
            "ON v.profile_id=p.profile_id JOIN accounts a ON a.user_id=p.user_id"
        )
        connection.execute(
            "CREATE VIEW report_generations AS SELECT h.*,"
            "CASE WHEN h.kind='vc' THEN h.transcript WHEN h.kind='tts' THEN h.input_text "
            "ELSE NULL END AS synthesis_text FROM generation_history h"
        )

    def _record_generation(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        snapshot: Profile | None,
        *,
        historical: bool = False,
    ) -> None:
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        billing = connection.execute(
            "SELECT * FROM job_billing WHERE job_id=?", (job_id,)
        ).fetchone()
        fingerprint = None
        if snapshot is not None and not historical:
            references = [
                {
                    "sha256": hashlib.sha256(
                        (self.data_dir / sample.reference_audio).read_bytes()
                    ).hexdigest(),
                    "text": sample.reference_text,
                }
                for sample in snapshot.reference_samples
            ]
            fingerprint = hashlib.sha256(
                json.dumps(
                    references, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        status = str(row["status"])
        generated = status in ("ready", "sending", "sent", "uncertain") or (
            status == "failed" and row["failure"] == "delivery_error"
        )
        delivery = (
            status
            if status in ("ready", "sending", "sent", "uncertain")
            else ("failed" if generated else None)
        )
        charged = None if historical else 0
        if billing is not None and billing["state"] == "settled":
            charged = int(billing["charged_ms"])
        connection.execute(
            "INSERT INTO generation_history(job_id,user_id,kind,created,updated,historical,"
            "profile_id,profile_name,voice_snapshot_json,reference_fingerprint,input_text,"
            "preview,exempt,status,generation_status,delivery_status,failure,cancel_requested,"
            "charged_ms,billing_state,delivered_message_id,result_profile_id) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                row["chat_id"],
                row["kind"],
                row["created"],
                row["updated"],
                int(historical),
                snapshot.id if snapshot else None,
                snapshot.name if snapshot else None,
                (str(row["profile"]) if historical else self._voice_metadata_json(snapshot))
                if snapshot
                else None,
                fingerprint,
                row["text"],
                (
                    int(billing["preview"])
                    if billing is not None
                    else int(row["kind"] in ("profile", "reference"))
                )
                if not historical
                else None,
                int(billing["exempt"]) if billing is not None and not historical else None,
                status,
                "ready" if generated else status,
                delivery,
                row["failure"],
                row["cancel_requested"],
                charged,
                billing["state"] if billing is not None else None,
                row["delivered_message_id"],
                row["result_profile_id"],
            ),
        )

    def record_transcript(self, job_id: str, lease: str, text: str) -> None:
        """Persist successful local ASR before synthesis; retries may repeat the same text."""
        if not text.strip() or len(text) > MAX_TEXT or "\0" in text:
            raise ValueError("Invalid recognized text")
        with self._transaction() as connection:
            row = self._leased(connection, job_id, lease)
            if row["cancel_requested"]:
                raise StaleLease("Cancellation has been requested")
            if row["kind"] != "vc":
                raise ValueError("Only speech input has a recognition transcript")
            history = connection.execute(
                "SELECT transcript FROM generation_history WHERE job_id=?", (job_id,)
            ).fetchone()
            if history is None:
                raise ValueError("Generation history is missing")
            if history[0] is not None and history[0] != text:
                raise ValueError("Recognized text was already recorded")
            connection.execute(
                "UPDATE generation_history SET transcript=? WHERE job_id=?", (text, job_id)
            )

    def _ensure_account(
        self, connection: sqlite3.Connection, chat_id: int, *, add_default: bool = True
    ) -> None:
        self._validate_panel_id(chat_id)
        connection.execute(
            "INSERT OR IGNORE INTO accounts(user_id,balance_ms,created) VALUES(?,?,?)",
            (chat_id, 0 if chat_id in self.admin_user_ids else TRIAL_MS, time.time()),
        )
        if add_default:
            connection.execute(
                "UPDATE accounts SET is_admin=? WHERE user_id=?",
                (int(chat_id in self.admin_user_ids), chat_id),
            )
            self._add_default_voice(connection, chat_id)

    def _recover_default_voice_copies(self, connection: sqlite3.Connection) -> None:
        """Resolve interrupted onboarding before legacy files can be assigned to the owner."""
        ownership_exists = (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='profile_owners'"
            ).fetchone()
            is not None
        )
        for marker in (self.data_dir / "profile-trash").glob("default-*.json"):
            identifier = marker.stem.removeprefix("default-")
            self.job_dir(identifier)
            if marker.is_symlink() or not marker.is_file():
                raise ValueError("Invalid default voice recovery marker")
            metadata = json.loads(marker.read_text())
            profile_id = "voice-" + identifier
            if not isinstance(metadata, dict) or set(metadata) != {"profile_id", "user_id"}:
                raise ValueError("Invalid default voice recovery metadata")
            self._validate_panel_id(metadata["user_id"])
            if metadata["profile_id"] != profile_id:
                raise ValueError("Default voice recovery profile does not match its marker")
            committed = (
                ownership_exists
                and connection.execute(
                    "SELECT 1 FROM profile_owners WHERE profile_id=? AND user_id=?",
                    (profile_id, metadata["user_id"]),
                ).fetchone()
                is not None
            )
            directory = self.data_dir / "profiles" / profile_id
            if directory.is_symlink():
                raise ValueError("Invalid default voice recovery directory")
            if not committed and directory.exists():
                shutil.rmtree(directory)
            marker.unlink()

    def _default_voice_template(self, connection: sqlite3.Connection) -> tuple[Profile, int] | None:
        directory = self.data_dir / "default-voice"
        if directory.is_symlink():
            raise ValueError("Default voice template cannot be a symbolic link")
        if not directory.exists():
            configured = connection.execute(
                "SELECT value FROM settings WHERE key='default_voice_configured'"
            ).fetchone()
            if configured is not None and configured[0] == "1":
                raise ValueError("The configured default voice template is missing")
            return None
        if not directory.is_dir():
            raise ValueError("Invalid default voice template directory")
        files = list(directory.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in files):
            raise ValueError("Default voice template requires regular private files")
        profile = Profile.model_validate_json((directory / "profile.json").read_text())
        origin = json.loads((directory / "origin.json").read_text())
        if not isinstance(origin, dict) or set(origin) != {"source_profile_id", "owner_id"}:
            raise ValueError("Invalid default voice origin")
        self._validate_panel_id(origin["owner_id"])
        if origin["source_profile_id"] != profile.id:
            raise ValueError("Default voice origin does not match its profile")
        references = {sample.reference_audio for sample in profile.reference_samples}
        if not references <= {"reference.wav", "reference-2.wav", "reference-3.wav"}:
            raise ValueError("Invalid default voice reference path")
        required = references | {"profile.json", "origin.json", "example.ogg"}
        names = {path.name for path in files}
        if not required <= names or not names <= required | {"whisper-context.json"}:
            raise ValueError("Default voice template is incomplete or contains unexpected files")
        connection.execute(
            "INSERT OR IGNORE INTO settings(key,value) VALUES('default_voice_configured','1')"
        )
        return profile, int(origin["owner_id"])

    def _add_default_voice(self, connection: sqlite3.Connection, chat_id: int) -> None:
        row = connection.execute(
            "SELECT default_voice_added FROM accounts WHERE user_id=?", (chat_id,)
        ).fetchone()
        if row is None or row[0]:
            return
        template = self._default_voice_template(connection)
        if template is None:
            return
        profile, source_owner_id = template
        original = connection.execute(
            "SELECT profile_id FROM profile_owners WHERE user_id=? AND profile_id=?",
            (chat_id, profile.id),
        ).fetchone()
        if chat_id == source_owner_id and original is not None:
            selected_id = profile.id
        else:
            from .config import atomic_private

            identifier = uuid.uuid4().hex
            selected_id = "voice-" + identifier
            target = self.data_dir / "profiles" / selected_id
            marker = self.data_dir / "profile-trash" / f"default-{identifier}.json"
            if target.exists() or target.is_symlink() or marker.exists() or marker.is_symlink():
                raise ValueError("Default voice profile already exists")
            self._pending_profiles[connection].append((target, marker))
            atomic_private(
                marker, json.dumps({"profile_id": selected_id, "user_id": chat_id}).encode()
            )
            target.mkdir(mode=0o700)
            source = self.data_dir / "default-voice"
            filenames = {sample.reference_audio for sample in profile.reference_samples}
            filenames.add("example.ogg")
            if (source / "whisper-context.json").exists():
                filenames.add("whisper-context.json")
            for filename in sorted(filenames):
                shutil.copyfile(source / filename, target / filename)
                (target / filename).chmod(0o600)
            personal = profile.model_copy(update={"id": selected_id})
            atomic_private(target / "profile.json", personal.model_dump_json().encode())
            connection.execute(
                "INSERT INTO profile_owners(profile_id,user_id,is_default) VALUES(?,?,1)",
                (selected_id, chat_id),
            )
            self._sync_voice_metadata(connection, personal)
        connection.execute(
            "UPDATE accounts SET default_voice_added=1,"
            "selected_profile_id=COALESCE(selected_profile_id,?) WHERE user_id=?",
            (selected_id, chat_id),
        )

    def _account(self, connection: sqlite3.Connection, chat_id: int) -> Account:
        self._ensure_account(connection, chat_id)
        row = connection.execute(
            "SELECT balance_ms,username,display_name FROM accounts WHERE user_id=?", (chat_id,)
        ).fetchone()
        reserved = connection.execute(
            "SELECT COALESCE(SUM(reserved_ms),0) FROM job_billing "
            "WHERE user_id=? AND state='reserved'",
            (chat_id,),
        ).fetchone()[0]
        return Account(
            chat_id,
            chat_id in self.admin_user_ids,
            int(row[0]),
            int(reserved),
            str(row[1]) if row[1] is not None else None,
            str(row[2]) if row[2] is not None else None,
        )

    def update_account_user(self, chat_id: int, username: str | None, display_name: str) -> Account:
        """Keep only this person's Telegram identity supplied by an authorized update."""
        if username is not None and (
            not 1 <= len(username) <= 32
            or any(not char.isascii() or not (char.isalnum() or char == "_") for char in username)
        ):
            raise ValueError("Invalid Telegram username")
        readable_name = "".join(" " if category(char) == "Cc" else char for char in display_name)
        normalized_name = " ".join(readable_name.split())[:256].strip() or None
        with self._transaction() as connection:
            self._ensure_account(connection, chat_id)
            connection.execute(
                "UPDATE accounts SET username=?,display_name=? WHERE user_id=?",
                (username, normalized_name, chat_id),
            )
            return self._account(connection, chat_id)

    def account(self, chat_id: int) -> Account:
        """Create one personal account; reopening it never replenishes its initial trial."""
        with self._transaction() as connection:
            return self._account(connection, chat_id)

    @staticmethod
    def _release_reservations(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE job_billing SET reserved_ms=0,state='released' WHERE state='reserved' "
            "AND NOT EXISTS(SELECT 1 FROM jobs WHERE jobs.id=job_billing.job_id "
            "AND jobs.status IN ('queued','running'))"
        )

    @staticmethod
    def _validate_panel_id(value: int) -> None:
        if type(value) is not int or not 0 < value < 2**63:
            raise ValueError("Panel identifiers must be positive SQLite integers")

    def panel_id(self, chat_id: int) -> int | None:
        """Read the persistent panel anchor after the caller authorizes this chat."""
        self._validate_panel_id(chat_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT message_id FROM ui_panels WHERE chat_id=?", (chat_id,)
            ).fetchone()
            return int(row[0]) if row is not None else None

    def set_panel(self, chat_id: int, message_id: int) -> None:
        self._validate_panel_id(chat_id)
        self._validate_panel_id(message_id)
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO ui_panels(chat_id,message_id) VALUES(?,?) "
                "ON CONFLICT(chat_id) DO UPDATE SET "
                "text=CASE WHEN message_id=excluded.message_id THEN text ELSE NULL END,"
                "keyboard_json=CASE WHEN message_id=excluded.message_id "
                "THEN keyboard_json ELSE NULL END,message_id=excluded.message_id",
                (chat_id, message_id),
            )

    def panel_content(self, chat_id: int) -> tuple[str, str] | None:
        """Read the last successful base view, excluding transient job status."""
        self._validate_panel_id(chat_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT text,keyboard_json FROM ui_panels WHERE chat_id=?", (chat_id,)
            ).fetchone()
            if row is None or row[0] is None or row[1] is None:
                return None
            return str(row[0]), str(row[1])

    def replace_panel(self, chat_id: int, message_id: int, text: str, keyboard_json: str) -> None:
        """Atomically replace the anchor and its base after Telegram confirms a new panel."""
        self._validate_panel_id(chat_id)
        self._validate_panel_id(message_id)
        if not 1 <= len(text) <= 3500 or not 1 <= len(keyboard_json) <= 20000:
            raise ValueError("Panel content exceeds the accepted limits")
        if not isinstance(json.loads(keyboard_json), dict):
            raise ValueError("Panel keyboard must be a JSON object")
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO ui_panels(chat_id,message_id,text,keyboard_json) VALUES(?,?,?,?) "
                "ON CONFLICT(chat_id) DO UPDATE SET message_id=excluded.message_id,"
                "text=excluded.text,keyboard_json=excluded.keyboard_json",
                (chat_id, message_id, text, keyboard_json),
            )

    def set_panel_content(
        self, chat_id: int, message_id: int, text: str, keyboard_json: str
    ) -> bool:
        """Save a rendered view only while its message is still the active panel."""
        self._validate_panel_id(chat_id)
        self._validate_panel_id(message_id)
        if not 1 <= len(text) <= 3500 or not 1 <= len(keyboard_json) <= 20000:
            raise ValueError("Panel content exceeds the accepted limits")
        if not isinstance(json.loads(keyboard_json), dict):
            raise ValueError("Panel keyboard must be a JSON object")
        with self._transaction() as connection:
            return (
                connection.execute(
                    "UPDATE ui_panels SET text=?,keyboard_json=? WHERE chat_id=? AND message_id=?",
                    (text, keyboard_json, chat_id, message_id),
                ).rowcount
                == 1
            )

    def clear_panel(self, chat_id: int, message_id: int) -> bool:
        """Forget a failed anchor without erasing a concurrently replaced panel."""
        self._validate_panel_id(chat_id)
        self._validate_panel_id(message_id)
        with self._transaction() as connection:
            return (
                connection.execute(
                    "DELETE FROM ui_panels WHERE chat_id=? AND message_id=?",
                    (chat_id, message_id),
                ).rowcount
                == 1
            )

    def recover(self) -> None:
        now = time.time()
        with self._transaction() as connection:
            self._recover_reference_swaps(connection)
            self._recover_profile_deletions(connection)
            connection.execute(
                "UPDATE jobs SET status='failed',failure='restart',text=NULL,lease=NULL,updated=? "
                "WHERE status='running'",
                (now,),
            )
            connection.execute(
                "UPDATE jobs SET status='uncertain',updated=? WHERE status='sending'", (now,)
            )
            self._release_reservations(connection)
        self.maintain(now)

    def _recover_profile_deletions(self, connection: sqlite3.Connection) -> None:
        """Restore a voice moved to trash before its deletion transaction committed."""
        for row in connection.execute(
            "SELECT id,profile_id FROM profile_drafts WHERE stage='delete'"
        ):
            identifier = str(row["id"])
            self.draft_dir(identifier)
            tomb = self.data_dir / "profile-trash" / identifier
            if tomb.is_symlink():
                raise ValueError("Invalid deleted profile directory")
            if not tomb.is_dir():
                continue
            metadata = tomb / "profile.json"
            if metadata.is_symlink():
                raise ValueError("Invalid deleted profile metadata")
            profile = Profile.model_validate_json(metadata.read_text())
            if profile.id != row["profile_id"]:
                raise ValueError("Deleted profile does not match its pending confirmation")
            target = self.data_dir / "profiles" / profile.id
            if target.exists() or target.is_symlink():
                raise ValueError("Ambiguous interrupted profile deletion")
            tomb.rename(target)

    def _recover_reference_swaps(self, connection: sqlite3.Connection) -> None:
        """Recover approval interrupted between filesystem rename and SQLite commit."""
        for marker in (self.data_dir / "profile-trash").glob("reference-*.json"):
            metadata = json.loads(marker.read_text())
            identifier = str(metadata["change_id"])
            prepared = self._reference_change_dir(identifier)
            profile_id = str(metadata["profile_id"])
            # Reuse the profile identifier validator before constructing local paths.
            from typing import Annotated

            from pydantic import Field, TypeAdapter

            TypeAdapter(
                Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")]
            ).validate_python(profile_id)
            tomb = self.data_dir / "profile-trash" / identifier
            target = self.data_dir / "profiles" / profile_id
            pending = connection.execute(
                "SELECT id FROM reference_changes WHERE id=? AND profile_id=?",
                (identifier, profile_id),
            ).fetchone()
            if tomb.is_dir():
                if not target.exists():
                    tomb.rename(target)
                elif pending is not None:
                    if prepared.exists():
                        raise ValueError("Ambiguous interrupted reference approval")
                    target.rename(prepared)
                    tomb.rename(target)
                else:
                    shutil.rmtree(tomb)
            marker.unlink()

    def profiles(self, chat_id: int) -> list[Profile]:
        with self._transaction() as connection:
            self._ensure_account(connection, chat_id)
            return self._owned_profiles(connection, chat_id)

    def _owned_profiles(self, connection: sqlite3.Connection, chat_id: int) -> list[Profile]:
        owned = {
            str(row[0])
            for row in connection.execute(
                "SELECT profile_id FROM profile_owners WHERE user_id=?", (chat_id,)
            )
        }
        return [
            self._read_profile(path)
            for path in sorted((self.data_dir / "profiles").glob("*/profile.json"))
            if path.parent.name in owned
        ]

    def _profiles(self) -> list[Profile]:
        """Enumerate existing files only for the one-time ownership migration."""
        return [
            self._read_profile(path)
            for path in sorted((self.data_dir / "profiles").glob("*/profile.json"))
        ]

    def _read_profile(self, path: Path) -> Profile:
        if path.is_symlink():
            raise ValueError("Invalid profile metadata path")
        profile = Profile.model_validate_json(path.read_text())
        if path.parent.name != profile.id:
            raise ValueError("Profile directory does not match its identifier")
        self._profile_reference_paths(profile)
        return profile

    def _profile_reference_paths(self, profile: Profile) -> tuple[Path, ...]:
        folder = self.data_dir / "profiles" / profile.id
        directory = folder.resolve()
        if folder.is_symlink() or directory.parent != (self.data_dir / "profiles").resolve():
            raise ValueError("Invalid profile directory")
        paths = []
        for sample in profile.reference_samples:
            relative = Path(sample.reference_audio)
            path = (directory / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.is_relative_to(directory)
            ):
                raise ValueError("Invalid profile audio path")
            if not path.is_file():
                raise NoProfile("Profile reference is missing")
            paths.append(path)
        return tuple(paths)

    @staticmethod
    def _with_references(profile: Profile, samples: tuple[ReferenceSample, ...]) -> Profile:
        if not 1 <= len(samples) <= 3:
            raise ValueError("A voice requires one to three references")
        return profile.model_copy(
            update={
                "reference_audio": samples[0].reference_audio,
                "reference_text": samples[0].reference_text,
                "extra_references": samples[1:],
            }
        )

    @staticmethod
    def _reference_filename(index: int) -> str:
        return "reference.wav" if index == 0 else f"reference-{index + 1}.wav"

    def _reference_fingerprint(self, profile: Profile) -> str:
        references = [
            {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": sample.reference_text}
            for sample, path in zip(
                profile.reference_samples, self._profile_reference_paths(profile), strict=True
            )
        ]
        return hashlib.sha256(
            json.dumps(
                references, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def _copy_profile_snapshot(self, profile: Profile, directory: Path) -> Profile:
        samples = []
        for index, (sample, path) in enumerate(
            zip(profile.reference_samples, self._profile_reference_paths(profile), strict=True)
        ):
            name = self._reference_filename(index)
            target = directory / name
            shutil.copyfile(path, target)
            target.chmod(0o600)
            samples.append(
                ReferenceSample(
                    reference_audio=f"jobs/{directory.name}/{name}",
                    reference_text=sample.reference_text,
                )
            )
        return self._with_references(profile, tuple(samples))

    def selected_profile(self, chat_id: int) -> Profile | None:
        with self._transaction() as connection:
            self._ensure_account(connection, chat_id)
            profiles = self._owned_profiles(connection, chat_id)
            row = connection.execute(
                "SELECT selected_profile_id FROM accounts WHERE user_id=?", (chat_id,)
            ).fetchone()
            selected = next((profile for profile in profiles if profile.id == row[0]), None)
            if selected is not None:
                return selected
            selected = profiles[0] if profiles else None
            connection.execute(
                "UPDATE accounts SET selected_profile_id=? WHERE user_id=?",
                (selected.id if selected else None, chat_id),
            )
            return selected

    def select_profile(self, chat_id: int, profile_id: str) -> Profile:
        with self._transaction() as connection:
            profile = self._find_profile(connection, chat_id, profile_id)
            connection.execute(
                "UPDATE accounts SET selected_profile_id=? WHERE user_id=?", (profile.id, chat_id)
            )
            return profile

    def get_profile(self, chat_id: int, profile_id: str) -> Profile:
        with self._transaction() as connection:
            return self._find_profile(connection, chat_id, profile_id)

    def is_default_voice(self, chat_id: int, profile_id: str) -> bool:
        with self._transaction() as connection:
            self._find_profile(connection, chat_id, profile_id)
            row = connection.execute(
                "SELECT is_default FROM profile_owners WHERE profile_id=? AND user_id=?",
                (profile_id, chat_id),
            ).fetchone()
            return bool(row[0])

    def update_voice_settings(
        self,
        chat_id: int,
        profile_id: str,
        *,
        delivery: VoiceDelivery | None = None,
        speed: float | None = None,
        changes: dict[str, object] | None = None,
    ) -> Profile:
        with self._transaction() as connection:
            profile = self._find_profile(connection, chat_id, profile_id)
            updates = dict(changes or {})
            if delivery is not None:
                updates["delivery"] = delivery
            if speed is not None:
                updates["speed"] = speed
            updated = self._changed_voice_settings(profile, updates)
            self._write_profile(connection, updated)
            return updated

    def _find_profile(
        self, connection: sqlite3.Connection, chat_id: int, profile_id: str
    ) -> Profile:
        row = connection.execute(
            "SELECT profile_id FROM profile_owners WHERE profile_id=? AND user_id=?",
            (profile_id, chat_id),
        ).fetchone()
        if row is None:
            raise NoProfile("Profile no longer available")
        path = self.data_dir / "profiles" / str(row[0]) / "profile.json"
        if not path.is_file():
            raise NoProfile("Profile no longer available")
        return self._read_profile(path)

    def _write_profile(self, connection: sqlite3.Connection, profile: Profile) -> None:
        from .config import atomic_private

        path = self.data_dir / "profiles" / profile.id / "profile.json"
        self._profile_writes[connection].setdefault(path, path.read_bytes())
        atomic_private(path, profile.model_dump_json().encode())
        self._sync_voice_metadata(connection, profile)

    @staticmethod
    def _changed_voice_settings(profile: Profile, changes: dict[str, object]) -> Profile:
        from pydantic import TypeAdapter

        allowed = {
            "delivery",
            "speed",
            "native_tags",
            "style_instruction",
            "temperature",
            "top_p",
            "top_k",
            "seed",
        }
        if not changes.keys() <= allowed:
            raise ValueError("Unknown voice setting")
        values = profile.model_dump()
        values["native_tags"] = profile.effective_native_tags
        values["delivery"] = "neutral"
        if "delivery" in changes:
            delivery: VoiceDelivery = TypeAdapter(VoiceDelivery).validate_python(
                changes["delivery"]
            )
            values["native_tags"] = LEGACY_DELIVERY_TAGS[delivery]
        values.update({key: value for key, value in changes.items() if key != "delivery"})
        return Profile.model_validate(values)

    def toggle_native_tag(self, chat_id: int, profile_id: str, tag: str) -> Profile:
        with self._transaction() as connection:
            profile = self._find_profile(connection, chat_id, profile_id)
            tags = list(profile.effective_native_tags)
            if tag in tags:
                tags.remove(tag)
            else:
                if len(tags) >= 3:
                    raise VoiceTagLimit("At most three native cues are allowed")
                tags.append(tag)
            updated = self._changed_voice_settings(profile, {"native_tags": tags})
            self._write_profile(connection, updated)
            return updated

    def adjust_voice_setting(
        self, chat_id: int, profile_id: str, code: str, direction: int
    ) -> Profile:
        from .voice_controls import PARAMETERS

        if code not in PARAMETERS or direction not in (-1, 1):
            raise ValueError("Unknown parameter adjustment")
        parameter = PARAMETERS[code]
        with self._transaction() as connection:
            profile = self._find_profile(connection, chat_id, profile_id)
            current = getattr(profile, parameter.key)
            value = max(
                parameter.minimum, min(parameter.maximum, current + direction * parameter.step)
            )
            value = int(value) if parameter.key in ("top_k", "seed") else round(value, 2)
            updated = self._changed_voice_settings(profile, {parameter.key: value})
            self._write_profile(connection, updated)
            return updated

    def update_seen(self, update_id: int) -> bool:
        with self._transaction() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM updates WHERE update_id=?", (update_id,)
                ).fetchone()
                is not None
            )

    @staticmethod
    def profile_name(value: str) -> str:
        name = value.strip()
        if not name or len(name) > MAX_PROFILE_NAME or any(ord(char) < 32 for char in name):
            raise ValueError("Profile name must contain 1 to 80 printable characters")
        return name

    def accept_callback(self, callback_id: str) -> bool:
        with self._transaction() as connection:
            return (
                connection.execute(
                    "INSERT OR IGNORE INTO ui_callbacks(callback_id,created) VALUES(?,?)",
                    (callback_id, time.time()),
                ).rowcount
                == 1
            )

    def draft_dir(self, draft_id: str) -> Path:
        self.job_dir(draft_id)  # IDs use the same controlled 32-character representation.
        return self.data_dir / "drafts" / draft_id

    @staticmethod
    def _draft(row: sqlite3.Row) -> ProfileDraft:
        from pydantic import TypeAdapter

        return ProfileDraft(
            str(row["id"]),
            TypeAdapter(DraftStage).validate_python(row["stage"]),
            str(row["profile_id"]) if row["profile_id"] else None,
            float(row["updated"]),
            TypeAdapter(VoiceSettingKey).validate_python(row["setting_key"])
            if row["setting_key"] is not None
            else None,
        )

    def profile_draft(self, chat_id: int) -> ProfileDraft | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM profile_drafts WHERE chat_id=?", (chat_id,)
            ).fetchone()
            return self._draft(row) if row is not None else None

    def reference_change(self, chat_id: int, profile_id: str) -> ReferenceChange | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT r.*,j.status AS job_status,j.failure AS job_failure "
                "FROM reference_changes r LEFT JOIN jobs j ON j.id=r.job_id "
                "WHERE r.chat_id=? AND r.profile_id=?",
                (chat_id, profile_id),
            ).fetchone()
            if row is None:
                return None
            from pydantic import TypeAdapter

            stage = str(row["stage"])
            failure: FailureReason | None = None
            if stage == "processing" and row["job_status"] in ("failed", "cancelled"):
                stage = "failed"
                failure = TypeAdapter(FailureReason).validate_python(
                    row["job_failure"] or "model_error"
                )
            return ReferenceChange(
                str(row["id"]),
                str(row["profile_id"]),
                TypeAdapter(Literal["upload", "processing", "ready", "failed"]).validate_python(
                    stage
                ),
                str(row["job_id"]) if row["job_id"] is not None else None,
                int(row["candidate_count"]),
                float(row["updated"]),
                failure,
            )

    def _reference_change_dir(self, change_id: str) -> Path:
        self.job_dir(change_id)
        return self.data_dir / "reference-changes" / change_id

    def _begin_reference_change(
        self,
        chat_id: int,
        profile_id: str,
        operation: Literal["add", "remove"],
        sample_index: int | None = None,
    ) -> ReferenceChange:
        with self._lock:
            if self.is_default_voice(chat_id, profile_id):
                raise PermissionError("Original recordings of the included voice are private")
            profile = self.get_profile(chat_id, profile_id)
            count = len(profile.reference_samples)
            if operation == "add" and count >= 3:
                raise ValueError("A voice already has three references")
            if operation == "remove" and (
                count <= 1 or type(sample_index) is not int or not 0 <= sample_index < count
            ):
                raise ValueError("At least one reference must remain")
            previous = self.reference_change(chat_id, profile_id)
            if previous is not None:
                if previous.stage != "failed":
                    raise ValueError("A reference change is already in progress")
                self.discard_reference_change(chat_id, previous.id)
            identifier, now = uuid.uuid4().hex, time.time()
            draft = self.profile_draft(chat_id) if operation == "add" else None
            with self._transaction() as connection:
                if draft is not None and draft.stage == "reference":
                    connection.execute(
                        "DELETE FROM reference_changes WHERE id=? AND stage='upload'", (draft.id,)
                    )
                connection.execute(
                    "INSERT INTO reference_changes(id,chat_id,profile_id,stage,operation,"
                    "sample_index,baseline,candidate_count,updated) "
                    "VALUES(?,?,?,'upload',?,?,?,?,?)",
                    (
                        identifier,
                        chat_id,
                        profile_id,
                        operation,
                        sample_index,
                        self._reference_fingerprint(profile),
                        count + (1 if operation == "add" else -1),
                        now,
                    ),
                )
                if operation == "add":
                    connection.execute(
                        "INSERT INTO profile_drafts(chat_id,id,stage,profile_id,updated,"
                        "setting_key) "
                        "VALUES(?,?,'reference',?,?,NULL) ON CONFLICT(chat_id) DO UPDATE SET "
                        "id=excluded.id,stage=excluded.stage,profile_id=excluded.profile_id,"
                        "updated=excluded.updated,setting_key=NULL",
                        (chat_id, identifier, profile_id, now),
                    )
            if draft is not None:
                directory = self.draft_dir(draft.id)
                if directory.exists():
                    shutil.rmtree(directory)
            return ReferenceChange(
                identifier,
                profile_id,
                "upload",
                None,
                count + (1 if operation == "add" else -1),
                now,
            )

    def begin_reference_add(self, chat_id: int, profile_id: str) -> ReferenceChange:
        return self._begin_reference_change(chat_id, profile_id, "add")

    def enqueue_reference_add(
        self,
        update_id: int,
        chat_id: int,
        change_id: str,
        source: Path,
    ) -> str | None:
        return self.enqueue(
            update_id, chat_id, "reference", source=source, reference_change_id=change_id
        )

    def enqueue_reference_remove(self, chat_id: int, profile_id: str, index: int) -> str:
        change = self._begin_reference_change(chat_id, profile_id, "remove", index)
        try:
            job_id = self.enqueue(None, chat_id, "reference", reference_change_id=change.id)
            if job_id is None:
                raise ValueError("Reference change could not be queued")
            return job_id
        except BaseException:
            self.discard_reference_change(chat_id, change.id)
            raise

    def discard_reference_change(self, chat_id: int, change_id: str) -> bool:
        directory = self._reference_change_dir(change_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT job_id FROM reference_changes WHERE id=? AND chat_id=?",
                (change_id, chat_id),
            ).fetchone()
            if row is None:
                return False
            if row["job_id"] is not None:
                connection.execute(
                    "UPDATE jobs SET cancel_requested=1,updated=? "
                    "WHERE id=? AND status IN ('running','sending')",
                    (time.time(), row["job_id"]),
                )
                connection.execute(
                    "UPDATE jobs SET status='cancelled',cancel_requested=1,text=NULL,updated=? "
                    "WHERE id=? AND status IN ('queued','ready')",
                    (time.time(), row["job_id"]),
                )
                cancelled = connection.execute(
                    "SELECT id FROM jobs WHERE id=? AND status='cancelled'", (row["job_id"],)
                ).fetchone()
                if cancelled is not None:
                    self._remove_input(str(row["job_id"]))
                    (self.job_dir(str(row["job_id"])) / "result.ogg").unlink(missing_ok=True)
            connection.execute(
                "DELETE FROM profile_drafts WHERE chat_id=? AND id=?", (chat_id, change_id)
            )
            connection.execute(
                "DELETE FROM reference_changes WHERE id=? AND chat_id=?", (change_id, chat_id)
            )
            if directory.exists():
                shutil.rmtree(directory)
        return True

    def reference_sample_path(self, chat_id: int, profile_id: str, index: int) -> Path:
        with self._lock:
            if self.is_default_voice(chat_id, profile_id):
                raise PermissionError("Original recordings of the included voice are private")
            paths = self._profile_reference_paths(self.get_profile(chat_id, profile_id))
            if type(index) is not int or not 0 <= index < len(paths):
                raise ValueError("Unknown reference sample")
            return paths[index]

    def reference_change_preview_path(self, chat_id: int, change_id: str) -> Path:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT id FROM reference_changes WHERE id=? AND chat_id=? AND stage='ready'",
                (change_id, chat_id),
            ).fetchone()
            if row is None:
                raise NoProfile("Reference preview is no longer available")
            path = self._reference_change_dir(change_id) / "example.ogg"
            if path.is_symlink() or not path.is_file():
                raise NoProfile("Reference preview is no longer available")
            return path

    def begin_profile_draft(
        self,
        chat_id: int,
        stage: DraftStage = "upload",
        profile_id: str | None = None,
        setting_key: VoiceSettingKey | None = None,
    ) -> ProfileDraft:
        with self._lock:
            if (stage == "setting") != (setting_key is not None):
                raise ValueError("Setting draft requires a setting key")
            if stage in ("rename", "delete", "setting", "reference") and not any(
                profile.id == profile_id for profile in self.profiles(chat_id)
            ):
                raise NoProfile("Profile no longer available")
            previous = self.profile_draft(chat_id)
            if previous is not None and previous.stage in ("upload", "name") and stage == "upload":
                return previous
            draft = ProfileDraft(uuid.uuid4().hex, stage, profile_id, time.time(), setting_key)
            with self._transaction() as connection:
                if previous is not None and previous.stage == "reference":
                    connection.execute(
                        "DELETE FROM reference_changes WHERE id=? AND stage='upload'",
                        (previous.id,),
                    )
                connection.execute(
                    "INSERT INTO profile_drafts(chat_id,id,stage,profile_id,updated,setting_key) "
                    "VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(chat_id) DO UPDATE SET id=excluded.id,stage=excluded.stage,"
                    "profile_id=excluded.profile_id,updated=excluded.updated,setting_key=excluded.setting_key",
                    (
                        chat_id,
                        draft.id,
                        draft.stage,
                        draft.profile_id,
                        draft.updated,
                        draft.setting_key,
                    ),
                )
            if previous is not None:
                directory = self.draft_dir(previous.id)
                if directory.exists():
                    shutil.rmtree(directory)
            return draft

    def save_voice_setting(
        self, update_id: int, chat_id: int, draft_id: str, value: str
    ) -> Profile | None:
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM updates WHERE update_id=?", (update_id,)
            ).fetchone():
                return None
            row = connection.execute(
                "SELECT * FROM profile_drafts WHERE chat_id=? AND id=? AND stage='setting'",
                (chat_id, draft_id),
            ).fetchone()
            if row is None:
                return None
            draft = self._draft(row)
            if draft.profile_id is None or draft.setting_key is None:
                raise ValueError("Invalid setting draft")
            profile = self._find_profile(connection, chat_id, draft.profile_id)
            key = draft.setting_key
            if key == "native_tag":
                changes: dict[str, object] = {
                    "native_tags": [*profile.effective_native_tags, value]
                }
            elif key in ("temperature", "top_p"):
                changes = {key: float(value.strip().replace(",", "."))}
            elif key in ("top_k", "seed"):
                changes = {key: int(value.strip())}
            else:
                changes = {key: value}
            updated = self._changed_voice_settings(profile, changes)
            self._write_profile(connection, updated)
            connection.execute(
                "DELETE FROM profile_drafts WHERE chat_id=? AND id=?", (chat_id, draft_id)
            )
            connection.execute(
                "INSERT INTO updates(update_id,job_id) VALUES(?,?)",
                (update_id, "setting:" + draft_id),
            )
            return updated

    def cancel_profile_draft(self, chat_id: int, draft_id: str) -> bool:
        draft = self.profile_draft(chat_id)
        if draft is not None and draft.id == draft_id and draft.stage == "reference":
            return self.discard_reference_change(chat_id, draft_id)
        with self._transaction() as connection:
            removed = (
                connection.execute(
                    "DELETE FROM profile_drafts WHERE chat_id=? AND id=?",
                    (chat_id, draft_id),
                ).rowcount
                == 1
            )
            directory = self.draft_dir(draft_id)
            if removed and directory.exists():
                shutil.rmtree(directory)
            return removed

    def save_profile_upload(
        self,
        update_id: int,
        chat_id: int,
        draft_id: str,
        source: Path,
    ) -> bool:
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM updates WHERE update_id=?", (update_id,)
            ).fetchone():
                return False
            row = connection.execute(
                "SELECT id FROM profile_drafts WHERE chat_id=? AND id=? AND stage='upload'",
                (chat_id, draft_id),
            ).fetchone()
            if row is None:
                return False
            directory = self.draft_dir(draft_id)
            directory.mkdir(mode=0o700, exist_ok=True)
            shutil.copyfile(source, directory / "input.audio")
            connection.execute(
                "UPDATE profile_drafts SET stage='name',updated=? WHERE chat_id=? AND id=?",
                (time.time(), chat_id, draft_id),
            )
            connection.execute(
                "INSERT INTO updates(update_id,job_id) VALUES(?,?)",
                (update_id, "draft:" + draft_id),
            )
            return True

    def rename_profile(self, update_id: int, chat_id: int, draft_id: str, name: str) -> bool:
        name = self.profile_name(name)
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM updates WHERE update_id=?", (update_id,)
            ).fetchone():
                return False
            row = connection.execute(
                "SELECT profile_id FROM profile_drafts WHERE chat_id=? AND id=? AND stage='rename'",
                (chat_id, draft_id),
            ).fetchone()
            if row is None:
                return False
            profile = self._find_profile(connection, chat_id, str(row[0]))
            if profile is None:
                raise NoProfile("Profile no longer available")
            updated = profile.model_copy(update={"name": name})
            self._write_profile(connection, updated)
            connection.execute(
                "DELETE FROM profile_drafts WHERE chat_id=? AND id=?", (chat_id, draft_id)
            )
            connection.execute(
                "INSERT INTO updates(update_id,job_id) VALUES(?,?)",
                (update_id, "rename:" + draft_id),
            )
            return True

    def delete_profile(self, chat_id: int, draft_id: str) -> bool:
        tomb = self.data_dir / "profile-trash" / draft_id
        moved: Path | None = None
        changes: list[sqlite3.Row] = []
        try:
            with self._transaction() as connection:
                row = connection.execute(
                    "SELECT profile_id FROM profile_drafts WHERE chat_id=? AND id=? "
                    "AND stage='delete'",
                    (chat_id, draft_id),
                ).fetchone()
                if row is None:
                    return False
                profile = self._find_profile(connection, chat_id, str(row[0]))
                if profile is None:
                    raise NoProfile("Profile no longer available")
                self.draft_dir(draft_id)
                moved = self.data_dir / "profiles" / profile.id
                moved.rename(tomb)
                changes = connection.execute(
                    "SELECT id,job_id FROM reference_changes WHERE profile_id=?", (profile.id,)
                ).fetchall()
                for change in changes:
                    connection.execute(
                        "UPDATE jobs SET cancel_requested=1,updated=? "
                        "WHERE id=? AND status IN ('running','sending')",
                        (time.time(), change["job_id"]),
                    )
                    connection.execute(
                        "UPDATE jobs SET status='cancelled',cancel_requested=1,text=NULL,updated=? "
                        "WHERE id=? AND status IN ('queued','ready')",
                        (time.time(), change["job_id"]),
                    )
                connection.execute(
                    "DELETE FROM reference_changes WHERE profile_id=?", (profile.id,)
                )
                connection.execute(
                    "DELETE FROM profile_owners WHERE profile_id=? AND user_id=?",
                    (profile.id, chat_id),
                )
                remaining = self._owned_profiles(connection, chat_id)
                connection.execute(
                    "UPDATE accounts SET selected_profile_id=? "
                    "WHERE user_id=? AND selected_profile_id=?",
                    (remaining[0].id if remaining else None, chat_id, profile.id),
                )
                connection.execute(
                    "DELETE FROM profile_drafts WHERE chat_id=? AND id=?", (chat_id, draft_id)
                )
        except BaseException:
            if moved is not None and tomb.exists() and not moved.exists():
                tomb.rename(moved)
            raise
        shutil.rmtree(tomb)
        for change in changes:
            candidate = self._reference_change_dir(str(change["id"]))
            if candidate.exists():
                shutil.rmtree(candidate)
        return True

    def finish_profile(self, job_id: str, lease: str, prepared: Path, transcript: str) -> Profile:
        from .config import atomic_private

        transcript = transcript.strip()
        if not transcript or len(transcript) > MAX_TEXT:
            raise ValueError("Invalid reference transcript")
        created: Path | None = None
        try:
            with self._transaction() as connection:
                row = self._leased(connection, job_id, lease)
                if row["cancel_requested"]:
                    raise StaleLease("Cancellation has been requested")
                if row["kind"] != "profile":
                    raise ValueError("Job is not a profile preparation")
                profile = Profile(
                    id="voice-" + job_id[:12],
                    name=self.profile_name(str(row["text"])),
                    reference_audio="reference.wav",
                    reference_text=transcript,
                )
                target = self.data_dir / "profiles" / profile.id
                if target.exists():
                    raise ValueError("Profile already exists")
                atomic_private(prepared / "profile.json", profile.model_dump_json().encode())
                shutil.copyfile(prepared / "example.ogg", self.job_dir(job_id) / "result.ogg")
                prepared.rename(target)
                created = target
                connection.execute(
                    "UPDATE jobs SET status='ready',text=NULL,lease=NULL,updated=?,"
                    "result_profile_id=? "
                    "WHERE id=?",
                    (time.time(), profile.id, job_id),
                )
                self._ensure_account(connection, int(row["chat_id"]))
                connection.execute(
                    "INSERT INTO profile_owners(profile_id,user_id) VALUES(?,?)",
                    (profile.id, row["chat_id"]),
                )
                self._sync_voice_metadata(connection, profile)
                connection.execute(
                    "UPDATE generation_history SET result_voice_snapshot_json=? WHERE job_id=?",
                    (self._voice_metadata_json(profile), job_id),
                )
                connection.execute(
                    "UPDATE accounts SET selected_profile_id=? WHERE user_id=?",
                    (profile.id, row["chat_id"]),
                )
        except BaseException:
            if created is not None and created.exists():
                shutil.rmtree(created)
            raise
        self._remove_input(job_id)
        return profile

    def reference_job(self, job_id: str, lease: str) -> tuple[Profile, bool]:
        with self._transaction() as connection:
            row = self._leased(connection, job_id, lease)
            if row["kind"] != "reference" or row["text"] not in ("add", "remove"):
                raise ValueError("Job is not a reference preparation")
            if row["cancel_requested"]:
                raise StaleLease("Reference change was cancelled")
            return Profile.model_validate_json(str(row["profile"])), row["text"] == "add"

    def finish_reference(
        self,
        job_id: str,
        lease: str,
        prepared: Path,
        reference_texts: list[str],
    ) -> Profile:
        """Publish a private candidate and its preview; the selected voice is untouched."""
        from .config import atomic_private

        created: Path | None = None
        try:
            with self._transaction() as connection:
                row = self._leased(connection, job_id, lease)
                change = connection.execute(
                    "SELECT * FROM reference_changes WHERE job_id=? AND stage='processing'",
                    (job_id,),
                ).fetchone()
                if row["cancel_requested"] or change is None:
                    raise StaleLease("Reference change is no longer active")
                if row["kind"] != "reference" or row["text"] not in ("add", "remove"):
                    raise ValueError("Job is not a reference preparation")
                try:
                    current = self._find_profile(
                        connection, int(change["chat_id"]), str(change["profile_id"])
                    )
                except NoProfile as error:
                    raise StaleLease("Voice was deleted") from error
                if self._reference_fingerprint(current) != change["baseline"]:
                    raise ValueError("Reference samples have changed")
                snapshot = Profile.model_validate_json(str(row["profile"]))
                expected_count = len(snapshot.reference_samples) + (
                    1 if row["text"] == "add" else 0
                )
                if not 1 <= expected_count <= 3 or len(reference_texts) != expected_count:
                    raise ValueError("Unexpected reference count")
                samples = tuple(
                    ReferenceSample(
                        reference_audio=self._reference_filename(index), reference_text=text
                    )
                    for index, text in enumerate(reference_texts)
                )
                for index, sample in enumerate(snapshot.reference_samples):
                    original = (self.data_dir / sample.reference_audio).resolve()
                    if not original.is_relative_to(self.job_dir(job_id).resolve()):
                        raise ValueError("Reference snapshot is outside its job")
                    received = prepared / self._reference_filename(index)
                    if (
                        received.is_symlink()
                        or not received.is_file()
                        or original.read_bytes() != received.read_bytes()
                        or sample.reference_text != reference_texts[index]
                    ):
                        raise ValueError("Original reference was modified")
                allowed = {sample.reference_audio for sample in samples} | {
                    "example.ogg",
                    "whisper-context.json",
                }
                if prepared.is_symlink() or any(
                    path.name not in allowed or path.is_symlink() or not path.is_file()
                    for path in prepared.iterdir()
                ):
                    raise ValueError("Invalid candidate files")
                for sample in samples:
                    path = prepared / sample.reference_audio
                    if not path.is_file():
                        raise ValueError("Candidate reference is missing")
                    path.chmod(0o600)
                candidate = self._with_references(snapshot, samples)
                atomic_private(prepared / "profile.json", candidate.model_dump_json().encode())
                target = self._reference_change_dir(str(change["id"]))
                if target.exists() or target.is_symlink():
                    raise ValueError("Candidate already exists")
                shutil.copyfile(prepared / "example.ogg", self.job_dir(job_id) / "result.ogg")
                prepared.rename(target)
                created = target
                now = time.time()
                connection.execute(
                    "UPDATE generation_history SET result_voice_snapshot_json=? WHERE job_id=?",
                    (self._voice_metadata_json(candidate), job_id),
                )
                connection.execute(
                    "UPDATE reference_changes SET stage='ready',updated=? WHERE id=?",
                    (now, change["id"]),
                )
                connection.execute(
                    "UPDATE jobs SET status='ready',text=NULL,lease=NULL,updated=?,"
                    "result_profile_id=? WHERE id=?",
                    (now, snapshot.id, job_id),
                )
        except BaseException:
            if created is not None and created.exists():
                shutil.rmtree(created)
            raise
        self._remove_input(job_id)
        return candidate

    def apply_reference_change(self, chat_id: int, change_id: str) -> Profile | None:
        """Swap the prepared voice only after approval, preserving current controls and name."""
        from .config import atomic_private

        prepared = self._reference_change_dir(change_id)
        tomb = self.data_dir / "profile-trash" / change_id
        marker = tomb.with_name(f"reference-{change_id}.json")
        target: Path | None = None
        moved_old = False
        moved_new = False
        try:
            with self._transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM reference_changes WHERE id=? AND chat_id=? AND stage='ready'",
                    (change_id, chat_id),
                ).fetchone()
                if row is None:
                    return None
                current = self._find_profile(connection, chat_id, str(row["profile_id"]))
                if self._reference_fingerprint(current) != row["baseline"]:
                    raise ValueError("Reference samples have changed; prepare a new comparison")
                if prepared.is_symlink() or not prepared.is_dir():
                    raise NoProfile("Reference comparison is no longer available")
                candidate = Profile.model_validate_json((prepared / "profile.json").read_text())
                if candidate.id != current.id:
                    raise ValueError("Candidate belongs to another voice")
                updated = self._with_references(current, candidate.reference_samples)
                for sample in updated.reference_samples:
                    path = prepared / sample.reference_audio
                    if path.is_symlink() or not path.is_file() or path.parent != prepared:
                        raise ValueError("Invalid candidate reference path")
                atomic_private(prepared / "profile.json", updated.model_dump_json().encode())
                target = self.data_dir / "profiles" / current.id
                atomic_private(
                    marker, json.dumps({"change_id": change_id, "profile_id": current.id}).encode()
                )
                target.rename(tomb)
                moved_old = True
                prepared.rename(target)
                moved_new = True
                self._sync_voice_metadata(connection, updated)
                connection.execute(
                    "DELETE FROM reference_changes WHERE id=? AND chat_id=?", (change_id, chat_id)
                )
                connection.execute(
                    "DELETE FROM profile_drafts WHERE chat_id=? AND id=?", (chat_id, change_id)
                )
        except BaseException:
            if target is not None:
                if moved_new and target.exists():
                    target.rename(prepared)
                if moved_old and tomb.exists():
                    tomb.rename(target)
            marker.unlink(missing_ok=True)
            raise
        with self._lock:
            if tomb.exists():
                shutil.rmtree(tomb)
            marker.unlink(missing_ok=True)
        return updated

    def enqueue(
        self,
        update_id: int | None,
        chat_id: int,
        kind: JobKind,
        *,
        text: str | None = None,
        source: Path | None = None,
        draft_id: str | None = None,
        profile_id: str | None = None,
        reference_change_id: str | None = None,
        preview: bool = False,
    ) -> str | None:
        self._validate_panel_id(chat_id)
        if preview and (kind != "tts" or profile_id is None or text != PREVIEW_TEXT):
            raise ValueError("A preview requires a specific personal voice")
        if kind == "tts" and (text is None or not text.strip() or len(text) > MAX_TEXT):
            raise ValueError("Invalid text length")
        if kind == "vc" and source is None:
            raise ValueError("Voice source is required")
        if kind == "profile":
            text = self.profile_name(text or "")
            if source is None and draft_id is None:
                raise ValueError("Profile reference is required")
            if profile_id is not None:
                raise ValueError("New profile cannot use an existing profile")
        if (kind == "reference") != (reference_change_id is not None):
            raise ValueError("Reference job requires its pending change")
        if kind == "reference":
            if update_id is not None and self.update_seen(update_id):
                return None
        # Profile selection and file snapshot are protected by the same local lock.
        with self._lock:
            reference_change = None
            if kind == "reference":
                with self._transaction() as connection:
                    reference_change = connection.execute(
                        "SELECT * FROM reference_changes WHERE id=? AND chat_id=? "
                        "AND stage='upload'",
                        (reference_change_id, chat_id),
                    ).fetchone()
                if reference_change is None:
                    raise ValueError("Reference change is no longer available")
                profile = self.get_profile(chat_id, str(reference_change["profile_id"]))
                if self._reference_fingerprint(profile) != reference_change["baseline"]:
                    raise ValueError("Reference samples have changed")
                text = str(reference_change["operation"])
                if (text == "add") != (source is not None):
                    raise ValueError("Reference upload does not match the operation")
                if text == "remove":
                    index = int(reference_change["sample_index"])
                    profile = self._with_references(
                        profile,
                        tuple(
                            sample
                            for position, sample in enumerate(profile.reference_samples)
                            if position != index
                        ),
                    )
            elif kind == "profile":
                profile = None
            elif profile_id is not None:
                profile = self.get_profile(chat_id, profile_id)
            else:
                profile = self.selected_profile(chat_id)
            if profile is None and kind != "profile":
                raise NoProfile("No voice profile configured")
            job_id = uuid.uuid4().hex
            directory = self.job_dir(job_id)
            try:
                with self._transaction() as connection:
                    if connection.execute(
                        "SELECT 1 FROM updates WHERE update_id=?", (update_id,)
                    ).fetchone():
                        return None
                    account = self._account(connection, chat_id)
                    reserve_ms = 0
                    if kind in ("tts", "vc") and not account.is_admin and not preview:
                        if account.available_ms <= 0:
                            raise InsufficientBalance("No synthesis time remains")
                        reserve_ms = min(account.available_ms, MAX_OUTPUT_MS)
                    waiting = connection.execute(
                        "SELECT COUNT(*) FROM jobs WHERE status='queued'"
                    ).fetchone()[0]
                    if waiting >= MAX_WAITING:
                        raise QueueFull("Queue has three waiting jobs")
                    if draft_id is not None:
                        draft = connection.execute(
                            "SELECT id FROM profile_drafts WHERE chat_id=? AND id=? "
                            "AND stage='name'",
                            (chat_id, draft_id),
                        ).fetchone()
                        if draft is None:
                            raise ValueError("Profile draft no longer active")
                        source = self.draft_dir(draft_id) / "input.audio"
                    directory.mkdir(mode=0o700)
                    snapshot = None
                    if profile is not None:
                        snapshot = self._copy_profile_snapshot(profile, directory)
                    if source is not None:
                        shutil.copyfile(source, directory / "input.audio")
                    now = time.time()
                    connection.execute(
                        "INSERT INTO jobs(id,chat_id,kind,text,profile,status,created,updated) "
                        "VALUES(?,?,?,?,?,'queued',?,?)",
                        (
                            job_id,
                            chat_id,
                            kind,
                            text,
                            snapshot.model_dump_json() if snapshot is not None else "null",
                            now,
                            now,
                        ),
                    )
                    if kind in ("tts", "vc"):
                        connection.execute(
                            "INSERT INTO job_billing(job_id,user_id,exempt,preview,"
                            "reserved_ms,state) VALUES(?,?,?,?,?,'reserved')",
                            (job_id, chat_id, int(account.is_admin), int(preview), reserve_ms),
                        )
                    self._record_generation(connection, job_id, snapshot)
                    if reference_change is not None:
                        changed = connection.execute(
                            "UPDATE reference_changes SET stage='processing',job_id=?,updated=? "
                            "WHERE id=? AND chat_id=? AND stage='upload'",
                            (job_id, now, reference_change_id, chat_id),
                        ).rowcount
                        if changed != 1:
                            raise ValueError("Reference change is no longer available")
                        connection.execute(
                            "UPDATE jobs SET result_profile_id=? WHERE id=?",
                            (reference_change["profile_id"], job_id),
                        )
                        connection.execute(
                            "DELETE FROM profile_drafts WHERE chat_id=? AND id=?",
                            (chat_id, reference_change_id),
                        )
                    if update_id is not None:
                        connection.execute(
                            "INSERT INTO updates(update_id,job_id) VALUES(?,?)", (update_id, job_id)
                        )
                    if draft_id is not None:
                        connection.execute("DELETE FROM profile_drafts WHERE chat_id=?", (chat_id,))
            except BaseException:
                if directory.exists():
                    shutil.rmtree(directory)
                raise
            if draft_id is not None:
                shutil.rmtree(self.draft_dir(draft_id), ignore_errors=False)
        return job_id

    def job_dir(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
            raise ValueError("Invalid job identifier")
        return self.data_dir / "jobs" / job_id

    def claim(self) -> WorkerJob | None:
        now = time.time()
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone():
                return None
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            lease = secrets.token_urlsafe(32)
            connection.execute(
                "UPDATE jobs SET status='running',lease=?,lease_until=?,started=?,updated=? "
                "WHERE id=?",
                (lease, now + LEASE_SECONDS, now, now, row["id"]),
            )
            return WorkerJob.model_validate(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "text": row["text"],
                    "profile": json.loads(row["profile"]),
                    "lease": lease,
                }
            )

    def check_lease(self, job_id: str, lease: str) -> bool:
        with self._transaction() as connection:
            row = self._leased(connection, job_id, lease)
            return bool(row["cancel_requested"])

    @staticmethod
    def _leased(connection: sqlite3.Connection, job_id: str, lease: str) -> sqlite3.Row:
        row: sqlite3.Row | None = connection.execute(
            "SELECT * FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        now = time.time()
        if (
            row is None
            or row["status"] != "running"
            or not lease
            or not secrets.compare_digest(row["lease"] or "", lease)
            or row["lease_until"] < now
            or row["started"] + JOB_TIMEOUT_SECONDS < now
        ):
            raise StaleLease("Job lease is no longer active")
        return row

    def heartbeat(self, job_id: str, lease: str) -> bool:
        with self._transaction() as connection:
            row = self._leased(connection, job_id, lease)
            connection.execute(
                "UPDATE jobs SET lease_until=? WHERE id=?", (time.time() + LEASE_SECONDS, job_id)
            )
            return bool(row["cancel_requested"])

    def finish(self, job_id: str, lease: str, result: Path, *, duration_ms: int) -> None:
        """Charge the verified output duration once, atomically with publishing its result."""
        if type(duration_ms) is not int or not 0 < duration_ms <= MAX_OUTPUT_MS:
            raise ValueError("Invalid output duration")
        insufficient = False
        with self._transaction() as connection:
            row = self._leased(connection, job_id, lease)
            if row["cancel_requested"]:
                raise StaleLease("Cancellation has been requested")
            if row["kind"] in ("profile", "reference"):
                raise ValueError("Profile preparation requires its complete result")
            history = connection.execute(
                "SELECT historical,transcript FROM generation_history WHERE job_id=?", (job_id,)
            ).fetchone()
            if row["kind"] == "vc" and not history[0] and history[1] is None:
                raise ValueError("Recognized text must be recorded before synthesis completes")
            billing = connection.execute(
                "SELECT * FROM job_billing WHERE job_id=? AND state='reserved'", (job_id,)
            ).fetchone()
            if billing is None:
                raise StaleLease("The job has no active billing reservation")
            if billing["preview"] and duration_ms > MAX_PREVIEW_MS:
                raise ValueError("Preview exceeds its allowed duration")
            connection.execute(
                "UPDATE generation_history SET duration_ms=? WHERE job_id=?", (duration_ms, job_id)
            )
            charged = 0 if billing["exempt"] or billing["preview"] else duration_ms
            if charged > int(billing["reserved_ms"]):
                insufficient = True
            if not insufficient and charged:
                insufficient = (
                    connection.execute(
                        "UPDATE accounts SET balance_ms=balance_ms-? "
                        "WHERE user_id=? AND balance_ms>=?",
                        (charged, row["chat_id"], charged),
                    ).rowcount
                    != 1
                )
            if insufficient:
                connection.execute(
                    "UPDATE jobs SET status='failed',failure='insufficient_balance',"
                    "text=NULL,lease=NULL,updated=? WHERE id=?",
                    (time.time(), job_id),
                )
                self._release_reservations(connection)
            else:
                result.replace(self.job_dir(job_id) / "result.ogg")
                connection.execute(
                    "UPDATE job_billing SET state='settled',reserved_ms=0,charged_ms=? "
                    "WHERE job_id=?",
                    (charged, job_id),
                )
                connection.execute(
                    "UPDATE jobs SET status='ready',text=NULL,lease=NULL,updated=? WHERE id=?",
                    (time.time(), job_id),
                )
        self._remove_input(job_id)
        if insufficient:
            raise InsufficientBalance("Generated speech exceeds the available balance")

    def fail(self, job_id: str, lease: str, reason: FailureReason) -> None:
        with self._transaction() as connection:
            self._leased(connection, job_id, lease)
            connection.execute(
                "UPDATE jobs SET status='failed',failure=?,text=NULL,lease=NULL,updated=? "
                "WHERE id=?",
                (reason, time.time(), job_id),
            )
            self._release_reservations(connection)
        self._remove_input(job_id)

    def request_cancel(self, job_id: str, chat_id: int) -> str:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT status FROM jobs WHERE id=? AND chat_id=?", (job_id, chat_id)
            ).fetchone()
            if row is None:
                return "missing"
            if row[0] == "queued":
                connection.execute(
                    "UPDATE jobs SET status='cancelled',text=NULL,updated=? WHERE id=?",
                    (time.time(), job_id),
                )
            elif row[0] == "running":
                connection.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
                return "requested"
            else:
                return "finished"
            self._release_reservations(connection)
        self._remove_input(job_id)
        return "cancelled"

    def confirm_cancel(self, job_id: str, lease: str) -> None:
        with self._transaction() as connection:
            row = self._leased(connection, job_id, lease)
            if not row["cancel_requested"]:
                raise StaleLease("Cancellation was not requested")
            connection.execute(
                "UPDATE jobs SET status='cancelled',text=NULL,lease=NULL,updated=? WHERE id=?",
                (time.time(), job_id),
            )
            self._release_reservations(connection)
        self._remove_input(job_id)

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        # Pydantic validates persisted state instead of unchecked typing casts.
        from pydantic import TypeAdapter

        return Job(
            id=str(row["id"]),
            chat_id=int(row["chat_id"]),
            status=TypeAdapter(JobStatus).validate_python(row["status"]),
            kind=TypeAdapter(JobKind).validate_python(row["kind"]),
            created=float(row["created"]),
            updated=float(row["updated"]),
            failure=TypeAdapter(FailureReason | None).validate_python(row["failure"]),
            cancel_requested=bool(row["cancel_requested"]),
            profile_id=str(row["result_profile_id"]) if row["result_profile_id"] else None,
        )

    def list_jobs(self, chat_id: int, limit: int = 8) -> list[Job]:
        with self._transaction() as connection:
            return [
                self._job(row)
                for row in connection.execute(
                    "SELECT * FROM jobs WHERE chat_id=? ORDER BY created DESC LIMIT ?",
                    (chat_id, limit),
                )
            ]

    def delivery_claim(self) -> Job | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='ready' ORDER BY created LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE jobs SET status='sending',updated=? WHERE id=?", (time.time(), row["id"])
            )
            return self._job(row)

    @staticmethod
    def _validate_status_state(state: str) -> None:
        from pydantic import TypeAdapter

        status, separator, cancelled = state.partition(":")
        if separator != ":" or cancelled not in ("0", "1"):
            raise ValueError("Invalid displayed job state")
        TypeAdapter(JobStatus).validate_python(status)

    @staticmethod
    def _detach_panel_statuses(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE jobs SET status_message_id=NULL,status_notified='unavailable' "
            "WHERE status_message_id IS NOT NULL AND EXISTS(SELECT 1 FROM ui_panels p "
            "WHERE p.chat_id=jobs.chat_id AND p.message_id=jobs.status_message_id)"
        )

    def status_entry(self, job_id: str, chat_id: int) -> tuple[Job, int | None] | None:
        """Read one person's job and its separate processing notice, never a menu anchor."""
        self._validate_panel_id(chat_id)
        self.job_dir(job_id)
        with self._transaction() as connection:
            self._detach_panel_statuses(connection)
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=? AND chat_id=?", (job_id, chat_id)
            ).fetchone()
            if row is None:
                return None
            anchor = row["status_message_id"]
            return self._job(row), int(anchor) if anchor is not None else None

    def claim_status(self, job_id: str, chat_id: int) -> bool:
        """Claim one initial send; an uncertain Telegram response never triggers a duplicate."""
        self._validate_panel_id(chat_id)
        self.job_dir(job_id)
        with self._transaction() as connection:
            return (
                connection.execute(
                    "UPDATE jobs SET status_notified='announcing' WHERE id=? AND chat_id=? "
                    "AND status_message_id IS NULL AND status_notified IS NULL",
                    (job_id, chat_id),
                ).rowcount
                == 1
            )

    def release_status_claim(self, job_id: str, chat_id: int) -> bool:
        """Allow a retry only when Telegram definitively rejected the initial send."""
        self._validate_panel_id(chat_id)
        self.job_dir(job_id)
        with self._transaction() as connection:
            return (
                connection.execute(
                    "UPDATE jobs SET status_notified=NULL WHERE id=? AND chat_id=? "
                    "AND status_message_id IS NULL AND status_notified='announcing'",
                    (job_id, chat_id),
                ).rowcount
                == 1
            )

    def attach_status(self, job_id: str, chat_id: int, message_id: int, state: str) -> bool:
        self._validate_panel_id(chat_id)
        self._validate_panel_id(message_id)
        self.job_dir(job_id)
        self._validate_status_state(state)
        with self._transaction() as connection:
            return (
                connection.execute(
                    "UPDATE jobs SET status_message_id=?,status_notified=? "
                    "WHERE id=? AND chat_id=? AND status_message_id IS NULL "
                    "AND status_notified='announcing' "
                    "AND NOT EXISTS(SELECT 1 FROM ui_panels p "
                    "WHERE p.chat_id=? AND p.message_id=?) "
                    "AND NOT EXISTS(SELECT 1 FROM jobs j WHERE j.chat_id=? "
                    "AND j.status_message_id=? AND j.id!=?)",
                    (
                        message_id,
                        state,
                        job_id,
                        chat_id,
                        chat_id,
                        message_id,
                        chat_id,
                        message_id,
                        job_id,
                    ),
                ).rowcount
                == 1
            )

    def drop_status(self, job_id: str, chat_id: int, message_id: int) -> bool:
        """Forget a deleted/uneditable notice without automatically recreating it."""
        self._validate_panel_id(chat_id)
        self._validate_panel_id(message_id)
        self.job_dir(job_id)
        with self._transaction() as connection:
            return (
                connection.execute(
                    "UPDATE jobs SET status_message_id=NULL,status_notified='unavailable' "
                    "WHERE id=? AND chat_id=? AND status_message_id=?",
                    (job_id, chat_id, message_id),
                ).rowcount
                == 1
            )

    def status_updates(self) -> list[tuple[Job, int | None, str]]:
        with self._transaction() as connection:
            self._detach_panel_statuses(connection)
            rows = connection.execute(
                "SELECT * FROM jobs WHERE "
                "(status_message_id IS NULL AND status_notified IS NULL) OR "
                "(status_message_id IS NOT NULL AND "
                "(status_notified IS NULL "
                "OR status_notified != status || ':' || cancel_requested)) "
                "ORDER BY created,id"
            )
            return [
                (
                    self._job(row),
                    int(row["status_message_id"]) if row["status_message_id"] is not None else None,
                    f"{row['status']}:{row['cancel_requested']}",
                )
                for row in rows
            ]

    def status_notified(self, job_id: str, message_id: int, state: str) -> bool:
        self._validate_panel_id(message_id)
        self.job_dir(job_id)
        self._validate_status_state(state)
        with self._transaction() as connection:
            return (
                connection.execute(
                    "UPDATE jobs SET status_notified=? WHERE id=? AND status_message_id=? "
                    "AND status || ':' || cancel_requested=? "
                    "AND NOT EXISTS(SELECT 1 FROM ui_panels p "
                    "WHERE p.chat_id=jobs.chat_id AND p.message_id=jobs.status_message_id)",
                    (state, job_id, message_id, state),
                ).rowcount
                == 1
            )

    def delivery_done(
        self, job_id: str, message_id: int | None, *, uncertain: bool = False
    ) -> None:
        status = "uncertain" if uncertain else "sent" if message_id is not None else "failed"
        with self._transaction() as connection:
            connection.execute(
                "UPDATE jobs SET status=?,delivered_message_id=?,failure=?,updated=? "
                "WHERE id=? AND status='sending'",
                (
                    status,
                    message_id,
                    "delivery_error" if status == "failed" else None,
                    time.time(),
                    job_id,
                ),
            )

    def retry_delivery(self, job_id: str, chat_id: int, callback_id: str | None = None) -> bool:
        with self._transaction() as connection:
            if (
                callback_id is not None
                and connection.execute(
                    "SELECT 1 FROM delivery_callbacks WHERE callback_id=?", (callback_id,)
                ).fetchone()
            ):
                return False
            if not (self.job_dir(job_id) / "result.ogg").is_file():
                return False
            changed = connection.execute(
                "UPDATE jobs SET status='ready',failure=NULL,updated=? WHERE id=? AND chat_id=? "
                "AND cancel_requested=0 "
                "AND (status IN ('uncertain','sent') OR "
                "(status='failed' AND failure='delivery_error'))",
                (time.time(), job_id, chat_id),
            ).rowcount
            if changed == 1 and callback_id is not None:
                connection.execute(
                    "INSERT INTO delivery_callbacks(callback_id,job_id) VALUES(?,?)",
                    (callback_id, job_id),
                )
            return changed == 1

    def _remove_input(self, job_id: str) -> None:
        for name in ("input.audio", "reference.wav", "reference-2.wav", "reference-3.wav"):
            (self.job_dir(job_id) / name).unlink(missing_ok=True)

    def maintain(self, now: float | None = None) -> None:
        timestamp = time.time() if now is None else now
        with self._transaction() as connection:
            connection.execute(
                "UPDATE jobs SET status='failed',failure='worker_lost',text=NULL,lease=NULL,"
                "updated=? "
                "WHERE status='running' AND (lease_until<? OR started<?)",
                (timestamp, timestamp, timestamp - JOB_TIMEOUT_SECONDS),
            )
            expired = connection.execute(
                "SELECT id FROM jobs WHERE created<? AND status NOT IN ('running','sending')",
                (timestamp - RETENTION_SECONDS,),
            ).fetchall()
            for row in expired:
                connection.execute("DELETE FROM delivery_callbacks WHERE job_id=?", (row[0],))
                connection.execute("DELETE FROM jobs WHERE id=?", (row[0],))
            self._release_reservations(connection)
            for row in connection.execute(
                "SELECT id FROM jobs WHERE status NOT IN ('queued','running')"
            ):
                self._remove_input(str(row[0]))
            live_ids = {str(row[0]) for row in connection.execute("SELECT id FROM jobs")}
            # Keep the lock until orphan removal ends: a concurrent enqueue creates its directory
            # inside this lock, before committing its row.
            for directory in (self.data_dir / "jobs").iterdir():
                if directory.is_dir() and directory.name not in live_ids:
                    shutil.rmtree(directory)
            for path in (self.data_dir / "incoming").glob("*.audio"):
                if path.stat().st_mtime < timestamp - RETENTION_SECONDS:
                    path.unlink(missing_ok=True)
            connection.execute(
                "DELETE FROM profile_drafts WHERE updated<?",
                (timestamp - RETENTION_SECONDS,),
            )
            connection.execute(
                "DELETE FROM ui_callbacks WHERE created<?",
                (timestamp - RETENTION_SECONDS,),
            )
            active_drafts = {
                str(row[0]) for row in connection.execute("SELECT id FROM profile_drafts")
            }
            for directory in (self.data_dir / "drafts").iterdir():
                if directory.is_dir() and directory.name not in active_drafts:
                    shutil.rmtree(directory)
            connection.execute(
                "DELETE FROM reference_changes WHERE updated<? OR "
                "(job_id IS NOT NULL AND job_id NOT IN (SELECT id FROM jobs))",
                (timestamp - RETENTION_SECONDS,),
            )
            change_ids = {
                str(row[0]) for row in connection.execute("SELECT id FROM reference_changes")
            }
            for directory in (self.data_dir / "reference-changes").iterdir():
                if directory.is_dir() and directory.name not in change_ids:
                    shutil.rmtree(directory)
            for directory in (self.data_dir / "profile-trash").iterdir():
                if directory.is_dir():
                    shutil.rmtree(directory)
