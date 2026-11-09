"""Reject distribution archives containing anything outside the reviewed application package."""

from __future__ import annotations

import gzip
import io
import re
import stat
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

PROJECT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_EXPANDED_BYTES = 4 * 1024 * 1024
MAX_ENTRIES = 128
PACKAGE_FILES = {
    "__init__.py",
    "__main__.py",
    "app.py",
    "audio.py",
    "bot.py",
    "cli.py",
    "config.py",
    "protocol.py",
    "py.typed",
    "store.py",
    "voice_controls.py",
    "views.py",
    "whisper_cache.py",
    "worker.py",
}
SDIST_METADATA = {"README.md", "pyproject.toml", "uv.lock", "PKG-INFO", ".gitignore"}
WHEEL_METADATA = {"METADATA", "WHEEL", "RECORD", "entry_points.txt"}
DISTRIBUTION_NOTICES = {"NOTICE", "licenses/FISH-AUDIO-RESEARCH-LICENSE.md"}


class DistributionError(ValueError):
    """Messages are fixed diagnostic codes, never archive paths or file contents."""


@dataclass(frozen=True)
class DistributionCheck:
    kind: Literal["sdist", "wheel"]
    files: int
    uncompressed_bytes: int
    archive_bytes: int


def expected_paths(
    path: Path,
) -> tuple[Literal["sdist", "wheel"], set[str], set[str], str, set[str]]:
    if re.fullmatch(r"shift-[A-Za-z0-9_.+]+\.tar\.gz", path.name):
        root = path.name.removesuffix(".tar.gz")
        files = {f"{root}/src/shift/{name}" for name in PACKAGE_FILES}
        files.update(f"{root}/{name}" for name in SDIST_METADATA)
        notices = {f"{root}/{name}" for name in DISTRIBUTION_NOTICES}
        files.update(notices)
        app = f"{root}/src/shift/app.py"
        kind: Literal["sdist", "wheel"] = "sdist"
    elif match := re.fullmatch(
        r"shift-([A-Za-z0-9_.+]+)-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl",
        path.name,
    ):
        metadata = f"shift-{match.group(1)}.dist-info"
        files = {f"shift/{name}" for name in PACKAGE_FILES}
        files.update(f"{metadata}/{name}" for name in WHEEL_METADATA)
        notices = {f"{metadata}/licenses/{name}" for name in DISTRIBUTION_NOTICES}
        files.update(notices)
        app = "shift/app.py"
        kind = "wheel"
    else:
        raise DistributionError("unexpected_archive_name")
    directories = {
        str(parent)
        for name in files
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    return kind, files, directories, app, notices


def validate_entry(
    name: str,
    size: int,
    *,
    directory: bool,
    allowed_files: set[str],
    allowed_dirs: set[str],
    seen: set[str],
) -> None:
    canonical = name.removesuffix("/") if directory else name
    if (
        not canonical
        or "\\" in canonical
        or "\x00" in canonical
        or PurePosixPath(canonical).is_absolute()
        or str(PurePosixPath(canonical)) != canonical
        or ".." in PurePosixPath(canonical).parts
    ):
        raise DistributionError("noncanonical_archive_path")
    if canonical in seen:
        raise DistributionError("duplicate_archive_entry")
    seen.add(canonical)
    if len(seen) > MAX_ENTRIES:
        raise DistributionError("too_many_archive_entries")
    if canonical not in (allowed_dirs if directory else allowed_files):
        raise DistributionError("archive_path_not_allowed")
    if size < 0 or size > MAX_ARCHIVE_BYTES or (directory and size != 0):
        raise DistributionError("uncompressed_size_limit")


def verify_distribution(path: Path) -> DistributionCheck:
    if path.is_symlink() or not path.is_file():
        raise DistributionError("archive_must_be_regular_file")
    archive_size = path.stat().st_size
    if not 0 < archive_size <= MAX_ARCHIVE_BYTES:
        raise DistributionError("archive_size_limit")
    kind, allowed_files, allowed_dirs, required_app, required_notices = expected_paths(path)
    seen: set[str] = set()
    regular: set[str] = set()
    total_size = 0
    if kind == "sdist":
        # Bound gzip expansion before tarfile parses PAX headers or other extended metadata.
        with gzip.open(path, "rb") as stream:
            expanded = stream.read(MAX_EXPANDED_BYTES + 1)
        if len(expanded) > MAX_EXPANDED_BYTES:
            raise DistributionError("expanded_tar_size_limit")
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
            for member in archive:
                if not (member.isfile() or member.isdir()) or member.issparse():
                    raise DistributionError("archive_links_or_special_files")
                validate_entry(
                    member.name,
                    member.size,
                    directory=member.isdir(),
                    allowed_files=allowed_files,
                    allowed_dirs=allowed_dirs,
                    seen=seen,
                )
                if member.isfile():
                    regular.add(member.name)
                    total_size += member.size
                if total_size > MAX_ARCHIVE_BYTES:
                    raise DistributionError("uncompressed_size_limit")
    else:
        with zipfile.ZipFile(path) as wheel:
            for entry in wheel.infolist():
                file_type = stat.S_IFMT(entry.external_attr >> 16)
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise DistributionError("archive_links_or_special_files")
                validate_entry(
                    entry.filename,
                    entry.file_size,
                    directory=entry.is_dir(),
                    allowed_files=allowed_files,
                    allowed_dirs=allowed_dirs,
                    seen=seen,
                )
                if not entry.is_dir():
                    regular.add(entry.filename)
                    total_size += entry.file_size
                if total_size > MAX_ARCHIVE_BYTES:
                    raise DistributionError("uncompressed_size_limit")
            # Verify compressed members and CRCs only after their aggregate size is bounded.
            for entry in wheel.infolist():
                with wheel.open(entry) as content:
                    if len(content.read(MAX_ARCHIVE_BYTES + 1)) != entry.file_size:
                        raise DistributionError("archive_member_size_mismatch")
    if required_app not in regular:
        raise DistributionError("application_missing_from_archive")
    if not required_notices <= regular:
        raise DistributionError("required_notice_missing_from_archive")
    return DistributionCheck(kind, len(regular), total_size, archive_size)


def main() -> int:
    paths = sorted((PROJECT / "dist").glob("shift-*.tar.gz"))
    paths.extend(sorted((PROJECT / "dist").glob("shift-*.whl")))
    try:
        checks = [verify_distribution(path) for path in paths]
        if {check.kind for check in checks} != {"sdist", "wheel"}:
            raise DistributionError("both_distribution_formats_required")
    except (DistributionError, OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as error:
        reason = str(error) if isinstance(error, DistributionError) else type(error).__name__
        print(f"Проверка дистрибутивов не пройдена: {reason}.", file=sys.stderr)
        return 1
    print(
        f"Дистрибутивы проверены: архивов {len(checks)}, "
        f"файлов {sum(check.files for check in checks)}, "
        f"байт {sum(check.archive_bytes for check in checks)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
