"""Telegram voice bot: one persistent panel, button navigation and real voice results."""

import asyncio
import logging
import random
import secrets
import sqlite3
import time
from collections.abc import Buffer, Callable
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal

from aiogram import Bot
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.types import (
    Audio,
    CallbackQuery,
    Chat,
    Document,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardRemove,
    ReplyParameters,
    Update,
    User,
    Voice,
)
from pydantic import TypeAdapter

from . import views
from .audio import (
    AudioError,
    to_voice,
    validate_enrollment_reference,
    validate_result,
    validate_voice,
)
from .protocol import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_SECONDS,
    MAX_TEXT,
    PREVIEW_TEXT,
    Profile,
    VoiceSettingKey,
)
from .store import (
    InsufficientBalance,
    Job,
    NoProfile,
    ProfileDraft,
    QueueFull,
    ReferenceChange,
    Store,
    VoiceTagLimit,
)
from .voice_controls import CATEGORY_CODES, CUES, PARAMETERS

logger = logging.getLogger("shift.bot")


class LimitedBuffer(BytesIO):
    def write(self, buffer: Buffer, /) -> int:
        if self.tell() + len(memoryview(buffer)) > MAX_AUDIO_BYTES:
            raise AudioError("Audio exceeds size limit")
        return super().write(buffer)


class BotController:
    def __init__(
        self,
        bot: Bot,
        store: Store,
        *,
        worker_available: Callable[[], bool] | None = None,
    ) -> None:
        self.bot = bot
        self.store = store
        self.worker_available = worker_available
        self._panel_locks: dict[int, asyncio.Lock] = {}
        self._retry_at: dict[int, float] = {}
        self._status_locks: dict[int, asyncio.Lock] = {}
        self._status_retry_at: dict[str, float] = {}

    def authorized(self, message: Message) -> bool:
        return self.authorized_sender(message.from_user, message.chat)

    def authorized_sender(self, user: User | None, chat: Chat) -> bool:
        return bool(
            user is not None
            and self.store.allowed_user(user.id)
            and not user.is_bot
            and chat.type == "private"
            and chat.id == user.id
        )

    async def home_panel(self, chat_id: int) -> views.Panel:
        profile = await asyncio.to_thread(self.store.selected_profile, chat_id)
        account = await asyncio.to_thread(self.store.account, chat_id)
        return views.start_panel(
            profile, account, self.worker_available is None or self.worker_available()
        )

    @classmethod
    def job_status_panel(cls, job: Job) -> views.Panel:
        rows: list[list[InlineKeyboardButton]] = []
        if job.status in ("queued", "running"):
            if job.cancel_requested:
                text = "Останавливаю…"
            else:
                text = (
                    "Образец в очереди."
                    if job.status == "queued" and job.kind in ("profile", "reference")
                    else "В очереди."
                    if job.status == "queued"
                    else "Готовлю голос…"
                    if job.kind == "profile"
                    else "Готовлю пример…"
                    if job.kind == "reference"
                    else "Готовлю голосовое…"
                )
                rows.append(
                    [InlineKeyboardButton(text="Отменить", callback_data=f"cancel:{job.id}")]
                )
        elif job.status in ("ready", "sending"):
            text = "Отправляю голосовое…"
        elif job.status == "sent":
            text = "Готово."
        elif job.status == "cancelled":
            text = "Отменено."
        elif job.status == "uncertain":
            text = "Telegram не подтвердил отправку. Проверьте чат: голосовое могло прийти."
        else:
            failure = cls.failure_text(job)
            text = failure[0].upper() + failure[1:] + "."
        if job.status == "uncertain" or (
            job.status == "failed" and job.failure == "delivery_error"
        ):
            rows.append(
                [InlineKeyboardButton(text="Повторить отправку", callback_data=f"retry:{job.id}")]
            )
        return views.Panel(text, rows)

    async def send_notice(
        self, chat_id: int, text: str, *, reply_to_message_id: int | None = None
    ) -> int | None:
        """Place a rejected request's explanation beside that request, outside navigation."""
        try:
            message = await self.bot.send_message(
                chat_id,
                text,
                reply_parameters=ReplyParameters(
                    message_id=reply_to_message_id, allow_sending_without_reply=True
                )
                if reply_to_message_id is not None
                else None,
            )
            return message.message_id
        except (TelegramAPIError, TimeoutError):
            logger.warning("request_notice_failed reason=network_or_api")
            return None

    async def render(
        self, chat_id: int, panel: views.Panel, *, notice: str | None = None
    ) -> int | None:
        if chat_id <= 0:
            return None
        lock = self._panel_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            return await self._render_locked(chat_id, panel, save=True, notice=notice)

    async def move_home_to_bottom(self, chat_id: int) -> None:
        """Only an explicit /start replaces the existing control message."""
        if chat_id <= 0:
            return
        lock = self._panel_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            if time.monotonic() < self._retry_at.get(chat_id, 0):
                return
            previous_anchor = await asyncio.to_thread(self.store.panel_id, chat_id)
            panel = await self.home_panel(chat_id)
            keyboard = panel.keyboard
            try:
                message = await self.bot.send_message(
                    chat_id,
                    panel.text,
                    reply_markup=keyboard,
                    parse_mode="HTML",
                )
            except TelegramRetryAfter as error:
                self._retry_at[chat_id] = time.monotonic() + min(error.retry_after, 60)
                logger.warning("panel_move_failed reason=rate_limited")
                return
            except (TelegramAPIError, TimeoutError):
                # An uncertain send must neither forget nor delete the working old panel.
                logger.warning("panel_move_failed reason=network_or_api")
                return
            await asyncio.to_thread(
                self.store.replace_panel,
                chat_id,
                message.message_id,
                panel.text,
                panel.keyboard.model_dump_json(),
            )
            if previous_anchor is not None and previous_anchor != message.message_id:
                try:
                    await self.bot.delete_message(chat_id, previous_anchor)
                except TelegramBadRequest as error:
                    if "message to delete not found" not in error.message.lower():
                        logger.warning("panel_delete_failed reason=telegram_bad_request")
                except (TelegramAPIError, TimeoutError):
                    # The new anchor is already durable. Never retry or delete another message.
                    logger.warning("panel_delete_failed reason=network_or_api")

    async def _render_locked(
        self, chat_id: int, panel: views.Panel, *, save: bool, notice: str | None = None
    ) -> int | None:
        anchor = await asyncio.to_thread(self.store.panel_id, chat_id)
        if time.monotonic() < self._retry_at.get(chat_id, 0):
            return None
        keyboard = panel.keyboard
        text = (notice + "\n\n" if notice else "") + panel.text
        try:
            if anchor is None:
                message = await self.bot.send_message(
                    chat_id, text, reply_markup=keyboard, parse_mode="HTML"
                )
                anchor = message.message_id
                await asyncio.to_thread(self.store.set_panel, chat_id, anchor)
            else:
                try:
                    await self.bot.edit_message_text(
                        text=text,
                        chat_id=chat_id,
                        message_id=anchor,
                        reply_markup=keyboard,
                        parse_mode="HTML",
                    )
                except TelegramBadRequest as error:
                    reason = error.message.lower()
                    if "message is not modified" not in reason:
                        if (
                            "message to edit not found" in reason
                            or "message can't be edited" in reason
                        ):
                            await asyncio.to_thread(self.store.clear_panel, chat_id, anchor)
                        raise
            if save:
                await asyncio.to_thread(
                    self.store.set_panel_content,
                    chat_id,
                    anchor,
                    panel.text,
                    panel.keyboard.model_dump_json(),
                )
            return anchor
        except TelegramRetryAfter as error:
            self._retry_at[chat_id] = time.monotonic() + min(error.retry_after, 60)
        except TelegramBadRequest as error:
            reason = error.message.lower()
            code = (
                "message_not_editable"
                if "message can't be edited" in reason
                else "message_missing"
                if "message to edit not found" in reason
                else "telegram_bad_request"
            )
            logger.warning("panel_edit_failed reason=%s", code)
        except (TelegramAPIError, TimeoutError):
            logger.warning("panel_edit_failed reason=network_or_api")
        return None

    async def refresh_panel(self, chat_id: int) -> bool:
        if chat_id <= 0:
            return False
        lock = self._panel_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            if await asyncio.to_thread(self.store.panel_id, chat_id) is None:
                return False
            saved = await asyncio.to_thread(self.store.panel_content, chat_id)
            if saved is None:
                panel = await self.home_panel(chat_id)
            else:
                text, markup = saved
                panel = views.Panel(
                    text, InlineKeyboardMarkup.model_validate_json(markup).inline_keyboard
                )
            if panel.text.startswith("<b>Timbre</b>"):
                panel = await self.home_panel(chat_id)
            elif panel.text.startswith("<b>Профиль</b>"):
                account = await asyncio.to_thread(self.store.account, chat_id)
                panel = views.account_panel(account)
            actions = [button.callback_data or "" for row in panel.rows for button in row]
            if any(action.startswith("samplediscard:") for action in actions):
                profile_id = next(
                    (
                        action.removeprefix("card:")
                        for action in actions
                        if action.startswith("card:")
                    ),
                    None,
                )
                if profile_id is not None:
                    try:
                        profile = await self.reference_profile(chat_id, profile_id)
                    except NoProfile:
                        panel = await self.home_panel(chat_id)
                    else:
                        change = await asyncio.to_thread(
                            self.store.reference_change, chat_id, profile_id
                        )
                        panel = views.samples_panel(
                            profile,
                            stage=change.stage if change else None,
                            change_id=change.id if change else None,
                            candidate_count=change.candidate_count if change else None,
                        )
                    return await self._render_locked(chat_id, panel, save=True) is not None
            if saved is not None and (panel.text, panel.keyboard.model_dump_json()) == saved:
                return True
            return await self._render_locked(chat_id, panel, save=True) is not None

    async def say(
        self, chat_id: int, text: str, keyboard: InlineKeyboardMarkup | None = None
    ) -> int | None:
        if keyboard is not None:
            return await self.render(chat_id, views.Panel(text, keyboard.inline_keyboard))
        if chat_id <= 0:
            return None
        lock = self._panel_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            saved = await asyncio.to_thread(self.store.panel_content, chat_id)
            panel = (
                await self.home_panel(chat_id)
                if saved is None
                else views.Panel(
                    saved[0], InlineKeyboardMarkup.model_validate_json(saved[1]).inline_keyboard
                )
            )
            return await self._render_locked(chat_id, panel, save=True, notice=text)

    async def on_message(self, message: Message, event_update: Update) -> None:
        if not self.authorized(message) or message.from_user is None:
            return
        try:
            await asyncio.to_thread(
                self.store.update_account_user,
                message.chat.id,
                username=message.from_user.username,
                display_name=message.from_user.full_name,
            )
            await self._on_message(message, event_update)
        except (OSError, sqlite3.Error):
            logger.error("message_failed reason=storage_error")
            await self.send_notice(
                message.chat.id,
                "Не удалось сохранить изменения. Попробуйте ещё раз чуть позже.",
                reply_to_message_id=message.message_id,
            )
        except ValueError:
            logger.error("message_failed reason=invalid_profile")
            await self.send_notice(
                message.chat.id,
                "Этот голос недоступен. Выберите другой или добавьте новый.",
                reply_to_message_id=message.message_id,
            )

    async def _on_message(self, message: Message, event_update: Update) -> None:
        text = message.text
        command = text.split(maxsplit=1)[0].split("@")[0] if text and text.strip() else ""
        if command == "/start":
            draft = await asyncio.to_thread(self.store.profile_draft, message.chat.id)
            if draft is not None and draft.stage == "reference":
                await asyncio.to_thread(
                    self.store.discard_reference_change, message.chat.id, draft.id
                )
            elif draft is not None and draft.stage in ("setting", "rename", "delete"):
                await asyncio.to_thread(self.store.cancel_profile_draft, message.chat.id, draft.id)
            await self.move_home_to_bottom(message.chat.id)
            return
        if command.startswith("/"):
            await self.send_notice(
                message.chat.id,
                "Откройте меню через /start. Выбрать и настроить голос можно кнопками.",
                reply_to_message_id=message.message_id,
            )
            return
        if await asyncio.to_thread(self.store.update_seen, event_update.update_id):
            return
        try:
            draft = await asyncio.to_thread(self.store.profile_draft, message.chat.id)
            if draft is not None:
                await self.handle_profile_message(message, event_update.update_id, draft)
                return
            if text is not None:
                if not text.strip() or len(text) > MAX_TEXT:
                    await self.send_notice(
                        message.chat.id,
                        "Пришлите текст длиной от 1 до 2000 символов.",
                        reply_to_message_id=message.message_id,
                    )
                    return
                job_id = await asyncio.to_thread(
                    self.store.enqueue, event_update.update_id, message.chat.id, "tts", text=text
                )
            elif message.voice is not None:
                job_id = await self.receive_voice(message, event_update.update_id)
            else:
                await self.send_notice(
                    message.chat.id,
                    "Пришлите текст или голосовое. "
                    "Чтобы добавить голос из аудиофайла, нажмите «Добавить голос».",
                    reply_to_message_id=message.message_id,
                )
                return
            if job_id is not None:
                await self.announce_job(
                    message.chat.id, job_id, reply_to_message_id=message.message_id
                )
        except InsufficientBalance:
            await self.send_notice(
                message.chat.id,
                "Для этого сообщения не хватает времени на балансе. Пополнение появится позже.",
                reply_to_message_id=message.message_id,
            )
        except QueueFull:
            await self.send_notice(
                message.chat.id,
                "Очередь заполнена. Дождитесь результата или отмените одну из своих озвучек.",
                reply_to_message_id=message.message_id,
            )
        except NoProfile:
            await self.send_notice(
                message.chat.id,
                "Сначала добавьте голос в разделе «Мои голоса».",
                reply_to_message_id=message.message_id,
            )
        except AudioError:
            await self.send_notice(
                message.chat.id,
                "Не удалось прочитать голосовое. Пришлите запись до 2 минут и 10 МБ.",
                reply_to_message_id=message.message_id,
            )
        except (TelegramAPIError, TimeoutError):
            await self.send_notice(
                message.chat.id,
                "Не удалось скачать запись. Попробуйте отправить её ещё раз.",
                reply_to_message_id=message.message_id,
            )

    async def announce_job(
        self, chat_id: int, job_id: str, *, reply_to_message_id: int | None = None
    ) -> None:
        if chat_id <= 0:
            return
        lock = self._status_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            if time.monotonic() < self._status_retry_at.get(job_id, 0):
                return
            entry = await asyncio.to_thread(self.store.status_entry, job_id, chat_id)
            if entry is None or entry[1] is not None:
                return
            if not await asyncio.to_thread(self.store.claim_status, job_id, chat_id):
                return
            job, _ = entry
            panel = self.job_status_panel(job)
            state = f"{job.status}:{int(job.cancel_requested)}"
            try:
                message = await self.bot.send_message(
                    chat_id,
                    panel.text,
                    reply_markup=panel.keyboard,
                    reply_parameters=ReplyParameters(
                        message_id=reply_to_message_id, allow_sending_without_reply=True
                    )
                    if reply_to_message_id is not None
                    else None,
                )
            except TelegramRetryAfter as error:
                self._status_retry_at[job_id] = time.monotonic() + error.retry_after
                await asyncio.to_thread(self.store.release_status_claim, job_id, chat_id)
                logger.warning("job_status_send_failed reason=rate_limited")
                return
            except (TelegramAPIError, TimeoutError):
                # An ambiguous send may already exist. Never create a second status automatically.
                logger.warning("job_status_send_failed reason=network_or_api")
                return
            await asyncio.to_thread(
                self.store.attach_status, job_id, chat_id, message.message_id, state
            )
            self._status_retry_at.pop(job_id, None)

    async def refresh_job_status(self, job: Job, message_id: int) -> bool:
        lock = self._status_locks.setdefault(job.chat_id, asyncio.Lock())
        async with lock:
            if time.monotonic() < self._status_retry_at.get(job.id, 0):
                return False
            entry = await asyncio.to_thread(self.store.status_entry, job.id, job.chat_id)
            if entry is None or entry[1] != message_id:
                return False
            job = entry[0]
            state = f"{job.status}:{int(job.cancel_requested)}"
            panel = self.job_status_panel(job)
            try:
                await self.bot.edit_message_text(
                    text=panel.text,
                    chat_id=job.chat_id,
                    message_id=message_id,
                    reply_markup=panel.keyboard,
                )
            except TelegramRetryAfter as error:
                self._status_retry_at[job.id] = time.monotonic() + error.retry_after
                return False
            except TelegramBadRequest as error:
                reason = error.message.lower()
                if "message is not modified" not in reason:
                    if "message to edit not found" in reason or "message can't be edited" in reason:
                        await asyncio.to_thread(
                            self.store.drop_status, job.id, job.chat_id, message_id
                        )
                        self._status_retry_at.pop(job.id, None)
                        return True
                    logger.warning("job_status_edit_failed reason=telegram_bad_request")
                    self._status_retry_at[job.id] = time.monotonic() + 5
                    return False
            except (TelegramAPIError, TimeoutError):
                logger.warning("job_status_edit_failed reason=network_or_api")
                self._status_retry_at[job.id] = time.monotonic() + 5
                return False
            self._status_retry_at.pop(job.id, None)
            return await asyncio.to_thread(self.store.status_notified, job.id, message_id, state)

    async def show_draft(
        self, chat_id: int, draft: ProfileDraft, *, notice: str | None = None
    ) -> None:
        profile = (
            await self.settings_profile(chat_id, draft.profile_id) if draft.profile_id else None
        )
        await self.render(
            chat_id,
            views.draft_panel(
                draft.stage,
                draft.id,
                profile_name=profile.name if profile else None,
                setting_key=draft.setting_key,
            ),
            notice=notice,
        )

    async def download_profile_input(self, media: Voice | Audio | Document, path: Path) -> None:
        if media.file_size is not None and media.file_size > MAX_AUDIO_BYTES:
            raise AudioError("Profile file exceeds size limit")
        if isinstance(media, (Voice, Audio)) and media.duration > 20:
            raise AudioError("Profile duration exceeds limit")
        with LimitedBuffer() as destination:
            async with asyncio.timeout(45):
                await self.bot.download(media, destination=destination, timeout=40)
            await asyncio.to_thread(path.write_bytes, destination.getvalue())

    @staticmethod
    async def profile_input_problem(path: Path, error: AudioError) -> str:
        if str(error) in {"Profile file exceeds size limit", "Audio exceeds size limit"}:
            return (
                "Файл больше 10 МиБ. Пришлите более компактную запись длительностью 15–20 секунд."
            )
        if str(error) == "Profile duration exceeds limit":
            return "Образец длиннее 20 секунд. Укоротите запись до 15–20 секунд и пришлите снова."
        try:
            # Reinspect only a rejected reference to distinguish quiet speech from clipping;
            # successful uploads keep the single validation pass.
            info = await validate_voice(path, max_duration=20)
        except AudioError:
            if str(error) == "Превышена длительность или запись слишком короткая.":
                return (
                    "Длительность образца вне диапазона 15–20 секунд. Пришлите подходящий фрагмент."
                )
            return (
                "Не удалось прочитать аудиофайл. Пришлите голосовое или "
                "WAV/MP3/OGG/FLAC с одной звуковой дорожкой без видео."
            )
        if info.duration < 15:
            return "Образец слишком короткий. Запишите не меньше 15 и не больше 20 секунд речи."
        if info.rms_db < -40:
            return "Запись слишком тихая или почти без звука. Запишите голос ближе к микрофону."
        if info.clipping_fraction > 0.01:
            return "Звук перегружен и искажён. Уменьшите громкость записи или отодвиньте микрофон."
        return (
            "Образец не удалось проверить. Пришлите чистую русскую речь длительностью 15–20 секунд."
        )

    async def handle_profile_message(
        self,
        message: Message,
        update_id: int,
        draft: ProfileDraft,
    ) -> None:
        if draft.stage == "setting":
            if message.text is None:
                await self.show_draft(message.chat.id, draft)
                return
            try:
                changed = await asyncio.to_thread(
                    self.store.save_voice_setting,
                    update_id,
                    message.chat.id,
                    draft.id,
                    message.text,
                )
            except ValueError:
                await self.show_draft(
                    message.chat.id,
                    draft,
                    notice="Значение не подходит. Проверьте диапазон и введите его ещё раз.",
                )
                return
            if changed is not None:
                await self.show_voice_settings(
                    message.chat.id, changed.id, notice="Настройка сохранена."
                )
            return
        if draft.stage in ("upload", "reference"):
            media = message.voice or message.audio or message.document
            if media is None:
                await self.show_draft(message.chat.id, draft)
                return
            incoming = self.store.data_dir / "incoming"
            await asyncio.to_thread(incoming.mkdir, mode=0o700, exist_ok=True)
            path = incoming / f"{secrets.token_hex(16)}.audio"
            try:
                await self.download_profile_input(media, path)
                await validate_enrollment_reference(path)
                if draft.stage == "reference":
                    if draft.profile_id is None:
                        raise NoProfile("Reference profile no longer available")
                    job_id = await asyncio.to_thread(
                        self.store.enqueue_reference_add,
                        update_id,
                        message.chat.id,
                        draft.id,
                        path,
                    )
                    await self.show_samples(message.chat.id, draft.profile_id)
                    if job_id is not None:
                        await self.announce_job(
                            message.chat.id, job_id, reply_to_message_id=message.message_id
                        )
                else:
                    saved = await asyncio.to_thread(
                        self.store.save_profile_upload,
                        update_id,
                        message.chat.id,
                        draft.id,
                        path,
                    )
                    current = await asyncio.to_thread(self.store.profile_draft, message.chat.id)
                    if saved and current is not None:
                        await self.show_draft(message.chat.id, current)
            except AudioError as error:
                await self.send_notice(
                    message.chat.id,
                    await self.profile_input_problem(path, error),
                    reply_to_message_id=message.message_id,
                )
            finally:
                await asyncio.to_thread(path.unlink, missing_ok=True)
            return
        if draft.stage in ("name", "rename"):
            try:
                name = self.store.profile_name(message.text or "")
            except ValueError:
                await self.show_draft(message.chat.id, draft)
                return
            if draft.stage == "name":
                job_id = await asyncio.to_thread(
                    self.store.enqueue,
                    update_id,
                    message.chat.id,
                    "profile",
                    text=name,
                    draft_id=draft.id,
                )
                if job_id is not None:
                    await self.render(message.chat.id, await self.home_panel(message.chat.id))
                    await self.announce_job(
                        message.chat.id, job_id, reply_to_message_id=message.message_id
                    )
            else:
                renamed = await asyncio.to_thread(
                    self.store.rename_profile,
                    update_id,
                    message.chat.id,
                    draft.id,
                    name,
                )
                if renamed:
                    await self.show_voice_settings(
                        message.chat.id, draft.profile_id, notice="Название изменено."
                    )
            return
        await self.show_draft(message.chat.id, draft)

    async def receive_voice(self, message: Message, update_id: int) -> str | None:
        voice = message.voice
        if voice is None:
            raise ValueError("Voice message required")
        if voice.duration > MAX_AUDIO_SECONDS or (
            voice.file_size is not None and voice.file_size > MAX_AUDIO_BYTES
        ):
            raise AudioError("Audio metadata exceeds limits")
        # Fail missing setup before downloading private recordings.
        if await asyncio.to_thread(self.store.selected_profile, message.chat.id) is None:
            raise NoProfile("No profile")
        account = await asyncio.to_thread(self.store.account, message.chat.id)
        if not account.is_admin and account.available_ms == 0:
            raise InsufficientBalance("No remaining time")
        incoming = self.store.data_dir / "incoming"
        await asyncio.to_thread(incoming.mkdir, mode=0o700, exist_ok=True)
        path = incoming / f"{secrets.token_hex(16)}.audio"
        try:
            with LimitedBuffer() as destination:
                async with asyncio.timeout(45):
                    await self.bot.download(voice, destination=destination, timeout=40)
                await asyncio.to_thread(path.write_bytes, destination.getvalue())
            await validate_voice(path)
            return await asyncio.to_thread(
                self.store.enqueue,
                update_id,
                message.chat.id,
                "vc",
                source=path,
            )
        finally:
            await asyncio.to_thread(path.unlink, missing_ok=True)

    async def settings_profile(self, chat_id: int, profile_id: str) -> Profile:
        return await asyncio.to_thread(self.store.get_profile, chat_id, profile_id)

    async def reference_profile(self, chat_id: int, profile_id: str) -> Profile:
        profile = await self.settings_profile(chat_id, profile_id)
        if await asyncio.to_thread(self.store.is_default_voice, chat_id, profile_id):
            raise NoProfile("Built-in reference samples are private")
        return profile

    async def show_account(self, chat_id: int, *, notice: str | None = None) -> None:
        account = await asyncio.to_thread(self.store.account, chat_id)
        await self.render(chat_id, views.account_panel(account), notice=notice)

    async def show_profiles(self, chat_id: int) -> None:
        profiles = await asyncio.to_thread(self.store.profiles, chat_id)
        selected = await asyncio.to_thread(self.store.selected_profile, chat_id)
        panel = views.catalog_panel(profiles, selected.id if selected else None)
        draft = await asyncio.to_thread(self.store.profile_draft, chat_id)
        if draft is not None and draft.stage in ("upload", "name"):
            panel = views.Panel(
                panel.text,
                panel.rows
                + [[InlineKeyboardButton(text="Продолжить добавление", callback_data="resume")]],
            )
        await self.render(chat_id, panel)

    async def show_voice_settings(
        self, chat_id: int, profile_id: str | None = None, *, notice: str | None = None
    ) -> None:
        profile = (
            await self.settings_profile(chat_id, profile_id)
            if profile_id
            else await asyncio.to_thread(self.store.selected_profile, chat_id)
        )
        if profile is None:
            await self.show_profiles(chat_id)
            return
        selected = await asyncio.to_thread(self.store.selected_profile, chat_id)
        await self.render(
            chat_id,
            views.voice_panel(
                profile,
                selected.id if selected else None,
                allow_samples=not await asyncio.to_thread(
                    self.store.is_default_voice, chat_id, profile.id
                ),
            ),
            notice=notice,
        )

    async def show_samples(
        self, chat_id: int, profile_id: str, *, notice: str | None = None
    ) -> None:
        profile = await self.reference_profile(chat_id, profile_id)
        change = await asyncio.to_thread(self.store.reference_change, chat_id, profile_id)
        await self.render(
            chat_id,
            views.samples_panel(
                profile,
                stage=change.stage if change else None,
                change_id=change.id if change else None,
                candidate_count=change.candidate_count if change else None,
            ),
            notice=notice,
        )

    async def reference_change_details(self, chat_id: int, change_id: str) -> ReferenceChange:
        for profile in await asyncio.to_thread(self.store.profiles, chat_id):
            change = await asyncio.to_thread(self.store.reference_change, chat_id, profile.id)
            if change is not None and change.id == change_id:
                await self.reference_profile(chat_id, profile.id)
                return change
        raise NoProfile("Reference change no longer available")

    async def send_reference_audio(
        self, chat_id: int, path: Path, *, caption: str, sample: bool = False
    ) -> None:
        converted: Path | None = None
        try:
            if sample:
                incoming = self.store.data_dir / "incoming"
                await asyncio.to_thread(incoming.mkdir, mode=0o700, exist_ok=True)
                converted = incoming / f"{secrets.token_hex(16)}.ogg"
                await to_voice(path, converted)
                path = converted
            await validate_result(path)
            async with asyncio.timeout(65):
                await self.bot.send_voice(
                    chat_id,
                    FSInputFile(path),
                    caption=caption,
                    reply_markup=ReplyKeyboardRemove(),
                    request_timeout=60,
                )
        except (TelegramNetworkError, TimeoutError):
            await self.say(
                chat_id,
                "Telegram не подтвердил отправку. Проверьте чат перед повторным прослушиванием.",
            )
        except (TelegramAPIError, AudioError, OSError):
            await self.say(chat_id, "Не получилось отправить пример. Нажмите «Прослушать» ещё раз.")
        finally:
            if converted is not None:
                await asyncio.to_thread(converted.unlink, missing_ok=True)

    async def show_cue_category(self, chat_id: int, profile_id: str, code: str, page: int) -> None:
        await self.render(
            chat_id,
            views.cue_category_panel(await self.settings_profile(chat_id, profile_id), code, page),
        )

    async def show_cue(self, chat_id: int, profile_id: str, cue_id: str) -> None:
        await self.render(
            chat_id,
            views.cue_panel(await self.settings_profile(chat_id, profile_id), cue_id),
        )

    async def show_selected_cues(self, chat_id: int, profile_id: str) -> None:
        await self.render(
            chat_id,
            views.selected_delivery_panel(await self.settings_profile(chat_id, profile_id)),
        )

    async def show_instruction(self, chat_id: int, profile_id: str) -> None:
        await self.render(
            chat_id,
            views.instruction_panel(await self.settings_profile(chat_id, profile_id)),
        )

    async def show_speed(self, chat_id: int, profile_id: str) -> None:
        await self.render(
            chat_id,
            views.speed_panel(await self.settings_profile(chat_id, profile_id)),
        )

    async def show_advanced(self, chat_id: int, profile_id: str) -> None:
        await self.render(
            chat_id,
            views.advanced_panel(await self.settings_profile(chat_id, profile_id)),
        )

    async def show_parameter(self, chat_id: int, profile_id: str, code: str) -> None:
        await self.render(
            chat_id,
            views.parameter_panel(await self.settings_profile(chat_id, profile_id), code),
        )

    async def queue_preview(self, chat_id: int, profile: Profile) -> None:
        job_id = await asyncio.to_thread(
            self.store.enqueue,
            None,
            chat_id,
            "tts",
            text=PREVIEW_TEXT,
            profile_id=profile.id,
            preview=True,
        )
        if job_id is not None:
            await self.announce_job(chat_id, job_id)

    async def reset_section(self, chat_id: int, profile_id: str, code: str) -> None:
        profile = await self.settings_profile(chat_id, profile_id)
        if code.startswith("cat-"):
            category = CATEGORY_CODES[code.removeprefix("cat-")]
            removed = {c.tag for c in CUES if c.category == category}
            changes: dict[str, object] = {
                "native_tags": tuple(t for t in profile.effective_native_tags if t not in removed)
            }
        elif code == "tags":
            changes = {"native_tags": ()}
        elif code == "ins":
            changes = {"style_instruction": None}
        elif code == "speed":
            changes = {"speed": 1.0}
        elif code == "advanced":
            changes = {p.key: p.default for p in PARAMETERS.values()}
        else:
            changes = {PARAMETERS[code].key: PARAMETERS[code].default}
        if code == "speed":
            await asyncio.to_thread(
                self.store.update_voice_settings, chat_id, profile_id, speed=1.0
            )
        else:
            await asyncio.to_thread(
                self.store.update_voice_settings, chat_id, profile_id, changes=changes
            )
        if code.startswith("cat-"):
            await self.show_cue_category(chat_id, profile_id, code.removeprefix("cat-"), 0)
        elif code == "tags":
            await self.show_selected_cues(chat_id, profile_id)
        elif code == "ins":
            await self.show_instruction(chat_id, profile_id)
        elif code == "speed":
            await self.show_speed(chat_id, profile_id)
        elif code == "advanced":
            await self.show_advanced(chat_id, profile_id)
        else:
            await self.show_parameter(chat_id, profile_id, code)

    @staticmethod
    def failure_text(job: Job) -> str:
        if job.failure == "insufficient_balance":
            return (
                "для этой озвучки не хватило времени на балансе; время не списано. "
                "Пополнение появится позже"
            )
        if job.failure == "out_of_memory":
            return "сейчас не получается озвучить сообщение; попробуйте чуть позже"
        if job.failure == "invalid_audio":
            return "не удалось прочитать запись; пришлите её ещё раз"
        if job.failure == "delivery_error":
            return "не получилось отправить голосовое; нажмите «Повторить отправку»"
        if job.failure == "not_ready":
            return "сейчас не получается создать голосовое; попробуйте чуть позже"
        if job.failure == "profile_invalid":
            return "образец не подошёл; добавьте 15–20 секунд чистой речи без музыки и шума"
        if job.failure == "reference_transcription_failed":
            return "не удалось разобрать русскую речь; добавьте более отчётливый образец"
        if job.failure == "speech_unrecognized":
            return "не удалось разобрать речь; пришлите более отчётливую запись"
        if job.failure == "speech_too_long":
            return "получилось больше 2000 символов текста; разделите запись на несколько частей"
        if job.failure == "whisper_preparation_failed":
            return "не получилось подготовить шёпот для этого голоса. Попробуйте другой образец"
        if job.failure in ("restart", "worker_lost"):
            return "озвучка прервалась; отправьте сообщение ещё раз"
        if job.failure == "timeout":
            return "озвучка заняла слишком много времени; попробуйте сообщение покороче"
        if job.failure == "storage_error":
            return "не удалось сохранить голосовое; попробуйте чуть позже"
        return "не получилось создать голосовое; попробуйте сообщение покороче"

    async def on_callback(self, callback: CallbackQuery) -> None:
        message = callback.message
        if not isinstance(message, Message) or not self.authorized_sender(
            callback.from_user, message.chat
        ):
            return
        chat_id = message.chat.id
        action, _, identifier = (callback.data or "").partition(":")
        try:
            await asyncio.to_thread(
                self.store.update_account_user,
                chat_id,
                username=callback.from_user.username,
                display_name=callback.from_user.full_name,
            )
            if action in ("cancel", "retry"):
                entry = await asyncio.to_thread(self.store.status_entry, identifier, chat_id)
                if entry is None or entry[1] != message.message_id:
                    await self.answer(callback, "Эта кнопка больше не действует.")
                    return
                if not await asyncio.to_thread(self.store.accept_callback, callback.id):
                    await self.answer(callback, "Уже выполнено.")
                    return
                if action == "cancel":
                    await asyncio.to_thread(self.store.request_cancel, identifier, chat_id)
                else:
                    await asyncio.to_thread(
                        self.store.retry_delivery, identifier, chat_id, callback.id
                    )
                current_status = await asyncio.to_thread(
                    self.store.status_entry, identifier, chat_id
                )
                if current_status is not None and current_status[1] is not None:
                    job, status_id = current_status
                    assert status_id is not None
                    await self.refresh_job_status(job, status_id)
                await self.refresh_panel(chat_id)
                await self.answer(callback)
                return
            if await asyncio.to_thread(self.store.panel_id, chat_id) != message.message_id:
                await self.answer(callback, "Эта панель устарела. Откройте текущую через /start.")
                return
            if not await asyncio.to_thread(self.store.accept_callback, callback.id):
                await self.answer(callback, "Уже выполнено.")
                return
            previous = await asyncio.to_thread(self.store.profile_draft, chat_id)
            if (
                previous is not None
                and previous.stage == "reference"
                and action in ("add", "rename", "delete", "ce")
            ):
                await asyncio.to_thread(self.store.discard_reference_change, chat_id, previous.id)
            if (
                previous is not None
                and previous.stage in ("setting", "rename", "delete")
                and action not in ("draftcancel", "deleteyes", "resume")
            ):
                await asyncio.to_thread(self.store.cancel_profile_draft, chat_id, previous.id)
            if action == "home":
                await self.render(chat_id, await self.home_panel(chat_id))
            elif action == "account":
                await self.show_account(chat_id)
            elif action == "voices":
                await self.show_profiles(chat_id)
            elif action == "card":
                await self.show_voice_settings(chat_id, identifier)
            elif action == "profile":
                await asyncio.to_thread(self.store.select_profile, chat_id, identifier)
                await self.show_voice_settings(
                    chat_id, identifier, notice="Голос выбран для новых сообщений."
                )
            elif action == "add":
                draft = await asyncio.to_thread(self.store.begin_profile_draft, chat_id)
                await self.show_draft(chat_id, draft)
            elif action == "resume":
                if previous is None:
                    await self.show_profiles(chat_id)
                else:
                    await self.show_draft(chat_id, previous)
            elif action in ("rename", "delete"):
                profile = await self.settings_profile(chat_id, identifier)
                stage: Literal["rename", "delete"] = "rename" if action == "rename" else "delete"
                draft = await asyncio.to_thread(
                    self.store.begin_profile_draft, chat_id, stage, identifier
                )
                if action == "delete":
                    await self.render(chat_id, views.delete_panel(profile, draft.id))
                else:
                    await self.show_draft(chat_id, draft)
            elif action == "draftcancel":
                if (
                    previous is not None
                    and previous.id == identifier
                    and previous.stage == "reference"
                ):
                    await asyncio.to_thread(
                        self.store.discard_reference_change, chat_id, identifier
                    )
                    if previous.profile_id is not None:
                        await self.show_samples(chat_id, previous.profile_id)
                else:
                    await asyncio.to_thread(self.store.cancel_profile_draft, chat_id, identifier)
                    if previous is not None and previous.id == identifier and previous.profile_id:
                        await self.show_voice_settings(chat_id, previous.profile_id)
                    else:
                        await self.show_profiles(chat_id)
            elif action == "deleteyes":
                if await asyncio.to_thread(self.store.delete_profile, chat_id, identifier):
                    await self.show_profiles(chat_id)
                else:
                    await self.say(chat_id, "Удаление недоступно. Вернитесь к списку голосов.")
            elif action == "cw":
                await self.render(
                    chat_id,
                    views.pronunciation_panel(await self.settings_profile(chat_id, identifier)),
                )
            elif action == "cc":
                profile_id, code, page = identifier.rsplit(":", 2)
                await self.show_cue_category(chat_id, profile_id, code, int(page))
            elif action in ("cd", "cg"):
                profile_id, cue_id = identifier.rsplit(":", 1)
                cue = next((c for c in CUES if c.id == cue_id), None)
                if cue is None:
                    raise ValueError("Unknown cue")
                if action == "cg":
                    await asyncio.to_thread(
                        self.store.toggle_native_tag, chat_id, profile_id, cue.tag
                    )
                await self.show_cue(chat_id, profile_id, cue_id)
            elif action in ("ct", "cu"):
                profile_id = identifier
                if action == "cu":
                    profile_id, token = identifier.rsplit(":", 1)
                    profile = await self.settings_profile(chat_id, profile_id)
                    tag = next(
                        (
                            t
                            for t in profile.effective_native_tags
                            if sha256(t.encode()).hexdigest()[:12] == token
                        ),
                        None,
                    )
                    if tag is None:
                        raise ValueError("Cue unavailable")
                    await asyncio.to_thread(self.store.toggle_native_tag, chat_id, profile_id, tag)
                await self.show_selected_cues(chat_id, profile_id)
            elif action in ("ci", "cs", "ca"):
                display = {
                    "ci": self.show_instruction,
                    "cs": self.show_speed,
                    "ca": self.show_advanced,
                }[action]
                await display(chat_id, identifier)
            elif action in ("cp", "cb", "cx"):
                if action == "cb":
                    profile_id, code, direction = identifier.rsplit(":", 2)
                    if direction not in ("+", "-"):
                        raise ValueError("Invalid direction")
                    await asyncio.to_thread(
                        self.store.adjust_voice_setting,
                        chat_id,
                        profile_id,
                        code,
                        1 if direction == "+" else -1,
                    )
                    await self.show_parameter(chat_id, profile_id, code)
                else:
                    profile_id, code = identifier.rsplit(":", 1)
                    if action == "cx":
                        await self.reset_section(chat_id, profile_id, code)
                    else:
                        await self.show_parameter(chat_id, profile_id, code)
            elif action == "ce":
                profile_id, code = identifier.rsplit(":", 1)
                key = (
                    "native_tag"
                    if code == "tag"
                    else "style_instruction"
                    if code == "ins"
                    else PARAMETERS[code].key
                )
                draft = await asyncio.to_thread(
                    self.store.begin_profile_draft,
                    chat_id,
                    "setting",
                    profile_id,
                    TypeAdapter(VoiceSettingKey).validate_python(key),
                )
                await self.show_draft(chat_id, draft)
            elif action in ("speed", "reset"):
                if action == "reset":
                    profile = await asyncio.to_thread(
                        self.store.update_voice_settings,
                        chat_id,
                        identifier,
                        delivery="neutral",
                        speed=1.0,
                        changes={
                            "native_tags": (),
                            "style_instruction": None,
                            **{p.key: p.default for p in PARAMETERS.values()},
                        },
                    )
                    await self.show_voice_settings(
                        chat_id, profile.id, notice="Восстановлены обычные настройки."
                    )
                else:
                    profile_id, value = identifier.rsplit(":", 1)
                    await asyncio.to_thread(
                        self.store.update_voice_settings, chat_id, profile_id, speed=float(value)
                    )
                    await self.show_speed(chat_id, profile_id)
            elif action == "listen":
                await self.queue_preview(chat_id, await self.settings_profile(chat_id, identifier))
            elif action == "samples":
                await self.show_samples(chat_id, identifier)
            elif action == "sample":
                profile_id, index = identifier.rsplit(":", 1)
                await self.render(
                    chat_id,
                    views.sample_panel(
                        await self.reference_profile(chat_id, profile_id), int(index)
                    ),
                )
            elif action == "sampleadd":
                current = await asyncio.to_thread(self.store.reference_change, chat_id, identifier)
                if current is not None and current.stage != "failed":
                    await self.show_samples(chat_id, identifier)
                else:
                    profile = await self.reference_profile(chat_id, identifier)
                    if len(profile.reference_samples) >= 3:
                        await self.show_samples(
                            chat_id, identifier, notice="Можно сохранить до трёх образцов."
                        )
                    else:
                        await asyncio.to_thread(self.store.begin_reference_add, chat_id, identifier)
                        upload_draft = await asyncio.to_thread(self.store.profile_draft, chat_id)
                        if upload_draft is None or upload_draft.stage != "reference":
                            raise NoProfile("Reference upload unavailable")
                        await self.show_draft(chat_id, upload_draft)
            elif action == "sampleremove":
                profile_id, index = identifier.rsplit(":", 1)
                current = await asyncio.to_thread(self.store.reference_change, chat_id, profile_id)
                if current is not None and current.stage != "failed":
                    await self.show_samples(chat_id, profile_id)
                else:
                    job_id = await asyncio.to_thread(
                        self.store.enqueue_reference_remove, chat_id, profile_id, int(index)
                    )
                    await self.show_samples(chat_id, profile_id)
                    await self.announce_job(chat_id, job_id)
            elif action == "samplelisten":
                profile_id, index = identifier.rsplit(":", 1)
                profile = await self.reference_profile(chat_id, profile_id)
                path = await asyncio.to_thread(
                    self.store.reference_sample_path, chat_id, profile_id, int(index)
                )
                await self.send_reference_audio(
                    chat_id, path, caption=f"{profile.name} · Образец {int(index) + 1}", sample=True
                )
            elif action == "samplepreview":
                change = await self.reference_change_details(chat_id, identifier)
                path = await asyncio.to_thread(
                    self.store.reference_change_preview_path, chat_id, identifier
                )
                profile = await self.reference_profile(chat_id, change.profile_id)
                await self.show_samples(chat_id, profile.id)
                await self.send_reference_audio(
                    chat_id, path, caption=f"{profile.name} · Новый пример"
                )
            elif action == "sampleapply":
                change = await self.reference_change_details(chat_id, identifier)
                try:
                    updated = await asyncio.to_thread(
                        self.store.apply_reference_change, chat_id, identifier
                    )
                    if updated is None:
                        raise ValueError("Reference change no longer applicable")
                except (NoProfile, ValueError):
                    await self.show_samples(
                        chat_id,
                        change.profile_id,
                        notice="Образцы уже изменились или пример ещё не готов. "
                        "Оставьте прежний вариант и подготовьте пример заново.",
                    )
                else:
                    await self.show_samples(chat_id, updated.id, notice="Образцы обновлены.")
            elif action == "samplediscard":
                change = await self.reference_change_details(chat_id, identifier)
                await asyncio.to_thread(self.store.discard_reference_change, chat_id, identifier)
                await self.show_samples(
                    chat_id, change.profile_id, notice="Оставлен прежний голос."
                )
            else:
                await self.answer(callback, "Эта кнопка больше не действует. Откройте /start.")
                return
        except InsufficientBalance:
            await self.send_notice(
                chat_id,
                "Для этого сообщения не хватает времени на балансе. Пополнение появится позже.",
            )
        except QueueFull:
            await self.send_notice(
                chat_id,
                "Очередь заполнена. Дождитесь результата или отмените одну из своих озвучек.",
            )
        except VoiceTagLimit:
            await self.say(
                chat_id,
                "Можно сочетать до трёх вариантов подачи. Сначала выключите один из выбранных.",
                InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="Выбранная подача",
                                callback_data=f"ct:{identifier.rsplit(':', 1)[0]}",
                            )
                        ]
                    ]
                ),
            )
        except (NoProfile, KeyError, ValueError, PermissionError):
            if action in ("cancel", "retry"):
                await self.answer(callback, "Эта кнопка больше не действует.")
                return
            await self.say(
                chat_id, "Эта настройка или голос больше недоступны. Вернитесь к списку голосов."
            )
        except (OSError, sqlite3.Error):
            logger.error("callback_failed reason=storage_error")
            if action in ("cancel", "retry"):
                await self.answer(callback, "Не удалось сохранить изменения. Попробуйте позже.")
                return
            await self.say(
                chat_id, "Не удалось сохранить изменения. Попробуйте ещё раз чуть позже."
            )
        await self.answer(callback)

    async def answer(self, callback: CallbackQuery, text: str | None = None) -> None:
        try:
            await callback.answer(text)
        except (TelegramAPIError, TimeoutError):
            logger.warning("callback_answer_failed reason=network_or_api")

    async def deliver_once(self) -> bool:
        await asyncio.to_thread(self.store.cancel_unauthorized_jobs)
        job = await asyncio.to_thread(self.store.delivery_claim)
        if job is None:
            return False
        if job.chat_id <= 0:
            await asyncio.to_thread(self.store.delivery_done, job.id, None)
            return True
        try:
            entry = await asyncio.to_thread(self.store.status_entry, job.id, job.chat_id)
            if entry is not None and entry[1] is not None:
                current, status_id = entry
                assert status_id is not None
                await self.refresh_job_status(current, status_id)
        except (OSError, sqlite3.Error):
            logger.warning("job_status_edit_failed reason=storage_error")
        result = self.store.job_dir(job.id) / "result.ogg"
        try:
            await validate_result(result)
            for attempt in range(3):
                try:
                    if job.chat_id <= 0:
                        await asyncio.to_thread(self.store.delivery_done, job.id, None)
                        return True
                    async with asyncio.timeout(65):
                        message = await self.bot.send_voice(
                            job.chat_id,
                            FSInputFile(result),
                            caption="Новый пример голоса" if job.kind == "reference" else None,
                            # Clear a previous version's reply keyboard on the audio result.
                            # Control panels must be created with inline markup to stay editable.
                            reply_markup=ReplyKeyboardRemove(),
                            request_timeout=60,
                        )
                    await asyncio.to_thread(self.store.delivery_done, job.id, message.message_id)
                    return True
                except TelegramRetryAfter as error:
                    if attempt == 2 or error.retry_after > 60:
                        await asyncio.to_thread(self.store.delivery_done, job.id, None)
                        return True
                    await asyncio.sleep(error.retry_after + random.uniform(0.1, 0.7))
        except (TelegramNetworkError, TimeoutError):
            # A transport timeout cannot prove that Telegram did not accept sendVoice.
            await asyncio.to_thread(self.store.delivery_done, job.id, None, uncertain=True)
        except (TelegramAPIError, AudioError, OSError):
            await asyncio.to_thread(self.store.delivery_done, job.id, None)
        finally:
            # Balance and prepared samples remain current even if Telegram lost the status.
            await self.refresh_panel(job.chat_id)
        return True

    async def deliver_loop(self) -> None:
        while True:
            try:
                await self.refresh_statuses()
                delivered = await self.deliver_once()
            except (OSError, sqlite3.Error):
                logger.error("delivery_failed reason=storage_error")
                delivered = False
                await asyncio.sleep(4)
            if not delivered:
                await asyncio.sleep(1)

    async def refresh_statuses(self) -> None:
        updates = await asyncio.to_thread(self.store.status_updates)
        changed_chats: set[int] = set()
        for job, message_id, _ in updates:
            if not self.store.allowed_user(job.chat_id):
                continue
            if job.status not in ("queued", "running"):
                changed_chats.add(job.chat_id)
            if message_id is None:
                await self.announce_job(job.chat_id, job.id)
                continue
            await self.refresh_job_status(job, message_id)
        for chat_id in changed_chats:
            await self.refresh_panel(chat_id)
