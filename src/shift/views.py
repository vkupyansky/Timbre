"""Pure views for the bot's single, replaceable Telegram panel."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from html import escape

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from .protocol import Profile, VoiceSettingKey
from .store import Account
from .voice_controls import (
    CATEGORIES,
    CATEGORY_CODES,
    CUES,
    PARAMETERS,
    PRONUNCIATION_HELP,
    VoiceCue,
)


@dataclass(frozen=True)
class Panel:
    text: str
    rows: list[list[InlineKeyboardButton]]

    @property
    def keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(inline_keyboard=self.rows)


def _button(text: str, data: str) -> InlineKeyboardButton:
    if not 1 <= len(data.encode()) <= 64:
        raise ValueError("Callback exceeds the Telegram limit")
    return InlineKeyboardButton(text=text, callback_data=data)


def _pairs(buttons: Sequence[InlineKeyboardButton]) -> list[list[InlineKeyboardButton]]:
    return [list(buttons[index : index + 2]) for index in range(0, len(buttons), 2)]


def _name(tag: str) -> str:
    return next((cue.name for cue in CUES if cue.tag == tag), tag)


def _cue(cue_id: str) -> VoiceCue:
    cue = next((item for item in CUES if item.id == cue_id), None)
    if cue is None:
        raise ValueError("Unknown voice cue")
    return cue


def _back(profile: Profile) -> list[InlineKeyboardButton]:
    return [_button("← К голосу", f"card:{profile.id}")]


def _number(value: float | int) -> str:
    return str(value).replace(".", ",")


def _heading(text: str) -> str:
    return f"<b>{escape(text, quote=False)}</b>"


def _time(milliseconds: int) -> str:
    seconds = max(milliseconds, 0) // 1000
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes} мин {seconds:02d} сек"


def _account_rows() -> list[list[InlineKeyboardButton]]:
    return [[_button("Мои голоса", "voices"), _button("Профиль", "account")]]


def start_panel(profile: Profile | None, account: Account, worker_ready: bool = True) -> Panel:
    if profile:
        prompt = f"🎙 {_heading(profile.name)}\nОтправьте текст или голосовое."
    else:
        prompt = "Начните с голоса.\nОткройте «Мои голоса» и добавьте образец записи."
    balance = "Безлимит" if account.is_admin else f"Осталось {_time(account.available_ms)}"
    if not worker_ready:
        prompt += "\n\nПока не получается озвучить сообщение. Попробуйте чуть позже."
    return Panel(
        f"{_heading('Timbre')}\nВаши слова — выбранным голосом.\n\n{prompt}\n\n{balance}",
        _account_rows(),
    )


def account_panel(account: Account) -> Panel:
    username = f"@{escape(account.username, quote=False)}" if account.username else "Не указан"
    name = escape(account.display_name, quote=False) if account.display_name else "Не указано"
    text = (
        f"{_heading('Профиль')}\n\nИмя: {name}\n"
        f"Имя пользователя: {username}\nID: <code>{account.user_id}</code>"
    )
    if account.is_admin:
        text += (
            f"\n\n{_heading('Администратор')}\nБаланс: {_heading('Безлимит')}.\n"
            "Можно озвучивать сообщения без ограничений по балансу. Время не списывается."
        )
    else:
        text += (
            f"\n\nОсталось {_heading(_time(account.available_ms))}.\n\n"
            "10 пробных минут выдаются один раз на аккаунт и подходят для всех ваших голосов. "
            "Время списывается по длительности готовой озвучки, без округления до минуты.\n\n"
            "Пополнение появится позже."
        )
        if account.reserved_ms:
            text += (
                "\n\nЧасть времени отложена для сообщений в обработке. "
                "Остаток вернётся после завершения."
            )
    return Panel(text, [[_button("← На главную", "home")]])


def catalog_panel(profiles: Sequence[Profile], selected_id: str | None) -> Panel:
    rows = _pairs(
        [
            _button(
                ("✓ " if profile.id == selected_id else "") + profile.name, f"card:{profile.id}"
            )
            for profile in profiles
        ]
    )
    if profiles:
        text = "Откройте голос, чтобы послушать пример или выбрать его для своих сообщений."
    else:
        text = "Здесь будут ваши голоса.\nДобавьте первый по образцу записи."
    rows.append([_button("+ Добавить голос", "add")])
    rows.append([_button("← На главную", "home")])
    return Panel(f"{_heading('Мои голоса')}\n\n{text}", rows)


def voice_panel(profile: Profile, selected_id: str | None, *, allow_samples: bool = True) -> Panel:
    chosen = (
        "Сейчас ваши сообщения звучат этим голосом."
        if profile.id == selected_id
        else "Послушайте пример. Нравится — нажмите «Применить»."
    )
    delivery = ", ".join(_name(tag) for tag in profile.effective_native_tags) or "Нейтральная"
    text = (
        f"🎙 {_heading(profile.name)}\n{chosen}\n\n"
        f"Подача: {escape(delivery, quote=False)}.\n"
        f"Темп: { {0.9: 'медленнее', 1.0: 'обычный', 1.1: 'быстрее'}[profile.speed] }.\n\n"
        "Настройте звучание ниже. Оно сохранится для этого голоса."
    )
    rows = [
        [
            _button("Применить", f"profile:{profile.id}"),
            _button("▶ Прослушать", f"listen:{profile.id}"),
        ]
    ]
    rows.extend(
        _pairs(
            [
                _button(CATEGORIES[category][0], f"cc:{profile.id}:{code}:0")
                for code, category in CATEGORY_CODES.items()
            ]
            + [
                _button("Темп", f"cs:{profile.id}"),
                _button("Инструкция", f"ci:{profile.id}"),
                _button("Дополнительно", f"ca:{profile.id}"),
                _button("Выбранная подача", f"ct:{profile.id}"),
            ]
        )
    )
    if allow_samples:
        rows.append(
            [_button(f"Образцы · {len(profile.reference_samples)} из 3", f"samples:{profile.id}")]
        )
    rows.append(
        [
            _button("Произношение", f"cw:{profile.id}"),
            _button("Сбросить настройки", f"reset:{profile.id}"),
        ]
    )
    rows.append(
        [
            _button("Переименовать", f"rename:{profile.id}"),
            _button("Удалить", f"delete:{profile.id}"),
        ]
    )
    rows.append([_button("← Мои голоса", "voices")])
    return Panel(text, rows)


def samples_panel(
    profile: Profile,
    *,
    stage: str | None = None,
    change_id: str | None = None,
    candidate_count: int | None = None,
) -> Panel:
    count = len(profile.reference_samples)
    rows = _pairs(
        [_button(f"Образец {index + 1}", f"sample:{profile.id}:{index}") for index in range(count)]
    )
    text = f"{_heading(f'{profile.name} · Образцы')}\n\nСейчас {count} из 3 образцов."
    if stage is None:
        text += "\nДобавляйте записи того же человека. Новый вариант сначала можно будет послушать."
        if count < 3:
            rows.append([_button("+ Добавить образец", f"sampleadd:{profile.id}")])
    else:
        if change_id is None or candidate_count is None:
            raise ValueError("Reference change is incomplete")
        if stage == "upload":
            text += "\n\nДобавление ещё одной записи не завершено."
            rows.append([_button("Продолжить добавление", "resume")])
        elif stage == "processing":
            text += "\n\nНовый вариант ещё не применён. Текущий голос работает как прежде."
        elif stage == "ready":
            samples = "1 образцом" if candidate_count == 1 else f"{candidate_count} образцами"
            text += (
                f"\n\nПример с {samples} готов.\nСравните новый и прежний варианты, затем выберите."
            )
            rows.extend(
                [
                    [
                        _button("▶ Новый пример", f"samplepreview:{change_id}"),
                        _button("▶ Прежний", f"listen:{profile.id}"),
                    ],
                    [_button("Применить", f"sampleapply:{change_id}")],
                ]
            )
        elif stage == "failed":
            text += "\n\nНе получилось подготовить пример. Текущий голос не изменён."
            if count < 3:
                rows.append([_button("Другой образец", f"sampleadd:{profile.id}")])
        else:
            raise ValueError("Unknown reference change stage")
        rows.append([_button("Оставить прежний", f"samplediscard:{change_id}")])
    rows.append(_back(profile))
    return Panel(text, rows)


def sample_panel(profile: Profile, index: int) -> Panel:
    if not 0 <= index < len(profile.reference_samples):
        raise ValueError("Reference sample no longer available")
    rows = [[_button("▶ Прослушать", f"samplelisten:{profile.id}:{index}")]]
    text = f"{_heading(f'{profile.name} · Образец {index + 1}')}"
    if len(profile.reference_samples) > 1:
        text += "\n\nПосле удаления сначала подготовлю пример. До вашего выбора голос не изменится."
        rows.append([_button("Удалить образец", f"sampleremove:{profile.id}:{index}")])
    else:
        text += "\n\nЭто единственный образец. Чтобы убрать его, сначала добавьте другой."
    rows.append([_button("← К образцам", f"samples:{profile.id}")])
    return Panel(text, rows)


def cue_category_panel(profile: Profile, code: str, page: int) -> Panel:
    category = CATEGORY_CODES[code]
    cues = [cue for cue in CUES if cue.category == category]
    pages = max(1, (len(cues) + 5) // 6)
    if not 0 <= page < pages:
        raise ValueError("Unknown voice cue page")
    rows = _pairs(
        [
            _button(
                ("✓ " if cue.tag in profile.effective_native_tags else "") + cue.name,
                f"cd:{profile.id}:{cue.id}",
            )
            for cue in cues[page * 6 : (page + 1) * 6]
        ]
    )
    navigation = []
    if page:
        navigation.append(_button("←", f"cc:{profile.id}:{code}:{page - 1}"))
    if page + 1 < pages:
        navigation.append(_button("→", f"cc:{profile.id}:{code}:{page + 1}"))
    if navigation:
        rows.append(navigation)
    rows.append([_button("Сбросить этот раздел", f"cx:{profile.id}:cat-{code}")])
    rows.append(_back(profile))
    title, description = CATEGORIES[category]
    return Panel(
        f"{_heading(f'{profile.name} · {title}')}\n\n{escape(description, quote=False)}\n"
        f"Откройте вариант, чтобы узнать, как он звучит.\n\n{page + 1} из {pages}",
        rows,
    )


def cue_panel(profile: Profile, cue_id: str) -> Panel:
    cue = _cue(cue_id)
    enabled = cue.tag in profile.effective_native_tags
    code = next(code for code, category in CATEGORY_CODES.items() if category == cue.category)
    page = [item.id for item in CUES if item.category == cue.category].index(cue_id) // 6
    rows = []
    rows.append([_button("Выключить" if enabled else "Включить", f"cg:{profile.id}:{cue.id}")])
    rows.append([_button("← К вариантам", f"cc:{profile.id}:{code}:{page}")])
    return Panel(
        f"{_heading(cue.name)}\n\n{escape(cue.description, quote=False)}\n\n"
        f"{'Уже выбрано для этого голоса.' if enabled else 'Можно добавить к звучанию голоса.'}\n"
        f"Выбрано {len(profile.effective_native_tags)} из 3 особенностей.\n\n"
        "Чтобы оценить результат, вернитесь к голосу и нажмите «Прослушать».",
        rows,
    )


def selected_delivery_panel(profile: Profile) -> Panel:
    tags = profile.effective_native_tags
    text = (
        "\n".join(f"• {escape(_name(tag), quote=False)}" for tag in tags) or "Нейтральная подача."
    )
    rows = []
    rows.extend(
        _pairs(
            [
                _button(
                    "Убрать: " + _name(tag),
                    f"cu:{profile.id}:{sha256(tag.encode()).hexdigest()[:12]}",
                )
                for tag in tags
            ]
        )
    )
    rows.append(
        [
            _button("Своя подача", f"ce:{profile.id}:tag"),
            _button("Сбросить подачу", f"cx:{profile.id}:tags"),
        ]
    )
    rows.append(_back(profile))
    return Panel(
        f"{_heading(f'{profile.name} · Выбранная подача')}\n\n{text}\n\n"
        "Сочетайте до трёх особенностей. Выберите их в разделах или опишите свою подачу.",
        rows,
    )


def instruction_panel(profile: Profile) -> Panel:
    rows = []
    rows.append(
        [
            _button("Изменить", f"ce:{profile.id}:ins"),
            _button("Очистить", f"cx:{profile.id}:ins"),
        ]
    )
    rows.append(_back(profile))
    instruction = (
        escape(profile.style_instruction, quote=False)
        if profile.style_instruction
        else "Инструкции пока нет."
    )
    return Panel(
        f"{_heading(f'{profile.name} · Инструкция')}\n\n"
        "Как говорить этому голосу? Например: тепло и спокойно, как в разговоре с другом.\n"
        "До 500 символов одной строкой.\n\n"
        f"{instruction}",
        rows,
    )


def speed_panel(profile: Profile) -> Panel:
    rows = []
    rows.extend(
        _pairs(
            [
                _button(
                    ("✓ " if profile.speed == value else "") + label,
                    f"speed:{profile.id}:{value}",
                )
                for value, label in ((0.9, "Медленнее"), (1.0, "Обычно"), (1.1, "Быстрее"))
            ]
        )
    )
    rows.append([_button("Обычный темп", f"cx:{profile.id}:speed")])
    rows.append(_back(profile))
    return Panel(
        f"{_heading(f'{profile.name} · Темп')}\n\n"
        "Чуть медленнее или чуть быстрее? Высота голоса останется прежней.\n\n"
        f"Сейчас: { {0.9: 'медленнее', 1.0: 'обычный темп', 1.1: 'быстрее'}[profile.speed] }.",
        rows,
    )


def advanced_panel(profile: Profile) -> Panel:
    rows = _pairs(
        [
            _button(
                f"{parameter.name}: {_number(getattr(profile, parameter.key))}",
                f"cp:{profile.id}:{code}",
            )
            for code, parameter in PARAMETERS.items()
        ]
    )
    rows.append([_button("Обычные значения", f"cx:{profile.id}:advanced")])
    rows.append(_back(profile))
    return Panel(
        f"{_heading(f'{profile.name} · Дополнительно')}\n\n"
        "Здесь можно точнее подобрать звучание. Откройте настройку, чтобы узнать, что она меняет.\n"
        "Если результат не понравится, верните обычные значения.",
        rows,
    )


def parameter_panel(profile: Profile, code: str) -> Panel:
    parameter = PARAMETERS[code]
    rows = []
    rows.extend(
        [
            [
                _button(f"−{_number(parameter.step)}", f"cb:{profile.id}:{code}:-"),
                _button(f"+{_number(parameter.step)}", f"cb:{profile.id}:{code}:+"),
            ],
            [
                _button("Ввести число", f"ce:{profile.id}:{code}"),
                _button("Обычное значение", f"cx:{profile.id}:{code}"),
            ],
        ]
    )
    rows.append([_button("← К дополнительным", f"ca:{profile.id}")])
    return Panel(
        f"{_heading(parameter.name)}\n\n{escape(parameter.description, quote=False)}\n\n"
        f"Сейчас: {_number(getattr(profile, parameter.key))}.",
        rows,
    )


def pronunciation_panel(profile: Profile) -> Panel:
    return Panel(
        f"{_heading(f'{profile.name} · Произношение')}\n\n"
        f"{escape(PRONUNCIATION_HELP, quote=False)}",
        [_back(profile)],
    )


def draft_panel(
    stage: str,
    draft_id: str,
    *,
    profile_name: str | None = None,
    setting_key: VoiceSettingKey | None = None,
) -> Panel:
    title = f"Голос «{profile_name}»" if profile_name else "Добавление голоса"
    if stage in ("upload", "reference"):
        prompt = (
            "Пришлите 15–20 секунд чистой связной речи одного человека в ровной манере. "
            "Подойдёт голосовое или аудиофайл до 10 МиБ."
        )
        if stage == "reference":
            title = f"{profile_name} · Новый образец" if profile_name else "Новый образец"
            prompt += "\n\nНужен тот же человек. Текущий голос изменится только после «Применить»."
    elif stage == "name":
        prompt = "Как назовём этот голос?\nОтправьте название, до 80 символов."
    elif stage == "rename":
        prompt = "Какое имя ему дать?\nОтправьте новое название, до 80 символов."
    elif stage == "delete":
        prompt = "Вернитесь к карточке голоса, чтобы подтвердить удаление."
    elif stage == "setting":
        if setting_key == "native_tag":
            prompt = (
                "Опишите подачу парой слов: "
                "до 100 символов одной строкой, без квадратных скобок. "
                "Например, soft tone — мягко. Английские описания обычно понятнее модели.\n\n"
                "После сохранения послушайте пример."
            )
        elif setting_key == "style_instruction":
            prompt = (
                "Как должен говорить этот голос? Опишите своими словами.\n"
                "До 500 символов одной строкой. Инструкция сохранится для новых сообщений."
            )
        else:
            parameter = next(
                (item for item in PARAMETERS.values() if item.key == setting_key), None
            )
            if parameter is None:
                raise ValueError("Unknown setting draft")
            prompt = parameter.description + "\n\nОтправьте новое число."
    else:
        raise ValueError("Unknown draft stage")
    return Panel(
        f"{_heading(title)}\n\n{escape(prompt, quote=False)}",
        [[_button("Отмена", f"draftcancel:{draft_id}")]],
    )


def delete_panel(profile: Profile, draft_id: str) -> Panel:
    rows = []
    rows.append(
        [
            _button("Удалить", f"deleteyes:{draft_id}"),
            _button("Отмена", f"draftcancel:{draft_id}"),
        ]
    )
    return Panel(
        f"{_heading(f'Удалить голос «{profile.name}»?')}\n\n"
        "Образцы и настройки этого голоса исчезнут. "
        "Если захотите вернуть его, понадобится образец.\n"
        "Сообщения, которые уже готовятся, останутся с этим голосом.",
        rows,
    )
