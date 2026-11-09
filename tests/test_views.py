from collections.abc import Iterator
from html import escape
from html.parser import HTMLParser

import pytest

from shift import views
from shift.protocol import Profile, ReferenceSample
from shift.store import Account
from shift.voice_controls import CATEGORY_CODES, CUES, PARAMETERS

ACCOUNT = Account(user_id=7, is_admin=True, balance_ms=0, reserved_ms=0)


def profile(identifier: str = "vika", name: str = "Вика") -> Profile:
    return Profile(
        id=identifier, name=name, reference_audio="reference.wav", reference_text="Образец речи."
    )


def actions(panel: views.Panel) -> list[str]:
    return [button.callback_data or "" for row in panel.rows for button in row]


def test_start_has_one_entry_and_catalog_opens_cards_without_applying() -> None:
    current = profile()
    start = views.start_panel(current, ACCOUNT)
    assert actions(start) == ["voices", "account"] and "Вика" in start.text
    assert start.text.startswith("<b>Timbre</b>\nВаши слова — выбранным голосом.")
    assert "Пока не получается озвучить" in views.start_panel(None, ACCOUNT, False).text
    profiles = [current, profile("vlad", "Влад"), profile("third", "Третий")]
    catalog = views.catalog_panel(profiles, "vlad")
    assert [len(row) for row in catalog.rows] == [2, 1, 1, 1]
    assert actions(catalog) == ["card:vika", "card:vlad", "card:third", "add", "home"]
    assert catalog.rows[0][1].text == "✓ Влад"
    assert catalog.rows[-2][0].text == "+ Добавить голос"
    assert actions(views.catalog_panel([], None)) == ["add", "home"]


def test_voice_card_groups_controls_and_keeps_apply_explicit() -> None:
    card = views.voice_panel(profile(), None)
    assert actions(card)[:2] == ["profile:vika", "listen:vika"]
    assert card.rows[0][1].text == "▶ Прослушать"
    assert all(len(row) <= 2 for row in card.rows)
    for required in (
        "cc:vika:e:0",
        "cc:vika:s:0",
        "cc:vika:p:0",
        "cc:vika:n:0",
        "cs:vika",
        "ci:vika",
        "ca:vika",
        "ct:vika",
        "cw:vika",
        "reset:vika",
        "rename:vika",
        "delete:vika",
        "samples:vika",
        "voices",
    ):
        assert required in actions(card)
    primary = card.text + " ".join(button.text for row in card.rows for button in row)
    assert "native" not in primary.lower() and "теги" not in primary.lower()


def all_voice_panels(current: Profile) -> Iterator[views.Panel]:
    yield views.voice_panel(current, current.id)
    yield views.selected_delivery_panel(current)
    yield views.instruction_panel(current)
    yield views.speed_panel(current)
    yield views.advanced_panel(current)
    yield views.pronunciation_panel(current)
    yield views.delete_panel(current, "a" * 32)
    yield views.samples_panel(current)
    yield views.sample_panel(current, 0)
    for stage in ("upload", "processing", "ready", "failed"):
        yield views.samples_panel(current, stage=stage, change_id="a" * 32, candidate_count=2)
    for code in PARAMETERS:
        yield views.parameter_panel(current, code)
    for code, category in CATEGORY_CODES.items():
        count = sum(cue.category == category for cue in CUES)
        for page in range((count + 5) // 6):
            yield views.cue_category_panel(current, code, page)
    for cue in CUES:
        yield views.cue_panel(current, cue.id)


def test_every_profile_owner_can_manage_their_own_voice_without_mutating_views() -> None:
    current = profile()
    before = current.model_dump()
    panels = list(all_voice_panels(current))
    assert "rename:vika" in actions(panels[0])
    assert "samples:vika" in actions(panels[0])
    assert current.model_dump() == before


def test_account_balance_is_human_readable_without_payment_controls() -> None:
    admin = views.account_panel(ACCOUNT)
    assert "Безлимит" in admin.text and "0" not in admin.text
    account = Account(user_id=8, is_admin=False, balance_ms=531234, reserved_ms=120000)
    balance = views.account_panel(account)
    assert "6 мин 51 сек" in balance.text and "один раз" in balance.text
    assert "отложена" in balance.text and actions(balance) == ["home"]
    home = views.start_panel(None, account)
    assert "Мои голоса" in home.text and "6 мин 51 сек" in home.text


def test_every_documented_cue_is_reachable_and_navigation_keeps_its_page() -> None:
    current = profile()
    reached: set[str] = set()
    for code, category in CATEGORY_CODES.items():
        matching = [cue for cue in CUES if cue.category == category]
        for page in range((len(matching) + 5) // 6):
            panel = views.cue_category_panel(current, code, page)
            buttons = [action for action in actions(panel) if action.startswith("cd:")]
            assert 1 <= len(buttons) <= 6
            reached.update(action.rsplit(":", 1)[1] for action in buttons)
            for cue in matching[page * 6 : (page + 1) * 6]:
                details = views.cue_panel(current, cue.id)
                assert cue.description in details.text
                assert actions(details)[-1] == f"cc:vika:{code}:{page}"
                assert actions(details)[0] == f"cg:vika:{cue.id}"
    assert reached == {cue.id for cue in CUES}


def test_telegram_limits_hold_for_every_screen_at_maximum_profile_lengths() -> None:
    current = Profile(
        id="a" * 48,
        name="я" * 80,
        reference_audio="reference.wav",
        reference_text="Речь.",
        native_tags=("a" * 100, "b" * 100, "c" * 100),
        style_instruction="я" * 500,
    )
    panels = [views.start_panel(current, ACCOUNT), views.catalog_panel([current], current.id)]
    panels.extend(all_voice_panels(current))
    for panel in panels:
        assert len(panel.text) <= 4096
        assert panel.keyboard.inline_keyboard == panel.rows
        assert all(len(row) <= 2 for row in panel.rows)
        assert all(1 <= len(action.encode()) <= 64 for action in actions(panel))


def test_reset_buttons_target_their_own_scope() -> None:
    current = profile()
    for code in CATEGORY_CODES:
        assert f"cx:vika:cat-{code}" in actions(views.cue_category_panel(current, code, 0))
    assert "cx:vika:tags" in actions(views.selected_delivery_panel(current))
    assert "cx:vika:ins" in actions(views.instruction_panel(current))
    assert "cx:vika:speed" in actions(views.speed_panel(current))
    assert "cx:vika:advanced" in actions(views.advanced_panel(current))
    for code in PARAMETERS:
        assert f"cx:vika:{code}" in actions(views.parameter_panel(current, code))


def test_draft_prompts_explain_limits_and_keep_the_existing_cancel_identity() -> None:
    draft_id = "a" * 32
    upload = views.draft_panel("upload", draft_id)
    assert "15–20 секунд" in upload.text
    assert actions(upload) == [f"draftcancel:{draft_id}"]
    assert "80 символов" in views.draft_panel("name", draft_id).text
    assert "Вика" in views.draft_panel("rename", draft_id, profile_name="Вика").text
    custom = views.draft_panel("setting", draft_id, profile_name="Вика", setting_key="native_tag")
    instruction = views.draft_panel("setting", draft_id, setting_key="style_instruction")
    assert "100 символов" in custom.text and "500 символов" in instruction.text
    for parameter in PARAMETERS.values():
        number = views.draft_panel("setting", draft_id, setting_key=parameter.key)
        assert parameter.description in number.text
    with pytest.raises(ValueError):
        views.draft_panel("setting", draft_id)
    with pytest.raises(ValueError):
        views.cue_category_panel(profile(), "e", -1)


def test_delete_confirmation_uses_draft_instead_of_profile_identifier() -> None:
    draft_id = "a" * 32
    panel = views.delete_panel(profile(), draft_id)
    assert actions(panel) == [f"deleteyes:{draft_id}", f"draftcancel:{draft_id}"]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_reference_management_enforces_visible_limits_and_requires_ready_preview(
    count: int,
) -> None:
    current = profile().model_copy(
        update={
            "extra_references": tuple(
                ReferenceSample(reference_audio=f"reference-{index}.wav", reference_text="Речь.")
                for index in range(1, count)
            )
        }
    )
    card = views.voice_panel(current, current.id)
    assert any(button.text == f"Образцы · {count} из 3" for row in card.rows for button in row)
    samples = views.samples_panel(current)
    assert ("sampleadd:vika" in actions(samples)) == (count < 3)
    assert [action for action in actions(samples) if action.startswith("sample:")] == [
        f"sample:vika:{index}" for index in range(count)
    ]
    for index in range(count):
        item = views.sample_panel(current, index)
        assert f"samplelisten:vika:{index}" in actions(item)
        assert (f"sampleremove:vika:{index}" in actions(item)) == (count > 1)
    for stage in ("upload", "processing", "ready", "failed"):
        pending = views.samples_panel(
            current, stage=stage, change_id="a" * 32, candidate_count=min(count + 1, 3)
        )
        assert (f"sampleapply:{'a' * 32}" in actions(pending)) == (stage == "ready")
        assert f"samplediscard:{'a' * 32}" in actions(pending)
        assert (f"samplepreview:{'a' * 32}" in actions(pending)) == (stage == "ready")
    with pytest.raises(ValueError):
        views.sample_panel(current, -1)
    with pytest.raises(ValueError):
        views.sample_panel(current, count)


def test_additional_reference_prompt_keeps_voice_name_and_never_asks_for_a_new_name() -> None:
    prompt = views.draft_panel("reference", "a" * 32, profile_name="Вика")
    assert "Вика" in prompt.text and "15–20 секунд" in prompt.text
    assert "тот же человек" in prompt.text and "Применить" in prompt.text
    assert actions(prompt) == [f"draftcancel:{'a' * 32}"]


class PanelHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.open_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        assert tag in {"b", "code"} and attrs == []
        self.open_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        assert self.open_tags.pop() == tag


def test_html_escapes_profile_names_instructions_and_custom_cues_on_every_screen() -> None:
    name = '<a href="tg://user?id=7">Вика & Влад</a>'
    instruction = '</b><script>alert("hi")</script> Говори мягко & спокойно'
    tag = "<i>soft & calm</i>"
    current = profile(name=name).model_copy(
        update={"style_instruction": instruction, "native_tags": (tag,)}
    )
    panels = [views.start_panel(current, ACCOUNT)]
    panels.extend(all_voice_panels(current))
    panels.append(views.draft_panel("rename", "a" * 32, profile_name=name))
    for panel in panels:
        parser = PanelHTML()
        parser.feed(panel.text)
        parser.close()
        assert not parser.open_tags
        assert name not in panel.text and instruction not in panel.text and tag not in panel.text
    assert escape(name, quote=False) in views.start_panel(current, ACCOUNT).text
    assert escape(instruction, quote=False) in views.instruction_panel(current).text
    assert escape(tag, quote=False) in views.selected_delivery_panel(current).text
    assert views.catalog_panel([current], None).rows[0][0].text == name


def test_escaped_maximum_input_fits_saved_panel() -> None:
    current = profile(name="&" * 80).model_copy(
        update={"style_instruction": "&" * 500, "native_tags": ("&" * 100,)}
    )
    for panel in all_voice_panels(current):
        assert len(panel.text) <= 3500


def test_default_voice_has_all_sound_controls_without_exposing_reference_samples() -> None:
    card = views.voice_panel(profile(), "vika", allow_samples=False)
    assert "samples:vika" not in actions(card)
    assert {"rename:vika", "delete:vika", "listen:vika", "cs:vika", "cc:vika:e:0"} <= set(
        actions(card)
    )
    assert "samples:vika" in actions(views.voice_panel(profile(), "vika"))


def test_account_identity_is_escaped_and_never_shows_a_foreign_identifier() -> None:
    account = Account(
        user_id=42,
        is_admin=False,
        balance_ms=600000,
        reserved_ms=0,
        username="test<&>",
        display_name="</b><script>Имя & фамилия</script>",
    )
    panel = views.account_panel(account)
    assert "@test&lt;&amp;&gt;" in panel.text
    assert "&lt;/b&gt;&lt;script&gt;Имя &amp; фамилия&lt;/script&gt;" in panel.text
    assert "<code>42</code>" in panel.text
    parser = PanelHTML()
    parser.feed(panel.text)
    parser.close()
    assert not parser.open_tags
