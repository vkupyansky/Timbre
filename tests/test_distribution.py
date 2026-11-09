import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_dist import MAX_ARCHIVE_BYTES, DistributionError, verify_distribution

ROOT = "shift-0.1.0"
SOURCE_APP = f"{ROOT}/src/shift/app.py"
WHEEL_APP = "shift/app.py"
SOURCE_NOTICES = [
    (f"{ROOT}/NOTICE", b"model attribution"),
    (f"{ROOT}/licenses/FISH-AUDIO-RESEARCH-LICENSE.md", b"model license"),
]
WHEEL_NOTICES = [
    (f"{ROOT}.dist-info/licenses/NOTICE", b"model attribution"),
    (f"{ROOT}.dist-info/licenses/licenses/FISH-AUDIO-RESEARCH-LICENSE.md", b"model license"),
]


def sdist(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def wheel(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)


def test_expected_package_and_metadata_accepted(tmp_path: Path) -> None:
    source = tmp_path / f"{ROOT}.tar.gz"
    binary = tmp_path / f"{ROOT}-py3-none-any.whl"
    sdist(
        source,
        [
            (SOURCE_APP, b"pass\n"),
            (f"{ROOT}/src/shift/whisper_cache.py", b"pass\n"),
            (f"{ROOT}/PKG-INFO", b"metadata"),
            *SOURCE_NOTICES,
        ],
    )
    wheel(
        binary,
        [
            (WHEEL_APP, b"pass\n"),
            ("shift/whisper_cache.py", b"pass\n"),
            (f"{ROOT}.dist-info/METADATA", b"metadata"),
            *WHEEL_NOTICES,
        ],
    )
    assert verify_distribution(source).files == 5
    assert verify_distribution(binary).files == 5


@pytest.mark.parametrize("omitted", [0, 1])
def test_distribution_requires_both_model_notices(tmp_path: Path, omitted: int) -> None:
    source = tmp_path / f"{ROOT}.tar.gz"
    binary = tmp_path / f"{ROOT}-py3-none-any.whl"
    sdist(source, [(SOURCE_APP, b"pass\n"), SOURCE_NOTICES[1 - omitted]])
    wheel(binary, [(WHEEL_APP, b"pass\n"), WHEEL_NOTICES[1 - omitted]])
    for path in (source, binary):
        with pytest.raises(DistributionError, match="required_notice_missing"):
            verify_distribution(path)


@pytest.mark.parametrize(
    "private_path",
    [
        ".shift/secrets/telegram.token",
        ".shift/data/shift.sqlite3",
        ".shift/data/profiles/anna/whisper-context.json",
        "models/weights.bin",
        "artifacts/voice.ogg",
        "tests/test_bot.py",
        ".venv/bin/python",
        "src/shift/__pycache__/app.pyc",
        "src/shift/secrets.py",
        "src/shift/whisper-context.json",
        "licenses/worker.token",
    ],
)
def test_source_archive_rejects_private_and_unreviewed_paths(
    tmp_path: Path,
    private_path: str,
) -> None:
    path = tmp_path / f"{ROOT}.tar.gz"
    sdist(path, [(SOURCE_APP, b"pass\n"), (f"{ROOT}/{private_path}", b"marker-not-a-credential")])
    with pytest.raises(DistributionError, match="archive_path_not_allowed"):
        verify_distribution(path)


def test_wheel_rejects_secret_path_even_with_valid_application(tmp_path: Path) -> None:
    path = tmp_path / f"{ROOT}-py3-none-any.whl"
    wheel(path, [(WHEEL_APP, b"pass\n"), (".shift/secrets/worker.token", b"marker-only")])
    with pytest.raises(DistributionError, match="archive_path_not_allowed"):
        verify_distribution(path)


@pytest.mark.parametrize("unsafe", ["../outside", "/absolute", "shift/../app.py", "shift\\app.py"])
def test_noncanonical_paths_rejected(tmp_path: Path, unsafe: str) -> None:
    path = tmp_path / f"{ROOT}-py3-none-any.whl"
    wheel(path, [(WHEEL_APP, b"pass\n"), (unsafe, b"marker")])
    with pytest.raises(DistributionError, match="noncanonical_archive_path"):
        verify_distribution(path)


def test_tar_and_wheel_links_rejected(tmp_path: Path) -> None:
    source = tmp_path / f"{ROOT}.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        member = tarfile.TarInfo(SOURCE_APP)
        member.type = tarfile.SYMTYPE
        member.linkname = "outside"
        archive.addfile(member)
    binary = tmp_path / f"{ROOT}-py3-none-any.whl"
    with zipfile.ZipFile(binary, "w") as archive:
        entry = zipfile.ZipInfo(WHEEL_APP)
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, "outside")
    for path in (source, binary):
        with pytest.raises(DistributionError, match="archive_links_or_special_files"):
            verify_distribution(path)


def test_archive_size_rejected_before_opening(tmp_path: Path) -> None:
    path = tmp_path / f"{ROOT}.tar.gz"
    path.write_bytes(b"x" * (MAX_ARCHIVE_BYTES + 1))
    with pytest.raises(DistributionError, match="archive_size_limit"):
        verify_distribution(path)


def test_compressed_expansion_is_bounded(tmp_path: Path) -> None:
    source = tmp_path / f"{ROOT}.tar.gz"
    binary = tmp_path / f"{ROOT}-py3-none-any.whl"
    content = b"x" * (MAX_ARCHIVE_BYTES + 1)
    sdist(source, [(SOURCE_APP, content)])
    wheel(binary, [(WHEEL_APP, content)])
    for path in (source, binary):
        assert path.stat().st_size < MAX_ARCHIVE_BYTES
        with pytest.raises(DistributionError, match="uncompressed_size_limit"):
            verify_distribution(path)


def test_missing_application_and_duplicate_members_rejected(tmp_path: Path) -> None:
    path = tmp_path / f"{ROOT}.tar.gz"
    sdist(path, [(f"{ROOT}/PKG-INFO", b"metadata")])
    with pytest.raises(DistributionError, match="application_missing"):
        verify_distribution(path)
    sdist(path, [(SOURCE_APP, b"first"), (SOURCE_APP, b"second")])
    with pytest.raises(DistributionError, match="duplicate_archive_entry"):
        verify_distribution(path)
