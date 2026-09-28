"""Загрузка набора данных Empa Aurora с Zenodo с докачкой и проверкой целостности."""

import hashlib
import logging
import zipfile
from pathlib import Path

import requests
import typer
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATASET_URL = "https://zenodo.org/api/records/15481956/files/Dataset-rocrate.zip/content"
EXPECTED_MD5 = "eaec9549b74b59d998e5138dab965b5d"

app = typer.Typer(help="Загрузка набора данных Empa Aurora.")


def _remote_size(url: str) -> int | None:
    """Размер удалённого файла по заголовку ответа."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        length = response.headers.get("Content-Length")
        return int(length) if length else None
    except requests.RequestException:
        return None


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Загружает файл с докачкой по Range при обрыве соединения."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_size = _remote_size(url)
    while True:
        local_size = dest.stat().st_size if dest.exists() else 0
        if remote_size is not None and local_size >= remote_size:
            break
        headers = {"Range": f"bytes={local_size}-"} if local_size else {}
        with requests.get(url, stream=True, headers=headers, timeout=60) as response:
            if local_size and response.status_code == 200:
                # Сервер не поддержал Range: начинаем заново.
                local_size = 0
            response.raise_for_status()
            total = remote_size or int(response.headers.get("Content-Length", 0))
            mode = "ab" if local_size else "wb"
            with (
                open(dest, mode) as handle,
                tqdm(
                    total=total,
                    initial=local_size,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
                        bar.update(len(chunk))
        if remote_size is None:
            break
    return dest


def verify_md5(path: Path, expected: str, chunk_size: int = 1 << 22) -> bool:
    """Проверяет контрольную сумму MD5 файла."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        logger.error("Несовпадение MD5: ожидалось %s, получено %s", expected, actual)
        return False
    logger.info("Контрольная сумма MD5 подтверждена: %s", actual)
    return True


def extract_archive(archive: Path, dest_dir: Path) -> Path:
    """Распаковывает архив набора данных в каталог назначения."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    logger.info("Архив распакован в %s", dest_dir)
    return dest_dir


@app.command()
def main(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Каталог для сырых данных."),
    skip_md5: bool = typer.Option(False, help="Пропустить проверку контрольной суммы."),
) -> None:
    """Загружает архив набора данных и распаковывает его в data/raw/aurora/."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    archive = raw_dir / "Dataset-rocrate.zip"
    logger.info("Загрузка %s в %s", DATASET_URL, archive)
    download_file(DATASET_URL, archive)
    if not skip_md5 and not verify_md5(archive, EXPECTED_MD5):
        raise typer.Exit(code=1)
    extract_archive(archive, raw_dir / "aurora")
    logger.info("Загрузка и распаковка завершены.")


if __name__ == "__main__":
    app()
