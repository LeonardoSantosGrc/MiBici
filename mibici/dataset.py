import csv
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger
import requests
import typer

from mibici.config import PROJ_ROOT, RAW_DATA_DIR

app = typer.Typer()

OPEN_DATA_URL = "https://mibici.net/en/open-data/"
MANIFEST_PATH = PROJ_ROOT / "references" / "data_manifest.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
}


def discover_csv_urls() -> list[str]:
    """Discover CSV files published on the official MiBici open-data page."""
    logger.info(f"Reading MiBici catalog: {OPEN_DATA_URL}")

    response = requests.get(
        OPEN_DATA_URL,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    csv_urls = {
        urljoin(OPEN_DATA_URL, link["href"])
        for link in soup.find_all("a", href=True)
        if link["href"].lower().split("?")[0].endswith(".csv")
    }

    return sorted(
        csv_urls,
        key=lambda url: Path(urlparse(url).path).name,
    )


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a local file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_file(url: str, destination: Path) -> str:
    """Download a CSV file to raw storage if it is not already present."""
    if destination.exists() and destination.stat().st_size > 0:
        logger.info(f"Skipping existing file: {destination.name}")
        return "existing"

    temporary_path = destination.with_suffix(destination.suffix + ".part")

    logger.info(f"Downloading: {destination.name}")

    with requests.get(
        url,
        headers=HEADERS,
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()

        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    if not temporary_path.exists() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {destination.name}")

    temporary_path.replace(destination)

    return "downloaded"


def write_manifest(
    records: list[dict[str, str | int]],
    manifest_path: Path,
) -> None:
    """Write the raw-data manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "url",
                "size_bytes",
                "sha256",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


@app.command()
def main(
    limit: int | None = typer.Option(
        None,
        help="Maximum number of CSV files to process. Useful for testing.",
    ),
) -> None:
    """Download MiBici open-data CSV files into data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    urls = discover_csv_urls()

    logger.info(f"CSV files discovered: {len(urls)}")

    if limit is not None:
        urls = urls[:limit]
        logger.info(f"Download limit enabled: {limit}")

    records: list[dict[str, str | int]] = []

    for index, url in enumerate(urls, start=1):
        filename = Path(urlparse(url).path).name
        destination = RAW_DATA_DIR / filename

        logger.info(f"[{index}/{len(urls)}] {filename}")

        status = download_file(url, destination)

        records.append(
            {
                "filename": filename,
                "url": url,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "status": status,
            }
        )

    manifest_path = (
        PROJ_ROOT / "references" / "data_manifest_test.csv" if limit is not None else MANIFEST_PATH
    )

    write_manifest(records, manifest_path)

    logger.success(f"Raw data ready: {len(records)} files. Manifest: {manifest_path}")


if __name__ == "__main__":
    app()
