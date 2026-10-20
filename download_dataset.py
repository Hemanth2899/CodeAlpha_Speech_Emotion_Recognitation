"""Download and extract the official RAVDESS audio-speech dataset."""
from pathlib import Path
import urllib.request
import zipfile
import sys

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "Audio_Speech_Actors_01-24.zip"
RAVDESS_DIR = DATA_DIR / "ravdess"
URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"


def progress(block_num, block_size, total_size):
    if total_size > 0:
        percent = min(block_num * block_size * 100 / total_size, 100)
        print(f"\rDownloading: {percent:5.1f}%", end="", flush=True)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if any(RAVDESS_DIR.rglob("*.wav")) if RAVDESS_DIR.exists() else False:
        print(f"Dataset already appears to be extracted at: {RAVDESS_DIR}")
        print("If you want to download it again, remove that folder first.")
        return

    if not ZIP_PATH.exists():
        print("Downloading RAVDESS speech audio dataset (this may take a while)...")
        print("Source: official RAVDESS record on Zenodo")
        try:
            urllib.request.urlretrieve(URL, ZIP_PATH, progress)
            print("\nDownload complete.")
        except Exception as exc:
            print(f"\nDownload failed: {exc}")
            print("Open the RAVDESS Zenodo page from README.md and download")
            print("Audio_Speech_Actors_01-24.zip manually into the data folder.")
            sys.exit(1)
    else:
        print(f"Using existing archive: {ZIP_PATH}")

    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(RAVDESS_DIR)
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid ZIP. Delete it and run again.")
        sys.exit(1)

    wav_count = sum(1 for _ in RAVDESS_DIR.rglob("*.wav"))
    print(f"Extracted {wav_count} WAV files to: {RAVDESS_DIR}")
    if wav_count == 0:
        print("No WAV files were found. Check the archive structure.")
        sys.exit(1)
    print("Dataset setup complete. Next run: python main.py --train")


if __name__ == "__main__":
    main()
