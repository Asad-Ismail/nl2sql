#!/usr/bin/env python3
"""
Download Spider SQLite databases for evaluation.

The databases are required for executing SQL queries during evaluation.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

# URLs to try (in order of preference)
DOWNLOAD_URLS = [
    "https://github.com/taoyds/spider/releases/download/v1.0/databases.zip",
    "http://yangleshu.com/Spider/databases.zip",
]

def download_file(url: str, dest_path: Path) -> bool:
    """Download file from URL to dest_path"""
    print(f"Attempting to download from: {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    db_dir = base_dir / "database" / "spider_data" / "database"
    db_dir.mkdir(parents=True, exist_ok=True)

    zip_path = base_dir / "database" / "spider_data" / "databases.zip"

    print("=" * 70)
    print("Spider Database Downloader")
    print("=" * 70)
    print(f"\nTarget directory: {db_dir}\n")

    # Try each URL
    for url in DOWNLOAD_URLS:
        if download_file(url, zip_path):
            print(f"\n✓ Downloaded: {zip_path}")
            print(f"  Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
            break
    else:
        print("\n❌ All download attempts failed!")
        print("\nPlease download manually from one of these URLs:")
        for url in DOWNLOAD_URLS:
            print(f"  - {url}")
        print(f"\nAnd extract to: {db_dir}")
        return 1

    # Extract
    print("\nExtracting databases...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(db_dir.parent)
        print("✓ Extracted successfully")

        # Count databases
        sqlite_files = list(db_dir.rglob("*.sqlite"))
        print(f"\n✓ Found {len(sqlite_files)} SQLite databases")

        # Clean up zip file
        zip_path.unlink()
        print(f"\n✓ Setup complete! Databases ready at: {db_dir}")

        return 0
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
