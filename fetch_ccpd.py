#!/usr/bin/env python3
import argparse
import subprocess
import json
import sys
import os
import time
import collections
import requests
from tqdm import tqdm

BUCKET = "licenseplate-dataset"
FILE = "CCPD_AUGMENTED.tar"
OUTPUT = FILE

SPEED_THRESHOLD = 100 * 1024 * 1024  # 100 MB/s in bytes
WINDOW_SIZE = 5.0                     # seconds for sliding window rate
GRACE_PERIOD = 3.0                    # seconds below threshold before killing
BACKOFF = 5.0                         # seconds to wait before reconnecting
READ_CHUNK = 4 * 1024 * 1024          # 4MB read chunks
TIMEOUT = 30                          # socket timeout in seconds


def get_download_url():
    result = subprocess.run(
        ["b2", "get-download-url-with-auth", "--duration", "86400", BUCKET, FILE],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"b2 get-download-url-with-auth failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def get_file_size():
    result = subprocess.run(
        ["b2", "ls", "--json", f"b2://{BUCKET}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"b2 ls failed:\n{result.stderr.strip()}")
    files = json.loads(result.stdout)
    matches = [f for f in files if f["fileName"] == FILE]
    if not matches:
        raise RuntimeError(f"File '{FILE}' not found in bucket '{BUCKET}'")
    return matches[0]["size"]


def download(url, file_size, output_path=None):
    """Download to output_path, or to stdout if output_path is None."""
    stdout_mode = output_path is None
    log = sys.stderr if stdout_mode else sys.stdout

    cursor = 0

    if not stdout_mode:
        if os.path.exists(output_path):
            cursor = os.path.getsize(output_path)
            if cursor == file_size:
                print("File already fully downloaded.")
                return
            print(f"Resuming from byte {cursor:,} ({cursor / 1024**3:.2f} GB)")
        else:
            with open(output_path, "wb") as f:
                f.truncate(file_size)

    attempt = 0

    with tqdm(
        total=file_size,
        initial=cursor,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=FILE,
        dynamic_ncols=True,
        file=log,
    ) as pbar:
        out = sys.stdout.buffer if stdout_mode else open(output_path, "r+b")
        try:
            while cursor < file_size:
                attempt += 1
                headers = {"Range": f"bytes={cursor}-{file_size - 1}"}

                print(f"\n[attempt {attempt}] Connecting from byte {cursor:,}...", file=log)

                try:
                    with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
                        resp.raise_for_status()
                        if not stdout_mode:
                            out.seek(cursor)

                        slow_since = None
                        window = collections.deque()  # (timestamp, bytes)

                        for chunk in resp.iter_content(chunk_size=READ_CHUNK):
                            if not chunk:
                                continue

                            out.write(chunk)
                            if stdout_mode:
                                out.flush()
                            cursor += len(chunk)
                            pbar.update(len(chunk))

                            now = time.monotonic()
                            window.append((now, len(chunk)))

                            # Evict entries older than WINDOW_SIZE seconds
                            while window and now - window[0][0] > WINDOW_SIZE:
                                window.popleft()

                            # Compute rate over the window
                            if len(window) > 1:
                                window_elapsed = now - window[0][0]
                                window_bytes = sum(b for _, b in window)
                                rate = window_bytes / window_elapsed if window_elapsed > 0 else None
                            else:
                                rate = None

                            if rate is not None and rate < SPEED_THRESHOLD:
                                if slow_since is None:
                                    slow_since = now
                                elif now - slow_since >= GRACE_PERIOD:
                                    print(
                                        f"\n[throttle] Speed {rate / 1024**2:.1f} MB/s "
                                        f"below threshold for {GRACE_PERIOD}s — "
                                        f"dropping connection, backing off {BACKOFF}s...",
                                        file=log,
                                    )
                                    break
                            else:
                                slow_since = None

                        else:
                            break  # iter_content exhausted cleanly — done

                except (requests.exceptions.RequestException, OSError) as e:
                    print(f"\n[error] {e}", file=sys.stderr)

                print(f"Backing off {BACKOFF}s...", file=log)
                time.sleep(BACKOFF)
        finally:
            if not stdout_mode:
                out.close()

    if not stdout_mode:
        actual_size = os.path.getsize(output_path)
        if actual_size != file_size:
            raise RuntimeError(f"Size mismatch: expected {file_size}, got {actual_size}")
        print(f"\nDone: {output_path} ({actual_size:,} bytes)")
    else:
        print(f"\nDone: streamed {cursor:,} bytes to stdout", file=log)


def main():
    parser = argparse.ArgumentParser(description="Download CCPD dataset from Backblaze B2")
    parser.add_argument(
        "--stdout", action="store_true",
        help="Stream downloaded bytes to stdout instead of writing to a file"
    )
    args = parser.parse_args()

    log = sys.stderr if args.stdout else sys.stdout

    print("Getting download URL...", file=log)
    url = get_download_url()

    print("Getting file size...", file=log)
    file_size = get_file_size()
    print(f"File size: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)", file=log)

    download(url, file_size, output_path=None if args.stdout else OUTPUT)


if __name__ == "__main__":
    main()
