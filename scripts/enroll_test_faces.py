#!/usr/bin/env python3
import argparse
import time
import sys
from pathlib import Path
import httpx

SAMPLE_DIR = Path(__file__).parent.parent / "sample"


def enroll_all(base_url: str) -> None:
    if not SAMPLE_DIR.exists():
        print(f"[ERROR] {SAMPLE_DIR} not found.")
        sys.exit(1)

    images = sorted(
        p for p in SAMPLE_DIR.glob("*.jpg")
        if not p.name.startswith("query_")
    )

    if not images:
        print(f"[ERROR] No images found in {SAMPLE_DIR}")
        sys.exit(1)

    print(f"\n── Enrolling {len(images)} identities → {base_url}/enroll ──────")

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        for img_path in images:
            identity_id = img_path.name.split("_")[0]
            t0 = time.perf_counter()

            with open(img_path, "rb") as f:
                response = client.post(
                    "/enroll",
                    params={"id": identity_id},
                    files={"image": (img_path.name, f, "image/jpeg")},
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000

            if response.status_code in (200, 201):
                body = response.json()
                print(
                    f"  ✓  {identity_id:<8} {img_path.name:<35} "
                    f"{elapsed_ms:6.0f} ms  "
                    f"enrolled_count={body.get('enrolled_count', '?')}"
                )
            else:
                print(
                    f"  ✗  {identity_id:<8} {img_path.name:<35} "
                    f"HTTP {response.status_code}: {response.text}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    enroll_all(args.base_url)