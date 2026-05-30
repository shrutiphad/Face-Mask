import argparse
import sys
import time
from pathlib import Path

import httpx

SAMPLE_DIR = Path(__file__).parent.parent / "sample"


def extract_id(filename: str) -> str:
    
    return filename.split("_")[0]


def enroll_all(base_url: str) -> None:
    if not SAMPLE_DIR.exists():
        print(
            f"[ERROR] {SAMPLE_DIR} not found.\n"
            "Run:  python scripts/get_sample_faces.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Collect all .jpg files EXCEPT the query image
    # The query file is named query_should_match_id0.jpg — it starts with "query_"
    images = sorted(
        p for p in SAMPLE_DIR.glob("*.jpg")
        if not p.name.startswith("query_")
    )

    if not images:
        print(
            f"[ERROR] No enrolment images found in {SAMPLE_DIR}.\n"
            "Run:  python scripts/get_sample_faces.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n── Enrolling {len(images)} identities → {base_url}/enroll ──────")
    latencies = []

    
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        for img_path in images:
            identity_id = extract_id(img_path.name)

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
                    f"  ✓  {identity_id:<8}  {img_path.name:<40}  "
                    f"{elapsed_ms:6.0f} ms  "
                    f"enrolled_count={body.get('enrolled_count', '?')}"
                )
                latencies.append(elapsed_ms)
            else:
                print(
                    f"  ✗  {identity_id:<8}  {img_path.name:<40}  "
                    f"HTTP {response.status_code}: {response.text}",
                    file=sys.stderr,
                )

    if latencies:
        print(f"\ Summary ")
        print(f"  Enrolled : {len(latencies)}/{len(images)}")
        print(f"  Avg time : {sum(latencies)/len(latencies):.0f} ms/image")
        print(f"  Total    : {sum(latencies):.0f} ms")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrol all sample faces")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running Face Match Service",
    )
    args = parser.parse_args()
    enroll_all(args.base_url)