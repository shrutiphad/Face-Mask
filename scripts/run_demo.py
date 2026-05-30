import argparse
import json
import time
import sys
from pathlib import Path
import httpx


THRESHOLD = 0.35          # must match app/config.py → match_threshold
SAMPLE_BASE = Path(__file__).parent.parent / "sample_faces"


def post_search(client: httpx.Client, img_path: Path) -> dict:
    """POST /search and return the parsed JSON response."""
    with open(img_path, "rb") as f:
        t0 = time.perf_counter()
        response = client.post(
            "/search",
            files={"image": (img_path.name, f, "image/jpeg")},
            timeout=60.0,
        )
        wall_ms = (time.perf_counter() - t0) * 1000

    if response.status_code != 200:
        print(f"ERROR: HTTP {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    data = response.json()
    data["_wall_ms"] = round(wall_ms, 1)
    return data


def print_result(label: str, data: dict, expect_match: bool) -> None:
    top = data.get("top_match") or {}
    identity_id = top.get("identity_id", "—")
    score = top.get("score", 0.0)
    is_match = data.get("is_match", False)
    search_ms = data.get("latency_ms", 0.0)
    wall_ms = data.get("_wall_ms", 0.0)

    status = "✓  PASS" if (is_match == expect_match) else "✗  FAIL"
    match_str = "MATCH  " if is_match else "NO-MATCH"

    print(f"\n{'─'*60}")
    print(f"  Test        : {label}")
    print(f"  Result      : {status}")
    print(f"  Top match   : {identity_id}")
    print(f"  Cosine score: {score:.4f}")
    print(f"  Decision    : {match_str}  (threshold={THRESHOLD})")
    print(f"  Qdrant ANN  : {search_ms:.1f} ms  (wall-clock: {wall_ms:.0f} ms)")


def main(base_url: str) -> None:
    print(f"\n{'═'*60}")
    print("  Face Match Demo — results for RESULTS.md")
    print(f"{'═'*60}")
    print(f"  Service  : {base_url}")
    print(f"  Threshold: {THRESHOLD}")

    # Check the service is up
    with httpx.Client(base_url=base_url, timeout=10) as client:
        try:
            r = client.get("/health")
            health = r.json()
            print(f"  Status   : {health.get('status', 'unknown')}")
            print(f"  Enrolled : {health.get('enrolled_count', '?')} identities")
        except Exception as exc:
            print(f"\nCannot reach {base_url}: {exc}", file=sys.stderr)
            print("Is the service running?  uvicorn app.main:app --port 8000")
            sys.exit(1)

    query_images = sorted((SAMPLE_BASE / "query").glob("*.jpg"))
    neg_images = sorted((SAMPLE_BASE / "hard_negative").glob("*.jpg"))

    if not query_images:
        print("No query images found. Run: python scripts/get_sample_faces.py", file=sys.stderr)
        sys.exit(1)

    all_latencies = []

    with httpx.Client(base_url=base_url) as client:
      
        print(f"\n{'─'*60}")
        print("  Test 1: Genuine query (different photo of enrolled identity)")
        for img in query_images:
            data = post_search(client, img)
            print_result(f"genuine — {img.name}", data, expect_match=True)
            all_latencies.append(data["latency_ms"])

        print(f"\n  Test 2: Hard negative (look-alike, should be rejected)")
        for img in neg_images:
            data = post_search(client, img)
            print_result(f"hard-neg — {img.name}", data, expect_match=False)
            all_latencies.append(data["latency_ms"])

   
    if all_latencies:
        print(f"\n{'─'*60}")
        print("  Search latency (Qdrant ANN only, excludes embedding time)")
        print(f"  Avg : {sum(all_latencies)/len(all_latencies):.1f} ms")
        print(f"  Min : {min(all_latencies):.1f} ms")
        print(f"  Max : {max(all_latencies):.1f} ms")

  
    print(f"\n{'═'*60}")
    print("  THRESHOLD JUSTIFICATION")
    print(f"{'═'*60}")
    print(f"""
  Threshold: {THRESHOLD}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    main(args.base_url)