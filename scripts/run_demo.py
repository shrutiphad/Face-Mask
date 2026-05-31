#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import httpx

BASE_URL  = "http://localhost:8000"
THRESHOLD = 0.35
SAMPLE    = Path(__file__).parent.parent / "sample"


# def post_search(client, img_path):
#     with open(img_path, "rb") as f:
#         t0 = time.perf_counter()
#         r  = client.post(
#             "/search",
#             files={"image": (img_path.name, f, "image/jpeg")},
#             timeout=120.0,
#         )
#         wall_ms = (time.perf_counter() - t0) * 1000
#     data = r.json()
#     data["_wall_ms"] = round(wall_ms, 1)
#     return data

def post_search(client, img_path):
    with open(img_path, "rb") as f:
        t0 = time.perf_counter()
        r  = client.post(
            "/search",
            files={"image": (img_path.name, f, "image/jpeg")},
            timeout=120.0,
        )
        wall_ms = (time.perf_counter() - t0) * 1000
    
    print(f"  HTTP status: {r.status_code}")
    print(f"  Response: {r.text[:200]}")
    
    data = r.json()
    data["_wall_ms"] = round(wall_ms, 1)
    return data

def main(base_url):
    print(f"\n{'='*60}")
    print("  Face Match Demo  —  numbers for RESULTS.md")
    print(f"{'='*60}")
    print(f"  Service   : {base_url}")
    print(f"  Threshold : {THRESHOLD}")

    # ── Check service ────────────────────────────────────────────────
    with httpx.Client(base_url=base_url, timeout=15) as client:
        h = client.get("/health").json()
        print(f"  Status    : {h.get('ok')}")
        print(f"  Enrolled  : {h.get('enrolled_count', 0)} identities")

        if h.get("enrolled_count", 0) == 0:
            print("\n  [ERROR] Nothing enrolled. Run enroll_test_faces.py first.")
            return

    # ── Locate query image ───────────────────────────────────────────
    query_file = SAMPLE / "query_should_match_id0.jpg"
    if not query_file.exists():
        print(f"\n  [ERROR] File not found: {query_file}")
        print("  Add a second photo of id0 named query_should_match_id0.jpg to sample/")
        return

    # ── Locate enrol images for hard-negative test ───────────────────
    enrol_images = sorted(
        p for p in SAMPLE.glob("*.jpg")
        if not p.name.startswith("query_")
    )

    latencies = []

    with httpx.Client(base_url=base_url) as client:

        # ── TEST 1: Genuine query ────────────────────────────────────
        print(f"\n{'-'*60}")
        print("  Test 1 — Genuine query")
        print(f"  File: {query_file.name}")

        data     = post_search(client, query_file)
        match_id = data.get("match_id")
        cosine   = data.get("cosine", 0.0)
        is_match = cosine >= THRESHOLD
        correct  = (match_id == "id0")
        latencies.append(data["_wall_ms"])

        print(f"\n  Top match   : {match_id}")
        print(f"  Cosine score: {cosine:.4f}")
        print(f"  is_match    : {is_match}  (threshold={THRESHOLD})")
        print(f"  Correct?    : {'yes ✓' if correct else 'NO ✗  (expected id0)'}")
        print(f"  Latency     : {data['_wall_ms']:.1f} ms")

        # ── TEST 2: Hard negative ────────────────────────────────────
        print(f"\n{'-'*60}")
        print("  Test 2 — Hard negative")
        print("  Searching each enrol image, looking for cross-identity confusion")

        hard_score = 0.0
        hard_pair  = ("—", "—")

        for img in enrol_images:
            probe_id = img.name.split("_")[0]
            d        = post_search(client, img)
            returned = d.get("match_id")
            score    = d.get("cosine", 0.0)
            latencies.append(d["_wall_ms"])

            if returned != probe_id and score > hard_score:
                hard_score = score
                hard_pair  = (probe_id, returned)

        if hard_pair[0] == "—":
            print("\n  No cross-identity confusion found — good separation.")
            print("  Showing self-match scores for reference:")
            for img in enrol_images:
                probe_id = img.name.split("_")[0]
                d        = post_search(client, img)
                print(f"    {probe_id} → {d.get('match_id')}  score={d.get('cosine',0):.4f}")
        else:
            print(f"\n  Hard-negative pair  : {hard_pair[0]} vs {hard_pair[1]}")
            print(f"  Cosine score        : {hard_score:.4f}")
            print(f"  Decision            : {'MATCH (false positive!)' if hard_score >= THRESHOLD else 'NO-MATCH ✓ (correctly rejected)'}")

    # ── Latency report ───────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  Latency")
    print(f"  Avg : {sum(latencies)/len(latencies):.1f} ms")
    print(f"  Min : {min(latencies):.1f} ms")
    print(f"  Max : {max(latencies):.1f} ms")

    # ── Threshold justification ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("  THRESHOLD JUSTIFICATION")
    print(f"{'='*60}")
    print(f"""
  Threshold: {THRESHOLD}

  ArcFace (buffalo_l) genuine pairs score >= 0.40.
  Impostors and look-alikes score <= 0.28.
  0.35 sits in the gap — conservative toward false negatives.

  False positive (wrong person accepted) = security breach. Worse.
  False negative (right person rejected) = user retries. Fine.
  So we set threshold above the impostor ceiling (0.28).
""")
    print("  Copy the numbers above into RESULTS.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    main(args.base_url)