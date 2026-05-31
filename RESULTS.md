Live URL: https://shrutidphad-face-mask.hf.space

Embedding model: InsightFace buffalo_s (ArcFace MobileNet)
Face detector/aligner: RetinaFace + 5-point affine (built into InsightFace)
Qdrant: Qdrant Cloud (free tier)

Enrolled identities:
id0 - personal photo (Shruti)
id1 - personal photo (Himi)
id2 - personal photo (Anki)
id3 - personal photo (Shubhrah)

Query result:
Image: query_should_match_id0.jpg
Top match: id0
Cosine score: 0.6028
Correct: yes

Hard negative:
No cross-identity confusion found at threshold 0.35
All self-match scores: 1.0000
Threshold: 0.35
Why: genuine pairs score >= 0.40, impostors <= 0.28
False positive = security breach (worse)
False negative = user retries (recoverable)

Search latency:
Avg: 97 ms (Qdrant ANN only)
End-to-end avg: 1394 ms (includes embedding)