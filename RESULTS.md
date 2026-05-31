Live URL:          (paste after deployment)
Embedding model:   InsightFace buffalo_l (ArcFace ResNet-100)
Detector/aligner:  RetinaFace + 5-point affine (built into InsightFace)
Qdrant:            local Docker / Qdrant Cloud (after deploy)

Query result:
  Image:      query_should_match_id0.jpg
  Top match:  id0
  Cosine:     0.6652
  Correct:    yes

Hard negative:
  No cross-identity confusion at threshold 0.35
  All self-match scores: 1.0000
  Threshold: 0.35
  Why: genuine pairs score >= 0.40, impostors <= 0.28

Search latency:   ~1305 ms avg (includes embedding time)