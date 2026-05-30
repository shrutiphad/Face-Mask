
import os
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import numpy as np

OUT = "sample"
os.makedirs(OUT, exist_ok=True)

# people with >=2 images so we can take an enroll + a separate query for #0
data = fetch_lfw_people(min_faces_per_person=2, color=True, resize=1.0)
images, targets, names = data.images, data.target, data.target_names

picked, seen = [], set()
for idx, t in enumerate(targets):
    if t not in seen:
        seen.add(t)
        picked.append((t, idx))
    if len(picked) == 5:
        break

for rank, (t, idx) in enumerate(picked):
    arr = (images[idx] * 255).astype(np.uint8) if images[idx].max() <= 1 else images[idx].astype(np.uint8)
    Image.fromarray(arr).save(f"{OUT}/id{rank}_{names[t].replace(' ', '_')}.jpg")

# a DIFFERENT image of identity #0 as the query
t0 = picked[0][0]
for idx, t in enumerate(targets):
    if t == t0 and idx != picked[0][1]:
        arr = (images[idx] * 255).astype(np.uint8) if images[idx].max() <= 1 else images[idx].astype(np.uint8)
        Image.fromarray(arr).save(f"{OUT}/query_should_match_id0.jpg")
        break

print("Wrote 5 enroll images + 1 query to ./sample")
print("Hard negative: pick any two visually similar identities yourself and document it.")
