import logging
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from app.config import get_settings

logger = logging.getLogger(__name__)


class FaceEmbedder:

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("Loading InsightFace model pack '%s' ...",
                    settings.insightface_model_pack)
        self._app = FaceAnalysis(
            name=settings.insightface_model_pack,
            root=settings.insightface_model_dir,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(
            ctx_id=0,
            det_size=(settings.det_size, settings.det_size),
        )
        logger.info("InsightFace ready.")

    def embed(self, image_bytes: bytes) -> np.ndarray:
        # Step 1 — decode image bytes into a numpy BGR array
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr is None:
            raise ValueError(
                "Could not decode image. Make sure it is a valid JPEG or PNG."
            )

        # Step 2 — detect + align faces (RetinaFace inside InsightFace)
        faces = self._app.get(bgr)

        if len(faces) == 0:
            raise ValueError(
                "No face detected in the image. "
                "Make sure the face is clearly visible and well-lit."
            )

        # Step 3 — pick the face with highest detection confidence
        best_face = max(faces, key=lambda f: float(f.det_score))

        # Step 4 — extract the ArcFace embedding (already computed by get())
        raw_embedding = best_face.embedding

        if raw_embedding is None:
            raise ValueError(
                "InsightFace returned a face but no embedding. "
                "The model pack may not have downloaded correctly."
            )

        # Step 5 — convert to float32 numpy array
        embedding = np.array(raw_embedding, dtype=np.float32)

        # Step 6 — L2 normalise so cosine similarity = dot product
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Embedding norm is zero — invalid image.")

        embedding = embedding / norm
        return embedding