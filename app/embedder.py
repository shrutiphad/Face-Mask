import logging
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from app.config import get_settings

logger = logging.getLogger(__name__)


class FaceEmbedder:
   

    def __init__(self) -> None:
        settings = get_settings()
        logger.info(
            "Loading InsightFace model pack '%s' …",
            settings.insightface_model_pack,
        )
        
        
        # FaceAnalysis downloads the model pack on first run and caches it
       
        self._app = FaceAnalysis(
            name=settings.insightface_model_pack,
            root=settings.insightface_model_dir,
            providers=["CPUExecutionProvider"],
        )
        
        # prepare() compiles the ONNX graph and sets the detection input size.
        
        self._app.prepare(
            ctx_id=0,
            det_size=(settings.det_size, settings.det_size),
        )
        logger.info("InsightFace ready — det_size=%d", settings.det_size)


    def embed(self, image_bytes: bytes) -> np.ndarray:
      
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(
                "Could not decode image — check that the file is a valid "
                "JPEG, PNG, or WebP."
            )

       
        faces = self._app.get(bgr)

        if not faces:
            raise ValueError(
                "No face detected in the image.  Make sure the face is "
                "clearly visible, well-lit, and not occluded."
            )

        
        best_face = max(faces, key=lambda f: f.det_score)
        logger.debug(
            "Detected %d face(s), using best (det_score=%.3f)",
            len(faces),
            best_face.det_score,
        )

    
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Embedding has zero norm — this should never happen.")
        embedding = embedding / norm 

        return embedding