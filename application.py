from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel

import nutrition_extractor.detection

application = FastAPI()
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nutrition_extractor.detection.load_model()
nutrition_extractor.detection.load_text_model()



@application.get("/")
async def root():
    return {"message": "Hello World"}

class LabelIdModel(BaseModel):
    label_id: str

@application.post("/ocr")
async def ocr(label_id_model: LabelIdModel):
    if not Path(label_id_model.label_id).exists():
        raise HTTPException(status_code=404, detail="label image not found")

    # for now, label_id should be the name of an image located in the /backend directory.
    # in future, based on our implementation plan, this will fetch the image from an AWS S3 bucket.
    res: dict = nutrition_extractor.detection.detect(label_id_model.label_id, True)

    return res

class B64Image(BaseModel):
    b64: str

@application.post("/ocrb64")
async def ocr(b64_im: B64Image):
    from PIL import Image
    import base64
    from io import BytesIO
    b64 = b64_im.b64
    if ";base64," in b64:
        b64 = b64.split(";base64,")[1]

    bytes_decoded = base64.b64decode(b64)
    img = Image.open(BytesIO(bytes_decoded))
    img.save("tmp.jpeg")

    # for now, b64 is a base64-encoded image.
    # in future, based on our implementation plan, this will fetch the image from an AWS S3 bucket.
    res: dict = nutrition_extractor.detection.detect("tmp.jpeg", True)
    print(res)
    return res
