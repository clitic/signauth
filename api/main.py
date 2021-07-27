from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
from rich.console import Console
from torch import jit
from signauth import transform, processimage
import os
import shutil
import uuid


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
console = Console()

console.log("Loading signauth model")
model = jit.load("model.pth")
model.eval()
console.log("SignAuth model loaded")


@app.post("/upload_image/")
async def create_upload_image(request: Request, file: UploadFile = File(...), scan: Optional[bool] = Query(False, description="add a scan filter to image while processing")):
    if file.content_type in ["image/jpeg", "image/jpg", "image/png"]:

        if int(request.headers.get("content-length", 0)) > 10000000:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "too large image to process (maximum: 10 M.B.)")

        image_path = os.path.join("uploaded_images", str(uuid.uuid1()))
        image_log_path = image_path + ".log"
        image_path += file.content_type.replace("image/", ".")

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        console.log(f"Making predictions for {image_path}")
        classes = ["fake", "real", None]
        try:
            img = processimage(image_path, scan=scan)
            img = transform(img).float().unsqueeze(0)
            output = model(img)
            index = output.data.numpy().argmax()
            console.log(f"Prediction for {image_path} was {classes[index]}")
        except:
            console.log(f"Predictions couldn't be made for {image_path}")
            index = 2

        try:
            os.remove(image_path)
            console.log(f"Successfully removed {image_path}")
            new_image_path, _ = os.path.splitext(image_path)
            new_image_path += "_signauth_processed.jpg"
            os.remove(new_image_path)
            console.log(f"Successfully removed {new_image_path}")
        except:
            console.log(f"Failed to remove {image_path}")

        return {"prediction": classes[index], "scan": scan}

    else:
        raise HTTPException(status.HTTP_405_METHOD_NOT_ALLOWED, "cannot process non image files")


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
