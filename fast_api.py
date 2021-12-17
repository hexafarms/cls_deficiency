from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
from predict import predict_api as predict

app = FastAPI()

@app.get("/")
def read_root():
    return {"Deficiency": "Prediction"}

@app.post("/deficiency")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"fast_api/input/{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    ############################ Current Best Model ###############################
    config_file = "/weights/v0/config.py"
    checkpoint_file = "/weights/v0/weights.pth"
    ################################################################################

    input_dir = f"fast_api/input/{file.filename}"
    output_dir = "fast_api/output/"

    predict(config_file, checkpoint_file, input_dir, output_dir) 
    image = open(output_dir+f"/{file.filename}", 'rb')

    return StreamingResponse(image, media_type=("image/jpeg"or"image/png"))
    
