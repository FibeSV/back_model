import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import keras

# App creation and model loading
app = FastAPI()
model = keras.models.load_model("./model.h5")

class MFcc(BaseModel):
    """
    Input features validation for the ML model
    """
    coefs: list



@app.post('/predict')
def predict(data: MFcc):
    """
    :param iris: input data from the post request
    :return: predicted iris type
    """
    features = [data.coefs]
    prediction = model.predict(features).tolist()[0]
    return {
        "prediction": prediction
    }


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
