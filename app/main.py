from fastapi import FastAPI
from starlette.responses import JSONResponse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os


app = FastAPI()

class PytorchMultiClass(nn.Module):
  def __init__(self, num_features):
    super(PytorchMultiClass, self).__init__()
    
    self.layer_1 = nn.Linear(num_features, 32)
    self.layer_out = nn.Linear(32, 104)
    self.softmax = nn.Softmax(dim=1)
  def forward(self, x):
    x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
    x = self.layer_out(x)
    return self.softmax(x)
    
model = PytorchMultiClass(6)

model.load_state_dict(torch.load('../models/pytorch_multi_beer_evaluation.pt'))

eval = model.eval()


@app.get('/')
def read_root():
  return 'A brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project'

@app.get('/ready', status_code=200)
def healthcheck():
  return 'Welcome. NN is all ready to go!'

def format_features(brewery_name: float, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, review_overall: float):
  return {
  	'brewery_name': [brewery_name],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'review_overall': [review_overall]
    }

@app.post("/beers/type/")
def predict(brewery_name: float, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, review_overall: float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste,review_overall)
    obs = sc.transform(pd.DataFrame(features))
    tens = torch.from_numpy(obs).float()
    pred = nn_model(tens)
    pred = pred.argmax(1)
    pred = enc.inverse_transform(pred)
    return JSONResponse(pred.tolist())

@app.post("/beer/type/")
def predict(beer_type):
    pred = enc.inverse_transform(model(beer_type))
    return JSONResponse(pred.tolist())

@app.get("/model/architecture/")
def mod_arch():
    return model

