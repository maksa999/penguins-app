import json
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def feedforward(self, inputs):
        return sigmoid(np.dot(inputs, self.weights) + self.bias)

class NeuronLinear:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def feedforward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

class ClassificationNetwork:
    def __init__(self, data):
        self.hidden_layer = [Neuron(n["weights"], n["bias"]) for n in data["hidden"]]
        self.output_layer = [Neuron(n["weights"], n["bias"]) for n in data["output"]]
        self.mn = np.array(data["mn"])
        self.mx = np.array(data["mx"])
        self.names = data["names"]

    def feedforward(self, inputs):
        hidden = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        return np.array([n.feedforward(hidden) for n in self.output_layer])

    def predict(self, inputs):
        x = (inputs - self.mn) / (self.mx - self.mn)
        outputs = self.feedforward(x)
        return self.names[int(np.argmax(outputs))]

class RegressionNetwork:
    def __init__(self, data):
        self.hidden_layer = [Neuron(n["weights"], n["bias"]) for n in data["hidden"]]
        self.output_neuron = NeuronLinear(
            data["output"]["weights"],
            data["output"]["bias"],
        )
        self.mn = np.array(data["mn"])
        self.mx = np.array(data["mx"])
        self.ymn = data["ymn"]
        self.ymx = data["ymx"]

    def feedforward(self, inputs):
        hidden = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        return self.output_neuron.feedforward(hidden)

    def predict(self, inputs):
        x = (inputs - self.mn) / (self.mx - self.mn)
        y_n = self.feedforward(x)
        return y_n * (self.ymx - self.ymn) + self.ymn

with open("classification_weights.json", "r", encoding="utf-8") as f:
    clf = ClassificationNetwork(json.load(f))

with open("regression_weights.json", "r", encoding="utf-8") as f:
    reg = RegressionNetwork(json.load(f))

print("Модели загружены.")
print("Классы:", clf.names)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )

@app.get("/classify", response_class=HTMLResponse)
def classify_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="classify.html",
        context={"prediction": None, "form": None},
    )

@app.post("/classify", response_class=HTMLResponse)
def classify_post(
    request: Request,
    bill_length: float = Form(...),
    bill_depth: float = Form(...),
    flipper_length: float = Form(...),
    sex: str = Form(...),
):
    is_male = 1.0 if sex == "male" else 0.0
    is_female = 1.0 if sex == "female" else 0.0

    features = np.array([bill_length, bill_depth, flipper_length, is_male, is_female])
    prediction = clf.predict(features)

    form = {
        "bill_length": bill_length,
        "bill_depth": bill_depth,
        "flipper_length": flipper_length,
        "sex": sex,
    }

    return templates.TemplateResponse(
        request=request,
        name="classify.html",
        context={"prediction": prediction, "form": form},
    )

@app.get("/regress", response_class=HTMLResponse)
def regress_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="regress.html",
        context={"prediction": None, "form": None},
    )

@app.post("/regress", response_class=HTMLResponse)
def regress_post(
    request: Request,
    bill_length: float = Form(...),
    bill_depth: float = Form(...),
    flipper_length: float = Form(...),
    sex: str = Form(...),
):
    is_male = 1.0 if sex == "male" else 0.0
    is_female = 1.0 if sex == "female" else 0.0

    features = np.array([bill_length, bill_depth, flipper_length, is_male, is_female])
    prediction = reg.predict(features)

    form = {
        "bill_length": bill_length,
        "bill_depth": bill_depth,
        "flipper_length": flipper_length,
        "sex": sex,
    }

    return templates.TemplateResponse(
        request=request,
        name="regress.html",
        context={"prediction": round(float(prediction), 1), "form": form},
    )