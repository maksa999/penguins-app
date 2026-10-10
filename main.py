import json
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


# ============ НЕЙРОННАЯ СЕТЬ (та же, что в Colab) ============
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, input_size, activation="sigmoid"):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.activation = activation

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        return sigmoid(total) if self.activation == "sigmoid" else total


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, output_activation="sigmoid"):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size, activation=output_activation)
                             for _ in range(output_size)]

    def feedforward(self, inputs):
        hidden = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        outputs = np.array([n.feedforward(hidden) for n in self.output_layer])
        return hidden, outputs

    def predict(self, inputs):
        _, out = self.feedforward(inputs)
        return int(np.argmax(out))

    def predict_value(self, inputs):
        _, out = self.feedforward(inputs)
        return float(out[0])


# ============ ЗАГРУЗКА МОДЕЛЕЙ ИЗ JSON ============
def load_network(path):
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)

    arch = saved["architecture"]   # [5, 8, 3] или [5, 8, 1]
    activation = "sigmoid" if arch[2] > 1 else "linear"

    net = SimpleNeuralNetwork(arch[0], arch[1], arch[2], output_activation=activation)

    for neuron, params in zip(net.hidden_layer, saved["hidden"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]
    for neuron, params in zip(net.output_layer, saved["output"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]

    return net, saved


net_cls, meta_cls = load_network("classification_weights.json")
net_reg, meta_reg = load_network("regression_weights.json")

# Общие параметры нормализации
MN = np.array(meta_cls["mn"])
MX = np.array(meta_cls["mx"])
CLASS_NAMES = meta_cls["classes"]

# Параметры нормализации массы
MASS_MN = meta_reg["mass_mn"]
MASS_MX = meta_reg["mass_mx"]

print("Модели загружены.")
print("Классы:", CLASS_NAMES)


# ============ FASTAPI ============
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/classify", response_class=HTMLResponse)
def classify_form(request: Request):
    return templates.TemplateResponse(request, "classify.html",
                                      {"result": None, "form": {}})


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

    x = np.array([bill_length, bill_depth, flipper_length, is_male, is_female])
    x_norm = (x - MN) / (MX - MN)

    _, probs = net_cls.feedforward(x_norm)
    idx = int(np.argmax(probs))
    result = {
        "species": CLASS_NAMES[idx],
        "probs": {CLASS_NAMES[i]: round(float(probs[i]), 3) for i in range(len(CLASS_NAMES))},
    }
    form = {"bill_length": bill_length, "bill_depth": bill_depth,
            "flipper_length": flipper_length, "sex": sex}
    return templates.TemplateResponse(request, "classify.html",
                                      {"result": result, "form": form})


@app.get("/regress", response_class=HTMLResponse)
def regress_form(request: Request):
    return templates.TemplateResponse(request, "regress.html",
                                      {"result": None, "form": {}})


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

    x = np.array([bill_length, bill_depth, flipper_length, is_male, is_female])
    x_norm = (x - MN) / (MX - MN)

    mass_norm = net_reg.predict_value(x_norm)
    mass_g = mass_norm * (MASS_MX - MASS_MN) + MASS_MN

    result = {"mass_g": round(float(mass_g), 1)}
    form = {"bill_length": bill_length, "bill_depth": bill_depth,
            "flipper_length": flipper_length, "sex": sex}
    return templates.TemplateResponse(request, "regress.html",
                                      {"result": result, "form": form})