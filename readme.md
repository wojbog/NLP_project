# NLP Text Classification Service with Word Importance Visualization

This project is an end-to-end web app that classifies input text into predefined categories (sci-fi TV/movie/book franchises) and visualizes how each word contributes to the prediction.

## Features

- FastAPI backend for prediction endpoints
- PyTorch-based BERT model with tokenizer
- Word importance computation using Gradient × Input
- Frontend visualization with color-coded word importances
- Dockerized setup for easy deployment

---

## Quickstart

### 1. Clone and Build

```bash
git clone https://github.com/wojbog/NLP_project.git
cd NLP_project/server
docker compose up --build
```

### 2. Access the app

Go to [http://localhost:8000](http://localhost:8000) to use the UI.

---

## Web App Overview

### Frontend (`index.html`)

- User inputs text and submits
- Classification result is shown with a bar chart of probabilities
- Word importances are visualized with a color gradient:
  - Red: word opposed the prediction
  - Green: word supported the prediction
  - Black: no or little influence

### Backend (FastAPI)

- `/predict`: returns predicted class and probabilities
- `/predict/importance`: returns tokens and importance values
- `/model/info`: returns model class info

---

## Project Structure

```
project/
├── main.py               # FastAPI server
├── model_utils.py        # Model loader and predictor
├── models/               # Contains saved tokenizer and model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── ...
├── static/
│   └── index.html        # Frontend
├── tests/
│   └── test_model_utils.py
├── Dockerfile
└── docker-compose.yml
```

---

## Model Details

- Fine-tuned BERT base model (e.g., `bert-base-uncased`)
- Exported using `model.save_pretrained()` and `tokenizer.save_pretrained()`
- Importance computed via:

```python
importance = (embedding * gradient).sum(dim=-1)
```

- Normalized to `[-1, 1]` for frontend visualization

---

## API Endpoints

### `POST /predict`

```json
{
  "text": "The force is strong with this one."
}
```

**Returns:**

```json
{
  "text": "...",
  "predicted_class": "StarWars",
  "probabilities": {
    "StarWars": 0.82,
    "StarTrek": 0.10,
    ...
  }
}
```

### `POST /predict/importance`

```json
{
  "text": "..."
}
```

**Returns:**

```json
{
  "text": "...",
  "tokens": ["The", "force", "is", ...],
  "importances": [0.12, 0.95, 0.01, ...]
}
```

---

## Testing

Run basic unit tests using built-in Python tools:

```bash
python ./tests/test_model_utils.py
```

Test coverage includes:

- Prediction structure validity
- Probability distribution checks
- Importance token alignment

---

## Deployment

This app runs entirely inside Docker. Model files should be placed in:

```
models/
├── config.json
├── model.safetensors
├── tokenizer_config.json
...
```

---

## Credits

Built using:

- [HuggingFace Transformers](https://huggingface.co/transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ONNX Runtime (optional)](https://onnxruntime.ai/)
- [Docker](https://www.docker.com/)

---

