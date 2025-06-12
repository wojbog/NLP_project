import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


class TextClassifier:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir)
        except Exception:
            import logging
            logging.exception("Failed to load model from %s", model_dir)
            raise

        self.model.eval()

        model_path = os.path.join(
            model_dir, "bert_base_uncased_fine_tuend.onnx")
        self.session = ort.InferenceSession(model_path)

        self.input_name = self.session.get_inputs()[0].name

    def _predict_logits(self, text):
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='np',
            return_token_type_ids=True
        )

        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64),
            'token_type_ids': inputs['token_type_ids'].astype(np.int64)
        }

        outputs = self.session.run(None, ort_inputs)

        logits = outputs[0]
        return logits

    def predict_simple(self, text):
        logits = self._predict_logits(text)
        predicted_class = np.argmax(logits, axis=1)[0]

        return self.id2label(predicted_class)

    def predict_distribution(self, text):
        logits = self._predict_logits(text)
        probabilities = np.exp(logits) / np.sum(
            np.exp(logits), axis=1, keepdims=True)

        return {
            self.id2label(i): prob for
            i, prob in enumerate(probabilities[0])}

    def predict(self, text) -> dict:
        distribution = self.predict_distribution(text)
        predicted_class = max(distribution, key=distribution.get)

        return {
                "predicted_class": predicted_class,
                "distribution": distribution
            }

    def compute_importance(self, text):
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        if 'input_ids' not in inputs or inputs['input_ids'] is None:
            raise ValueError("Tokenizer failed to return input_ids.")

        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        token_type_ids = inputs.get('token_type_ids')

        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.retain_grad()

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = outputs.logits
        target = logits.argmax(dim=1)

        logits[0, target].backward()

        grads = embeddings.grad[0]
        importances = (grads * embeddings[0]).sum(dim=1).detach().cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return list(zip(tokens, importances))

    @staticmethod
    def id2label(id):
        labels = {
                0: "Futurama",
                1: "DoctorWho",
                2: "XFiles",
                3: "Stargate",
                4: "StarTrek",
                5: "Farscape",
                6: "Babylon5",
                7: "StarWarsRebels",
                8: "Fringe",
                9: "DoctorWhoSpinoffs",
                10: "StarWarsBooks",
                }
        return labels.get(id, "Unknown")


global_model = None


def get_model():
    global global_model
    if global_model is None:
        model_dir = "models"
        global_model = TextClassifier(model_dir)
    return global_model


if __name__ == "__main__":
    model = get_model()
    text = "Doctor who is a popular science fiction television series."

    print(model.predict_simple(text))
    print(model.predict_distribution(text))

    logits = model._predict_logits(text)
    print(logits)
