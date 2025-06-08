import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

class TextClassifier:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        model_path = os.path.join(model_dir, "bert_base_uncased_fine_tuend.onnx")
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
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        return {self.id2label(i): prob for i, prob in enumerate(probabilities[0])}

    def predict(self, text) -> dict:
        distribution = self.predict_distribution(text)
        predicted_class = max(distribution, key=distribution.get)

        return {
                "predicted_class": predicted_class,
                "distribution": distribution
            }




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
