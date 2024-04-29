import gradio as gr
from optimum.pipelines import pipeline
from datasets import load_dataset
from src import data
import random

ner_pipeline = pipeline(task="token-classification", model="model/finer-debt-distilbert-cased/onnx", accelerator="ort")

dataset = data.process(load_dataset("nlpaueb/finer-139")["test"])
random.seed(42)
random_samples = random.choices(range(100), k=5)
print(random_samples)
examples = [" ".join(dataset[i]["tokens"]) for i in random_samples]
print(examples)


def ner(text):
    output = ner_pipeline(text)
    return {"text": text, "entities": output}


demo = gr.Interface(ner, gr.Textbox(placeholder="Enter sentence here..."), gr.HighlightedText(), examples=examples)

demo.launch()
