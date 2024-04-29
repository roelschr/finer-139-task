# finer-139-task

Task: Build a Named Entity Recognition Classifier
Dataset: FiNER
Model: DistilBERT

## Get started

gradio

## Follow through

Setup your environment using [poetry](https://python-poetry.org/) and Python 3.10.

```shell
# poetry config virtualenvs.in-project true
poetry install
poetry shell
```

Then go through our EDA notebook (feel free to press that Run All button).

Next, it's time to fine-tune the model using our new dataset. I chose to not do that with a notebook, because that's not how I would move to production. Instead, training and serving would be packaged and deployed for orchestration (most probably with Docker). 

```shell
python src/train.py
```

I've trained the model on a RTX3090 (24GB VRAM). Feel free to use mlflow and check the training metrics:

```shell
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
# then navigate to http://127.0.0.1:8080/#/experiments/0/runs/95cedfd54135453880dfd277799058f9/model-metrics
```

The model looks fairly good, with weighted f1 score of 93% in the test set. A better evaluation would require better business understanding in order to judge recall and precision per class in terms of cost/revenue.

Here is the confusion matrix for the following classes:
```json
{
    0: "O",
    1: "B-DebtInstrumentBasisSpreadOnVariableRate1",
    2: "B-DebtInstrumentInterestRateStatedPercentage",
    3: "B-LineOfCreditFacilityMaximumBorrowingCapacity",
    4: "B-DebtInstrumentInterestRateEffectivePercentage",
}
```

![Confusion matrix](cm.jpeg)

And this is final classification report.

|                                                  | Precision | Recall | F1-Score | Support |
|--------------------------------------------------|-----------|--------|----------|---------|
| DebtInstrumentBasisSpreadOnVariableRate1         | 0.92      | 0.94   | 0.93     | 1198    |
| DebtInstrumentInterestRateEffectivePercentage   | 0.80      | 0.84   | 0.82     | 264     |
| DebtInstrumentInterestRateStatedPercentage      | 0.93      | 0.94   | 0.94     | 1331    |
| LineOfCreditFacilityMaximumBorrowingCapacity    | 0.92      | 0.95   | 0.93     | 1235    |
||||||
| **micro avg**                                   | 0.91      | 0.94   | 0.93     | 4028    |
| **macro avg**                                   | 0.89      | 0.92   | 0.91     | 4028    |
| **weighted avg**                                | 0.92      | 0.94   | 0.93     | 4028    |

Finally, the model is converted to onnx. Because no quantization or optimization is applied, the model performance is expected to be the same.

