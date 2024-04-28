# finer-139-task

Task: Build a Named Entity Recognition Classifier
Dataset: FiNER
Model: DistilBERT

## Follow through

Setup your environment using [poetry](https://python-poetry.org/) and Python 3.10.

```shell
# poetry config virtualenvs.in-project true
poetry install
```

Then go through our EDA notebook (feel free to press that Run All button).

Next, it's time to fine-tune the model using our new dataset. I chose to not do that with a notebook, because that's not how I would move to production. 
Instead, training and serving would be packaged and deployed for orchestration. 

