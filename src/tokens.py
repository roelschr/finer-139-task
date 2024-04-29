from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")


def tokenize_and_align_labels(samples):
    """
    Tokenize the text and align the labels with the tokens.

    This function processes each example by tokenizing the text and ensuring that
    the labels align with the tokens produced by the tokenizer. It accounts for the
    possibility that a tokenizer may split a token into multiple sub-tokens. Only the
    first sub-token of a split token retains the original label, while the subsequent
    sub-tokens are assigned a label of -100, indicating that they should be ignored
    in the loss calculation during model training.

    Parameters:
    - samples (dict): A dictionary containing two keys: 'tokens' and 'ner_tags'. The
      'tokens' key holds a list of words, and the 'ner_tags' key contains the corresponding
      list of entity labels for those words.

    Returns:
    - dict: A dictionary with keys corresponding to the outputs from the tokenizer (such as
      'input_ids', 'attention_mask', etc.) and an additional key 'labels' which contains
      the adjusted labels that align with the tokenizer's output tokens.
    """
    tokenized_inputs = tokenizer(
        samples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=True,
        max_length=128,
    )
    labels = []
    for i, label in enumerate(samples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
