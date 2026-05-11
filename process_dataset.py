import json
from transformers import AutoTokenizer
import os


def create_data(input_len, batch_size, model_path, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if os.path.exists(f'GSM8K-in{input_len}-bs{batch_size}.jsonl'):
        print("dataset already exists...")
        exit(0)

    if not os.path.exists('./GSM8K.jsonl'):
        print("gsm8k dataset not exists...")
        exit(0)

    dataset = []
    with open('./GSM8K.jsonl', 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data['question'])

    # repeat input_len
    dataset_2k = []
    for sentence in dataset:
        words = tokenizer.tokenize(sentence)
        if len(words) == 0:
            continue
        len_num = len(words) // input_len
        if len_num == 0:
            multiplier = (input_len // len(words)) + 1
            repeated_len = words * multiplier
            words = repeated_len[:input_len]
            decoded_text = tokenizer.convert_tokens_to_string(words)
            dataset_2k.append(decoded_text)

    # repeat to batch_size
    batch_num = len(dataset_2k) // batch_size
    if batch_num == 0:
        multiplier = (batch_size // len(dataset_2k)) + 1
        repeated_batch = dataset_2k * multiplier
        dataset_2k = repeated_batch[:batch_size]
    else:
        dataset_2k = dataset_2k[:batch_size]

    json_str = json.dumps(dataset_2k, ensure_ascii=False, indent=4)
    base_name = os.path.basename(os.path.normpath(model_path))
    with open(os.path.join(save_path, f'GSM8K-in{input_len}-bs{batch_size}-{base_name}.jsonl'), 'w', encoding='utf-8') as f:
        for i in range(len(dataset_2k)):
            f.write(json.dumps({"question": dataset_2k[i], "answer": "none"}, ensure_ascii=False))
            f.write("\n")
