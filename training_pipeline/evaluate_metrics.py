import os
import re
import json
import torch
import yaml
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, load_dataset_builder
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from difflib import SequenceMatcher

def load_config(path="config_eval.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def normalized_edit_distance(pred, gt):
    return 1 - SequenceMatcher(None, pred, gt).ratio()


# let's turn that into JSON
def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}
        

def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = PaliGemmaProcessor.from_pretrained(cfg["HF_MODEL_NAME"])
    model = PaliGemmaForConditionalGeneration.from_pretrained(cfg["HF_MODEL_NAME"]).to(device)

    builder = load_dataset_builder(cfg["DATASET_HF"])
    num_samples = builder.info.splits[cfg["EVALUATION_SPLIT"]].num_examples
    ds_stream = load_dataset(cfg["DATASET_HF"], split=cfg["EVALUATION_SPLIT"], streaming=True)

    results = []
    os.makedirs(cfg["EVALUATION_RESULT_PATH"], exist_ok=True)
    output_json = os.path.join(cfg["EVALUATION_RESULT_PATH"], "predictions.jsonl")
    metrics_path = os.path.join(cfg["EVALUATION_RESULT_PATH"], "metrics.json")

    for idx, sample in tqdm(enumerate(ds_stream), total=num_samples, desc="Evaluating"):
        image = sample["image"].convert("RGB")
        answer = sample["ground_truth"]  # customize this key as per your dataset

        inputs = processor(image, text=cfg["PROMPT_TOKEN"], return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=cfg["MAX_LENGTH"])
        
        num_prompt_tokens = (
            (generated_ids == model.config.image_token_index).sum().item()
            + len(processor.tokenizer.encode(cfg["PROMPT_TOKEN"])) + 2
        )
        decoded = processor.batch_decode(generated_ids[:, num_prompt_tokens:], skip_special_tokens=False)[0]

        pred_clean = re.sub(r"(?:(?<=>) | (?=</s_))", "", decoded)
        json_pred = token2json(pred_clean, processor)
        ed_score = normalized_edit_distance(pred_clean, answer)

        results.append({
            "id": idx,
            "prediction": json_pred,
            "raw_prediction": pred_clean,
            "ground_truth": answer,
            "edit_distance": ed_score
        })

        with open(output_json, "a") as f:
            f.write(json.dumps(results[-1]) + "\n")

    mean_ed = np.mean([r["edit_distance"] for r in results])
    with open(metrics_path, "w") as f:
        json.dump({"mean_normalized_edit_distance": mean_ed}, f, indent=2)

    print(f"✅ Evaluation complete. Metrics saved to `{metrics_path}`")


if __name__ == "__main__":
    config = load_config()
    evaluate(config)