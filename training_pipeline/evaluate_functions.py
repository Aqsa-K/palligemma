from nltk import edit_distance
import numpy as np

def evaluate_predictions(predictions, answers):
  scores = []
  for pred, answer in zip(predictions, answers):
      pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
      scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

      print(f"Prediction: {pred}")
      print(f"    Answer: {answer}")
      print(f" Normed ED: {scores[0]}")

  print(np.mean(scores))


test_dataset = CustomDataset(HF_DATASET, split="test")
test_dataloader = DataLoader(test_dataset, collate_fn=eval_collate_fn, batch_size=4, shuffle=False)



from transformers import PaliGemmaForConditionalGeneration

model = PaliGemmaForConditionalGeneration.from_pretrained(FINETUNED_MODEL_ID)

input_ids, attention_mask, pixel_values, answers = next(iter(test_dataloader))


generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
# turn them back into text, chopping of the prompt
# important: we don't skip special tokens here, because we want to see them in the output
predictions = processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)