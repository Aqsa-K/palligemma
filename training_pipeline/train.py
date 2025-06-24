from datasets import load_dataset
import json
from torch.utils.data import Dataset
from typing import Any, List, Dict
import random
import json
from transformers import AutoProcessor
from torch.utils.data import DataLoader
import numpy as np
import yaml


from transformers import AutoProcessor
from transformers import PaliGemmaProcessor
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader

import lightning as L
import torch
from torch.utils.data import DataLoader
import re
from nltk import edit_distance

from transformers import PaliGemmaForConditionalGeneration

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from huggingface_hub import HfApi

from lightning.pytorch.loggers import WandbLogger
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



with open("config.yaml", "r") as f:
        config_yaml = yaml.safe_load(f)

# Load the dataset from Hugging Face Hub
HF_DATASET = config_yaml["HF_DATASET"]
REPO_ID = config_yaml["REPO_ID"]
FINETUNED_MODEL_ID = config_yaml["FINETUNED_MODEL_ID"]
MAX_LENGTH = config_yaml["MAX_LENGTH"]
WANDB_PROJECT = config_yaml["WANDB_PROJECT"]
WANDB_NAME = config_yaml["WANDB_NAME"]
PROMPT = "extract JSON."

dataset = load_dataset(HF_DATASET)

example = dataset['train'][0]
image = example["image"]
# resize image for smaller displaying
width, height = image.size
# image = image.resize((int(0.3*width), int(0.3*height)))
print(f"Image size: {image.size}")

ground_truth = json.loads(example["ground_truth"])
ground_truth["gt_parse"]


class CustomDataset(Dataset):
    """
    PyTorch Dataset. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = False,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # print("inside get item")

        # inputs
        image = sample["image"].convert("RGB")
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1

        return image, target_sequence
    

train_dataset = CustomDataset(HF_DATASET, split="train")
val_dataset = CustomDataset(HF_DATASET, split="validation")



processor = PaliGemmaProcessor.from_pretrained(REPO_ID)
# The number of image tokens for PaliGemma is available as image_seq_length
num_image_tokens = processor.image_seq_length
print(num_image_tokens)

# text_lengths = []
# for example in dataset["train"]:
#     toks = processor.tokenizer(example["ground_truth"], add_special_tokens=True)
#     text_lengths.append(len(toks["input_ids"]))

# print("Max text tokens:", max(text_lengths))
# print("95th pct text tokens:", np.percentile(text_lengths, 95))

def train_collate_fn(examples):
  images = [example[0] for example in examples]
  texts = [PROMPT for _ in range(len(images))]
  labels = [example[1] for example in examples]

  inputs = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding=True,
                     truncation="only_second", max_length=MAX_LENGTH,
                     tokenize_newline_separately=False)

  input_ids = inputs["input_ids"]
  token_type_ids = inputs["token_type_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]
  labels = inputs["labels"]

  return input_ids, token_type_ids, attention_mask, pixel_values, labels


def eval_collate_fn(examples):
  images = [example[0] for example in examples]
  # Convert PIL images to NumPy arrays with channel dimension
  # images = [np.array(image.convert("RGB")) for image in images]
  texts = [PROMPT for _ in range(len(images))]
  answers = [example[1] for example in examples]

  inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, tokenize_newline_separately=False)

  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]

  return input_ids, attention_mask, pixel_values, answers


train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=2, shuffle=True)
input_ids, token_type_ids, attention_mask, pixel_values, labels = next(iter(train_dataloader))

print(processor.batch_decode(input_ids))

for id, label in zip(input_ids[0][-30:], labels[0][-30:]):
  print(processor.decode([id.item()]), processor.decode([label.item()]))


val_dataloader = DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=2, shuffle=False)
input_ids, attention_mask, pixel_values, answers = next(iter(val_dataloader))

print(processor.batch_decode(input_ids))

class PaliGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                pixel_values=pixel_values,
                                labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance",
                  np.mean(scores),
                  on_step=False,
                  on_epoch=True,
                  prog_bar=True,
                  logger=True,
                  add_dataloader_idx=False
                 )

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=0)
    

device = "cuda"
model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID, torch_dtype=torch.bfloat16).to(device)

# for param in model.vision_tower.parameters():
#     param.requires_grad = False

# for param in model.multi_modal_projector.parameters():
#     param.requires_grad = False


config = {"max_epochs": 2,
          "val_check_interval": 0.5, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
        #   "accumulate_grad_batches": 4,
          "lr": 1e-4,
          "batch_size": 1,
          # "seed":2022,
          "num_nodes": 1,
          # "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
}

model_module = PaliGemmaModelPLModule(config, processor, model)
model_module.model.gradient_checkpointing_enable()

api = HfApi()

hf_model_name = "AqsaK/paligemma_finetuned_census_data_subset"

class ShowFewSamples(Callback):
    """
    Print at most `num_samples` examples every `every_n_epochs` epochs.
    Works even if validation_step only returns scalar metrics.
    """

    def __init__(self, every_n_epochs: int = 1, num_samples: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_start(self, trainer, pl_module):
        # Should we show samples this epoch?
        pl_module._show_examples = trainer.current_epoch % self.every_n_epochs == 0
        pl_module._examples_left = self.num_samples if pl_module._show_examples else 0



class ToggleVerbose(Callback):
    """
    Keep pl_module.config["verbose"] == True for the first *n* validation epochs,
    then switch it to False so printing stops.
    """
    def __init__(self, off_after_epoch: int = 0):  # 0 → only epoch-0 is verbose
        super().__init__()
        self.off_after_epoch = off_after_epoch

    def on_validation_epoch_start(self, trainer, pl_module):
        # True while current_epoch <= off_after_epoch, then False
        pl_module.config["verbose"] = trainer.current_epoch <= self.off_after_epoch


class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(hf_model_name,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")
        
        pl_module.processor.push_to_hub(hf_model_name,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")   

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(hf_model_name,
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub(hf_model_name,
                                    commit_message=f"Training done")


class PushOnCheckpoint(Callback):
    def __init__(self, checkpoint_callback: ModelCheckpoint, hf_model_name: str):
        super().__init__()
        self.ckpt_cb = checkpoint_callback
        self.hf_model_name = hf_model_name
        self._last_ckpt = None
        # Ensure the repository exists before pushing
        api.create_repo(repo_id=self.hf_model_name, exist_ok=True)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        new_ckpt = self.ckpt_cb.last_model_path
        if new_ckpt and new_ckpt != self._last_ckpt:
            self._last_ckpt = new_ckpt
            print(f"➡️ Pushing new checkpoint to Hub: {os.path.basename(new_ckpt)}")

            pl_module.model.push_to_hub(
                repo_id=self.hf_model_name,
                commit_message=f"Checkpoint {os.path.basename(new_ckpt)}"
            )
            # Push processor/config
            pl_module.processor.push_to_hub(
                repo_id=self.hf_model_name,
                commit_message=f"Checkpoint {os.path.basename(new_ckpt)}"
            )


class StepLogger(Callback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.log_every_n_steps == 0:
            print(f">>> Completed global step {step}")

# then add to your Trainer
step_logger = StepLogger(log_every_n_steps=100)

train_loader = model_module.train_dataloader()
steps_per_epoch = len(train_loader)

# Quarter of one epoch (must be an integer)
quarter_steps = max(1, steps_per_epoch // 4)
        
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",                      # where to save
    filename="donut-epoch{epoch:02d}-step{step}",# naming convention
    save_top_k=-1,                               # if you want *all* quarter‐epoch checkpoints; set -1 to keep every save
    every_n_train_steps=quarter_steps,           # save every quarter‐epoch (in # of steps)
    verbose=True,
)

# early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=5, verbose=True, mode="min")

wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        # accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        val_check_interval=config["val_check_interval"],
        precision="bf16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        # logger=wandb_logger,
        callbacks=[PushToHubCallback(), step_logger, checkpoint_callback, ShowFewSamples(every_n_epochs=1, num_samples=2)],
        enable_progress_bar=True
)

trainer.fit(model_module)
