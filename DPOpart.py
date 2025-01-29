"""
The DPOTrainer class is a custom subclass of Hugging Face's Trainer designed specifically for Direct Preference Optimization (DPO). 
It overrides the compute_loss method to use the dpo_loss function, which calculates the loss difference between a preferred (chosen) response and an alternative (rejected) response. 
This approach fine-tunes a language model to align with human preferences by encouraging the model to generate responses closer to the chosen examples while discouraging rejected ones.
"""
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
train_data_path = "/content/drive/MyDrive/SFTmodel/generated_dataTripAdvisor.json"
val_data_path = "/content/drive/MyDrive/SFTmodel/generated_Val_dataTripAdvisor.json"
huggingface_model_path = "/content/drive/MyDrive/SFTmodel/huggingface_model"
best_model_path = "/content/drive/MyDrive/SFTmodel/best_model"

# Step 1: Load Pre-trained GPT-2 Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Save the model and tokenizer to Hugging Face format
model.save_pretrained(huggingface_model_path)
tokenizer.save_pretrained(huggingface_model_path)

# Step 2: Load Training and Validation Datasets
train_dataset = load_dataset("json", data_files={"train": train_data_path})["train"]
val_dataset = load_dataset("json", data_files={"validation": val_data_path})["validation"]

# Step 3: Tokenize the Dataset
def preprocess_function(examples):
    """
    Tokenizes input examples for a language model, processing both "chosen" and optionally "rejected" responses.

    Args:
        examples (dict): A dictionary containing dataset examples. 
                         Must include a "chosen" key and optionally a "rejected" key.

    Returns:
        dict: A dictionary with the following tokenized outputs:
            - "input_ids" (list of int): Tokenized input IDs for the "chosen" response.
            - "attention_mask" (list of int): Attention mask for the "chosen" response.
            - "rejected_input_ids" (list of int, optional): Tokenized input IDs for the "rejected" response (if provided).
            - "rejected_attention_mask" (list of int, optional): Attention mask for the "rejected" response (if provided).

    Raises:
        ValueError: If the "chosen" key is missing from the input examples.

    Example:
        >>> examples = {
        ...     "chosen": "This is the preferred response.",
        ...     "rejected": "This is an alternative response."
        ... }
        >>> preprocess_function(examples)
        {
            "input_ids": [...],
            "attention_mask": [...],
            "rejected_input_ids": [...],
            "rejected_attention_mask": [...]
        }
    """
    if "chosen" in examples:
        chosen = tokenizer(
            examples["chosen"], truncation=True, padding="max_length", max_length=512
        )
    else:
        raise ValueError("Dataset examples must include a 'chosen' field.")

    if "rejected" in examples and examples["rejected"]:
        rejected = tokenizer(
            examples["rejected"], truncation=True, padding="max_length", max_length=512
        )
        return {
            "input_ids": chosen["input_ids"],
            "attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }
    else:
        return {
            "input_ids": chosen["input_ids"],
            "attention_mask": chosen["attention_mask"],
        }
# Apply preprocessing to both train and validation datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
def dpo_loss(model, inputs):
    """
    Computes the Direct Preference Optimization (DPO) loss for a given model.

    This function calculates the loss for a "chosen" response and, if available, 
    a "rejected" response, returning the difference between them to encourage 
    preference learning.

    Args:
        model (torch.nn.Module): The language model used for computing the loss.
        inputs (dict): A dictionary containing tokenized inputs with the following keys:
            - "input_ids" (torch.Tensor): Tokenized input IDs for the chosen response.
            - "attention_mask" (torch.Tensor): Attention mask for the chosen response.
            - "rejected_input_ids" (torch.Tensor, optional): Tokenized input IDs for the rejected response.
            - "rejected_attention_mask" (torch.Tensor, optional): Attention mask for the rejected response.

    Returns:
        torch.Tensor: The computed loss. If "rejected_input_ids" is provided, 
                      the function returns the difference between the chosen loss 
                      and the rejected loss. Otherwise, it returns only the chosen loss.

    Example:
        >>> inputs = {
        ...     "input_ids": torch.tensor([[1, 2, 3]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1]]),
        ...     "rejected_input_ids": torch.tensor([[4, 5, 6]]),
        ...     "rejected_attention_mask": torch.tensor([[1, 1, 1]])
        ... }
        >>> loss = dpo_loss(model, inputs)
        >>> print(loss)
    """
    # Move inputs to the GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute loss for the chosen response
    chosen_outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"]
    )

    # Compute loss for the rejected response if available
    if "rejected_input_ids" in inputs:
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            labels=inputs["rejected_input_ids"]
        )
        return chosen_outputs.loss - rejected_outputs.loss
    else:
        return chosen_outputs.loss
# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/SFTmodel/dpo-finetuned-model",  # Save to Google Drive
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",
    push_to_hub=False,
    fp16=True,  # Enable mixed precision training for better GPU performance
)
# Step 6: Custom Trainer for Direct Preference Optimization (DPO)
class DPOTrainer(Trainer):
    """
    A custom Trainer class for Direct Preference Optimization (DPO).

    This subclass overrides the `compute_loss` method to use the `dpo_loss` function,
    which computes the loss based on the difference between the chosen and rejected responses.

    Args:
        model (torch.nn.Module): The model to be trained.
        inputs (dict): Tokenized inputs for training, including "chosen" and optionally "rejected" responses.
        return_outputs (bool, optional): Whether to return model outputs along with the loss. Default is False.
        num_items_in_batch (int, optional): Placeholder argument, not used in this implementation.

    Returns:
        torch.Tensor or tuple: The computed loss, and optionally the model outputs if `return_outputs` is True.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = dpo_loss(model, inputs)
        return (loss, None) if return_outputs else loss

# Initialize the trainer with training arguments and datasets
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Fine-Tuned Model to Google Drive
output_dir = "/content/drive/MyDrive/SFTmodel/dpo-finetuned-model"

# Save the model and tokenizer
model.save_pretrained(output_dir)  
tokenizer.save_pretrained(output_dir)
