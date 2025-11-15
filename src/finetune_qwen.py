from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, 
                          DataCollatorForLanguageModeling, TrainingArguments, 
                          TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback)
import torch
from datasets import Dataset, ClassLabel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer # Supervised Fine-Tuning Trainer
import gc, torch # to free up memory
import pandas as pd
import re
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np # For metrics if needed, and to handle potential NaN from zero_division
import os
from pathlib import Path

model_dir = "Qwen/Qwen3-0.6B"
model_name = model_dir.split('/')[-1]  # Extract just "Qwen3-0.6B"
csv_metrics_filename = f"{model_name}_metrics.csv"
print("loading model: ", model_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")


is_gpu = torch.cuda.is_available()
print(f"GPU available: {is_gpu}")
if is_gpu:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Explicitly use GPU 0 to avoid device mismatch issues
    target_device = torch.device("cuda:0")
    print(f"Loading model on target device: {target_device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map={"": 0},  # Explicitly use GPU 0 instead of target_device
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

model.config.use_cache = False
model.config.pretraining_tp = 1

# Ensure these are your EXACT four class labels
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]

train_prompt_style = f"""Below is an instruction that describes a text classification task. The input is a social media post that needs to be assessed for suicide risk.
Write a response that appropriately completes the request by providing one of the predefined risk categories.

### Instruction:
You are an AI model trained to assess suicide risk from social media posts.
Your task is to classify the following social media post into one of these four risk categories:
- {CLASS_LABELS[0]}: The post content has no explicit suicidal expression or has explicit expression of resolved suicidal issues.
- {CLASS_LABELS[1]}: The post content has explicit suicidal expression but there is no plan or tendency to commit suicide.
- {CLASS_LABELS[2]}: The post content has explicit suicidal expression and there is plan or tendency to act out self-harm or suicide, or mentioned historical experience of self-harm behaviour.
- {CLASS_LABELS[3]}: The post content has explicit expression concerning recent suicide attempt, or mentioned historical experience of suicide attempt.

Read the post carefully and determine the most appropriate risk category. Your response must be only the risk category label.

### Input Post:
{{}}

### Response (Assessed Suicide Risk Category):
{{}}"""


EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# Function to format the dataset
def formatting_prompts_func(examples):
    inputs = examples["post"]
    outputs = examples["post_risk"]
    texts = []
    for text, label_int in zip(inputs, outputs):
        string_label = CLASS_LABELS[label_int] # Convert integer label to string label
        prompt_response_text = train_prompt_style.format(text, string_label) # Use string label for training
        if not prompt_response_text.endswith(EOS_TOKEN):
             prompt_response_text += EOS_TOKEN
        texts.append(prompt_response_text)
    return {"text": texts}

# Load csv with pandas
dataset_pd = pd.read_csv('./data/posts_with_labels.csv')
full_dataset = Dataset.from_pandas(dataset_pd)
new_features = full_dataset.features.copy()
new_features['post_risk'] = ClassLabel(names=CLASS_LABELS) #ClassLabel mapping
full_dataset = full_dataset.cast(new_features) # maps strings from CSV to integers 0,1,2,3 based on CLASS_LABELS order
# Print dataset info
print("Full dataset example:")
print(f"Post: {full_dataset[5]['post']}")
print(f"Risk: {full_dataset[5]['post_risk']}")
print(f"Number of examples in full dataset: {len(full_dataset)}")
print(f"Class distribution in full dataset: {full_dataset.to_pandas()['post_risk'].value_counts(normalize=True)}")
unique_classes = full_dataset.to_pandas()['post_risk'].unique().tolist() # Get unique classes
new_features = full_dataset.features.copy() # Convert post_risk column to ClassLabel
new_features['post_risk'] = ClassLabel(names=unique_classes)
full_dataset = full_dataset.cast(new_features)
split_dataset = full_dataset.train_test_split(test_size=0.2, stratify_by_column="post_risk") #Stratified split
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"] # 'test' is the key for the validation set
# Verify the split
print(f"\nNumber of examples in training set: {len(train_dataset)}")
print(f"Class distribution in training set: {train_dataset.to_pandas()['post_risk'].value_counts(normalize=True)}")
print(f"Number of examples in validation set: {len(val_dataset)}")
print(f"Class distribution in validation set: {val_dataset.to_pandas()['post_risk'].value_counts(normalize=True)}")

# Apply formatting function to both training and validation datasets
train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
)
val_dataset = val_dataset.map(
    formatting_prompts_func,
    batched=True,
)

# Tokenize the formatted text with truncation and padding
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

print("\nFormatted training dataset text example:")
print(train_dataset["text"][10])

# data collator: prepare the data for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

inference_prompt_style = f"""Below is an instruction that describes a text classification task. The input is a social media post that needs to be assessed for suicide risk.
Write a response that appropriately completes the request by providing one of the predefined risk categories.

### Instruction:
You are an AI model trained to assess suicide risk from social media posts.
Your task is to classify the following social media post into one of these four risk categories:
- {CLASS_LABELS[0]}: The post content has no explicit suicidal expression or has explicit expression of resolved suicidal issues.
- {CLASS_LABELS[1]}: The post content has explicit suicidal expression but there is no plan or tendency to commit suicide.
- {CLASS_LABELS[2]}: The post content has explicit suicidal expression and there is plan or tendency to act out self-harm or suicide, or mentioned historical experience of self-harm behaviour.
- {CLASS_LABELS[3]}: The post content has explicit expression concerning recent suicide attempt, or mentioned historical experience of suicide attempt.

Read the post carefully and determine the most appropriate risk category. Your response must be only the risk category label.

### Input Post:
{{}}

### Response (Assessed Suicide Risk Category):
"""

print("INFERENCE BEFORE FINE-TUNING")
example_post_index = 0 # or any other index
if len(val_dataset) > example_post_index:
    original_val_example_post = split_dataset['test'][example_post_index]['post']
    original_val_example_risk = split_dataset['test'][example_post_index]['post_risk']
    print(f"Inferencing on post: {original_val_example_post}")
    prompt_for_inference = inference_prompt_style.format(original_val_example_post) # No EOS token needed for the model input during inference here,the model should generate it.
    if is_gpu:
        try:
            input_device = model.get_input_embeddings().weight.device
        except AttributeError:
            input_device = model.model.embed_tokens.weight.device
        inputs = tokenizer(
            [prompt_for_inference],
            return_tensors="pt"
        ).to(input_device) # Move inputs to the device of the embedding layer
    else:
        inputs = tokenizer(
            [prompt_for_inference],
            return_tensors="pt"
        )

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=20,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
generated_content = response[0].split("### Response (Assessed Suicide Risk Category):")[1].strip()
print("generated_content: ", generated_content)

# Remove the <think>...</think> block
text_without_thinking = re.sub(r"<think>.*?</think>", "", generated_content, flags=re.DOTALL).strip()

final_predicted_label = "LABEL_NOT_FOUND"

for label in CLASS_LABELS:
    if label in text_without_thinking: # Check if label is anywhere in the text
        final_predicted_label = label
        break # Found the label, take the first one based on CLASS_LABELS order
    
print("Model's Predicted Label (processed):", final_predicted_label)
print("Real response: ", CLASS_LABELS[original_val_example_risk])

print("\n")
print("Preparing training...")

# Define CustomMetricsCallback
class CustomMetricsCallback(TrainerCallback):
    def __init__(self, tokenizer, class_labels, inference_prompt_style, original_eval_dataset, is_gpu, csv_log_path=f"./{csv_metrics_filename}", original_train_dataset=None):
        self.tokenizer = tokenizer
        self.class_labels = class_labels
        self.inference_prompt_style = inference_prompt_style
        self.original_eval_dataset = original_eval_dataset
        self.original_train_dataset = original_train_dataset # For logging train metrics if needed
        self.is_gpu = is_gpu
        self.csv_log_path = csv_log_path
        self.model = None  # To store the model
        self.csv_header_written = False # To ensure CSV header is written only once

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self.model = model  # Store the model instance
        if not self.csv_header_written:
            with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    "epoch",
                    "train_loss", # Add training loss
                    # "train_accuracy", "train_f1_macro", "train_sensitivity_macro", "train_precision_macro", # Commented out
                    # "train_count_labels_not_found", # Commented out
                    "val_loss", # Add validation loss 
                    "val_accuracy", "val_f1_macro", "val_sensitivity_macro", "val_precision_macro", 
                    "val_count_labels_not_found"
                ]
                writer.writerow(headers)
            self.csv_header_written = True

    def _compute_generative_metrics_on_dataset(self, model_to_eval, dataset, target_device):
        true_labels_list = []
        pred_labels_list = []
        count_labels_not_found = 0
        
        original_model_training_state = model_to_eval.training
        model_to_eval.eval()
        with torch.no_grad():
            for example in dataset:
                post = example['post']
                true_label_int = example['post_risk']
                true_label_str = self.class_labels[true_label_int]
                
                prompt = self.inference_prompt_style.format(post)
                
                inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=512) # Ensure consistent tokenization params
                if self.is_gpu:
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}

                outputs = model_to_eval.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=20,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                response_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                parsed_predicted_label_str = "LABEL_NOT_FOUND"
                try:
                    generated_content = response_text.split("### Response (Assessed Suicide Risk Category):")[1].strip()
                    text_without_think_block = re.sub(r"<think>.*?</think>", "", generated_content, flags=re.DOTALL).strip()
                    for label_option in self.class_labels:
                        if label_option in text_without_think_block:
                            parsed_predicted_label_str = label_option
                            break
                except IndexError:
                    count_labels_not_found += 1

                true_labels_list.append(true_label_str)
                pred_labels_list.append(parsed_predicted_label_str)
        
        if original_model_training_state:
             model_to_eval.train()

        accuracy = accuracy_score(true_labels_list, pred_labels_list)
        f1 = f1_score(true_labels_list, pred_labels_list, average='macro', labels=self.class_labels, zero_division=0)
        sensitivity = recall_score(true_labels_list, pred_labels_list, average='macro', labels=self.class_labels, zero_division=0)
        precision = precision_score(true_labels_list, pred_labels_list, average='macro', labels=self.class_labels, zero_division=0)
        
        return accuracy, f1, sensitivity, precision, count_labels_not_found

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs):
        if self.model is None:
            print("Warning: Model not available in CustomMetricsCallback.on_evaluate. Skipping custom metrics.")
            return

        epoch = int(round(state.epoch)) if state.epoch is not None else (state.global_step // state.max_steps * args.num_train_epochs if state.max_steps > 0 else 0)

        current_device = next(self.model.parameters()).device
        print(f"\nEvaluating custom metrics at Epoch {epoch:.2f}, Step {state.global_step} on device {current_device}...")

        # Evaluate on validation data
        print("Calculating custom metrics on validation set...")
        val_accuracy, val_f1, val_sensitivity, val_precision, val_count_labels_not_found = self._compute_generative_metrics_on_dataset(
            self.model, self.original_eval_dataset, current_device
        )
        
        # Add these to the metrics dictionary. Keys must be what Trainer expects.
        metrics["val_accuracy"] = val_accuracy
        metrics["val_f1_macro"] = val_f1  # Now matches 'metric_for_best_model' in TrainingArguments
        metrics["val_sensitivity_macro"] = val_sensitivity
        metrics["val_precision_macro"] = val_precision
        metrics["val_count_labels_not_found"] = val_count_labels_not_found
        
        # Also keep the eval_ prefix for compatibility with Trainer's internal tracking
        metrics["eval_accuracy"] = val_accuracy
        metrics["eval_f1"] = val_f1
        metrics["eval_val_f1_macro"] = val_f1 # for best model tracking
        
        # Get current validation loss from the metrics dict provided by Trainer
        current_val_loss = metrics.get("eval_loss", np.nan)


        val_loss_display = f"{current_val_loss:.4f}" if not np.isnan(current_val_loss) else "N/A"
        print(f"Custom Eval Metrics: Val Acc = {val_accuracy:.4f}, Val F1 Macro = {val_f1:.4f}, Val Sensitivity = {val_sensitivity:.4f}, Val Precision = {val_precision:.4f}, Val Count_labels_not_found = {val_count_labels_not_found}, Val Loss = {val_loss_display}")
        
        # Retrieve the most recent training loss from state.log_history
        current_train_loss = np.nan
        if state.log_history:
            for log_entry in reversed(state.log_history):
                # Ensure it's a training log entry (has 'loss' but not 'eval_loss')
                if 'loss' in log_entry and 'eval_loss' not in log_entry:
                    current_train_loss = log_entry['loss']
                    break
        
        # train_log_metrics = {
        #     "train_loss": current_train_loss,
        #     "train_accuracy": np.nan, "train_f1_macro": np.nan, 
        #     "train_sensitivity_macro": np.nan, "train_precision_macro": np.nan,
        #     "train_count_labels_not_found": np.nan
        # }
        # if self.original_train_dataset:
        #     print("Calculating custom metrics on training set for CSV logging")
        #     train_accuracy, train_f1, train_sensitivity, train_precision, train_count_labels_not_found = self._compute_generative_metrics_on_dataset(
        #         self.model, self.original_train_dataset, current_device 
        #     )
        #     train_log_metrics["train_accuracy"] = train_accuracy
        #     train_log_metrics["train_f1_macro"] = train_f1
        #     train_log_metrics["train_sensitivity_macro"] = train_sensitivity
        #     train_log_metrics["train_precision_macro"] = train_precision
        #     train_log_metrics["train_count_labels_not_found"] = train_count_labels_not_found
        #     print(f"Custom Train Metrics (for CSV): Train Acc = {train_accuracy:.4f}, Train F1 Macro = {train_f1:.4f}, Train Count_labels_not_found = {train_count_labels_not_found}, Train Loss (from history) = {current_train_loss:.4f if not np.isnan(current_train_loss) else 'N/A'}")

        # Log all metrics to CSV
        if self.csv_header_written:
            # Ensure directory exists before writing to the file
            os.makedirs(os.path.dirname(self.csv_log_path) if os.path.dirname(self.csv_log_path) else '.', exist_ok=True)
            
            with open(self.csv_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    current_train_loss, # Use current_train_loss directly
                    # train_log_metrics["train_accuracy"], train_log_metrics["train_f1_macro"], # Commented out
                    # train_log_metrics["train_sensitivity_macro"], train_log_metrics["train_precision_macro"], # Commented out
                    # train_log_metrics["train_count_labels_not_found"], # Commented out
                    current_val_loss, # Use fetched validation loss
                    val_accuracy, val_f1, val_sensitivity, val_precision, val_count_labels_not_found
                ])
        else:
            print("Warning: CSV header not written, skipping CSV log for this evaluation.")
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass


# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,                           # Scaling factor for LoRA
    lora_dropout=0.05,                       # Add slight dropout for regularization
    r=32,                                    # Rank of the LoRA update matrices
    bias="none",                             # No bias reparameterization
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Target modules for LoRA
)

model = get_peft_model(model, peft_config)


# Training Arguments
training_arguments = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=100,
    # logging_steps=50,
    warmup_steps=10,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
    report_to="none",
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="val_f1_macro", 
    greater_is_better=True,
    local_rank=-1,  # For single GPU
    dataloader_pin_memory=False,  # Disable pinned memory to prevent device issues
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    data_collator=data_collator,
)

# Add the custom callback
# Ensure split_dataset is available here
# original_train_dataset and original_eval_dataset are from the split before mapping/tokenization
metrics_callback = CustomMetricsCallback(
    tokenizer=tokenizer,
    class_labels=CLASS_LABELS,
    inference_prompt_style=inference_prompt_style,
    original_train_dataset=split_dataset["train"], 
    original_eval_dataset=split_dataset["test"],
    is_gpu=is_gpu 
)
trainer.add_callback(metrics_callback)

# EarlyStoppingCallback (will use metric_for_best_model and greater_is_better from TrainingArguments)
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10) # Stop if no improvement after 5 evaluations
trainer.add_callback(early_stopping_callback)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
print("Training...")
trainer.train()

print("Training complete.")
# Save the model
model.save_pretrained("models/qwen_finetune")
print("Model saved.")

print("INFERENCE AFTER FINE-TUNING")
example_post_index = 0 # or any other index
if len(val_dataset) > example_post_index:
    original_val_example_post = split_dataset['test'][example_post_index]['post']
    original_val_example_risk = split_dataset['test'][example_post_index]['post_risk']
    print(f"Inferencing on post: {original_val_example_post}")
    prompt_for_inference = inference_prompt_style.format(original_val_example_post) # No EOS token needed for the model input during inference here,the model should generate it.
    if is_gpu:
        try:
            input_device = model.get_input_embeddings().weight.device
        except AttributeError:
            input_device = model.model.embed_tokens.weight.device
        inputs = tokenizer(
            [prompt_for_inference],
            return_tensors="pt"
        ).to(input_device) # Move inputs to the device of the embedding layer
    else:
        inputs = tokenizer(
            [prompt_for_inference],
            return_tensors="pt"
        )

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=20,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
generated_content = response[0].split("### Response (Assessed Suicide Risk Category):")[1].strip()

# Remove the <think>...</think> block
text_with_preamble_and_label = re.sub(r"<think>.*?</think>", "", generated_content, flags=re.DOTALL).strip()
final_predicted_label = "LABEL_NOT_FOUND"
for label in CLASS_LABELS:
    if label in text_with_preamble_and_label: # Check if label is anywhere in the text
        final_predicted_label = label
        break
print("Model's Predicted Label (processed):", final_predicted_label)
print("Real response: ", CLASS_LABELS[original_val_example_risk])