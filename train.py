from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
)

def format_example(example):
    text = f"""### Instrucción:
{example["instruction"]}

### Texto OCR:
{example["input"]}

### Respuesta:
{example["output"]}{tokenizer.eos_token}"""
    return {"text": text}

dataset = dataset.map(format_example)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=max_seq_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=20,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    ),
)

trainer.train()

model.save_pretrained("factura-qwen-lora")
tokenizer.save_pretrained("factura-qwen-lora")

print("Entrenamiento terminado.")
print("Modelo LoRA guardado en: factura-qwen-lora")