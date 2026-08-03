from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


max_seq_length = 2048
strict_instruction = (
    "Converti este texto OCR de una factura ARCA en un unico objeto JSON valido. "
    "No inventes datos: si falta un dato usa null; para iva, tributos e items usa array vacio. "
    "No agregues texto antes o despues del JSON. "
    "Usa exactamente el schema ARCA con estas claves raiz: tipo_comprobante, codigo_comprobante, "
    "punto_venta, numero_comprobante, numero_factura, fecha_emision, emisor, receptor, moneda, "
    "tipo_cambio, subtotal, importe_no_gravado, importe_exento, iva_total, tributos_total, "
    "impuestos, total, cae, fecha_vencimiento_cae, iva, tributos, items. "
    "emisor y receptor deben tener: nombre, doc_tipo, doc_nro, cuit, condicion_iva. "
    "Normaliza valores: fechas YYYY-MM-DD, CUIT con guiones en cuit, doc_nro sin guiones, "
    "punto_venta de 5 digitos, numero_comprobante de 8 digitos, numero_factura como 00000-00000000. "
    "Usa moneda ARCA: PES para pesos argentinos y DOL para dolares. "
    "No incluyas etiquetas OCR como CUIT:, Cliente:, Comp. Nro: dentro de los valores."
)

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
    text = f"""### Instruccion:
{strict_instruction}

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
        warmup_steps=5,
        max_steps=80,
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
