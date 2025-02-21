from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-xl"
model_dir = "D:\PHD\Development\Python\Cardio_CAD\large_language_models\model"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_dir)
    interpreter = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    prompt = "Translate this to French: Hello, world!"
    output = interpreter(prompt)
    print(output) #Check the output
    generated_text = output[0]["generated_text"]
    print(generated_text)
    pass
except Exception as e:
    print(f"Error: {e}")