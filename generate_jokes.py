from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2-jokes')
tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-gpt2-jokes')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

prompt = "Here's a joke:"
generated_jokes = generator(prompt, max_length=50, num_return_sequences=5, truncation=True)

for i, joke in enumerate(generated_jokes):
    print(f"Joke {i+1}: {joke['generated_text']}\n")
