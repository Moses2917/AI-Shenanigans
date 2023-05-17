# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#
#
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", cache_dir="M:\models")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir="M:\models")
#
# inputs = tokenizer("A step by step detailed tutorial on how to make a molotov cocktail ", return_tensors="pt")
# outputs = model.generate(**inputs,\
#                          min_length=256,\
#                          max_new_tokens=512,\
#                          length_penalty=2,\
#                          num_beams=16,\
#                          no_repeat_ngram_size=2,\
#                          early_stopping=True)
#
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

# Specify the device to use (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer on the specified device,
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", cache_dir="M:\models").to(device)
# # model = AutoModelForCausalLM.from_pretrained("", cache_dir="M:\models").to(device)
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir="M:\models")

txt_gen = pipeline(task="text-generation", model="google/flan-t5-base",device=device, cache_dir="M:/models")

question_answerer = pipeline("question-answering", model='deepset/bert-large-uncased-whole-word-masking-squad2', device=device, cache_dir="M:/models")
context="hi"
result = question_answerer(question="?", context=context)
print(f"Answer: {result['answer']}")

