import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import langchain
from langchain import HuggingFaceHub
# tokenizer = AutoTokenizer.from_pretrained("KoboldAI/OPT-6.7B-Erebus")
# model = AutoModelForCausalLM.from_pretrained("KoboldAI/OPT-6.7B-Erebus",cache_dir="M:/models")
#
from transformers import pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"
generator = pipeline('text-generation', model='facebook/opt-1.3b',device=device)
question = "Hello, I'm am conscious and"
print(generator(question, do_sample=True, min_length=50, max_length=128))

# from langchain import PromptTemplate, HuggingFaceHub, LLMChain
# template = """Question: {question}
#
# Answer: Let's think step by step."""
#
# prompt = PromptTemplate(template=template, input_variables=["question"])
# llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id="KoboldAI/OPT-6.7B-Erebus", model_kwargs={"temperature":0.5, "max_length":64},huggingfacehub_api_token="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"))
#
# question = "Write me a sexy story about how a shy young man uses his secret ability to stop time to fuck and have sex with anyone"
#
# print(llm_chain.run(question))