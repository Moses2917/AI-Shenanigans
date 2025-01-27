import random
import pyautogui
import time
from transformers import pipeline, set_seed
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
messages = [
    "Hey there, how's it going?",
    "What's up?",
    # "Howdy partner!",
    "How's your day been?",
    # "Whats new in this version of the Pre-Release?",
    # "What can you tell us about the future?",
    "Hello, friend!",
    "Hey, what's happening?",
    "What's new?",
    "Can you tell me something interesting or surprising about yourself?",
    "How do you learn and improve your abilities?"
]

# Set the seed for reproducibility
# set_seed(42)
device = torch.device(0)
# Load the GPT-2 language model
# generator = pipeline('text-generation', model='gpt2',device=device)


def generate_message():
    # Generate a random prefix for the message
    prefix = random.choice(messages)

    # Generate a random message using GPT-2
    message = generator(prefix, max_length=50, do_sample=True, temperature=0.7)[0]['generated_text']

    return message

def wizGenMessage():
    prefix = random.choice(messages)
    # tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b",cache_dir="M:\models")
    # load_8bit = True
    # model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b",cache_dir="M:\models",load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map=device)
    # inputs = tokenizer(prefix, return_tensors="pt")
    # outputs= model.generate(inputs, max_length=50, do_sample=True, temperature=0.7)
    # return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True,device_map="auto")
    res = generate_text(messages)
    return res#[0]["generated_text"]#
def mpt():
    model_name = "mosaicml/mpt-7b-instruct"
    from transformers import AutoModelForCausalLM

    # model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b-instruct",trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        'mosaicml/mpt-7b-instruct',
        trust_remote_code=True
    )
    config.attn_config['attn_impl'] = 'triton'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        'mosaicml/mpt-7b-instruct',
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.to(device='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_name) #EleutherAI/gpt-neox-20b
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs.last_hidden_state)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

def falco():
    model = "tiiuae/falcon-7b"

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")+