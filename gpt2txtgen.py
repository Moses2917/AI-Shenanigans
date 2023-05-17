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
    model_name = "OccamRazor/mpt-7b-storywriter-4bit-128g"
    from transformers import AutoModelForCausalLM

    # model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b-instruct",trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        'OccamRazor/mpt-7b-storywriter-4bit-128g',
        trust_remote_code=True
    )
    config.attn_config['attn_impl'] = 'triton'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        'OccamRazor/mpt-7b-storywriter-4bit-128g',
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.to(device='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_name) #EleutherAI/gpt-neox-20b
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs.last_hidden_state)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


count = 0
model_name = "mosaicml/mpt-7b-instruct"#"databricks/dolly-v2-3b"
# generate_text = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,device_map="auto")
mpt()
for x in range(int(time.time())): #int(time.time())
    # message = wizGenMessage()
    prompt = random.choice(messages)
    prompt = "One positive thing you would tell them, make it very inspirational and wise?"
    message = generate_text(prompt)[0]["generated_text"]
    # print(prompt + ": " + message)
    print(f"Prompt: {prompt} \nResponse Message: {message}".format(prompt,message))
    # if count > 5:
    #     time.sleep(120)
    #     count = 0
    # pyautogui.typewrite(message[:50],interval=0.009)
    # pyautogui.press('enter')
    # count += 1
    # time.sleep(random.uniform(60,70))