# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets
from transformers import AutoTokenizer, WhisperForConditionalGeneration


def whisperRecog(audioFile,transcript):
  import torch
  from transformers import pipeline, WhisperProcessor, AutoProcessor, AutoModelForSpeechSeq2Seq
  from pydub import AudioSegment

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  # audio = AudioSegment.from_file(audioFile, format="mp3")
  # audio.export("temp.wav", format="wav")
  # with open("temp.wav", "rb") as f:
  #   wav_data = f.read()


  processor = AutoProcessor.from_pretrained("openai/whisper-small",language="Armenian", task="transcribe")
  mode = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
  forced_decoder_ids = processor.get_decoder_prompt_ids(language="armenian", task="transcribe")
  f = open(audioFile, "rb")
  transcript = open(transcript + ".txt", "w", encoding='utf-8')
  pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    chunk_length_s=30,
    device=device,
    # forced_decoder_ids=forced_decoder_ids
    # tokenizer=processor
  )
  prediction = pipe(audioFile)["text"]
  # input_features = processor(wav_data,sampling_rate=16_000, return_tensors="pt").input_features
  # predicted_ids = mode.generate(input_features, forced_decoder_ids=forced_decoder_ids)
  # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
  transcript.write(str("transcription") + prediction)
  f.close()



  # for chunk in prediction:
  #   transcript.write(str(chunk) + "\n")

def qaPipeline(ques):

  import torch
  from transformers import pipeline
  device = "cuda:0" if torch.cuda.is_available() else "cpu" # deepset/bert-large-uncased-whole-word-masking-squad2
  question_answerer = pipeline("question-answering", model='deepset/roberta-base-squad2', device=device, cache_dir="M:/models")
  # txt_gen = pipeline(task="text-generation", model="google/flan-t5-small",device=device, cache_dir="M:/models")

  content = open("bookContent.txt", "r", encoding='UTF-8')
  context = content.read()
  # context = "Your name is 'My name is crackGPT', my name is Movses, when asked about names answer in a complete sentence"
  exQues = "in the new era the large sectors of American business started doing, is it consolidation,simplification, or deregulation"
  exQues = "Bob and Sally are sitting together on a bench, sally is holding a gun to Bob's head"
  result = question_answerer(question=str(ques) + "?", context=context)
  print(f"Answer: {result['answer']}")
  # lResult = txt_gen("in the new era the large sectors of American business started")
  # print(lResult)
  ##Try training Flan-T5 on squad v2 for a LLM combined/trained on question answering

#M:/new downloads/videoplayback.mp3

whisperRecog(transcript="ch23lec", audioFile="M:/new downloads/lec.wav")

def whisp():
  import torch
  from transformers import WhisperTokenizer
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Armenian", task="transcribe")
# input("What is your question?\n")
# qaPipeline("how can users post copyrighted material on TikTok")
