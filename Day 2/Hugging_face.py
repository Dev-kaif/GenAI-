import os # Imports the 'os' module, which provides a way to interact with the operating system, like setting environment variables.
from transformers import AutoTokenizer # Imports AutoTokenizer, a class that automatically selects the correct tokenizer for a given pre-trained model.
from transformers import AutoModelForCausalLM # Imports AutoModelForCausalLM, a class that automatically selects the correct causal language model for a given pre-trained model.
from transformers import pipeline # Imports the 'pipeline' function, a high-level API for quickly using pre-trained models for various tasks.
import torch # Imports the PyTorch library, essential for tensor operations and deep learning, especially for specifying data types.
from dotenv import load_dotenv

# ---
# Hugging Face Authentication
# ---

Hugging_Face_Token = os.getenv("Hugging_Face")

# Sets an environment variable named "HF_TOKEN" with your Hugging Face API token.
# This token is often required to access certain models, especially private ones or those with stricter access policies.
# It ensures you're authenticated when downloading models and tokenizers from Hugging Face.
os.environ["HF_TOKEN"] = Hugging_Face_Token

# ---
# Model and Tokenizer Loading
# ---

# Defines the name of the pre-trained model we want to use.
# In this case, it's "google/gemma-3-1b-it", which is a specific version of Google's Gemma model.
model_name = "google/gemma-3-1b-it"

# Initializes the tokenizer.
# 'AutoTokenizer.from_pretrained(model_name)' automatically downloads and loads the appropriate tokenizer
# associated with the 'model_name'. A tokenizer converts text into numerical IDs that the model can understand.
our_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initializes the pre-trained causal language model.
# 'AutoModelForCausalLM.from_pretrained(model_name)' downloads and loads the model's architecture and weights.
# 'torch_dtype=torch.bfloat16' is an important optimization. It loads the model's weights in 'bfloat16'
# (brain float 16) precision instead of the default 'float32'. This significantly reduces memory usage
# and can speed up inference on compatible hardware (like GPUs) without a significant loss in performance.
our_model = AutoModelForCausalLM.from_pretrained(
    model_name ,
    torch_dtype = torch.bfloat16
)

# ---
# Setting up the Text Generation Pipeline
# ---

# Creates a text generation pipeline.
# The 'pipeline' function simplifies using models for specific tasks.
# We tell it to use our 'our_model' and 'our_tokenizer' for the "text-generation" task.
# This sets up an easy-to-use interface for generating text.
gen_pipeline = pipeline("text-generation",model = our_model,tokenizer=our_tokenizer)

# ---
# Generating Text
# ---

# Calls the text generation pipeline with an input prompt.
# "hye there, how are you?" is the starting text for the model to continue.
# 'max_new_tokens=25' limits the model to generate a maximum of 25 new tokens (words or sub-word units)
# after the input prompt. This controls the length of the generated response.
gen_pipeline("hye there, how are you?", max_new_tokens=25)