# Agriculture-Assistant-Bot--Draconis
## Problem Statement
To create a LLM model with data regarding the different disease affecting common crops grown in kerala

## Solution
### Transformer 

**Self attention:**  
Finding the important word in the sentence.  
Query, key, value  
=softmax(QK/√(Dk))V
- Dk - dimensionality
- Q - query
- K - key/response
- V - value  
Softmax(a) = e^a/∑e^b  

**6 layers:**  
1. Input embedding - converting to vectors ( numbers)  
2. Positional encoding - maps to another number  
3. Self attention - gives a new set of numbers
4. Feed forward - in-depth feature extraction
5. Normalization 
6. Output
		
*Eg: the quick brown fox jumps over the lazy dog*  
1. E1…….e9
	Each with their own vector eg: [.1,.3,.5,.7]
1. Positional encoding
    ``` 
    Pe(pos,2i) = sin(pos/100002i/dmodel)  
	Pe (pos,2i+1)= cos(pos/100002i/dmodel)  
	0<= i < d/2
    ```
1. Self attention  
	Q = xwq (seeking vector)  
	K = xwk (relevant vector)  
	V = xwv (validator) (aggregation score) features of data  
	Softmax(qk/root(d))v  
1. Feed forward  
	```
    FFN = Max(0,xw1+b1)w2+ b2  
		    ^        ^  
		  Hidden   output  
	Bring to higher dimension
    ```
1. Normalization  
	`Layer norm = ((x−u)/σ)γ+β `  
	Gamma and beta acquired through model training
1. Multiheaded attention  
	`Multihead(q,k,v) = Concate(head1,head2,…)`  
	`Headi=attention(QWiq,KWik,VWiv)`  
	Perform more feature extraction
1. Feedforward  
	Gate->Up->Down  
	Gate - high dimensional projection  
	Up - deeper high dimensional projection  
	Down - back to embedded projection  
1. Normalization



### Architecture
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DbxjnrSEhoLEIFb9arz2-U0OR7N4abfM#scrollTo=F2ikhLFfuon_) : Python Intro

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1P6hG2t0ijSnVQ3Y0wx1ixTiuotN0DeLC?usp=sharing#scrollTo=YCSiAqIjeNMD) : Transformers_Architecture

- Generating text through transformers  
- Vocab creation  
- Indexing token and vice versa  
- Transformer: embedding -> forward  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnLOHFRdrg3Hw7-DN7FiuPreAq6nLKnD?usp=sharing) : Transformers_Architecture

- Split the sentence and check each adjacent elements
- Create pairs and their frequency count
- Merge pairs based on highest freq

*Using sentencepiece module*  
- Train using data  
- Use the model to encode as pieces  
```
original sentence: I have a dog
tokenized sentence: ['_','I','_ha','v','e','_','a','_dog']
```

### Fine tuning
**Low rank adaptation:**   
Create an adaptor model using existing layers, and merge the adaptor model with original model, Reducing the rank of large matrix to get smaller rank matrix to reduce computational complexity 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RdePhapMcBCplrhtN4N5-27ttjb4OVhO?authuser=2) : LLM_Fine-Tuning_new

![Training](https://github.com/ShawnFrostX/Agriculture-Assistant-Bot--Draconis/blob/ce610d019fdaa3196275047ce1b8a0632e673368/Report.png)

## Limitations
- Blunder generation
- Hallucination
- Unexpected response for unknown inputs
  

## Applying RAG and overcoming limitations

Most of the limitation can be solved using RAG   
RAG helped train on newly added data  
Use langchain and db vectors for optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qkeulBrOrRftebrVqFr4ivE_6pRj4saK?authuser=2#scrollTo=MloXh3yjC7iM) : quant_rag

### Limitations

Could not perform well when greeting input was given. It responded with the answers for random Questions from the data.

## Further research

Can be furher improved by adding:
- Greetings
- Giving the responses a flow rather than sounding robotic
- Building an interface easier user interaction

## Deployment

### Deployment as an OpenAI Compatible API

#### Install vLLM + Haystack

- we install vLLM using pip 
- for production use cases, there are many other options, including Docker 

``` 
!pip install vllm haystack-ai 
```

```py
# we prepend "nohup" and postpend "&" to make the Colab cell run in background
! nohup python -m vllm.entrypoints.openai.api_server \
				--model '/content/drive/MyDrive/Colab Notebooks/addon GENAI/final_weights_new' \
				--dtype auto \
				--max-model-len 2048 \
				> vllm.log &
```
``` py
# we check the logs until the server has been started correctly
!while ! grep -q "Application startup complete" vllm.log; do tail -n 1 vllm.log; sleep 5; done
```
```py
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
import string
import random

# initializing size of string
N = 20

# using random.choices()
# generating random strings
res = ''.join(random.choices(string.ascii_uppercase +
							string.digits, k=N))

generator = OpenAIChatGenerator(
	api_key=Secret.from_token(res),  # for compatibility with the OpenAI API, a placeholder api_key is needed
	model="/content/drive/MyDrive/Colab Notebooks/addon GENAI/final_weights_new",
	api_base_url="http://localhost:8000/v1",
	generation_kwargs = {"max_tokens": 1024}
)

```
```py
messages = []

while True:
msg = input("Enter your message or Q to exit\n🧑 ")
if msg=="Q":
	break
messages.append(ChatMessage.from_user(msg))
response = generator.run(messages=messages)
assistant_resp = response['replies'][0]
print("🤖 "+assistant_resp.content)
messages.append(assistant_resp)
```
### GGUF Model Deployment Guide

Sample deployment configuration

#### Prerequisites

- Python 3.8 or higher

#### Installation

1. Install the required Python packages.

#### Ubuntu CPU

```sh
pip install llama-cpp-python
pip install flask
```
#### Ubuntu with CUDA

```sh
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
pip install flask
```

#### Windows

Download and install Anaconda python from [here](https://www.anaconda.com/download)

```sh
conda create -n deeplearning python=3.8
conda activate deeplearning
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install llama-cpp-python
pip install flask
```

2. Download the required model file from [here](https://drive.google.com/file/d/13UUBxOuFUrbrTGGuPxT7WXJpldSjbqXO/view).

#### CPU 
1. Run the `app_cpu.py` script to start the Flask server.

```sh
python app_cpu.py
```
#### CUDA GPU
1. Run the `app_cuda_gpu.py` script to start the Flask server.

```sh
python app_cuda_gpu.py
```


2. In a new terminal, run the `post_request.py` script to send a POST request to the server.

```sh
python post_request.py
```

#### Code Explanation

`app.py` is the main server file that uses Flask to create a web API. It uses the Llama library to generate responses based on the input message.

`post_request.py` is a script that sends a POST request to the server with a message. The server then uses the Llama library to generate a response and sends it back to the client.

#### Sample Request

You can send a POST request to `http://localhost:5000/api/deployment` with the following JSON body:

```json
{
    "message": "what are the best pesticides for crops in Kerala?"
}
```

The server will respond with the AI-generated response.

#### Deploying flask in production
Clicke [here](https://medium.com/techfront/step-by-step-visual-guide-on-deploying-a-flask-application-on-aws-ec2-8e3e8b82c4f7) to learn more on how to deploy a flask application in production



## Reference

TinyLlama-1.1B-Chat-v1.0  
https://pypi.org/project/llama-cpp-python/
