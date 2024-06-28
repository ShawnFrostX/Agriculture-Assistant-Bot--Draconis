# Agriculture-Assistant-Bot--Draconis
## Transformer 

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



## Architecture
[Python Intro](https://colab.research.google.com/drive/1DbxjnrSEhoLEIFb9arz2-U0OR7N4abfM#scrollTo=F2ikhLFfuon_)

[Transformers_Architecture.ipynb - Colab (google.com)](https://colab.research.google.com/drive/1P6hG2t0ijSnVQ3Y0wx1ixTiuotN0DeLC?usp=sharing#scrollTo=YCSiAqIjeNMD)

- Generating text through transformers  
- Vocab creation  
- Indexing token and vice versa  
- Transformer: embedding -> forward  

[Tokenization_with_bpe_sentencepiece.ipynb - Colab (google.com)](https://colab.research.google.com/drive/1CnLOHFRdrg3Hw7-DN7FiuPreAq6nLKnD?usp=sharing)

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

## Fine tuning
**Low rank adaptation:**   
Create an adaptor model using existing layers, and merge the adaptor model with original model, Reducing the rank of large matrix to get smaller rank matrix to reduce computational complexity 

[LLM_Fine-Tuning_new.ipynb - Colab (google.com)](https://colab.research.google.com/drive/1RdePhapMcBCplrhtN4N5-27ttjb4OVhO?authuser=2)

![Training](https://github.com/ShawnFrostX/Agriculture-Assistant-Bot--Draconis/blob/ce610d019fdaa3196275047ce1b8a0632e673368/Report.png)

*Had limitations like Blunder generation and Hallucination* 
*Generates unexpected responses for unknown inputs*  

## RAG(Retrieval Augmented Generation)  
Using RAG facilitates modifying data and further improve the model  
Uses *langchain* and *db vectors* for optimization

[quant_rag.ipynb - Colab (google.com)](https://colab.research.google.com/drive/1qkeulBrOrRftebrVqFr4ivE_6pRj4saK?authuser=2#scrollTo=MloXh3yjC7iM)

*The RAG helped add more data but could not perform well when greeting input was given. It responded with the answers for random Questions from the data.*
