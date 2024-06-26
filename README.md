# Agriculture-Assistant-Bot--Draconis
## Transformer 
**Self attention:**
Finding the important word in the sentence.
Query, key, value
= softmax(QK/√(Dk))V
- Dk - dimensionality
- Q - query
- K - key/response
- V - value
Softmax(a) = e^a/∑e^b 
6 layers:
	1.Input embedding - converting to vectors ( numbers)
	2.Positional encoding - maps to another number
	3.Self attention - gives a new set of numbers
	4.Feed forward - in-depth feature extraction
	5.Normalization 
	6.Output
		
Eg: the quick brown fox jumps over the lazy dog
i. E1…….e9
	Each with their own vector eg: [.1,.3,.5,.7]
ii. Pe(pos,2i) = sin(pos/100002i/dmodel)
	Pe (pos,2i+1)= cos(pos/100002i/dmodel)
	0<= i < d/2
iii. Self attention 
	Q = xwq (seeking vector)
	K = xwk (relevant vector)
	V = xwv (validator) (aggregation score) features of data
	Softmax(qk/root(d))v
iv. Feed forward
	FFN = Max(0,xw1+b1)w2+ b2 
		    ^        ^
		  Hidden   output
	Bring to higher dimension
v. Normalization 
	Layer norm = ((x−u)/σ)γ+β
	Gamma and beta acquired through model training
vi. Multiheaded attention
	Multihead(q,k,v) = Concate(head1,head2,…)
	Headi=attention(QWiq,KWik,VWiv)
	Perform more feature extraction
vii. Feedforward 
	Gate->up->down
	Gate - high dimensional projection
	Up - deeper high dimensional projection
	Down - back to embedded projection
viii. Normalization
## Architecture
Transformers_Architecture.ipynb - Colab (google.com)

Generating text through transformers
Vocab creation
Indexing token and vice versa
Transformer: embedding -> forward

Python3 Fundamentals.ipynb - Colab (google.com)

Tokenization_with_bpe_sentencepiece.ipynb - Colab (google.com)

Split the sentence and check each adjacent elements
Create pairs and their frequency count
Merge pairs based on highest freq

Using sentencepiece module
Train using data
Use the model to encode as pieces
	original sentence: I have a dog
	tokenized sentence: ['_','I','_ha','v','e','_','a','_dog']



Fine tuning
	
Low rank adaptation: Create an adaptor model using existing layers, and merge the adaptor model with original model,
Reducing the rank of large matrix to get smaller rank matrix to reduce computational complexity


