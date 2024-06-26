# Agriculture-Assistant-Bot--Draconis
## Transformer
### Self attention
Finding the important word in the sentence 
Query, key, value
= softmax(QKT/√(d_k ))V
	Dk - dimensionality
	Q - query
	K - key/response
	V - value
	Softmax(a) = ea/∑▒e^b 
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
			ii. Pe(pos,2i) = sin(pos/100002i/dmodel
				Pe (pos,2i+1)= cos(pos/100002i/dmodel
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
