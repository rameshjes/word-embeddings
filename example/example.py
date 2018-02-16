import gensim
from gensim.models import Word2Vec
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.decomposition import PCA

'''
Load the text file
Convert text into tokenized sentences and words
Create word2vec model of the text file
Plot using PCA for visualization
'''

class Embeddings:

	'''
	Convert text into tokenized sentences and words
	for word2vec model
	'''

	def preProcessData(self, text):
			
		tokenize_sent = sent_tokenize(text)
		number_of_sent = len(tokenize_sent)
		
		tokenized_words = []
		count = 0
		for i in tokenize_sent:
			word_token = word_tokenize(i)
			count = count + len(word_token)
			tokenized_words.append(word_token)
		print "total number of words ", count
		return tokenized_words

	'''
	Load text from file
	'''


	def loadFile(self):

		f = open('./simple_passage.txt')

		return f.read()


if __name__ == '__main__':


	emb = Embeddings()
	file = emb.loadFile()
	data = emb.preProcessData(file)

	# model = Word2Vec(data, min_count = 1)

	#summarize the model
	# print model

	#summarize the vocabulary
	# words = list(model.wv.vocab)
	# print words
	
	#save model
	# model.save('simple_passage_model.bin')

	# load model
	new_model = Word2Vec.load('simple_passage_model.bin')

	X = new_model[new_model.wv.vocab]
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)

	# create a scatter plot of the projection
	plt.scatter(result[:, 0], result[:, 1])
	words = list(new_model.wv.vocab)
	for i, word in enumerate(words):
		plt.annotate(word, xy=(result[i, 0], result[i, 1]))
	plt.grid(True)
	plt.show()
	