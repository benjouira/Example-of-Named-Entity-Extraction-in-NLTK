# Example -1
# In the below example of named entity recognition in NLTK, we have taken a text from times of India and have applied tokenization and POS tagging to the text.
import nltk
from nltk import word_tokenize,pos_tag
# important to download
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = "NASA awarded Elon Musk’s SpaceX a $2.9 billion contract to build the lunar lander."
tokens = word_tokenize(text)
tag=pos_tag(tokens)
print(tag)

ne_tree = nltk.ne_chunk(tag)
print(ne_tree)

# Example -2
# Let us see one more example where we have used already present tagged sentences provided by the NLTK library.
nltk.download('treebank')
sent = nltk.corpus.treebank.tagged_sents()
print(sent)
print(nltk.ne_chunk(sent[0]))
