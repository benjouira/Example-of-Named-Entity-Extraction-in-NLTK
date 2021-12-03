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
