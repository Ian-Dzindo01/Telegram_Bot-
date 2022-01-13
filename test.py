from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

examples = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']

for w in examples:
    print(ps.stem(w))

print(lemmatizer.lemmatize("geese"))
