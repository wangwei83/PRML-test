### 文档的向量化
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
'Jobs was the chairman of Apple Inc.',
'I like to use apple computer.',
'And I also like to eat apple.']

vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

print("=====================================")

###经过停用词过滤后的文档向量化
import nltk
from nltk.corpus import stopwords
nltk.data.path.append(r'C:\nltk_data')

# from nltk.book import *
# nltk.download('stopwords')
print(stopwords.words('english'))