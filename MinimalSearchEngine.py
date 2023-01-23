import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import sqrt
import numpy as np


# Make a request to the website
r = requests.get('https://in.ign.com/')
soup = BeautifulSoup(r.content, 'html.parser')
link = []
for i in soup.find('section', {'class':'vspotlight'}).find_all('a'):
    link.append(i['href'])

documents = []
for i in link:
    r = requests.get(i)
    soup = BeautifulSoup(r.content, 'html.parser')
  
    # Retrieve all paragraphs and combine it as one
    sen = []
    for i in soup.find('article', {'class':'article-section article-page'}).find_all('p'):
        sen.append(i.text)
    # Add the combined paragraphs to documents
    documents.append(' '.join(sen))

documents_clean = []
for d in documents:
    # Remove Unicode
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
    # Remove Mentions
    document_test = re.sub(r'@\w+', '', document_test)

    document_test = re.sub(r'[^\w\s]', ' ', document_test)
    document_test = re.sub(r'[0-9]', '', document_test)
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    documents_clean.append(document_test)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents_clean)
X = X.T.toarray()
# Create a DataFrame and set the vocabulary as the index
df = pd.DataFrame(X, index=vectorizer.get_feature_names())


def get_similar_articles(q, df):
  print("query:", q)
  # Convert the query become a vector
  q = [q]
  q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
  sim = {}

  # Calculate the similarity
  for i in range(len(documents_clean)):
    sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
  

  sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
  for k, v in sim_sorted:
    if v != 0.0:
      print(documents_clean[k])
      print()


#Driver 
print("Enter Query")
q1 = input()
get_similar_articles(q1, df)