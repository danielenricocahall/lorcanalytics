import ssl
import string
from typing import List, Any, Dict

import nltk
import requests
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

LORCANA_API_ENDPOINT = "https://api.lorcana-api.com/cards"


def fetch_cards_data():
    page_num = 1
    results = []
    while True:
        response = requests.get(f'{LORCANA_API_ENDPOINT}/all?page={page_num}&pagesize=1000')
        if not response.json():
            break
        results += response.json()
        page_num += 1
    return results

# Counter({'Amber': 102, 'Steel': 102, 'Ruby': 102, 'Emerald': 102, 'Sapphire': 102, 'Amethyst': 102})

def filter_cards(cards: List[Dict[str, Any]], colors: List[str]) -> List[Dict[str, Any]]:
    return list(filter(lambda card: card['Color'] in colors, cards))

def remove_stopwords(text: str) -> str:
    return ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_words])


def remove_punctuation(text: str) -> str:
    return ' '.join([word for word in word_tokenize(text) if word not in string.punctuation])


def remove_lorcana_symbols(text: str) -> str:
    return text.replace("{i}", "").replace("{e}", "").replace("{s}", "").replace("{p}","").replace("{l}", "")


def cluster_body_texts(cards):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # A smaller, faster model suitable for this task
    texts = [card['Body_Text'] for card in cards if 'Body_Text' in card]
    texts = list(map(lambda text: remove_punctuation(remove_lorcana_symbols(text)), texts))
    embeddings = model.encode(texts)

    # Perform k-means clustering
    sse = {}
    for k in range(1, 60):
        kmeans = KMeans(n_clusters=k, init="random").fit(embeddings)
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()
    """
    num_clusters = 25
    kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)

    labels = kmeans.labels_
    predicted_labels = kmeans.predict(embeddings)
    for i in range(num_clusters):
        print(f'Cluster {i}:')
        cluster_texts = [texts[j] for j in range(len(texts)) if predicted_labels[j] == i]
        print(cluster_texts[:5])


    # Plotting results (optional)
    # Reduce dimensions for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Clustering of Card Body Texts')
    plt.show()

    return labels
    """



if __name__ == "__main__":
    cards_data = fetch_cards_data()
    cards_data = filter_cards(cards_data, ['Ruby', 'Amber'])
    cluster_labels = cluster_body_texts(cards_data)
