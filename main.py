import json
import re
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

LORCANA_API = "https://lorcania.com/api/cardsSearch"


def fetch_cards_data():
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
    }

    data = {
        "colors": [],
        "sets": [],
        "traits": [],
        "keywords": [],
        "costs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "inkwell": [],
        "rarity": [],
        "language": "English",
        "options": [],
        "sorting": "default"
    }

    response = requests.post(LORCANA_API, headers=headers, data=json.dumps(data))
    results = response.json()
    return results


# Counter({'Amber': 102, 'Steel': 102, 'Ruby': 102, 'Emerald': 102, 'Sapphire': 102, 'Amethyst': 102})

def filter_cards(cards: List[Dict[str, Any]], colors: List[str]) -> List[Dict[str, Any]]:
    return list(filter(lambda card: card['Color'] in colors, cards))


def remove_stopwords(text: str) -> str:
    return ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_words])


def remove_punctuation(text: str) -> str:
    return ' '.join([word for word in word_tokenize(text) if word not in string.punctuation])


def remove_mark(text: str) -> str:
    return re.sub(r'<mark>.*?</mark>', '', text)


def remove_italic(text: str) -> str:
    return re.sub(r'<i>.*?</i>', '', text)


def remove_bold(text: str) -> str:
    # We're doing this because we want to keep the text inside the bold tags
    # as the card data contains relevant information inside the bold tags e.g;
    # Evasive, Challenger, etc.
    return text.replace('<b>', '').replace('</b>', '')


def remove_line_breaks(text: str) -> str:
    return text.replace('<br>', ' ').replace('<br />', ' ')

def find_elbow_point(sse):
    diffs = [sse[i] - sse[i + 1] for i in range(1, len(sse) - 1)]
    return np.argmin(diffs) + 1


def cluster_body_texts(cards):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # A smaller, faster model suitable for this task
    names_and_texts = [(card['name'], card['action']) for card in cards if card.get('action')]
    names, texts = zip(*names_and_texts)
    texts = list(map(lambda text: remove_punctuation(remove_line_breaks(remove_bold(remove_italic(remove_mark(text))))), texts))
    embeddings = model.encode(texts)

    # Perform k-means clustering
    sse = {}
    for k in range(1, 200):
        kmeans = KMeans(n_clusters=k, init="random").fit(embeddings)
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()
    elbow = find_elbow_point(sse)

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


if __name__ == "__main__":
    cards_data = fetch_cards_data()
    # cards_data = filter_cards(cards_data, ['Ruby', 'Amber'])
    cluster_labels = cluster_body_texts(cards_data['cards'])
