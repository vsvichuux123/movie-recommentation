from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

data = {
    'title': [
        'The Matrix', 'John Wick', 'Toy Story', 'Finding Nemo', 'Interstellar',
        'KGF', 'Avengers', 'Inception', 'Titanic', 'Gladiator'
    ],
    'genres': [
        'Action Sci-Fi', 'Action Thriller', 'Animation Comedy', 'Animation Adventure', 'Sci-Fi Drama',
        'Action Drama', 'Action Sci-Fi', 'Action Sci-Fi Thriller', 'Drama Romance', 'Action Drama History'
    ]
}
df = pd.DataFrame(data)

count = CountVectorizer(tokenizer=lambda x: x.split())
count_matrix = count.fit_transform(df['genres'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

def get_recommendations(title):
    title = title.strip().lower()  # normalize input to lowercase
    titles_lower = [t.lower() for t in df['title']]
    if title not in titles_lower:
        return []
    idx = titles_lower.index(title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # top 3 similar movies excluding itself
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie = data.get('movie') if data else None
    recommendations = get_recommendations(movie) if movie else []
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
