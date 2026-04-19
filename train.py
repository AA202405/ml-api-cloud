import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
# 1. Dataset (Replace with a larger dataset like IMDB for better accuracy)
texts = [
"I love this movie, it is fantastic!",
"Absolutely terrible, worst experience ever.",
"Great acting and wonderful plot.",
"I hated every minute of it.",
"A beautiful and inspiring story.",
"Boring, dull, and a waste of time."
]
labels = ["positive", "negative", "positive", "negative", "positive", "negative"]
# 2. Build and Train the Model
# We use a pipeline to vectorize the text and classify it in one step
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)
# 3. Export the Trained Model
joblib.dump(model, 'sentiment_model.joblib')
print("Model successfully trained and saved as sentiment_model.joblib") 