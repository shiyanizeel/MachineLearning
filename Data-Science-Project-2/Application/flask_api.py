from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('../Models/sentiment_model.h5')

# Load the tokenizer (use the same parameters as used during training)
tokenizer = Tokenizer(num_words=5000)

# Define the maximum sequence length (same as used during training)
max_length = 100

# Create a function to predict sentiment
def predict_sentiment(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, truncating='post')

    # Predict the sentiment
    prediction = model.predict(padded)
    
    return prediction

# Define the API endpoint for sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Check if 'text' is in the JSON
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    user_input = data['text']

    # Call the prediction function
    prediction = predict_sentiment(user_input)

    # Determine the sentiment based on the prediction
    sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax()]

    # Return the result as JSON
    return jsonify({'sentiment': sentiment})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
