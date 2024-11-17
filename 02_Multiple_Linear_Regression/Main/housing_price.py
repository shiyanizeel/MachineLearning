import pandas as pd
import joblib 
# Load the model
loaded_model = joblib.load('../Models/linear_regression_model.pkl')

# Function to predict house price
def predict_price(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus):
    return predicted_price

if __name__ == '__main__':
    area = 1500
    bedrooms = 3
    bathrooms = 2
    stories = 2
    parking = 2
    mainroad = True
    guestroom = False
    basement = True
    hotwaterheating = True
    airconditioning = False
    prefarea = True
    furnishingstatus = 'semi-furnished'

    predicted_price = predict_price(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus)
    print(f"Predicted Price: {predicted_price:.2f}")


