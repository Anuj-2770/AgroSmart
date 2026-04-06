# Importing essential libraries and modules

from pyexpat import model
from flask import Flask, redirect, render_template, request, session
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from flask import jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# =========================================================================================
# ================= NLP CHATBOT MODEL =================

training_sentences = [
    "Which crop is best in summer?",
    "Suggest crop for high rainfall",
    "Best fertilizer for low nitrogen soil",
    "What crop should I grow in winter?",
    "How to improve soil fertility?",
    "Crop for hot weather",
    "Rainy season crop",
    "Low phosphorus fertilizer suggestion"
]

labels = [
    "summer_crop",
    "high_rainfall_crop",
    "fertilizer",
    "winter_crop",
    "soil_improvement",
    "summer_crop",
    "high_rainfall_crop",
    "fertilizer"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)

chatbot_model = LogisticRegression()
chatbot_model.fit(X, labels)


def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = chatbot_model.predict(input_vector)[0]

    responses = {
        "summer_crop": "Maize and Cotton are suitable crops for summer season.",
        "high_rainfall_crop": "Rice and Sugarcane perform well in high rainfall areas.",
        "fertilizer": "Use Nitrogen-rich or balanced NPK fertilizer based on soil deficiency.",
        "winter_crop": "Wheat and Mustard grow well during winter season.",
        "soil_improvement": "Add organic compost and proper irrigation to improve soil fertility."
    }

    return responses.get(prediction, "Please provide more soil or weather details.")

# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# =========================================================================================
crop_df = pd.read_csv("data-processed/crop_details.csv")
market_df = pd.read_csv("data-processed/market_price.csv")
crop_df['crop'] = crop_df['crop'].str.strip().str.lower()
market_df['crop'] = market_df['crop'].str.strip().str.lower()


# Custom functions for calculations
def weather_fetch(city_name):
    try:
        api_key = config.weather_api_key
        base_url = "https://api.openweathermap.org/data/2.5/weather?"
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name

        response = requests.get(complete_url, timeout=5)
        x = response.json()

        if str(x.get("cod")) == "200":
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2)
            humidity = y["humidity"]
            return temperature, humidity
        else:
            return None
    except Exception as e:
        print("Weather API Error:", e)
        return None

import torch.nn.functional as F

def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(img)).convert("RGB")
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        outputs = model(img_u)
        probabilities = F.softmax(outputs, dim=1)

        confidence, preds = torch.max(probabilities, dim=1)

    confidence_value = confidence.item()
    prediction = disease_classes[preds[0].item()]

    return prediction, confidence_value

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)   
app.secret_key = "agrosmart_secret_key"

# render home page
@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page

# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        weather = weather_fetch(city)

        if weather is not None:
            temperature, humidity = weather

            # ---- ML Prediction ----
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = str(my_prediction[0]).strip()
            display_prediction = final_prediction.title()

            # ==============================
            # ---- Crop Details ----
            # ==============================
            crop_row = crop_df[crop_df['crop'] == final_prediction]

            if not crop_row.empty:
                crop_row = crop_row.iloc[0]
                info = {
                    "duration": crop_row['duration'],
                    "temperature": crop_row['temperature'],
                    "soil": crop_row['soil'],
                    "yield": crop_row['yield'],
                    "season": crop_row['season']
                }
            else:
                info = {
                    "duration": "Not Available",
                    "temperature": "Not Available",
                    "soil": "Not Available",
                    "yield": "Not Available",
                    "season": "Not Available"
                }

            # ==============================
            # ---- Market Details ----
            # ==============================
            market_row = market_df[market_df['crop'] == final_prediction]

            if not market_row.empty:
                market_row = market_row.iloc[0]
                price = float(market_row['price_per_kg'])
                yield_estimate = float(market_row['yield_kg_per_hectare'])
                estimated_income = price * yield_estimate
                income_formatted = "₹ {:,.0f}".format(estimated_income)
            else:
                price = 0
                yield_estimate = 0
                income_formatted = "Data Not Available"

            # ==============================
            # ---- Confidence & Risk ----
            # ==============================
            proba = crop_recommendation_model.predict_proba(data)
            confidence = round(max(proba[0]) * 100, 2)
            risk_value = round(100 - confidence, 2)

            if confidence > 85:
                risk = "Low Risk 🟢"
            elif confidence > 60:
                risk = "Medium Risk 🟡"
            else:
                risk = "High Risk 🔴"

            return render_template(
                "crop-result.html",
                prediction=display_prediction,
                price=price,
                yield_estimate=yield_estimate,
                income=income_formatted,
                confidence=confidence,
                risk=risk,
                risk_value=risk_value,
                info=info
            )

        else:
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page
@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('app/Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)

        try:
            # Read uploaded image
            img = file.read()

            # Get prediction and confidence
            prediction, confidence = predict_image(img)

            # Check if confidence is high enough
            if confidence < 0.95:
                result = "Please upload a clear leaf image."
            else:
                result = disease_dic[prediction]

            # Make safe for HTML rendering
            prediction = Markup(str(result))

            return render_template('disease-result.html',
                                   prediction=prediction,
                                   title=title)

        except Exception as e:
            print("Error:", e)
            return render_template('disease.html', title=title)

    # GET request
    return render_template('disease.html', title=title)

# ===============================================================================================
#Flask Route add 
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"].lower()

    # 🔄 Reset Chat
    if any(word in user_message for word in ["reset", "restart", "start over", "clear"]):
        session.clear()
        return jsonify({
        "reply": ""
    })

    # 🆘 Help Command
    if any(word in user_message for word in ["help", "options", "how can help you ","what can you do"]):
       return jsonify({
        "reply": "I can help you with:\n"
                 "🌾 Crop suggestions by season\n"
                 "💰 Cost estimation\n"
                 "⚠️ Risk analysis\n"
                 "🌡️ Ideal temperature info\n"
                 "🧪 Fertilizer guidance\n"
                 "🌿 Leaf disease prediction\n\n"
                 "Just tell me what you need 😊"
    })

    # 1️⃣ Greeting
    if any(word in user_message for word in ["hi", "hello", "hey"]):
        return jsonify({
            "reply": "Hello 👋 I am AgroSmart AI. How can I help you today regarding crop recommendations?"
        })
    # 2️⃣ thanku handling
    if any(word in user_message for word in ["thank", "thanks", "thank you", "thx"]):
        return jsonify({
        "reply": "You're welcome 😊 Would you like suggestions for another crop or season?"
    })
    # 3️⃣  ok handling
    if any(word in user_message for word in ["ok", "okay", "good", "nice"]):
        return jsonify({
        "reply": "Great 👍 Let me know the season or soil details to suggest crops."
    })

    # 4️⃣  Season handling
       # 🌞 Summer
    if "summer" in user_message:
        return jsonify({
            "reply": "For Summer 🌞 season, Maize 🌽, Cotton, and Groundnut are suitable crops."
        })

    # 🌧️ Monsoon / Rainy
    if "monsoon" in user_message or "rainy" in user_message:
        return jsonify({
            "reply": "During Monsoon 🌧️ season, Rice 🌾, Soybean, and Maize perform very well."
        })

    # ❄️ Winter
    if "winter" in user_message:
        return jsonify({
            "reply": "For Winter ❄️ season, Wheat 🌾, Mustard, and Peas are good crop choices."
        })
       # 🌸 Spring
    if "spring" in user_message:
        return jsonify({
            "reply": "In Spring 🌸 season, Crops like Sunflower 🌻, Vegetables, and Barley are suitable."
        })
    if "vegetable" in user_message:
        return jsonify({
        "reply": "Popular vegetables for spring include Tomato 🍅, Cucumber 🥒, Spinach 🥬, Carrot 🥕, and Capsicum."
    })


    # 🍂 Autumn / Post-Monsoon
    if "autumn" in user_message or "post monsoon" in user_message:
        return jsonify({
            "reply": "In Autumn 🍂 season, Crops like Millet and Pulses can be cultivated."
        })


    # 5️⃣  Crop / farming related keywords
    if any(word in user_message for word in ["crop", "food", "season"]):
        return jsonify({
            "reply": "Please provide details like rainfall, soil type, or season so I can suggest the best crop."
        })
    
    # 6️⃣ Follow-up Logic
    if "fertilizer" in user_message:
        return jsonify({
         "reply": "Please enter your soil NPK values in the fertilizer prediction section for accurate advice."
        })
    
    # 7️⃣  Yes / No Handling
    if any(word in user_message for word in ["yes", "haan","yeah"]):
       return jsonify({
        "reply": "Great 👍 Please tell me if you need fertilizer advice, crop advice, or disease detection."
       })
    # no handling
    if any(word in user_message for word in ["no","nahi","nope"]):
        return jsonify({
        "reply": "No problem 😊 Would you like help with another season or crop?"
        })
    # ph Handling
    if "ph" in user_message:
        return jsonify({
        "reply": "Soil pH tells how acidic or alkaline your soil is. pH 7 = neutral, below 7 = acidic, above 7 = alkaline. Most crops grow best between pH 6.0 and 7.5."
    })
    # NPK Handling
    if "npk" in user_message or "npk value" in user_message:
        return jsonify({
        "reply": "NPK stands for Nitrogen (N), Phosphorus (P), and Potassium (K). These are the 3 main nutrients that plants need for growth, root development, and overall strength."
    })
    if "nitrogen" in user_message or "n value" in user_message:
        return jsonify({
        "reply": "Nitrogen helps plants grow leaves and maintain green color. Low nitrogen causes yellow leaves and slow growth."
    })
    if "phosphorus" in user_message or "p value" in user_message:
        return jsonify({
        "reply": "Phosphorus helps plants develop strong roots, flowers, and fruits. It is essential for energy transfer in plants."
    })
    if "potassium" in user_message or "k value" in user_message:
        return jsonify({
        "reply": "Potassium increases plant strength, improves water usage, and helps crops resist diseases and harsh weather."
    })
    # Soil handling 
    if "soil" in user_message or "what is soil" in user_message:
        return jsonify({
        "reply": "Soil is a natural mixture of minerals, organic matter, air, and water that supports plant growth. Soil fertility depends on pH, NPK nutrients, organic carbon, moisture, and texture.\n\nCommon soil types:\n• Sandy Soil – Fast drainage, best for peanut, carrot, coconut.\n• Clay Soil – High water holding, good for rice and wheat.\n• Loamy Soil – Best for farming; mixture of sand, silt, clay.\n• Silt Soil – Smooth and fertile, good for vegetables.\n• Peaty Soil – High organic matter, good moisture retention."
    })
    # organic carbon handler 
    if "organic carbon" in user_message or "oc" in user_message:
        return jsonify({
        "reply": "Organic Carbon improves soil health, increases nutrient availability, and helps in maintaining moisture. OC above 0.75% is considered good for crops."
    })
    # 🌱 Soil Behavior
    if "acidic soil" in user_message:
        return jsonify({"reply": "Acidic soil has pH below 7. Crops like potato, tomato, tea, and pineapple grow well in acidic soil."})

    if "alkaline soil" in user_message:
        return jsonify({"reply": "Alkaline soil has pH above 7. Mustard, barley, and cotton grow well in alkaline soil."})


    # 🌾 Crop Suitability
    if "sandy soil" in user_message:
        return jsonify({"reply": "Sandy soil is ideal for crops like watermelon, carrot, peanut, coconut, and cotton."})

    if "less water" in user_message or "low water" in user_message:
        return jsonify({"reply": "Millet, mustard, sorghum, and chickpea are crops that require very little water."})

    if "winter crop" in user_message or "crop for winter" in user_message:
        return jsonify({"reply": "Winter crops include wheat, mustard, peas, carrot, and cabbage."})

    if "summer crop" in user_message or "crop for summer" in user_message:
        return jsonify({"reply": "Summer crops include maize, groundnut, cotton, and cucumber."})


    # 🌧 Rainfall / Drought
    if "rainfall for rice" in user_message or "rainfall" in user_message:
        return jsonify({"reply": "Most crops require 150–300 mm rainfall for healthy growth. Low-water crops can grow in 50–150 mm rainfall, while rice and other water-loving crops prefer 300–500 mm rainfall."})

    if "drought crop" in user_message:
        return jsonify({"reply": "Millet, barley, sorghum, and pulses are drought-resistant crops."})


    # 🌡 Temperature
    if "temperature for wheat" in user_message or "temperature" in user_message:
        return jsonify({"reply":"Most crops grow well between 10°C to 30°C. Cool-season crops prefer 10°C–20°C, while warm-season crops grow best in 20°C–30°C temperature."})


    # 🍂 Disease / Leaf issues
    if "yellow leaves" in user_message or "yellow leaf" in user_message:
        return jsonify({"reply": "Yellow leaves occur due to overwatering, nitrogen deficiency, or fungal disease."})
    if "disease" in user_message:
        return jsonify({
         "reply": "Please upload leaf photo of your crop in disease prediction section for accurate advice."
        })
    if "leaf spot" in user_message or "spots on leaves" in user_message:
        return jsonify({"reply": "Leaf spots are caused by fungal or bacterial infections. Use organic fungicide and avoid wet leaves."})
    # 8️⃣  Unnecessary / unrelated question
    return jsonify({
    "reply": "⚠️ I am AgroSmart AI 🌾, designed to assist farmers with crop recommendations, fertilizer advice,  and seasonal crop planning. Kindly ask queries related to agriculture so I can help you effectively."
})


# ==========================================================================================================
if __name__ == '__main__':
    app.run(debug=True)
