from flask import Flask, request, jsonify, render_template
import random
import logging
from datetime import datetime


app = Flask(__name__, static_folder='static', template_folder='templates')

logging.basicConfig(filename='gear_requests.log', level=logging.INFO)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/how-it-works') 
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/why-us')
def why_us():
    return render_template('why-us.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

# these recommendations are placeholders

@app.route('/recommend', methods=['POST'])
def recommend_gear():
    data = request.get_json()

    logging.info(f"[{datetime.now()}] /recommend requested with data: {data}")

    all_recommendations = [
        {
            "message": "Considering you're managing multiple dogs and your dog is a German Shepherd, pulls, and is being trained as a guard dog, we think this gear will fit your dog well!",
            "collar": "Tuff Pupper Heavy Duty Dog Collar with Handle, Large: 18 to 26-in neck, 1.5-in wide",
            "harness": "ICEFANG Tactical Dog Harness, No Pull, Reflective, Large: 28 to 35-in chest",
            "leash": "BAAPET Strong Dog Leash, Black, Large: 6-ft long, 1-in wide, with Padded Handle"
        },
        {
            "message": "Considering your dog is a Chihuahua, does not pull, and is still growing, we think this gear will fit your dog well!",
            "collar": "Reflective Nylon Collar",
            "harness": "Front-Clip No-Pull Harness",
            "leash": "6ft Padded Handle Leash"
        },
        {
            "message": "Considering your dog is a Golden Retriever, does not pull, and you're on a tight budget, we think this gear will fit your dog well!",
            "collar": "Blueberry Pet Classic Solid Nylon Dog Collar, Large: 18 to 26-in neck, 1-in wide",
            "harness": "Chai's Choice Outdoor Adventure 3M Reflective Dog Harness, Large: 22 to 35-in chest",
            "leash": "Frisco Solid Nylon Dog Leash, Large: 6-ft long, 1-in wide"
        },
        {
            "message": "Considering your dog is a young Doberman, pulls, is being trained as a guard dog, and you're on a moderate budget, we think this gear will fit your dog well!",
            "collar": "PetSafe Quick Snap Buckle Nylon Martingale Dog Collar, Medium: 14 to 20-in neck",
            "harness": "Rabbitgoo No-Pull Dog Harness, Adjustable and Reflective, Large: 17 to 34-in chest",
            "leash": "iYoShop Heavy Duty Rope Dog Leash with Padded Handle, 5-ft long"
        }
    ]

    selected = random.sample(all_recommendations, 4)

    return jsonify({
        "recommendations": selected,
        "input": data
    })


# fallback for undefined pages
@app.errorhandler(404)
def page_not_found(e):
    return "Page not found", 404

if __name__ == '__main__':
    app.run(debug=True)
