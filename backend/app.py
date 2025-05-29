from flask import Flask, request, jsonify, send_from_directory
import os, random

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# testing the new branch. New change

@app.route('/recommend', methods=['POST'])
def recommend_gear():
    data = request.get_json()

    recommendation_1 = {
        "message": "Considering you're managing multiple dogs and your dog is a German Shepherd, pulls, and is being trained as a guard dog, we think this gear will fit your dog well!",
        "collar": "Tuff Pupper Heavy Duty Dog Collar with Handle, Large: 18 to 26-in neck, 1.5-in wide",
        "harness": "ICEFANG Tactical Dog Harness, No Pull, Reflective, Large: 28 to 35-in chest",
        "leash": "BAAPET Strong Dog Leash, Black, Large: 6-ft long, 1-in wide, with Padded Handle"
    }

    recommendation_2 = {
        "message": "Considering your dog is a Chihuahua, does not pull, and is still growing, we think this gear will fit your dog well!",
        "collar": "Reflective Nylon Collar",
        "harness": "Front-Clip No-Pull Harness",
        "leash": "6ft Padded Handle Leash"
    }

    recommendation_3 = {
        "message": "Considering your dog is a Golden Retriever, does not pull, and you're on a tight budget, we think this gear will fit your dog well!",
        "collar": "Blueberry Pet Classic Solid Nylon Dog Collar, Large: 18 to 26-in neck, 1-in wide",
        "harness": "Chai's Choice Outdoor Adventure 3M Reflective Dog Harness, Large: 22 to 35-in chest",
        "leash": "Frisco Solid Nylon Dog Leash, Large: 6-ft long, 1-in wide"
    }

    recommendation_4 = {
        "message": "Considering your dog is a young Doberman, pulls, is being trained as a guard dog, and you're on a moderate budget, we think this gear will fit your dog well!",
        "collar": "PetSafe Quick Snap Buckle Nylon Martingale Dog Collar, Medium: 14 to 20-in neck",
        "harness": "Rabbitgoo No-Pull Dog Harness, Adjustable and Reflective, Large: 17 to 34-in chest",
        "leash": "iYoShop Heavy Duty Rope Dog Leash with Padded Handle, 5-ft long"
    }


    # Randomly pick one
    recommendation = random.choice([recommendation_1, recommendation_2, recommendation_3, recommendation_4])

    return jsonify({
        "recommendation": recommendation,
        "input": data
    })




@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
