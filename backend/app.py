from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# testing the new branch

@app.route('/recommend', methods=['POST'])
def recommend_gear():
    data = request.get_json()

    # photo derived data or user input data (depends)
    weight = data.get("weight")

    #photo derived data
    chest_bridge_length = data.get("chest_bridge_length")
    neck_circumference = data.get("neck_circumference")
    back_bridge_length = data.get("back_bridge_length")
    belly_circumference = data.get("belly_circumference")

    # user input data
    breed = data.get("breed")
    activity = data.get("pull")
    guard_dog = data.get("gaurd_dog")
    total_dogs = data.get("total_dogs")
    budget = data.get("budget")



    recommendation = {
        "collar": "Reflective Nylon Collar",
        "harness": "Front-Clip No-Pull Harness",
        "leash": "6ft Padded Handle Leash"
    }

    return jsonify({
        "recommendation": recommendation,
        "input": data
    })

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
