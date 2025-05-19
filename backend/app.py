from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend_gear():
    data = request.get_json()
    
    # This is placeholder logic; replace with real recommendation logic later
    breed = data.get("breed")
    size = data.get("size")
    activity = data.get("activity_level")

    recommendation = {
        "collar": "Reflective Nylon Collar",
        "harness": "Front-Clip No-Pull Harness",
        "leash": "6ft Padded Handle Leash"
    }

    return jsonify({
        "recommendation": recommendation,
        "input": data
    })

if __name__ == '__main__':
    app.run(debug=True)
