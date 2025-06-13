from flask import Flask, request, jsonify, render_template
import random
import logging
import pandas as pd
from datetime import datetime


app = Flask(__name__, static_folder='static', template_folder='templates')

# 🧾 Logs user gear recommendation requests
logging.basicConfig(filename='gear_requests.log', level=logging.INFO)

# 📄 Load CSV of gear options (you’ll create this later)
gear_data = pd.read_csv('gear_data.csv')

# 🌐 Web page routes
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


@app.route('/recommend', methods=['POST'])
def recommend_gear():
    data = request.get_json()

    # Get user answers from the demo form   
    breed = data.get("breed", "").lower()
    size = data.get("size", "").lower()
    puller = data.get("puller", "").lower()
    budget = data.get("budget", "").lower()

   # 🧠 Very basic filtering: check if any matching rows exist
    filtered = gear_data[
        (gear_data["breed"].str.lower().str.contains(breed.lower(), na=False)) |
        (gear_data["size"].str.lower().str.contains(size.lower(), na=False)) |
        (gear_data["pulls"].str.lower().str.contains(puller.lower(), na=False)) |
        (gear_data["budget"].str.lower().str.contains(budget.lower(), na=False))   
    ]

    if filtered.empty:
        recommendation = gear_data.sample(1).to_dict(orient="records")[0]
    else:
         recommendation = filtered.sample(1).to_dict(orient="records")[0]    


    # 🧾 Log this request
    logging.info(f"[{datetime.now()}] /recommend requested with data: {data}")

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
