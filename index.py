from flask import Flask, render_template, request, jsonify
from model import make_recommendation 

app = Flask(__name__, template_folder='./')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/get')
def get_from_api():
    query = request.args.get('msg') 
    try:
        if not query:
            return jsonify({"error": "No message provided."}), 400
        
        recommendation = make_recommendation(str(query))
        
        return jsonify(recommendation)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(debug=True)
