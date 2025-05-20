from flask import Flask,render_template, request,jsonify,redirect,flash,url_for
import os
import pandas as pd
import json

from utils.processing_csv import process_info_csv
from utils.llm import general_query_llm

app = Flask(__name__)
app.secret_key = "olee1234"  

UPLOAD_FOLDER = "exercise_info_folder"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

api_key = "0177126f09f9b3897b18fce813639930f8ed8aa39765677036d8edd53ebbd3a5"

# ───── Routes ─────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data.get("query", "")

    print(user_query)

    if not user_query.strip():
        return jsonify({"response": "Please enter a valid question."})
    
    json_output = general_query_llm(user_query, api_key)

    try:
        parsed = json.loads(json_output)
        if parsed["fitness_related"]:
            if parsed["exp_level"] == "None":
                # Ask for level
                return jsonify({"response": parsed["response"]})
            else:
                pass
        else:
            return jsonify({"response": parsed["response"]})
    except Exception as e:
        print("Failed to parse:", e)
        return jsonify({"response": "Sorry, something went wrong. Please try again."})



@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("csv")
        
        if not file:
            flash("No file selected.", "danger")
            return redirect(request.url)
        
        if not file.filename.endswith(".csv"):
            flash("Only CSV files are allowed.", "danger")
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            process_info_csv(df)
        except Exception as e:
            flash(f"Error reading CSV: {e}", "danger")
            return redirect(request.url)

        flash("Exercise CSV uploaded successfully!", "success")
        return redirect(url_for("upload"))

    return render_template("upload_exercise_info.html")


if __name__ == "__main__":
    app.run(debug=True)