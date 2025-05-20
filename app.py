from flask import Flask,render_template, request,jsonify,redirect,flash,url_for,session
import os
import pandas as pd
import json

from utils.processing_csv import process_info_csv
from utils.llm import general_query_llm,call_fitness_llm
from utils.retriever import retrieve_similar_chunks

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

    # Initialize session memory if not exists
    if "chat_history" not in session:
        session["chat_history"] = []
    if "fitness_level" not in session:
        session["fitness_level"] = "None"
    if "fitness_problem" not in session:
        session["fitness_problem"] = "None"

    if not user_query.strip():
        return jsonify({"response": "Please enter a valid question."})

    # Add user message to history
    session["chat_history"].append({"role": "user", "content": user_query})

    json_output = general_query_llm(user_query, api_key, session["chat_history"])

    try:
        parsed = json.loads(json_output)

        # Update stored level if a new one is detected
        if parsed.get("exp_level") and parsed["exp_level"] != "None":
            session["fitness_level"] = parsed["exp_level"]

        # Update stored problem if a new one is detected
        if parsed.get("fitness_problem") and parsed["fitness_problem"] != "None":
            session["fitness_problem"] = parsed["fitness_problem"]

        if parsed["fitness_related"]:
            if parsed["exp_level"] == "None":
                session["chat_history"].append({"role": "assistant", "content": parsed["response"]})
                session.modified = True  
                return jsonify({"response": parsed["response"]})
            else:
                # If routing to fitness LLM, you still need to save history
                session["chat_history"].append({"role": "assistant", "content": parsed["response"]})
                session.modified = True

                retrieved_chunks = retrieve_similar_chunks(user_query, k=5)

                response = call_fitness_llm(user_query,session["fitness_level"],session["fitness_problem"], api_key,retrieved_chunks)

                session["chat_history"].append({"role": "assistant", "content": response})
                
                return jsonify({"response": response})
        else:
            session["chat_history"].append({"role": "assistant", "content": parsed["response"]})
            session.modified = True  
            return jsonify({"response": parsed["response"]})

    except Exception as e:
        print("Failed to parse:", e)
        
        error_response = "Sorry, something went wrong. Please try again."
        session["chat_history"].append({"role": "assistant", "content": error_response})
        session.modified = True 
        return jsonify({"response": error_response})




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