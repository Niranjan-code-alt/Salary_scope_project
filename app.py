from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_salary = None
    if request.method == "POST":
        try:
            age = int(request.form["age"])

            gender_map = {
                "Male": 1,
                "Female": 0,
                "Transgender": 0,
                "Non-Binary": 0,
                "Genderqueer": 0,
                "Prefer Not to Say": 0,
                "Other": 0
            }
            gender = gender_map.get(request.form["gender"], 0)

            marital_map = {
                "Married": 1
            }
            marital_status = marital_map.get(request.form["marital_status"], 0)

            experience = float(request.form["experience"])
            edu_numeric = int(request.form["education_numeric"])
            hours_per_week = int(request.form["hours_per_week"])

            area_map = {
                "Urban": 1,
                "Rural": 0
            }
            area_type = area_map.get(request.form["area_type"], 0)

            religion_map = {
                "Hindu": 1,
                "Muslim": 2,
                "Christian": 3,
                "Sikh": 4,
                "Buddhist": 5,
                "Jain": 6,
                "Parsi": 7,
                "Jewish": 8,
                "Bahai": 9,
                "Tribal/Indigenous": 10,
                "Atheist": 11,
                "Other": 0,
                "Prefer Not to Say": 0
            }
            religion = religion_map.get(request.form["religion"], 0)

            # Adjust input features order and number as per your model training
            input_data = np.array([[age, gender, marital_status, experience,
                                    edu_numeric, hours_per_week, area_type, religion]])
            

            predicted_salary = model.predict(input_data)[0]
            predicted_salary = f"₹{predicted_salary/100000:.1f} lakh per year"

        except Exception as e:
            predicted_salary = f"Error: {e}"

    return render_template("index.html", predicted_salary=predicted_salary)


if __name__ == "__main__":
    app.run(debug=True)
