from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# -------- TRAIN MODEL ONCE --------
df = pd.read_csv("Housing.csv")

X = df[['area','bedrooms','bathrooms','stories','mainroad','guestroom',
        'basement','hotwaterheating','airconditioning','parking',
        'prefarea','furnishingstatus']].copy()
y = df['price']

yes_no = ['mainroad','guestroom','basement',
          'hotwaterheating','airconditioning','prefarea']
for col in yes_no:
    X[col] = X[col].map({'yes':1,'no':0})

X = pd.get_dummies(X, columns=['furnishingstatus'], drop_first=True)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
# ---------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        # -------- INPUTS FROM HTML --------
        area = int(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        parking = int(request.form["parking"])

        mainroad = request.form["mainroad"]
        guestroom = request.form["guestroom"]
        basement = request.form["basement"]
        hotwater = request.form["hotwater"]
        aircon = request.form["aircon"]
        prefarea = request.form["prefarea"]
        furnishing = request.form["furnishing"]

        # -------- BUILD DATAFRAME --------
        user_df = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': 1 if mainroad=="yes" else 0,
            'guestroom': 1 if guestroom=="yes" else 0,
            'basement': 1 if basement=="yes" else 0,
            'hotwaterheating': 1 if hotwater=="yes" else 0,
            'airconditioning': 1 if aircon=="yes" else 0,
            'parking': parking,
            'prefarea': 1 if prefarea=="yes" else 0,
            'furnishingstatus': furnishing
        }])

        user_df = pd.get_dummies(user_df)
        user_df = user_df.reindex(columns=X.columns, fill_value=0)

        price = 1.5*model.predict(user_df)[0]

        if price >= 1e7:
            result = f"Estimated Price: ₹{price:,.2f} (~{price/1e7:.2f} Cr)"
        else:
            result = f"Estimated Price: ₹{price:,.2f} (~{price/1e5:.2f} Lakh)"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000,debug=True)

