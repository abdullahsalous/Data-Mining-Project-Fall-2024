from flask import Flask, render_template
import regression

app = Flask(__name__)

@app.route("/")
def home():
    total = regression.calculate_sum(10, 20)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)