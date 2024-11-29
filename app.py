from flask import Flask, render_template, request
import regression

app = Flask(__name__)

@app.route("/")
def home():
    greeting = regression.greet_user("John Doe")
    total = regression.calculate_sum(10, 20)

    return render_template("index.html", greeting=greeting, total=total)

@app.route("/calculate", methods=["POST"])
def calculate():
    # Get user inputs from the form
    num1 = int(request.form["num1"])
    num2 = int(request.form["num2"])
    
    # Use the calculate_sum function from utils.py
    result = regression.calculate_sum(num1, num2)
    
    # Pass the result back to the template
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)