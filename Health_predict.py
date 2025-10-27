import joblib
import numpy as np
from flask import Flask, request, render_template_string

# Initialize the Flask application
app = Flask(__name__)

# --- Load Your Model ---
# This loads the model file you uploaded.
# Ensure 'best_risk_level_model.joblib' is in the same directory as this app.py file.
try:
    model = joblib.load('best_risk_level_model.joblib')
except FileNotFoundError:
    print("Model file 'best_risk_level_model.joblib' not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Define the HTML Template ---
# We are embedding the HTML directly in our Python file for simplicity.
# This template includes a form for user input and a place to show the prediction.
# It's styled with Tailwind CSS for a clean, responsive look.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Patient Risk Level Predictor</title>
    <!-- Load Tailwind CSS for styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* A simple style for the prediction result */
        #prediction-result {
            transition: all 0.3s ease;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center font-sans p-4">

    <div class="bg-white p-6 sm:p-10 rounded-xl shadow-lg w-full max-w-3xl">
        
        <h1 class="text-2xl sm:text-3xl font-bold text-gray-800 mb-6 text-center">
            Patient Risk Level Predictor
        </h1>

        <!-- The form posts data to our /predict endpoint -->
        <form action="/predict" method="POST">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <!-- Numerical Inputs -->
                <div>
                    <label for="Respiratory_Rate" class="block mb-2 text-sm font-medium text-gray-700">Respiratory Rate (breaths/min)</label>
                    <input type="number" step="1" id="Respiratory_Rate" name="Respiratory_Rate" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500" placeholder="e.g., 20" required>
                </div>
                
                <div>
                    <label for="Oxygen_Saturation" class="block mb-2 text-sm font-medium text-gray-700">Oxygen Saturation (%)</label>
                    <input type="number" step="0.1" id="Oxygen_Saturation" name="Oxygen_Saturation" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500" placeholder="e.g., 98.5" required>
                </div>

                <div>
                    <label for="O2_Scale" class="block mb-2 text-sm font-medium text-gray-700">O2 Scale (L/min)</label>
                    <input type="number" step="0.1" id="O2_Scale" name="O2_Scale" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500" placeholder="e.g., 1.5" required>
                </div>

                <div>
                    <label for="Systolic_BP" class="block mb-2 text-sm font-medium text-gray-700">Systolic BP (mmHg)</label>
                    <input type="number" step="1" id="Systolic_BP" name="Systolic_BP" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500" placeholder="e.g., 120" required>
                </div>

                <div>
                    <label for="Heart_Rate" class="block mb-2 text-sm font-medium text-gray-700">Heart Rate (bpm)</label>
                    <input type="number" step="1" id="Heart_Rate" name="Heart_Rate" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500" placeholder="e.g., 80" required>
                </div>

                <div>
                    <label for="Temperature" class="block mb-2 text-sm font-medium text-gray-700">Temperature (°C)</label>
                    <input type="number" step="0.1" id="Temperature" name="Temperature" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500" placeholder="e.g., 36.6" required>
                </div>

                <!-- Categorical Inputs -->
                <div>
                    <label for="On_Oxygen" class="block mb-2 text-sm font-medium text-gray-700">On Oxygen?</label>
                    <select id="On_Oxygen" name="On_Oxygen" class="w-full p-3 border border-gray-300 rounded-lg bg-white focus:ring-blue-500 focus:border-blue-500">
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>

                <div>
                    <label for="Consciousness" class="block mb-2 text-sm font-medium text-gray-700">Consciousness Level</label>
                    <select id="Consciousness" name="Consciousness" class="w-full p-3 border border-gray-300 rounded-lg bg-white focus:ring-blue-500 focus:border-blue-500">
                        <option value="Alert">Alert (A)</option>
                        <option value="Confused">Confused (C)</option>
                        <option value="Voice">Responds to Voice (V)</option>
                        <option value="Pain">Responds to Pain (P)</option>
                        <option value="Unresponsive">Unresponsive (U)</option>
                    </select>
                </div>
            </div>

            <!-- Submit Button -->
            <div class="mt-8">
                <button type="submit" class="w-full bg-blue-600 text-white p-3 rounded-lg font-bold text-lg hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-300 transition-all duration-300 shadow-md">
                    Predict Risk Level
                </button>
            </div>
        </form>

        <!-- Prediction Result Area -->
        <!-- The {{ prediction_text }} part is where Flask will insert the result -->
        {% if prediction_text %}
        <div id="prediction-result" class="mt-6 text-center p-4 bg-blue-50 border-l-4 border-blue-500 rounded-lg">
            <p class="text-xl font-semibold text-blue-800">{{ prediction_text }}</p>
        </div>
        {% endif %}

        {% if error_text %}
        <div id="error-result" class="mt-6 text-center p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
            <p class="text-xl font-semibold text-red-800">{{ error_text }}</p>
        </div>
        {% endif %}

    </div>
</body>
</html>
"""

# --- Define Application Routes ---

@app.route('/')
def home():
    """Renders the home page with the input form."""
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text="Error: Model could not be loaded. Please check server logs.")
    # render_template_string formats our HTML string, inserting any variables we pass.
    return render_template_string(HTML_TEMPLATE, prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, processes input, and returns the prediction."""
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text="Error: Model is not loaded. Cannot make prediction.")

    try:
        # --- 1. Get Data from Form ---
        # Get numerical values and convert them to float
        resp_rate = float(request.form['Respiratory_Rate'])
        oxy_sat = float(request.form['Oxygen_Saturation'])
        o2_scale = float(request.form['O2_Scale'])
        sys_bp = float(request.form['Systolic_BP'])
        heart_rate = float(request.form['Heart_Rate'])
        temp = float(request.form['Temperature'])
        
        # --- 2. Preprocess Categorical Data ---
        # Convert 'On_Oxygen' (Yes/No) to binary (1/0)
        on_oxygen = 1 if request.form['On_Oxygen'] == 'Yes' else 0
        
        # Get the consciousness level
        consciousness = request.form['Consciousness']
        
        # One-hot encode the consciousness level based on the features the model expects
        # (Consciousness_A, Consciousness_C, Consciousness_P, Consciousness_U, Consciousness_V)
        # Note: The model's feature names seem to use 'P' for Pain and 'V' for Voice.
        consciousness_a = 1 if consciousness == 'Alert' else 0
        consciousness_c = 1 if consciousness == 'Confused' else 0
        consciousness_p = 1 if consciousness == 'Pain' else 0
        consciousness_u = 1 if consciousness == 'Unresponsive' else 0
        consciousness_v = 1 if consciousness == 'Voice' else 0

        # --- 3. Create Feature Array ---
        # The order MUST match the order the model was trained on:
        # ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale', 'Systolic_BP', 
        #  'Heart_Rate', 'Temperature', 'On_Oxygen', 'Consciousness_A', 
        #  'Consciousness_C', 'Consciousness_P', 'Consciousness_U', 'Consciousness_V']
        features = [
            resp_rate, oxy_sat, o2_scale, sys_bp, heart_rate, temp,
            on_oxygen, consciousness_a, consciousness_c, consciousness_p,
            consciousness_u, consciousness_v
        ]
        
        # Convert to a 2D numpy array, as scikit-learn models expect a 2D array
        final_features = [np.array(features)]
        
        # --- 4. Make Prediction ---
        prediction = model.predict(final_features)
        
        # Get the first (and only) prediction result
        output = prediction[0]
        
        # Format the output text
        prediction_result_text = f"Predicted Risk Level: {output}"

        # --- 5. Return Result ---
        # Render the same page, but this time with the prediction text
        return render_template_string(HTML_TEMPLATE, prediction_text=prediction_result_text)

    except Exception as e:
        # Handle any errors during prediction
        print(f"Error during prediction: {e}")
        return render_template_string(HTML_TEMPLATE, error_text=f"Error processing input. Please check values and try again.")

# --- Run the Application ---
if __name__ == "__main__":
    # Setting host='0.0.0.0' makes the app accessible on your local network
    # You can access it by going to http://127.0.0.1:5000 or http://localhost:5000 in your browser
    app.run(debug=True, host='0.0.0.0', port=5000)
