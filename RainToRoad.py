from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

app = Flask(__name__)

# =========================================================
# 1. ADDIS ABABA ROUTES / SEFERS
# =========================================================

ROUTES = {
    "Addis Ketema": [
        "Addis Ketema", "Merkato", "Amanuel Area", "American Gibi",
        "Aserasement", "Atekelet Tera", "Autobus Tera",
        "Berberee Berenda", "Bomb Tera", "Chew Berenda",
        "Chid Tera", "Cinima Ras", "Dubai Tera", "Minalesh Tera"
    ],

    "Akaki-Kality": [
        "Kality", "Saris", "Saris Abo"
    ],

    "Arada": [
        "Abacoran Sefer", "Arat Kilo", "Datsun Sefer",
        "Doro Manekiya", "Eri Bekentu", "Gedam Seffer",
        "Gojjam Berenda", "Habete Giorgis", "Kebena",
        "Piassa (piazza)", "Semien Mezegaja", "Sengatera",
        "Taliyan Sefer", "Tekelehaymanot"
    ],

    "Bole": [
        "Bole Airport", "Bole Ayat", "Bole Mikael", "Gerji",
        "Bole Medhanialem", "Megenagna", "Bole Ruwanda",
        "Gurd Shola", "Urael", "Wello Sefer", "Bole Japan"
    ],

    "Gulele": [
        "Abadina Area", "Addisu Gebeya", "Enqulal Faberika",
        "Paster", "Shero Meda"
    ],

    "Kera": [
        "Aroge Kera", "Bekelo Bet", "Kera Meberat",
        "Gotera", "Kera", "La Gare", "Mexico"
    ],

    "Kirkos": [
        "Ambassador", "Beherawi", "Bulgariya Mazoriya",
        "Lancha", "Meskel Flower", "Mobil",
        "Olympia", "Wolo Sefer"
    ],

    "Kolfe Keranio": [
        "Agusta", "Asko Area", "Asko Bercheko Faberika Area",
        "Atena Tera", "Ayertena", "Kolfe Keranyo",
        "Zenebework"
    ],

    "Lideta": [
        "Abenet", "Coca", "Darmar", "Geja Seffer",
        "Golla Mikael", "Goma Kuteba", "Goma Tera",
        "Mechare Meda", "Molla Maru", "Sarbet",
        "Sebategna", "Tor Hiylloch"
    ],

    "Nefas Silk Lafto": [
        "Besrat Gebriel", "Jemo", "Lafto",
        "Lebu", "Mekanisa"
    ],

    "Yeka": [
        "Aware", "Ayat", "Enderase", "Kazanchis", "Kotebe"
    ]
}


# =========================================================
# 2. COORDINATES
# =========================================================

COORDINATES = {
    "Bole Airport": (8.9779, 38.7993),
    "Megenagna": (9.0185, 38.8021),
    "Merkato": (9.0305, 38.7398),
    "Piassa (piazza)": (9.0350, 38.7520),
    "Sarbet": (8.9950, 38.7320),
    "Kazanchis": (9.0170, 38.7670),
    "Gotera": (8.9830, 38.7580),
    "Arat Kilo": (9.0330, 38.7630),
    "Jemo": (8.9450, 38.7110),
    "Ayat": (9.0250, 38.8650),
    "Kality": (8.8950, 38.7610),
    "Addisu Gebeya": (9.0620, 38.7420),
    "Mexico": (9.0100, 38.7470)
}


# =========================================================
# 3. DISTANCE CALCULATION
# =========================================================

def calculate_exact_distance(loc1, loc2):

    if loc1 in COORDINATES and loc2 in COORDINATES:

        lat1, lon1 = COORDINATES[loc1]
        lat2, lon2 = COORDINATES[loc2]

        earth_radius = 6371.0

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon / 2) ** 2
        )

        c = 2 * np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a)
        )

        return round(float(earth_radius * c), 2)

    return 7.2


# =========================================================
# 4. SYNTHETIC ML DATASET
# =========================================================

np.random.seed(42)

N = 8452

distances = np.random.uniform(1.0, 25.0, N)
traffic = np.random.uniform(10.0, 95.0, N)
rainfall = np.random.uniform(0.0, 30.0, N)
temperature = np.random.uniform(10.0, 32.0, N)
time_hour = np.random.uniform(6.0, 22.0, N)

road_types = np.random.choice(
    ["Arterial", "Secondary", "Highway", "Local"],
    size=N,
    p=[0.40, 0.25, 0.15, 0.20]
)

road_effect = {
    "Arterial": 0.0,
    "Secondary": 1.2,
    "Highway": -0.3,
    "Local": 2.2
}

road_effect_values = np.array(
    [road_effect[r] for r in road_types]
)

noise = np.random.normal(0, 2.5, N)

travel_times = (
    -5.7
    + (2.745 * distances)
    + (0.181 * traffic)
    + (0.309 * rainfall)
    + (0.026 * temperature)
    + (0.151 * time_hour)
    + road_effect_values
    + noise
)

travel_times = np.maximum(travel_times, 3.0)


# =========================================================
# 5. DATAFRAME
# =========================================================

df = pd.DataFrame({
    "distance": distances,
    "traffic": traffic,
    "rainfall": rainfall,
    "temperature": temperature,
    "time_hour": time_hour,
    "road_type": road_types,
    "travel_time": travel_times
})


# =========================================================
# 6. ONE-HOT ENCODE ROAD TYPE
# =========================================================

df_encoded = pd.get_dummies(
    df,
    columns=["road_type"],
    drop_first=True,
    dtype=float
)

FEATURE_COLUMNS = [
    "distance",
    "traffic",
    "rainfall",
    "temperature",
    "time_hour",
    "road_type_Highway",
    "road_type_Local",
    "road_type_Secondary"
]

X = df_encoded[FEATURE_COLUMNS]
y = df_encoded["travel_time"]


# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# 8. PRIMARY MODEL
# =========================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)


# =========================================================
# 9. PRIMARY MODEL METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

mae_display = round(mae, 2)
mse_display = round(mse, 2)
rmse_display = round(rmse, 2)
r2_display = round(r2, 2)


# =========================================================
# 10. ACTUAL VS PREDICTED DATA
# =========================================================

rng = np.random.default_rng(42)

sample_size = min(100, len(y_test))

sample_positions = rng.choice(
    len(y_test),
    size=sample_size,
    replace=False
)

actual_sample = y_test.iloc[sample_positions].values
pred_sample = y_pred[sample_positions]

actual_vs_pred = []

for actual, predicted in zip(
    actual_sample,
    pred_sample
):

    error = abs(actual - predicted)

    actual_vs_pred.append({
        "actual": round(float(actual), 2),
        "predicted": round(float(predicted), 2),
        "error": round(float(error), 2)
    })


# =========================================================
# 11. RESIDUALS
# =========================================================

residuals = []

for actual, predicted in zip(
    actual_sample,
    pred_sample
):

    residual = actual - predicted

    residuals.append({
        "predicted": round(float(predicted), 2),
        "residual": round(float(residual), 2)
    })


# =========================================================
# 12. PERMUTATION FEATURE IMPORTANCE
# =========================================================

permutation = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

importance_values = permutation.importances_mean

feature_importances = []

for feature, importance in zip(
    FEATURE_COLUMNS,
    importance_values
):

    feature_importances.append({
        "feature": feature,
        "importance": round(
            max(float(importance), 0.0),
            4
        )
    })

feature_importances.sort(
    key=lambda x: x["importance"],
    reverse=True
)


# =========================================================
# 13. MODEL COMPARISON
# =========================================================

comparison = []

comparison.append({
    "model": "Linear Regression",
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "r2": round(r2, 2)
})


ridge = Ridge(alpha=1.0)

ridge.fit(
    X_train,
    y_train
)

ridge_pred = ridge.predict(X_test)

ridge_mae = mean_absolute_error(
    y_test,
    ridge_pred
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ridge_pred
    )
)

ridge_r2 = r2_score(
    y_test,
    ridge_pred
)

comparison.append({
    "model": "Ridge Regression",
    "mae": round(ridge_mae, 2),
    "rmse": round(ridge_rmse, 2),
    "r2": round(ridge_r2, 2)
})


polynomial = Pipeline([
    (
        "poly",
        PolynomialFeatures(
            degree=2,
            include_bias=False
        )
    ),
    (
        "linear",
        LinearRegression()
    )
])

polynomial.fit(
    X_train,
    y_train
)

poly_pred = polynomial.predict(X_test)

poly_mae = mean_absolute_error(
    y_test,
    poly_pred
)

poly_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        poly_pred
    )
)

poly_r2 = r2_score(
    y_test,
    poly_pred
)

comparison.append({
    "model": "Polynomial (Degree 2)",
    "mae": round(poly_mae, 2),
    "rmse": round(poly_rmse, 2),
    "r2": round(poly_r2, 2)
})


# =========================================================
# 14. HELPER: BUILD MODEL INPUT
# =========================================================

def create_model_input(
    distance,
    traffic_value,
    rainfall_value,
    temperature_value,
    hour,
    road_type
):

    data = {
        "distance": distance,
        "traffic": traffic_value,
        "rainfall": rainfall_value,
        "temperature": temperature_value,
        "time_hour": hour,
        "road_type_Highway": 0.0,
        "road_type_Local": 0.0,
        "road_type_Secondary": 0.0
    }

    if road_type == "Highway":
        data["road_type_Highway"] = 1.0

    elif road_type == "Local":
        data["road_type_Local"] = 1.0

    elif road_type == "Secondary":
        data["road_type_Secondary"] = 1.0

    return pd.DataFrame(
        [data],
        columns=FEATURE_COLUMNS
    )


# =========================================================
# 15. DYNAMIC PREDICTION
# =========================================================

def compute_prediction(
    origin,
    destination,
    rainfall_val,
    temp_val,
    traffic_val,
    time_str,
    road_type
):

    distance_val = calculate_exact_distance(
        origin,
        destination
    )

    try:

        parts = time_str.split(":")

        hour = (
            float(parts[0])
            + float(parts[1]) / 60.0
        )

    except Exception:

        hour = 17.5

    input_data = create_model_input(
        distance=distance_val,
        traffic_value=traffic_val,
        rainfall_value=rainfall_val,
        temperature_value=temp_val,
        hour=hour,
        road_type=road_type
    )

    est_time = float(
        model.predict(input_data)[0]
    )

    est_time = max(
        est_time,
        3.0
    )

    est_time = round(
        est_time,
        1
    )

    normal_input = create_model_input(
        distance=distance_val,
        traffic_value=15.0,
        rainfall_value=0.0,
        temperature_value=temp_val,
        hour=hour,
        road_type=road_type
    )

    norm_time = float(
        model.predict(normal_input)[0]
    )

    norm_time = max(
        norm_time,
        3.0
    )

    norm_time = round(
        norm_time,
        1
    )

    delay = round(
        est_time - norm_time,
        1
    )

    return {
        "origin": origin,
        "destination": destination,
        "distance": distance_val,
        "est_time": est_time,
        "norm_time": norm_time,
        "delay": delay
    }


# =========================================================
# 16. ROAD TYPES
# =========================================================

ROAD_TYPES = [
    "Arterial",
    "Secondary",
    "Highway",
    "Local"
]


# =========================================================
# 17. HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        routes=ROUTES,
        road_types=ROAD_TYPES
    )


# =========================================================
# 18. ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# 19. WEB PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        origin = request.form.get(
            "origin",
            "Bole Airport"
        )

        destination = request.form.get(
            "destination",
            "Megenagna"
        )

        rainfall_val = float(
            request.form.get(
                "rainfall",
                8.4
            )
        )

        temp_val = float(
            request.form.get(
                "temperature",
                18.9
            )
        )

        traffic_val = float(
            request.form.get(
                "traffic",
                72
            )
        )

        time_str = request.form.get(
            "time_of_day",
            "17:30"
        )

        road_type = request.form.get(
            "road_type",
            "Arterial"
        )

        result = compute_prediction(
            origin,
            destination,
            rainfall_val,
            temp_val,
            traffic_val,
            time_str,
            road_type
        )

        return render_template(
            "result.html",
            origin=origin,
            destination=destination,
            est_time=result["est_time"],
            norm_time=result["norm_time"],
            delay=result["delay"],
            traffic=traffic_val,
            rainfall=rainfall_val,
            temperature=temp_val,
            distance=result["distance"]
        )

    except Exception as error:

        app.logger.exception(
            "Prediction error"
        )

        return (
            f"Prediction error: {error}",
            400
        )


# =========================================================
# 20. API PREDICTION
# =========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    try:

        data = request.get_json() or {}

        result = compute_prediction(

            data.get(
                "origin",
                "Bole Airport"
            ),

            data.get(
                "destination",
                "Megenagna"
            ),

            float(
                data.get(
                    "rainfall",
                    8.4
                )
            ),

            float(
                data.get(
                    "temperature",
                    18.9
                )
            ),

            float(
                data.get(
                    "traffic",
                    72
                )
            ),

            data.get(
                "time_of_day",
                "17:30"
            ),

            data.get(
                "road_type",
                "Arterial"
            )
        )

        return jsonify(result)

    except Exception as error:

        app.logger.exception(
            "API prediction error"
        )

        return jsonify({
            "error": str(error)
        }), 400


# =========================================================
# 21. ML INSIGHTS
# =========================================================

@app.route("/insights")
def insights():

    metrics = {

        "dataset_size": f"{N:,}",

        "train_size": f"{len(X_train):,}",

        "test_size": f"{len(X_test):,}",

        "mae": mae_display,

        "mse": mse_display,

        "rmse": rmse_display,

        "r2": r2_display,

        "variance_explained":
            f"{r2 * 100:.1f}"
    }


    # -----------------------------------------------------
    # Coefficients
    # -----------------------------------------------------

    regression_coefficients = []

    regression_coefficients.append({

        "feature": "Intercept",

        "coefficient": round(
            float(model.intercept_),
            3
        ),

        "interpretation":
            "Model baseline"
    })


    coefficient_descriptions = {

        "distance":
            "Travel time change per additional km",

        "traffic":
            "Travel time change per 1% traffic",

        "rainfall":
            "Travel time change per 1 mm rainfall",

        "temperature":
            "Travel time change per 1°C",

        "time_hour":
            "Travel time change per additional hour",

        "road_type_Highway":
            "Highway relative to Arterial",

        "road_type_Local":
            "Local road relative to Arterial",

        "road_type_Secondary":
            "Secondary road relative to Arterial"
    }


    for feature, coefficient in zip(
        FEATURE_COLUMNS,
        model.coef_
    ):

        regression_coefficients.append({

            "feature": feature,

            "coefficient": round(
                float(coefficient),
                3
            ),

            "interpretation":
                coefficient_descriptions.get(
                    feature,
                    "Model coefficient"
                )
        })


    # -----------------------------------------------------
    # Chart.js feature importance
    # -----------------------------------------------------

    importance_labels = [
        item["feature"]
        for item in feature_importances
    ]

    importance_scores = [
        item["importance"]
        for item in feature_importances
    ]

    feature_importance_chart = {
        "labels": importance_labels,
        "scores": importance_scores
    }


    # -----------------------------------------------------
    # Chart.js actual vs predicted
    # -----------------------------------------------------

    actual_chart_data = [
        {
            "x": index + 1,
            "y": item["actual"]
        }
        for index, item in enumerate(actual_vs_pred)
    ]

    predicted_chart_data = [
        {
            "x": index + 1,
            "y": item["predicted"]
        }
        for index, item in enumerate(actual_vs_pred)
    ]


    actual_predicted_chart = {
        "actual": actual_chart_data,
        "predicted": predicted_chart_data
    }


    # -----------------------------------------------------
    # Chart.js residual data
    # -----------------------------------------------------

    residual_chart_data = [
        {
            "x": item["predicted"],
            "y": item["residual"]
        }
        for item in residuals
    ]


    return render_template(

        "insights.html",

        metrics=metrics,

        regression_coefficients=regression_coefficients,

        actual_vs_pred=actual_predicted_chart,

        residuals=residual_chart_data,

        feature_importances=feature_importance_chart,

        comparison=comparison
    )


# =========================================================
# 22. RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )