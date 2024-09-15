from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib



# Initialize the Flask application
app = Flask(__name__)

file_path = 'train.csv'
train = pd.read_csv(file_path)
print(train.head())

# Define the directory where uploaded files will be saved
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Random Forest model
model_path = 'pkl/LGBM_model.pkl'
random_forest_model = joblib.load(model_path)


import pandas as pd

def calculate_means(df, x):
    """
    Calculate the mean values of specified columns for rows where winPlacePerc is within [x, x + 0.05].

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x (float): The reference value for winPlacePerc.

    Returns:
    pd.Series: Mean values of the specified columns.
    """
    # Filter rows where winPlacePerc is in the range [x, x + 0.05]
    filtered_df = df[(df['winPlacePerc'] >= x) & (df['winPlacePerc'] <= x + 0.005)]
    
    # Columns to calculate the mean
    columns_to_mean = [
        'walkDistance',
        'killPlace',
        'total_distance',
        'boosts',
        'HealsPerDist',
        'killP/maxP',
        'total_team_damage',
        'total_kills_by_team',
        'weaponsAcquired',
        'Heals_Boosts',
        'KillsPerDist',
        'matchDuration',
        'rideDistance',
        'kills',
        'longestKill'
    ]
    
    # Calculate mean values for the specified columns
    mean_values = filtered_df[columns_to_mean].mean()
    
    return mean_values






from sklearn.impute import SimpleImputer

def preprocessing(test):
    test = test.fillna(0)
    test['player_played'] = test.groupby('matchId')['matchId'].transform('count')
    test['total_distance'] = test.rideDistance + test.walkDistance + test.swimDistance
    test['kills_without_moving'] = ((test['kills'] > 0) & (test['total_distance'] == 0))
    test.drop(test[test['kills_without_moving'] == True].index, inplace=True)
    test.drop(['kills_without_moving'], axis=1, inplace=True)
    test['players_joined'] = test.groupby('matchId')['matchId'].transform('count')
    test['players_in_a_team'] = test.groupby('groupId').groupId.transform('count')
    team_mapper = lambda x: 1 if ('solo' in x) else 2 if ('duo' in x) else 4
    test["max_team"] = test['matchType'].apply(team_mapper)
    test['a'] = test["players_in_a_team"] > test["max_team"]
    test.drop(test[test['a'] == True].index, inplace=True)
    test.drop(['a'], axis=1, inplace=True)
    test.drop(test[test['roadKills'] >= 10].index, inplace=True)
    test.drop(test[test['kills'] >= 35].index, inplace=True)
    test.drop(test[test['longestKill'] >= 1000].index, inplace=True)
    test.drop(test[test['walkDistance'] >= 10000].index, inplace=True)
    test.drop(test[test.rideDistance >= 15000].index, inplace=True)
    test.drop(test[test.swimDistance >= 1000].index, inplace=True)
    test.drop(test[test.total_distance >= 15000].index, inplace=True)
    test.drop(test[test.weaponsAcquired >= 50].index, inplace=True)
    test.drop(test[test.heals >= 40].index, inplace=True)
    test = pd.get_dummies(test, columns=['matchType'])

    return test

def feature_engineering(data1):
    ratio = data1['headshotKills'] / data1['kills'].replace(0, 1)
    data1['headshot/kill'] = ratio

    AR = data1['assists'] + data1['revives']
    data1['Assist_Revive'] = AR

    KWalkDist = data1['kills'] / (data1['walkDistance'].replace(0, 1) + 1)
    data1['KillsPerDist'] = KWalkDist

    hb = data1['heals'] + data1['boosts']
    data1['Heals_Boosts'] = hb

    HWalkDist = data1['Heals_Boosts'] / (data1['walkDistance'].replace(0, 1) + 1)
    data1['HealsPerDist'] = HWalkDist

    killmax = data1['killPlace'] / (data1['maxPlace'].replace(0, 1) + 1)
    data1['killP/maxP'] = killmax

    data1['total_team_damage'] = data1.groupby('groupId')['damageDealt'].transform('sum')
    data1['total_kills_by_team'] = data1.groupby('groupId')['kills'].transform('sum')

    data1.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)

    return data1

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save the uploaded file

        # Process the CSV file
        data = pd.read_csv(file_path)  # Load the CSV using pandas

        # Preprocessing and feature engineering
        data = preprocessing(data)
        data = feature_engineering(data)

        # Select relevant columns for prediction
        cols_15 = ['walkDistance', 'killPlace', 'total_distance', 'boosts',
                   'HealsPerDist', 'killP/maxP', 'total_team_damage',
                   'total_kills_by_team', 'weaponsAcquired', 'Heals_Boosts',
                   'KillsPerDist', 'matchDuration', 'rideDistance', 'kills',
                   'longestKill']

        # Ensure columns are present in the data
        missing_cols = [col for col in cols_15 if col not in data.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {', '.join(missing_cols)}"}), 400

        # Get the features needed for the model
        input_data = data[cols_15]

        # Use the model to predict
        predictions = random_forest_model.predict(input_data)
        data['predictions'] = predictions
        # print('hi')
        print(predictions)
        # mean_values = {}

        # Convert the results to a dictionary or JSON
        results = data[['predictions']].to_dict(orient='records')
        additional_data = data.drop(columns='predictions').to_dict(orient='records')
        mean_values = calculate_means(train, predictions[0])
        print(mean_values)
        
        return render_template('result.html', predictions=results, data=additional_data, mean_values=mean_values)
    else:
        return jsonify({"error": "Invalid file format. Only CSV files are allowed."}), 400


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
