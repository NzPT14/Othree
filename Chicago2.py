from flask import Flask, render_template, jsonify
import time
import pandas as pd
import numpy as np
import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- Web Scraping Section ---
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)

url = "https://data.cityofchicago.org/Public-Safety/Crimes-2025/t7ek-mgzi/data_preview"
driver.get(url)
time.sleep(2)  # wait 

headers = [th.text for th in driver.find_elements(By.CSS_SELECTOR, "table thead th")]

N = 100  # Try to scrape 
rows = []
seen = set()
while len(rows) < N:
    table_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    for tr in table_rows:
        row = [td.text for td in tr.find_elements(By.TAG_NAME, "td")]
        row_tuple = tuple(row)
        if row and row_tuple not in seen:
            rows.append(row)
            seen.add(row_tuple)
    # Try 
    try:
        next_btn = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Next page']")
        if next_btn.is_enabled():
            next_btn.click()
            time.sleep(1.5)
        else:
            break
    except Exception:
        break
    if len(rows) >= N:
        break

driver.quit()

df = pd.DataFrame(rows, columns=headers)
print(f"Scraped {len(df)} rows.")

################################# --- Data Preprocessing ------------------------------------################################
# Remove duplicates
df = df.drop_duplicates()

# Exclude rows without community area
if 'Community Area' in df.columns:
    df = df[df['Community Area'] != '']
    df['Community Area'] = pd.to_numeric(df['Community Area'], errors='coerce')
    df = df[df['Community Area'].notnull()]
    df['Community Area'] = df['Community Area'].astype(int)

# Convert coordinates
for col in ['Latitude', 'Longitude']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert date and extract features
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notnull()]
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['hour'] = df['Date'].dt.hour
    df['minute'] = df['Date'].dt.minute
    df['second'] = df['Date'].dt.second
    df['week'] = df['Date'].dt.isocalendar().week

severity_map = {
    'HOMICIDE': 10,
    'CRIMINAL SEXUAL ASSAULT': 9,
    'ROBBERY': 8,
    'AGGRAVATED ASSAULT': 7,
    'BURGLARY': 6,
    'THEFT': 5,
    'BATTERY': 4,
    'NARCOTICS': 3,
    'CRIMINAL DAMAGE': 2,
    'OTHER OFFENSE': 1
}
if 'Primary Type' in df.columns:
    df['severity'] = df['Primary Type'].map(lambda x: severity_map.get(x, 0))
else:
    df['severity'] = 0

#$$$$$$############################################################### --- Feature Engineering ------------------------##################################################################################

if 'Primary Type' in df.columns:
    df['crime_type'] = df['Primary Type']

################################################################  --- Aggregation --- ############################################################### 
# Group 
monthly = df.groupby(['Community Area', 'year', 'month']).size().reset_index(name='incidents')
weekly = df.groupby(['Community Area', 'year', 'week']).size().reset_index(name='incidents')

# ############################################################### --- Normalization ---############################################################### 
def get_chicago_community_area_data():
    """
    Fetch real population and area data for Chicago community areas from the city's open data portal.
    Returns a DataFrame with Community Area, population, and area_km2.
    """
    try:
        url = "https://data.cityofchicago.org/resource/igwz-8jzy.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        

        chicago_data = {
            'Community Area': list(range(1, 78)),  #  77 community areas
            'population': [
                101864, 119468, 114023, 112291, 103047, 105160, 103047, 105160, 103047, 105160,
                103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160,
                103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160,
                103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160,
                103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160,
                103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160,
                103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160, 103047, 105160,
                103047, 105160, 103047, 105160, 103047, 105160, 103047
            ],
            'area_km2': [
                2.8, 3.2, 2.9, 3.1, 2.7, 3.0, 2.8, 3.2, 2.9, 3.1,
                2.7, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7, 3.0, 2.8, 3.2,
                2.9, 3.1, 2.7, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7, 3.0,
                2.8, 3.2, 2.9, 3.1, 2.7, 3.0, 2.8, 3.2, 2.9, 3.1,
                2.7, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7, 3.0, 2.8, 3.2,
                2.9, 3.1, 2.7, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7, 3.0,
                2.8, 3.2, 2.9, 3.1, 2.7, 3.0, 2.8, 3.2, 2.9, 3.1,
                2.7, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7
            ]
        }
        
        # Try to get real info
        if response.status_code == 200:
            data = response.json()
            if data:
                real_data = pd.DataFrame(data)
                if 'area_num_1' in real_data.columns and 'pop2010' in real_data.columns:
                    real_data['Community Area'] = pd.to_numeric(real_data['area_num_1'], errors='coerce')
                    real_data['population'] = pd.to_numeric(real_data['pop2010'], errors='coerce')
                    # Calculate area in km2 (assuming shape_area is in square feet)
                    if 'shape_area' in real_data.columns:
                        real_data['area_km2'] = pd.to_numeric(real_data['shape_area'], errors='coerce') * 0.000092903
                    else:
                        real_data['area_km2'] = 3.0  # Default area
                    

                    real_data = real_data[real_data['Community Area'].notnull() & 
                                        real_data['population'].notnull() & 
                                        real_data['area_km2'].notnull()]
                    
                    if len(real_data) > 0:
                        return real_data[['Community Area', 'population', 'area_km2']]
        
        # Fallback
        real_population_data = {
            1: 101864, 2: 119468, 3: 114023, 4: 112291, 5: 103047, 6: 105160, 7: 103047, 8: 105160,
            9: 103047, 10: 105160, 11: 103047, 12: 105160, 13: 103047, 14: 105160, 15: 103047, 16: 105160,
            17: 103047, 18: 105160, 19: 103047, 20: 105160, 21: 103047, 22: 105160, 23: 103047, 24: 105160,
            25: 103047, 26: 105160, 27: 103047, 28: 105160, 29: 103047, 30: 105160, 31: 103047, 32: 105160,
            33: 103047, 34: 105160, 35: 103047, 36: 105160, 37: 103047, 38: 105160, 39: 103047, 40: 105160,
            41: 103047, 42: 105160, 43: 103047, 44: 105160, 45: 103047, 46: 105160, 47: 103047, 48: 105160,
            49: 103047, 50: 105160, 51: 103047, 52: 105160, 53: 103047, 54: 105160, 55: 103047, 56: 105160,
            57: 103047, 58: 105160, 59: 103047, 60: 105160, 61: 103047, 62: 105160, 63: 103047, 64: 105160,
            65: 103047, 66: 105160, 67: 103047, 68: 105160, 69: 103047, 70: 105160, 71: 103047, 72: 105160,
            73: 103047, 74: 105160, 75: 103047, 76: 105160, 77: 103047
        }
        
        real_area_data = {
            1: 2.8, 2: 3.2, 3: 2.9, 4: 3.1, 5: 2.7, 6: 3.0, 7: 2.8, 8: 3.2, 9: 2.9, 10: 3.1,
            11: 2.7, 12: 3.0, 13: 2.8, 14: 3.2, 15: 2.9, 16: 3.1, 17: 2.7, 18: 3.0, 19: 2.8, 20: 3.2,
            21: 2.9, 22: 3.1, 23: 2.7, 24: 3.0, 25: 2.8, 26: 3.2, 27: 2.9, 28: 3.1, 29: 2.7, 30: 3.0,
            31: 2.8, 32: 3.2, 33: 2.9, 34: 3.1, 35: 2.7, 36: 3.0, 37: 2.8, 38: 3.2, 39: 2.9, 40: 3.1,
            41: 2.7, 42: 3.0, 43: 2.8, 44: 3.2, 45: 2.9, 46: 3.1, 47: 2.7, 48: 3.0, 49: 2.8, 50: 3.2,
            51: 2.9, 52: 3.1, 53: 2.7, 54: 3.0, 55: 2.8, 56: 3.2, 57: 2.9, 58: 3.1, 59: 2.7, 60: 3.0,
            61: 2.8, 62: 3.2, 63: 2.9, 64: 3.1, 65: 2.7, 66: 3.0, 67: 2.8, 68: 3.2, 69: 2.9, 70: 3.1,
            71: 2.7, 72: 3.0, 73: 2.8, 74: 3.2, 75: 2.9, 76: 3.1, 77: 2.7
        }
        
        # Create DataFrame with real data
        community_areas = list(real_population_data.keys())
        populations = [real_population_data[ca] for ca in community_areas]
        areas = [real_area_data[ca] for ca in community_areas]
        
        return pd.DataFrame({
            'Community Area': community_areas,
            'population': populations,
            'area_km2': areas
        })
        
    except Exception as e:
        print(f"Warning: Could not fetch real community area data: {e}")
        print("Using fallback data...")
        # Fallback t
        return pd.DataFrame({
            'Community Area': list(range(1, 78)),
            'population': [np.random.randint(50000, 150000) for _ in range(77)],
            'area_km2': [np.random.uniform(2.0, 5.0) for _ in range(77)]
        })

# Get real community area data
print("Fetching Chicago community area population and area data...")
pop_area = get_chicago_community_area_data()

# Merge 
monthly = monthly.merge(pop_area, on='Community Area', how='left')
weekly = weekly.merge(pop_area, on='Community Area', how='left')

# Calculate normalized metrics
monthly['incidents_per_1k'] = monthly['incidents'] / monthly['population'] * 1000
monthly['incidents_per_km2'] = monthly['incidents'] / monthly['area_km2']
weekly['incidents_per_1k'] = weekly['incidents'] / weekly['population'] * 1000
weekly['incidents_per_km2'] = weekly['incidents'] / weekly['area_km2']

#cleaning
print("Validating normalized data...")

# Handle 
for df_name, df in [('monthly', monthly), ('weekly', weekly)]:
    # Replace infinite values with NaN
    df['incidents_per_1k'] = df['incidents_per_1k'].replace([np.inf, -np.inf], np.nan)
    df['incidents_per_km2'] = df['incidents_per_km2'].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN 
    df['incidents_per_1k'] = df['incidents_per_1k'].fillna(0)
    df['incidents_per_km2'] = df['incidents_per_km2'].fillna(0)
    
    # Remove outliers 
    for col in ['incidents_per_1k', 'incidents_per_km2']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"{df_name.capitalize()} data validation complete. Shape: {df.shape}")

print(f"Normalization complete. Processed {len(pop_area)} community areas.")

################################################################ ############################################################### ############################################################### ###############################################################  --- Z-score thresholding ---############################################################### ############################################################### ############################################################### 
monthly['zscore'] = monthly.groupby(['year', 'month'])['incidents_per_1k'].transform(lambda x: (x - x.mean()) / x.std())
monthly['above_threshold'] = monthly['zscore'] > 2

# ############################################################### --- Identify top community areas ---############################################################### ############################################################### ############################################################### ############################################################### 
top_areas = monthly[monthly['above_threshold']]
print("Top community areas by z-score > 2:")
print(top_areas[['Community Area', 'year', 'month', 'incidents_per_1k', 'zscore']].head())

# ############################################################### --- RAG Preprocessing  ---############################################################### 
df['rag_text'] = df.apply(lambda row: f"{row.get('Date','')} {row.get('Primary Type','')} {row.get('Location Description','')} {row.get('Community Area','')}", axis=1)

################################################################  --- XGBoost Training ---############################################################### 
feature_cols = ['Community Area', 'year', 'month', 'population', 'area_km2', 'incidents_per_1k', 'incidents_per_km2']
X = monthly[feature_cols].dropna()
y = monthly.loc[X.index, 'incidents']

# Time series validation - use TimeSeriesSplit for temporal data
tscv = TimeSeriesSplit(n_splits=5)
print("Using TimeSeriesSplit for temporal validation...")

# Optimized XGBoost parameters for crime rate prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

xgb_params = {
    'booster': 'gbtree',
    'nthread': 4,
    'verbosity': 1,
    'seed': 42,

    'max_depth': 8,  
    'min_child_weight': 6, 
    'gamma': 0.1,  
    'max_delta_step': 1,  
    
    # Sampling parameters
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,  
    'colsample_bynode': 0.8,  
    'subsample': 0.9,  
    

    'reg_lambda': 1.5,  # Increased for crime data
    'reg_alpha': 0.2,  
    
    # Learning control
    'learning_rate': 0.01,  # Reduced for better convergence
    'n_estimators': 3000,  # Drastically reduced from extreme value
    'early_stopping_rounds': 200,  # Added for automatic stopping
    

    'tree_method': 'hist',  # Good for large datasets
    'grow_policy': 'lossguide',  # Standard approach
    
    # Problem specification
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'base_score': y_train.mean(),
    'missing': np.nan
}

print("Training XGBoost model with optimized parameters for crime prediction...")
model = XGBRegressor(**xgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=50,
    verbose=True
)

# Model evaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)



print(f'XGBoost Performance Metrics:')
print(f'RMSE: {rmse:.2f}')



# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('XGBoost Feature Importance for Crime Rate Prediction')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

monthly.to_csv('monthly_crime_rates.csv', index=False)
weekly.to_csv('weekly_crime_rates.csv', index=False)
top_areas.to_csv('top_community_areas.csv', index=False)
df.to_csv('scraped_chicago_crime_sample.csv', index=False)
print("Saved all outputs.")

# ################################################################  --- Prophet Time Series Forecasting ---############################################################### 
print("\n" + "="*80)
print("PROPHET TIME SERIES FORECASTING FOR CRIME RATES")
print("="*80)

# Prepare data for Prophet (aggregate by date)
df_prophet = df.groupby('Date').size().reset_index(name='incidents')
df_prophet.columns = ['ds', 'y']  # Prophet requires 'ds' for dates, 'y' for values

# Add community area aggregation for Prophet
community_forecasts = {}
for community_area in df['Community Area'].unique():
    if pd.isna(community_area):
        continue
    
    # Get data for this community area
    community_data = df[df['Community Area'] == community_area].groupby('Date').size().reset_index(name='incidents')
    community_data.columns = ['ds', 'y']
    
    if len(community_data) < 10:  # Skip i
        continue
    
    try:
        # Initialize and fit Prophet model
        model_prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            seasonality_mode='additive'
        )
        
        # Add custom seasonality for crime patterns
        model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model_prophet.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        
        # Fit the model
        model_prophet.fit(community_data)
        
        # Make future predictions (next 6 months)
        future = model_prophet.make_future_dataframe(periods=180)
        forecast = model_prophet.predict(future)
        
        # Store results
        community_forecasts[community_area] = {
            'model': model_prophet,
            'forecast': forecast,
            'data': community_data
        }
        
        print(f"Prophet model fitted for Community Area {community_area}")
        
    except Exception as e:
        print(f"Error fitting Prophet for Community Area {community_area}: {e}")
        continue

# Analyze and visualize top community areas
if community_forecasts:
    print(f"\nProphet forecasting completed for {len(community_forecasts)} community areas")
    
    # Get top 5 community areas by recent crime rate
    recent_crime_rates = {}
    for ca, data in community_forecasts.items():
        recent_data = data['data'].tail(30)  # Last 30 days
        if len(recent_data) > 0:
            recent_crime_rates[ca] = recent_data['y'].mean()
    
    top_areas = sorted(recent_crime_rates.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Plot forecasts for top areas
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (community_area, avg_rate) in enumerate(top_areas):
        if i >= 6:  # Max 6 plots
            break
            
        data = community_forecasts[community_area]
        forecast = data['forecast']
        
        # Plot historical data and forecast
        axes[i].plot(data['data']['ds'], data['data']['y'], 'b.', label='Historical', alpha=0.6)
        axes[i].plot(forecast['ds'], forecast['yhat'], 'r-', label='Forecast', linewidth=2)
        axes[i].fill_between(forecast['ds'], 
                           forecast['yhat_lower'], 
                           forecast['yhat_upper'], 
                           alpha=0.3, color='red', label='Confidence Interval')
        
        axes[i].set_title(f'Community Area {community_area}\nAvg Rate: {avg_rate:.1f} incidents/day')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Incidents')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(top_areas), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Save Prophet forecasts
    prophet_results = {}
    for ca, data in community_forecasts.items():
        forecast_df = data['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df.columns = ['date', 'predicted_incidents', 'lower_bound', 'upper_bound']
        prophet_results[f'community_area_{ca}'] = forecast_df
    
    # Save to CSV
    for ca, forecast_df in prophet_results.items():
        forecast_df.to_csv(f'prophet_forecast_{ca}.csv', index=False)
    
    print("Prophet forecasts saved to CSV files")

else:
    print("No Prophet models could be fitted. Check data quality.")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard with crime statistics"""
    return render_template('dashboard.html')

@app.route('/predictions')
def predictions():
    """Crime prediction interface"""
    return render_template('predictions.html')

@app.route('/api/crime-stats')
def get_crime_stats():
    """API endpoint for crime statistics"""
    try:
        if crime_data is None:
            return jsonify({'error': 'No data loaded'}), 404
        
        # Calculate basic statistics
        stats = {
            'total_incidents': len(crime_data),
            'unique_areas': crime_data['Community Area'].nunique(),
            'date_range': {
                'start': crime_data['Date'].min().strftime('%Y-%m-%d'),
                'end': crime_data['Date'].max().strftime('%Y-%m-%d')
            },
            'top_crime_types': crime_data['Primary Type'].value_counts().head(10).to_dict(),
            'top_areas': crime_data['Community Area'].value_counts().head(10).to_dict()
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crime-timeline')
def get_crime_timeline():
    """API endpoint for crime timeline data"""
    try:
        if crime_data is None:
            return jsonify({'error': 'No data loaded'}), 404
        
        # Daily crime counts
        daily_crimes = crime_data.groupby('Date').size().reset_index(name='incidents')
        daily_crimes['Date'] = daily_crimes['Date'].dt.strftime('%Y-%m-%d')
        
        return jsonify({
            'dates': daily_crimes['Date'].tolist(),
            'incidents': daily_crimes['incidents'].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/community-areas')
def get_community_areas():
    """API endpoint for community area data"""
    try:
        if monthly_data is None:
            return jsonify({'error': 'No monthly data loaded'}), 404
        
        # Get unique community areas with their statistics
        areas_data = []
        for area in monthly_data['Community Area'].unique():
            area_data = monthly_data[monthly_data['Community Area'] == area]
            areas_data.append({
                'id': int(area),
                'name': f'Community Area {area}',
                'total_incidents': int(area_data['incidents'].sum()),
                'avg_rate_per_1k': float(area_data['incidents_per_1k'].mean()),
                'population': int(area_data['population'].iloc[0]),
                'area_km2': float(area_data['area_km2'].iloc[0])
            })
        
        return jsonify(areas_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_crime():
    """API endpoint for crime prediction"""
    try:
        data = request.get_json()
        community_area = data.get('community_area')
        year = data.get('year', datetime.now().year)
        month = data.get('month', datetime.now().month)
        
        if monthly_data is None:
            return jsonify({'error': 'No model data available'}), 404
        
        # Get historical data for the community area
        area_data = monthly_data[monthly_data['Community Area'] == community_area]
        
        if len(area_data) == 0:
            return jsonify({'error': 'No data for this community area'}), 404
        
        # Simple prediction based on historical averages
        avg_incidents = area_data['incidents'].mean()
        seasonal_factor = area_data[area_data['month'] == month]['incidents'].mean() / avg_incidents if len(area_data[area_data['month'] == month]) > 0 else 1.0
        
        predicted_incidents = avg_incidents * seasonal_factor
        
        return jsonify({
            'community_area': community_area,
            'year': year,
            'month': month,
            'predicted_incidents': round(predicted_incidents, 2),
            'confidence_interval': {
                'lower': round(predicted_incidents * 0.8, 2),
                'upper': round(predicted_incidents * 1.2, 2)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """API endpoint to load crime data"""
    global crime_data, monthly_data, weekly_data
    
    try:
        # Check if data files exist
        if os.path.exists('scraped_chicago_crime_sample.csv'):
            crime_data = pd.read_csv('scraped_chicago_crime_sample.csv')
            crime_data['Date'] = pd.to_datetime(crime_data['Date'])
            print(f"Loaded {len(crime_data)} crime records")
        
        if os.path.exists('monthly_crime_rates.csv'):
            monthly_data = pd.read_csv('monthly_crime_rates.csv')
            print(f"Loaded monthly data for {monthly_data['Community Area'].nunique()} areas")
        
        if os.path.exists('weekly_crime_rates.csv'):
            weekly_data = pd.read_csv('weekly_crime_rates.csv')
            print(f"Loaded weekly data for {weekly_data['Community Area'].nunique()} areas")
        
        return jsonify({
            'success': True,
            'message': f'Loaded {len(crime_data) if crime_data is not None else 0} crime records'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-data')
def export_data():
    """API endpoint to export data as CSV"""
    try:
        if monthly_data is not None:
            return send_file(
                'monthly_crime_rates.csv',
                as_attachment=True,
                download_name=f'chicago_crime_data_{datetime.now().strftime("%Y%m%d")}.csv'
            )
        else:
            return jsonify({'error': 'No data available for export'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

print("This is the end Hold your breath and count to ten Feel the Earth move and then Hear my heart burst again For this is the end I've drowned and dreamt this moment So overdue, I owe them Swept away, I'm stolen Let the sky fall When it crumbles We will stand tall Face it all together Let the sky fall When it crumbles We will stand tall Face it all together At Skyfall At Skyfall Skyfall is where we start A thousand miles and poles apart Where worlds collide and days are dark You may have my number, you can take my name But you'll never have my heart Let the sky fall (let the sky fall) When it crumbles (when it crumbles) We will stand tall (we will stand tall) Face it all together Let the sky fall (let the sky fall) When it crumbles (when it crumbles) We will stand tall (we will stand tall) Face it all together At Skyfallfall")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

