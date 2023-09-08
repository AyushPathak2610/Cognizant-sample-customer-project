import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(f"{path}")
    return df

def create_X_y(data,target):    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def model_train_And_prediction(X,y):
    model = RandomForestRegressor()
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    trained_model = model.fit(X_train, y_train)
    y_pred = trained_model.predict(X_test)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    print(mae)
    
def main():
    df=load_data("path/to/csv/")
    X,y = create_X_y(df,"estimated_stock_pct")
    model_train_And_prediction(X,y)