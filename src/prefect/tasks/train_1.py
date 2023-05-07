import optuna
import pandas as pd
from prefect import task
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

from typing import Tuple, Dict

def get_preprocess_transformer(data: pd.DataFrame) -> ColumnTransformer:
    
    num_columns = data.select_dtypes(exclude=[object]).columns.values
    cat_columns = data.select_dtypes(include=[object]).columns.values
    
    print("num_columns: ", num_columns)
    print("cat_columns: ", cat_columns)
    
    preprocesser = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_columns)
        ]
    )
    
    return preprocesser

@task(log_stdout=True, nout=2)
def load_dataset_task() -> Tuple[Tuple[pd.DataFrame], Tuple[pd.DataFrame]]:
    
    sql = "SELECT * FROM apartments WHERE transaction_real_price IS NOT NULL LIMIT 5000"
    
    engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
    
    data = pd.read_sql(sql, con=engine)
    label = data['transaction_real_price']
    
    data.drop(columns=['apartment_id', 'transaction_id', 'transaction_real_price', 'jibun', 'apt', 'addr_kr', 'dong'], axis=1, inplace=True)
    
    x_train, x_valid, y_train, y_valid = train_test_split(
        data, label, test_size=0.2, random_state=42
    )
    
    return (x_train, y_train), (x_valid, y_valid)


@task(log_stdout=True, nout=3)
def hpo_task(train: Tuple[pd.DataFrame, pd.DataFrame], 
             valid: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[ColumnTransformer, Dict, float]:
    x_train, y_train = train
    x_valid, y_valid = valid
    
    preprocesser = get_preprocess_transformer(data=x_train)
    
    x_train = preprocesser.fit_transform(x_train)
    x_valid = preprocesser.transform(x_valid)
    
    print("HPO Start")
    
    def objectiveModel(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree": trial.suggest_float("subsample", 0.5, 1),
        }
        
        model = XGBRegressor(**params) # = XGBRegressor(max_depth=params["max_depth"]....)
        
        model.fit(x_train, y_train,
                  eval_metric="mae",
                  eval_set=[(x_valid, y_valid)],
                  early_stopping_rounds=10,
                  verbose=0
                  )

        valid_score = mean_absolute_error(y_valid, model.predict(x_valid))
        
        return valid_score
    
    storage = optuna.storages.RDBStorage(url="postgresql://postgres:postgres@localhost:5432/optuna")
    
    study = optuna.create_study(direction="minimize", storage=storage)
    study.optimize(objectiveModel, n_trials=10)
    
    print("Best params: ", study.best_params, "Best MAE: ", study.best_value)
    
    return preprocesser, study.best_params, study.best_value

from sklearn.pipeline import Pipeline

@task(log_stdout=True)
def train_task(preprocesser: ColumnTransformer, 
               train: Tuple[pd.DataFrame, pd.DataFrame],
               valid: Tuple[pd.DataFrame, pd.DataFrame],
               best_params: Dict,) -> Pipeline:
    x_train, y_train = train
    x_valid, y_valid = valid
    
    x_train = preprocesser.transform(x_train)
    x_valid = preprocesser.transform(x_valid)
    
    model = XGBRegressor(**best_params)
    
    model.fit(x_train, y_train,
              eval_metric="mae",
              eval_set=[(x_valid, y_valid)],
              early_stopping_rounds=10,
              verbose=0)
    
    print("Training Done.", "Best MAE: ", mean_absolute_error(y_valid, model.predict(x_valid)))
    
    pipeline = Pipeline(steps=[("preprocessor", preprocesser), ("model", model)])
    
    return pipeline