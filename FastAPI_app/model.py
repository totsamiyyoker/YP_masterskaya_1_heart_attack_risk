import dill
import pandas as pd
from pathlib import Path


class Model:
    def __init__(self):
        model_path = Path(__file__).parent / 'best_pipe.pkl'
        with open(model_path, 'rb') as f:
            self.pipeline = dill.load(f)
        # Признаки, на которых обучалась модель
        self.feature_cols = self.pipeline.named_steps['model'].feature_names_
        # Признаки в формате index: feature, для отбора нужных из передаваемых данных
        self.feature_map = self.pipeline.feature_map
        # Названия всех признаков в формате snake_case
        self.snake_case = self.pipeline.snake_cols
        # Порог
        self.threshold = self.pipeline.best_threshold
    
    def predict_row(self, df: pd.DataFrame) -> int:
        """
        Выполняет предсказание по одной переданной строке.
        """
        try:
            prob = self.pipeline.predict_proba(df)[:, 1]
            pred = (prob >= self.threshold).astype(int)  
            return int(pred[0])
        except Exception as e:
            raise e
        
    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет предсказание по датафрейму, получаемому из CSV-файла.
        """
        try:
            probs = self.pipeline.predict_proba(df)[:, 1]
            preds = (probs >= self.threshold).astype(float)
            df_result = pd.DataFrame({'id': df.index.values, 'prediction': preds})
            return df_result
        except Exception as e:
            return ValueError(f'Ошибка при предсказании: {e}')