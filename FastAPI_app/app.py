import argparse
import logging

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, Request, Form, File
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model import Model


app = FastAPI()

app.mount('/tmp', StaticFiles(directory='tmp'), name='images')
templates = Jinja2Templates(directory='templates')

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

# Инициализиурем модель с лучшим порогом
model = Model()

# Endpoint с проверкой работы приложения
@app.get('/health')
def health():
    return {'status': "OK"}

# Игнорирование запросов chrome
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    # Игнорируем специфические chrome/devtools запросы
    if str(request.url).endswith("chrome.devtools.json"):
        return PlainTextResponse("Ignored", status_code=200)
    return PlainTextResponse("Not Found", status_code=404)

# GET endpoint: загружает главную страницу
@app.get('/')
def main(request: Request):
    return templates.TemplateResponse('start_form.html',
                                      {'request': request})

# POST endpoint: выполняет предсказание по переданной одной строке 
@app.post('/predict_row')
def predict_row(row: str = Form(...)): 
    """
    Выполняет предсказание для переданной строки.
    """ 
    # Получаем словарь входных признаков с индексами
    feature_map = model.feature_map
    # Получаем признаки
    feature_cols = model.feature_cols

    # Разделяем строку по запятым
    values = [x.strip() for x in row.split(',')]

    # Достаём id 
    patient_id = values[-1]
    # Удаляем из строки индекс и id
    values = values[1:-1]

    # Заменяем пустые строки на NaN
    values = [np.nan if v == '' else v for v in values]

    # Составляем словарь {индекс: значение}
    raw_data = {i: val for i, val in enumerate(values)}
    # Оставляем только признаки, которые были использованы при обучении модели
    selected_values = [raw_data[i] for i in feature_map.keys()]

    # Создаём датафрейм из строки
    df = pd.DataFrame([selected_values], columns=feature_cols)
    # Столбец gender имеет тип object, оставляем только количественные столбцы
    num_cols = [c for c in feature_cols if c != 'gender']
    # Кодируем количественные столбцы как тип float64
    df[num_cols] = df[num_cols].astype('float64')

    # Выполняем предсказание 
    pred = model.predict_row(df)

    return {'id': patient_id, 'prediction': pred}


# POST endpoint: выполняет предсказание по переданному CSV-файлу
@app.post('/predict_csv')
async def predict_csv(file: UploadFile = File(...)):
    """
    Выполняет предсказание на переданном csv файле.
    Позволяет скачать полученный csv файл с предсказаниями,
    а также передаёт в /predictions_json предсказания для вывода в формате JSON.
    """
    try:
        if not file.filename.endswith('.csv'):
            return {'status': 'ERROR', 'message': 'Файл должен быть в формате CSV'}
        
        df = pd.read_csv(file.file, index_col=0)
        # Делаем столбец `id` индексом
        df.set_index('id', inplace=True)
        # Приводим названия к snake_case
        df.columns = model.snake_case

        # Выполняем проверку, что CSV-файл имеет такую же структуру, как в тестовой выборке
        required_cols = set(model.snake_case)
        uploaded_cols = set(df.columns)
        missing = required_cols - uploaded_cols
        if missing:
            return {'status': 'ERROR', 'message': f'Отсутствуют столбцы: {", ".join(missing)}'}

        # Оставляем только признаки, на которых обучалась модель
        df = df[model.feature_cols]

        # Выполняем предсказание и сохраняем в файл
        df_pred = model.predict_df(df)
        df_pred.to_csv('tmp/predictions.csv', index=False)

        # Создаём глобальную переменную для вывода в формате JSON
        global predictions_store
        predictions_store = df_pred.set_index('id')['prediction'].to_dict()
        return {'status': 'OK', 'result_path': 'tmp/predictions.csv'}
    except Exception as e:
        return {'status': 'ERROR', 'message': str(e)}

# GET endpoint: выводит на экран предсказания вида id: prediction в формате JSON
@app.get('/predictions_json')
def get_predictions_json():
    """
    Выводит информацию в формате JSON на новой странице,
    если таковая была передана.
    """
    if predictions_store is None:
        return {'status': 'ERROR', 'message': 'Нет предсказаний'}
    return {'status': 'OK', 'predictions': predictions_store}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8010, type=int, dest='port')
    parser.add_argument('--host', default='0.0.0.0', type=str, dest='host')
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
