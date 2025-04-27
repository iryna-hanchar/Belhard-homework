import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class Time:
  def __init__(self, file_path, target_column, columns_drop=None, categorical_columns=None):
    self.file_path = file_path #путь к файлу
    self.target_column = target_column #целевая переменная
    self.columns_drop = columns_drop #список столбцов для удаления
    self.categorical_columns = categorical_columns #список категориальных столбцов
    self.data = None
    self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    self.model = None
    self.label_encoders = {}

  #загрузка данных
  def load_data(self, file_path=None):
    self.data = pd.read_csv (self.file_path)
    print ('Данные успешно загружены.')
    return self.data
  #предобработка данных
  def preprocess_data (self):
    #удаление неиспользуемых столбцов
    if self.columns_drop:
      self.data.drop(self.columns_drop, axis=1, inplace=True)
      print (f'Столбцы {self.columns_drop} удалены')
  
  #преобразование категориальных переменных в числовые
    for col in self.categorical_columns:
      if col in self.data.columns:
        le = LabelEncoder()
        self.data[col] = le.fit_transform(self.data[col])
        self.label_encoders[col] = le
        print (f'Столбец {col} преобразован в числовой формат.')

  #разделим на обучающую и тестовую выборки
  def split_data (self, test_size=0.3, random_state=42):
    X = self.data.drop(self.target_column, axis=1)
    y = self.data[self.target_column]
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)
    return self.X_train, self.X_test, self.y_train, self.y_test
  
  #обучение модели
  def train_model (self, **kwargs):
    self.model=LinearRegression(**kwargs)
    self.model.fit(self.X_train, self.y_train)
    print('Обучена линейная регрессия:')
    print (f'Коэффициенты: {self.model.coef_}')
    print (f'Пересечение: {self.model.intercept_}')
    return self.model

  #предсказание на новых данных
  def predict(self, X=None):
    if X is None:
      X=self.X_test
    predictions=self.model.predict(X)
    return predictions

  #оценка модели
  def evaluate_model (self):
    if self.model is None:
      raise ValueError ('Модель не обучена')
   
    y_pred = self.model.predict (self.X_test)
    mse = mean_squared_error (self.y_test, y_pred)
    r2 = r2_score (self.y_test, y_pred)
    print ('\nОценка регрессионной модели:')
    print (f'mse: {mse:.4f}')
    print (f'R^2: {r2:.4f}')

    return {'mse': mse, 'r2': r2}
  
  #визулизация
  def visualize_resul (self,figsize=(15,8), step=50):
   if self.model is None:
    raise ValueError("Модель не обучена")

   #y_pred = self.model.predict(self.X_test)
   
   plt.figure(figsize=(15, 8))
   n_show=40
   x_axis=range(n_show)
   plt.plot(x_axis, self.y_test.values[:n_show], 'b-', label='Истинное значение', linewidth=2)
   plt.plot(x_axis, self.y_pred[:n_show], 'g--', label='Предсказанное значение', linewidth=1.5)
   plt.title('Истинное vs Предсказанные значения')
   plt.xlabel('Индекс')
   plt.ylabel('Average Screen Time (hours)')
   plt.legend()
   plt.grid(True)
   plt.tight_layout()
   plt.show()
