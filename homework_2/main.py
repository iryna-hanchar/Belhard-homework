import hw2 as modul

file_path = 'Global_Space_Dataset.csv' #путь к файлу
target_column = 'Success Rate (%)' #целевая переменная
columns_drop = ['Country', 'Mission Name', 'Launch Site', 'Satellite Type', 'Technology Used', 'Collaborating Countries']  #столбцы для удаления
categorical_columns = {'Mission Type':{'Unmanned':0, 'Manned':1},'Environmental Impact': {'Low': 1, 'Medium': 2, 'High': 3}} #замена на числовые значения

model = modul.Mission (
    file_path = file_path,
    target_column = target_column,
    columns_drop = columns_drop,
    categorical_columns = categorical_columns
)

model.load_data(file_path)
model.preprocess_data()
model.split_data (test_size=0.3, random_state=50)
model.train_model()
metrics = model.evaluate_model()
model.visualize_results()

