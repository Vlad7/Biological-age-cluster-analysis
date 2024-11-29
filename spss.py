from savReaderWriter import SavReader

file_path = "datasets/HN07_ALL.sav"

with SavReader(file_path, ioUtf8=True) as reader:
    records = reader.all()  # Чтение данных
    header = reader.header  # Заголовки столбцов

print(header)
print(records[:5])  # Первые 5 строк

"""
import chardet
import pyreadstat

file_path = 'datasets/HN07_ALL.sav'

# Чтение небольшого фрагмента файла для анализа
with open(file_path, "rb") as f:
    rawdata = f.read(10000)  # Читаем первые 10КБ для анализа
    result = chardet.detect(rawdata)

print("Detected Encoding:", result['encoding'])

# Используем найденную кодировку
df, meta = pyreadstat.read_sav(file_path, encoding=result['encoding'])
"""
"""import chardet

# Определение кодировки файла
with open("datasets/HN07_ALL.sav", "rb") as f:
    result = chardet.detect(f.read())

print("Detected Encoding:", result['encoding'])
"""

"""
# Encoding: windows-1251

# you need pandas >= 0.25.0 for this
import pyreadstat

# Укажите путь к вашему файлу .sav
file_path = "datasets/HN07_ALL.sav"

# Чтение файла
df, meta = pyreadstat.read_sav(file_path, encoding_errors='ignore', encoding='windows-1251')
"""

"""import pandas as pd
df = pd.read_spss('Ldatasets/HN07_AL.sav')
print(df)"""