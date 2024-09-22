import pandas as pd
import requests  # Для выполнения HTTP-запросов к API
import time  # Для добавления задержки между повторными попытками запросов
import re  # Импортируем модуль для работы с регулярными выражениями
import datetime as dt
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer # Метод для анализа слов

# Функция для получения географических координат (широта и долгота) с API OpenCage
def get_location_from_api(address, api_key, retries=3, delay=5):
    """
    Выполняет запрос к API OpenCage для получения широты и долготы по адресу.
    
    Аргументы:
    address -- строковый адрес для получения координат
    api_key -- ключ API для доступа к OpenCage Geocoding API
    retries -- количество попыток запроса при неудаче (по умолчанию 3)
    delay -- задержка между попытками в секундах (по умолчанию 5)

    Возвращает:
    Кортеж (широта, долгота) или (None, None) в случае ошибки.
    """
    address = str(address)  # Преобразуем адрес в строку, если он не строкового типа
    for attempt in range(retries):  # Пытаемся выполнить запрос несколько раз
        try:
            # Выполняем GET-запрос к API OpenCage с указанным адресом и ключом API
            response = requests.get(f'https://api.opencagedata.com/geocode/v1/json?q={address}&key={api_key}')
            response.raise_for_status()  # Проверка статуса ответа, вызывает исключение при ошибке HTTP
            data = response.json()  # Преобразуем ответ в формат JSON

            # Проверяем, есть ли результаты в ответе API
            if data['results']:
                # Извлекаем координаты (широта и долгота) из первого результата
                location = data['results'][0]['geometry']
                latitude = location['lat']
                longitude = location['lng']
                print(f'Response OK: Time {dt.datetime.now()}')
                return latitude, longitude  # Возвращаем координаты
            else:
                # Если API не вернул результаты, выводим сообщение об ошибке
                print(f'Error for address "{address}": No results found')
                return None, None  # Возвращаем None, если данные не найдены
        except (requests.RequestException, ValueError) as e:
            # Обрабатываем ошибки, связанные с запросом или с некорректным JSON
            print(f'Attempt {attempt + 1} failed: {e}')
            time.sleep(delay)  # Делаем задержку перед повторной попыткой

    # Если все попытки неудачны, возвращаем None
    return None, None


def outliers_iqr_mod_log(data, feature, left=1.5, right=1.5, log_scale=False):
    """
    Поиск выбросов на основании метода межквартильных размахов (метод Тьюки).
    
    Аргументы:
    data: pandas.DataFrame
        Датафрейм, содержащий данные.
    feature: str
        Название столбца, по которому производится поиск выбросов.
    left: float, по умолчанию 1.5
        Коэффициент для определения нижней границы выбросов (множитель IQR).
    right: float, по умолчанию 1.5
        Коэффициент для определения верхней границы выбросов (множитель IQR).
    log_scale: bool, по умолчанию False
        Если True, данные перед обработкой логарифмируются (используется для обработки данных с экспоненциальным распределением).

    Возвращает:
    outliers: pandas.DataFrame
        Датафрейм, содержащий выбросы, которые находятся за пределами границ.
    cleaned: pandas.DataFrame
        Датафрейм с "очищенными" данными, не содержащий выбросов (находящихся внутри границ).
    """
    # Если логарифмическое преобразование включено, применяем логарифм к данным
    if log_scale:
        x = np.log(data[feature])
    else:
        # Иначе используем данные как есть (без логарифмирования)
        x = data[feature]
    
    # Вычисляем первый (25-й процентиль) и третий (75-й процентиль) квартиль данных
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75)
    
    # Вычисляем межквартильный размах (IQR)
    iqr = quartile_3 - quartile_1
    
    # Определяем нижнюю границу для выбросов
    lower_bound = quartile_1 - (iqr * left)
    
    # Определяем верхнюю границу для выбросов
    upper_bound = quartile_3 + (iqr * right)
    
    # Определяем выбросы: данные, которые находятся ниже нижней границы или выше верхней границы
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    
    # Оставляем очищенные данные: те, которые находятся внутри границ (не являются выбросами)
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    
    # Возвращаем два датафрейма: выбросы и очищенные данные
    return outliers, cleaned


# Функция для извлечения категорий из тэгов
def extract_tags_info(tags):
    """
    Извлечение категорий и информации о поездке из строки с тегами.

    Аргументы:
    tags: str
        Строка, содержащая список тегов (категорий) в текстовом виде, которую нужно преобразовать.

    Возвращает:
    pandas.Series
        Серия из 7 значений:
        - trip_type: 1, если 'Leisure trip' присутствует в тегах, иначе 0.
        - business_type: 1, если 'Business trip' присутствует в тегах, иначе 0.
        - couple_flag: 1, если 'Couple' присутствует в тегах, иначе 0.
        - solo_flag: 1, если 'Solo traveler' присутствует в тегах, иначе 0.
        - group_flag: 1, если 'Group' присутствует в тегах, иначе 0.
        - nights_stayed: количество ночей, если указано ('Stayed X nights'), иначе 0.
        - mobile_flag: 1, если отзыв был оставлен с мобильного устройства ('mobile'), иначе 0.
    """
    
    # Преобразуем строку с тегами в список (из текстового формата в Python-список)
    tags = eval(tags)
    
    # Проверяем наличие различных категорий в тегах и присваиваем бинарные значения (1 или 0)
    trip_type = 1 if any('Leisure trip' in tag for tag in tags) else 0  # Признак для Leisure trip
    business_type = 1 if any('Business trip' in tag for tag in tags) else 0  # Признак для Business trip
    couple_flag = 1 if any('Couple' in tag for tag in tags) else 0  # Признак для пар
    solo_flag = 1 if any('Solo traveler' in tag for tag in tags) else 0  # Признак для одиночных путешественников
    group_flag = 1 if any('Group' in tag for tag in tags) else 0  # Признак для групп
    mobile_flag = 1 if any('mobile' in tag for tag in tags) else 0  # Признак для мобильных устройств

    # Инициализируем переменную для хранения количества ночей
    nights_stayed = 0
    
    # Поиск информации о количестве ночей с помощью регулярного выражения 'Stayed X nights'
    for tag in tags:
        match = re.search(r'Stayed (\d+) nights', tag)  # Ищем паттерн "Stayed X nights" в каждом теге
        if match:
            nights_stayed = int(match.group(1))  # Извлекаем количество ночей
            break  # Останавливаем цикл после нахождения первого совпадения

    # Возвращаем результат в виде pandas.Series
    return pd.Series([trip_type, business_type, couple_flag, solo_flag, group_flag, nights_stayed, mobile_flag])


# Функция для вычисления расстояния между двумя координатами с использованием формулы Хаверсина
def haversine(lat1, lng1, lat2, lng2):
    """
    Вычисление расстояния между двумя географическими координатами с использованием формулы Хаверсина.
    
    Аргументы:
    lat1: float
        Широта первой точки (в градусах).
    lng1: float
        Долгота первой точки (в градусах).
    lat2: float
        Широта второй точки (в градусах).
    lng2: float
        Долгота второй точки (в градусах).
    
    Возвращает:
    float
        Расстояние между двумя точками в километрах.
    """
    
    # Радиус Земли в километрах (средний радиус)
    R = 6371.0
    
    # Преобразуем значения широты и долготы из градусов в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    
    # Разница координат широты и долготы
    dlat = lat2 - lat1  # Разница широт
    dlng = lng2 - lng1  # Разница долгот
    
    # Формула Хаверсина для расчета расстояния
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Вычисление расстояния: радиус Земли умножаем на результат формулы Хаверсина
    distance = R * c
    
    return distance

# Словарь с координатами центров городов
city_centers = {
    'london': (51.5074, -0.1278), 
    'barcelona': (41.3851, 2.1734), 
    'paris': (48.8566, 2.3522), 
    'amsterdam': (52.3676, 4.9041), 
    'vienna': (48.2082, 16.3738), 
    'milan': (45.4642, 9.1900)
}

# Функция для вычисления расстояния от гостиницы до центра города
def calculate_distance(row):
    """
    Вычисление расстояния от гостиницы до центра города с использованием формулы Хаверсина.
    
    Аргументы:
    row: pandas.Series
        Строка DataFrame, содержащая информацию о гостинице, включая координаты и город.
    
    Возвращает:
    float
        Расстояние от гостиницы до центра города в километрах, или NaN, если город не найден.
    """
    
    # Извлекаем название города из столбца city
    city = row['city']  
    
    # Проверяем, есть ли город в словаре city_centers
    if city in city_centers:
        # Извлекаем координаты центра города (широта и долгота)
        center_lat, center_lng = city_centers[city]
        
        # Извлекаем координаты гостиницы из столбцов lat и lng
        hotel_lat, hotel_lng = row['lat'], row['lng']
        
        # Вычисляем расстояние между гостиницей и центром города с помощью формулы Хаверсина
        return haversine(hotel_lat, hotel_lng, center_lat, center_lng)
    
    else:
        # Если город не найден в словаре, возвращаем NaN (означает "не число" или отсутствующее значение)
        return np.nan
    

# Создаем объект SentimentIntensityAnalyzer из библиотеки VADER для анализа тональности текста
sia = SentimentIntensityAnalyzer()

# Используем словарь для кэширования результатов анализа тональности
# Это позволяет не анализировать одно и то же значение несколько раз, ускоряя выполнение программы
sentiment_cache = {}

# Функция для анализа тональности текста с использованием кэширования
def analyze_sentiment_cached(text):
    """
    Функция принимает текст (строку), проводит анализ его тональности и возвращает тональный коэффициент (compound score).
    Если текст уже был проанализирован ранее, результат возвращается из кэша.
    
    Параметры:
    - text (str): текст отзыва, который нужно проанализировать.
    
    Возвращает:
    - sentiment_score (float): тональный коэффициент текста (compound score).
                               Значение может варьироваться от -1 (очень негативный) до 1 (очень позитивный).
    """
    # Проверяем, есть ли уже результат анализа данного текста в кэше
    if text in sentiment_cache:
        # Если результат уже сохранен в кэше, возвращаем его
        return sentiment_cache[text]
    else:
        # Если текста нет в кэше, проводим анализ
        # Если текст содержит стандартные фразы, указывающие на отсутствие отзыва, возвращаем 0
        if text.lower() in ['no positive', 'no negative', 'nothing', 'n a']:
            sentiment_cache[text] = 0
        else:
            # Иначе анализируем текст с помощью SentimentIntensityAnalyzer и сохраняем результат в кэш
            sentiment_cache[text] = sia.polarity_scores(text)['compound']
        
        # Возвращаем результат анализа тональности
        return sentiment_cache[text]
