import pandas as pd  # за работа с таблици

# 1. Зареждане на CSV файла
df = pd.read_csv('cars.csv')

# 2. Печатане на първите редове
print(df.head())

# 3. Кодиране на категориалните данни (fuel, engine_size, weight, car_type)
fuel_mapping = {'gasoline': 0, 'diesel': 1, 'electric': 2}
engine_mapping = {'small': 0, 'medium': 1, 'large': 2}
weight_mapping = {'light': 0, 'medium': 1, 'heavy': 2}
car_type_mapping = {'sport': 0, 'family': 1, 'city': 2}

# Замяна на текстовете с числа
df['fuel'] = df['fuel'].map(fuel_mapping)
df['engine_size'] = df['engine_size'].map(engine_mapping)
df['weight'] = df['weight'].map(weight_mapping)
df['car_type'] = df['car_type'].map(car_type_mapping)

# Проверяваме отново как изглеждат данните
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 4. Разделяне на данните
X = df[['doors', 'fuel', 'engine_size', 'weight']]
y = df['car_type']

# 5. Разделяме на тренировъчни и тестови данни
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Създаваме и обучаваме модела KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 7. Тестваме точността
accuracy = knn.score(X_test, y_test)
print(f"Точност на модела: {accuracy:.2f}")

# 8. Прогноза за нова кола

print("\nНека въведем нова кола за предсказване:")

# Въвеждаме характеристиките
doors = int(input("Брой врати (примерно 2, 4, 5): "))
fuel_input = input("Тип гориво (gasoline/diesel/electric): ")
engine_input = input("Размер на двигател (small/medium/large): ")
weight_input = input("Тегло на колата (light/medium/heavy): ")

# Превеждаме текстовете в числа
fuel = fuel_mapping[fuel_input]
engine_size = engine_mapping[engine_input]
weight = weight_mapping[weight_input]

# Създаваме нова кола за прогнозиране
new_car = [[doors, fuel, engine_size, weight]]

# Прогноза
prediction = knn.predict(new_car)
predicted_type = [key for key, value in car_type_mapping.items() if value == prediction[0]][0]

print(f"\nПредсказание: Тази кола е '{predicted_type.upper()}' тип.")
