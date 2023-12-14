from tensorflow.keras.models import load_model
import pandas as pd
import random
import numpy as np

# Kalori Excedded calculation (The first user use the app)
def calculate_bmr(weight_kg, height_cm, age, gender):
    if gender == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender == 'female':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        raise ValueError("Invalid gender")
    return bmr

def calculate_tdee(bmr, activity_factor):
    tdee = bmr * activity_factor
    return tdee

# Contoh penggunaan
weight = 90  # Berat badan dalam kilogram
height = 180  # Tinggi badan dalam sentimeter
age = 21  # Usia
gender = 'male'  # Jenis kelamin ('male' atau 'female')
activity_factor = 1.375  # Faktor aktivitas fisik (contoh: sedang aktif)

estimated_kalori = calculate_bmr(weight, height, age, gender)
kalori_user = calculate_tdee(estimated_kalori, activity_factor) # kalau makan tinggal di tambah yg bagian (tdee)

kalori_excess = kalori_user - estimated_kalori
print('Excess calories',kalori_excess)

# Machine Learning

# Load the ensemble models
loaded_models = []
num_models = 3
for i in range(num_models):
    model_path = f'Model/ensemble_model2_{i + 1}.h5'
    loaded_model = load_model(model_path)
    loaded_models.append(loaded_model)

dataset=pd.read_csv('Dataset/Exercise_Output_5.csv',index_col=0)

# Reshape the input to (1, 1)
kalori_input = np.array([[kalori_excess]])

# Initialize ensemble probabilities
ensemble_probabilities = np.zeros((1, 20))
# Make predictions using each loaded model
for model in loaded_models:
    predictions = model.predict(kalori_input)
    ensemble_probabilities += predictions
flat_data = ensemble_probabilities[0]
max_index = np.argmax(flat_data)
value = np.random.rand(20)
value[max_index] = 1.0

# Average the probabilities
print(value)
# Create a list of tuples containing label and probability
label_prob_tuples = [(label, probability) for label, probability in enumerate(value)]


# Sorting
# Sort the list based on probabilities in descending order
sorted_label_prob_tuples = sorted(label_prob_tuples, key=lambda x: x[1], reverse=True)

# Extract only the labels from the sorted list of tuples
sorted_labels = [label for label, _ in sorted_label_prob_tuples]
# label_5=sorted_labels[:5]
# Convert label_5 to a pandas Series for comparison
label_5_series = pd.Series(sorted_labels[0:5])

# Use loc to filter rows where 'label' is in label_5_series
selected_rows = dataset.loc[dataset['label'].isin(label_5_series), ['label', 'activity']]
selected_rows.sort_values(by='label',ascending=True,inplace=True)
# Print or use the selected rows
print(selected_rows.drop_duplicates(subset=['label']))