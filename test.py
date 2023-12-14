from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
# Load the ensemble models
loaded_models = []
num_models = 3
for i in range(num_models):
    model_path = f'Model/ensemble_model2_{i + 1}.h5'
    loaded_model = load_model(model_path)
    loaded_models.append(loaded_model)

dataset=pd.read_csv('Dataset/Exercise_Output_5.csv',index_col=0)

# Assuming 'kalori' is your input value
# kalori = 2015.125
kalori = 100000000

# Reshape the input to (1, 1)
kalori_input = np.array([[kalori]])

# Initialize ensemble probabilities
ensemble_probabilities = np.zeros((1, 20))
# Make predictions using each loaded model
for model in loaded_models:
    predictions = model.predict(kalori_input)
    ensemble_probabilities += predictions

# Average the probabilities
ensemble_probabilities /= num_models
print(ensemble_probabilities)
# Create a list of tuples containing label and probability
label_prob_tuples = [(label, probability) for label, probability in enumerate(ensemble_probabilities.ravel())]


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
selected_rows.sort_values(by='label',ascending=False,inplace=True)
# Print or use the selected rows
print(selected_rows.drop_duplicates(subset=['label']))