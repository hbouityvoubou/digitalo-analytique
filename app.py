from fastapi import FastAPI
import uvicorn
import numpy as np
import pandas as pd
import pickle
import json
import tensorflow as tf

from pydantic import BaseModel

class caracteristiques(BaseModel):
    Agricultural_Crops : float 
    Precipitation : float 
    Air_Temperature : float 
    Relative_Humidity : float 
    Atmospheric_Pressure : float 
    Wind_Speed : float 
    Cloud_Cover : float 
    Sunshine_Duration : float 
    Visibility : float 
    	

app = FastAPI()
#load the model
mlp_model = pickle.load(open('mlpmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
agricultural_tasks = ['Sow', 'Irrigate', 'Fertilize', 'Weed', 'Prick', 'Prune', 'Hoe', 'Harvest']

@app.get('/')
def index():
    return {'message':'Bonjour Henri'}

@app.post('/predict')
def prediction(Data:caracteristiques):
    data = pd.json_normalize(Data.dict())
    data_scaled = scaler.transform(data)
    probabilities = mlp_model.predict_proba(data_scaled)
    # Afficher les probabilités pour chaque exemple de l'ensemble de test
    for i, prob in enumerate(probabilities):
        # print(f"Exemple {i + 1}:")
        # Créer une liste vide
        tableau = []
        for j, p in enumerate(prob):
            # print(f"  Classe {j}: Probabilité {p:.4f}")
            tableau.append(p)
            
        #print()
    
        # # Trouvez l'indice de la classe avec la probabilité la plus élevée
        predicted_class_index = tf.argmax(tableau).numpy()
        #print("index de la probabilité la plus forte :",predicted_class_index)
        #print("la probabilité la plus forte :",tableau[predicted_class_index])
        taux = tableau[predicted_class_index]*100
        # # Utilisez cet indice pour obtenir l'étiquette prédite
        predicted_class = agricultural_tasks[predicted_class_index]
        # # Affichez l'étiquette prédite
        return f"certain à {taux:.2f} %, qu'il faut {predicted_class}"


if __name__=='__main__':
    uvicorn.run(app)