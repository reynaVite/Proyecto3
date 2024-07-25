from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging 

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo, el escalador y el codificador
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        habilidad_lectura = request.form['habilidad_lectura']
        habilidad_escritura = request.form['habilidad_escritura']
        habilidad_matematicas = request.form['habilidad_matematicas']
        participacion = request.form['participacion']
        comportamiento = request.form['comportamiento']
        
        # Crear un DataFrame con los datos
        nuevo_dato = pd.DataFrame({
            'habilidad_lectura': [habilidad_lectura],
            'habilidad_escritura': [habilidad_escritura],
            'habilidad_matematicas': [habilidad_matematicas],
            'participacion': [participacion],
            'comportamiento': [comportamiento]
        })
        
        # Codificar las variables categóricas del nuevo dato
        nuevo_dato_encoded = ordinal_encoder.transform(nuevo_dato)

        # Convertir los datos codificados en un DataFrame para mantener los nombres de columnas
        nuevo_dato_encoded_df = pd.DataFrame(nuevo_dato_encoded, columns=nuevo_dato.columns)

        # Escalar el nuevo dato
        nuevo_dato_scaled = scaler.transform(nuevo_dato_encoded_df)

        # Realizar la predicción
        prediccion = rf_model.predict(nuevo_dato_scaled)

        # Convertir la predicción a un tipo serializable
        prediccion_serializable = int(prediccion[0])

        app.logger.debug(f'Predicción: {prediccion_serializable}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediction': prediccion_serializable})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
