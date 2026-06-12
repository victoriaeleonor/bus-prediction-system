# ==============================================================================
# ENTRENAMIENTO Y EVALUACIÓN DE CNN 1D (ALINEADO ESTRICTAMENTE CON XGBOOST)
# ==============================================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. RUTAS Y CONFIGURACIONES DE COLUMNAS
# Usamos el CSV final exportado por el preprocesamiento de XGBoost
DATA_PATH = '/Users/florenciagonzalez/Documents/bus-prediction-system/ml/eta/data/eta_preprocessed_xgboost_final.csv'
MODEL_SAVE_PATH = 'cnn_model_eta_final'

# Las 12 características seleccionadas en la Celda 11 de XGBoost
features = [
    'DistanceFromStop', 'dist_to_dest_m', 'distance_close', 'schedule_delay',
    'speed_kmh', 'speed_roll3', 'direction', 'hour_sin', 'hour_cos', 
    'is_am_rush', 'is_pm_rush', 'proximity_enc'
]
target = 'TimeToArrival'

print("Cargando dataset unificado desde XGBoost...")
df_final = pd.read_csv(DATA_PATH, low_memory=False)

# 2. SEPARACIÓN TEMPORAL POR ÍNDICE (Garantiza el aislamiento de velocidades)
# El archivo ya viene ordenado internamente en bloques: primeros 80% Train, últimos 20% Test
split_idx = int(len(df_final) * 0.80)

df_train_raw = df_final.iloc[:split_idx].copy()
df_test_raw  = df_final.iloc[split_idx:].copy()

print(f"Registros iniciales extraídos -> Train: {len(df_train_raw):,} | Test: {len(df_test_raw):,}")

# 3. LIMPIEZA DE NULOS RESIDUALES (Igual que en la extracción de matrices de XGBoost)
df_train_clean = df_train_raw.dropna(subset=features + [target])
df_test_clean  = df_test_raw.dropna(subset=features + [target])

X_train = df_train_clean[features].values
Y_train = df_train_clean[target].values

X_test  = df_test_clean[features].values
Y_test  = df_test_clean[target].values

print(f"✓ Matrices numéricas listas (sin nulos).")
print(f"  Train shape: X={X_train.shape} | Y={Y_train.shape}")
print(f"  Test shape:  X={X_test.shape} | Y={Y_test.shape}")

# 4. ESCALADO DE CARACTERÍSTICAS (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# El test set se transforma usando EXCLUSIVAMENTE la media y desviación del train set
X_test_scaled = scaler.transform(X_test)

# 5. REDIMENSIONAMIENTO TRIDIMENSIONAL PARA CNN 1D
# Estructura requerida por Keras: (muestras, secuencias/features, canales)
# En este caso: (muestras, 12, 1)
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

print(f"\nDimensiones preparadas para la CNN 1D:")
print(f"  X_train_cnn shape: {X_train_cnn.shape}")
print(f"  X_test_cnn shape:  {X_test_cnn.shape}")

# 6. ARQUITECTURA DEL MODELO CNN 1D
model = Sequential([
    # Primera capa convolucional adaptada dinámicamente a las 12 características de entrada
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])),
    
    Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    
    Flatten(),
    Dropout(0.2),
    
    Dense(units=256, activation='relu'),
    Dense(units=1, activation='linear') # Capa de salida lineal para regresión (Predicción de minutos)
])

# Compilación utilizando el optimizador Adam con la tasa de aprendizaje de tu diseño original
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

print("\nResumen de la estructura de la CNN:")
model.summary()

# 7. ENTRENAMIENTO CON EARLY STOPPING
# Detiene el entrenamiento si la pérdida de validación deja de mejorar para evitar sobreajuste
early_stopping = EarlyStopping(
    monitor='loss', 
    patience=5, 
    restore_best_weights=True
)

print("\nIniciando entrenamiento del modelo CNN...")
history = model.fit(
    X_train_cnn,
    Y_train,
    epochs=32,
    batch_size=256, # Incrementado ligeramente para agilizar el procesamiento del volumen de datos
    callbacks=[early_stopping],
    verbose=1
)

# 8. PREDICCIONES Y EVALUACIÓN DE MÉTRICAS RESULTANTES
print("\nGenerando predicciones sobre el conjunto de prueba (Futuro)...")
Y_pred = model.predict(X_test_cnn).flatten()

# Cálculo de las métricas estadísticas primarias
mae  = mean_absolute_error(Y_test, Y_pred)
mse  = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(Y_test, Y_pred)

print("\n=========================================================================")
print("   MÉTRICAS DE EVALUACIÓN RESULTANTES — RECONSTRUCCIÓN CNN 1D")
print("=========================================================================")
print(f" Error Absoluto Medio (MAE):   {mae:.4f} min  ({mae * 60:.1f} segundos)")
print(f" Error Cuadrático Medio (MSE): {mse:.4f}")
print(f" Raíz del ECM (RMSE):          {rmse:.4f} min ({rmse * 60:.1f} segundos)")
print(f" Coeficiente de Det. (R2):     {r2:.4f}")
print("=========================================================================")

# 9. GUARDAR MODELO ENTRENADO
model.save(MODEL_SAVE_PATH)
print(f"\n✓ Modelo CNN guardado exitosamente en: '{MODEL_SAVE_PATH}'")