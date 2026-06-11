"""
Mejoras de feature engineering para el modelo ETA
Basado en resultados experimentales sobre dataset MTA y literatura:
- Sun et al. (2007), Transportation Research Record
- Chien & Ding (2002), Journal of Transportation Engineering  
- Dunne & McArdle (2023), Intelligent Transport Systems Conference
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def haversine_meters(lat1, lon1, lat2, lon2):
    """Distancia haversine en metros entre dos puntos GPS."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def parse_mta_schedule_time(sched_str, reference_dt):
    """
    Convierte el formato de hora MTA '24:06:14' a datetime.
    MTA usa horas > 23 para indicar servicio nocturno del día siguiente.
    """
    try:
        parts = str(sched_str).split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        base = reference_dt.normalize()
        return base + pd.Timedelta(hours=h, minutes=m, seconds=s)
    except Exception:
        return pd.NaT


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construye todas las features a partir del CSV raw del MTA.

    Columnas requeridas del CSV original (mta_1706.csv):
        RecordedAtTime, ExpectedArrivalTime, ScheduledArrivalTime,
        DistanceFromStop, PublishedLineName, DirectionRef,
        OriginLat, OriginLong, DestinationLat, DestinationLong,
        VehicleLocation.Latitude, VehicleLocation.Longitude,
        VehicleRef, NextStopPointName

    Retorna DataFrame listo para entrenar con features y columna TimeToArrival.
    """
    df = df_raw.copy()

    # --- Parsear fechas ---
    df['RecordedAtTime'] = pd.to_datetime(df['RecordedAtTime'])
    df['ExpectedArrivalTime'] = pd.to_datetime(df['ExpectedArrivalTime'])

    # Eliminar filas sin columnas clave
    required = [
        'ExpectedArrivalTime', 'RecordedAtTime', 'DistanceFromStop',
        'OriginLat', 'OriginLong', 'DestinationLat', 'DestinationLong',
        'VehicleLocation.Latitude', 'VehicleLocation.Longitude'
    ]
    df = df.dropna(subset=required).copy()

    # Ordenar para calcular features de lag correctamente
    df = df.sort_values(['VehicleRef', 'PublishedLineName', 'RecordedAtTime'])

    # ================================================================
    # TARGET: minutos hasta llegada a la próxima parada
    # ================================================================
    df['TimeToArrival'] = (
        (df['ExpectedArrivalTime'] - df['RecordedAtTime'])
        .dt.total_seconds() / 60
    )
    df = df[(df['TimeToArrival'] > 0) & (df['TimeToArrival'] < 60)].copy()

    # ================================================================
    # FEATURE 1: Approach speed (velocidad de acercamiento a la parada)
    # Fuente: Sun et al. (2007) - GPS-based bus arrival time prediction
    # Calcula cuántos metros por segundo se acerca el bus a la MISMA parada
    # entre pings consecutivos del mismo vehículo en la misma ruta.
    # 
    # NOTA: Con GPS cada ~10 min (MTA), esta feature tiene baja cobertura (~4%)
    # y baja importancia. Con GPS de alta frecuencia (Raspberry Pi, segundos)
    # sería mucho más potente.
    # ================================================================
    grp = df.groupby(['VehicleRef', 'PublishedLineName', 'NextStopPointName'])
    df['prev_dist'] = grp['DistanceFromStop'].shift(1)
    df['prev_time_s'] = (
        df['RecordedAtTime'] - grp['RecordedAtTime'].shift(1)
    ).dt.total_seconds()

    valid_speed = (
        df['prev_dist'].notna() &
        (df['prev_time_s'] > 0) &
        (df['prev_time_s'] < 800)   # descartar gaps entre viajes
    )
    df.loc[valid_speed, 'approach_speed_mps'] = (
        (df.loc[valid_speed, 'prev_dist'] - df.loc[valid_speed, 'DistanceFromStop'])
        / df.loc[valid_speed, 'prev_time_s']
    ).clip(0, 20)
    df['approach_speed_mps'] = df['approach_speed_mps'].fillna(0)

    # ================================================================
    # FEATURE 2: Progreso en la ruta y distancia al destino
    # Fuente: Dunne & McArdle (2023) - predicting whole route vs segment
    # El porcentaje completado de la ruta captura el contexto espacial
    # del bus (inicio, mitad, fin de recorrido).
    # ================================================================
    df['total_route_dist_m'] = haversine_meters(
        df['OriginLat'], df['OriginLong'],
        df['DestinationLat'], df['DestinationLong']
    )
    df['dist_traveled_m'] = haversine_meters(
        df['OriginLat'], df['OriginLong'],
        df['VehicleLocation.Latitude'], df['VehicleLocation.Longitude']
    )
    df['route_completion'] = (
        df['dist_traveled_m'] / df['total_route_dist_m'].replace(0, np.nan)
    ).clip(0, 1).fillna(0.5)

    # Distancia restante al destino (proxy de "cuántas paradas faltan")
    # Chien & Ding (2002) identifican el número de paradas restantes
    # como predictor clave. Como no tenemos esa info directamente,
    # usamos la distancia geográfica al destino.
    df['dist_to_dest_m'] = haversine_meters(
        df['VehicleLocation.Latitude'], df['VehicleLocation.Longitude'],
        df['DestinationLat'], df['DestinationLong']
    )

    # ================================================================
    # FEATURE 3: Delay vs horario programado
    # Fuente: Chien & Ding (2002), Sun et al. (2007)
    # El atraso acumulado vs el horario programado es predictor clave.
    # Positivo = el bus va tarde, negativo = va adelantado.
    # ================================================================
    sched_dts = df.apply(
        lambda r: parse_mta_schedule_time(r['ScheduledArrivalTime'], r['ExpectedArrivalTime']),
        axis=1
    )
    df['delay_min'] = (
        (df['ExpectedArrivalTime'] - sched_dts).dt.total_seconds() / 60
    ).clip(-30, 30).fillna(0)

    # ================================================================
    # FEATURES TEMPORALES
    # ================================================================
    df['hour'] = df['RecordedAtTime'].dt.hour
    df['minute'] = df['RecordedAtTime'].dt.minute
    df['day_of_week'] = df['RecordedAtTime'].dt.dayofweek

    # Representación cíclica de la hora (23h y 0h son consecutivas)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # ================================================================
    # FEATURE: Ruta del bus y dirección
    # ================================================================
    le = LabelEncoder()
    df['line_encoded'] = le.fit_transform(df['PublishedLineName'].astype(str))
    df['direction'] = df['DirectionRef'].fillna(0).astype(int)

    return df, le


# ================================================================
# FEATURES FINALES
# ================================================================
FEATURES_BASELINE = [
    # Las 4 features del modelo original
    'DistanceFromStop', 'hour', 'minute', 'day_of_week',
]

FEATURES_IMPROVED = [
    # --- Original ---
    'DistanceFromStop',

    # --- Tiempo ---
    'hour', 'minute', 'day_of_week',
    'hour_sin', 'hour_cos',           # representación cíclica

    # --- Nuevas features documentadas ---
    'approach_speed_mps',             # velocidad de acercamiento a la parada
    'route_completion',               # % completado de la ruta
    'dist_to_dest_m',                 # proxy de paradas restantes
    'delay_min',                      # atraso vs horario programado

    # --- Contexto ---
    'line_encoded',                   # qué línea de bus
    'direction',                      # dirección del recorrido
]

TARGET = 'TimeToArrival'


if __name__ == '__main__':
    import zipfile
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    import joblib

    # --- Cargar datos ---
    ZIP_PATH = '/Users/jazgonzalez/Documents/capstone/bus-prediction-system/ml/eta/data/mta_1706.csv.zip'   # ajusta tu ruta
    print('Cargando dataset...')
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open('mta_1706.csv') as f:
            df_raw = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
    print(f'Filas cargadas: {len(df_raw):,}')

    # --- Feature engineering ---
    print('Construyendo features...')
    df_feat, line_encoder = build_features(df_raw)

    # --- Dataset final ---
    df_model = df_feat[FEATURES_IMPROVED + [TARGET]].dropna()
    print(f'Dataset final: {len(df_model):,} filas, {len(FEATURES_IMPROVED)} features')

    # --- Split ---
    X = df_model[FEATURES_IMPROVED]
    y = df_model[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Entrenar ---
    print('Entrenando modelo...')
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- Evaluar ---
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nMAE:  {mae:.3f} min ({mae*60:.1f} seg)')
    print(f'R²:   {r2:.4f}')

    # --- Guardar ---
    joblib.dump(model, 'eta_model_v3.pkl')
    joblib.dump(line_encoder, 'line_encoder_v3.pkl')
    print('\nModelo guardado: eta_model_v3.pkl')
    print('Encoder guardado: line_encoder_v3.pkl')