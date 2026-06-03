"""
Bus Prediction System — FastAPI Backend
Loads occupancy (XGBoost/RF) and ETA (RF) models and serves predictions.
"""
import os
import math
import pickle
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── paths ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
OCC_DIR  = BASE / "ml" / "occupancy" / "models"
ETA_DIR  = BASE / "ml" / "eta" / "model"

# ── load occupancy models ──────────────────────────────────────────────────
SUFFIX = "with_lags"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

xgb_model      = load_pkl(OCC_DIR / f"xgboost_occupancy_{SUFFIX}.pkl")
xgb_encoder    = load_pkl(OCC_DIR / f"xgboost_label_encoder_{SUFFIX}.pkl")
xgb_features   = load_pkl(OCC_DIR / f"xgboost_feature_names_{SUFFIX}.pkl")
label_encoders = load_pkl(BASE / "ml" / "occupancy" / "data" / f"sunt_2024_03_{SUFFIX}_encoders.pkl")

try:
    eta_model = joblib.load(ETA_DIR / "rf_model_eta.pkl")
    print("✓ ETA model loaded")
except FileNotFoundError:
    eta_model = None
    print("⚠ ETA model not found — will return dummy ETA")

print("✓ All models loaded")

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Bus Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── pydantic schemas ───────────────────────────────────────────────────────
class BusPayload(BaseModel):
    bus_id: str
    route: str
    lat: float
    lon: float
    timestamp: str
    occupancy: int
    hour: int
    day_of_week: int
    is_rush_hour: int
    route_progress: float
    stop_index: int = 0   # ← nuevo, default 0 para compatibilidad

class PredictionResponse(BaseModel):
    bus_id: str
    route: str
    lat: float
    lon: float
    timestamp: str
    occupancy_raw: int
    occupancy_class: str
    occupancy_pct: float
    eta_minutes: float
    hour: int
    day_of_week: int
    is_rush_hour: int

# ── route stops (coordenadas reales de la linea 38 — Asuncion) ─────────────
ROUTE_STOPS = [
    (-25.3863252, -57.4976859),
    (-25.3863252, -57.4976859),
    (-25.3859144, -57.4973199),
    (-25.3851929, -57.4966801),
    (-25.3847335, -57.4962726),
    (-25.3839298, -57.495803),
    (-25.3833575, -57.4954686),
    (-25.3831084, -57.4953226),
    (-25.3825319, -57.494974),
    (-25.3815804, -57.4944221),
    (-25.3811034, -57.4941683),
    (-25.3808609, -57.4940399),
    (-25.3804046, -57.4938178),
    (-25.3797617, -57.493512),
    (-25.3793231, -57.4933248),
    (-25.3786856, -57.4930827),
    (-25.3786303, -57.4930652),
    (-25.3782926, -57.492976),
    (-25.3775332, -57.4927958),
    (-25.3764615, -57.4925578),
    (-25.376039, -57.4924593),
    (-25.3753657, -57.4923655),
    (-25.3749787, -57.4922851),
    (-25.3747617, -57.4922424),
    (-25.3731256, -57.4919764),
    (-25.3721025, -57.4918354),
    (-25.3718868, -57.491814),
    (-25.3713409, -57.4917796),
    (-25.370561, -57.4917354),
    (-25.3695743, -57.4916797),
    (-25.3694164, -57.4916613),
    (-25.3680534, -57.4912486),
    (-25.3666988, -57.490828),
    (-25.3660467, -57.490654),
    (-25.3658579, -57.4906036),
    (-25.3642771, -57.4904137),
    (-25.36403, -57.4903901),
    (-25.363709, -57.4903528),
    (-25.3628977, -57.4902616),
    (-25.3611723, -57.490045),
    (-25.3606042, -57.4900255),
    (-25.3602151, -57.4900579),
    (-25.3599109, -57.4901907),
    (-25.359605, -57.4903616),
    (-25.3589454, -57.4907164),
    (-25.3584819, -57.4908329),
    (-25.3576299, -57.4909479),
    (-25.3567276, -57.4909285),
    (-25.355826, -57.4908775),
    (-25.3553296, -57.4908553),
    (-25.3548081, -57.4941158),
    (-25.3542948, -57.4975845),
    (-25.3540101, -57.4993812),
    (-25.3537025, -57.5011164),
    (-25.3535693, -57.5018628),
    (-25.3530147, -57.5017494),
    (-25.3512658, -57.5014137),
    (-25.3511425, -57.5014109),
    (-25.3500692, -57.5020257),
    (-25.3487893, -57.5028459),
    (-25.348588, -57.5029725),
    (-25.3482033, -57.5031908),
    (-25.3475979, -57.5035494),
    (-25.3464332, -57.5041875),
    (-25.3452314, -57.5048568),
    (-25.3441832, -57.5054835),
    (-25.344088, -57.5055354),
    (-25.3429009, -57.5062254),
    (-25.3417728, -57.5069027),
    (-25.3402915, -57.5077081),
    (-25.3384063, -57.5087981),
    (-25.3381197, -57.5089729),
    (-25.3380362, -57.5090273),
    (-25.3379577, -57.5090867),
    (-25.3373175, -57.5096368),
    (-25.3370013, -57.5099294),
    (-25.3370021, -57.5099749),
    (-25.3369973, -57.5100769),
    (-25.336969, -57.5103123),
    (-25.3369655, -57.5103437),
    (-25.3369423, -57.5104881),
    (-25.3369276, -57.5105626),
    (-25.33687, -57.5107241),
    (-25.3367491, -57.5109126),
    (-25.3352849, -57.5123078),
    (-25.3349542, -57.5125931),
    (-25.3336472, -57.5136603),
    (-25.3330033, -57.5141786),
    (-25.3326535, -57.5144618),
    (-25.3314631, -57.5154118),
    (-25.3314274, -57.5154413),
    (-25.3294313, -57.5170853),
    (-25.3293454, -57.5171562),
    (-25.3281673, -57.518039),
    (-25.3266024, -57.5191339),
    (-25.3248962, -57.5204055),
    (-25.324645, -57.5205927),
    (-25.3245102, -57.52069),
    (-25.3242421, -57.5208836),
    (-25.3241237, -57.5209694),
    (-25.3237761, -57.5212214),
    (-25.3223181, -57.5222782),
    (-25.3222302, -57.5223419),
    (-25.3209339, -57.5232777),
    (-25.3208898, -57.5233094),
    (-25.3193744, -57.5243842),
    (-25.3186608, -57.5248948),
    (-25.3181638, -57.5252505),
    (-25.3175917, -57.5256795),
    (-25.3173421, -57.5258667),
    (-25.3169424, -57.5261665),
    (-25.3163067, -57.5266471),
    (-25.315152, -57.5275196),
    (-25.3147123, -57.5278517),
    (-25.3141243, -57.5282959),
    (-25.3136949, -57.5286218),
    (-25.3130163, -57.5291365),
    (-25.3127107, -57.5293668),
    (-25.311928, -57.5299573),
    (-25.3109957, -57.530663),
    (-25.3108736, -57.5307552),
    (-25.310062, -57.5313687),
    (-25.3091485, -57.5320604),
    (-25.3088143, -57.5323135),
    (-25.3071469, -57.5335698),
    (-25.3067129, -57.5339001),
    (-25.3063359, -57.5342217),
    (-25.3061666, -57.5344113),
    (-25.3058376, -57.5347797),
    (-25.3052954, -57.5354281),
    (-25.305038, -57.5357381),
    (-25.3048466, -57.5360005),
    (-25.3041962, -57.5372088),
    (-25.3041241, -57.5373704),
    (-25.3040532, -57.5375777),
    (-25.304031, -57.5378214),
    (-25.3040559, -57.5381042),
    (-25.3042059, -57.5387583),
    (-25.3043545, -57.5394282),
    (-25.3044005, -57.5398821),
    (-25.3044164, -57.5402545),
    (-25.3044358, -57.5408943),
    (-25.3044719, -57.5414002),
    (-25.3045434, -57.5418255),
    (-25.3045936, -57.542032),
    (-25.3047515, -57.5425983),
    (-25.305037, -57.5432815),
    (-25.3052218, -57.5436474),
    (-25.3054167, -57.5440439),
    (-25.3054799, -57.5441565),
    (-25.3062623, -57.5457233),
    (-25.306421, -57.5460431),
    (-25.3067341, -57.5466688),
    (-25.3070145, -57.5472156),
    (-25.3071803, -57.5475697),
    (-25.3074393, -57.5481885),
    (-25.3076414, -57.5488445),
    (-25.3080376, -57.5503036),
    (-25.3080625, -57.5503957),
    (-25.3081622, -57.5508651),
    (-25.3081737, -57.5510247),
    (-25.3081522, -57.5513505),
    (-25.3081082, -57.551562),
    (-25.3080463, -57.5518619),
    (-25.3079173, -57.5522604),
    (-25.3078456, -57.552419),
    (-25.3075989, -57.5529025),
    (-25.3074468, -57.5531709),
    (-25.3073532, -57.5533378),
    (-25.3072288, -57.5535614),
    (-25.3071024, -57.5537947),
    (-25.3070288, -57.5539307),
    (-25.3066604, -57.5545579),
    (-25.3064108, -57.5549771),
    (-25.3059535, -57.5557699),
    (-25.3048038, -57.5578513),
    (-25.3047509, -57.5579471),
    (-25.3041281, -57.5591061),
    (-25.304078, -57.5591993),
    (-25.3036103, -57.5600601),
    (-25.3034737, -57.5603186),
    (-25.303422, -57.5604144),
    (-25.3025845, -57.5619677),
    (-25.3024342, -57.5622391),
    (-25.3022847, -57.5625203),
    (-25.3014212, -57.5640999),
    (-25.3007415, -57.5653407),
    (-25.3002018, -57.5663657),
    (-25.3001424, -57.5664785),
    (-25.2992595, -57.5681117),
    (-25.2990356, -57.5685082),
    (-25.2988984, -57.568768),
    (-25.2985651, -57.5693873),
    (-25.2984029, -57.5696886),
    (-25.2981023, -57.5705164),
    (-25.2977134, -57.5714751),
    (-25.2975369, -57.571853),
    (-25.2975011, -57.571928),
    (-25.2971003, -57.5729112),
    (-25.2966502, -57.5739179),
    (-25.2962649, -57.574813),
    (-25.2962074, -57.5749465),
    (-25.2956106, -57.5763126),
    (-25.2953351, -57.5769446),
    (-25.2952957, -57.5770349),
    (-25.2952599, -57.5771332),
    (-25.2950169, -57.5777078),
    (-25.2949245, -57.5779234),
    (-25.2948883, -57.5780078),
    (-25.2945015, -57.578885),
    (-25.2944599, -57.5789756),
    (-25.2941673, -57.579663),
    (-25.2941039, -57.5798119),
    (-25.293703, -57.5807139),
    (-25.2934928, -57.5811955),
    (-25.2933117, -57.5816103),
    (-25.2929166, -57.5825289),
    (-25.292536, -57.583422),
    (-25.2921401, -57.5843111),
    (-25.2917533, -57.5851781),
    (-25.2915387, -57.5856569),
    (-25.2909677, -57.5869304),
    (-25.2905832, -57.5878472),
    (-25.290553, -57.5879197),
    (-25.2901594, -57.5887937),
    (-25.2900301, -57.5890618),
    (-25.2899624, -57.5891947),
    (-25.2898436, -57.5893433),
    (-25.2897719, -57.5893917),
    (-25.2895916, -57.589499),
    (-25.2895312, -57.5895414),
    (-25.2894268, -57.5895806),
    (-25.2886891, -57.5893321),
    (-25.288177, -57.589177),
    (-25.287472, -57.588966),
    (-25.2874033, -57.5889448),
    (-25.2865929, -57.5886951),
    (-25.2865045, -57.5886702),
    (-25.2864364, -57.5886468),
    (-25.2846951, -57.5881072),
    (-25.2839356, -57.5878745),
    (-25.2829927, -57.587586),
    (-25.281982, -57.5872761),
    (-25.2800236, -57.5866761),
    (-25.2798499, -57.5866229),
    (-25.2772721, -57.5858622),
    (-25.2763927, -57.5856088),
    (-25.2753874, -57.5853059),
    (-25.2742744, -57.5849773),
    (-25.2739833, -57.5848914),
    (-25.2734662, -57.5847337),
    (-25.2726797, -57.5845011),
    (-25.2724771, -57.5844403),
    (-25.2716105, -57.5841804),
    (-25.2709672, -57.5839754),
    (-25.2707073, -57.5838988),
    (-25.2694641, -57.5835318),
    (-25.2684866, -57.5832441),
    (-25.2678491, -57.5830561),
    (-25.2670348, -57.5828161),
    (-25.2657857, -57.5824486),
    (-25.2639636, -57.5819063),
    (-25.2627633, -57.5815592),
    (-25.2618311, -57.5812849),
    (-25.2613517, -57.5811441),
    (-25.2608889, -57.5810077),
    (-25.2600204, -57.5807522),
    (-25.2598671, -57.5807023),
    (-25.2589203, -57.5799655),
    (-25.2588688, -57.5799236),
    (-25.2587977, -57.5798663),
    (-25.2580955, -57.5792746),
    (-25.2573197, -57.5786703),
    (-25.2558904, -57.5775735),
    (-25.2558262, -57.5775256),
    (-25.255199, -57.5768768),
    (-25.2545501, -57.5761163),
    (-25.2539754, -57.5754115),
    (-25.2539373, -57.5753632),
    (-25.253887, -57.5752806),
    (-25.2538592, -57.5751761),
    (-25.2537399, -57.5753567),
    (-25.2538223, -57.5752848),
    (-25.2538592, -57.5751761),
    (-25.2538479, -57.5750825),
    (-25.2537729, -57.5749772),
    (-25.2536561, -57.574941),
    (-25.2535501, -57.5749815),
    (-25.2534852, -57.5750701),
    (-25.25347, -57.5751827),
    (-25.2535079, -57.5752872),
    (-25.2535869, -57.5753559),
    (-25.2536864, -57.5753721),
    (-25.2536864, -57.5753721),
    (-25.2536753, -57.5753991),
    (-25.2530098, -57.57646),
    (-25.2524044, -57.5776412),
    (-25.2524142, -57.5777914),
    (-25.2523796, -57.5779283),
    (-25.2522768, -57.5777438),
    (-25.2525967, -57.5772245),
    (-25.253517, -57.5754023),
    (-25.2534926, -57.5753185),
]

# ── helpers ────────────────────────────────────────────────────────────────
OCC_LABELS = {0: "low", 1: "medium", 2: "high", 3: "very_high"}

def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def get_next_stop(lat, lon, stop_index):
    """Distancia en metros a la próxima parada según índice actual."""
    total = len(ROUTE_STOPS)
    next_idx = (stop_index + 1) % total
    next_stop = ROUTE_STOPS[next_idx]
    return haversine_meters(lat, lon, next_stop[0], next_stop[1])

def predict_occupancy(payload: BusPayload) -> tuple[str, float]:
    now = datetime.fromisoformat(payload.timestamp)
    row = {f: 0 for f in xgb_features}
    mapping = {
        "hour": now.hour,
        "day_of_week": payload.day_of_week,
        "is_rush_hour": payload.is_rush_hour,
        "route_progress": payload.route_progress,
        "loading": payload.occupancy / 100.0,
        "month": now.month,
    }
    for k, v in mapping.items():
        if k in row:
            row[k] = v
    if "route_short_name" in row and "route_short_name" in label_encoders:
        enc = label_encoders["route_short_name"]
        try:
            row["route_short_name"] = enc.transform([payload.route])[0]
        except Exception:
            row["route_short_name"] = 0
    X = pd.DataFrame([row])[xgb_features]
    pred_encoded = xgb_model.predict(X)[0]
    occ_class = xgb_encoder.inverse_transform([pred_encoded])[0] \
        if hasattr(xgb_encoder, "inverse_transform") else OCC_LABELS.get(pred_encoded, "medium")
    pct_map = {"low": 15.0, "medium": 40.0, "high": 65.0, "very_high": 88.0}
    return str(occ_class), pct_map.get(str(occ_class), 50.0)

def predict_eta(payload: BusPayload) -> float:
    if eta_model is None:
        return round((1 - payload.route_progress) * 5, 2)
    now = datetime.fromisoformat(payload.timestamp)
    distance = get_next_stop(payload.lat, payload.lon, payload.stop_index)
    X = pd.DataFrame([{
        "DistanceFromStop": distance,
        "month":       now.month,
        "day":         now.day,
        "day_of_week": payload.day_of_week,
        "hour":        payload.hour,
        "minute":      now.minute,
    }])
    eta = float(eta_model.predict(X)[0])
    return round(max(0.1, eta), 2)

# ── routes ─────────────────────────────────────────────────────────────────
@app.post("/predict/eta", response_model=PredictionResponse)
async def predict(payload: BusPayload):
    occ_class, occ_pct = predict_occupancy(payload)
    eta = predict_eta(payload)
    return PredictionResponse(
        bus_id          = payload.bus_id,
        route           = payload.route,
        lat             = payload.lat,
        lon             = payload.lon,
        timestamp       = payload.timestamp,
        occupancy_raw   = payload.occupancy,
        occupancy_class = occ_class,
        occupancy_pct   = occ_pct,
        eta_minutes     = eta,
        hour            = payload.hour,
        day_of_week     = payload.day_of_week,
        is_rush_hour    = payload.is_rush_hour,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "models": ["xgboost_occupancy", "rf_eta"]}

# ── WebSocket broadcast ────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

@app.post("/predict/eta/broadcast", response_model=PredictionResponse)
async def predict_and_broadcast(payload: BusPayload):
    result = await predict(payload)
    print(f"[WS] clientes conectados: {len(manager.active)}")
    await manager.broadcast(result.dict())
    return result

# ── serve frontend ─────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_path / "index.html"))