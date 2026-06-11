
### Folder structure for the BACKEND


backend/
├── main.py                  # solo FastAPI app + registrar routers
├── routers/ #todos los enpoints HTTP y WebSockets
│   ├── predictions.py       # POST /predict/eta, POST /predict/eta/broadcast
│   ├── routes.py            # GET /route, GET /route/{relation_id}
│   └── health.py            # GET /health
├── services/
│   ├── occupancy_service.py # lógica XGBoost SUNT OD
│   ├── eta_service.py       # lógica RF MTA
│   ├── route_service.py     # llama Overpass, parsea JSON, cachea resultado
│   └── ws_manager.py        # ConnectionManager (ya existe en main.py)
├── models/
│   └── schemas.py           # BusPayload, PredictionResponse (los Pydantic actuales)
└── utils/
    └── geo.py               # haversine_meters, get_next_stop


### Requirements
`pydantic` `httpx` `fastapi`