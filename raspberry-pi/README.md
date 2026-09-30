# Telemetry source

Everything that feeds the backend lives here. At the moment that is a
simulator: `simulator.py` drives virtual buses along the real geometry of each
line and posts the same payload a Raspberry Pi with a GPS unit would send. The
backend cannot tell the difference, which is the point — replacing the
simulator with real hardware is a matter of producing the same JSON.

```
raspberry-pi/
├── simulator.py      The simulator
├── route_38.json     Raw OpenStreetMap relation for Línea 38
└── route_15_1.json   Raw OpenStreetMap relation for Línea 15-1
```

The route files are unmodified Overpass API responses, so they contain nodes,
ways and the relation that ties them together rather than a ready-made list of
coordinates. Both the simulator and the backend rebuild the polyline from them,
using the same algorithm, so the drawn route and the traveled route always
agree.

## Running it

```bash
pip install requests
python simulator.py --line 38
```

Against a deployed backend:

```bash
python simulator.py --line 15-1 \
  --backend-url https://<your-app>.onrender.com/predict/eta/broadcast
```

`--line` accepts `38` or `15-1` and defaults to `38`. `--backend-url` defaults
to `http://localhost:8000/predict/eta/broadcast`. One process drives one line;
run two processes to have both lines live at once.

The script prints one line per bus per tick with the occupancy class and ETA
the backend returned, so it doubles as a check that inference is working.

## What it sends

Every five seconds, per bus:

| Field | Meaning |
|---|---|
| `bus_id`, `route`, `route_id` | Which bus, on which line |
| `lat`, `lon` | Current position |
| `timestamp` | Local ISO timestamp |
| `occupancy` | Passengers on board, 0–100 |
| `hour`, `day_of_week`, `is_rush_hour` | Clock features for the occupancy model |
| `route_progress` | 0 to 1 within the current leg |
| `stop_index` | Index of the last stop passed |
| `travel_direction` | +1 outbound, -1 on the return leg |
| `speed_kmh` | Smoothed instantaneous speed |

It posts to `/predict/eta/broadcast`, so each payload is predicted on and
pushed to every connected browser in one round trip.

## How the simulation behaves

**Route.** The relation file is loaded, ways are reordered and reversed as
needed to form a continuous line, every second point is kept, and the result is
trimmed to the stretch between the first and last real stop. The trim matters:
the raw relation usually continues past the last named stop to a depot or
turnaround — for Línea 38 that is roughly 247 of 548 points, over twenty
minutes of travel at this pace. Without trimming, the bus disappears down a
dead stretch while its stop index has nothing left to advance to, which looks
exactly like the app having frozen.

**Movement.** Each tick advances one point along the polyline. At the end the
bus turns around and retraces the same points, so `travel_direction` flips to
-1 and `route_progress` restarts within the return leg. Sending progress that
just cycled 0 to 1 forever would make it meaningless on the way back.

**Stops.** `stop_index` only advances when the bus comes within five meters of
the next stop in its current direction of travel — not to the nearest stop
generally. That keeps the ETA falling monotonically as the bus approaches, in
both directions, instead of jumping when the nearest stop changes.

**Speed.** Raw distance over time between consecutive points swings wildly,
because OpenStreetMap places many points through curves and few along straight
stretches. The reported speed is exponentially smoothed against the previous
reading and capped at 60 km/h.

**Occupancy.** Generated from the hour of day: 60–100 passengers at rush hour,
30–60 midday, 0–20 overnight, 10–40 otherwise. This is the one part of the
payload that is invented rather than derived from real geometry.

**Multiple buses.** Each line's `bus_ids` list starts with its primary bus,
which departs immediately. Any others wait until the primary has passed the
first stop, so two buses never leave the terminal stacked on top of each other.
A line can also list `return_bus_ids`, which spawn at the opposite terminal
already on the return leg; nothing else starts there, so they need no stagger.
Every bus keeps its own position, speed and stop index in its own `BusState`.

## Adding a line

Add an entry to `LINES` at the top of `simulator.py` with the line's bus ids,
display name, route id, route file name, OpenStreetMap relation id, and its
ordered list of stop coordinates. Save the Overpass response for that relation
next to it as `route_<line>.json`.

The stop list has to be ordered by actual position along the polyline, not by
any numbering scheme the operator uses. Línea 15-1 originally had two stops
appended at the end of the list when they physically sit between stops 9 and
10, which broke the stop index and every ETA that depended on it.

This registry intentionally duplicates `backend/routes_config.py`. The
simulator is meant to run standalone on a Pi without the backend package
installed, so it carries its own lightweight copy. Keep the two in sync when
adding a line.

## Moving to real hardware

The path is deliberately short. Replace `build_payload()` with real readings —
position and speed from the GPS module, passenger count from whatever counter
is installed — keep the field names, and point `--backend-url` at the server.
Everything downstream stays as it is.

Two details already handled for a Pi: the script reconfigures stdout to UTF-8,
since a non-UTF-8 console would otherwise fail on the status characters it
prints, and it falls back to downloading the route from Overpass if the local
route file is missing.
