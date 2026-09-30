# Frontend

The dashboard: a Leaflet map with live bus positions, a sidebar with the
selected bus's ETA and occupancy, and two arrival calculators.

It is one file. `index.html` contains the markup, the styles and the logic, with
no build step, no framework and no bundler. The backend mounts this folder at
`/static` and serves `index.html` at `/`, so there is nothing to compile or
deploy separately — editing the file and reloading the page is the whole
workflow.

Two libraries load from a CDN: Leaflet 1.9.4 for the map, and Inter and Space
Grotesk from Google Fonts. Map tiles come from OpenStreetMap. Everything else is
first-party.

## How it gets its data

On load the page does four things:

1. Fetches `GET /route/lines` to populate the line picker.
2. Fetches `GET /route?line=38` for the polyline, the stop coordinates and the
   street names, and draws them.
3. Opens a WebSocket to `/ws`.
4. Renders a "start the simulator" hint until the first message arrives.

From then on it is push-driven. Every prediction the backend broadcasts arrives
on the WebSocket and is cached per bus id. Buses belonging to the line currently
being viewed get their marker moved; the rest are kept in memory so that
switching lines can redraw them instantly. If the socket closes, the connection
indicator turns grey and the page retries every three seconds.

The route fetch has its own retry, backing off up to five attempts, because it
happens during the Render instance's cold start often enough to matter.

Two calls go the other way. `POST /predict/occupancy/for-viewer` recomputes the
selected bus's occupancy against the browser's local clock, and
`POST /predict/trip` answers both arrival calculators.

## Reading the map

Everything on screen encodes something, and the color assignments avoid
collisions so that two pieces of information are never confused for each other.

- **The bus** is a circle. Its fill is the occupancy class — green, amber,
  orange, red — and its border and pulse are the travel direction: teal
  outbound, lime on the return leg. Direction is a property of the moment, not
  of the bus, so a bus changes color when it turns around at a terminal. The
  selected bus is drawn larger.
- **The route line** is colored per line, cyan for Línea 38 and violet for
  Línea 15-1, drawn over a dark outline so it stays legible on light tiles.
- **Stops** are small indigo dots. The one the bus is heading for turns amber
  and pulses faster. Hovering a stop shows its street name.
- **Selection rings** mark stops chosen in a calculator: pink for "when does my
  bus arrive", red for the trip planner's origin and destination. Only one
  group's rings show at a time, so a stop never wears both.
- **Your location** is a blue dot, from geolocation or from tapping the map.

Clicking a bus selects it, and the entire sidebar then follows that bus. There
is no separate bus dropdown — the map is the picker.

## The sidebar

**Line picker** switches lines, which reloads the route, clears the old
markers, selects the new line's primary bus and rebuilds markers for any buses
already known on it.

**ETA and next stop** for the selected bus, updated on every broadcast.
Durations render as seconds under a minute, minutes and seconds above it, hours
and minutes past an hour — never as a bare "125 min".

**Occupancy** shows a class badge, a percentage bar and three person icons,
filled in proportion to how full the bus is. It is recomputed against the
viewer's own clock rather than the simulator's, so it reflects the time where
the reader is.

**When does my bus arrive?** takes a stop three ways: from the dropdown, from
browser geolocation, or by tapping a point on the map. The last two pick the
nearest stop to that point and ring it on the map with a label. Once a result is
showing it stays live, recalculating on each broadcast instead of freezing at
the moment the button was pressed.

**Trip planner** takes an origin and a destination and returns the wait, the
ride duration and the arrival time. Switching lines clears any result, since a
stop index from one line means nothing on another.

## Languages

Spanish and English, toggled from the header, with Spanish as the default.

Static text carries a `data-i18n` attribute. Text set from JavaScript —
geolocation outcomes, calculator errors — goes through a helper that records
which key produced it, so those strings retranslate on a switch too instead of
being stuck in whichever language was active when they appeared. A switch also
regenerates the map labels, the stop dropdown entries and the bus popup.

## On a phone

The layout is a CSS grid that collapses the sidebar into a full-width overlay
below 720px, with a backdrop that dismisses it. A first visit on a narrow
screen starts with the map full-bleed; after that the choice is remembered in
`localStorage`.

Height is set with `100dvh` and a `100vh` fallback. Safari measures `100vh`
with its address bar hidden, which makes the page taller than the visible area
and lets the fixed header slide under the status bar when the bar collapses;
`dvh` tracks the actual visible height instead.

Because the sidebar covers the map on a phone, tapping a bus also opens a popup
on the marker itself with its occupancy, so the answer does not require opening
the sidebar.

## Notes for editing

- Bus markers only rebuild their DOM when the occupancy, direction or selection
  actually changed. Rebuilding on every tick made markers flicker.
- Highlighting the next stop swaps two icons rather than rebuilding all 21.
- Stop icons are rebuilt through `rebuildStopIcon()`, never `makeStopIcon()`
  directly, so an existing selection ring survives the swap.
- `LINE_BUS_ID` maps a line to its primary bus, not to its only one. It sets
  which bus is selected by default on a line switch, and keys the route track
  color.
