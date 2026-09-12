# Modular Asset Slots & Customization Guide

This directory contains modular graphic and animation asset slots for the **Montesinho Wildfire Response 3D Command Center**.

The dashboard automatically checks and loads assets from these standardized slots. You can hot-swap or drop in custom company logos, drone models, custom Lottie animations, 3D textures, or tactical crew insignia without modifying code.

---

## Directory Layout

```
web/static/assets/
├── icons/
│   ├── crew_marker.svg      # Response crew marker / tactical shield icon
│   ├── flame_icon.svg       # High-risk wildfire thermal icon
│   ├── drone_icon.svg       # Aerial reconnaissance drone icon
│   ├── shield_icon.svg      # 4-crew cell quota constraint indicator
│   └── wind_icon.svg        # Meteorological wind velocity vector icon
├── images/
│   └── montesinho_terrain.svg # Topographic contour map overlay of Montesinho Park
├── animations/
│   ├── README.md            # Instructions for custom Lottie / WebM / CSS animations
│   ├── radar_sweep.svg      # Real-time tactical radar sweep overlay animation
│   └── fire_pulse.svg       # Thermal hotspot heartbeat / pulse animation
└── custom_images/
    └── README.md            # Drop folder for custom incident commander logos or map textures
```

---

## How to Customize Assets

1. **Replace Crew Marker**:
   - Place your custom icon (e.g. fire engine, helicopter, wildland firefighter badge) at `web/static/assets/icons/crew_marker.svg` or `crew_marker.png`.
2. **Add Custom Animations**:
   - Drop custom animations (SVG, Lottie JSON, GIF, or WebM) into `web/static/assets/animations/`.
3. **Add Agency / Incident Badges**:
   - Drop agency logos or satellite terrain ortho-imagery into `web/static/assets/custom_images/`.
4. **Instant Verification**:
   - The web server's `/api/assets` endpoint automatically enumerates all detected assets and reports their status to the frontend HUD.
