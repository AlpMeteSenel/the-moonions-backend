#!/usr/bin/env bash
# Downloads and precaches all public lunar datasets used by the backend.
# Re-runnable: skips files already present.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
get() { local url="$1" out="${2:-$(basename "$1")}"; [ -s "$out" ] && { echo "have $out"; return; }; echo "get  $out"; curl -fsSL --retry 3 -C - -o "$out.part" "$url" && mv "$out.part" "$out"; }
SVS=https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720
LOLA=https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr
NAIF=https://naif.jpl.nasa.gov/pub/naif/generic_kernels
# Ephemerides: Sun/Earth/Moon positions + lunar principal-axis orientation
get https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp
get $NAIF/pck/moon_pa_de421_1900-2050.bpc
get $NAIF/fk/satellites/moon_080317.tf
get $NAIF/pck/pck00011.tpc || true
# NASA SVS CGI Moon Kit: LROC WAC colour mosaic + LOLA displacement (for 3D globe)
get $SVS/lroc_color_poles_8k.tif
get $SVS/ldem_16.tif
# LOLA GDR global DEM at 64 px/deg (~474 m/px), int16 metres*0.5 offset from 1737.4 km
get $LOLA/cylindrical/img/ldem_64.img
get $LOLA/cylindrical/img/ldem_64.lbl
# LOLA polar DEMs (80 m/px) for high-detail south & north pole analysis
get $LOLA/polar/img/ldem_80s_80m.img
get $LOLA/polar/img/ldem_80s_80m.lbl
get $LOLA/polar/img/ldem_80n_80m.img
get $LOLA/polar/img/ldem_80n_80m.lbl
# LOLA polar 20 m south (lat <= -80): ground truth for AI training labels (1.85 GB)
get $LOLA/polar/img/ldem_80s_20m.img
get $LOLA/polar/img/ldem_80s_20m.lbl
# Robbins 2018 global crater database (~1.3 M craters)
get https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/lunar_crater_database_robbins_2018 robbins_2018.zip || echo "WARN: crater DB download failed"
echo done
