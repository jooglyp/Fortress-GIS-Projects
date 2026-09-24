# Hydrology inputs

DEM tiles as GeoTIFF. Any number of tiles; `fortress_gis.domains.hydrology.load_dem` mosaics
them with `rasterio.merge` and needs them to share a CRS and resolution. A projected CRS in
metres is expected because catchment area, flowline length and slope are reported in km and
m per km.

The synthetic tiles written by the notebook are `dem_tile_r{row}_c{col}.tif`: a 240 by 300
cell surface at 100 m in EPSG:32615, split two by two. The CLI default is every `*.tif` in
this folder.

Optional inputs the pipeline accepts but the demo does not use:

- a boundary polygon (`boundary=`) to clip the mosaic before processing
- a flowline layer (`flowlines=`) such as a national hydrography product, in any CRS, with an
  id column; elevation statistics are then sampled along those lines instead of lines traced
  from the terrain
