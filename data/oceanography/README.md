# Oceanography inputs

One CSV per acquisition, laid out as a wide grid: the first column holds one coordinate axis
(longitude by default), the header row holds the other (latitude), and each cell is the value
at that lattice point. Empty cells are treated as missing (cloud, swath gap).

```
longitude,25.0000,25.5085,26.0169,...
-40.0000,0.2793,0.2764,0.1339,...
-39.4915,0.2606,0.2398,0.1448,...
```

The acquisition date comes from the filename: the first `YYYYDDD` token (year and day of
year) becomes the value column `chlor_YYYYDDD`. The synthetic files are named
`AYYYYDDD_chlor_a.csv`, which is the layout of common ocean colour level 3 products. Files
with a different token position still work as long as the seven digits appear once; a
different prefix goes through `load_composites(paths, prefix=...)`.

Axis order can be swapped with `read_grid_csv_as_points(first_axis="latitude",
second_axis="longitude")`. Coordinates are assumed to be WGS84.

Season assignment uses `fortress_gis.features.seasonal.SeasonCalendar`. The default is the
two season split used for chlorophyll work (winter for day of year below 104 or from 288,
summer between); `SeasonCalendar.four_seasons()` gives meteorological seasons, and any set of
`(start, end, label)` bounds can be passed.
