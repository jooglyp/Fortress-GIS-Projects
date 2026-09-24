"""Command line entry points: ``fortress-gis <domain> ...``.

Each command runs one domain workflow on files under ``data/<domain>/`` (or paths you give it)
and writes a QGIS bundle plus a Kepler.gl HTML under ``exports/<domain>/``. The notebooks in
``demos/`` call the same functions with more commentary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from fortress_gis import log
from fortress_gis.config import domain_data_dir, domain_export_dir

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    log.init("DEBUG" if verbose else "INFO")


@app.command()
def hydrology(
    dem: Annotated[
        list[Path] | None, typer.Option(help="DEM tile(s). Default: data/hydrology/*.tif")
    ] = None,
    pour_x: Annotated[float | None, typer.Option(help="Pour point x in DEM CRS.")] = None,
    pour_y: Annotated[float | None, typer.Option(help="Pour point y in DEM CRS.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output folder. Default: exports/hydrology")
    ] = None,
    no_kepler: bool = False,
) -> None:
    """Delineate the watershed above a pour point and export it."""
    import numpy as np

    from fortress_gis.domains import hydrology as hy

    tiles = dem or sorted(domain_data_dir("hydrology").glob("*.tif"))
    if not tiles:
        raise typer.BadParameter(
            "No DEM tiles found; pass --dem or place GeoTIFFs under data/hydrology/"
        )
    raster = hy.load_dem(tiles)
    if pour_x is None or pour_y is None:
        # default: the DEM's lowest edge cell, which is where a basin drains
        edge = np.full(raster.shape, np.nan)
        m = raster.masked()
        edge[0, :], edge[-1, :], edge[:, 0], edge[:, -1] = m[0, :], m[-1, :], m[:, 0], m[:, -1]
        r, c = np.unravel_index(np.nanargmin(edge), edge.shape)
        xs, ys = raster.cell_centers()
        pour_x, pour_y = float(xs[r, c]), float(ys[r, c])
        typer.echo(f"Pour point not given; using lowest edge cell at ({pour_x:.1f}, {pour_y:.1f})")
    result = hy.run_watershed_pipeline(raster, (pour_x, pour_y))
    typer.echo(result.summary().to_string())
    paths = hy.export_artifacts(
        result, out or domain_export_dir("hydrology"), kepler_html=not no_kepler
    )
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


@app.command()
def oceanography(
    grids: Annotated[
        list[Path] | None, typer.Option(help="Grid CSVs. Default: data/oceanography/*.csv")
    ] = None,
    hex_radius_m: Annotated[float | None, typer.Option(help="Hex circumradius in metres.")] = None,
    method: Annotated[str, typer.Option(help="gstar or lisa")] = "gstar",
    permutations: int = 499,
    out: Path | None = None,
    no_kepler: bool = False,
) -> None:
    """Aggregate gridded composites to hexes and map hotspots of the seasonal covariates."""
    from fortress_gis.domains import oceanography as oc

    files = grids or sorted(domain_data_dir("oceanography").glob("*.csv"))
    if not files:
        raise typer.BadParameter(
            "No grid CSVs found; pass --grids or place them under data/oceanography/"
        )
    points = oc.load_composites(files)
    stack = oc.build_hex_covariates(points, hex_radius_m=hex_radius_m)
    attrs = [c for c in stack.covariate_columns if c.endswith("_mean")]
    results = oc.screen_hotspots(stack.covariates, attrs, method=method, permutations=permutations)
    for attr, res in results.items():
        typer.echo(f"{attr}: {res.summary().to_dict()}")
    paths = oc.export_artifacts(
        stack, results, out or domain_export_dir("oceanography"), kepler_html=not no_kepler
    )
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


@app.command()
def airspace(
    positions: Annotated[
        Path | None,
        typer.Option(
            help="Parquet/CSV of positions. Default: data/airspace/aircraft_positions.parquet"
        ),
    ] = None,
    hours: Annotated[tuple[int, int], typer.Option(help="UTC hour band [start, end)")] = (7, 19),
    max_altitude_ft: float = 500.0,
    minutes_per_hex: float = 5.0,
    dask: Annotated[
        bool,
        typer.Option(
            "--dask/--no-dask", help="Read and detect with dask; prints the dashboard link."
        ),
    ] = False,
    out: Path | None = None,
    no_kepler: bool = False,
) -> None:
    """Detect proximity events between tracks and test hexes for excess risk."""
    from fortress_gis.domains import airspace as air
    from fortress_gis.features.proximity import ProximityConfig

    src = positions or domain_data_dir("airspace") / "aircraft_positions.parquet"
    if not src.exists():
        raise typer.BadParameter(f"{src} not found")
    if dask:
        from fortress_gis.compute.cluster import get_dask_client

        get_dask_client()
    flt = air.AirspaceFilter(hours=hours, altitude_ft=(0.0, max_altitude_ft))
    result = air.run_airspace_pipeline(
        src, flt=flt, config=ProximityConfig(), minutes_per_hex=minutes_per_hex, use_dask=dask
    )
    typer.echo(result.summary().to_string())
    paths = air.export_artifacts(
        result, out or domain_export_dir("airspace"), kepler_html=not no_kepler
    )
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


@app.command()
def maritime(
    telemetry: Annotated[
        Path | None,
        typer.Option(help="CSV/parquet of telemetry. Default: data/maritime/vessel_telemetry.*"),
    ] = None,
    target: str = "fuel_demand_kg_h",
    cv: int = 5,
    n_iter: int = 20,
    out: Path | None = None,
    no_kepler: bool = False,
) -> None:
    """Fit inference and predictive models of a telemetry target and map the voyage."""
    import pandas as pd

    from fortress_gis.domains import maritime as mar

    if telemetry is None:
        folder = domain_data_dir("maritime")
        candidates = [folder / "vessel_telemetry.parquet", folder / "vessel_telemetry.csv"]
        src = next((c for c in candidates if c.exists()), candidates[0])
    else:
        src = telemetry
    if not src.exists():
        raise typer.BadParameter(f"{src} not found")
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    result = mar.run_maritime_pipeline(df, target=target, cv=cv, n_iter=n_iter)
    typer.echo(result.summary().to_string())
    typer.echo(result.leaderboard.to_string(index=False))
    paths = mar.export_artifacts(
        result, df, out or domain_export_dir("maritime"), kepler_html=not no_kepler
    )
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()
