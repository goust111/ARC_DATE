from __future__ import annotations

import random
from typing import Any

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
PALETTE = tuple(range(10))
MAX_GEN_ATTEMPTS = 500
_GEN_COUNTER = 0


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _paint_vertical(
    out: np.ndarray,
    r0: int,
    r1: int,
    c: int,
    color: int,
    *,
    bg_only: int | None = None,
) -> None:
    h, w = out.shape
    cc = int(c)
    if not (0 <= cc < w):
        return
    for rr in range(max(0, int(r0)), min(h - 1, int(r1)) + 1):
        if bg_only is None or int(out[rr, cc]) == int(bg_only):
            out[rr, cc] = int(color)


def _make_basin(
    c0: int,
    bottom_row: int,
    inner_width: int,
    depth: int,
    short_side: str,
    tall_extra: int,
) -> dict[str, int | str]:
    inner_width = int(inner_width)
    depth = int(depth)
    bottom_row = int(bottom_row)
    c0 = int(c0)
    c1 = c0 + inner_width + 1
    short_h = depth + 1
    tall_h = short_h + int(tall_extra)
    if short_side == "left":
        left_h, right_h = short_h, tall_h
        spill_col = c0 - 1
    else:
        left_h, right_h = tall_h, short_h
        spill_col = c1 + 1
    left_top = bottom_row - left_h + 1
    right_top = bottom_row - right_h + 1
    spill_row = bottom_row - depth
    return {
        "c0": c0,
        "c1": c1,
        "bottom_row": bottom_row,
        "inner_width": inner_width,
        "depth": depth,
        "capacity": inner_width * depth,
        "short_side": short_side,
        "spill_col": spill_col,
        "spill_row": spill_row,
        "left_top": left_top,
        "right_top": right_top,
        "left_h": left_h,
        "right_h": right_h,
        "top_row": min(left_top, right_top),
    }


def _draw_basin(grid: np.ndarray, basin: dict[str, int | str], wall_color: int) -> None:
    r = int(basin["bottom_row"])
    c0 = int(basin["c0"])
    c1 = int(basin["c1"])
    lt = int(basin["left_top"])
    rt = int(basin["right_top"])
    grid[r, c0 : c1 + 1] = int(wall_color)
    grid[lt : r + 1, c0] = int(wall_color)
    grid[rt : r + 1, c1] = int(wall_color)


def _parse_basin_component(coords: np.ndarray) -> dict[str, int | str] | None:
    r0, c0, r1, c1 = _bbox_coords(coords)
    if c1 - c0 < 2:
        return None
    cells = {(int(r), int(c)) for r, c in coords}
    if any((r1, c) not in cells for c in range(c0, c1 + 1)):
        return None
    left_rows = sorted(int(r) for r, c in coords if int(c) == c0)
    right_rows = sorted(int(r) for r, c in coords if int(c) == c1)
    if not left_rows or not right_rows:
        return None
    if left_rows != list(range(left_rows[0], r1 + 1)):
        return None
    if right_rows != list(range(right_rows[0], r1 + 1)):
        return None
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr == r1:
            continue
        if cc not in (c0, c1):
            return None
    left_h = r1 - left_rows[0] + 1
    right_h = r1 - right_rows[0] + 1
    if left_h == right_h:
        return None
    depth = min(left_h, right_h) - 1
    if depth < 1:
        return None
    short_side = "left" if left_h < right_h else "right"
    basin = _make_basin(c0, r1, c1 - c0 - 1, depth, short_side, abs(left_h - right_h))
    if int(basin["left_top"]) != left_rows[0]:
        return None
    if int(basin["right_top"]) != right_rows[0]:
        return None
    if int(basin["top_row"]) != r0:
        return None
    if len(cells) != (c1 - c0 + 1) + (left_h - 1) + (right_h - 1):
        return None
    return basin


def _infer_scene(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 2, "expected exactly two foreground colors"

    wall_candidates: list[tuple[int, list[dict[str, int | str]]]] = []
    for color in colors:
        components = _components(g == int(color), connectivity=4)
        if not components:
            continue
        parsed: list[dict[str, int | str]] = []
        ok = True
        for comp in components:
            basin = _parse_basin_component(comp)
            if basin is None:
                ok = False
                break
            parsed.append(basin)
        if ok:
            parsed.sort(key=lambda b: (int(b["spill_row"]), int(b["c0"])))
            wall_candidates.append((int(color), parsed))
    assert len(wall_candidates) == 1, "wall color must be uniquely identifiable by U-shapes"
    wall_color, basins = wall_candidates[0]
    water_color = next(int(c) for c in colors if int(c) != int(wall_color))

    sources: list[dict[str, Any]] = []
    for comp in _components(g == water_color, connectivity=8):
        r0, c0, r1, c1 = _bbox_coords(comp)
        cols = sorted({int(c) for _r, c in comp})
        if len(cols) == 1:
            drop_col = cols[0]
        else:
            assert (c1 - c0) % 2 == 0, "wide sources must have a unique center column"
            drop_col = (c0 + c1) // 2
        assert r1 + 1 < h, "sources must hang above the scene"
        sources.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "volume": int(comp.shape[0]),
                "drop_col": int(drop_col),
                "start_row": int(r1 + 1),
            }
        )
    assert sources, "expected at least one source component"
    sources.sort(key=lambda s: (int(s["bbox"][0]), int(s["drop_col"])))

    return {
        "grid": g,
        "shape": (h, w),
        "bg": bg,
        "wall_color": wall_color,
        "water_color": water_color,
        "basins": basins,
        "sources": sources,
    }


def _find_next_basin(
    basins: list[dict[str, int | str]],
    start_row: int,
    col: int,
) -> int | None:
    candidates: list[tuple[int, int, int]] = []
    for idx, basin in enumerate(basins):
        if int(basin["c0"]) < int(col) < int(basin["c1"]) and int(basin["spill_row"]) >= int(start_row):
            candidates.append((int(basin["spill_row"]), int(basin["bottom_row"]), idx))
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[0][2])


def _simulate(grid: np.ndarray, *, return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    scene = _infer_scene(grid)
    g = scene["grid"]
    h, _w = scene["shape"]
    bg = int(scene["bg"])
    water = int(scene["water_color"])
    basins = scene["basins"]
    sources = scene["sources"]

    out = g.copy()
    basin_units = [0 for _ in basins]
    source_first_hits: list[int | None] = []
    overflow_events = 0
    outflow_events = 0

    for source in sources:
        row = int(source["start_row"])
        col = int(source["drop_col"])
        volume = int(source["volume"])
        first_hit: int | None = None
        while volume > 0:
            basin_idx = _find_next_basin(basins, row, col)
            if basin_idx is None:
                _paint_vertical(out, row, h - 1, col, water, bg_only=bg)
                outflow_events += 1
                break
            basin = basins[basin_idx]
            if first_hit is None:
                first_hit = basin_idx
            _paint_vertical(out, row, int(basin["spill_row"]) - 1, col, water, bg_only=bg)
            basin_units[basin_idx] += volume
            capacity = int(basin["capacity"])
            if basin_units[basin_idx] > capacity:
                volume = basin_units[basin_idx] - capacity
                basin_units[basin_idx] = capacity
                row = int(basin["spill_row"])
                col = int(basin["spill_col"])
                overflow_events += 1
            else:
                volume = 0
        source_first_hits.append(first_hit)

    fill_layers: list[int] = []
    for idx, basin in enumerate(basins):
        width = int(basin["inner_width"])
        units = int(basin_units[idx])
        assert units % width == 0, "ambiguous partial layer would violate uniqueness"
        layers = units // width
        assert 0 <= layers <= int(basin["depth"])
        fill_layers.append(layers)
        for layer in range(layers):
            rr = int(basin["bottom_row"]) - 1 - layer
            out[rr, int(basin["c0"]) + 1 : int(basin["c1"])] = water

    meta = {
        "bg": bg,
        "wall_color": int(scene["wall_color"]),
        "water_color": water,
        "basins": basins,
        "sources": sources,
        "basin_units": basin_units,
        "fill_layers": fill_layers,
        "overflow_events": overflow_events,
        "outflow_events": outflow_events,
        "source_first_hits": source_first_hits,
    }
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each source's conserved water volume through cascaded basins, level-filling each basin before spilling from its shorter wall.

    Input structure:
    - The grid has a dominant background, one wall color forming several disconnected upward-open U-shaped basins, and one water color forming one or more hanging source components.
    - In every basin the two side walls have different heights, so the lower rim uniquely determines the spill side.
    - Each source component stands above open air and has a single vertical drop column beneath it.

    Rule:
    - Treat the pixel count of each water source component as its conserved water volume.
    - Paint a vertical falling trace only through background cells until the stream reaches the first basin whose opening contains that column.
    - Inside a basin, store water by full horizontal layers from the bottom upward; if the basin reaches the shorter wall's top and volume remains, send all leftover units down the outside column next to that shorter wall and repeat.
    - Keep all basin walls unchanged and add the final leveled water plus every air-column fall trace.

    Key insight:
    - The source component size is the quantity that propagates through the cascade, while the geometry of each U-shape converts that quantity into a local capacity and a unique overflow direction.
    """
    return _simulate(np.asarray(grid, dtype=int), return_meta=False)


def _copy_basin(basin: dict[str, int | str], dr: int, dc: int) -> dict[str, int | str]:
    moved = dict(basin)
    for key in ("c0", "c1", "spill_col"):
        moved[key] = int(moved[key]) + int(dc)
    for key in ("bottom_row", "spill_row", "left_top", "right_top", "top_row"):
        moved[key] = int(moved[key]) + int(dr)
    return moved


def _shift_chain(chain: dict[str, Any], dr: int, dc: int) -> dict[str, Any]:
    return {
        "source": {
            "r0": int(chain["source"]["r0"]) + int(dr),
            "r1": int(chain["source"]["r1"]) + int(dr),
            "col": int(chain["source"]["col"]) + int(dc),
            "volume": int(chain["source"]["volume"]),
        },
        "basins": [_copy_basin(basin, dr, dc) for basin in chain["basins"]],
        "incoming": list(chain["incoming"]),
        "final_mode": str(chain["final_mode"]),
    }


def _chain_reserved_bbox(chain: dict[str, Any]) -> tuple[int, int, int, int]:
    source = chain["source"]
    min_r = int(source["r0"])
    max_r = int(source["r1"])
    min_c = int(source["col"])
    max_c = int(source["col"])
    for basin in chain["basins"]:
        min_r = min(min_r, int(basin["top_row"]))
        max_r = max(max_r, int(basin["bottom_row"]))
        min_c = min(min_c, int(basin["c0"]), int(basin["spill_col"]))
        max_c = max(max_c, int(basin["c1"]), int(basin["spill_col"]))
    return min_r, min_c, max_r, max_c


def _build_chain(
    length: int,
    final_mode: str,
    *,
    max_source_units: int,
    require_multi_layer: bool,
    directions: list[str] | None = None,
) -> dict[str, Any] | None:
    for _ in range(300):
        specs: list[dict[str, int | str]] = []
        for idx in range(length):
            short_side = directions[idx] if directions is not None else random.choice(["left", "right"])
            if length >= 3 and idx < length - 1:
                inner_width = 2
                depth = 1
            else:
                inner_width = random.choice([2, 3])
                depth = random.choice([1, 2])
            tall_extra = random.choice([1, 2])
            specs.append(
                {
                    "inner_width": inner_width,
                    "depth": depth,
                    "short_side": short_side,
                    "tall_extra": tall_extra,
                    "capacity": inner_width * depth,
                }
            )

        if final_mode == "contained":
            final_layers = random.randint(1, int(specs[-1]["depth"]))
            incoming_last = int(specs[-1]["inner_width"]) * final_layers
        else:
            incoming_last = int(specs[-1]["capacity"]) + random.randint(1, 2)

        incoming = [0] * length
        incoming[-1] = int(incoming_last)
        for idx in range(length - 2, -1, -1):
            incoming[idx] = int(specs[idx]["capacity"]) + int(incoming[idx + 1])

        source_units = int(incoming[0])
        if not (2 <= source_units <= max_source_units):
            continue

        source = {"r0": 0, "r1": source_units - 1, "col": 0, "volume": source_units}
        basins: list[dict[str, int | str]] = []
        prev_marker_row = int(source["r1"])
        incoming_col = int(source["col"])
        for idx, spec in enumerate(specs):
            tall_extra = int(spec["tall_extra"])
            gap = random.randint(tall_extra, tall_extra + (2 if idx == 0 else 3))
            spill_row = prev_marker_row + gap + 1
            bottom_row = spill_row + int(spec["depth"])
            hit_offset = random.randint(1, int(spec["inner_width"]))
            c0 = incoming_col - hit_offset
            basin = _make_basin(
                c0,
                bottom_row,
                int(spec["inner_width"]),
                int(spec["depth"]),
                str(spec["short_side"]),
                tall_extra,
            )
            basins.append(basin)
            prev_marker_row = int(basin["spill_row"])
            incoming_col = int(basin["spill_col"])

        fills = [min(int(v), int(b["capacity"])) for v, b in zip(incoming, basins)]
        fill_layers = [fill // int(basin["inner_width"]) for fill, basin in zip(fills, basins)]
        if require_multi_layer and max(fill_layers) < 2:
            continue

        chain = {"source": source, "basins": basins, "incoming": incoming, "final_mode": final_mode}
        _r0, c0, r1, c1 = _chain_reserved_bbox(chain)
        if c1 - c0 + 1 > 12 or r1 + 3 > 28:
            continue
        return chain
    return None


def _draw_scene(
    chains: list[dict[str, Any]],
    bg: int,
    wall: int,
    water: int,
    *,
    bottom_margin: int,
    right_margin: int,
) -> np.ndarray:
    max_r = 0
    max_c = 0
    for chain in chains:
        _r0, _c0, r1, c1 = _chain_reserved_bbox(chain)
        max_r = max(max_r, int(r1))
        max_c = max(max_c, int(c1))
    grid = np.full((max_r + 1 + bottom_margin, max_c + 1 + right_margin), int(bg), dtype=int)
    for chain in chains:
        src = chain["source"]
        grid[int(src["r0"]) : int(src["r1"]) + 1, int(src["col"])] = int(water)
        for basin in chain["basins"]:
            _draw_basin(grid, basin, wall)
    return grid


def _expected_from_chains(grid: np.ndarray, chains: list[dict[str, Any]], bg: int, water: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, _w = out.shape
    basin_fills: list[tuple[dict[str, int | str], int]] = []
    for chain in chains:
        src = chain["source"]
        row = int(src["r1"]) + 1
        col = int(src["col"])
        volume = int(src["volume"])
        for basin in chain["basins"]:
            _paint_vertical(out, row, int(basin["spill_row"]) - 1, col, water, bg_only=bg)
            fill_units = min(volume, int(basin["capacity"]))
            basin_fills.append((basin, fill_units))
            if volume > int(basin["capacity"]):
                volume -= int(basin["capacity"])
                row = int(basin["spill_row"])
                col = int(basin["spill_col"])
            else:
                volume = 0
                break
        if volume > 0:
            _paint_vertical(out, row, h - 1, col, water, bg_only=bg)
    for basin, fill_units in basin_fills:
        layers = fill_units // int(basin["inner_width"])
        for layer in range(layers):
            rr = int(basin["bottom_row"]) - 1 - layer
            out[rr, int(basin["c0"]) + 1 : int(basin["c1"])] = int(water)
    return out


def _validate_scene(grid: np.ndarray, chains: list[dict[str, Any]], bg: int, wall: int, water: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    expected = _expected_from_chains(grid, chains, bg, water)
    try:
        actual, meta = _simulate(grid, return_meta=True)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if meta["wall_color"] != wall or meta["water_color"] != water:
        return False
    if len(meta["basins"]) != sum(len(chain["basins"]) for chain in chains):
        return False
    if len(meta["sources"]) != len(chains):
        return False
    if len({int(chain["source"]["col"]) for chain in chains}) != len(chains):
        return False
    if any(hit is None for hit in meta["source_first_hits"]):
        return False
    if len(set(int(hit) for hit in meta["source_first_hits"])) != len(meta["source_first_hits"]):
        return False
    if meta["overflow_events"] < 1:
        return False
    if max(meta["fill_layers"]) < 2:
        return False
    if np.count_nonzero(grid != bg) >= grid.size * 0.4:
        return False
    return True


def _generate_mode(mode: int) -> np.ndarray | None:
    bg, wall, water = random.sample(PALETTE, 3)

    if mode == 0:
        chains = [
            _build_chain(
                2,
                "contained",
                max_source_units=8,
                require_multi_layer=True,
                directions=["left", "right"],
            )
        ]
    elif mode == 1:
        chains = [
            _build_chain(
                2,
                "outflow",
                max_source_units=8,
                require_multi_layer=False,
                directions=["right", "right"],
            ),
            _build_chain(
                1,
                "contained",
                max_source_units=6,
                require_multi_layer=True,
                directions=[random.choice(["left", "right"])],
            ),
        ]
    elif mode == 2:
        chains = [
            _build_chain(
                1,
                "contained",
                max_source_units=6,
                require_multi_layer=True,
                directions=[random.choice(["left", "right"])],
            ),
            _build_chain(
                2,
                "contained",
                max_source_units=8,
                require_multi_layer=False,
                directions=["right", "left"],
            ),
        ]
    else:
        chains = [
            _build_chain(
                3,
                "contained",
                max_source_units=9,
                require_multi_layer=True,
                directions=["left", "right", "left"],
            )
        ]

    if any(chain is None for chain in chains):
        return None
    chains = [chain for chain in chains if chain is not None]
    random.shuffle(chains)

    placed: list[dict[str, Any]] = []
    cursor_c = random.randint(1, 2)
    for chain in chains:
        r_shift = random.randint(0, 2)
        _r0, c0, _r1, c1 = _chain_reserved_bbox(chain)
        moved = _shift_chain(chain, r_shift, cursor_c - c0)
        placed.append(moved)
        _mr0, _mc0, _mr1, mc1 = _chain_reserved_bbox(moved)
        cursor_c = mc1 + random.randint(5, 7)

    grid = _draw_scene(
        placed,
        bg,
        wall,
        water,
        bottom_margin=random.randint(1, 3),
        right_margin=random.randint(1, 3),
    )
    if max(grid.shape) > 30:
        return None
    if not _validate_scene(grid, placed, bg, wall, water):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    mode = _GEN_COUNTER % 4
    _GEN_COUNTER += 1
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_mode(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid conserved-water cascade scene")
