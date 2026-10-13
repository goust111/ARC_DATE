from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
CONTROL = 6
MARKER_COLORS = (1, 2, 3, 7)
VERTICAL_STEPS = ((-1, 0), (1, 0))
ALL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    height, width = grid.shape

    for start_r, start_c in np.argwhere(mask):
        sr = int(start_r)
        sc = int(start_c)
        if visited[sr, sc]:
            continue
        stack = [(sr, sc)]
        visited[sr, sc] = True
        coords: list[Coord] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        arr = np.array(sorted(coords), dtype=int)
        components.append(
            {
                "color": int(color),
                "coords": arr,
                "bbox": _bbox_coords(arr),
                "area": int(arr.shape[0]),
            }
        )
    components.sort(key=lambda comp: tuple(map(int, comp["bbox"][:2])))
    return components


def _band_ranges(rows: list[int], height: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for idx, row in enumerate(rows):
        if idx + 1 < len(rows):
            end = (int(row) + int(rows[idx + 1])) // 2
        else:
            end = height - 1
        bands.append((int(start), int(end)))
        start = end + 1
    return bands


def _marker_records(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    colors = [
        int(color)
        for color in np.unique(grid)
        if int(color) not in (bg, GREY, CONTROL)
    ]
    markers: list[dict] = []
    height, width = grid.shape

    for color in colors:
        coords = np.argwhere(grid == int(color))
        assert coords.shape[0] == 1, f"marker color {color} must appear exactly once"
        r, c = map(int, coords[0])
        controls = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) == CONTROL
        ]
        assert len(controls) == 1, f"marker at {(r, c)} must touch exactly one control"
        control = controls[0]
        step = (r - control[0], c - control[1])
        assert abs(step[0]) + abs(step[1]) == 1, "ray direction must be cardinal"
        markers.append(
            {
                "color": int(color),
                "pos": (r, c),
                "control": control,
                "step": step,
            }
        )

    assert 3 <= len(markers) <= 4, "expected three or four markers"
    markers.sort(key=lambda marker: (marker["pos"][0], marker["pos"][1]))
    rows = [marker["pos"][0] for marker in markers]
    assert len(set(rows)) == len(rows), "marker rows must be unique"

    for marker, band in zip(markers, _band_ranges(rows, grid.shape[0])):
        marker["band"] = band
    return markers


def _ray_cells(start: Coord, step: Coord, shape: tuple[int, int]) -> set[Coord]:
    height, width = shape
    r = int(start[0]) + int(step[0])
    c = int(start[1]) + int(step[1])
    ray: set[Coord] = set()
    while 0 <= r < height and 0 <= c < width:
        ray.add((r, c))
        r += int(step[0])
        c += int(step[1])
    return ray


def _grey_objects(grid: np.ndarray) -> list[dict]:
    return _find_color_components(grid, GREY)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored marker into a horizontal band frame, then erase only the grey targets hit by that marker's ray inside the same band.

    Input structure:
    - The background is black and the foreground contains three or four single-pixel colored markers on distinct rows.
    - Every marker has one adjacent pink control pixel that points opposite the ray direction.
    - Grey pixels form several small 4-connected target objects spread across the bands induced by the marker rows.

    Rule:
    - Sort the markers by row and split the grid into horizontal bands at the floored midpoints between consecutive marker rows.
    - For each marker, color the full top and bottom rows of its band, the left and right edges of that band, and the entire marker row with the marker color.
    - Cast a straight ray from the marker away from its adjacent pink control pixel to the grid edge.
    - Erase a grey target object iff the ray touches at least one of its pixels and every pixel of that object lies within the marker's own band.
    - Remove every pink control pixel at the end; untouched grey objects remain.

    Key insight:
    - Ray intersection is only half the filter: a hit object disappears only when the whole connected grey component also belongs to the same horizontal band as the firing marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = _get_bg(g)
    markers = _marker_records(g)
    objects = _grey_objects(g)

    out = g.copy()
    height, width = g.shape

    for marker in markers:
        color = int(marker["color"])
        row, _col = marker["pos"]
        lo, hi = marker["band"]
        out[lo, :] = color
        out[hi, :] = color
        out[lo : hi + 1, 0] = color
        out[lo : hi + 1, width - 1] = color
        out[row, :] = color

    erase_indices: set[int] = set()
    for marker in markers:
        lo, hi = marker["band"]
        ray = _ray_cells(marker["pos"], marker["step"], g.shape)
        for idx, obj in enumerate(objects):
            coords = obj["coords"]
            hit = any((int(r), int(c)) in ray for r, c in coords)
            same_band = all(lo <= int(r) <= hi for r, _c in coords)
            if hit and same_band:
                erase_indices.add(idx)

    for idx in erase_indices:
        coords = objects[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for marker in markers:
        cr, cc = marker["control"]
        out[cr, cc] = bg

    return out


OBJECT_PATTERNS: tuple[tuple[Coord, ...], ...] = (
    ((0, 0), (1, 0)),
    ((0, 0), (-1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (0, -1)),
    ((0, 0), (-1, 0), (1, 0)),
    ((0, 0), (0, -1), (0, 1)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (1, 0), (0, -1)),
    ((0, 0), (-1, 0), (0, 1)),
    ((0, 0), (-1, 0), (0, -1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, -1), (1, -1)),
    ((0, 0), (0, 1), (-1, 1)),
    ((0, 0), (0, -1), (-1, -1)),
    ((0, 0), (1, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (0, -1), (1, -1)),
    ((0, 0), (-1, 0), (0, 1), (-1, 1)),
    ((0, 0), (-1, 0), (0, -1), (-1, -1)),
    ((0, 0), (-1, 0), (1, 0), (0, 1)),
    ((0, 0), (-1, 0), (1, 0), (0, -1)),
    ((0, 0), (0, -1), (0, 1), (1, 0)),
    ((0, 0), (0, -1), (0, 1), (-1, 0)),
)


def _expand8(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[1:, 1:] |= mask[:-1, :-1]
    expanded[1:, :-1] |= mask[:-1, 1:]
    expanded[:-1, 1:] |= mask[1:, :-1]
    expanded[:-1, :-1] |= mask[1:, 1:]
    return expanded


def _choose_marker_rows(height: int, n_markers: int) -> tuple[list[int], list[tuple[int, int]]] | None:
    for _ in range(200):
        rows = []
        for idx in range(n_markers):
            base = int(round((idx + 0.5) * height / n_markers - 0.5))
            rows.append(base + random.randint(-1, 1))
        rows = [max(1, min(height - 2, row)) for row in rows]
        if len(set(rows)) != n_markers:
            continue
        rows.sort()
        if any(rows[idx + 1] - rows[idx] < 4 for idx in range(n_markers - 1)):
            continue
        bands = _band_ranges(rows, height)
        if any(hi - lo + 1 < 5 for lo, hi in bands):
            continue
        if any(not (lo + 2 <= row <= hi - 2) for row, (lo, hi) in zip(rows, bands)):
            continue
        return rows, bands
    return None


def _choose_marker_cols(width: int, n_markers: int) -> list[int] | None:
    for _ in range(200):
        pool = list(range(2, width - 2))
        random.shuffle(pool)
        chosen: list[int] = []
        for col in pool:
            if all(abs(col - other) >= 3 for other in chosen):
                chosen.append(col)
                if len(chosen) == n_markers:
                    random.shuffle(chosen)
                    return chosen
    return None


def _band_mask(
    shape: tuple[int, int],
    band: tuple[int, int],
    forbidden_rows: set[int],
    forbidden_cols: set[int],
    include_rows: set[int] | None = None,
) -> np.ndarray:
    height, width = shape
    lo, hi = band
    mask = np.zeros(shape, dtype=bool)
    r0 = max(1, lo + 1)
    r1 = min(height - 2, hi - 1)
    if r0 > r1 or width < 3:
        return mask
    mask[r0 : r1 + 1, 1 : width - 1] = True
    include_rows = set() if include_rows is None else {int(row) for row in include_rows}
    for row in include_rows:
        if lo <= row <= hi and 0 <= row < height:
            mask[row, 1 : width - 1] = True
    for row in forbidden_rows - include_rows:
        if 0 <= row < height:
            mask[row, :] = False
    for col in forbidden_cols:
        if 0 <= col < width:
            mask[:, col] = False
    return mask


def _pattern_cells(seed: Coord, pattern: tuple[Coord, ...]) -> list[Coord]:
    return [(int(seed[0]) + dr, int(seed[1]) + dc) for dr, dc in pattern]


def _place_object(
    grid: np.ndarray,
    allowed_mask: np.ndarray,
    seeds: list[Coord],
) -> tuple[Coord, ...] | None:
    if not seeds:
        return None
    blocked = _expand8(grid != BG)
    seeds = seeds[:]
    random.shuffle(seeds)
    patterns = list(OBJECT_PATTERNS)
    for seed in seeds:
        random.shuffle(patterns)
        for pattern in patterns:
            cells = _pattern_cells(seed, pattern)
            if len(set(cells)) != len(cells):
                continue
            if any(
                not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or not allowed_mask[r, c] or blocked[r, c]
                for r, c in cells
            ):
                continue
            for r, c in cells:
                grid[r, c] = GREY
            return tuple(sorted(cells))
    return None


def _hit_seeds(marker: dict, shape: tuple[int, int]) -> list[Coord]:
    height, width = shape
    row, col = marker["pos"]
    lo, hi = marker["band"]
    dr, dc = marker["step"]
    if dr < 0:
        return [(r, col) for r in range(lo + 1, row)]
    if dr > 0:
        return [(r, col) for r in range(row + 1, hi)]
    if dc < 0:
        return [(row, c) for c in range(1, col)]
    return [(row, c) for c in range(col + 1, width - 1)]


def _object_band_index(obj: tuple[Coord, ...], bands: list[tuple[int, int]]) -> int:
    rows = {r for r, _c in obj}
    for idx, (lo, hi) in enumerate(bands):
        if all(lo <= row <= hi for row in rows):
            return idx
    raise AssertionError("object must belong to exactly one band")


def _build_case() -> tuple[np.ndarray, dict] | None:
    n_markers = random.choice((3, 4))
    height = random.randint(18, 24) if n_markers == 3 else random.randint(22, 28)
    width = random.randint(15, 20)

    row_setup = _choose_marker_rows(height, n_markers)
    if row_setup is None:
        return None
    rows, bands = row_setup

    cols = _choose_marker_cols(width, n_markers)
    if cols is None:
        return None

    colors = random.sample(list(MARKER_COLORS), n_markers)
    grid = np.zeros((height, width), dtype=int)

    if random.random() < 0.5:
        cross_source = random.randrange(0, n_markers - 1)
        cross_step = (1, 0)
        cross_target = cross_source + 1
    else:
        cross_source = random.randrange(1, n_markers)
        cross_step = (-1, 0)
        cross_target = cross_source - 1

    steps = [random.choice(ALL_STEPS) for _ in range(n_markers)]
    steps[cross_source] = cross_step

    markers: list[dict] = []
    for idx in range(n_markers):
        row = rows[idx]
        col = cols[idx]
        color = colors[idx]
        step = steps[idx]
        control = (row - step[0], col - step[1])
        if not (0 <= control[0] < height and 0 <= control[1] < width):
            return None
        grid[row, col] = color
        grid[control] = CONTROL
        markers.append(
            {
                "color": color,
                "pos": (row, col),
                "control": control,
                "step": step,
                "band": bands[idx],
            }
        )

    all_marker_rows = {row for row in rows}
    all_marker_cols = {col for col in cols}

    same_band_hit: list[tuple[Coord, ...]] = []
    for marker in markers:
        marker_col = marker["pos"][1]
        is_vertical = marker["step"][0] != 0
        band_allowed = _band_mask(
            grid.shape,
            marker["band"],
            forbidden_rows=all_marker_rows,
            forbidden_cols=(all_marker_cols - {marker_col}) if is_vertical else all_marker_cols,
            include_rows={marker["pos"][0]} if not is_vertical else None,
        )
        obj = _place_object(grid, band_allowed, _hit_seeds(marker, grid.shape))
        if obj is None:
            return None
        same_band_hit.append(obj)

    source_col = markers[cross_source]["pos"][1]
    cross_allowed = _band_mask(
        grid.shape,
        markers[cross_target]["band"],
        forbidden_rows=all_marker_rows,
        forbidden_cols=all_marker_cols - {source_col},
    )
    cross_lo, cross_hi = markers[cross_target]["band"]
    cross_seeds = [(row, source_col) for row in range(cross_lo + 1, cross_hi)]
    cross_object = _place_object(grid, cross_allowed, cross_seeds)
    if cross_object is None:
        return None

    same_band_unhit: list[tuple[Coord, ...]] = []
    for marker in markers:
        band_allowed = _band_mask(
            grid.shape,
            marker["band"],
            forbidden_rows=all_marker_rows,
            forbidden_cols=all_marker_cols,
        )
        seeds = [tuple(map(int, coord)) for coord in np.argwhere(band_allowed)]
        obj = _place_object(grid, band_allowed, seeds)
        if obj is None:
            return None
        same_band_unhit.append(obj)

    extras: list[tuple[Coord, ...]] = []
    extra_limit = 2 if n_markers == 3 else 1
    for _ in range(random.randint(0, extra_limit)):
        band_order = list(range(n_markers))
        random.shuffle(band_order)
        placed = None
        for band_idx in band_order:
            allowed = _band_mask(
                grid.shape,
                markers[band_idx]["band"],
                forbidden_rows=all_marker_rows,
                forbidden_cols=all_marker_cols,
            )
            seeds = [tuple(map(int, coord)) for coord in np.argwhere(allowed)]
            placed = _place_object(grid, allowed, seeds)
            if placed is not None:
                extras.append(placed)
                break
        if placed is None:
            return None

    metadata = {
        "markers": markers,
        "bands": bands,
        "same_band_hit": same_band_hit,
        "same_band_unhit": same_band_unhit,
        "cross_source": cross_source,
        "cross_target": cross_target,
        "cross_object": cross_object,
        "extras": extras,
    }
    return grid, metadata


def _validate_case(grid: np.ndarray, info: dict) -> bool:
    out = main(grid)
    markers = info["markers"]
    bands = info["bands"]
    rays = [_ray_cells(marker["pos"], marker["step"], grid.shape) for marker in markers]

    for marker in markers:
        if int(out[marker["control"]]) != BG:
            return False

    for obj in info["same_band_hit"]:
        arr = np.array(obj, dtype=int)
        if not np.all(out[arr[:, 0], arr[:, 1]] == BG):
            return False

    keepers = list(info["same_band_unhit"]) + [info["cross_object"]] + list(info["extras"])
    for obj in keepers:
        arr = np.array(obj, dtype=int)
        if not np.all(out[arr[:, 0], arr[:, 1]] == GREY):
            return False

    cross_object = info["cross_object"]
    source_idx = int(info["cross_source"])
    target_idx = int(info["cross_target"])
    if not any(cell in rays[source_idx] for cell in cross_object):
        return False
    if _object_band_index(cross_object, bands) != target_idx:
        return False

    objects = _grey_objects(grid)
    if not (6 <= len(objects) <= 10):
        return False

    counts = [0] * len(bands)
    for obj in objects:
        band_hits = [
            idx
            for idx, (lo, hi) in enumerate(bands)
            if all(lo <= int(r) <= hi for r, _c in obj["coords"])
        ]
        if len(band_hits) != 1:
            return False
        counts[band_hits[0]] += 1
    if any(count < 2 for count in counts):
        return False

    for idx, marker in enumerate(markers):
        lo, hi = marker["band"]
        same_band_hit = False
        same_band_unhit = False
        for obj in objects:
            coords = [tuple(map(int, coord)) for coord in obj["coords"]]
            belongs = all(lo <= r <= hi for r, _c in coords)
            hit = any(cell in rays[idx] for cell in coords)
            erased = all(int(out[r, c]) == BG for r, c in coords)
            if belongs and hit and erased:
                same_band_hit = True
            if belongs and not hit and all(int(out[r, c]) == GREY for r, c in coords):
                same_band_unhit = True
        if not (same_band_hit and same_band_unhit):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_case()
        if built is None:
            continue
        grid, info = built
        if _validate_case(grid, info):
            return grid
    raise AssertionError("failed to generate a valid band-limited ray-erasure puzzle")
