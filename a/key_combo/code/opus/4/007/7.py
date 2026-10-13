from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 5
PALETTE = (1, 2, 3, 4, 6)
GRID_MIN = 16
GRID_MAX = 22
REGION_COUNT_RANGE = (3, 5)
AREA_CHOICES = tuple(range(4, 13))
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 250
SHAPE_RETRIES = 200

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: Iterable[Coord]) -> BBox:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: Iterable[Coord]) -> list[Coord]:
    pts = list(coords)
    r0, c0, _, _ = bbox_coords(pts)
    return sorted((r - r0, c - c0) for r, c in pts)


def _transform_coords(coords: Iterable[Coord], mode: int) -> list[Coord]:
    pts = list(coords)
    r0, c0, r1, c1 = bbox_coords(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    base = [(r - r0, c - c0) for r, c in pts]

    if mode == 0:
        out = base
    elif mode == 1:
        out = [(c, h - 1 - r) for r, c in base]
    elif mode == 2:
        out = [(h - 1 - r, w - 1 - c) for r, c in base]
    elif mode == 3:
        out = [(w - 1 - c, r) for r, c in base]
    elif mode == 4:
        out = [(r, w - 1 - c) for r, c in base]
    elif mode == 5:
        out = [(h - 1 - r, c) for r, c in base]
    elif mode == 6:
        out = [(c, r) for r, c in base]
    elif mode == 7:
        out = [(w - 1 - c, h - 1 - r) for r, c in base]
    else:
        raise ValueError("invalid transform mode")
    return _normalize(out)


def _halo(coords: Iterable[Coord]) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in coords:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                out.add((rr, cc))
    return out


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen: set[Coord] = set()
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, MARKER) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) == color:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": sorted(coords),
                    "area": len(coords),
                    "bbox": bbox_coords(coords),
                }
            )
    return comps


def _marker_columns(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    return [int(c) for c in np.flatnonzero(g[0] == MARKER)]


def _first_hit_component(col: int, grid: np.ndarray, cell_to_comp: dict[Coord, int]) -> int:
    g = np.asarray(grid, dtype=int)
    for r in range(1, g.shape[0]):
        color = int(g[r, col])
        if color not in (BG, MARKER):
            return cell_to_comp[(r, col)]
    raise AssertionError(f"marker at column {col} does not hit a region")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the first region hit by each top border marker and straighten it into a row segment.

    Input structure:
    - A black background grid contains 3-5 disconnected monochrome foreground regions inside the canvas.
    - Grey markers lie on the top border, sorted visually from left to right.
    - A marker's inward view is the vertical ray straight down its own column.

    Rule:
    - Find all grey top-border markers and sort them by column.
    - For each marker, cast a downward ray and take the 4-connected foreground region containing the first non-black, non-grey pixel hit by that ray.
    - Record that region's color and pixel count, then build an output grid with one row per marker and width equal to the largest selected region area.
    - In output row i, draw a left-aligned horizontal segment of the selected region's color whose length equals that region's area.

    Key insight:
    - The border markers are remote selectors: their left-to-right order decides output row order, while the only preserved property of each selected region is its color and area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _find_components(g)
    assert comps, "expected at least one foreground region"
    cell_to_comp = {
        (r, c): idx
        for idx, comp in enumerate(comps)
        for r, c in comp["coords"]
    }

    cols = _marker_columns(g)
    assert cols, "expected at least one top marker"

    rows: list[tuple[int, int]] = []
    seen_targets: set[int] = set()
    for col in cols:
        comp_idx = _first_hit_component(col, g, cell_to_comp)
        rows.append((int(comps[comp_idx]["color"]), int(comps[comp_idx]["area"])))
        seen_targets.add(comp_idx)

    assert len(seen_targets) == len(cols), "markers must select distinct regions"
    width = max(area for _, area in rows)
    out = np.full((len(rows), width), BG, dtype=int)
    for r, (color, area) in enumerate(rows):
        out[r, :area] = color
    return out


def _random_polyomino(area: int) -> list[Coord]:
    for _ in range(SHAPE_RETRIES):
        cells: set[Coord] = {(0, 0)}
        while len(cells) < area:
            frontier = sorted(
                {
                    nbr
                    for r, c in cells
                    for nbr in neighbors4(r, c)
                    if nbr not in cells
                }
            )
            cells.add(random.choice(frontier))

        coords = _transform_coords(cells, random.randrange(8))
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w < 2:
            continue
        if h > 5 or w > 6:
            continue
        if area == h * w:
            continue
        if len({r for r, _ in coords}) < 2 or len({c for _, c in coords}) < 2:
            continue
        return coords
    raise RuntimeError(f"failed to grow a polyomino of area {area}")


def _place_regions(h: int, w: int, count: int) -> list[dict] | None:
    colors = random.sample(PALETTE, count)
    max_area = random.randint(count + 3, AREA_CHOICES[-1])
    areas = random.sample(tuple(range(AREA_CHOICES[0], max_area + 1)), count)
    specs = list(zip(colors, areas))
    random.shuffle(specs)

    regions: list[dict] = []
    blocked: set[Coord] = set()
    for color, area in specs:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            shape = _random_polyomino(area)
            _, _, r1, c1 = bbox_coords(shape)
            shape_h = r1 + 1
            shape_w = c1 + 1
            top = random.randint(2, h - shape_h)
            left = random.randint(0, w - shape_w)
            cells = [(top + r, left + c) for r, c in shape]
            if any((r, c) in blocked for r, c in cells):
                continue
            regions.append(
                {
                    "color": int(color),
                    "coords": sorted(cells),
                    "area": int(area),
                    "bbox": bbox_coords(cells),
                }
            )
            for cell in _halo(cells):
                rr, cc = cell
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add(cell)
            placed = True
            break
        if not placed:
            return None
    return regions


def _render_regions(h: int, w: int, regions: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for region in regions:
        for r, c in region["coords"]:
            grid[r, c] = int(region["color"])
    return grid


def _accessible_columns(grid: np.ndarray, regions: list[dict]) -> dict[int, list[int]]:
    cell_to_region = {
        (r, c): idx
        for idx, region in enumerate(regions)
        for r, c in region["coords"]
    }
    accessible = {idx: [] for idx in range(len(regions))}
    for c in range(grid.shape[1]):
        for r in range(1, grid.shape[0]):
            if int(grid[r, c]) != BG:
                accessible[cell_to_region[(r, c)]].append(c)
                break
    return accessible


def _inversion_count(order_a: list[int], order_b: list[int]) -> int:
    pos = {rid: i for i, rid in enumerate(order_b)}
    seq = [pos[rid] for rid in order_a]
    inv = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                inv += 1
    return inv


def _assignment_score(regions: list[dict], assignment: dict[int, int]) -> tuple[int, int]:
    marker_order = sorted(assignment, key=lambda rid: assignment[rid])
    bbox_left_order = sorted(range(len(regions)), key=lambda rid: (regions[rid]["bbox"][1], regions[rid]["bbox"][0]))
    center_order = sorted(
        range(len(regions)),
        key=lambda rid: (regions[rid]["bbox"][1] + regions[rid]["bbox"][3], regions[rid]["bbox"][0]),
    )
    top_order = sorted(range(len(regions)), key=lambda rid: (regions[rid]["bbox"][0], regions[rid]["bbox"][1]))

    inversion = (
        _inversion_count(marker_order, bbox_left_order)
        + _inversion_count(marker_order, center_order)
        + _inversion_count(marker_order, top_order)
    )
    interior = 0
    for rid, col in assignment.items():
        cols = sorted(c for _, c in regions[rid]["coords"])
        if cols[0] < col < cols[-1]:
            interior += 1
    return inversion, inversion * 10 + interior


def _choose_markers(regions: list[dict], accessible: dict[int, list[int]]) -> dict[int, int] | None:
    if any(not cols for cols in accessible.values()):
        return None

    region_ids = sorted(accessible, key=lambda rid: (len(accessible[rid]), regions[rid]["bbox"][0], regions[rid]["bbox"][1]))
    best_assignment: dict[int, int] | None = None
    best_key: tuple[int, int] | None = None
    chosen: dict[int, int] = {}
    used_cols: list[int] = []

    def search(i: int) -> None:
        nonlocal best_assignment, best_key
        if i == len(region_ids):
            inversion, score = _assignment_score(regions, chosen)
            key = (inversion, score)
            if best_key is None or key > best_key:
                best_key = key
                best_assignment = dict(chosen)
            return

        rid = region_ids[i]
        candidates = sorted(
            accessible[rid],
            key=lambda col: (
                min(abs(col - other) for other in used_cols) if used_cols else 99,
                -abs(col - (regions[rid]["bbox"][1] + regions[rid]["bbox"][3]) / 2.0),
                col,
            ),
            reverse=True,
        )
        for col in candidates:
            if any(abs(col - prev) < 2 for prev in used_cols):
                continue
            chosen[rid] = col
            used_cols.append(col)
            search(i + 1)
            used_cols.pop()
            del chosen[rid]

    search(0)
    if best_assignment is None:
        return None
    inversion, _ = _assignment_score(regions, best_assignment)
    if inversion == 0:
        return None
    return best_assignment


def _build_output_spec(regions: list[dict], markers: dict[int, int]) -> list[tuple[int, int, int]]:
    return sorted(
        [(int(col), int(regions[rid]["color"]), int(regions[rid]["area"])) for rid, col in markers.items()],
        key=lambda item: item[0],
    )


def _output_from_spec(spec: list[tuple[int, int, int]]) -> np.ndarray:
    width = max(area for _, _, area in spec)
    out = np.full((len(spec), width), BG, dtype=int)
    for r, (_col, color, area) in enumerate(spec):
        out[r, :area] = color
    return out


def _generate_case(with_meta: bool = False):
    for _ in range(GENERATION_RETRIES):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        count = random.randint(REGION_COUNT_RANGE[0], REGION_COUNT_RANGE[1])
        regions = _place_regions(h, w, count)
        if regions is None:
            continue

        grid = _render_regions(h, w, regions)
        accessible = _accessible_columns(grid, regions)
        markers = _choose_markers(regions, accessible)
        if markers is None:
            continue

        for col in markers.values():
            grid[0, col] = MARKER

        spec = _build_output_spec(regions, markers)
        if with_meta:
            return grid, {"regions": regions, "markers": markers, "spec": spec}
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case(with_meta=False)


if __name__ == "__main__":
    grid, meta = _generate_case(with_meta=True)
    expected = _output_from_spec(meta["spec"])
    actual = main(grid)
    assert np.array_equal(actual, expected)
    print(grid)
    print()
    print(actual)
