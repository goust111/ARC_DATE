from __future__ import annotations

import random

import numpy as np


BLACK = 0
AZURE = 8
MAROON = 9
POPULATION_COLORS = (1, 2, 3, 4, 5, 6, 7)
PATTERN_NAMES = (
    "hbar",
    "vbar",
    "square",
    "diag_down",
    "diag_up",
    "l_tl",
    "l_tr",
    "l_bl",
    "l_br",
    "plus",
    "frame",
)

# --------------- solver ---------------


def _pattern_cells(name: str, step: int) -> set[tuple[int, int]]:
    assert step >= 1

    if name == "hbar":
        return {(0, c) for c in range(step)}
    if name == "vbar":
        return {(r, 0) for r in range(step)}
    if name == "square":
        return {(r, c) for r in range(step) for c in range(step)}
    if name == "diag_down":
        return {(i, i) for i in range(step)}
    if name == "diag_up":
        return {(i, step - 1 - i) for i in range(step)}
    if name == "l_tl":
        return {(0, c) for c in range(step)} | {(r, 0) for r in range(step)}
    if name == "l_tr":
        return {(0, c) for c in range(step)} | {(r, step - 1) for r in range(step)}
    if name == "l_bl":
        return {(step - 1, c) for c in range(step)} | {(r, 0) for r in range(step)}
    if name == "l_br":
        return {(step - 1, c) for c in range(step)} | {(r, step - 1) for r in range(step)}
    if name == "plus":
        arm = step - 1
        size = 2 * arm + 1
        mid = arm
        return {(mid, c) for c in range(size)} | {(r, mid) for r in range(size)}
    if name == "frame":
        if step == 1:
            return {(0, 0)}
        cells = set()
        for c in range(step):
            cells.add((0, c))
            cells.add((step - 1, c))
        for r in range(step):
            cells.add((r, 0))
            cells.add((r, step - 1))
        return cells
    raise ValueError(f"unknown pattern family: {name}")


def _pattern_dims(name: str, step: int) -> tuple[int, int]:
    cells = _pattern_cells(name, step)
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _separator_positions(g: np.ndarray) -> tuple[list[int], list[int]]:
    sep_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == AZURE)]
    sep_cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == AZURE)]
    assert sep_rows and sep_cols, "expected both horizontal and vertical azure separators"
    assert all(b - a > 1 for a, b in zip(sep_rows, sep_rows[1:])), "separator rows must not touch"
    assert all(b - a > 1 for a, b in zip(sep_cols, sep_cols[1:])), "separator cols must not touch"
    return sep_rows, sep_cols


def _intervals(limit: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + [int(v) for v in separators] + [int(limit)]
    spans = []
    for a, b in zip(bounds, bounds[1:]):
        start = a + 1
        end = b
        assert start < end, "empty region interval"
        spans.append((start, end))
    return spans


def _parse_region(region: np.ndarray) -> dict:
    assert not np.any(region == AZURE), "azure should only appear on separator lines"

    marker_coords = np.argwhere(region == MAROON)
    assert len(marker_coords) >= 1, "each region needs a maroon step marker"
    marker_rows = sorted(set(int(r) for r, _c in marker_coords))
    assert len(marker_rows) == 1, "marker must be a single horizontal run"
    marker_row = marker_rows[0]
    assert marker_row == region.shape[0] - 1, "marker must sit on the bottom edge"
    marker_cols = sorted(int(c) for _r, c in marker_coords)
    expected_cols = list(range(region.shape[1] - len(marker_cols), region.shape[1]))
    assert marker_cols == expected_cols, "marker must be right-aligned in its region"
    step = len(marker_cols)

    colors = sorted(int(v) for v in np.unique(region) if int(v) not in {BLACK, MAROON})
    assert len(colors) == 1, "each region must contain exactly one population color"
    color = colors[0]

    coords = np.argwhere(region == color)
    assert len(coords) >= 1
    anchor_r = int(coords[:, 0].min())
    anchor_c = int(coords[:, 1].min())
    shape = {(int(r - anchor_r), int(c - anchor_c)) for r, c in coords}

    return {
        "color": color,
        "step": int(step),
        "anchor": (anchor_r, anchor_c),
        "shape": shape,
        "region_shape": tuple(map(int, region.shape)),
    }


def _infer_family(records: list[dict]) -> str:
    matches = []
    for name in PATTERN_NAMES:
        if all(record["shape"] == _pattern_cells(name, int(record["step"])) for record in records):
            matches.append(name)
    assert len(matches) == 1, f"expected one growth family match, got {matches}"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color regions into populations, order each population by the maroon
    step marker, identify the shape-growth rule, and advance the latest snapshot
    by one step in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows, sep_cols = _separator_positions(g)
    row_spans = _intervals(g.shape[0], sep_rows)
    col_spans = _intervals(g.shape[1], sep_cols)

    groups: dict[int, list[dict]] = {}
    region_map: dict[tuple[int, int], tuple[int, int]] = {}
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            info = _parse_region(g[r0:r1, c0:c1])
            groups.setdefault(int(info["color"]), []).append(info)
            region_map[(int(info["color"]), int(info["step"]))] = (int(r0), int(c0))

    out = g.copy()
    for color, records in groups.items():
        assert len(records) >= 2, "each population must appear in multiple regions"
        steps = [int(record["step"]) for record in records]
        assert len(set(steps)) == len(steps), "step markers must be unique within a population"
        ordered = sorted(records, key=lambda record: int(record["step"]))
        family = _infer_family(ordered)

        target = ordered[-1]
        next_step = int(target["step"]) + 1
        next_cells = _pattern_cells(family, next_step)
        anchor_r, anchor_c = target["anchor"]
        region_h, region_w = target["region_shape"]
        r0, c0 = region_map[(int(color), int(target["step"]))]

        max_r = max(r for r, _c in next_cells)
        max_c = max(c for _r, c in next_cells)
        assert anchor_r + max_r < region_h, "predicted shape exceeds region height"
        assert anchor_c + max_c < region_w, "predicted shape exceeds region width"

        region_out = out[r0 : r0 + region_h, c0 : c0 + region_w]
        region_out[region_out == int(color)] = BLACK
        region_out[region_out == MAROON] = BLACK

        for dr, dc in next_cells:
            region_out[anchor_r + dr, anchor_c + dc] = int(color)

        marker_start = region_w - next_step
        assert marker_start >= 0, "region too narrow for advanced marker"
        assert not np.any(region_out[region_h - 1, marker_start:] == int(color)), "marker must stay separate from the shape"
        region_out[region_h - 1, marker_start:] = MAROON

    return out


# --------------- generator ---------------

LAYOUTS = (
    (2, 3, 2),
    (3, 2, 2),
    (3, 3, 3),
)
REGION_HEIGHT_RANGE = (8, 9)
REGION_WIDTH_RANGE = (8, 9)
STEPS = (1, 2, 3)
MAX_STAGE = 4
MAX_GENERATION_ATTEMPTS = 300


def _canvas_with_regions(rows: int, cols: int, region_h: int, region_w: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    height = rows * region_h + (rows - 1)
    width = cols * region_w + (cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)

    origins = []
    for rr in range(rows):
        for cc in range(cols):
            origins.append((rr * (region_h + 1), cc * (region_w + 1)))

    for rr in range(1, rows):
        grid[rr * (region_h + 1) - 1, :] = AZURE
    for cc in range(1, cols):
        grid[:, cc * (region_w + 1) - 1] = AZURE
    return grid, origins


def _sample_anchor(region_h: int, region_w: int, family: str) -> tuple[int, int]:
    max_h, max_w = _pattern_dims(family, MAX_STAGE)
    max_r0 = region_h - 1 - max_h
    max_c0 = region_w - max_w
    assert max_r0 >= 0 and max_c0 >= 0
    return random.randint(0, max_r0), random.randint(0, max_c0)


def _paint_region(region: np.ndarray, color: int, family: str, step: int, anchor: tuple[int, int]) -> None:
    ar, ac = anchor
    for dr, dc in _pattern_cells(family, step):
        region[ar + dr, ac + dc] = int(color)
    region[-1, region.shape[1] - step :] = MAROON


def _render_pair(specs: list[dict], rows: int, cols: int, region_h: int, region_w: int) -> tuple[np.ndarray, np.ndarray]:
    inp, origins = _canvas_with_regions(rows, cols, region_h, region_w)
    out = inp.copy()
    for spec, (r0, c0) in zip(specs, origins):
        inp_region = inp[r0 : r0 + region_h, c0 : c0 + region_w]
        out_region = out[r0 : r0 + region_h, c0 : c0 + region_w]
        _paint_region(inp_region, int(spec["color"]), str(spec["family"]), int(spec["step"]), spec["anchor"])
        out_step = MAX_STAGE if int(spec["step"]) == max(STEPS) else int(spec["step"])
        _paint_region(out_region, int(spec["color"]), str(spec["family"]), out_step, spec["anchor"])
    return inp, out


def _build_specs(rows: int, cols: int, population_count: int, region_h: int, region_w: int) -> list[dict]:
    region_count = rows * cols
    assert region_count == population_count * len(STEPS)

    colors = random.sample(POPULATION_COLORS, population_count)
    families = random.sample(PATTERN_NAMES, population_count)

    specs = []
    for color, family in zip(colors, families):
        steps = list(STEPS)
        random.shuffle(steps)
        for step in steps:
            specs.append(
                {
                    "color": int(color),
                    "family": str(family),
                    "step": int(step),
                    "anchor": _sample_anchor(region_h, region_w, str(family)),
                }
            )
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    rows, cols, population_count = random.choice(LAYOUTS)
    region_h = random.randint(*REGION_HEIGHT_RANGE)
    region_w = random.randint(*REGION_WIDTH_RANGE)
    specs = _build_specs(rows, cols, population_count, region_h, region_w)
    inp, expected = _render_pair(specs, rows, cols, region_h, region_w)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
