from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)

# --------------- solver ---------------


def _seed_components(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (GREY, MAROON):
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "r0": min(rows),
                    "r1": max(rows),
                    "c0": min(cols),
                    "c1": max(cols),
                    "size": len(coords),
                }
            )

    return comps


def _soil_strips(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    strips: list[dict] = []
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != MAROON:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == MAROON:
                c += 1
            strips.append({"row": r, "c0": c0, "c1": c})
            c += 1
    return strips


def _pair_seed(strip: dict, components: list[dict]) -> dict | None:
    soil_row = int(strip["row"])
    soil_center = (int(strip["c0"]) + int(strip["c1"])) / 2.0
    candidates = []
    for comp in components:
        if int(comp["r1"]) != soil_row - 1:
            continue
        comp_center = (int(comp["c0"]) + int(comp["c1"])) / 2.0
        if not (int(strip["c0"]) <= comp_center <= int(strip["c1"])):
            continue
        candidates.append((abs(comp_center - soil_center), int(comp["c0"]), comp))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _grow_from_seed(out: np.ndarray, seed: dict, strip: dict) -> None:
    color = int(seed["color"])
    r0 = int(seed["r0"])
    c0 = int(seed["c0"])
    h = int(seed["r1"]) - r0 + 1
    n_layers = int(seed["size"])
    soil_row = int(strip["row"])
    left = int(strip["c0"])
    right = int(strip["c1"])
    grid_h, grid_w = map(int, out.shape)

    mirrored = [(h - 1 - (rr - r0), cc - c0) for rr, cc in seed["coords"]]
    for k in range(1, n_layers + 1):
        for mr, mc in mirrored:
            rr = soil_row + k + mr
            if not (0 <= rr < grid_h):
                continue
            for dc in range(-k, k + 1):
                cc = c0 + mc + dc
                if cc < left or cc > right or not (0 <= cc < grid_w):
                    continue
                if int(out[rr, cc]) == GREY:
                    out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each seed with the maroon soil strip directly below it, then grow a mirrored,
    horizontally expanding copy below the soil. The number of growth layers equals the
    seed's pixel count, and every layer is clipped to the soil strip's horizontal span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    components = _seed_components(g)
    for strip in _soil_strips(g):
        seed = _pair_seed(strip, components)
        if seed is None:
            continue
        _grow_from_seed(out, seed, strip)
    return out


# --------------- generator ---------------

AREA_TO_SHAPES: dict[int, tuple[np.ndarray, ...]] = {
    1: (np.array([[1]], dtype=bool),),
    2: (np.array([[1, 1]], dtype=bool),),
    3: (
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
    ),
    4: (
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    ),
}

PAIR_COUNT_RANGE = (2, 4)
TOP_MARGIN_RANGE = (1, 3)
BOTTOM_MARGIN_RANGE = (1, 3)
INTER_BAND_GAP_RANGE = (1, 2)
GRID_WIDTH_RANGE = (16, 30)
MIN_SOIL_LEN = 2
GENERATION_RETRIES = 300


def _sample_specs() -> list[dict]:
    count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(SEED_COLORS, count)
    specs: list[dict] = []
    for color in colors:
        area = random.randint(1, 4)
        shape = random.choice(AREA_TO_SHAPES[area])
        specs.append(
            {
                "color": int(color),
                "area": int(area),
                "shape": shape,
                "clip": random.choice((False, True)),
                "gap": random.randint(*INTER_BAND_GAP_RANGE),
            }
        )
    return specs


def _grid_height(specs: list[dict], top_margin: int, bottom_margin: int) -> int:
    cursor = int(top_margin)
    for spec in specs:
        h = int(spec["shape"].shape[0])
        n = int(spec["area"])
        soil_row = cursor + h
        growth_bottom = soil_row + n + h - 1
        cursor = growth_bottom + int(spec["gap"]) + 1
    return cursor + int(bottom_margin)


def _soil_length(seed_w: int, layers: int, clip: bool, width_cap: int) -> int:
    full_width = int(seed_w + 2 * layers)
    if clip and full_width - 1 >= max(seed_w, MIN_SOIL_LEN):
        low = max(seed_w, MIN_SOIL_LEN)
        high = min(full_width - 1, width_cap)
        if low <= high:
            return random.randint(low, high)
    low = max(full_width, MIN_SOIL_LEN)
    high = min(width_cap, full_width + 6)
    return random.randint(low, high)


def _draw_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(shape)
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        grid[top + int(rr), left + int(cc)] = int(color)


def _construct_input(specs: list[dict]) -> np.ndarray | None:
    max_full_width = max(int(spec["shape"].shape[1]) + 2 * int(spec["area"]) for spec in specs)
    min_width = max(GRID_WIDTH_RANGE[0], max_full_width + 4)
    if min_width > GRID_WIDTH_RANGE[1]:
        return None
    grid_w = random.randint(min_width, GRID_WIDTH_RANGE[1])
    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    grid_h = _grid_height(specs, top_margin, bottom_margin)
    grid = np.full((grid_h, grid_w), GREY, dtype=int)

    cursor = top_margin
    for spec in specs:
        shape = spec["shape"]
        color = int(spec["color"])
        seed_h, seed_w = map(int, shape.shape)
        layers = int(spec["area"])

        soil_row = cursor + seed_h
        soil_len = _soil_length(seed_w, layers, bool(spec["clip"]), grid_w - 2)
        if soil_len > grid_w - 2:
            return None
        soil_c0 = random.randint(1, grid_w - soil_len - 1)
        soil_c1 = soil_c0 + soil_len - 1
        seed_left = soil_c0 + (soil_len - seed_w) // 2
        seed_top = soil_row - seed_h

        grid[soil_row, soil_c0 : soil_c1 + 1] = MAROON
        _draw_shape(grid, shape, seed_top, seed_left, color)

        growth_bottom = soil_row + layers + seed_h - 1
        cursor = growth_bottom + int(spec["gap"]) + 1

    return grid


def _has_visible_growth(inp: np.ndarray, out: np.ndarray) -> bool:
    return bool(np.any(inp != out))


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    inp = _construct_input(specs)
    if inp is None:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if not _has_visible_growth(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
