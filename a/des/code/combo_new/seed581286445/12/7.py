from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
ORANGE = 7
BAND_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
CANVAS = 30
MAX_GEN_RETRIES = 300
MAX_PLACE_TRIES = 160
MARGIN_MAX = 3


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((int(cr), int(cc)))
            for dr, dc in steps:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _perp(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    if dr != 0:
        return (0, 1)
    return (1, 0)


def _parse_band(grid: np.ndarray, color: int) -> dict[str, object]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
    assert len(coords) >= 2, "band must have length >= 2"

    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    cells = set(coords)

    if len(rows) == 1:
        r = next(iter(rows))
        c0 = min(cols)
        c1 = max(cols)
        expected = {(int(r), int(c)) for c in range(int(c0), int(c1) + 1)}
        assert cells == expected, "horizontal band must be contiguous"
        ordered = [(int(r), int(c)) for c in range(int(c0), int(c1) + 1)]
        endpoints = [(int(r), int(c0)), (int(r), int(c1))]
        orientation = "h"
    else:
        assert len(cols) == 1, "band must be axis-aligned"
        c = next(iter(cols))
        r0 = min(rows)
        r1 = max(rows)
        expected = {(int(r), int(c)) for r in range(int(r0), int(r1) + 1)}
        assert cells == expected, "vertical band must be contiguous"
        ordered = [(int(r), int(c)) for r in range(int(r0), int(r1) + 1)]
        endpoints = [(int(r0), int(c)), (int(r1), int(c))]
        orientation = "v"

    return {
        "color": int(color),
        "coords": ordered,
        "endpoints": endpoints,
        "orientation": orientation,
    }


def _parse_arrow(comp: list[tuple[int, int]]) -> dict[str, object]:
    cells = {(int(r), int(c)) for r, c in comp}
    assert len(cells) >= 3, "arrow must have shaft + 2 head pixels"
    length = len(cells) - 2

    matches: list[dict[str, object]] = []
    for tip_r, tip_c in cells:
        for name, (dr, dc) in DIRS.items():
            pr, pc = _perp((dr, dc))
            shaft = {(int(tip_r - dr * k), int(tip_c - dc * k)) for k in range(length)}
            head = {
                (int(tip_r + dr + pr), int(tip_c + dc + pc)),
                (int(tip_r + dr - pr), int(tip_c + dc - pc)),
            }
            expected = shaft | head
            if expected != cells:
                continue
            base = (int(tip_r - dr * (length - 1)), int(tip_c - dc * (length - 1)))
            matches.append(
                {
                    "direction": name,
                    "length": int(length),
                    "tip": (int(tip_r), int(tip_c)),
                    "base": base,
                    "coords": sorted(cells),
                }
            )

    assert len(matches) == 1, "orange component must match exactly one arrow geometry"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each orange arrow as a projection instruction that replicates its neighboring color band.

    Input structure:
    - The background is black.
    - Non-orange foreground objects are monochrome horizontal or vertical bands,
      each using a distinct color and having length at least 2.
    - Orange objects are arrows drawn as a straight shaft plus a two-pixel diagonal
      arrowhead; one end of each shaft is 4-neighbor adjacent to an endpoint of a band.

    Rule:
    - Detect every orange arrow, reading its pointing direction from the arrowhead and
      its projection distance from the shaft length.
    - Match each arrow to the unique band endpoint adjacent to the shaft base.
    - Keep the original band, erase the arrow, and add translated copies of that band
      at offsets 1 through L along the arrow direction, where L is the shaft length.

    Key insight:
    - The arrowhead gives the translation direction while the shaft length gives how
      many successive translated copies must be painted.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    band_colors = [int(v) for v in np.unique(g) if int(v) not in (BG, ORANGE)]
    bands = [_parse_band(g, color) for color in band_colors]

    orange_comps = _components(g == ORANGE, connectivity=8)
    assert orange_comps, "expected at least one arrow"
    arrows = [_parse_arrow(comp) for comp in orange_comps]

    out = g.copy()
    out[out == ORANGE] = BG

    for arrow in arrows:
        direction = DIRS[str(arrow["direction"])]
        needed_orientation = "v" if direction[1] != 0 else "h"
        matches: list[dict[str, object]] = []
        base_r, base_c = arrow["base"]
        for band in bands:
            if band["orientation"] != needed_orientation:
                continue
            for endpoint in band["endpoints"]:
                if abs(int(endpoint[0]) - int(base_r)) + abs(int(endpoint[1]) - int(base_c)) == 1:
                    matches.append(band)
                    break
        assert len(matches) == 1, "each arrow must match exactly one adjacent band"
        band = matches[0]

        dr, dc = direction
        for step in range(1, int(arrow["length"]) + 1):
            for r, c in band["coords"]:
                rr = int(r + dr * step)
                cc = int(c + dc * step)
                assert 0 <= rr < h and 0 <= cc < w, "projection must remain inside the grid"
                out[rr, cc] = int(band["color"])

    return out


def _normalize(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(int(r - min_r), int(c - min_c)) for r, c in cells}
    return shifted, int(min_r), int(min_c)


def _family_spec(rng: np.random.Generator, direction_name: str, color: int) -> dict[str, object]:
    direction = DIRS[direction_name]
    band_len = int(rng.integers(2, 8))
    arrow_len = int(rng.integers(2, 7))
    attach_far_end = bool(rng.integers(0, 2))

    if direction_name in ("up", "down"):
        band = {(0, int(c)) for c in range(band_len)}
        endpoint_c = band_len - 1 if attach_far_end else 0
        endpoint = (0, int(endpoint_c))
    else:
        band = {(int(r), 0) for r in range(band_len)}
        endpoint_r = band_len - 1 if attach_far_end else 0
        endpoint = (int(endpoint_r), 0)

    dr, dc = direction
    shaft = {(int(endpoint[0] + dr * k), int(endpoint[1] + dc * k)) for k in range(1, arrow_len + 1)}
    tip = (int(endpoint[0] + dr * arrow_len), int(endpoint[1] + dc * arrow_len))
    pr, pc = _perp(direction)
    head = {
        (int(tip[0] + dr + pr), int(tip[1] + dc + pc)),
        (int(tip[0] + dr - pr), int(tip[1] + dc - pc)),
    }

    projections: set[tuple[int, int]] = set()
    for step in range(1, arrow_len + 1):
        for r, c in band:
            projections.add((int(r + dr * step), int(c + dc * step)))

    footprint = set(band) | shaft | head | projections
    footprint_norm, min_r, min_c = _normalize(footprint)

    def shift_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(int(r - min_r), int(c - min_c)) for r, c in cells}

    return {
        "color": int(color),
        "direction": direction_name,
        "band": shift_cells(set(band)),
        "arrow": shift_cells(set(shaft) | head),
        "output": shift_cells(set(band) | projections),
        "footprint": footprint_norm,
    }


def _guard(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    guarded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                guarded.add((int(r + dr), int(c + dc)))
    return guarded


def _place_families(rng: np.random.Generator, families: list[dict[str, object]]) -> list[dict[str, object]] | None:
    blocked = np.zeros((CANVAS, CANVAS), dtype=bool)
    placed: list[dict[str, object]] = []

    for family in families:
        rows = [r for r, _ in family["footprint"]]
        cols = [c for _, c in family["footprint"]]
        box_h = max(rows) + 1
        box_w = max(cols) + 1

        success = False
        for _ in range(MAX_PLACE_TRIES):
            top = int(rng.integers(0, CANVAS - box_h + 1))
            left = int(rng.integers(0, CANVAS - box_w + 1))

            footprint = {(int(top + r), int(left + c)) for r, c in family["footprint"]}
            guard = _guard(footprint)
            if any(0 <= r < CANVAS and 0 <= c < CANVAS and blocked[r, c] for r, c in guard):
                continue

            for r, c in guard:
                if 0 <= r < CANVAS and 0 <= c < CANVAS:
                    blocked[r, c] = True

            def move(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
                return {(int(top + r), int(left + c)) for r, c in cells}

            placed.append(
                {
                    "color": int(family["color"]),
                    "band": move(family["band"]),
                    "arrow": move(family["arrow"]),
                    "output": move(family["output"]),
                    "footprint": footprint,
                }
            )
            success = True
            break

        if not success:
            return None

    return placed


def _crop_bounds(rng: np.random.Generator, placed: list[dict[str, object]]) -> tuple[int, int, int, int]:
    all_cells = set()
    for item in placed:
        all_cells |= item["footprint"]

    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)

    top_pad = int(rng.integers(0, min(MARGIN_MAX, min_r) + 1))
    bottom_pad = int(rng.integers(0, min(MARGIN_MAX, CANVAS - 1 - max_r) + 1))
    left_pad = int(rng.integers(0, min(MARGIN_MAX, min_c) + 1))
    right_pad = int(rng.integers(0, min(MARGIN_MAX, CANVAS - 1 - max_c) + 1))

    return (
        int(min_r - top_pad),
        int(max_r + bottom_pad),
        int(min_c - left_pad),
        int(max_c + right_pad),
    )


def _render(placed: list[dict[str, object]], bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    r0, r1, c0, c1 = bounds
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for item in placed:
        color = int(item["color"])
        for r, c in item["band"]:
            inp[int(r - r0), int(c - c0)] = color
        for r, c in item["arrow"]:
            inp[int(r - r0), int(c - c0)] = ORANGE
        for r, c in item["output"]:
            out[int(r - r0), int(c - c0)] = color

    return inp, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(2, 5))
    directions = list(rng.permutation(np.array(list(DIRS.keys()), dtype=object))[:count])
    colors = [int(c) for c in rng.choice(np.array(BAND_COLORS, dtype=int), size=count, replace=False)]
    families = [_family_spec(rng, str(direction), int(color)) for direction, color in zip(directions, colors)]

    placed = _place_families(rng, families)
    if placed is None:
        return None

    bounds = _crop_bounds(rng, placed)
    inp, expected = _render(placed, bounds)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
