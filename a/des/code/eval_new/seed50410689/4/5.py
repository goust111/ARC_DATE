from __future__ import annotations

import random

import numpy as np


BG = 0
AZURE = 8
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
SIDES = ("top", "bottom", "left", "right")
DIRECTION_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 300
GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (12, 30)
FRAME_HEIGHT_RANGE = (4, 9)
FRAME_WIDTH_RANGE = (4, 9)
MAX_OBJECTS = 4
MAX_MARKERS = 5
PLACEMENT_TRIES = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expand(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _analyze_color(grid: np.ndarray, color: int) -> dict[str, object]:
    comps = _components(grid == color)
    large = [comp for comp in comps if len(comp) > 1]
    singletons = [comp for comp in comps if len(comp) == 1]
    assert len(large) == 1, "each fence color must have exactly one non-singleton component"
    assert len(singletons) == 1, "each fence color must have exactly one projection start"

    frame = large[0]
    start = tuple(map(int, singletons[0][0]))
    r0, c0, r1, c1 = _bbox(frame)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame bbox must have non-empty interior"

    frame_set = {tuple(map(int, cell)) for cell in frame}
    border = _rect_border(r0, c0, r1, c1)
    assert frame_set <= border, "frame pixels must lie on their bbox border"

    missing = sorted(border - frame_set)
    assert len(missing) == 1, "frame border must have exactly one gap"
    gap_r, gap_c = missing[0]

    if gap_r == r0 and c0 < gap_c < c1:
        side = "top"
    elif gap_r == r1 and c0 < gap_c < c1:
        side = "bottom"
    elif gap_c == c0 and r0 < gap_r < r1:
        side = "left"
    elif gap_c == c1 and r0 < gap_r < r1:
        side = "right"
    else:
        raise AssertionError("gap must lie on exactly one non-corner edge")

    dr, dc = DIRECTION_BY_SIDE[side]
    assert start == (gap_r + dr, gap_c + dc), "projection start must sit just outside the gap"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert set(map(int, np.unique(interior))) <= {BG, AZURE}, "frame interior may only contain background and Azure markers"
    markers = np.argwhere(interior == AZURE) + np.array([r0 + 1, c0 + 1], dtype=int)

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "start": start,
        "direction": (dr, dc),
        "markers": [(int(r), int(c)) for r, c in markers],
        "count": int(len(markers)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast each same-colored fence's shadow outward, using the interior Azure count as the ray length.

    Input structure:
    - The background is black.
    - Each active object color forms a one-pixel-thick hollow rectangular fence with exactly one missing border pixel on one non-corner side.
    - Inside each fence are zero or more Azure marker pixels on black, and just outside the gap sits one same-colored singleton projection start.

    Rule:
    - For each fence color, recover the rectangular border bbox, find the unique missing border pixel, and use its side as the projection direction.
    - Count the Azure pixels strictly inside that fence; this count N is the number of outward steps to extend from the same-colored singleton start.
    - Paint those N outward ray cells in the fence color, clear the interior Azure markers to black, and leave the fence and start pixel unchanged.

    Key insight:
    - The output depends on chaining three separate cues per color: the rectangle gap gives direction, the external singleton gives the ray origin, and the Azure markers inside the fence give the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, AZURE)]
    assert colors, "expected at least one colored fence"

    for color in colors:
        info = _analyze_color(g, color)

        r0, c0, r1, c1 = info["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == AZURE] = BG

        start_r, start_c = info["start"]
        dr, dc = info["direction"]
        for step in range(1, int(info["count"]) + 1):
            rr = start_r + dr * step
            cc = start_c + dc * step
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "projection ray must stay inside the grid"
            assert int(out[rr, cc]) == BG, "projection ray must extend through empty space"
            out[rr, cc] = int(color)

    return out


def _build_spec(
    height: int,
    width: int,
    color: int,
    side: str,
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    dr, dc = DIRECTION_BY_SIDE[side]
    for _ in range(PLACEMENT_TRIES):
        frame_h = random.randint(*FRAME_HEIGHT_RANGE)
        frame_w = random.randint(*FRAME_WIDTH_RANGE)
        interior_area = (frame_h - 2) * (frame_w - 2)
        if interior_area < 2:
            continue

        marker_max = min(MAX_MARKERS, interior_area - 1)
        marker_count = random.randint(1, marker_max)

        top_margin = marker_count + 1 if side == "top" else 0
        bottom_margin = marker_count + 1 if side == "bottom" else 0
        left_margin = marker_count + 1 if side == "left" else 0
        right_margin = marker_count + 1 if side == "right" else 0

        r0_lo = top_margin
        r0_hi = height - frame_h - bottom_margin
        c0_lo = left_margin
        c0_hi = width - frame_w - right_margin
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1

        if side in ("top", "bottom"):
            gap_c = random.randint(c0 + 1, c1 - 1)
            gap_r = r0 if side == "top" else r1
        else:
            gap_r = random.randint(r0 + 1, r1 - 1)
            gap_c = c0 if side == "left" else c1

        start = (gap_r + dr, gap_c + dc)
        ray = {(start[0] + dr * step, start[1] + dc * step) for step in range(1, marker_count + 1)}
        if not all(0 <= r < height and 0 <= c < width for r, c in ray):
            continue

        border = _rect_border(r0, c0, r1, c1)
        frame = border - {(gap_r, gap_c)}
        interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        markers = set(random.sample(interior_cells, marker_count))

        reserved = set(frame)
        reserved.add(start)
        reserved.update(markers)
        reserved.update(ray)
        if any(cell in blocked for cell in reserved):
            continue

        return {
            "color": int(color),
            "bbox": (r0, c0, r1, c1),
            "side": side,
            "start": start,
            "frame": frame,
            "markers": markers,
            "ray": ray,
        }
    return None


def _expected_from_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["markers"]:
            out[int(r), int(c)] = BG
        for r, c in spec["ray"]:
            out[int(r), int(c)] = int(spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    if height * width >= 260:
        max_objects = MAX_OBJECTS
    elif height * width >= 170:
        max_objects = 3
    else:
        max_objects = 2
    object_count = random.randint(1, max_objects)

    colors = random.sample(FRAME_COLORS, object_count)
    sides = random.sample(SIDES, k=min(object_count, len(SIDES)))
    while len(sides) < object_count:
        sides.append(random.choice(SIDES))
    random.shuffle(sides)

    blocked: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []
    for color, side in zip(colors, sides):
        spec = _build_spec(height, width, color, side, blocked)
        if spec is None:
            return None

        for r, c in spec["frame"]:
            grid[int(r), int(c)] = int(color)
        sr, sc = spec["start"]
        grid[int(sr), int(sc)] = int(color)
        for r, c in spec["markers"]:
            grid[int(r), int(c)] = AZURE

        reserved = set(spec["frame"]) | {spec["start"]} | set(spec["markers"]) | set(spec["ray"])
        blocked.update(_expand(reserved, grid.shape))
        specs.append(spec)

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection puzzle input")
