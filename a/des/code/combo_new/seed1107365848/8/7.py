from __future__ import annotations

from collections import Counter
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
QUADRANTS = (0, 1, 2, 3)  # TL, TR, BL, BR
MISSING_CORNERS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
OBJECT_KINDS = ("wide", "tall")
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a non-empty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    full_rows: dict[int, list[int]] = {}
    full_cols: dict[int, list[int]] = {}

    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color):
            full_rows.setdefault(color, []).append(int(r))

    for c in range(w):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            full_cols.setdefault(color, []).append(int(c))

    candidates = [
        (color, rows[0], cols[0])
        for color, rows in full_rows.items()
        for cols in [full_cols.get(color, [])]
        if len(rows) == 1 and len(cols) == 1
    ]
    assert len(candidates) == 1, "expected one full-row/full-column divider cross"
    return tuple(map(int, candidates[0]))


def _find_components(grid: np.ndarray, mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            rows = coords_arr[:, 0]
            cols = coords_arr[:, 1]
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "area": int(coords_arr.shape[0]),
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                }
            )

    return components


def _quadrant_bounds(shape: tuple[int, int], mid_r: int, mid_c: int, quadrant: int) -> tuple[int, int, int, int]:
    h, w = shape
    if quadrant == 0:
        return 0, mid_r - 1, 0, mid_c - 1
    if quadrant == 1:
        return 0, mid_r - 1, mid_c + 1, w - 1
    if quadrant == 2:
        return mid_r + 1, h - 1, 0, mid_c - 1
    return mid_r + 1, h - 1, mid_c + 1, w - 1


def _quadrant_marker_cell(shape: tuple[int, int], quadrant: int) -> tuple[int, int]:
    h, w = shape
    if quadrant == 0:
        return 0, 0
    if quadrant == 1:
        return 0, w - 1
    if quadrant == 2:
        return h - 1, 0
    return h - 1, w - 1


def _mirror_completion_coords(coords: np.ndarray) -> set[tuple[int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())
    h, w = r1 - r0 + 1, c1 - c0 + 1
    assert (h, w) in {(2, 3), (3, 2)}, "each L must have a 2x3 or 3x2 bbox"

    patch = np.zeros((h, w), dtype=bool)
    patch[rows - r0, cols - c0] = True
    assert int(patch.sum()) == 5 and h * w == 6, "each object must be a single-corner-missing rectangle"

    missing = np.argwhere(~patch)
    assert missing.shape[0] == 1, "expected exactly one missing cell"
    miss_r, miss_c = map(int, missing[0])
    assert miss_r in {0, h - 1} and miss_c in {0, w - 1}, "missing cell must be a bbox corner"

    local_cells = [(int(r), int(c)) for r, c in zip(*np.where(patch), strict=False)]
    mirrored: set[tuple[int, int]] = set()

    if (h, w) == (2, 3):
        shift_c = -1 if miss_c == 0 else 1
        for rr, cc in local_cells:
            mirrored.add((rr, (w - 1 - cc) + shift_c))
    else:
        shift_r = -1 if miss_r == 0 else 1
        for rr, cc in local_cells:
            mirrored.add(((h - 1 - rr) + shift_r, cc))

    return {(r0 + rr, c0 + cc) for rr, cc in mirrored}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each quadrant L-shape with a marker-colored mirrored extension determined by its missing corner.

    Input structure
    - The background is the unique dominant color.
    - One non-background color forms exactly one full horizontal divider row and one full vertical divider column.
    - Each quadrant contains exactly one isolated one-pixel corner marker of its own color.
    - The remaining foreground objects are 4-connected 5-pixel L shapes whose bbox is either 2x3 or 3x2.

    Rule
    - Keep the divider cross, the corner markers, and every original L pixel unchanged.
    - For each 2x3 L, reflect it across a vertical axis and shift the mirrored copy one cell toward the missing left/right side.
    - For each 3x2 L, reflect it across a horizontal axis and shift the mirrored copy one cell toward the missing top/bottom side.
    - Paint only the new visible cells of that mirrored copy with the corner-marker color of the L's quadrant.

    Key insight
    - The geometry cue and the color cue come from different places: the L's missing corner picks the mirror direction, while the quadrant's isolated corner marker supplies the copy color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    divider_color, mid_r, mid_c = _find_divider(g, bg)
    out = g.copy()

    for quadrant in QUADRANTS:
        r0, r1, c0, c1 = _quadrant_bounds(g.shape, mid_r, mid_c, quadrant)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = (sub != bg) & (sub != divider_color)
        comps = _find_components(sub, mask)
        assert comps, "each quadrant must contain at least its corner marker"

        markers = [comp for comp in comps if comp["area"] == 1]
        shapes = [comp for comp in comps if comp["area"] == 5]
        others = [comp for comp in comps if comp["area"] not in {1, 5}]
        assert not others, "quadrants may contain only one-pixel markers and 5-pixel L shapes"
        assert len(markers) == 1, "each quadrant must contain exactly one corner marker"
        marker_color = int(markers[0]["color"])

        for comp in shapes:
            abs_coords = comp["coords"] + np.array([[r0, c0]])
            obj_color = int(comp["color"])
            for rr, cc in _mirror_completion_coords(abs_coords):
                assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "mirrored copy must stay inside the grid"
                existing = int(out[rr, cc])
                assert existing in {bg, obj_color}, "mirrored copy may only overlap background or its own source"
                if existing == bg:
                    out[rr, cc] = marker_color

    return out


# --------------- generator ---------------


def _original_local_cells(kind: str, missing_corner: tuple[str, str]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    vert, horiz = missing_corner
    if kind == "wide":
        foot_h, foot_w = 2, 4
        base_r = 0
        base_c = 1 if horiz == "left" else 0
        miss_r = 0 if vert == "top" else 1
        miss_c = 0 if horiz == "left" else 2
        obj_h, obj_w = 2, 3
    else:
        foot_h, foot_w = 4, 2
        base_r = 1 if vert == "top" else 0
        base_c = 0
        miss_r = 0 if vert == "top" else 2
        miss_c = 0 if horiz == "left" else 1
        obj_h, obj_w = 3, 2

    cells: list[tuple[int, int]] = []
    for r in range(obj_h):
        for c in range(obj_w):
            if (r, c) != (miss_r, miss_c):
                cells.append((base_r + r, base_c + c))
    return (foot_h, foot_w), cells


def _candidate_positions(
    bounds: tuple[int, int, int, int],
    foot_h: int,
    foot_w: int,
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    positions: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1 - foot_h + 2):
        for c in range(c0 + 1, c1 - foot_w + 2):
            if blocked[r - 1 : r + foot_h + 1, c - 1 : c + foot_w + 1].any():
                continue
            positions.append((r, c))
    random.shuffle(positions)
    return positions


def _place_object(
    blocked: np.ndarray,
    shape: tuple[int, int],
    mid_r: int,
    mid_c: int,
    kind: str,
    preferred_quadrants: list[int],
) -> tuple[int, int, int] | None:
    foot_h, foot_w = (2, 4) if kind == "wide" else (4, 2)
    all_quadrants = preferred_quadrants + [q for q in QUADRANTS if q not in preferred_quadrants]
    for quadrant in all_quadrants:
        bounds = _quadrant_bounds(shape, mid_r, mid_c, quadrant)
        positions = _candidate_positions(bounds, foot_h, foot_w, blocked)
        if positions:
            r, c = random.choice(positions)
            blocked[r - 1 : r + foot_h + 1, c - 1 : c + foot_w + 1] = True
            return quadrant, r, c
    return None


def _render_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(13, 29)
    w = random.randint(13, 29)
    mid_r = random.randint(6, h - 7)
    mid_c = random.randint(6, w - 7)

    palette = list(range(10))
    bg, divider = random.sample(palette, 2)
    remaining = [color for color in palette if color not in {bg, divider}]
    marker_colors = random.sample(remaining, 4)
    object_colors = [color for color in remaining if color not in set(marker_colors)]
    assert len(object_colors) >= 1

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    inp[mid_r, :] = out[mid_r, :] = divider
    inp[:, mid_c] = out[:, mid_c] = divider

    blocked = np.zeros((h, w), dtype=bool)
    blocked[mid_r, :] = True
    blocked[:, mid_c] = True

    for quadrant, marker_color in enumerate(marker_colors):
        rr, cc = _quadrant_marker_cell((h, w), quadrant)
        inp[rr, cc] = out[rr, cc] = marker_color
        blocked[rr, cc] = True

    placements: list[tuple[str, tuple[str, str], int, int, int, int]] = []
    mandatory_quadrants = random.sample(list(QUADRANTS), 2)
    mandatory_specs = [
        ("wide", mandatory_quadrants[0]),
        ("tall", mandatory_quadrants[1]),
    ]

    for kind, preferred in mandatory_specs:
        placed = _place_object(blocked, (h, w), mid_r, mid_c, kind, [preferred])
        if placed is None:
            return None
        quadrant, r, c = placed
        placements.append((kind, random.choice(MISSING_CORNERS), random.choice(object_colors), quadrant, r, c))

    for _ in range(random.randint(0, 3)):
        kind = random.choice(OBJECT_KINDS)
        placed = _place_object(blocked, (h, w), mid_r, mid_c, kind, random.sample(list(QUADRANTS), len(QUADRANTS)))
        if placed is None:
            break
        quadrant, r, c = placed
        placements.append((kind, random.choice(MISSING_CORNERS), random.choice(object_colors), quadrant, r, c))

    if len(placements) < 2:
        return None

    for kind, missing_corner, obj_color, quadrant, base_r, base_c in placements:
        _foot_shape, local_cells = _original_local_cells(kind, missing_corner)
        abs_coords = np.array([(base_r + dr, base_c + dc) for dr, dc in local_cells], dtype=int)
        inp[abs_coords[:, 0], abs_coords[:, 1]] = obj_color
        out[abs_coords[:, 0], abs_coords[:, 1]] = obj_color

        marker_color = marker_colors[quadrant]
        for rr, cc in _mirror_completion_coords(abs_coords):
            existing = int(out[rr, cc])
            assert existing in {bg, obj_color}
            if existing == bg:
                out[rr, cc] = marker_color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rendered = _render_attempt()
    if rendered is None:
        return None
    inp, expected = rendered
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
