from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER_COLOR = 9
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
SIDES = ("top", "bottom", "left", "right")
INWARD = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

MIN_LAYERS = 2
MAX_LAYERS = 5
MIN_GRID = 12
MAX_GRID = 30
GEN_ATTEMPTS = 400


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components


def _parse_frame(component: dict) -> dict:
    coords = component["coords"]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame bbox must be at least 3x3"

    coord_set = set(coords)
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    assert coord_set <= border, "frame pixels must lie on the rectangle border"
    missing = list(border - coord_set)
    assert len(missing) == 1, "each frame must have exactly one opening"
    opening = missing[0]

    if opening[0] == r0:
        side = "top"
        assert c0 < opening[1] < c1
    elif opening[0] == r1:
        side = "bottom"
        assert c0 < opening[1] < c1
    elif opening[1] == c0:
        side = "left"
        assert r0 < opening[0] < r1
    else:
        assert opening[1] == c1
        side = "right"
        assert r0 < opening[0] < r1

    return {
        "color": int(component["color"]),
        "bbox": (r0, c0, r1, c1),
        "opening": opening,
        "side": side,
    }


def _bbox_area(frame: dict) -> int:
    r0, c0, r1, c1 = frame["bbox"]
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _path_is_clear(grid: np.ndarray, start: tuple[int, int], opening: tuple[int, int], side: str) -> bool:
    dr, dc = INWARD[side]
    r, c = start
    orow, ocol = opening

    if dr != 0:
        if c != ocol or (orow - r) * dr <= 0:
            return False
        for rr in range(r + dr, orow + dr, dr):
            if int(grid[rr, c]) != BG:
                return False
    else:
        if r != orow or (ocol - c) * dc <= 0:
            return False
        for cc in range(c + dc, ocol + dc, dc):
            if int(grid[r, cc]) != BG:
                return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the outside marker through a nested chain of one-opening frames and keep only its final landing cell in the innermost frame.

    Input structure:
    - The background is black.
    - Several differently colored one-pixel-thick rectangular frames are nested inside one another.
    - Each frame has exactly one one-cell opening on one side, and a maroon marker sits just outside the outermost opening.

    Rule:
    - Enter the outermost frame through its opening and land on the first black cell just inside it.
    - Then read the next inner frame's opening side and move straight toward that opening, pass through it, and land on the first black cell just inside that frame.
    - Repeat until the marker has entered the innermost frame, erase the original outside marker, and draw only the final maroon landing cell.

    Key insight:
    - The openings form a directional instruction sequence across nested frames; each step preserves one coordinate from the previous landing point, so the whole opening sequence determines the final position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g, bg=BG)
    marker_components = [comp for comp in components if len(comp["coords"]) == 1]
    assert len(marker_components) == 1, "expected exactly one outside marker"
    marker = marker_components[0]
    marker_pos = marker["coords"][0]
    marker_color = int(marker["color"])

    frames = [_parse_frame(comp) for comp in components if comp is not marker]
    assert frames, "expected at least one frame"
    frames.sort(key=lambda frame: (-_bbox_area(frame), frame["bbox"][0], frame["bbox"][1]))

    out = g.copy()
    out[marker_pos] = BG

    current = marker_pos
    for i, frame in enumerate(frames):
        side = frame["side"]
        opening = frame["opening"]
        dr, dc = INWARD[side]
        landing = (opening[0] + dr, opening[1] + dc)

        if i == 0:
            expected_marker = (opening[0] - dr, opening[1] - dc)
            assert current == expected_marker, "outer marker must touch the outermost opening"

        assert _path_is_clear(g, current, opening, side), "path to the next opening must be clear"
        assert int(g[landing]) == BG, "landing cell inside the frame must be black"
        current = landing

    out[current] = marker_color
    return out


def _minimum_bbox_dim(remaining_inner_layers: int) -> int:
    return 3 + 4 * remaining_inner_layers


def _spans_containing(x: int, lo: int, hi: int, min_len: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if hi - lo + 1 < min_len:
        return spans
    for a in range(lo, min(hi, x - 1) + 1):
        b0 = max(a + min_len - 1, x + 1)
        for b in range(b0, hi + 1):
            spans.append((a, b))
    return spans


def _intervals_above(x: int, lo: int, hi: int, min_len: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if hi - lo + 1 < min_len:
        return spans
    for a in range(max(lo, x + 1), hi + 1):
        for b in range(a + min_len - 1, hi + 1):
            spans.append((a, b))
    return spans


def _intervals_below(x: int, lo: int, hi: int, min_len: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if hi - lo + 1 < min_len:
        return spans
    upper = min(hi, x - 1)
    for a in range(lo, upper + 1):
        for b in range(a + min_len - 1, upper + 1):
            spans.append((a, b))
    return spans


def _sample_inner_frame(
    prev_bbox: tuple[int, int, int, int],
    current: tuple[int, int],
    remaining_inner_layers: int,
) -> dict | None:
    pr0, pc0, pr1, pc1 = prev_bbox
    cr, cc = current
    min_h = _minimum_bbox_dim(remaining_inner_layers)
    min_w = _minimum_bbox_dim(remaining_inner_layers)

    row_lo, row_hi = pr0 + 2, pr1 - 2
    col_lo, col_hi = pc0 + 2, pc1 - 2

    rows_through = _spans_containing(cr, row_lo, row_hi, min_h)
    cols_through = _spans_containing(cc, col_lo, col_hi, min_w)
    rows_above = _intervals_above(cr, row_lo, row_hi, min_h)
    rows_below = _intervals_below(cr, row_lo, row_hi, min_h)
    cols_above = _intervals_above(cc, col_lo, col_hi, min_w)
    cols_below = _intervals_below(cc, col_lo, col_hi, min_w)

    choices: list[tuple[str, list[tuple[int, int]], list[tuple[int, int]]]] = []
    if rows_above and cols_through:
        choices.append(("top", rows_above, cols_through))
    if rows_below and cols_through:
        choices.append(("bottom", rows_below, cols_through))
    if rows_through and cols_above:
        choices.append(("left", rows_through, cols_above))
    if rows_through and cols_below:
        choices.append(("right", rows_through, cols_below))
    if not choices:
        return None

    side, row_choices, col_choices = random.choice(choices)
    r0, r1 = random.choice(row_choices)
    c0, c1 = random.choice(col_choices)

    if side == "top":
        opening = (r0, cc)
    elif side == "bottom":
        opening = (r1, cc)
    elif side == "left":
        opening = (cr, c0)
    else:
        opening = (cr, c1)

    return {
        "bbox": (r0, c0, r1, c1),
        "opening": opening,
        "side": side,
    }


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, opening: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[opening] = BG


def _build_attempt() -> np.ndarray | None:
    layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    outer_min = _minimum_bbox_dim(layers - 1)

    h = random.randint(max(MIN_GRID, outer_min + 2), MAX_GRID)
    w = random.randint(max(MIN_GRID, outer_min + 2), MAX_GRID)

    frame_h = random.randint(outer_min, h - 2)
    frame_w = random.randint(outer_min, w - 2)
    r0 = random.randint(1, h - frame_h - 1)
    c0 = random.randint(1, w - frame_w - 1)
    outer_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    side = random.choice(SIDES)
    if side in ("top", "bottom"):
        opening = (outer_bbox[0] if side == "top" else outer_bbox[2], random.randint(outer_bbox[1] + 1, outer_bbox[3] - 1))
    else:
        opening = (random.randint(outer_bbox[0] + 1, outer_bbox[2] - 1), outer_bbox[1] if side == "left" else outer_bbox[3])

    dr, dc = INWARD[side]
    marker_pos = (opening[0] - dr, opening[1] - dc)
    current = (opening[0] + dr, opening[1] + dc)

    frames = [{"bbox": outer_bbox, "opening": opening, "side": side}]
    prev_bbox = outer_bbox
    for depth in range(1, layers):
        remaining = layers - depth - 1
        frame = _sample_inner_frame(prev_bbox, current, remaining)
        if frame is None:
            return None
        frames.append(frame)
        fdr, fdc = INWARD[frame["side"]]
        current = (frame["opening"][0] + fdr, frame["opening"][1] + fdc)
        prev_bbox = frame["bbox"]

    colors = random.sample(FRAME_COLORS, layers)
    grid = np.full((h, w), BG, dtype=int)
    for frame, color in zip(frames, colors):
        _draw_frame(grid, frame["bbox"], color, frame["opening"])
    grid[marker_pos] = MARKER_COLOR
    return grid


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    return (
        isinstance(out, np.ndarray)
        and out.shape == grid.shape
        and int(np.count_nonzero(grid == MARKER_COLOR)) == 1
        and int(np.count_nonzero(out == MARKER_COLOR)) == 1
        and int(np.count_nonzero(grid != out)) == 2
    )


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None and _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid nested-frame instance")
