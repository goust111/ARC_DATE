from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_COLORS = [BLUE, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _indicator_pos(rect: dict) -> tuple[int, int]:
    r0, c0, h, w = rect["r0"], rect["c0"], rect["h"], rect["w"]
    side = rect["side"]
    if side == "left":
        return r0 + h // 2, c0
    if side == "right":
        return r0 + h // 2, c0 + w - 1
    if side == "up":
        return r0, c0 + w // 2
    if side == "down":
        return r0 + h - 1, c0 + w // 2
    raise ValueError(f"unknown side {side}")


def _render_rectangles(shape: tuple[int, int], rectangles: list[dict]) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    for rect in rectangles:
        r0, c0, h, w = rect["r0"], rect["c0"], rect["h"], rect["w"]
        if r0 < 0 or c0 < 0 or r0 + h > shape[0] or c0 + w > shape[1]:
            return None
        if np.any(grid[r0 : r0 + h, c0 : c0 + w] != BLACK):
            return None
        grid[r0 : r0 + h, c0 : c0 + w] = GREEN
        ir, ic = _indicator_pos(rect)
        grid[ir, ic] = RED
    return grid


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for comp in _components(grid != BLACK):
        colors = {int(grid[r, c]) for r, c in comp}
        if colors <= {GREEN, RED}:
            red_pixels = [(r, c) for r, c in comp if int(grid[r, c]) == RED]
            if len(red_pixels) != 1:
                raise AssertionError("each rectangle component must have exactly one red indicator")
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                raise AssertionError("rectangle component must fill its bounding box")
            rr, cc = red_pixels[0]
            sides = []
            if rr == r0:
                sides.append("up")
            if rr == r1:
                sides.append("down")
            if cc == c0:
                sides.append("left")
            if cc == c1:
                sides.append("right")
            if len(sides) != 1:
                raise AssertionError("indicator must lie on exactly one rectangle side")
            rectangles.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "side": sides[0],
                    "indicator": (rr, cc),
                }
            )
    return rectangles


def _extend_walls(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    for rect in _extract_rectangles(out):
        r0, c0, rh, rw = rect["r0"], rect["c0"], rect["h"], rect["w"]
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        ir, ic = rect["indicator"]
        side = rect["side"]
        if side == "left":
            out[r0 : r1 + 1, :c0] = GREEN
            out[ir, : ic + 1] = RED
        elif side == "right":
            out[r0 : r1 + 1, c1 + 1 :] = GREEN
            out[ir, ic:] = RED
        elif side == "up":
            out[:r0, c0 : c1 + 1] = GREEN
            out[: ir + 1, ic] = RED
        elif side == "down":
            out[r1 + 1 :, c0 : c1 + 1] = GREEN
            out[ir:, ic] = RED
        else:
            raise AssertionError("unsupported side")
    return out


def _region_components(stage: np.ndarray) -> list[list[tuple[int, int]]]:
    wall_mask = (stage == GREEN) | (stage == RED)
    return _components(~wall_mask)


def _region_markers(stage: np.ndarray, region: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    markers = []
    for r, c in region:
        color = int(stage[r, c])
        if color not in (BLACK, RED, GREEN):
            markers.append((r, c, color))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each indicator-marked green rectangle into a wall, then fill each resulting region with its marker color.

    Input structure:
    - The grid has a black background, 2-3 small solid green rectangles, and one red indicator pixel on the border of each rectangle.
    - Additional isolated marker pixels use colors other than black, green, and red.
    - The rectangles are positioned so that their extensions and the canvas boundary partition the grid into regions.

    Rule:
    - For every green rectangle, determine which side contains its red border indicator.
    - Extend a green band from that side to the canvas edge with the rectangle's perpendicular thickness, and draw a red line from the indicator pixel to the same edge.
    - Treat green walls and red lines as barriers, find every 4-connected non-wall region while keeping marker pixels inside their regions, and if a region contains exactly one marker pixel, fill all black cells in that region with that marker's color.

    Key insight:
    - The rectangle growth is not the final goal: it builds the walls that define the regions, and the single marker inside each region chooses that region's fill color.
    """
    stage = _extend_walls(np.asarray(grid, dtype=int))
    out = stage.copy()
    for region in _region_components(stage):
        markers = _region_markers(stage, region)
        if len(markers) != 1:
            continue
        fill_color = markers[0][2]
        for r, c in region:
            if out[r, c] == BLACK:
                out[r, c] = fill_color
    return out


def _full_horizontal(h: int, w: int, row0: int) -> dict:
    if random.choice([True, False]):
        return {"r0": row0, "c0": 0, "h": 3, "w": 2, "side": "right"}
    return {"r0": row0, "c0": w - 2, "h": 3, "w": 2, "side": "left"}


def _full_vertical(h: int, w: int, col0: int) -> dict:
    if random.choice([True, False]):
        return {"r0": 0, "c0": col0, "h": 2, "w": 3, "side": "down"}
    return {"r0": h - 2, "c0": col0, "h": 2, "w": 3, "side": "up"}


def _marker_candidates(inp: np.ndarray, region: list[tuple[int, int]], chosen: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = inp.shape
    candidates = []
    for r, c in region:
        if inp[r, c] != BLACK:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            continue
        ok = True
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if inp[nr, nc] in (GREEN, RED):
                    ok = False
        if not ok:
            continue
        if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in chosen):
            continue
        candidates.append((r, c))
    return candidates


def _sample_hvv_layout() -> tuple[int, int, list[dict]] | None:
    h = random.randint(14, 19)
    w = random.randint(16, 22)
    row_choices = list(range(3, h - 5))
    col_pairs = [
        (c0, c1)
        for c0 in range(3, w - 9)
        for c1 in range(c0 + 6, w - 3)
        if c1 + 3 <= w - 3
    ]
    if not row_choices or not col_pairs:
        return None
    hr = random.choice(row_choices)
    vc0, vc1 = random.choice(col_pairs)
    rects = [_full_horizontal(h, w, hr), _full_vertical(h, w, vc0), _full_vertical(h, w, vc1)]
    return h, w, rects


def _sample_vhh_layout() -> tuple[int, int, list[dict]] | None:
    h = random.randint(16, 22)
    w = random.randint(14, 19)
    col_choices = list(range(3, w - 5))
    row_pairs = [
        (r0, r1)
        for r0 in range(3, h - 9)
        for r1 in range(r0 + 6, h - 3)
        if r1 + 3 <= h - 3
    ]
    if not col_choices or not row_pairs:
        return None
    vc = random.choice(col_choices)
    hr0, hr1 = random.choice(row_pairs)
    rects = [_full_vertical(h, w, vc), _full_horizontal(h, w, hr0), _full_horizontal(h, w, hr1)]
    return h, w, rects


def _sample_cross_layout() -> tuple[int, int, list[dict]] | None:
    h = random.randint(14, 19)
    w = random.randint(14, 19)
    row_choices = list(range(3, h - 5))
    col_choices = list(range(3, w - 5))
    if not row_choices or not col_choices:
        return None
    hr = random.choice(row_choices)
    vc = random.choice(col_choices)
    rects = [_full_horizontal(h, w, hr), _full_vertical(h, w, vc)]
    return h, w, rects


def _sample_rectangles() -> tuple[np.ndarray, int] | None:
    modes = ["cross", "hvv", "vhh"]
    weights = [3, 2, 2]
    mode = random.choices(modes, weights=weights, k=1)[0]
    if mode == "cross":
        sampled = _sample_cross_layout()
    elif mode == "hvv":
        sampled = _sample_hvv_layout()
    else:
        sampled = _sample_vhh_layout()
    if sampled is None:
        return None
    h, w, rectangles = sampled
    grid = _render_rectangles((h, w), rectangles)
    if grid is None:
        return None
    if len(_extract_rectangles(grid)) != len(rectangles):
        return None
    return grid, len(rectangles)


def _validate_generated(inp: np.ndarray, rectangle_count: int) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    try:
        rectangles = _extract_rectangles(inp)
        if len(rectangles) != rectangle_count:
            return False
        marker_cells = [(int(r), int(c), int(inp[r, c])) for r, c in np.argwhere((inp != BLACK) & (inp != RED) & (inp != GREEN))]
        if not (3 <= len(marker_cells) <= 5):
            return False
        if len({color for _, _, color in marker_cells}) != len(marker_cells):
            return False
        stage = _extend_walls(inp)
        regions = _region_components(stage)
        if len(regions) <= len(marker_cells):
            return False

        marked_regions = 0
        unmarked_regions = 0
        for region in regions:
            markers = _region_markers(stage, region)
            if len(markers) == 0:
                unmarked_regions += 1
            elif len(markers) == 1:
                marked_regions += 1
            else:
                return False
        if marked_regions != len(marker_cells) or unmarked_regions < 1:
            return False

        out = main(inp)
        wall_mask = (stage == GREEN) | (stage == RED)
        if not np.array_equal(out[wall_mask], stage[wall_mask]):
            return False
        for region in regions:
            markers = _region_markers(stage, region)
            black_cells = [(r, c) for r, c in region if int(stage[r, c]) == BLACK]
            if len(markers) == 1:
                fill_color = markers[0][2]
                if any(int(out[r, c]) != fill_color for r, c in black_cells):
                    return False
            else:
                if any(int(out[r, c]) != BLACK for r, c in black_cells):
                    return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sampled = _sample_rectangles()
        if sampled is None:
            continue
        base, rectangle_count = sampled
        stage = _extend_walls(base)
        regions = _region_components(stage)
        if len(regions) == 4:
            marker_count = 3
        elif len(regions) == 6:
            marker_count = random.choice([4, 5])
        else:
            continue

        chosen_regions = random.sample(range(len(regions)), marker_count)
        colors = random.sample(MARKER_COLORS, marker_count)
        inp = base.copy()
        chosen_cells: list[tuple[int, int]] = []

        ok = True
        for region_idx, color in zip(chosen_regions, colors):
            candidates = _marker_candidates(inp, regions[region_idx], chosen_cells)
            if not candidates:
                ok = False
                break
            r, c = random.choice(candidates)
            inp[r, c] = color
            chosen_cells.append((r, c))
        if not ok:
            continue
        if _validate_generated(inp, rectangle_count):
            return inp
    raise RuntimeError("failed to generate valid puzzle inputs")
