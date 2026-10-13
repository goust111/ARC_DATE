import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDE_NAMES = ("top", "bottom", "left", "right")

GRID_MIN = 18
GRID_MAX = 28
RECT_COUNT_CHOICES = (1, 2, 3)
BOX_MIN = 5
BOX_MAX = 9
INPUT_PADDING = 2
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
TRUNCATE_PROB = 0.45


def _side_gap(side: np.ndarray):
    if side[0] != GREY or side[-1] != GREY:
        return None
    if np.any((side != GREY) & (side != BLACK)):
        return None

    interior = side[1:-1]
    black_idx = np.flatnonzero(interior == BLACK)
    if black_idx.size == 0:
        if np.all(interior == GREY):
            return ()
        return None

    if black_idx[-1] - black_idx[0] + 1 != black_idx.size:
        return None
    if black_idx[0] < 1 or black_idx[-1] > len(interior) - 2:
        return None
    if np.any(interior[: black_idx[0]] != GREY):
        return None
    if np.any(interior[black_idx[-1] + 1 :] != GREY):
        return None
    return ((int(black_idx[0] + 1), int(black_idx[-1] + 1)),)


def _validate_box(g: np.ndarray, sr: int, sc: int, r0: int, c0: int, r1: int, c1: int):
    if r1 - r0 + 1 < BOX_MIN or c1 - c0 + 1 < BOX_MIN:
        return None

    top_gap = _side_gap(g[r0, c0 : c1 + 1])
    bottom_gap = _side_gap(g[r1, c0 : c1 + 1])
    left_gap = _side_gap(g[r0 : r1 + 1, c0])
    right_gap = _side_gap(g[r0 : r1 + 1, c1])
    if None in (top_gap, bottom_gap, left_gap, right_gap):
        return None

    gaps = {
        "top": top_gap,
        "bottom": bottom_gap,
        "left": left_gap,
        "right": right_gap,
    }
    if not any(gaps[side] for side in SIDE_NAMES):
        return None

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(interior == GREY):
        return None
    colored = np.argwhere((interior != BLACK) & (interior != GREY))
    if len(colored) != 1:
        return None
    rr, cc = map(int, colored[0])
    if (r0 + 1 + rr, c0 + 1 + cc) != (sr, sc):
        return None

    seed_color = int(g[sr, sc])
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "seed": (int(sr), int(sc)),
        "color": seed_color,
        "gaps": gaps,
    }


def _find_box_for_seed(g: np.ndarray, sr: int, sc: int):
    grey_rows = np.flatnonzero(np.any(g == GREY, axis=1))
    rows_above = [
        int(r)
        for r in grey_rows
        if r < sr and np.any(g[r, :sc] == GREY) and np.any(g[r, sc + 1 :] == GREY)
    ]
    rows_below = [
        int(r)
        for r in grey_rows
        if r > sr and np.any(g[r, :sc] == GREY) and np.any(g[r, sc + 1 :] == GREY)
    ]
    rows_above.sort(reverse=True)
    rows_below.sort()

    best = None
    best_area = None
    for r0 in rows_above:
        for r1 in rows_below:
            common_cols = np.flatnonzero((g[r0] == GREY) & (g[r1] == GREY))
            c0_candidates = [int(c) for c in common_cols if c < sc]
            c1_candidates = [int(c) for c in common_cols if c > sc]
            c0_candidates.sort(reverse=True)
            c1_candidates.sort()
            for c0 in c0_candidates:
                for c1 in c1_candidates:
                    candidate = _validate_box(g, sr, sc, r0, c0, r1, c1)
                    if candidate is None:
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if best_area is None or area < best_area:
                        best = candidate
                        best_area = area
    assert best is not None, "each seed must lie in exactly one valid frame"
    return best


def _paint_branch(out: np.ndarray, original: np.ndarray, side: str, bbox, gap_span, color: int):
    h, w = out.shape
    r0, c0, r1, c1 = bbox

    if side in ("top", "bottom"):
        gap_c0, gap_c1 = gap_span
        width = gap_c1 - gap_c0 + 1
        row_start = r0 - width if side == "top" else r1 + 1
        row_stop = r0 if side == "top" else r1 + 1 + width
        for r in range(row_start, row_stop):
            if not (0 <= r < h):
                continue
            for c in range(gap_c0, gap_c1 + 1):
                if 0 <= c < w and original[r, c] == BLACK:
                    out[r, c] = color
    else:
        gap_r0, gap_r1 = gap_span
        width = gap_r1 - gap_r0 + 1
        col_start = c0 - width if side == "left" else c1 + 1
        col_stop = c0 if side == "left" else c1 + 1 + width
        for r in range(gap_r0, gap_r1 + 1):
            if not (0 <= r < h):
                continue
            for c in range(col_start, col_stop):
                if 0 <= c < w and original[r, c] == BLACK:
                    out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    seeds = np.argwhere((g != BLACK) & (g != GREY))
    for sr, sc in sorted((tuple(map(int, pos)) for pos in seeds)):
        frame = _find_box_for_seed(g, sr, sc)
        r0, c0, r1, c1 = frame["bbox"]
        color = frame["color"]

        for start, end in frame["gaps"]["top"]:
            _paint_branch(out, g, "top", (r0, c0, r1, c1), (c0 + start, c0 + end), color)
        for start, end in frame["gaps"]["bottom"]:
            _paint_branch(out, g, "bottom", (r0, c0, r1, c1), (c0 + start, c0 + end), color)
        for start, end in frame["gaps"]["left"]:
            _paint_branch(out, g, "left", (r0, c0, r1, c1), (r0 + start, r0 + end), color)
        for start, end in frame["gaps"]["right"]:
            _paint_branch(out, g, "right", (r0, c0, r1, c1), (r0 + start, r0 + end), color)

    return out


def _branch_cells(spec, shape):
    h, w = shape
    r0, c0, r1, c1 = spec["bbox"]
    cells = []

    for side, span in spec["gaps"].items():
        if side in ("top", "bottom"):
            gap_c0, gap_c1 = span
            width = gap_c1 - gap_c0 + 1
            row_start = r0 - width if side == "top" else r1 + 1
            row_stop = r0 if side == "top" else r1 + 1 + width
            for r in range(row_start, row_stop):
                if 0 <= r < h:
                    for c in range(gap_c0, gap_c1 + 1):
                        if 0 <= c < w:
                            cells.append((int(r), int(c)))
        else:
            gap_r0, gap_r1 = span
            width = gap_r1 - gap_r0 + 1
            col_start = c0 - width if side == "left" else c1 + 1
            col_stop = c0 if side == "left" else c1 + 1 + width
            for r in range(gap_r0, gap_r1 + 1):
                if 0 <= r < h:
                    for c in range(col_start, col_stop):
                        if 0 <= c < w:
                            cells.append((int(r), int(c)))
    return cells


def _gap_span(rng: random.Random, side_len: int):
    max_width = side_len - 4
    assert max_width >= 1
    width = rng.randint(1, max_width)
    before = rng.randint(1, side_len - width - 3)
    start = before
    end = start + width - 1
    return width, start, end


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int):
    rr0 = max(0, r0 - INPUT_PADDING)
    cc0 = max(0, c0 - INPUT_PADDING)
    rr1 = min(occupied.shape[0] - 1, r1 + INPUT_PADDING)
    cc1 = min(occupied.shape[1] - 1, c1 + INPUT_PADDING)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int):
    rr0 = max(0, r0 - INPUT_PADDING)
    cc0 = max(0, c0 - INPUT_PADDING)
    rr1 = min(occupied.shape[0] - 1, r1 + INPUT_PADDING)
    cc1 = min(occupied.shape[1] - 1, c1 + INPUT_PADDING)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_rectangle(rng: random.Random, h: int, w: int, occupied: np.ndarray, color: int):
    for _ in range(PLACEMENT_ATTEMPTS):
        side_count = rng.randint(1, 4)
        sides = rng.sample(SIDE_NAMES, side_count)
        near_side = rng.choice(sides) if rng.random() < TRUNCATE_PROB else None

        box_h = rng.randint(BOX_MIN, BOX_MAX)
        box_w = rng.randint(BOX_MIN, BOX_MAX)
        if near_side in ("top", "bottom"):
            box_w = max(box_w, 6)
        if near_side in ("left", "right"):
            box_h = max(box_h, 6)
        if box_h >= h - 1 or box_w >= w - 1:
            continue

        top_margin = 1 if near_side == "top" else 3
        bottom_margin = 1 if near_side == "bottom" else 3
        left_margin = 1 if near_side == "left" else 3
        right_margin = 1 if near_side == "right" else 3

        r0_min = top_margin
        r0_max = h - box_h - bottom_margin
        c0_min = left_margin
        c0_max = w - box_w - right_margin
        if r0_min > r0_max or c0_min > c0_max:
            continue

        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1
        if not _can_place(occupied, r0, c0, r1, c1):
            continue

        gaps = {}
        ok = True
        for side in sides:
            side_len = box_w if side in ("top", "bottom") else box_h
            width, start, end = _gap_span(rng, side_len)
            if side == near_side and width < 2 and side_len >= 6:
                width = rng.randint(2, side_len - 4)
                before = rng.randint(1, side_len - width - 3)
                start = before
                end = start + width - 1

            if side == "top":
                gap = (c0 + start + 1, c0 + end + 1)
            elif side == "bottom":
                gap = (c0 + start + 1, c0 + end + 1)
            elif side == "left":
                gap = (r0 + start + 1, r0 + end + 1)
            else:
                gap = (r0 + start + 1, r0 + end + 1)
            gaps[side] = gap

            distance = {
                "top": r0,
                "bottom": h - 1 - r1,
                "left": c0,
                "right": w - 1 - c1,
            }[side]
            if side != near_side and width > distance + 1:
                ok = False
                break
        if not ok:
            continue

        seed_r = rng.randint(r0 + 1, r1 - 1)
        seed_c = rng.randint(c0 + 1, c1 - 1)
        spec = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "seed": (int(seed_r), int(seed_c)),
            "color": int(color),
            "gaps": gaps,
        }
        _mark_occupied(occupied, r0, c0, r1, c1)
        return spec
    return None


def _draw_input(shape, specs):
    g = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = spec["color"]
        g[r0, c0 : c1 + 1] = GREY
        g[r1, c0 : c1 + 1] = GREY
        g[r0 : r1 + 1, c0] = GREY
        g[r0 : r1 + 1, c1] = GREY

        for side, span in spec["gaps"].items():
            if side == "top":
                g[r0, span[0] : span[1] + 1] = BLACK
            elif side == "bottom":
                g[r1, span[0] : span[1] + 1] = BLACK
            elif side == "left":
                g[span[0] : span[1] + 1, c0] = BLACK
            else:
                g[span[0] : span[1] + 1, c1] = BLACK

        seed_r, seed_c = spec["seed"]
        g[seed_r, seed_c] = color
    return g


def _render_expected(inp: np.ndarray, specs):
    out = inp.copy()
    usage = np.zeros(inp.shape, dtype=int)
    for spec in specs:
        for r, c in _branch_cells(spec, inp.shape):
            if inp[r, c] != BLACK:
                return None
            usage[r, c] += 1
            if usage[r, c] > 1:
                return None
            out[r, c] = spec["color"]
    return out


def _generate_case(rng: random.Random):
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    occupied = np.zeros((h, w), dtype=bool)

    max_rects = 1 if min(h, w) < 20 else 2 if min(h, w) < 24 else 3
    rect_count = rng.choice(RECT_COUNT_CHOICES[:max_rects])
    colors = rng.sample(PALETTE, rect_count)

    specs = []
    for color in colors:
        spec = _place_rectangle(rng, h, w, occupied, color)
        if spec is None:
            return None
        specs.append(spec)

    inp = _draw_input((h, w), specs)
    expected = _render_expected(inp, specs)
    if expected is None:
        return None
    return inp, specs


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("generate_input: exceeded retry budget")
