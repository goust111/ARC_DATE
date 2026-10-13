import random
from collections import Counter

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _reflect_mask(mask: np.ndarray, axis: int) -> np.ndarray:
    h, w = mask.shape
    cols = np.arange(w)
    mirror = 2 * axis - cols
    valid = (mirror >= 0) & (mirror < w)
    out = np.zeros((h, w), dtype=bool)
    out[:, valid] = mask[:, mirror[valid]]
    return out


def _rectangle_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if area != int(coords.shape[0]):
        return None
    return r0, c0, r1, c1


def _touches_mask(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            return True
    return False


def _adjacent_non_bg_colors(grid: np.ndarray, mask: np.ndarray, bg: int) -> set[int]:
    h, w = grid.shape
    colors: set[int] = set()
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                value = int(grid[nr, nc])
                if value != bg:
                    colors.add(value)
    return colors


def _candidate_marker_colors(grid: np.ndarray, wound: np.ndarray, bg: int) -> list[int]:
    candidates = []
    for color in _adjacent_non_bg_colors(grid, wound, bg):
        positions = np.argwhere(grid == color)
        if positions.size == 0:
            continue
        if all(_touches_mask(wound, int(r), int(c)) for r, c in positions):
            candidates.append(int(color))
    return candidates


def _axis_candidate(grid: np.ndarray, axis: int, bg: int) -> tuple[tuple[int, int, int, float], np.ndarray, int] | None:
    occ = grid != bg
    mirrored_occ = _reflect_mask(occ, axis)
    wound = (~occ) & mirrored_occ
    bbox = _rectangle_bbox(wound)
    if bbox is None:
        return None

    marker_candidates = _candidate_marker_colors(grid, wound, bg)
    if len(marker_candidates) != 1:
        return None
    marker = marker_candidates[0]

    matched = int(np.count_nonzero(occ & mirrored_occ))
    mismatched = int(np.count_nonzero(occ ^ mirrored_occ))
    center_bias = -abs(axis - (grid.shape[1] - 1) / 2.0)
    score = (matched - mismatched, matched, -mismatched, center_bias)
    return score, wound, marker


def _find_wound(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    best: tuple[tuple[int, int, int, float], np.ndarray, int] | None = None
    for axis in range(grid.shape[1]):
        candidate = _axis_candidate(grid, axis, bg)
        if candidate is None:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None, "expected a unique mirrored wound"
    _, wound, marker = best
    return wound, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the broken rectangular region in a vertically symmetric pattern and
    fill it with the unique boundary marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    wound, marker = _find_wound(g, bg)

    out = g.copy()
    out[wound] = marker
    return out


# --------------- generator ---------------

BG = 0
COLOR_POOL = tuple(range(1, 10))

HEIGHT_RANGE = (15, 24)
WIDTH_RANGE = (17, 29)  # odd widths only

SPAN_RANGE = (8, 16)
EDGE_HALF_RANGE = (2, 5)
HALF_WIDTH_CAP = 10
WIDTH_STEP_CAP = 2
MIN_OBJECT_AREA = 70
MAX_OBJECT_FRACTION = 0.44

PATTERN_COLOR_COUNT_RANGE = (3, 5)
PATTERN_MODES = ("mirror_noise", "row_bands", "diagonal_bands", "diamond_bands")

WOUND_HEIGHT_RANGE = (2, 6)
WOUND_WIDTH_RANGE = (2, 6)

GEN_RETRIES = 400


def _bounded_steps(total: int, slots: int, cap: int) -> list[int]:
    steps = []
    remaining = total
    for i in range(slots):
        left = slots - i - 1
        lo = max(0, remaining - cap * left)
        hi = min(cap, remaining)
        step = random.randint(lo, hi)
        steps.append(step)
        remaining -= step
    return steps


def _halfwidth_profile(span: int, max_half: int) -> list[int] | None:
    edge_hi = min(EDGE_HALF_RANGE[1], max_half)
    start = random.randint(EDGE_HALF_RANGE[0], edge_hi)
    end = random.randint(EDGE_HALF_RANGE[0], edge_hi)
    up_need = (max_half - start + WIDTH_STEP_CAP - 1) // WIDTH_STEP_CAP
    down_need = (max_half - end + WIDTH_STEP_CAP - 1) // WIDTH_STEP_CAP
    if up_need + down_need + 1 > span:
        return None

    peak_start = random.randint(up_need, span - 1 - down_need)
    peak_end = random.randint(peak_start, span - 1 - down_need)

    widths = [start]
    for step in _bounded_steps(max_half - start, peak_start, WIDTH_STEP_CAP):
        widths.append(widths[-1] + step)

    widths.extend([max_half] * (peak_end - peak_start))

    current = max_half
    for step in _bounded_steps(max_half - end, span - 1 - peak_end, WIDTH_STEP_CAP):
        current -= step
        widths.append(current)

    if len(widths) != span:
        return None
    return widths


def _paint_pair(grid: np.ndarray, axis: int, r: int, c: int, color: int) -> None:
    mirror_c = 2 * axis - c
    grid[r, c] = color
    grid[r, mirror_c] = color


def _canonical_cells(mask: np.ndarray, axis: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    cells = [(int(r), int(c)) for r, c in coords if int(c) <= axis]
    random.shuffle(cells)
    return cells


def _paint_pattern(mask: np.ndarray, axis: int, colors: list[int]) -> np.ndarray:
    k = len(colors)
    out = np.zeros(mask.shape, dtype=int)
    mode = random.choice(PATTERN_MODES)

    if mode == "mirror_noise":
        for r, c in _canonical_cells(mask, axis):
            _paint_pair(out, axis, r, c, random.choice(colors))
    elif mode == "row_bands":
        band = random.randint(1, 3)
        row_offsets = []
        current = random.randrange(k)
        for _ in range(mask.shape[0]):
            if random.random() < 0.35:
                current = random.randrange(k)
            row_offsets.append(current)
        for r0, c0 in np.argwhere(mask):
            r = int(r0)
            c = int(c0)
            dist = abs(c - axis)
            out[r, c] = colors[(row_offsets[r] + dist // band) % k]
    elif mode == "diagonal_bands":
        a = random.randint(1, k)
        b = random.randint(1, k)
        phase = random.randrange(k)
        for r0, c0 in np.argwhere(mask):
            r = int(r0)
            c = int(c0)
            dist = abs(c - axis)
            out[r, c] = colors[(phase + a * r + b * dist) % k]
    else:
        a = random.randint(1, k)
        b = random.randint(1, k)
        phase = random.randrange(k)
        center_row = random.randint(mask.shape[0] // 4, 3 * mask.shape[0] // 4)
        for r0, c0 in np.argwhere(mask):
            r = int(r0)
            c = int(c0)
            dist = abs(c - axis)
            out[r, c] = colors[(phase + a * abs(r - center_row) + b * dist) % k]

    cells = _canonical_cells(mask, axis)
    assert len(cells) >= len(colors)
    for color, (r, c) in zip(colors, cells):
        _paint_pair(out, axis, r, c, color)

    out[~mask] = BG
    return out


def _make_shape(height: int, width: int) -> tuple[np.ndarray, int] | None:
    span_hi = min(SPAN_RANGE[1], height - 2)
    if span_hi < SPAN_RANGE[0]:
        return None
    span = random.randint(SPAN_RANGE[0], span_hi)

    max_half_hi = min(HALF_WIDTH_CAP, width // 2 - 3)
    if max_half_hi < EDGE_HALF_RANGE[0]:
        return None
    max_half = random.randint(max(EDGE_HALF_RANGE[1], EDGE_HALF_RANGE[0]), max_half_hi)

    profile = _halfwidth_profile(span, max_half)
    if profile is None:
        return None

    axis_choices = [
        axis
        for axis in range(max_half + 1, width - max_half - 1)
        if abs(axis - width // 2) <= 2
    ]
    if not axis_choices:
        axis_choices = list(range(max_half + 1, width - max_half - 1))
    if not axis_choices:
        return None
    axis = random.choice(axis_choices)

    top = random.randint(1, height - span - 1)
    mask = np.zeros((height, width), dtype=bool)
    for i, half in enumerate(profile):
        r = top + i
        mask[r, axis - half : axis + half + 1] = True

    area = int(np.count_nonzero(mask))
    if area < MIN_OBJECT_AREA:
        return None
    if area >= int(MAX_OBJECT_FRACTION * height * width):
        return None
    return mask, axis


def _wound_candidates(mask: np.ndarray, axis: int) -> list[tuple[int, int, int, int, str]]:
    h, w = mask.shape
    candidates = []
    for rect_h in range(WOUND_HEIGHT_RANGE[0], min(WOUND_HEIGHT_RANGE[1], h - 2) + 1):
        for rect_w in range(WOUND_WIDTH_RANGE[0], min(WOUND_WIDTH_RANGE[1], w - 2) + 1):
            for r0 in range(1, h - rect_h - 1):
                r1 = r0 + rect_h - 1
                for c0 in range(1, w - rect_w - 1):
                    c1 = c0 + rect_w - 1
                    if c1 <= axis - 2:
                        side = "left"
                    elif c0 >= axis + 2:
                        side = "right"
                    else:
                        continue
                    if not bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
                        continue
                    if not bool(np.all(mask[r0 - 1, c0 : c1 + 1])):
                        continue
                    if not bool(np.all(mask[r1 + 1, c0 : c1 + 1])):
                        continue
                    if not bool(np.all(mask[r0 : r1 + 1, c0 - 1])):
                        continue
                    if not bool(np.all(mask[r0 : r1 + 1, c1 + 1])):
                        continue
                    candidates.append((r0, c0, rect_h, rect_w, side))
    return candidates


def _ring_cells(r0: int, c0: int, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0 - 1, c))
        cells.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        cells.append((r, c0 - 1))
        cells.append((r, c1 + 1))
    return cells


def _oracle_ok(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(WIDTH_RANGE[0], WIDTH_RANGE[1])
    if width % 2 == 0:
        width += 1
    if width > WIDTH_RANGE[1]:
        width -= 2

    shaped = _make_shape(height, width)
    if shaped is None:
        return None
    mask, axis = shaped

    color_count = random.randint(*PATTERN_COLOR_COUNT_RANGE)
    chosen = random.sample(COLOR_POOL, color_count + 1)
    pattern_colors = chosen[:color_count]
    marker_color = chosen[-1]

    base = _paint_pattern(mask, axis, pattern_colors)
    if not np.array_equal(base, np.fliplr(base)[:, :]):
        # Shape coloring is explicitly mirrored; this is a cheap sanity guard.
        mirrored = np.zeros_like(base)
        for r in range(height):
            for c in range(width):
                mc = 2 * axis - c
                if 0 <= mc < width:
                    mirrored[r, c] = base[r, mc]
        if not np.array_equal(base, mirrored):
            return None

    candidates = _wound_candidates(mask, axis)
    if not candidates:
        return None

    side = random.choice(("left", "right"))
    side_candidates = [cand for cand in candidates if cand[4] == side]
    if not side_candidates:
        side_candidates = candidates
    r0, c0, rect_h, rect_w, _side = random.choice(side_candidates)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    solved = base.copy()
    for r, c in _ring_cells(r0, c0, rect_h, rect_w):
        solved[r, c] = marker_color
    solved[r0 : r1 + 1, c0 : c1 + 1] = marker_color

    inp = solved.copy()
    inp[r0 : r1 + 1, c0 : c1 + 1] = BG

    if _background_color(inp) != BG:
        return None
    if not _oracle_ok(inp, solved):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
