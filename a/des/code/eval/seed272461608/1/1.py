import random

import numpy as np


BG_COLOR = 0
NON_BG_COLORS = list(range(1, 10))

GRID_MIN = 14
GRID_MAX = 24
FRAME_COUNT_RANGE = (1, 4)
INNER_H_RANGE = (2, 6)
INNER_W_RANGE = (2, 6)
FRAME_MARGIN_FROM_EDGE = 2
FRAME_GAP = 1
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300

DIRECTIONS = ("top", "bottom", "left", "right")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _single_contiguous_gap(seg: np.ndarray, color: int, bg: int) -> bool:
    gap_idx = np.flatnonzero(seg != color)
    if gap_idx.size == 0:
        return False
    if np.any(seg[gap_idx] != bg):
        return False
    return int(gap_idx[-1]) - int(gap_idx[0]) + 1 == int(gap_idx.size)


def _frame_from_component(grid: np.ndarray, comp: list[tuple[int, int]], bg: int):
    colors = {int(grid[r, c]) for r, c in comp}
    if len(colors) != 1:
        return None

    color = next(iter(colors))
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    if any(not (r in (r0, r1) or c in (c0, c1)) for r, c in comp):
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return None

    corners = [grid[r0, c0], grid[r0, c1], grid[r1, c0], grid[r1, c1]]
    if any(int(v) != color for v in corners):
        return None

    segments = {
        "top": grid[r0, c0 + 1 : c1],
        "bottom": grid[r1, c0 + 1 : c1],
        "left": grid[r0 + 1 : r1, c0],
        "right": grid[r0 + 1 : r1, c1],
    }

    missing = [side for side, seg in segments.items() if _single_contiguous_gap(seg, color, bg)]
    if len(missing) != 1:
        return None

    direction = missing[0]
    for side, seg in segments.items():
        if side == direction:
            if np.any((seg != color) & (seg != bg)):
                return None
            continue
        if not np.all(seg == color):
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "color": color,
        "direction": direction,
    }


def _source_segment(grid: np.ndarray, frame: dict, bg: int) -> np.ndarray:
    del bg
    r0, c0, r1, c1 = frame["bbox"]
    direction = frame["direction"]
    if direction == "top":
        return grid[0, c0 + 1 : c1]
    if direction == "bottom":
        return grid[-1, c0 + 1 : c1]
    if direction == "left":
        return grid[r0 + 1 : r1, 0]
    return grid[r0 + 1 : r1, -1]


def _fill_mask_from_source(source: np.ndarray, direction: str, inner_h: int, inner_w: int, bg: int) -> np.ndarray:
    if direction in ("top", "bottom"):
        row_mask = (source != bg).reshape(1, inner_w)
        return np.tile(row_mask, (inner_h, 1))
    col_mask = (source != bg).reshape(inner_h, 1)
    return np.tile(col_mask, (1, inner_w))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame from the edge stripe indicated by its missing side.
    The aligned edge segment supplies a binary pattern (bg vs non-bg); that
    pattern is repeated across the frame interior and recolored to the frame's
    border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    if h < 5 or w < 5:
        return g.copy()

    bg = _most_common_color(g)
    out = g.copy()

    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = g[1:-1, 1:-1] != bg

    frames = []
    for comp in _components(inner_mask):
        frame = _frame_from_component(g, comp, bg)
        if frame is not None:
            frames.append(frame)

    for frame in sorted(frames, key=lambda item: item["bbox"]):
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        source = _source_segment(g, frame, bg)
        mask = _fill_mask_from_source(source, frame["direction"], inner_h, inner_w, bg)
        out[r0 + 1 : r1, c0 + 1 : c1] = np.where(mask, frame["color"], bg)

    return out


def _partition_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [b - a for a, b in zip(bounds, bounds[1:], strict=False)]


def _make_stripe(length: int, palette: list[int]) -> np.ndarray:
    min_runs = min(4, length)
    max_runs = min(8, length)

    for _ in range(100):
        run_count = random.randint(min_runs, max_runs)
        run_lengths = _partition_positive(length, run_count)
        run_colors: list[int] = []
        used_bg = False
        used_fg = False

        for i in range(run_count):
            choices = [BG_COLOR] + palette
            if run_colors:
                choices = [c for c in choices if c != run_colors[-1]]

            runs_left = run_count - i
            if not used_bg and runs_left == 1:
                color = BG_COLOR
            elif not used_fg and runs_left == 1:
                color = random.choice(palette)
            else:
                weights = [3 if c == BG_COLOR else 1 for c in choices]
                color = random.choices(choices, weights=weights, k=1)[0]

            used_bg |= color == BG_COLOR
            used_fg |= color != BG_COLOR
            run_colors.append(color)

        stripe = np.empty(length, dtype=int)
        pos = 0
        for run_length, color in zip(run_lengths, run_colors, strict=False):
            stripe[pos : pos + run_length] = color
            pos += run_length

        if used_bg and used_fg and np.any(stripe[:-1] != stripe[1:]):
            return stripe

    raise RuntimeError("failed to build stripe")


def _mixed_source(seg: np.ndarray) -> bool:
    return bool(np.any(seg == BG_COLOR) and np.any(seg != BG_COLOR))


def _source_for_bbox(
    top: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    direction: str,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    if direction == "top":
        return top[c0 + 1 : c1]
    if direction == "bottom":
        return bottom[c0 + 1 : c1]
    if direction == "left":
        return left[r0 + 1 : r1]
    return right[r0 + 1 : r1]


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = frame["color"]
    direction = frame["direction"]
    gap_start = frame["gap_start"]
    gap_len = frame["gap_len"]

    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if direction == "top":
        start = c0 + 1 + gap_start
        grid[r0, start : start + gap_len] = BG_COLOR
    elif direction == "bottom":
        start = c0 + 1 + gap_start
        grid[r1, start : start + gap_len] = BG_COLOR
    elif direction == "left":
        start = r0 + 1 + gap_start
        grid[start : start + gap_len, c0] = BG_COLOR
    else:
        start = r0 + 1 + gap_start
        grid[start : start + gap_len, c1] = BG_COLOR


def _apply_frame_fill(grid: np.ndarray, out: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    source = _source_segment(grid, frame, BG_COLOR)
    mask = _fill_mask_from_source(source, frame["direction"], inner_h, inner_w, BG_COLOR)
    out[r0 + 1 : r1, c0 + 1 : c1] = np.where(mask, frame["color"], BG_COLOR)


def _bg_is_unique_mode(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    return winners.size == 1 and int(winners[0]) == BG_COLOR


def _place_frames(
    h: int,
    w: int,
    top: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    directions: list[str],
    frame_colors: list[int],
) -> list[dict] | None:
    frames: list[dict] = []
    max_inner_h = min(INNER_H_RANGE[1], h - 6)
    max_inner_w = min(INNER_W_RANGE[1], w - 6)
    if max_inner_h < INNER_H_RANGE[0] or max_inner_w < INNER_W_RANGE[0]:
        return None

    for direction, color in zip(directions, frame_colors, strict=False):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            inner_h = random.randint(INNER_H_RANGE[0], max_inner_h)
            inner_w = random.randint(INNER_W_RANGE[0], max_inner_w)
            outer_h = inner_h + 2
            outer_w = inner_w + 2

            max_r0 = h - outer_h - FRAME_MARGIN_FROM_EDGE
            max_c0 = w - outer_w - FRAME_MARGIN_FROM_EDGE
            if max_r0 < FRAME_MARGIN_FROM_EDGE or max_c0 < FRAME_MARGIN_FROM_EDGE:
                return None

            r0 = random.randint(FRAME_MARGIN_FROM_EDGE, max_r0)
            c0 = random.randint(FRAME_MARGIN_FROM_EDGE, max_c0)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)

            if any(not _boxes_separated(bbox, other["bbox"], FRAME_GAP) for other in frames):
                continue

            source = _source_for_bbox(top, bottom, left, right, direction, bbox)
            if not _mixed_source(source):
                continue

            gap_size = inner_w if direction in ("top", "bottom") else inner_h
            gap_len = random.randint(1, gap_size)
            gap_start = random.randint(0, gap_size - gap_len)

            frames.append(
                {
                    "bbox": bbox,
                    "color": color,
                    "direction": direction,
                    "gap_start": gap_start,
                    "gap_len": gap_len,
                }
            )
            placed = True
            break

        if not placed:
            return None

    return frames


def _generate_scene() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    min_side = max(GRID_MIN, 12 + 2 * (frame_count - 1))
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)

    stripe_palette = random.sample(NON_BG_COLORS, random.randint(2, 3))
    frame_palette = [c for c in NON_BG_COLORS if c not in stripe_palette]
    frame_colors = random.sample(frame_palette, frame_count)
    directions = [random.choice(DIRECTIONS) for _ in range(frame_count)]

    top = _make_stripe(w, random.sample(stripe_palette, random.randint(1, len(stripe_palette))))
    bottom = _make_stripe(w, random.sample(stripe_palette, random.randint(1, len(stripe_palette))))
    left = _make_stripe(h, random.sample(stripe_palette, random.randint(1, len(stripe_palette))))
    right = _make_stripe(h, random.sample(stripe_palette, random.randint(1, len(stripe_palette))))

    frames = _place_frames(h, w, top, bottom, left, right, directions, frame_colors)
    if frames is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[0, :] = top
    grid[-1, :] = bottom
    grid[:, 0] = left
    grid[:, -1] = right

    for frame in frames:
        _draw_frame(grid, frame)

    if not _bg_is_unique_mode(grid):
        return None

    expected = grid.copy()
    for frame in frames:
        _apply_frame_fill(grid, expected, frame)

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
