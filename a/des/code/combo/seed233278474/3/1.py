import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9


LEGEND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
GEN_RETRIES = 300
LEGEND_COUNT_RANGE = (3, 6)
LEGEND_SIZE_CHOICES = [2, 3]
LEGEND_GAP_CHOICES = [1, 2]
TOP_MARGIN_CHOICES = [0, 1]
LEGEND_TO_FENCE_GAP = (2, 4)
GRID_H_MAX = 28
GRID_W_MAX = 30
FRAME_RETRIES = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _is_monochrome(g: np.ndarray, pts: list[tuple[int, int]]) -> bool:
    vals = {int(g[r, c]) for r, c in pts}
    return len(vals) == 1


def _is_solid_rectangle(pts: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h >= 2 and w >= 2 and len(pts) == h * w


def _is_rectangular_frame(pts: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(pts) != 2 * (h + w) - 4:
        return False
    for r, c in pts:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _legend_colors(g: np.ndarray) -> list[int]:
    blocks = []
    for pts in _components(g != BLACK):
        if not _is_monochrome(g, pts) or not _is_solid_rectangle(pts):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        blocks.append((r0, c0, r1, c1, int(g[r0, c0])))
    assert blocks, "expected at least one legend block"
    top_row = min(r0 for r0, _, _, _, _ in blocks)
    top_blocks = [b for b in blocks if b[0] == top_row]
    top_blocks.sort(key=lambda item: item[1])
    return [color for _, _, _, _, color in top_blocks]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey fence by counting the maroon markers it encloses and using
    that count to index the top legend from left to right.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend = _legend_colors(g)

    for pts in _components(g == GREY):
        if not _is_rectangular_frame(pts):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(interior == MAROON))
        assert 1 <= marker_count <= len(legend), "marker count must map into the legend"
        new_color = legend[marker_count - 1]
        for r, c in pts:
            out[r, c] = new_color
        cleared = out[r0 + 1 : r1, c0 + 1 : c1]
        cleared[cleared == MAROON] = BLACK

    return out


def _legend_layout(rng: random.Random):
    size = rng.choice(LEGEND_SIZE_CHOICES)
    count = rng.randint(*LEGEND_COUNT_RANGE)
    gaps = [rng.choice(LEGEND_GAP_CHOICES) for _ in range(count - 1)]
    used_width = count * size + sum(gaps)
    top = rng.choice(TOP_MARGIN_CHOICES)
    gap_below = rng.randint(*LEGEND_TO_FENCE_GAP)
    min_h = top + size + gap_below + 8
    min_w = used_width + 2
    if min_h > GRID_H_MAX or min_w > GRID_W_MAX:
        return None
    height = rng.randint(max(16, min_h), GRID_H_MAX)
    width = rng.randint(max(16, min_w), GRID_W_MAX)
    start_c = rng.randint(1, width - used_width - 1)
    colors = rng.sample(LEGEND_COLOR_POOL, count)
    return {
        "size": size,
        "count": count,
        "gaps": gaps,
        "top": top,
        "gap_below": gap_below,
        "height": height,
        "width": width,
        "start_c": start_c,
        "colors": colors,
    }


def _max_marker_slots(inner_h: int, inner_w: int) -> int:
    return ((inner_h + 1) // 2) * ((inner_w + 1) // 2)


def _sample_frame_dims(rng: random.Random, marker_count: int) -> tuple[int, int]:
    for _ in range(100):
        frame_h = rng.randint(4, 9)
        frame_w = rng.randint(4, 12)
        if _max_marker_slots(frame_h - 2, frame_w - 2) >= marker_count:
            return frame_h, frame_w
    need = marker_count
    side = 4
    while _max_marker_slots(side - 2, side - 2) < need:
        side += 1
    return side, side


def _non_touching_marker_positions(
    rng: random.Random, r0: int, c0: int, r1: int, c1: int, count: int
) -> list[tuple[int, int]]:
    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))
    classes = []
    for row_parity in (0, 1):
        for col_parity in (0, 1):
            pts = [
                (r, c)
                for i, r in enumerate(rows)
                for j, c in enumerate(cols)
                if i % 2 == row_parity and j % 2 == col_parity
            ]
            if len(pts) >= count:
                classes.append(pts)
    assert classes, "frame interior must have enough isolated slots"
    return rng.sample(rng.choice(classes), count)


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = blocked.shape
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(hh, r0 + h + 1)
    pc1 = min(ww, c0 + w + 1)
    return not blocked[pr0:pr1, pc0:pc1].any()


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    hh, ww = blocked.shape
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(hh, r0 + h + 1)
    pc1 = min(ww, c0 + w + 1)
    blocked[pr0:pr1, pc0:pc1] = True


def _draw_frame(g: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _choose_marker_counts(rng: random.Random, legend_count: int, frame_count: int) -> list[int]:
    counts = list(range(1, legend_count + 1))
    rng.shuffle(counts)
    picked = counts[: min(frame_count, legend_count)]
    while len(picked) < frame_count:
        picked.append(rng.randint(1, legend_count))
    rng.shuffle(picked)
    return picked


def _frame_count(rng: random.Random, usable_h: int, usable_w: int) -> int:
    area = usable_h * usable_w
    max_frames = min(4, max(1, area // 45))
    choices = list(range(1, max_frames + 1))
    weights = [1, 3, 3, 2][: max_frames]
    return rng.choices(choices, weights=weights, k=1)[0]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _legend_layout(rng)
    if layout is None:
        return None

    h = layout["height"]
    w = layout["width"]
    g = np.zeros((h, w), dtype=int)

    r = layout["top"]
    c = layout["start_c"]
    legend_positions = []
    for idx, color in enumerate(layout["colors"]):
        legend_positions.append(c)
        size = layout["size"]
        g[r : r + size, c : c + size] = color
        if idx < layout["count"] - 1:
            c += size + layout["gaps"][idx]

    placement_top = layout["top"] + layout["size"] + layout["gap_below"]
    usable_h = h - placement_top - 1
    usable_w = w - 2
    if usable_h < 4 or usable_w < 4:
        return None

    blocked = np.zeros_like(g, dtype=bool)
    blocked[:placement_top, :] = True

    frame_count = _frame_count(rng, usable_h, usable_w)
    marker_counts = _choose_marker_counts(rng, layout["count"], frame_count)

    placed = 0
    for marker_count in marker_counts:
        frame_h, frame_w = _sample_frame_dims(rng, marker_count)
        max_r0 = h - frame_h - 1
        max_c0 = w - frame_w - 1
        if max_r0 < placement_top or max_c0 < 1:
            return None
        ok = False
        for _ in range(FRAME_RETRIES):
            r0 = rng.randint(placement_top, max_r0)
            c0 = rng.randint(1, max_c0)
            if not _can_place(blocked, r0, c0, frame_h, frame_w):
                continue
            _draw_frame(g, r0, c0, frame_h, frame_w, GREY)
            for mr, mc in _non_touching_marker_positions(
                rng, r0, c0, r0 + frame_h - 1, c0 + frame_w - 1, marker_count
            ):
                g[mr, mc] = MAROON
            _mark_blocked(blocked, r0, c0, frame_h, frame_w)
            placed += 1
            ok = True
            break
        if not ok:
            return None

    if placed == 0:
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
