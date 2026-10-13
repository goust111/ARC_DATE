import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

FRAME_COLORS = (BLUE, RED, GREEN, YELLOW)
PATTERN_COLORS = (5, 6, 7, 8, 9)
MAX_GEN_RETRIES = 400

_GEN_CALL_COUNT = 0


def _bbox_of(points):
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


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


def _is_rectangular_frame(mask: np.ndarray, points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox_of(points)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return False
    frame = np.zeros_like(mask, dtype=bool)
    frame[r0, c0 : c1 + 1] = True
    frame[r1, c0 : c1 + 1] = True
    frame[r0 : r1 + 1, c0] = True
    frame[r0 : r1 + 1, c1] = True
    return np.array_equal(mask, frame)


def _find_frames(grid: np.ndarray) -> dict[tuple[int, int, int, int], int]:
    frames: dict[tuple[int, int, int, int], int] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        color_mask = grid == color
        for comp in _components(color_mask):
            mask = np.zeros_like(color_mask, dtype=bool)
            for r, c in comp:
                mask[r, c] = True
            if _is_rectangular_frame(mask, comp):
                frames[_bbox_of(comp)] = color
    return frames


def _inset_bbox(bbox: tuple[int, int, int, int], delta: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + delta, c0 + delta, r1 - delta, c1 - delta)


def _pattern_value(pattern: np.ndarray, r: int, c: int, mode: int) -> int:
    ph, pw = pattern.shape
    if mode == BLUE:
        rr = min(max(r, 0), ph - 1)
        cc = c % pw
        return int(pattern[rr, cc])
    if mode == RED:
        rr = r % ph
        cc = min(max(c, 0), pw - 1)
        return int(pattern[rr, cc])
    if mode == GREEN:
        return int(pattern[r % ph, c % pw])
    counts = Counter(int(v) for v in pattern.flat)
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    winners.sort()
    return winners[0]


def _fill_gap(out: np.ndarray, outer_bbox, inner_bbox, pattern: np.ndarray, mode: int) -> None:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    pr0, pc0 = ir0 + 1, ic0 + 1
    for r in range(or0 + 1, or1):
        for c in range(oc0 + 1, oc1):
            if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                continue
            if out[r, c] != BLACK:
                continue
            rel_r = r - pr0
            rel_c = c - pc0
            out[r, c] = _pattern_value(pattern, rel_r, rel_c, mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect nested rectangular frames and fill the one-cell black gaps between
    them using the original innermost pattern. The outer frame color selects
    one of four fill modes: horizontal repeat, vertical repeat, 2D repeat, or
    solid majority color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    children: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    parents: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    for bbox in frames:
        child = _inset_bbox(bbox, 2)
        if child in frames:
            children[bbox] = child
            parents[child] = bbox

    roots = [bbox for bbox in frames if bbox not in parents and bbox in children]
    assert roots, "expected at least one nested frame structure"

    out = g.copy()
    for root in roots:
        chain = [root]
        while chain[-1] in children:
            chain.append(children[chain[-1]])
        assert 2 <= len(chain) <= 3, "each structure must have 2 or 3 frame layers"

        inner = chain[-1]
        ir0, ic0, ir1, ic1 = inner
        pattern = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
        assert pattern.shape[0] in (2, 3) and pattern.shape[1] in (2, 3)
        assert np.all(pattern != BLACK), "innermost pattern must be non-black"

        for outer_bbox, inner_bbox in zip(chain[:-1], chain[1:]):
            mode = frames[outer_bbox]
            assert mode in FRAME_COLORS, "frame colors must be blue/red/green/yellow"
            _fill_gap(out, outer_bbox, inner_bbox, pattern, mode)

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _random_pattern(rng: random.Random, h: int, w: int) -> np.ndarray:
    palette = list(PATTERN_COLORS)
    rng.shuffle(palette)
    n_colors = rng.choice((1, 2, 3))
    colors = palette[:n_colors]

    total = h * w
    if n_colors == 1:
        arr = np.full((h, w), colors[0], dtype=int)
        return arr

    majority = colors[0]
    arr = np.full((h, w), majority, dtype=int)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)

    if n_colors == 2:
        minority_count = rng.randint(1, max(1, total // 2 - 1))
        for r, c in cells[:minority_count]:
            arr[r, c] = colors[1]
        return arr

    while True:
        count_b = rng.randint(1, total - 2)
        count_c = rng.randint(1, total - 1 - count_b)
        majority_count = total - count_b - count_c
        if majority_count > max(count_b, count_c):
            break

    idx = 0
    for _ in range(count_b):
        r, c = cells[idx]
        idx += 1
        arr[r, c] = colors[1]
    for _ in range(count_c):
        r, c = cells[idx]
        idx += 1
        arr[r, c] = colors[2]
    return arr


def _object_outer_size(pattern_h: int, pattern_w: int, layers: int) -> tuple[int, int]:
    return pattern_h + 4 * layers - 2, pattern_w + 4 * layers - 2


def _placeable(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    H, W = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _paint_object(
    grid: np.ndarray,
    top: int,
    left: int,
    layers: int,
    frame_colors: list[int],
    pattern: np.ndarray,
) -> None:
    ph, pw = pattern.shape
    inner_r0 = top + 2 * (layers - 1)
    inner_c0 = left + 2 * (layers - 1)
    inner_bbox = (inner_r0, inner_c0, inner_r0 + ph + 1, inner_c0 + pw + 1)
    _draw_frame(grid, inner_bbox, frame_colors[-1])
    grid[inner_r0 + 1 : inner_r0 + 1 + ph, inner_c0 + 1 : inner_c0 + 1 + pw] = pattern

    current = inner_bbox
    for i in range(layers - 2, -1, -1):
        current = (current[0] - 2, current[1] - 2, current[2] + 2, current[3] + 2)
        _draw_frame(grid, current, frame_colors[i])


def _sample_frame_colors(rng: random.Random, layers: int, preferred_gap: int) -> list[int]:
    chosen = [preferred_gap]
    remaining = [c for c in FRAME_COLORS if c != preferred_gap]
    rng.shuffle(remaining)
    while len(chosen) < layers - 1:
        chosen.append(remaining.pop())
    innermost = remaining.pop() if remaining else preferred_gap
    chosen.append(innermost)
    return chosen


def _gen_attempt(rng: random.Random, force_gap: int, force_layers: int) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    max_objects = 4
    n_objects = rng.randint(1, max_objects)
    planned_layers = [force_layers]
    while len(planned_layers) < n_objects:
        planned_layers.append(rng.choice((2, 3)))
    rng.shuffle(planned_layers)

    forced_index = rng.randrange(n_objects)
    gap_choices = [rng.choice(FRAME_COLORS) for _ in range(n_objects)]
    gap_choices[forced_index] = force_gap

    objects = []
    for i in range(n_objects):
        layers = planned_layers[i]
        ph = rng.choice((2, 3))
        pw = rng.choice((2, 3))
        outer_h, outer_w = _object_outer_size(ph, pw, layers)
        if outer_h + 2 > h or outer_w + 2 > w:
            return None
        pattern = _random_pattern(rng, ph, pw)
        frame_colors = _sample_frame_colors(rng, layers, gap_choices[i])
        objects.append((outer_h, outer_w, layers, frame_colors, pattern))

    rng.shuffle(objects)
    for outer_h, outer_w, layers, frame_colors, pattern in objects:
        placements = []
        for r0 in range(h - outer_h + 1):
            for c0 in range(w - outer_w + 1):
                if _placeable(occupied, r0, c0, outer_h, outer_w, pad=1):
                    placements.append((r0, c0))
        if not placements:
            return None
        top, left = rng.choice(placements)
        _paint_object(grid, top, left, layers, frame_colors, pattern)
        occupied[top : top + outer_h, left : left + outer_w] = True

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    rng = random.Random()
    preferred_gap = FRAME_COLORS[_GEN_CALL_COUNT % len(FRAME_COLORS)]
    preferred_layers = 2 if (_GEN_CALL_COUNT // len(FRAME_COLORS)) % 2 == 0 else 3
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng, preferred_gap, preferred_layers)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to place valid nested structures")
