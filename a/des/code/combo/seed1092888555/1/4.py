import random
from collections import Counter

import numpy as np


BG_COLOR = 3
FRAME_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
FRAME_COUNT_RANGE = (2, 5)
FRAME_SIDE_RANGE = (4, 12)
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in frame_cells(bbox):
        mask[r, c] = True
    return mask


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
            out.append(comp)
    return out


def visible_bboxes(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        pts = np.argwhere(grid == color)
        assert pts.size > 0
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bboxes[color] = (int(r0), int(c0), int(r1), int(c1))
    return bboxes


def infer_order(
    grid: np.ndarray,
    bboxes: dict[int, tuple[int, int, int, int]],
) -> list[int]:
    colors = sorted(bboxes)
    frames = {color: frame_cells(bboxes[color]) for color in colors}
    edges: set[tuple[int, int]] = set()
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            a_over_b = 0
            b_over_a = 0
            for r, c in frames[a] & frames[b]:
                vis = int(grid[r, c])
                if vis == a:
                    a_over_b += 1
                elif vis == b:
                    b_over_a += 1
            if a_over_b and b_over_a:
                raise AssertionError("contradictory overlap order")
            if a_over_b:
                edges.add((b, a))
            elif b_over_a:
                edges.add((a, b))

    indeg = {color: 0 for color in colors}
    succ = {color: set() for color in colors}
    for lo, hi in edges:
        if hi not in succ[lo]:
            succ[lo].add(hi)
            indeg[hi] += 1

    order: list[int] = []
    ready = sorted(color for color in colors if indeg[color] == 0)
    while ready:
        if len(ready) != 1:
            raise AssertionError("layer order is not uniquely determined")
        color = ready.pop()
        order.append(color)
        for nxt in sorted(succ[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort()

    assert len(order) == len(colors)
    return order


def count_holes(
    shape: tuple[int, int],
    bboxes: dict[int, tuple[int, int, int, int]],
) -> dict[int, int]:
    wall_mask = np.zeros(shape, dtype=bool)
    for bbox in bboxes.values():
        wall_mask |= frame_mask(shape, bbox)

    counts: dict[int, int] = {}
    for color, (r0, c0, r1, c1) in bboxes.items():
        if r1 - r0 <= 1 or c1 - c0 <= 1:
            counts[color] = 0
            continue
        interior = ~wall_mask[r0 + 1 : r1, c0 + 1 : c1]
        counts[color] = len(components(interior))
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the stacking order of overlapping rectangle frames and summarize each
    layer by a row whose width equals the number of enclosed background holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bboxes = visible_bboxes(g, bg)
    order = infer_order(g, bboxes)
    holes = count_holes(g.shape, bboxes)

    widths = [max(holes[color], 1) for color in order]
    out = np.full((len(order), max(widths)), bg, dtype=int)
    for r, color in enumerate(order):
        out[r, : widths[r]] = color
    return out


def render_grid(
    shape: tuple[int, int],
    ordered_colors: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    bg: int,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color in ordered_colors:
        for r, c in frame_cells(bboxes[color]):
            grid[r, c] = color
    return grid


def side_visibility_ok(
    grid: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    top = np.any(grid[r0, c0 : c1 + 1] == color)
    bottom = np.any(grid[r1, c0 : c1 + 1] == color)
    left = np.any(grid[r0 : r1 + 1, c0] == color)
    right = np.any(grid[r0 : r1 + 1, c1] == color)
    return bool(top and bottom and left and right)


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return bool(frame_cells(a) & frame_cells(b))


def random_bbox(rng: random.Random, h: int, w: int, large: bool) -> tuple[int, int, int, int]:
    side_lo, side_hi = FRAME_SIDE_RANGE
    if large:
        fh = rng.randint(max(6, side_lo + 2), min(h - 2, side_hi + 2))
        fw = rng.randint(max(6, side_lo + 2), min(w - 2, side_hi + 2))
    else:
        fh = rng.randint(side_lo, min(h - 2, side_hi))
        fw = rng.randint(side_lo, min(w - 2, side_hi))
    r0 = rng.randint(1, h - fh - 1)
    c0 = rng.randint(1, w - fw - 1)
    return (r0, c0, r0 + fh - 1, c0 + fw - 1)


def sample_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n = rng.randint(*FRAME_COUNT_RANGE)
    colors = list(rng.sample(FRAME_COLORS, n))

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    placed: list[tuple[int, int, int, int]] = []

    for idx, color in enumerate(colors):
        for _ in range(120):
            bbox = random_bbox(rng, h, w, large=(idx == 0))
            if bbox in placed:
                continue
            if idx > 0 and not overlaps(bbox, placed[-1]):
                continue
            bboxes[color] = bbox
            placed.append(bbox)
            break
        else:
            return None

    grid = render_grid((h, w), colors, bboxes, BG_COLOR)
    if get_bg(grid) != BG_COLOR:
        return None

    visible = Counter(int(v) for v in grid.flat if int(v) != BG_COLOR)
    if any(count < 6 for count in visible.values()):
        return None
    if any(not side_visibility_ok(grid, color, bboxes[color]) for color in colors):
        return None

    try:
        inferred_bboxes = visible_bboxes(grid, BG_COLOR)
        inferred_order = infer_order(grid, inferred_bboxes)
    except AssertionError:
        return None
    if inferred_order != colors:
        return None

    holes = count_holes(grid.shape, bboxes)
    if max(holes.values()) < 2 and rng.random() < 0.65:
        return None

    out = main(grid)
    if out.shape[0] != n:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = sample_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
