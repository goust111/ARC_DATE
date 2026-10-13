from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
WALL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_SIDE_OPTIONS = [5, 7, 9]
GRID_H_RANGE = (28, 38)
GRID_W_RANGE = (28, 38)
WALL_COUNT_OPTIONS = [2, 3, 4]
EDGE_MARGIN = 4
MIN_FRAME_GAP = 6
MAX_ATTEMPTS = 500

PATTERN_TEMPLATES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rect_frame(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None
    if h % 2 == 0 or w % 2 == 0:
        return None
    if len(comp) != 2 * h + 2 * w - 4:
        return None
    patch = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        patch[r - r0, c - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(patch, expected):
        return None
    return r0, c0, r1, c1


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    colors = [int(v) for v in np.unique(grid) if int(v) not in {BG, GREY}]
    for color in colors:
        for comp in _components(grid == color):
            box = _is_rect_frame(comp)
            if box is not None:
                frames.append({"color": int(color), "bbox": box})
    assert len(frames) >= 2, "expected multiple colored walls"
    frame_colors = [frame["color"] for frame in frames]
    assert len(set(frame_colors)) == len(frame_colors), "wall colors must be unique"
    return frames


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _direction_from_to(src_box: tuple[int, int, int, int], dst_box: tuple[int, int, int, int]) -> str:
    sr, sc = _center(src_box)
    tr, tc = _center(dst_box)
    dr = tr - sr
    dc = tc - sc
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    if dc != 0:
        return "right" if dc > 0 else "left"
    assert dr != 0, "source and target walls must differ"
    return "down" if dr > 0 else "up"


def _point_rect_dist2(r: int, c: int, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    dr = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    dc = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return int(dr * dr + dc * dc)


def _nearest_side_candidates(
    src_box: tuple[int, int, int, int], dst_box: tuple[int, int, int, int]
) -> list[str]:
    r0, c0, r1, c1 = dst_box
    side_cells = {
        "up": [(r0, c) for c in range(c0, c1 + 1)],
        "down": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }
    scored: list[tuple[int, str]] = []
    for side, cells in side_cells.items():
        best = min(_point_rect_dist2(r, c, src_box) for r, c in cells)
        scored.append((int(best), side))
    best_dist = min(dist for dist, _side in scored)
    return [side for dist, side in scored if dist == best_dist]


def _nearest_side(src_box: tuple[int, int, int, int], dst_box: tuple[int, int, int, int]) -> str:
    return _direction_from_to(dst_box, src_box)


def _gap_cell(box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "up":
        return r0, (c0 + c1) // 2
    if side == "down":
        return r1, (c0 + c1) // 2
    if side == "left":
        return (r0 + r1) // 2, c0
    if side == "right":
        return (r0 + r1) // 2, c1
    raise AssertionError(f"unknown side {side}")


def _move_pattern(
    pattern: list[tuple[int, int]],
    src_box: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int]]:
    rows = [r for r, _ in pattern]
    cols = [c for _, c in pattern]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)
    r0, c0, r1, c1 = src_box
    if direction == "up":
        dr, dc = r0 - 1 - max_r, 0
    elif direction == "down":
        dr, dc = r1 + 1 - min_r, 0
    elif direction == "left":
        dr, dc = 0, c0 - 1 - max_c
    else:
        dr, dc = 0, c1 + 1 - min_c
    return [(r + dr, c + dc) for r, c in pattern]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each wall's key color, open a center gap on the referenced wall,
    then eject the grey interior pattern from the source wall toward the target wall.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames = _extract_frames(g)
    by_color = {frame["color"]: frame for frame in frames}

    out = g.copy()
    moves: list[tuple[list[tuple[int, int]], tuple[int, int, int, int], str]] = []
    gaps: list[tuple[int, int]] = []

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_local = np.argwhere(inside == GREY)
        assert grey_local.size > 0, "each wall must contain a grey pattern"
        key_local = np.argwhere((inside != BG) & (inside != GREY))
        assert key_local.shape[0] == 1, "each wall must contain exactly one key"

        key_r = int(key_local[0, 0] + r0 + 1)
        key_c = int(key_local[0, 1] + c0 + 1)
        key_color = int(g[key_r, key_c])
        assert key_color != frame["color"], "key color must reference another wall"
        assert key_color in by_color, "key color must match an existing wall"

        allowed = (inside == BG) | (inside == GREY)
        allowed[int(key_local[0, 0]), int(key_local[0, 1])] = True
        assert bool(np.all(allowed)), "interior may contain only grey pattern plus one key"

        pattern = [
            (int(rr + r0 + 1), int(cc + c0 + 1))
            for rr, cc in grey_local
        ]
        target = by_color[key_color]
        direction = _direction_from_to(frame["bbox"], target["bbox"])
        side = _nearest_side(frame["bbox"], target["bbox"])

        moves.append((pattern, frame["bbox"], direction))
        gaps.append(_gap_cell(target["bbox"], side))

        for rr, cc in pattern:
            out[rr, cc] = BG
        out[key_r, key_c] = BG

    for rr, cc in gaps:
        out[rr, cc] = BG

    for pattern, src_box, direction in moves:
        moved = _move_pattern(pattern, src_box, direction)
        for rr, cc in moved:
            assert 0 <= rr < h and 0 <= cc < w, "moved pattern must stay in bounds"
            assert out[rr, cc] == BG, "moved pattern must land on empty background"
        for rr, cc in moved:
            out[rr, cc] = GREY

    return out


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return max(row_gap, col_gap)


def _sample_boxes(
    h: int, w: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for bh, bw in sizes:
        placed = False
        for _ in range(200):
            r_min = EDGE_MARGIN
            r_max = h - EDGE_MARGIN - bh
            c_min = EDGE_MARGIN
            c_max = w - EDGE_MARGIN - bw
            if r_min > r_max or c_min > c_max:
                return None
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            if all(_bbox_gap(box, other) >= MIN_FRAME_GAP for other in boxes):
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return boxes


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    for _ in range(random.randint(0, 3)):
        out = np.rot90(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.astype(bool)


def _sample_pattern(inside_h: int, inside_w: int) -> tuple[np.ndarray, int, int]:
    templates = PATTERN_TEMPLATES[:]
    random.shuffle(templates)
    for base in templates:
        mask = _random_transform(base)
        ph, pw = mask.shape
        if ph <= inside_h and pw <= inside_w and int(mask.sum()) < inside_h * inside_w:
            r0 = random.randint(0, inside_h - ph)
            c0 = random.randint(0, inside_w - pw)
            return mask, r0, c0
    raise AssertionError("no pattern fits the chosen frame")


def _sample_derangement(n: int) -> list[int]:
    perm = list(range(n))
    while True:
        random.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm[:]


def _valid_mapping(boxes: list[tuple[int, int, int, int]], perm: list[int]) -> bool:
    directions: list[str] = []
    for i, j in enumerate(perm):
        src = boxes[i]
        dst = boxes[j]
        sr, sc = _center(src)
        tr, tc = _center(dst)
        if abs(tr - sr) == abs(tc - sc):
            return False
        directions.append(_direction_from_to(src, dst))
    return len(set(directions)) >= 2


def _place_horizontal_chain(
    h: int, w: int, sizes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int, int, int]], list[int]] | None:
    n = len(sizes)
    gaps = [random.randint(MIN_FRAME_GAP, MIN_FRAME_GAP + 3) for _ in range(n - 1)]
    max_h = max(bh for bh, _bw in sizes)
    total_w = sum(bw for _bh, bw in sizes) + sum(gaps)
    if h - EDGE_MARGIN - max_h < EDGE_MARGIN or w - EDGE_MARGIN - total_w < EDGE_MARGIN:
        return None
    row = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - max_h)
    col = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - total_w)
    boxes: list[tuple[int, int, int, int]] = []
    cur = col
    for i, (bh, bw) in enumerate(sizes):
        boxes.append((row, cur, row + bh - 1, cur + bw - 1))
        if i < n - 1:
            cur += bw + gaps[i]
    if n == 2:
        perm = [1, 0]
    else:
        perm = random.choice([[1, 2, 0], [2, 0, 1]])
    return boxes, perm


def _place_vertical_chain(
    h: int, w: int, sizes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int, int, int]], list[int]] | None:
    n = len(sizes)
    gaps = [random.randint(MIN_FRAME_GAP, MIN_FRAME_GAP + 3) for _ in range(n - 1)]
    max_w = max(bw for _bh, bw in sizes)
    total_h = sum(bh for bh, _bw in sizes) + sum(gaps)
    if h - EDGE_MARGIN - total_h < EDGE_MARGIN or w - EDGE_MARGIN - max_w < EDGE_MARGIN:
        return None
    row = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - total_h)
    col = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - max_w)
    boxes: list[tuple[int, int, int, int]] = []
    cur = row
    for i, (bh, bw) in enumerate(sizes):
        boxes.append((cur, col, cur + bh - 1, col + bw - 1))
        if i < n - 1:
            cur += bh + gaps[i]
    if n == 2:
        perm = [1, 0]
    else:
        perm = random.choice([[1, 2, 0], [2, 0, 1]])
    return boxes, perm


def _place_cycle_2x2(
    h: int, w: int, sizes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int, int, int]], list[int]] | None:
    assert len(sizes) == 4
    tl_size, tr_size, br_size, bl_size = sizes
    top_h = max(tl_size[0], tr_size[0])
    bottom_h = max(bl_size[0], br_size[0])
    left_w = max(tl_size[1], bl_size[1])
    right_w = max(tr_size[1], br_size[1])
    row_gap = random.randint(MIN_FRAME_GAP, MIN_FRAME_GAP + 3)
    col_gap = random.randint(MIN_FRAME_GAP, MIN_FRAME_GAP + 3)
    total_h = top_h + row_gap + bottom_h
    total_w = left_w + col_gap + right_w
    if h - EDGE_MARGIN - total_h < EDGE_MARGIN or w - EDGE_MARGIN - total_w < EDGE_MARGIN:
        return None
    top = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - total_h)
    left = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - total_w)

    tl = (top, left, top + tl_size[0] - 1, left + tl_size[1] - 1)
    tr = (top, left + left_w + col_gap, top + tr_size[0] - 1, left + left_w + col_gap + tr_size[1] - 1)
    bl = (top + top_h + row_gap, left, top + top_h + row_gap + bl_size[0] - 1, left + bl_size[1] - 1)
    br = (
        top + top_h + row_gap,
        left + left_w + col_gap,
        top + top_h + row_gap + br_size[0] - 1,
        left + left_w + col_gap + br_size[1] - 1,
    )
    boxes = [tl, tr, br, bl]
    perm = random.choice([[1, 2, 3, 0], [3, 0, 1, 2]])
    return boxes, perm


def _layout_boxes_and_mapping(
    h: int, w: int, sizes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int, int, int]], list[int]] | None:
    n = len(sizes)
    if n == 4:
        return _place_cycle_2x2(h, w, sizes)
    if random.random() < 0.5:
        return _place_horizontal_chain(h, w, sizes)
    return _place_vertical_chain(h, w, sizes)


def _build_input() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    wall_count = random.choice(WALL_COUNT_OPTIONS)
    wall_colors = random.sample(WALL_COLORS, wall_count)
    sizes = [(random.choice(FRAME_SIDE_OPTIONS), random.choice(FRAME_SIDE_OPTIONS)) for _ in range(wall_count)]
    layout = _layout_boxes_and_mapping(h, w, sizes)
    if layout is None:
        return None
    boxes, perm = layout
    if not _valid_mapping(boxes, perm):
        return None

    grid = np.full((h, w), BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(wall_colors, boxes):
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color

    for i, (color, box) in enumerate(zip(wall_colors, boxes)):
        r0, c0, r1, c1 = box
        inside_h = r1 - r0 - 1
        inside_w = c1 - c0 - 1
        pattern_mask, pr0, pc0 = _sample_pattern(inside_h, inside_w)
        ph, pw = pattern_mask.shape
        sub = grid[r0 + 1 : r1, c0 + 1 : c1]
        patch = sub[pr0 : pr0 + ph, pc0 : pc0 + pw]
        patch[pattern_mask] = GREY

        free = np.argwhere(sub == BG)
        if free.shape[0] == 0:
            return None
        kr, kc = map(int, random.choice(free.tolist()))
        key_color = wall_colors[perm[i]]
        assert key_color != color
        sub[kr, kc] = key_color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
