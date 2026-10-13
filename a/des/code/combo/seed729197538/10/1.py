from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
ANCHOR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rect_frame(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return None
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if len(comp) != len(expected):
        return None
    if set(comp) != expected:
        return None
    return (r0, c0, r1, c1)


def _find_frames(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    h, w = grid.shape
    frame_mask = np.zeros((h, w), dtype=bool)
    frames: list[tuple[int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        for comp in _components(grid == color):
            bbox = _is_rect_frame(comp)
            if bbox is None:
                continue
            frames.append(bbox)
            for r, c in comp:
                frame_mask[r, c] = True
    frames.sort()
    return frames, frame_mask


def _anchor_pairs(grid: np.ndarray, frame_mask: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    positions: dict[int, list[tuple[int, int]]] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or frame_mask[r, c]:
                continue
            positions.setdefault(color, []).append((r, c))

    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    used: set[int] = set()
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or frame_mask[r, c] or color in used:
                continue
            pts = positions.get(color, [])
            if len(pts) != 2:
                continue
            pts = sorted(pts)
            pairs.append((color, pts[0], pts[1]))
            used.add(color)
    return pairs


def _path_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    tr, tc = end
    points: list[tuple[int, int]] = []

    if sc <= tc:
        for c in range(sc, tc + 1):
            points.append((sr, c))
    else:
        for c in range(sc, tc - 1, -1):
            points.append((sr, c))

    if sr < tr:
        for r in range(sr + 1, tr + 1):
            points.append((r, tc))
    elif sr > tr:
        for r in range(sr - 1, tr - 1, -1):
            points.append((r, tc))

    return points


def _inside_bbox(pt: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = pt
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _inside_interior(pt: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = pt
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _traverses_frame(path: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    in_box = [_inside_bbox(pt, bbox) for pt in path]
    if not any(in_box):
        return False

    i = 0
    n = len(path)
    while i < n:
        if not in_box[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and in_box[j + 1]:
            j += 1
        enters = i > 0 and not in_box[i - 1]
        exits = j + 1 < n and not in_box[j + 1]
        has_interior = any(_inside_interior(path[k], bbox) for k in range(i, j + 1))
        if enters and exits and has_interior:
            return True
        i = j + 1
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region whose implied anchor-to-anchor path traverses it.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames, frame_mask = _find_frames(g)
    for color, start, end in _anchor_pairs(g, frame_mask):
        path = _path_points(start, end)
        for r0, c0, r1, c1 in frames:
            if _traverses_frame(path, (r0, c0, r1, c1)):
                out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _boxes_too_close(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_rectangles(
    rng: random.Random,
    height: int,
    width: int,
    count: int,
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    tries = 0
    while len(rects) < count and tries < 3000:
        tries += 1
        rect_h = rng.randint(4, min(9, height - 2))
        rect_w = rng.randint(4, min(9, width - 2))
        max_r0 = height - rect_h - 1
        max_c0 = width - rect_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None
        r0 = rng.randint(1, max_r0)
        c0 = rng.randint(1, max_c0)
        cand = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if any(_boxes_too_close(cand, other, gap=2) for other in rects):
            continue
        rects.append(cand)
    if len(rects) != count:
        return None
    rects.sort()
    return rects


def _anchor_candidates(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    ok = np.ones((height, width), dtype=bool)
    for r0, c0, r1, c1 in rects:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(height - 1, r1 + 1)
        cc1 = min(width - 1, c1 + 1)
        ok[rr0 : rr1 + 1, cc0 : cc1 + 1] = False
    return [(r, c) for r in range(height) for c in range(width) if ok[r, c]]


def _crossed_rect_indices(
    rects: list[tuple[int, int, int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> set[int]:
    path = _path_points(start, end)
    crossed: set[int] = set()
    for idx, bbox in enumerate(rects):
        if _traverses_frame(path, bbox):
            crossed.add(idx)
    return crossed


def _far_from_used(pt: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    return all(max(abs(pt[0] - ur), abs(pt[1] - uc)) > 1 for ur, uc in used)


def _sample_pair(
    rng: random.Random,
    candidates: list[tuple[int, int]],
    used: set[tuple[int, int]],
    rects: list[tuple[int, int, int, int]],
    mode: str,
    target_rect: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int], set[int]] | None:
    for _ in range(2500):
        a, b = rng.sample(candidates, 2)
        if a[0] == b[0] or a[1] == b[1]:
            continue
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 3:
            continue
        if not _far_from_used(a, used) or not _far_from_used(b, used):
            continue
        start, end = sorted((a, b))
        crossed = _crossed_rect_indices(rects, start, end)
        if len(crossed) > 3:
            continue
        if mode == "zero" and crossed:
            continue
        if mode == "cross" and not crossed:
            continue
        if mode == "cross_target" and (target_rect is None or target_rect not in crossed):
            continue
        return start, end, crossed
    return None


def _draw_rectangles(grid: np.ndarray, rects: list[tuple[int, int, int, int]]) -> None:
    for r0, c0, r1, c1 in rects:
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    rect_count = rng.randint(2, 6)
    rects = _place_rectangles(rng, height, width, rect_count)
    if rects is None:
        return None

    candidates = _anchor_candidates(height, width, rects)
    max_pairs = min(len(ANCHOR_COLORS), len(candidates) // 2)
    if max_pairs < 2:
        return None
    pair_count = rng.randint(2, min(5, max_pairs))

    need_overlap = pair_count >= 3 and rng.random() < 0.45
    need_zero = rng.random() < 0.55

    specs: list[tuple[str, int | None]] = []
    if need_overlap:
        target = rng.randrange(rect_count)
        specs.extend([("cross_target", target), ("cross_target", target)])
    if need_zero and len(specs) < pair_count:
        specs.append(("zero", None))
    while len(specs) < pair_count:
        if not any(mode.startswith("cross") for mode, _ in specs):
            specs.append(("cross", None))
        else:
            specs.append((rng.choice(("cross", "zero", "cross")), None))
    rng.shuffle(specs)

    used: set[tuple[int, int]] = set()
    pairs: list[tuple[tuple[int, int], tuple[int, int], set[int]]] = []
    for mode, target in specs:
        sampled = _sample_pair(rng, candidates, used, rects, mode, target)
        if sampled is None:
            return None
        start, end, crossed = sampled
        pairs.append((start, end, crossed))
        used.add(start)
        used.add(end)

    cross_counter = Counter(idx for _start, _end, crossed in pairs for idx in crossed)
    filled = set(cross_counter)
    if not filled:
        return None
    if rect_count >= 3 and len(filled) == rect_count:
        return None
    if need_overlap and not any(v >= 2 for v in cross_counter.values()):
        return None
    if need_zero and not any(not crossed for _start, _end, crossed in pairs):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    _draw_rectangles(grid, rects)

    colors = list(rng.sample(ANCHOR_COLORS, len(pairs)))
    rng.shuffle(colors)
    for color, (start, end, _crossed) in zip(colors, pairs):
        grid[start] = color
        grid[end] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    frame_cells = sum((r1 - r0 - 1) * (c1 - c0 - 1) for r0, c0, r1, c1 in rects)
    changed = int(np.count_nonzero((out != grid) & (grid == BLACK)))
    if changed <= 0 or changed >= frame_cells:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
