from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 0
GRID_MIN = 16
GRID_MAX = 30
GROUP_COUNT_RANGE = (2, 4)
SHAPE_SIDE_MIN = 3
SHAPE_SIDE_MAX = 6
PLACEMENT_PAD = 1
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(int(x) for x in pts[0])
    h, w = mask.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                q.append((int(nr), int(nc)))
    return len(seen) == int(mask.sum())


def _tight_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    r0 = min(rs)
    r1 = max(rs)
    c0 = min(cs)
    c1 = max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.tobytes()


def _hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(bg[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            region = []
            touches_border = False
            while q:
                cr, cc = q.popleft()
                region.append((int(cr), int(cc)))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(bg[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            if not touches_border:
                holes.append(region)
    return holes


def _extract_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != int(bg)):
        for pts in _components(grid == int(color)):
            mask, bbox = _tight_mask(pts)
            comps.append(
                {
                    "color": int(color),
                    "cells": [(int(r), int(c)) for r, c in pts],
                    "mask": mask,
                    "bbox": bbox,
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group equal shapes by translated mask, recolor the minority copy to the
    majority color, and fill the majority copies' holes with the minority color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    comps = _extract_components(g, bg)

    groups: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    for comp in comps:
        groups[_mask_key(comp["mask"])].append(comp)

    out = g.copy()
    for group in groups.values():
        if len(group) != 3:
            continue

        color_counts = Counter(int(comp["color"]) for comp in group)
        if sorted(color_counts.values()) != [1, 2]:
            continue

        majority_color = max(color_counts, key=lambda color: color_counts[color])
        minority_color = min(color_counts, key=lambda color: color_counts[color])

        for comp in group:
            if int(comp["color"]) == int(minority_color):
                for r, c in comp["cells"]:
                    out[int(r), int(c)] = int(majority_color)
            else:
                r0, c0, _r1, _c1 = comp["bbox"]
                for hole in _hole_regions(comp["mask"]):
                    for rr, cc in hole:
                        out[int(r0 + rr), int(c0 + cc)] = int(minority_color)
    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randrange(4))
    return np.array(out, dtype=bool, copy=True)


def _grow_shape(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    frontier = set(_neighbors4(start[0], start[1], height, width))
    while int(mask.sum()) < int(target_area) and frontier:
        choices = list(frontier)
        random.shuffle(choices)
        cell = None
        for cand in choices:
            if any(bool(mask[nr, nc]) for nr, nc in _neighbors4(cand[0], cand[1], height, width)):
                cell = cand
                break
        if cell is None:
            return None
        frontier.discard(cell)
        mask[cell] = True
        for nxt in _neighbors4(cell[0], cell[1], height, width):
            if not bool(mask[nxt]):
                frontier.add(nxt)

    if int(mask.sum()) != int(target_area):
        return None
    return _crop(mask)


def _carve_single_hole(mask: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not bool(mask[r, c]):
                continue
            if not all(bool(mask[nr, nc]) for nr, nc in _neighbors4(r, c, h, w)):
                continue
            carved = np.array(mask, copy=True)
            carved[r, c] = False
            if not _is_connected(carved):
                continue
            holes = _hole_regions(carved)
            if len(holes) == 1 and len(holes[0]) == 1:
                candidates.append(carved)
    if not candidates:
        return None
    return np.array(random.choice(candidates), dtype=bool, copy=True)


def _make_shape(want_hole: bool) -> np.ndarray | None:
    min_side = 4 if want_hole else SHAPE_SIDE_MIN
    for _ in range(200):
        box_h = random.randint(min_side, SHAPE_SIDE_MAX)
        box_w = random.randint(min_side, SHAPE_SIDE_MAX)
        box_area = box_h * box_w
        area_lo = max(box_h + box_w - 2, int(np.ceil(box_area * 0.45)))
        area_hi = max(area_lo, int(np.floor(box_area * 0.82)))
        target_area = random.randint(area_lo, area_hi)
        mask = _grow_shape(box_h, box_w, target_area)
        if mask is None:
            continue
        if min(mask.shape) < SHAPE_SIDE_MIN or int(mask.sum()) < 5:
            continue
        if _hole_regions(mask):
            continue
        if want_hole:
            mask = _carve_single_hole(mask)
            if mask is None:
                continue
        transformed = _random_transform(mask)
        if want_hole:
            holes = _hole_regions(transformed)
            if len(holes) != 1 or len(holes[0]) != 1:
                continue
        else:
            if _hole_regions(transformed):
                continue
        return transformed
    return None


def _choose_shapes(group_count: int) -> list[tuple[np.ndarray, bool]] | None:
    if group_count < 2:
        return None

    hole_flags = [True, False]
    while len(hole_flags) < group_count:
        hole_flags.append(bool(random.getrandbits(1)))
    random.shuffle(hole_flags)

    shapes: list[tuple[np.ndarray, bool]] = []
    seen = set()
    for want_hole in hole_flags:
        shape = _make_shape(bool(want_hole))
        if shape is None:
            return None
        key = _mask_key(shape)
        if key in seen:
            return None
        seen.add(key)
        shapes.append((shape, bool(want_hole)))
    return shapes


def _grid_shape(shapes: list[np.ndarray]) -> tuple[int, int] | None:
    padded_area = 0
    for mask in shapes:
        h, w = mask.shape
        padded_area += 3 * (h + 2 * PLACEMENT_PAD) * (w + 2 * PLACEMENT_PAD)
    side = max(GRID_MIN, int(np.ceil(np.sqrt(padded_area))))
    if side > GRID_MAX:
        return None
    h = random.randint(side, min(GRID_MAX, side + 4))
    w = random.randint(side, min(GRID_MAX, side + 4))
    return int(h), int(w)


def _place_group(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    colors: tuple[int, int],
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    placements = []

    for _ in range(300):
        if len(placements) == 3:
            break
        top = random.randint(0, h - mh)
        left = random.randint(0, w - mw)
        r0 = max(0, top - PLACEMENT_PAD)
        r1 = min(h, top + mh + PLACEMENT_PAD)
        c0 = max(0, left - PLACEMENT_PAD)
        c1 = min(w, left + mw + PLACEMENT_PAD)
        if bool(blocked[r0:r1, c0:c1].any()):
            continue
        placements.append((int(top), int(left), int(r0), int(r1), int(c0), int(c1)))
        blocked[r0:r1, c0:c1] = True

    if len(placements) != 3:
        return False

    majority_color, minority_color = colors
    minority_index = random.randrange(3)
    for idx, (top, left, _r0, _r1, _c0, _c1) in enumerate(placements):
        color = minority_color if idx == minority_index else majority_color
        patch = grid[top : top + mh, left : left + mw]
        patch[mask] = int(color)
    return True


def _valid_generated_grid(grid: np.ndarray, group_count: int) -> bool:
    if int(np.count_nonzero(grid == BG_COLOR)) <= grid.size // 2:
        return False

    comps = _extract_components(grid, BG_COLOR)
    if len(comps) != 3 * int(group_count):
        return False

    groups: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    for comp in comps:
        groups[_mask_key(comp["mask"])].append(comp)
    if len(groups) != int(group_count):
        return False

    has_hole = False
    has_no_hole = False
    for group in groups.values():
        if len(group) != 3:
            return False
        counts = Counter(int(comp["color"]) for comp in group)
        if sorted(counts.values()) != [1, 2]:
            return False
        holes = _hole_regions(group[0]["mask"])
        if holes:
            has_hole = True
            if len(holes) != 1 or len(holes[0]) != 1:
                return False
        else:
            has_no_hole = True
    return has_hole and has_no_hole


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(*GROUP_COUNT_RANGE)
    shape_specs = _choose_shapes(group_count)
    if shape_specs is None:
        return None

    shapes = [shape for shape, _want_hole in shape_specs]
    grid_shape = _grid_shape(shapes)
    if grid_shape is None:
        return None
    h, w = grid_shape

    color_pool = list(range(1, 10))
    random.shuffle(color_pool)
    if len(color_pool) < 2 * group_count:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for idx, mask in enumerate(shapes):
        majority_color = int(color_pool[2 * idx])
        minority_color = int(color_pool[2 * idx + 1])
        if not _place_group(grid, blocked, mask, (majority_color, minority_color)):
            return None

    if not _valid_generated_grid(grid, group_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
