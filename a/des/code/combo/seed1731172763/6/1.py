from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = list(range(1, 10))

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
GEN_ATTEMPTS = 400
GROUP_COUNT_OPTIONS = (2, 3)
HOLEY_COUNTS = (1, 2, 3)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(visited[r, c]):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(visited[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _hole_count_mask(mask: np.ndarray) -> int:
    holes = 0
    for comp in _mask_components(~mask):
        if any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _count_holes(coords: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return _hole_count_mask(mask)


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or bool(visited[r, c]):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not bool(visited[nr, nc]) and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                    "holes": _count_holes(coords),
                }
            )
    return objects


def _winner(objs: list[dict]) -> dict:
    return min(objs, key=lambda obj: (-obj["area"], obj["bbox"][0], obj["bbox"][1], obj["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group foreground objects by hole count and recolor each group to the color of
    that group's largest object.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in _analyze_objects(g):
        groups[int(obj["holes"])].append(obj)

    for objs in groups.values():
        group_color = int(_winner(objs)["color"])
        for obj in objs:
            rows = [r for r, _ in obj["coords"]]
            cols = [c for _, c in obj["coords"]]
            out[rows, cols] = group_color

    return out


def _template(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in lines], dtype=bool)


TEMPLATES_BY_HOLES = {
    0: [
        _template(("111", "111")),
        _template(("110", "010", "011")),
        _template(("111", "010", "010")),
        _template(("100", "111", "001")),
        _template(("110", "110", "011")),
    ],
    1: [
        _template(("111", "101", "111")),
        _template(("1111", "1001", "1001", "1111")),
        _template(("010", "101", "111")),
        _template(("111", "101", "111", "010")),
    ],
    2: [
        _template(("11111", "10101", "11111")),
        _template(("111", "101", "111", "101", "111")),
        _template(("11111", "10101", "11111", "00100")),
    ],
    3: [
        _template(("1111111", "1010101", "1111111")),
        _template(("111", "101", "111", "101", "111", "101", "111")),
    ],
}


def _split_total(parts: int, total: int) -> list[int]:
    vals = [1] * parts
    for _ in range(total - parts):
        vals[random.randrange(parts)] += 1
    return vals


def _scaled_mask(base: np.ndarray, role: str) -> np.ndarray:
    rows, cols = map(int, base.shape)
    max_dim = max(rows, cols)

    if role == "winner":
        max_total = 18
        low_mult, high_mult = 2.0, 3.2
    else:
        max_total = 13
        low_mult, high_mult = 1.2, 2.5

    h_lo = max(rows, int(np.ceil(rows * low_mult)))
    h_hi = min(max_total, max(rows, int(np.floor(rows * high_mult))))
    w_lo = max(cols, int(np.ceil(cols * low_mult)))
    w_hi = min(max_total, max(cols, int(np.floor(cols * high_mult))))
    target_h = random.randint(h_lo, h_hi)
    target_w = random.randint(w_lo, w_hi)
    row_heights = _split_total(rows, target_h)
    col_widths = _split_total(cols, target_w)

    out = np.zeros((sum(row_heights), sum(col_widths)), dtype=bool)
    r0 = 0
    for i, rh in enumerate(row_heights):
        c0 = 0
        for j, cw in enumerate(col_widths):
            if bool(base[i, j]):
                out[r0 : r0 + rh, c0 : c0 + cw] = True
            c0 += cw
        r0 += rh
    return out


def _random_mask(hole_count: int, role: str) -> np.ndarray:
    for _ in range(120):
        base = np.array(random.choice(TEMPLATES_BY_HOLES[hole_count]), copy=True)
        if random.random() < 0.5:
            base = np.fliplr(base)
        base = np.rot90(base, k=random.randint(0, 3))
        mask = _scaled_mask(base, role)
        if _hole_count_mask(mask) != hole_count:
            continue
        if len(_mask_components(mask)) != 1:
            continue
        return mask
    raise RuntimeError("failed to build mask")


def _make_group_masks(hole_count: int, group_size: int) -> list[np.ndarray]:
    for _ in range(200):
        masks = [_random_mask(hole_count, "winner")]
        masks.extend(_random_mask(hole_count, "loser") for _ in range(group_size - 1))
        areas = [int(mask.sum()) for mask in masks]
        max_area = max(areas)
        if areas.count(max_area) != 1 or areas[0] != max_area:
            continue
        return masks
    raise RuntimeError("failed to create group with unique largest object")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            out[rr[ok], cc[ok]] = True
    return out


def _place_objects(masks: list[np.ndarray], colors: list[int], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(masks)), key=lambda idx: int(masks[idx].sum()), reverse=True)

    for idx in order:
        mask = masks[idx]
        mh, mw = map(int, mask.shape)
        if mh > h - 2 or mw > w - 2:
            return None
        positions = [(r, c) for r in range(1, h - mh) for c in range(1, w - mw)]
        random.shuffle(positions)
        rows, cols = np.where(mask)
        placed = False
        for r0, c0 in positions:
            rr = rows + r0
            cc = cols + c0
            if bool(blocked[rr, cc].any()):
                continue
            grid[rr, cc] = int(colors[idx])
            placements[idx] = (r0, c0)

            dilated = _dilate8(mask)
            drs, dcs = np.where(dilated)
            blocked[drs + r0, dcs + c0] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _group_sizes(num_groups: int) -> list[int]:
    if num_groups == 2:
        return [random.randint(2, 3), random.randint(2, 3)]
    sizes = [2, 2, 2]
    if random.random() < 0.5:
        sizes[random.randrange(3)] = 3
    return sizes


def _planned_hole_counts(num_groups: int) -> list[int]:
    holey = random.sample(HOLEY_COUNTS, num_groups - 1)
    counts = [0] + holey
    random.shuffle(counts)
    return counts


def _build_candidate() -> np.ndarray | None:
    num_groups = random.choice(GROUP_COUNT_OPTIONS)
    hole_counts = _planned_hole_counts(num_groups)
    sizes = _group_sizes(num_groups)
    total_objects = sum(sizes)
    colors = random.sample(PALETTE, total_objects)

    masks: list[np.ndarray] = []
    planned_groups: list[tuple[int, int, list[int]]] = []
    color_idx = 0
    for hole_count, group_size in zip(hole_counts, sizes, strict=True):
        group_masks = _make_group_masks(hole_count, group_size)
        group_colors = colors[color_idx : color_idx + group_size]
        color_idx += group_size

        order = list(range(group_size))
        random.shuffle(order)
        winner_idx = max(range(group_size), key=lambda idx: int(group_masks[idx].sum()))
        if order[0] == winner_idx:
            swap_at = random.randrange(1, group_size)
            order[0], order[swap_at] = order[swap_at], order[0]

        for idx in order:
            masks.append(group_masks[idx])
        planned_groups.append((hole_count, group_size, [group_colors[idx] for idx in order]))

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    permuted_colors = [color for _hole_count, _size, group_colors in planned_groups for color in group_colors]
    grid = _place_objects(masks, permuted_colors, h, w)
    if grid is None:
        return None
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if np.any(grid < 0) or np.any(grid > 9):
        return False

    objects = _analyze_objects(grid)
    if len(objects) < 4:
        return False

    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        groups[int(obj["holes"])].append(obj)

    if 0 not in groups or len(groups) < 2:
        return False
    if not any(hole_count > 0 for hole_count in groups):
        return False
    if any(len(objs) < 2 for objs in groups.values()):
        return False
    if len(objects) > len(PALETTE):
        return False

    for objs in groups.values():
        areas = [int(obj["area"]) for obj in objs]
        if areas.count(max(areas)) != 1:
            return False
        colors = {int(obj["color"]) for obj in objs}
        if len(colors) != len(objs):
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    for objs in groups.values():
        group_color = int(_winner(objs)["color"])
        changed = False
        for obj in objs:
            rows = [r for r, _ in obj["coords"]]
            cols = [c for _, c in obj["coords"]]
            vals = set(map(int, out[rows, cols].tolist()))
            if vals != {group_color}:
                return False
            if int(obj["color"]) != group_color:
                changed = True
        if not changed:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC task input")
