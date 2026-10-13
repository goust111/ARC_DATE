from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG_COLOR = 7
LINE_COLOR = 3
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch, (r0, c0, r1, c1)


def _patch_key(patch: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, patch.shape)), patch.tobytes()


def _center_from_bbox(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _draw_path(out: np.ndarray, bg: int, start: tuple[int, int], end: tuple[int, int]) -> None:
    r1, c1 = start
    r2, c2 = end
    step_c = 1 if c2 >= c1 else -1
    for c in range(c1, c2 + step_c, step_c):
        if int(out[r1, c]) == bg:
            out[r1, c] = LINE_COLOR
    step_r = 1 if r2 >= r1 else -1
    for r in range(r1, r2 + step_r, step_r):
        if int(out[r, c2]) == bg:
            out[r, c2] = LINE_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify multiple dihedral-transform copies of the same multicolor sprite and
    connect equal-orientation pairs with green orthogonal L paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    comps = _find_components(g, bg)
    assert len(comps) >= 2

    groups: dict[tuple[tuple[int, int], bytes], list[tuple[int, int]]] = defaultdict(list)
    for coords in comps:
        patch, box = _extract_patch(g, coords, bg)
        groups[_patch_key(patch)].append(_center_from_bbox(box))

    out = g.copy()
    for centers in groups.values():
        assert len(centers) % 2 == 0, "each orientation must appear an even number of times"
        centers.sort()
        for i in range(0, len(centers), 2):
            _draw_path(out, bg, centers[i], centers[i + 1])
    return out


def _transform_patch(patch: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    out = np.fliplr(patch) if flip else patch.copy()
    return np.rot90(out, rot)


def _all_transforms(patch: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        for rot in range(4):
            cur = _transform_patch(patch, rot, flip)
            key = _patch_key(cur)
            if key not in seen:
                seen.add(key)
                variants.append(cur)
    return variants


def _random_connected_mask(size: int, area: int) -> np.ndarray:
    center = size // 2
    mask = np.zeros((size, size), dtype=bool)
    mask[center, center] = True
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS8:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    frontier.add((nr, nc))
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True
    return mask


def _make_template() -> np.ndarray:
    palette = [c for c in range(10) if c not in (BG_COLOR, LINE_COLOR)]
    for _ in range(400):
        size = random.choice([3, 5])
        max_area = size * size - 1
        area = random.randint(size + 2, min(max_area, size * size - 2))
        mask = _random_connected_mask(size, area)
        colors = random.sample(palette, random.randint(2, min(4, area)))
        patch = np.full((size, size), BG_COLOR, dtype=int)
        coords = list(map(tuple, np.argwhere(mask)))
        random.shuffle(coords)
        for i, (r, c) in enumerate(coords):
            patch[r, c] = colors[i % len(colors)]
        if len({_patch_key(v) for v in _all_transforms(patch)}) != 8:
            continue
        return patch
    raise RuntimeError("failed to build asymmetric template")


def _path_cells(start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    r1, c1 = start
    r2, c2 = end
    cells = set()
    step_c = 1 if c2 >= c1 else -1
    for c in range(c1, c2 + step_c, step_c):
        cells.add((r1, c))
    step_r = 1 if r2 >= r1 else -1
    for r in range(r1, r2 + step_r, step_r):
        cells.add((r, c2))
    return cells


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _patch_cells(patch: np.ndarray, top: int, left: int, bg: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(patch != bg):
        cells.add((top + int(r), left + int(c)))
    return cells


def _can_place_patch(
    patch: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    sprite_blocked: set[tuple[int, int]],
    bg: int,
) -> bool:
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    cells = _patch_cells(patch, top, left, bg)
    return not (cells & sprite_blocked)


def _place_pair(
    patch: np.ndarray,
    h: int,
    w: int,
    sprite_cells: set[tuple[int, int]],
    sprite_blocked: set[tuple[int, int]],
    corridor_reserved: set[tuple[int, int]],
    bg: int,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    ph, pw = patch.shape
    center = (ph // 2, pw // 2)
    positions = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
    random.shuffle(positions)

    for top1, left1 in positions[:160]:
        if not _can_place_patch(patch, top1, left1, h, w, sprite_blocked, bg):
            continue
        cells1 = _patch_cells(patch, top1, left1, bg)
        center1 = (top1 + center[0], left1 + center[1])

        positions2 = positions[:]
        random.shuffle(positions2)
        for top2, left2 in positions2[:160]:
            if abs(top1 - top2) + abs(left1 - left2) < 6:
                continue
            if not _can_place_patch(patch, top2, left2, h, w, sprite_blocked | _dilate(cells1, h, w), bg):
                continue
            cells2 = _patch_cells(patch, top2, left2, bg)
            center2 = (top2 + center[0], left2 + center[1])
            path = _path_cells(center1, center2)
            external_path = path - cells1 - cells2
            if external_path & sprite_cells:
                continue
            if external_path & corridor_reserved:
                continue
            return (top1, left1), (top2, left2), cells1 | cells2, external_path
    return None


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    ph, pw = patch.shape
    sub = patch != bg
    grid[top : top + ph, left : left + pw][sub] = patch[sub]


def generate_input() -> np.ndarray:
    for _ in range(300):
        template = _make_template()
        transforms = _all_transforms(template)
        group_count = random.randint(2, 4)
        chosen = random.sample(transforms, group_count)
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.full((h, w), BG_COLOR, dtype=int)

        sprite_cells: set[tuple[int, int]] = set()
        sprite_blocked: set[tuple[int, int]] = set()
        corridor_reserved: set[tuple[int, int]] = set()
        placed: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]] = []
        ok = True

        random.shuffle(chosen)
        for patch in chosen:
            pair = _place_pair(patch, h, w, sprite_cells, sprite_blocked, corridor_reserved, BG_COLOR)
            if pair is None:
                ok = False
                break
            (top1, left1), (top2, left2), new_cells, corridor = pair
            _paint_patch(grid, patch, top1, left1, BG_COLOR)
            _paint_patch(grid, patch, top2, left2, BG_COLOR)
            sprite_cells |= new_cells
            sprite_blocked = _dilate(sprite_cells, h, w) | corridor_reserved
            corridor_reserved |= corridor
            placed.append((patch, (top1, left1), (top2, left2)))
        if not ok:
            continue

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        if not np.any(out == LINE_COLOR):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
