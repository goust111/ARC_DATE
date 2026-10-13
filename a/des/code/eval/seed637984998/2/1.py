from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
        (r - 1, c - 1),
        (r - 1, c + 1),
        (r + 1, c - 1),
        (r + 1, c + 1),
    ]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors8(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _local_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _enclosed_region(comp: dict) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = comp["bbox"]
    mask = _local_mask(comp)
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if not mask[r, c] and not outside[r, c]:
                    outside[r, c] = True
                    q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    enclosed = set()
    for r in range(h):
        for c in range(w):
            if not mask[r, c] and not outside[r, c]:
                enclosed.add((r0 + r, c0 + c))
    return enclosed


def _ring_pair(components: list[dict]) -> tuple[dict, dict, int]:
    hollow = []
    for comp in components:
        interior = _enclosed_region(comp)
        if interior:
            item = dict(comp)
            item["interior"] = interior
            item["local_coords"] = {(r - comp["bbox"][0], c - comp["bbox"][1]) for r, c in comp["coords"]}
            hollow.append(item)

    candidates: list[tuple[int, int, dict, dict]] = []
    for big in hollow:
        br0, bc0, br1, bc1 = big["bbox"]
        big_h = br1 - br0 + 1
        big_w = bc1 - bc0 + 1
        for small in hollow:
            if big is small:
                continue
            sr0, sc0, sr1, sc1 = small["bbox"]
            small_h = sr1 - sr0 + 1
            small_w = sc1 - sc0 + 1
            if big_h <= small_h or big_w <= small_w:
                continue
            if big_h % small_h != 0 or big_w % small_w != 0:
                continue
            k = big_h // small_h
            if k <= 1 or big_w // small_w != k:
                continue
            projected = {(r // k, c // k) for r, c in big["local_coords"]}
            if projected == small["local_coords"]:
                candidates.append((big_h * big_w, len(big["coords"]), big, small))

    assert candidates, "failed to identify fence and miniature outline"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    big = candidates[0][2]
    small = candidates[0][3]
    k = (big["bbox"][2] - big["bbox"][0] + 1) // (small["bbox"][2] - small["bbox"][0] + 1)
    return big, small, int(k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the miniature fence map by projecting interior fragments onto the
    scaled-down outline and coloring each miniature region by fragment color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    components = _find_components(g, bg)
    big_ring, small_ring, k = _ring_pair(components)

    out = g.copy()
    big_r0, big_c0, _big_r1, _big_c1 = big_ring["bbox"]
    small_r0, small_c0, _small_r1, _small_c1 = small_ring["bbox"]
    big_inside = big_ring["interior"]
    small_inside = small_ring["interior"]

    best_area: dict[tuple[int, int], int] = {}
    best_color: dict[tuple[int, int], int] = {}

    for comp in components:
        if comp["bbox"] == big_ring["bbox"] and comp["coords"] == big_ring["coords"]:
            continue
        if comp["bbox"] == small_ring["bbox"] and comp["coords"] == small_ring["coords"]:
            continue
        coords = comp["coords"]
        if not all((r, c) in big_inside for r, c in coords):
            continue
        mapped = set()
        for r, c in coords:
            rr = (r - big_r0) // k
            cc = (c - big_c0) // k
            tgt = (small_r0 + rr, small_c0 + cc)
            if tgt in small_inside:
                mapped.add(tgt)
        area = len(coords)
        for tgt in mapped:
            if area > best_area.get(tgt, -1):
                best_area[tgt] = area
                best_color[tgt] = int(comp["color"])

    for (r, c), color in best_color.items():
        out[r, c] = color
    return out


def _connected(mask: np.ndarray, use_diagonal: bool = False) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in pts[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        step_neighbors = _neighbors8 if use_diagonal else _neighbors4
        for nr, nc in step_neighbors(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _shape_with_notches() -> np.ndarray | None:
    h = random.randint(6, 9)
    w = random.randint(6, 10)
    mask = np.ones((h, w), dtype=bool)
    for _ in range(random.randint(0, 4)):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            if h <= 4 or w <= 4:
                continue
            depth = random.randint(1, h - 3)
            span = random.randint(1, w - 2)
            c0 = random.randint(1, w - span - 1)
            if side == "top":
                mask[:depth, c0 : c0 + span] = False
            else:
                mask[h - depth :, c0 : c0 + span] = False
        else:
            if h <= 4 or w <= 4:
                continue
            depth = random.randint(1, w - 3)
            span = random.randint(1, h - 2)
            r0 = random.randint(1, h - span - 1)
            if side == "left":
                mask[r0 : r0 + span, :depth] = False
            else:
                mask[r0 : r0 + span, w - depth :] = False

    if not _connected(mask):
        return None
    small_inside = mask & (~_boundary(mask))
    if int(small_inside.sum()) < 4:
        return None
    if not _connected(_boundary(mask), use_diagonal=True):
        return None
    return mask


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _grow_shape(start: tuple[int, int], free: set[tuple[int, int]], size: int) -> set[tuple[int, int]] | None:
    shape = {start}
    while len(shape) < size:
        frontier = []
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in free and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(frontier))
    return shape


def _collision_patterns(k: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    assert k >= 3
    patterns = [
        ({(0, 0)}, {(k - 1, k - 2), (k - 1, k - 1)}),
        ({(0, k - 1)}, {(k - 2, 0), (k - 1, 0)}),
        ({(k - 1, 0)}, {(0, k - 2), (0, k - 1)}),
        ({(k - 1, k - 1)}, {(0, 0), (1, 0)}),
    ]
    return patterns


def _place_block_fragment(grid: np.ndarray, big_r0: int, big_c0: int, k: int, cells: set[tuple[int, int]], color: int) -> int:
    area = 0
    for sr, sc in cells:
        rr = big_r0 + sr * k
        cc = big_c0 + sc * k
        grid[rr : rr + k, cc : cc + k] = color
        area += k * k
    return area


def _attempt_generate() -> np.ndarray | None:
    collision_mode = random.random() < 0.5
    shape = None
    for _ in range(100):
        shape = _shape_with_notches()
        if shape is not None:
            break
    if shape is None:
        return None

    k = random.randint(3, 4) if collision_mode else random.randint(2, 4)
    small_ring = _boundary(shape)
    small_inside = shape & (~small_ring)
    if int(small_inside.sum()) < 4:
        return None

    big_fill = np.kron(shape.astype(int), np.ones((k, k), dtype=int)).astype(bool)
    big_ring = _boundary(big_fill)
    big_h, big_w = big_ring.shape
    small_h, small_w = small_ring.shape

    grid_h = big_h + small_h + random.randint(4, 9)
    grid_w = big_w + small_w + random.randint(4, 9)
    if grid_h < big_h + 2 or grid_w < big_w + 2:
        return None

    small_margin = random.randint(0, 1)
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        small_r0, small_c0 = small_margin, small_margin
    elif corner == "tr":
        small_r0, small_c0 = small_margin, grid_w - small_w - small_margin
    elif corner == "bl":
        small_r0, small_c0 = grid_h - small_h - small_margin, small_margin
    else:
        small_r0, small_c0 = grid_h - small_h - small_margin, grid_w - small_w - small_margin

    big_r0 = big_c0 = None
    for _ in range(200):
        rr = random.randint(0, grid_h - big_h)
        cc = random.randint(0, grid_w - big_w)
        if _boxes_separate((rr, cc, big_h, big_w), (small_r0, small_c0, small_h, small_w), pad=1):
            big_r0, big_c0 = rr, cc
            break
    if big_r0 is None or big_c0 is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    fence_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    grid[big_r0 : big_r0 + big_h, big_c0 : big_c0 + big_w][big_ring] = fence_color
    grid[small_r0 : small_r0 + small_h, small_c0 : small_c0 + small_w][small_ring] = fence_color

    interior_cells = [(int(r), int(c)) for r, c in np.argwhere(small_inside)]
    free = set(interior_cells)
    blocked: set[tuple[int, int]] = set()
    fragments: list[dict] = []

    if collision_mode:
        collision_choices = list(free)
        random.shuffle(collision_choices)
        if not collision_choices:
            return None
        cell = collision_choices[0]
        blocked.add(cell)
        for nr, nc in _neighbors4(*cell):
            blocked.add((nr, nc))
        pat_a, pat_b = random.choice(_collision_patterns(k))
        colors = random.sample([c for c in range(1, 10) if c != fence_color], 2)
        fragments.append({"mode": "collision", "cell": cell, "pixels": pat_a, "color": colors[0], "area": len(pat_a)})
        fragments.append({"mode": "collision", "cell": cell, "pixels": pat_b, "color": colors[1], "area": len(pat_b)})

    remaining_cells = [cell for cell in free if cell not in blocked]
    random.shuffle(remaining_cells)
    normal_target = random.randint(2, 4)
    max_colors = 9 - 1 - len(fragments)
    if normal_target > max_colors:
        normal_target = max_colors
    color_pool = [c for c in range(1, 10) if c != fence_color and c not in {frag["color"] for frag in fragments}]
    random.shuffle(color_pool)

    for color in color_pool:
        free_now = [cell for cell in free if cell not in blocked]
        if not free_now or len([frag for frag in fragments if frag["mode"] == "normal"]) >= normal_target:
            break
        start = random.choice(free_now)
        size = random.randint(1, min(4, len(free_now)))
        shape_cells = None
        for _ in range(25):
            candidate = _grow_shape(start, set(free_now), size)
            if candidate is not None:
                shape_cells = candidate
                break
            size = max(1, size - 1)
        if shape_cells is None:
            continue
        fragments.append({"mode": "normal", "cells": shape_cells, "color": color, "area": len(shape_cells) * k * k})
        for cell in shape_cells:
            blocked.add(cell)
            for nr, nc in _neighbors4(*cell):
                blocked.add((nr, nc))

    normal_count = len([frag for frag in fragments if frag["mode"] == "normal"])
    if normal_count < 2:
        return None

    for frag in fragments:
        if frag["mode"] == "normal":
            _place_block_fragment(grid, big_r0, big_c0, k, frag["cells"], frag["color"])
        else:
            sr, sc = frag["cell"]
            rr0 = big_r0 + sr * k
            cc0 = big_c0 + sc * k
            for dr, dc in frag["pixels"]:
                grid[rr0 + dr, cc0 + dc] = frag["color"]

    out = main(grid)
    small_patch_in = grid[small_r0 : small_r0 + small_h, small_c0 : small_c0 + small_w]
    small_patch_out = out[small_r0 : small_r0 + small_h, small_c0 : small_c0 + small_w]
    if np.array_equal(small_patch_in, small_patch_out):
        return None
    if len(np.unique(small_patch_out[small_patch_out != 0])) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
