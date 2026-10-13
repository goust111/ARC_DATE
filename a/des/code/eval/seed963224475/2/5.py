from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = DIRS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

# Generator controls.
GEN_RETRIES = 300
PLACEMENT_ATTEMPTS = 200
GRID_MAX = 30
GRID_MARGIN = 1
PAIR_COUNT_RANGE = (2, 4)
GAP_RANGE = (1, 3)
COLOR_POOL = tuple(range(1, 10))


def _scale2_mask(mask: np.ndarray) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((2, 2), dtype=np.uint8)).astype(bool)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    return comps


def _match_scaled_pair(comp_a: dict, comp_b: dict) -> tuple[int, int] | None:
    mask_a = comp_a["mask"]
    mask_b = comp_b["mask"]

    if (
        mask_a.shape[0] * 2 == mask_b.shape[0]
        and mask_a.shape[1] * 2 == mask_b.shape[1]
        and np.array_equal(_scale2_mask(mask_a), mask_b)
    ):
        return 0, 1

    if (
        mask_b.shape[0] * 2 == mask_a.shape[0]
        and mask_b.shape[1] * 2 == mask_a.shape[1]
        and np.array_equal(_scale2_mask(mask_b), mask_a)
    ):
        return 1, 0

    return None


def _find_pairs(comps: list[dict]) -> list[tuple[int, int]]:
    n = len(comps)
    assert n % 2 == 0 and n > 0, "expected an even number of objects"

    pairs: list[tuple[int, int]] = []
    degree = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            match = _match_scaled_pair(comps[i], comps[j])
            if match is None:
                continue
            small_idx = i if match[0] == 0 else j
            large_idx = j if match[1] == 1 else i
            pairs.append((small_idx, large_idx))
            degree[small_idx] += 1
            degree[large_idx] += 1

    assert len(pairs) * 2 == n, "every object must belong to exactly one scaled pair"
    assert all(d == 1 for d in degree), "scaled pairing must be unique"
    return pairs


def _relative_side(small_bbox: tuple[int, int, int, int], large_bbox: tuple[int, int, int, int]) -> str:
    sr0, sc0, sr1, sc1 = small_bbox
    lr0, lc0, lr1, lc1 = large_bbox

    horizontal_overlap = not (sc1 < lc0 or sc0 > lc1)
    vertical_overlap = not (sr1 < lr0 or sr0 > lr1)

    if horizontal_overlap and sr1 < lr0:
        return "up"
    if horizontal_overlap and sr0 > lr1:
        return "down"
    if vertical_overlap and sc1 < lc0:
        return "left"
    if vertical_overlap and sc0 > lc1:
        return "right"

    small_center_r = (sr0 + sr1) / 2.0
    small_center_c = (sc0 + sc1) / 2.0
    large_center_r = (lr0 + lr1) / 2.0
    large_center_c = (lc0 + lc1) / 2.0
    dr = small_center_r - large_center_r
    dc = small_center_c - large_center_c
    if abs(dc) >= abs(dr):
        return "left" if dc < 0 else "right"
    return "up" if dr < 0 else "down"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shaped monochrome objects related by an exact 2x scale factor.
    Erase each small object, then recolor the matching large object's edge strip
    from the side where the small object sits. The strip depth equals the small
    object's bbox size along that direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _find_components(g)
    pairs = _find_pairs(comps)
    out = g.copy()

    for small_idx, large_idx in pairs:
        small = comps[small_idx]
        large = comps[large_idx]
        side = _relative_side(small["bbox"], large["bbox"])
        small_h, small_w = map(int, small["mask"].shape)
        lr0, lc0, lr1, lc1 = large["bbox"]
        new_color = int(small["color"])

        for r, c in large["coords"]:
            if side == "left" and c - lc0 < small_w:
                out[r, c] = new_color
            elif side == "right" and lc1 - c < small_w:
                out[r, c] = new_color
            elif side == "up" and r - lr0 < small_h:
                out[r, c] = new_color
            elif side == "down" and lr1 - r < small_h:
                out[r, c] = new_color

        for r, c in small["coords"]:
            out[r, c] = BG

    return out


def _random_connected_mask(max_dim: int) -> np.ndarray:
    for _ in range(100):
        seed_h = random.randint(2, max_dim)
        seed_w = random.randint(2, max_dim)
        area = seed_h * seed_w
        target = random.randint(max(2, (area + 2) // 3), area)

        cells = {(random.randrange(seed_h), random.randrange(seed_w))}
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < seed_h and 0 <= nc < seed_w:
                    frontier.add((nr, nc))

        while len(cells) < target:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            pr, pc = pick
            for dr, dc in DIRS4:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < seed_h and 0 <= nc < seed_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        mask = np.zeros((seed_h, seed_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if int(mask.sum()) >= 2:
            return mask

    raise RuntimeError("failed to generate a connected mask")


def _make_pair_specs(pair_count: int) -> list[dict] | None:
    directions = random.sample(("up", "down", "left", "right"), pair_count)
    colors = random.sample(COLOR_POOL, 2 * pair_count)
    max_dim = 4 if pair_count <= 3 else 3
    used_component_shapes: set[tuple[tuple[int, int], bytes]] = set()
    specs: list[dict] = []

    for idx in range(pair_count):
        for _ in range(100):
            small_mask = _random_connected_mask(max_dim)
            key = _mask_key(small_mask)
            large_mask = _scale2_mask(small_mask)
            large_key = _mask_key(large_mask)
            if key in used_component_shapes or large_key in used_component_shapes:
                continue
            used_component_shapes.add(key)
            used_component_shapes.add(large_key)
            break
        else:
            return None

        gap = random.randint(*GAP_RANGE)
        sh, sw = map(int, small_mask.shape)
        lh, lw = map(int, large_mask.shape)
        direction = directions[idx]

        if direction in ("left", "right"):
            union_h, union_w = lh, sw + gap + lw
        else:
            union_h, union_w = sh + gap + lh, lw

        specs.append(
            {
                "small_mask": small_mask,
                "large_mask": large_mask,
                "small_color": colors[2 * idx],
                "large_color": colors[2 * idx + 1],
                "direction": direction,
                "gap": gap,
                "union_h": union_h,
                "union_w": union_w,
            }
        )

    return specs


def _choose_grid_shape(specs: list[dict]) -> tuple[int, int] | None:
    pair_count = len(specs)
    need_h = max(int(spec["union_h"]) for spec in specs) + 2 * GRID_MARGIN
    need_w = max(int(spec["union_w"]) for spec in specs) + 2 * GRID_MARGIN
    base = 16 + 3 * pair_count
    lo_h = max(base, need_h)
    lo_w = max(base, need_w)
    if lo_h > GRID_MAX or lo_w > GRID_MAX:
        return None
    return random.randint(lo_h, GRID_MAX), random.randint(lo_w, GRID_MAX)


def _place_pair(grid: np.ndarray, reserved: np.ndarray, spec: dict) -> bool:
    h, w = grid.shape
    small = spec["small_mask"]
    large = spec["large_mask"]
    sh, sw = map(int, small.shape)
    lh, lw = map(int, large.shape)
    gap = int(spec["gap"])
    direction = spec["direction"]

    for _ in range(PLACEMENT_ATTEMPTS):
        if direction == "left":
            row_offset = random.randint(0, lh - sh)
            lr0_min, lr0_max = GRID_MARGIN, h - GRID_MARGIN - lh
            lc0_min, lc0_max = GRID_MARGIN + sw + gap, w - GRID_MARGIN - lw
            if lr0_min > lr0_max or lc0_min > lc0_max:
                return False
            lr0 = random.randint(lr0_min, lr0_max)
            lc0 = random.randint(lc0_min, lc0_max)
            sr0 = lr0 + row_offset
            sc0 = lc0 - gap - sw
        elif direction == "right":
            row_offset = random.randint(0, lh - sh)
            lr0_min, lr0_max = GRID_MARGIN, h - GRID_MARGIN - lh
            lc0_min, lc0_max = GRID_MARGIN, w - GRID_MARGIN - lw - gap - sw
            if lr0_min > lr0_max or lc0_min > lc0_max:
                return False
            lr0 = random.randint(lr0_min, lr0_max)
            lc0 = random.randint(lc0_min, lc0_max)
            sr0 = lr0 + row_offset
            sc0 = lc0 + lw + gap
        elif direction == "up":
            col_offset = random.randint(0, lw - sw)
            lr0_min, lr0_max = GRID_MARGIN + sh + gap, h - GRID_MARGIN - lh
            lc0_min, lc0_max = GRID_MARGIN, w - GRID_MARGIN - lw
            if lr0_min > lr0_max or lc0_min > lc0_max:
                return False
            lr0 = random.randint(lr0_min, lr0_max)
            lc0 = random.randint(lc0_min, lc0_max)
            sr0 = lr0 - gap - sh
            sc0 = lc0 + col_offset
        else:
            col_offset = random.randint(0, lw - sw)
            lr0_min, lr0_max = GRID_MARGIN, h - GRID_MARGIN - lh - gap - sh
            lc0_min, lc0_max = GRID_MARGIN, w - GRID_MARGIN - lw
            if lr0_min > lr0_max or lc0_min > lc0_max:
                return False
            lr0 = random.randint(lr0_min, lr0_max)
            lc0 = random.randint(lc0_min, lc0_max)
            sr0 = lr0 + lh + gap
            sc0 = lc0 + col_offset

        ur0 = min(sr0, lr0)
        uc0 = min(sc0, lc0)
        ur1 = max(sr0 + sh - 1, lr0 + lh - 1)
        uc1 = max(sc0 + sw - 1, lc0 + lw - 1)
        pr0 = max(0, ur0 - 1)
        pc0 = max(0, uc0 - 1)
        pr1 = min(h - 1, ur1 + 1)
        pc1 = min(w - 1, uc1 + 1)
        if reserved[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
            continue

        _paint_mask(grid, large, lr0, lc0, int(spec["large_color"]))
        _paint_mask(grid, small, sr0, sc0, int(spec["small_color"]))
        reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    specs = _make_pair_specs(pair_count)
    if specs is None:
        return None

    shape = _choose_grid_shape(specs)
    if shape is None:
        return None

    grid = np.zeros(shape, dtype=int)
    reserved = np.zeros(shape, dtype=bool)
    order = sorted(specs, key=lambda spec: int(spec["union_h"]) * int(spec["union_w"]), reverse=True)

    for spec in order:
        if not _place_pair(grid, reserved, spec):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scaled-pair sample")
