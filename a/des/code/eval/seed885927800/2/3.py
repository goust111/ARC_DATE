from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARROW = 9
OBJECT_COLORS = tuple(range(1, 9))

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_SIZE_OPTIONS = (3, 4, 5)
ARROW_GAP_RANGE = (1, 3)
MAX_ARROWS = 6
MAX_MASK_ATTEMPTS = 200
MAX_LAYOUT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 300

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


_ARROW_RIGHT = np.array(
    [
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
    ],
    dtype=bool,
)

ARROW_INFO = {
    "right": {"vec": (0, 1), "patch": _ARROW_RIGHT, "tip": (1, 2)},
    "down": {"vec": (1, 0), "patch": np.rot90(_ARROW_RIGHT, -1), "tip": (2, 1)},
    "left": {"vec": (0, -1), "patch": np.rot90(_ARROW_RIGHT, 2), "tip": (1, 0)},
    "up": {"vec": (-1, 0), "patch": np.rot90(_ARROW_RIGHT, 1), "tip": (0, 1)},
}
ARROW_SHAPES = {
    tuple(map(tuple, info["patch"].astype(int))): name for name, info in ARROW_INFO.items()
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (BG, ARROW):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox(arr)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
            patch[arr[:, 0] - r0, arr[:, 1] - c0] = color
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                }
            )
    return objects


def _arrow_direction(coords: np.ndarray) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    rel = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    rel[coords[:, 0] - r0, coords[:, 1] - c0] = 1
    key = tuple(map(tuple, rel))
    assert key in ARROW_SHAPES, "unexpected maroon component geometry"
    return ARROW_SHAPES[key]


def _rotate_object_patch(patch: np.ndarray, turns_cw: int) -> np.ndarray:
    k = int(turns_cw) % 4
    return np.rot90(patch, -k) if k else patch.copy()


def _paste_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    for rr in range(h):
        for cc in range(w):
            color = int(patch[rr, cc])
            if color == BG:
                continue
            r = top + rr
            c = left + cc
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert int(out[r, c]) == BG
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Arrows point at objects; each object rotates clockwise once per incoming arrow, and arrows are erased.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    objects = _find_objects(g)
    assert objects, "expected at least one non-arrow object"

    owner = np.full((h, w), -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        owner[coords[:, 0], coords[:, 1]] = idx

    votes = [0] * len(objects)
    for comp in _find_components_of_color(g, ARROW):
        assert comp.shape[0] == 5, "each arrow must have exactly five pixels"
        direction = _arrow_direction(comp)
        info = ARROW_INFO[direction]
        r0, c0, _r1, _c1 = _bbox(comp)
        tip_r = r0 + int(info["tip"][0])
        tip_c = c0 + int(info["tip"][1])
        dr, dc = info["vec"]

        r = tip_r + dr
        c = tip_c + dc
        target = None
        while 0 <= r < h and 0 <= c < w:
            idx = int(owner[r, c])
            if idx != -1:
                target = idx
                break
            r += dr
            c += dc
        assert target is not None, "every arrow must hit an object"
        votes[target] += 1

    out = np.full_like(g, BG)
    for idx, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        patch = obj["patch"]
        rotated = _rotate_object_patch(patch, votes[idx])
        rh, rw = rotated.shape
        center_r2 = r0 + r1
        center_c2 = c0 + c1
        top_num = center_r2 - (rh - 1)
        left_num = center_c2 - (rw - 1)
        assert top_num % 2 == 0 and left_num % 2 == 0, "bbox center must preserve grid alignment"
        top = top_num // 2
        left = left_num // 2
        _paste_patch(out, rotated, int(top), int(left))

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _distinct_rotations(mask: np.ndarray) -> bool:
    keys = {np.rot90(mask, k).tobytes() for k in range(4)}
    return len(keys) == 4


def _random_connected_mask(size: int) -> np.ndarray:
    min_area = max(size + 1, 4)
    max_area = size * size - 1
    for _ in range(MAX_MASK_ATTEMPTS):
        area = random.randint(min_area, max_area)
        mask = np.zeros((size, size), dtype=bool)
        start = (random.randrange(size), random.randrange(size))
        mask[start] = True

        while int(mask.sum()) < area:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if int(mask.sum()) != area:
            continue
        if not _touches_all_sides(mask):
            continue
        if not _distinct_rotations(mask):
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric object mask")


def _place_objects(h: int, w: int, specs: list[dict]) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    for spec in sorted(specs, key=lambda item: int(item["size"]), reverse=True):
        size = int(spec["size"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - size):
            for c0 in range(1, w - size):
                r1 = r0 + size
                c1 = c0 + size
                window = occupied[max(0, r0 - 1) : min(h, r1 + 1), max(0, c0 - 1) : min(w, c1 + 1)]
                if window.any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        spec["top"] = r0
        spec["left"] = c0
        occupied[r0 : r0 + size, c0 : c0 + size] = True
    return True


def _render_objects(h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((h, w), BG, dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    for idx, spec in enumerate(specs):
        top = int(spec["top"])
        left = int(spec["left"])
        mask = spec["mask"]
        color = int(spec["color"])
        cells = np.argwhere(mask)
        grid[top + cells[:, 0], left + cells[:, 1]] = color
        owner[top + cells[:, 0], left + cells[:, 1]] = idx
    return grid, owner


def _trace_target(owner: np.ndarray, tip: tuple[int, int], vec: tuple[int, int]) -> int | None:
    h, w = owner.shape
    r, c = int(tip[0]) + int(vec[0]), int(tip[1]) + int(vec[1])
    while 0 <= r < h and 0 <= c < w:
        idx = int(owner[r, c])
        if idx != -1:
            return idx
        r += int(vec[0])
        c += int(vec[1])
    return None


def _boundary_hits(spec: dict) -> dict[str, list[tuple[int, int]]]:
    mask = spec["mask"]
    top = int(spec["top"])
    left = int(spec["left"])
    hits = {name: [] for name in ARROW_INFO}

    for lr in range(mask.shape[0]):
        cols = np.flatnonzero(mask[lr])
        if cols.size:
            hits["right"].append((top + lr, left + int(cols[0])))
            hits["left"].append((top + lr, left + int(cols[-1])))

    for lc in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, lc])
        if rows.size:
            hits["down"].append((top + int(rows[0]), left + lc))
            hits["up"].append((top + int(rows[-1]), left + lc))

    return hits


def _enumerate_arrow_candidates(base: np.ndarray, owner: np.ndarray, specs: list[dict]) -> list[dict[str, list[dict]]]:
    h, w = base.shape
    all_candidates: list[dict[str, list[dict]]] = []
    for idx, spec in enumerate(specs):
        grouped = {name: [] for name in ARROW_INFO}
        hits = _boundary_hits(spec)
        for name, boundary_cells in hits.items():
            info = ARROW_INFO[name]
            dr, dc = info["vec"]
            tip_off_r, tip_off_c = info["tip"]
            offsets = np.argwhere(info["patch"])
            for hit_r, hit_c in boundary_cells:
                for gap in range(ARROW_GAP_RANGE[0], ARROW_GAP_RANGE[1] + 1):
                    tip_r = int(hit_r) - dr * (gap + 1)
                    tip_c = int(hit_c) - dc * (gap + 1)
                    box_r0 = tip_r - int(tip_off_r)
                    box_c0 = tip_c - int(tip_off_c)
                    cells = [(box_r0 + int(rr), box_c0 + int(cc)) for rr, cc in offsets]
                    if not all(0 <= r < h and 0 <= c < w for r, c in cells):
                        continue
                    if any(int(base[r, c]) != BG for r, c in cells):
                        continue
                    if _trace_target(owner, (tip_r, tip_c), (dr, dc)) != idx:
                        continue
                    grouped[name].append(
                        {
                            "direction": name,
                            "tip": (tip_r, tip_c),
                            "cells": cells,
                        }
                    )
        all_candidates.append(grouped)
    return all_candidates


def _fits_arrow(candidate: dict, arrow_block: np.ndarray) -> bool:
    return all(not arrow_block[r, c] for r, c in candidate["cells"])


def _place_arrows(base: np.ndarray, candidates: list[dict[str, list[dict]]]) -> tuple[np.ndarray, list[int]] | None:
    h, w = base.shape
    input_grid = base.copy()
    votes = [0] * len(candidates)
    arrow_mask = np.zeros((h, w), dtype=bool)
    arrow_block = np.zeros((h, w), dtype=bool)

    total_options = sum(sum(len(v) for v in grouped.values()) for grouped in candidates)
    if total_options < 2:
        return None

    num_arrows = random.randint(2, min(MAX_ARROWS, total_options))
    mode = random.choice(("balanced", "focused"))

    for _ in range(num_arrows):
        feasible: list[tuple[int, dict[str, list[dict]]]] = []
        for obj_idx, grouped in enumerate(candidates):
            valid_grouped = {
                name: [cand for cand in lst if _fits_arrow(cand, arrow_block)]
                for name, lst in grouped.items()
            }
            valid_grouped = {name: lst for name, lst in valid_grouped.items() if lst}
            if valid_grouped:
                feasible.append((obj_idx, valid_grouped))
        if not feasible:
            return None

        if mode == "balanced":
            min_votes = min(votes[obj_idx] for obj_idx, _ in feasible)
            feasible = [(obj_idx, grouped) for obj_idx, grouped in feasible if votes[obj_idx] == min_votes]

        obj_idx, grouped = random.choice(feasible)
        direction = random.choice(list(grouped))
        candidate = random.choice(grouped[direction])
        for r, c in candidate["cells"]:
            input_grid[r, c] = ARROW
            arrow_mask[r, c] = True
        arrow_block = _dilate8(arrow_mask)
        votes[obj_idx] += 1

    if max(votes) == 0:
        return None
    return input_grid, votes


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1] - 1)
    w = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1] - 1)
    obj_count = random.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] - 1)
    colors = random.sample(OBJECT_COLORS, obj_count)

    specs = []
    for color in colors:
        size = random.choice(OBJECT_SIZE_OPTIONS)
        specs.append({"color": color, "size": size, "mask": _random_connected_mask(size)})

    if not _place_objects(h, w, specs):
        return None

    base, owner = _render_objects(h, w, specs)
    candidates = _enumerate_arrow_candidates(base, owner, specs)
    placed = _place_arrows(base, candidates)
    if placed is None:
        return None

    inp, votes = placed
    if len({v % 4 for v in votes}) == 1:
        return None

    try:
        out = main(inp)
    except AssertionError:
        return None
    if out.shape != inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _attempt_generate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid arrow-rotation puzzle")
