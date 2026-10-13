from __future__ import annotations

import random

import numpy as np


RED = 2

# --------------- shared helpers ---------------


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((int(rr), int(cc)))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    return len(_components(mask)) == 1


def _erode_once(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    interior = mask.copy()
    interior &= pad[:-2, 1:-1]
    interior &= pad[2:, 1:-1]
    interior &= pad[1:-1, :-2]
    interior &= pad[1:-1, 2:]
    return interior


def _skeleton_mask(mask: np.ndarray) -> np.ndarray:
    current = np.asarray(mask, dtype=bool).copy()
    while True:
        nxt = _erode_once(current)
        if not bool(nxt.any()):
            return current
        if not _is_connected(nxt):
            return current
        current = nxt


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    norm = {(int(r - r0), int(c - c0)) for r, c in coords}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    out = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        out[int(r), int(c)] = True
    return out


def _transform_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    if random.random() < 0.5:
        pts = [(int(r), int(-c)) for r, c in pts]
    for _ in range(random.randint(0, 3)):
        pts = [(int(-c), int(r)) for r, c in pts]
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0 = min(rows)
    c0 = min(cols)
    return {(int(r - r0), int(c - c0)) for r, c in pts}


def _dilate_manhattan(coords: set[tuple[int, int]], radius: int) -> np.ndarray:
    pts: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) <= radius:
                    pts.add((int(r + dr), int(c + dc)))
    return _coords_to_mask(pts)


# --------------- main ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the final peel-stable skeleton of each 4-connected foreground object.

    Repeatedly remove every foreground pixel that touches background in the
    4-neighborhood. Stop just before the next peel would empty the object or
    split it into multiple 4-connected components. Paint the remaining pixels red.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    fg_mask = g != bg

    out = np.full_like(g, int(bg))
    for comp in _components(fg_mask):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True
        skel = _skeleton_mask(local)
        rr, cc = np.where(skel)
        out[r0 + rr, c0 + cc] = int(RED)
    return out


# --------------- generator ---------------

_GEN_ATTEMPTS = 400
_OBJECT_ATTEMPTS = 80
_PLACEMENT_ATTEMPTS = 160
_GRID_H_RANGE = (18, 29)
_GRID_W_RANGE = (18, 29)
_INPUT_COLORS = [0, 1, 3, 4, 5, 6, 7, 8, 9]


def _make_strip_core() -> set[tuple[int, int]]:
    bends = random.choices([0, 1, 2], weights=[2, 4, 3])[0]
    lengths = [random.randint(7, 13)]
    if bends >= 1:
        lengths.append(random.randint(4, 8))
    if bends == 2:
        lengths.append(random.randint(5, 10))

    dirs = [(0, 1)]
    if bends >= 1:
        dirs.append(random.choice([(1, 0), (-1, 0)]))
    if bends == 2:
        dirs.append((0, 1))

    r = 0
    c = 0
    coords = {(0, 0)}
    for (dr, dc), seg_len in zip(dirs, lengths):
        for _ in range(seg_len):
            r += dr
            c += dc
            coords.add((int(r), int(c)))
    return _transform_coords(coords)


def _make_branch_core() -> set[tuple[int, int]]:
    trunk_len = random.randint(9, 15)
    coords = {(0, c) for c in range(trunk_len)}
    style = random.choice(["single", "t_shape", "split"])

    if style == "single":
        point = random.randint(2, trunk_len - 3)
        side = random.choice([-1, 1])
        branch_len = random.randint(3, max(4, trunk_len // 2))
        for step in range(1, branch_len + 1):
            coords.add((int(side * step), int(point)))

    elif style == "t_shape":
        point = random.randint(3, trunk_len - 4)
        for side in (-1, 1):
            branch_len = random.randint(2, max(3, trunk_len // 3))
            for step in range(1, branch_len + 1):
                coords.add((int(side * step), int(point)))

    else:
        p1, p2 = sorted(random.sample(range(2, trunk_len - 2), 2))
        for point in (p1, p2):
            side = random.choice([-1, 1])
            branch_len = random.randint(3, max(4, trunk_len // 3))
            for step in range(1, branch_len + 1):
                coords.add((int(side * step), int(point)))

    return _transform_coords(coords)


def _make_object_mask(kind: str) -> np.ndarray | None:
    for _ in range(_OBJECT_ATTEMPTS):
        core = _make_strip_core() if kind == "strip" else _make_branch_core()
        radius = random.choices([1, 2], weights=[3, 2])[0]
        mask = _dilate_manhattan(core, radius)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < 18:
            continue
        skel = _skeleton_mask(mask)
        if int(skel.sum()) < 2 or int(skel.sum()) >= int(mask.sum()):
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    return None


def _reserve_zone(mask: np.ndarray, top: int, left: int, blocked: set[tuple[int, int]]) -> None:
    for r, c in np.argwhere(mask):
        gr = int(top + r)
        gc = int(left + c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((int(gr + dr), int(gc + dc)))


def _place_object(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if mh + 2 > h or mw + 2 > w:
        return False

    coords = np.argwhere(mask)
    for _ in range(_PLACEMENT_ATTEMPTS):
        top = random.randint(1, h - mh - 1)
        left = random.randint(1, w - mw - 1)
        if any((int(top + r), int(left + c)) in blocked for r, c in coords):
            continue
        for r, c in coords:
            grid[int(top + r), int(left + c)] = int(color)
        _reserve_zone(mask, top, left, blocked)
        return True
    return False


def _choose_kinds(n_objects: int) -> list[str]:
    if n_objects == 1:
        return [random.choice(["strip", "branch"])]
    if n_objects >= 2 and random.random() < 0.5:
        kinds = ["strip", "branch"] + [random.choice(["strip", "branch"]) for _ in range(n_objects - 2)]
        random.shuffle(kinds)
        return kinds
    return [random.choice(["strip", "branch"]) for _ in range(n_objects)]


def _valid_generated_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    bg = _dominant_color(inp)
    fg_mask = inp != bg
    red_mask = out == RED
    if not bool(fg_mask.any()) or not bool(red_mask.any()):
        return False
    if not np.all((out == bg) | red_mask):
        return False
    if len(_components(fg_mask)) != len(_components(red_mask)):
        return False
    if int(np.count_nonzero(fg_mask)) >= int(inp.size // 2):
        return False
    if int(np.count_nonzero(red_mask)) >= int(np.count_nonzero(fg_mask)):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_H_RANGE)
    w = random.randint(*_GRID_W_RANGE)
    bg, fg = random.sample(_INPUT_COLORS, 2)
    grid = np.full((h, w), int(bg), dtype=int)

    max_objects = 4 if min(h, w) >= 24 else 3
    n_objects = random.randint(1, max_objects)
    kinds = _choose_kinds(n_objects)

    masks: list[np.ndarray] = []
    for kind in kinds:
        mask = _make_object_mask(kind)
        if mask is None:
            return None
        masks.append(mask)

    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    blocked: set[tuple[int, int]] = set()
    for mask in masks:
        if not _place_object(grid, mask, fg, blocked):
            return None

    out = main(grid)
    if not _valid_generated_pair(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
