import random
from collections import deque

import numpy as np


BG = 0
TREE_COLORS = [1, 2, 3, 4]
BAND_COLORS = [5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
SIDES = ["top", "bottom", "left", "right"]
GEN_ATTEMPTS = 400
LOCAL_ATTEMPTS = 200


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _border_side(cell: tuple[int, int], h: int, w: int) -> str | None:
    r, c = cell
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    return sides[0] if len(sides) == 1 else None


def _component_edge_count(cells: list[tuple[int, int]]) -> int:
    pts = set(cells)
    edges = 0
    for r, c in pts:
        if (r + 1, c) in pts:
            edges += 1
        if (r, c + 1) in pts:
            edges += 1
    return edges


def _read_band(grid: np.ndarray, root: tuple[int, int]) -> list[int] | None:
    h, w = grid.shape
    r, c = root
    side = _border_side(root, h, w)
    if side is None:
        return None

    if side in ("top", "bottom"):
        directions = [(0, -1), (0, 1)]
    else:
        directions = [(-1, 0), (1, 0)]

    candidates: list[list[int]] = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if int(grid[nr, nc]) == BG:
            continue
        band: list[int] = []
        while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
            band.append(int(grid[nr, nc]))
            nr += dr
            nc += dc
        if len(band) >= 2:
            candidates.append(band)

    return candidates[0] if len(candidates) == 1 else None


def _extract_trees(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    trees: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for cells in _components_mask(g == color):
            if len(cells) < 2:
                continue
            border_cells = [cell for cell in cells if _border_side(cell, h, w) is not None]
            if len(border_cells) != 1:
                continue
            if _component_edge_count(cells) != len(cells) - 1:
                continue
            band = _read_band(g, border_cells[0])
            if band is None:
                continue
            trees.append(
                {
                    "color": color,
                    "cells": cells,
                    "root": border_cells[0],
                    "band": band,
                }
            )
    return trees


def _depth_map(root: tuple[int, int], cells: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    pts = set(cells)
    depths = {root: 0}
    q = deque([root])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, 10**9, 10**9):
            if (nr, nc) in pts and (nr, nc) not in depths:
                depths[(nr, nc)] = depths[(r, c)] + 1
                q.append((nr, nc))
    assert len(depths) == len(pts)
    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each tree by BFS depth from its unique edge root using the adjacent edge palette.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    trees = _extract_trees(g)
    assert trees, "expected at least one valid tree"

    h, w = g.shape
    for tree in trees:
        cells = [(int(r), int(c)) for r, c in tree["cells"]]
        root = tuple(map(int, tree["root"]))
        band = [int(c) for c in tree["band"]]
        depths = {root: 0}
        q = deque([root])
        pts = set(cells)
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) in pts and (nr, nc) not in depths:
                    depths[(nr, nc)] = depths[(r, c)] + 1
                    q.append((nr, nc))
        assert len(depths) == len(pts)
        for r, c in cells:
            out[r, c] = band[depths[(r, c)] % len(band)]

    return out


def _path_candidates(
    current: tuple[int, int],
    occupied: set[tuple[int, int]],
    box_h: int,
    box_w: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = current
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < box_h and 0 <= nc < box_w):
            continue
        if nr == 0 or (nr, nc) in occupied:
            continue
        adj = 0
        for ar, ac in DIRS4:
            if (nr + ar, nc + ac) in occupied:
                adj += 1
        if adj == 1:
            out.append((nr, nc))
    return out


def _choose_step(
    current: tuple[int, int],
    candidates: list[tuple[int, int]],
    prev_dir: tuple[int, int] | None,
) -> tuple[int, int]:
    weights: list[int] = []
    r, c = current
    for nr, nc in candidates:
        dr, dc = nr - r, nc - c
        weight = 1
        if dr == 1:
            weight += 4
        if prev_dir is not None and (dr, dc) == prev_dir:
            weight += 2
        weights.append(weight)
    return random.choices(candidates, weights=weights, k=1)[0]


def _build_path_tree(box_h: int, box_w: int, root_col: int, target_depth: int) -> set[tuple[int, int]] | None:
    root = (0, root_col)
    for _ in range(LOCAL_ATTEMPTS):
        occupied = {root}
        current = root
        prev_dir: tuple[int, int] | None = None
        for _step in range(target_depth):
            candidates = _path_candidates(current, occupied, box_h, box_w)
            if not candidates:
                break
            nxt = _choose_step(current, candidates, prev_dir)
            prev_dir = (nxt[0] - current[0], nxt[1] - current[1])
            occupied.add(nxt)
            current = nxt
        if len(occupied) == target_depth + 1:
            return occupied
    return None


def _grow_branch(
    start: tuple[int, int],
    occupied: set[tuple[int, int]],
    box_h: int,
    box_w: int,
    branch_len: int,
    side_pref: int,
) -> list[tuple[int, int]] | None:
    local_occ = set(occupied)
    current = start
    prev_dir: tuple[int, int] | None = None
    path: list[tuple[int, int]] = []
    for step in range(branch_len):
        candidates = _path_candidates(current, local_occ, box_h, box_w)
        if step == 0:
            lateral = [p for p in candidates if p[0] == current[0] and p[1] - current[1] == side_pref]
            if not lateral:
                return None
            candidates = lateral
        if not candidates:
            return None
        nxt = _choose_step(current, candidates, prev_dir)
        prev_dir = (nxt[0] - current[0], nxt[1] - current[1])
        path.append(nxt)
        local_occ.add(nxt)
        current = nxt
    return path


def _build_branching_tree(box_h: int, box_w: int, root_col: int, target_depth: int) -> set[tuple[int, int]] | None:
    if target_depth < 2:
        return None
    root = (0, root_col)
    trunk = [(d, root_col) for d in range(target_depth + 1)]
    if trunk[-1][0] >= box_h:
        return None

    for _ in range(LOCAL_ATTEMPTS):
        occupied = set(trunk)
        attachments = list(range(1, target_depth))
        random.shuffle(attachments)
        branch_count = 1 + int(target_depth >= 3 and random.random() < 0.65)
        made_branch = False
        used_attach: set[int] = set()
        for _k in range(branch_count):
            choices = [d for d in attachments if d not in used_attach and target_depth - d >= 1]
            if not choices:
                break
            attach_depth = random.choice(choices)
            used_attach.add(attach_depth)
            branch_len = random.randint(1, min(3, target_depth - attach_depth))
            sides = [-1, 1]
            random.shuffle(sides)
            branch = None
            for side_pref in sides:
                branch = _grow_branch(
                    trunk[attach_depth],
                    occupied,
                    box_h,
                    box_w,
                    branch_len,
                    side_pref,
                )
                if branch is not None:
                    break
            if branch is None:
                continue
            occupied.update(branch)
            made_branch = True
        if made_branch:
            return occupied
    return None


def _make_band(length: int, used_bands: set[tuple[int, ...]]) -> tuple[int, ...] | None:
    for _ in range(100):
        band = tuple(random.sample(BAND_COLORS, length))
        if band not in used_bands:
            return band
    return None


def _build_local_object(
    tree_color: int,
    branching: bool,
    wrap: bool,
    used_bands: set[tuple[int, ...]],
) -> dict[str, object] | None:
    for _ in range(LOCAL_ATTEMPTS):
        if wrap:
            band_len = random.randint(2, 4)
            target_depth = random.randint(band_len, band_len + 4)
        else:
            band_len = random.randint(3, 5)
            target_depth = random.randint(2, band_len - 1)

        box_h = random.randint(max(6, target_depth + 2), min(12, target_depth + 4))
        min_w = max(band_len + 4, 7 if branching else 6)
        max_w = min(12, min_w + 3)
        if min_w > max_w:
            continue
        box_w = random.randint(min_w, max_w)
        band_dir = random.choice([-1, 1])

        if branching:
            if band_dir == 1:
                lo, hi = 2, box_w - band_len - 1
            else:
                lo, hi = band_len, box_w - 3
        else:
            if band_dir == 1:
                lo, hi = 1, box_w - band_len - 1
            else:
                lo, hi = band_len, box_w - 2
        if lo > hi:
            continue
        root_col = random.randint(lo, hi)

        cells = (
            _build_branching_tree(box_h, box_w, root_col, target_depth)
            if branching
            else _build_path_tree(box_h, box_w, root_col, target_depth)
        )
        if cells is None:
            continue

        band = _make_band(band_len, used_bands)
        if band is None:
            continue

        local = np.zeros((box_h, box_w), dtype=int)
        for r, c in cells:
            local[r, c] = tree_color
        if band_dir == 1:
            for i, color in enumerate(band, start=1):
                local[0, root_col + i] = color
        else:
            for i, color in enumerate(band, start=1):
                local[0, root_col - i] = color

        trees = _extract_trees(local)
        if len(trees) != 1:
            continue
        depths = _depth_map(tuple(map(int, trees[0]["root"])), [(int(r), int(c)) for r, c in trees[0]["cells"]])
        max_depth = max(depths.values())
        if (max_depth >= band_len) != wrap:
            continue
        degrees = _tree_degrees([(int(r), int(c)) for r, c in trees[0]["cells"]])
        has_branch = any(deg >= 3 for deg in degrees.values())
        if has_branch != branching:
            continue

        used_bands.add(band)
        return {
            "grid": local,
            "branching": branching,
            "wrap": wrap,
            "band": band,
        }
    return None


def _transform_local(local: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return local.copy()
    if side == "bottom":
        return np.flipud(local)
    if side == "left":
        return np.rot90(local, 1)
    if side == "right":
        return np.rot90(local, -1)
    raise ValueError(side)


def _tree_degrees(cells: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    pts = set(cells)
    out: dict[tuple[int, int], int] = {}
    for r, c in pts:
        out[(r, c)] = sum((r + dr, c + dc) in pts for dr, dc in DIRS4)
    return out


def _non_bg_cells(arr: np.ndarray) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(arr != BG):
        cells.append((int(r), int(c), int(arr[r, c])))
    return cells


def _can_place(
    blocked: np.ndarray,
    placed: list[tuple[int, int, int]],
    base_r: int,
    base_c: int,
    h: int,
    w: int,
) -> bool:
    for r, c, _color in placed:
        gr, gc = base_r + r, base_c + c
        if not (0 <= gr < h and 0 <= gc < w):
            return False
        if blocked[gr, gc]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, placed: list[tuple[int, int, int]], base_r: int, base_c: int):
    h, w = blocked.shape
    for r, c, _color in placed:
        gr, gc = base_r + r, base_c + c
        for dr, dc in DIRS8:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True
        blocked[gr, gc] = True


def _side_offsets(side: str, obj_h: int, obj_w: int, h: int, w: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    if side == "top":
        cols = list(range(1, w - obj_w))
        random.shuffle(cols)
        offsets = [(0, c0) for c0 in cols]
    elif side == "bottom":
        cols = list(range(1, w - obj_w))
        random.shuffle(cols)
        offsets = [(h - obj_h, c0) for c0 in cols]
    elif side == "left":
        rows = list(range(1, h - obj_h))
        random.shuffle(rows)
        offsets = [(r0, 0) for r0 in rows]
    else:
        rows = list(range(1, h - obj_h))
        random.shuffle(rows)
        offsets = [(r0, w - obj_w) for r0 in rows]
    return offsets


def _gen_attempt() -> np.ndarray | None:
    n_trees = random.randint(2, 4)
    tree_colors = random.sample(TREE_COLORS, n_trees)
    used_bands: set[tuple[int, ...]] = set()

    modes = [(False, False), (True, True)]
    extras = [(False, True), (True, False)]
    random.shuffle(extras)
    while len(modes) < n_trees:
        modes.append(extras[len(modes) - 2] if len(modes) - 2 < len(extras) else random.choice(modes))
    random.shuffle(modes)

    local_objects: list[dict[str, object]] = []
    for color, (branching, wrap) in zip(tree_colors, modes):
        obj = _build_local_object(color, branching, wrap, used_bands)
        if obj is None:
            return None
        local_objects.append(obj)

    sides = random.sample(SIDES, n_trees)
    transformed: list[dict[str, object]] = []
    for obj, side in zip(local_objects, sides):
        arr = _transform_local(np.asarray(obj["grid"], dtype=int), side)
        transformed.append(
            {
                "grid": arr,
                "side": side,
                "branching": obj["branching"],
                "wrap": obj["wrap"],
            }
        )

    top_h = max((int(o["grid"].shape[0]) for o in transformed if o["side"] == "top"), default=0)
    bottom_h = max((int(o["grid"].shape[0]) for o in transformed if o["side"] == "bottom"), default=0)
    left_w = max((int(o["grid"].shape[1]) for o in transformed if o["side"] == "left"), default=0)
    right_w = max((int(o["grid"].shape[1]) for o in transformed if o["side"] == "right"), default=0)
    max_h = max(int(o["grid"].shape[0]) for o in transformed)
    max_w = max(int(o["grid"].shape[1]) for o in transformed)

    min_h = max(18, top_h + bottom_h + 4, max_h + 2)
    min_w = max(18, left_w + right_w + 4, max_w + 2)
    if min_h > 30 or min_w > 30:
        return None
    h = random.randint(min_h, min(30, min_h + 5))
    w = random.randint(min_w, min(30, min_w + 5))

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    order = sorted(transformed, key=lambda o: int(np.count_nonzero(o["grid"])), reverse=True)

    for obj in order:
        arr = np.asarray(obj["grid"], dtype=int)
        obj_h, obj_w = arr.shape
        offsets = _side_offsets(str(obj["side"]), obj_h, obj_w, h, w)
        if not offsets:
            return None
        placed_cells = _non_bg_cells(arr)
        placed = False
        for base_r, base_c in offsets:
            if not _can_place(blocked, placed_cells, base_r, base_c, h, w):
                continue
            for r, c, color in placed_cells:
                grid[base_r + r, base_c + c] = color
            _mark_blocked(blocked, placed_cells, base_r, base_c)
            placed = True
            break
        if not placed:
            return None

    trees = _extract_trees(grid)
    if len(trees) != n_trees:
        return None
    if len({tuple(int(x) for x in tree["band"]) for tree in trees}) != n_trees:
        return None

    branch_flags = []
    wrap_flags = []
    for tree in trees:
        cells = [(int(r), int(c)) for r, c in tree["cells"]]
        degrees = _tree_degrees(cells)
        branch_flags.append(any(deg >= 3 for deg in degrees.values()))
        depths = _depth_map(tuple(map(int, tree["root"])), cells)
        wrap_flags.append(max(depths.values()) >= len(tree["band"]))

    if not any(branch_flags) or all(branch_flags):
        return None
    if not any(wrap_flags) or all(wrap_flags):
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid depth-recolor tree puzzle")
