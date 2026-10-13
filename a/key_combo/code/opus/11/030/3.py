import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CW_TURNS = {"U": 0, "R": 3, "D": 2, "L": 1}

SPRITE_TEMPLATES = [
    {
        "mask": np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 0, 1],
            ],
            dtype=int,
        ),
        "anchors": {"U": (2, 2), "D": (0, 1), "L": (1, 2), "R": (1, 0)},
    },
    {
        "mask": np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        "anchors": {"U": (2, 1), "D": (0, 0), "L": (1, 2), "R": (0, 0)},
    },
    {
        "mask": np.array(
            [
                [0, 1, 1],
                [1, 1, 0],
                [1, 0, 0],
            ],
            dtype=int,
        ),
        "anchors": {"U": (2, 0), "D": (0, 2), "L": (0, 2), "R": (1, 0)},
    },
]

GEN_MAX_ATTEMPTS = 400
GEN_H_RANGE = (18, 25)
GEN_W_RANGE = (18, 25)
GEN_BRANCH_COUNT_RANGE = (2, 4)
GEN_BRANCH_LEN_RANGE = (2, 3)
GEN_TRUNK_LEN_RANGE = (5, 10)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in DIRS4:
                nr, nc = cr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _color_patch(mask3: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros_like(mask3, dtype=int)
    out[mask3.astype(bool)] = color
    return out


def _scan_runs(mask: np.ndarray, axis: str) -> list[tuple[str, int, int, int, int]]:
    runs: list[tuple[str, int, int, int, int]] = []
    h, w = mask.shape
    if axis == "row":
        for r in range(h):
            c = 0
            while c < w:
                if not mask[r, c]:
                    c += 1
                    continue
                start = c
                while c + 1 < w and mask[r, c + 1]:
                    c += 1
                runs.append(("row", r, start, c, c - start + 1))
                c += 1
    else:
        for c in range(w):
            r = 0
            while r < h:
                if not mask[r, c]:
                    r += 1
                    continue
                start = r
                while r + 1 < h and mask[r + 1, c]:
                    r += 1
                runs.append(("col", c, start, r, r - start + 1))
                r += 1
    return runs


def _find_trunk(g: np.ndarray) -> tuple[str, int, int, int]:
    red = g == RED
    runs = _scan_runs(red, "row") + _scan_runs(red, "col")
    assert runs, "expected red pixels"
    longest = max(run[4] for run in runs)
    candidates = [run for run in runs if run[4] == longest]
    assert len(candidates) == 1, "expected a unique longest red straight segment"
    axis, fixed, lo, hi, _length = candidates[0]
    return axis, fixed, lo, hi


def _direction_name(dr: int, dc: int) -> str:
    if (dr, dc) == (-1, 0):
        return "U"
    if (dr, dc) == (1, 0):
        return "D"
    if (dr, dc) == (0, -1):
        return "L"
    if (dr, dc) == (0, 1):
        return "R"
    raise AssertionError("unexpected direction")


def _extract_branches(g: np.ndarray, axis: str, fixed: int, lo: int, hi: int) -> list[dict]:
    red = g == RED
    trunk = set()
    if axis == "row":
        for c in range(lo, hi + 1):
            trunk.add((fixed, c))
        checks = [(-1, 0), (1, 0)]
        attach_points = [(fixed, c) for c in range(lo, hi + 1)]
    else:
        for r in range(lo, hi + 1):
            trunk.add((r, fixed))
        checks = [(0, -1), (0, 1)]
        attach_points = [(r, fixed) for r in range(lo, hi + 1)]

    branches: list[dict] = []
    for ar, ac in attach_points:
        for dr, dc in checks:
            nr, nc = ar + dr, ac + dc
            if not _in_bounds(g.shape, nr, nc) or not red[nr, nc]:
                continue
            cells: list[tuple[int, int]] = []
            while _in_bounds(g.shape, nr, nc) and red[nr, nc] and (nr, nc) not in trunk:
                cells.append((nr, nc))
                nr += dr
                nc += dc
            assert cells, "branch must contain at least one red cell"
            branches.append(
                {
                    "attach": (ar, ac),
                    "cells": cells,
                    "end": cells[-1],
                    "dir": _direction_name(dr, dc),
                    "step": (dr, dc),
                }
            )

    assert 2 <= len(branches) <= 4, "expected 2-4 branches"
    lengths = {len(branch["cells"]) for branch in branches}
    assert len(lengths) == 1, "expected all branches to have the same length"
    return branches


def _blue_components(g: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps = _components(g == BLUE)
    coord_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for cell in comp:
            coord_to_comp[cell] = idx
    return comps, coord_to_comp


def _rotate_sprite_patch(g: np.ndarray, comp: list[tuple[int, int]], cw_turns: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(comp)
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    out = patch.copy()
    out[patch == BLUE] = BG
    rotated = np.rot90(patch, k=cw_turns)
    out[rotated == BLUE] = BLUE
    return out


def _solve_details(g: np.ndarray) -> tuple[np.ndarray, dict]:
    axis, fixed, lo, hi = _find_trunk(g)
    branches = _extract_branches(g, axis, fixed, lo, hi)
    comps, coord_to_comp = _blue_components(g)
    assert comps, "expected blue sprites"

    used: dict[int, str] = {}
    out = g.copy()
    for branch in branches:
        dr, dc = branch["step"]
        er, ec = branch["end"]
        anchor = (er + dr, ec + dc)
        assert _in_bounds(g.shape, *anchor), "sprite anchor must stay in bounds"
        assert int(g[anchor]) == BLUE, "expected a blue sprite one step beyond each branch end"
        comp_idx = coord_to_comp[anchor]
        assert comp_idx not in used, "each branch must own a distinct sprite"
        used[comp_idx] = branch["dir"]
        comp = comps[comp_idx]
        rotated_patch = _rotate_sprite_patch(g, comp, CW_TURNS[branch["dir"]])
        r0, c0, r1, c1 = _bbox(comp)
        out[r0 : r1 + 1, c0 : c1 + 1] = rotated_patch

    assert len(used) == len(comps), "every blue component must belong to one branch"
    return out, {
        "axis": axis,
        "trunk": (fixed, lo, hi),
        "branches": branches,
        "component_count": len(comps),
        "used_dirs": Counter(used.values()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each branch-end sprite according to the direction of the red branch that reaches it.

    Input structure:
    - A black background contains one connected red structure made of a longest straight trunk and several shorter perpendicular branches.
    - One blue 4-connected sprite sits one cell beyond the end of each branch.
    - All blue sprites share the same small asymmetric shape, while the red trunk/branches remain unchanged.

    Rule:
    - Find the unique longest red horizontal or vertical segment; this is the trunk.
    - For each perpendicular red branch, read its direction relative to the trunk: up, right, down, or left.
    - Map directions to clockwise rotations U->0, R->90, D->180, L->270, then rotate the adjacent blue sprite in place.
    - Keep the red structure unchanged and replace each sprite by its rotated version with the same bbox top-left corner.

    Key insight:
    - The red topology is not decoration: each branch direction encodes the rotation to apply to a separate blue object, so trunk detection and sprite transformation must be combined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out, _meta = _solve_details(g)
    return out


def _choose_spaced_positions(length: int, count: int, min_gap: int) -> list[int] | None:
    slots = list(range(1, length - 1))
    random.shuffle(slots)
    chosen: list[int] = []

    def search(pool: list[int]) -> bool:
        if len(chosen) == count:
            return True
        if len(pool) < count - len(chosen):
            return False
        pool = sorted(pool)
        order = pool[:]
        random.shuffle(order)
        for pos in order:
            next_pool = [x for x in pool if x > pos and x - pos >= min_gap]
            chosen.append(pos)
            if search(next_pool):
                return True
            chosen.pop()
        return False

    if search(slots):
        return chosen[:]
    return None


def _branch_side_counts(branch_count: int) -> tuple[int, int]:
    if branch_count == 2:
        return 1, 1
    if branch_count == 3:
        return random.choice([(1, 2), (2, 1)])
    return 2, 2


def _sprite_top_left(
    anchor_map: dict[str, tuple[int, int]], direction: str, anchor_r: int, anchor_c: int
) -> tuple[int, int]:
    lr, lc = anchor_map[direction]
    return anchor_r - lr, anchor_c - lc


def _place_sprite(
    grid: np.ndarray,
    sprite_mask: np.ndarray,
    anchor_map: dict[str, tuple[int, int]],
    direction: str,
    anchor_r: int,
    anchor_c: int,
) -> tuple[int, int, int, int]:
    r0, c0 = _sprite_top_left(anchor_map, direction, anchor_r, anchor_c)
    h, w = sprite_mask.shape
    assert h == 3 and w == 3
    assert 0 <= r0 and 0 <= c0 and r0 + h <= grid.shape[0] and c0 + w <= grid.shape[1]
    patch = _color_patch(sprite_mask, BLUE)
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    assert not np.any((patch == BLUE) & (sub != BG)), "sprite overlap"
    sub[patch == BLUE] = BLUE
    return r0, c0, r0 + h - 1, c0 + w - 1


def _boxes_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _template_is_good(mask3: np.ndarray) -> bool:
    if mask3.shape != (3, 3) or int(mask3.sum()) < 3 or int(mask3.sum()) > 5:
        return False
    if len(_components(mask3.astype(bool))) != 1:
        return False
    if not (mask3[0, :].any() and mask3[2, :].any() and mask3[:, 0].any() and mask3[:, 2].any()):
        return False
    variants = {np.rot90(mask3, k).tobytes() for k in range(4)}
    return len(variants) == 4


assert all(_template_is_good(item["mask"]) for item in SPRITE_TEMPLATES)


def _sample_layout() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    branch_count = random.randint(*GEN_BRANCH_COUNT_RANGE)
    branch_len = random.randint(*GEN_BRANCH_LEN_RANGE)
    axis = random.choice(["row", "col"])
    sprite_spec = random.choice(SPRITE_TEMPLATES)
    sprite_mask = sprite_spec["mask"]
    anchor_map = sprite_spec["anchors"]

    if branch_count >= 3:
        trunk_len_min = max(7, GEN_TRUNK_LEN_RANGE[0])
    else:
        trunk_len_min = GEN_TRUNK_LEN_RANGE[0]
    trunk_len = random.randint(trunk_len_min, GEN_TRUNK_LEN_RANGE[1])

    margin = branch_len + 3
    grid = np.zeros((h, w), dtype=int)

    if axis == "row":
        if h <= 2 * margin or w < trunk_len + 4:
            return None
        trunk_r = random.randint(margin, h - margin - 1)
        c0 = random.randint(2, w - trunk_len - 2)
        c1 = c0 + trunk_len - 1
        grid[trunk_r, c0 : c1 + 1] = RED

        up_count, down_count = _branch_side_counts(branch_count)
        up_positions = _choose_spaced_positions(trunk_len, up_count, 4)
        down_positions = _choose_spaced_positions(trunk_len, down_count, 4)
        if up_positions is None or down_positions is None:
            return None

        boxes: list[tuple[int, int, int, int]] = []
        used_attach = set()
        for rel in up_positions:
            used_attach.add(rel)
            col = c0 + rel
            for step in range(1, branch_len + 1):
                grid[trunk_r - step, col] = RED
            anchor_r, anchor_c = trunk_r - branch_len - 1, col
            box = _place_sprite(grid, sprite_mask, anchor_map, "U", anchor_r, anchor_c)
            if any(not _boxes_disjoint(box, prev) for prev in boxes):
                return None
            boxes.append(box)
        for rel in down_positions:
            if rel in used_attach:
                return None
            col = c0 + rel
            for step in range(1, branch_len + 1):
                grid[trunk_r + step, col] = RED
            anchor_r, anchor_c = trunk_r + branch_len + 1, col
            box = _place_sprite(grid, sprite_mask, anchor_map, "D", anchor_r, anchor_c)
            if any(not _boxes_disjoint(box, prev) for prev in boxes):
                return None
            boxes.append(box)
    else:
        if w <= 2 * margin or h < trunk_len + 4:
            return None
        trunk_c = random.randint(margin, w - margin - 1)
        r0 = random.randint(2, h - trunk_len - 2)
        r1 = r0 + trunk_len - 1
        grid[r0 : r1 + 1, trunk_c] = RED

        left_count, right_count = _branch_side_counts(branch_count)
        left_positions = _choose_spaced_positions(trunk_len, left_count, 4)
        right_positions = _choose_spaced_positions(trunk_len, right_count, 4)
        if left_positions is None or right_positions is None:
            return None

        boxes = []
        used_attach = set()
        for rel in left_positions:
            used_attach.add(rel)
            row = r0 + rel
            for step in range(1, branch_len + 1):
                grid[row, trunk_c - step] = RED
            anchor_r, anchor_c = row, trunk_c - branch_len - 1
            box = _place_sprite(grid, sprite_mask, anchor_map, "L", anchor_r, anchor_c)
            if any(not _boxes_disjoint(box, prev) for prev in boxes):
                return None
            boxes.append(box)
        for rel in right_positions:
            if rel in used_attach:
                return None
            row = r0 + rel
            for step in range(1, branch_len + 1):
                grid[row, trunk_c + step] = RED
            anchor_r, anchor_c = row, trunk_c + branch_len + 1
            box = _place_sprite(grid, sprite_mask, anchor_map, "R", anchor_r, anchor_c)
            if any(not _boxes_disjoint(box, prev) for prev in boxes):
                return None
            boxes.append(box)

    out, meta = _solve_details(grid)
    dirs = set(meta["used_dirs"])
    if axis == "row" and dirs != {"U", "D"}:
        return None
    if axis == "col" and dirs != {"L", "R"}:
        return None
    if np.array_equal(grid, out):
        return None
    return grid, out


def _check_generated_instance(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        solved, meta = _solve_details(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    red_cells = int(np.sum(inp == RED))
    blue_cells = int(np.sum(inp == BLUE))
    if red_cells <= blue_cells:
        return False
    if Counter(inp.flatten())[BG] <= red_cells + blue_cells:
        return False

    branches = meta["branches"]
    if len({len(branch["cells"]) for branch in branches}) != 1:
        return False
    if len(meta["used_dirs"]) != 2:
        return False
    if any(np.array_equal(inp == BLUE, np.roll(out == BLUE, shift=s, axis=a)) for a in (0, 1) for s in (-1, 1)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        sample = _sample_layout()
        if sample is None:
            continue
        inp, out = sample
        if _check_generated_instance(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
