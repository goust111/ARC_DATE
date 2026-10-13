from __future__ import annotations

import random

import numpy as np


BG = 0
L_COLOR = 1
OBSTACLE_COLOR = 2
CORNERS = ("TL", "TR", "BL", "BR")
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rot90(side: int, pos: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    return int(c), int(side - 1 - r)


def _corner_spec(corner: str, side: int) -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    if corner == "TL":
        return (0, 0), ((0, 1), (1, 0))
    if corner == "TR":
        return (0, side - 1), ((0, -1), (1, 0))
    if corner == "BL":
        return (side - 1, 0), ((-1, 0), (0, 1))
    assert corner == "BR"
    return (side - 1, side - 1), ((-1, 0), (0, -1))


def _corner_l_cells(side: int, k: int, corner: str) -> set[tuple[int, int]]:
    vertex, directions = _corner_spec(corner, side)
    cells = set()
    for dr, dc in directions:
        for step in range(k):
            cells.add((vertex[0] + dr * step, vertex[1] + dc * step))
    return cells


def _all_arm_paths(side: int, k: int, corner: str) -> list[tuple[tuple[int, int], ...]]:
    vertex, directions = _corner_spec(corner, side)
    arms = []
    for dr, dc in directions:
        arms.append(tuple((vertex[0] + dr * step, vertex[1] + dc * step) for step in range(k)))

    rotated = arms
    all_arms: list[tuple[tuple[int, int], ...]] = []
    for _ in range(4):
        all_arms.extend(rotated)
        rotated = [tuple(_rot90(side, cell) for cell in arm) for arm in rotated]
    return all_arms


def _scaled_coord(coord: tuple[int, int], in_side: int, out_side: int) -> tuple[int, int]:
    r, c = coord
    scale_num = out_side - 1
    scale_den = in_side - 1
    return (int((r * scale_num) // scale_den), int((c * scale_num) // scale_den))


def _input_l_cells(in_side: int, k: int, corner: str) -> set[tuple[int, int]]:
    if corner == "TL":
        return {(0, c) for c in range(k)} | {(r, 0) for r in range(k)}
    if corner == "TR":
        return {(0, c) for c in range(in_side - k, in_side)} | {(r, in_side - 1) for r in range(k)}
    if corner == "BL":
        return {(in_side - 1, c) for c in range(k)} | {(r, 0) for r in range(in_side - k, in_side)}
    assert corner == "BR"
    return {(in_side - 1, c) for c in range(in_side - k, in_side)} | {
        (r, in_side - 1) for r in range(in_side - k, in_side)
    }


def _parse_corner(touches_top: bool, touches_bottom: bool, touches_left: bool, touches_right: bool) -> str:
    assert touches_top != touches_bottom, "L must touch exactly one horizontal border"
    assert touches_left != touches_right, "L must touch exactly one vertical border"
    if touches_top and touches_left:
        return "TL"
    if touches_top and touches_right:
        return "TR"
    if touches_bottom and touches_left:
        return "BL"
    return "BR"


def _detect_scene(grid: np.ndarray) -> tuple[int, int, str, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    n = int(g.shape[0])
    assert n >= 5, "expected a non-trivial square grid"
    assert set(np.unique(g)).issubset({BG, L_COLOR, OBSTACLE_COLOR}), "unexpected colors present"

    blue_mask = g == L_COLOR
    blue_box = _bbox(blue_mask)
    assert blue_box is not None, "missing blue L"
    r0, c0, r1, c1 = blue_box
    k = int(r1 - r0 + 1)
    assert k == int(c1 - c0 + 1), "blue L bbox must be square"
    assert 3 <= k <= 5, "L arm length must be in 3..5"

    corner = _parse_corner(r0 == 0, r1 == n - 1, c0 == 0, c1 == n - 1)
    expected = np.zeros_like(blue_mask, dtype=bool)
    edge_row = r0 if r0 == 0 else r1
    edge_col = c0 if c0 == 0 else c1
    expected[edge_row, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, edge_col] = True
    assert np.array_equal(blue_mask, expected), "blue pixels must form a one-cell-thick corner L"

    obstacles = [tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE_COLOR)]
    assert 1 <= len(obstacles) <= 3, "expected 1..3 red obstacles"
    corners = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    l_cells = _input_l_cells(n, k, corner)
    assert not any(pos in corners for pos in obstacles), "obstacles cannot be on input corners"
    assert not any(pos in l_cells for pos in obstacles), "obstacles cannot overlap the input L"
    return n, k, corner, obstacles


def _render_output(in_side: int, k: int, corner: str, obstacles: list[tuple[int, int]]) -> np.ndarray:
    out_side = 2 * k - 1
    arms = _all_arm_paths(out_side, k, corner)
    out = np.full((out_side, out_side), BG, dtype=int)
    for arm in arms:
        for r, c in arm:
            out[r, c] = L_COLOR

    mapped = {_scaled_coord(pos, in_side, out_side) for pos in obstacles}
    for arm in arms:
        hit_index = next((idx for idx, cell in enumerate(arm) if cell in mapped), None)
        if hit_index is None:
            continue
        for r, c in arm[hit_index:]:
            out[r, c] = BG

    for r, c in mapped:
        out[r, c] = OBSTACLE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a corner L into a four-rotation pinwheel and truncate its arms where scaled obstacles land.

    Input structure:
    - The input is a square black background grid containing one blue one-cell-thick L and 1-3 red obstacle pixels.
    - The blue L touches exactly one grid corner, with equal horizontal and vertical arm lengths.
    - The red pixels are not on the L and not on the input corner cells.

    Rule:
    - Detect the blue L corner and its arm length k, then create a square output canvas of side 2k-1.
    - Recreate the same L in the corresponding output corner and rotate it by 90, 180, and 270 degrees about the output center to obtain four blue corner Ls.
    - Map each red obstacle by scaling its row and column with factor (2k-2)/(n-1), place it on the output, and for every rotated L arm scan from the arm's corner toward the center; from the first encountered red cell onward, erase the rest of that arm.

    Key insight:
    - The input L simultaneously fixes the enlarged canvas size and the rotation seed, while the red pixels only matter after that rotation because they clip the specific arm pixels reached in the rotated geometry.
    """
    in_side, k, corner, obstacles = _detect_scene(np.asarray(grid, dtype=int))
    return _render_output(in_side, k, corner, obstacles)


def _candidate_obstacles(in_side: int, k: int, corner: str) -> list[dict]:
    out_side = 2 * k - 1
    arms = _all_arm_paths(out_side, k, corner)
    hits_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for arm_idx, arm in enumerate(arms):
        for step, cell in enumerate(arm):
            hits_by_cell.setdefault(cell, []).append((arm_idx, step))

    l_cells = _input_l_cells(in_side, k, corner)
    input_corners = {(0, 0), (0, in_side - 1), (in_side - 1, 0), (in_side - 1, in_side - 1)}
    candidates = []
    for r in range(in_side):
        for c in range(in_side):
            pos = (r, c)
            if pos in l_cells or pos in input_corners:
                continue
            mapped = _scaled_coord(pos, in_side, out_side)
            if mapped not in hits_by_cell:
                continue
            candidates.append(
                {
                    "input": pos,
                    "output": mapped,
                    "hits": tuple(hits_by_cell[mapped]),
                    "min_step": min(step for _, step in hits_by_cell[mapped]),
                }
            )
    return candidates


def _pick_input_size(k: int) -> int:
    out_side = 2 * k - 1
    sizes = list(range(max(5, k + 1), out_side + 1))
    weights = [4 if size < out_side else 1 for size in sizes]
    return random.choices(sizes, weights=weights, k=1)[0]


def _select_obstacles(candidates: list[dict], k: int) -> list[tuple[int, int]] | None:
    if not candidates:
        return None
    shared = [cand for cand in candidates if len(cand["hits"]) == 2]
    for _ in range(200):
        count = random.choices((1, 2, 3), weights=(3, 4, 3), k=1)[0]
        chosen: list[dict] = []
        used_outputs: set[tuple[int, int]] = set()
        used_arms: set[int] = set()

        if shared and random.random() < 0.45:
            first = random.choice(shared)
            chosen.append(first)
            used_outputs.add(first["output"])
            used_arms.update(arm_idx for arm_idx, _ in first["hits"])

        while len(chosen) < count:
            pool = [
                cand
                for cand in candidates
                if cand["output"] not in used_outputs
                and used_arms.isdisjoint(arm_idx for arm_idx, _ in cand["hits"])
            ]
            if not pool:
                break
            weights = [
                1 + 2 * (k - 1 - cand["min_step"]) + (2 if len(cand["hits"]) == 2 else 0)
                for cand in pool
            ]
            nxt = random.choices(pool, weights=weights, k=1)[0]
            chosen.append(nxt)
            used_outputs.add(nxt["output"])
            used_arms.update(arm_idx for arm_idx, _ in nxt["hits"])

        if len(chosen) != count:
            continue
        if not any(cand["min_step"] < k - 1 for cand in chosen):
            continue
        return [cand["input"] for cand in chosen]
    return None


def _build_input(in_side: int, k: int, corner: str, obstacles: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((in_side, in_side), BG, dtype=int)
    for r, c in _input_l_cells(in_side, k, corner):
        grid[r, c] = L_COLOR
    for r, c in obstacles:
        grid[r, c] = OBSTACLE_COLOR
    return grid


def _arm_cells(out_side: int, k: int, corner: str) -> set[tuple[int, int]]:
    return {cell for arm in _all_arm_paths(out_side, k, corner) for cell in arm}


def _validate_generated(inp: np.ndarray, k: int, corner: str, obstacles: list[tuple[int, int]]) -> bool:
    in_side = int(inp.shape[0])
    out_side = 2 * k - 1
    out = main(inp)
    expected = _render_output(in_side, k, corner, obstacles)
    if not np.array_equal(out, expected):
        return False

    baseline = _render_output(in_side, k, corner, [])
    if np.array_equal(out, baseline):
        return False

    mapped = [_scaled_coord(pos, in_side, out_side) for pos in obstacles]
    if len(set(mapped)) != len(mapped):
        return False
    arm_cells = _arm_cells(out_side, k, corner)
    if not all(pos in arm_cells for pos in mapped):
        return False

    for obstacle in obstacles:
        reduced_obstacles = [pos for pos in obstacles if pos != obstacle]
        reduced_out = _render_output(in_side, k, corner, reduced_obstacles)
        if np.array_equal(reduced_out, out):
            return False

    for pos in mapped:
        if pos[0] in (0, out_side - 1) and pos[1] in (0, out_side - 1):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        k = random.randint(3, 5)
        corner = random.choice(CORNERS)
        in_side = _pick_input_size(k)
        candidates = _candidate_obstacles(in_side, k, corner)
        obstacles = _select_obstacles(candidates, k)
        if obstacles is None:
            continue
        inp = _build_input(in_side, k, corner, obstacles)
        if _validate_generated(inp, k, corner, obstacles):
            return inp
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
