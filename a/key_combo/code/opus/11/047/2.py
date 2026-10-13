from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    unseen = set(points)
    comps: list[set[tuple[int, int]]] = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    comp.add((nr, nc))
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _rotate_cw_cell(r: int, c: int, center_r2: int, center_c2: int) -> tuple[int, int] | None:
    rr2 = center_r2 + (2 * c - center_c2)
    cc2 = center_c2 - (2 * r - center_r2)
    if (rr2 & 1) or (cc2 & 1):
        return None
    return rr2 // 2, cc2 // 2


def _rotate_vec(dr: int, dc: int, steps: int) -> tuple[int, int]:
    steps %= 4
    for _ in range(steps):
        dr, dc = dc, -dr
    return dr, dc


def _candidate_info(
    blue: set[tuple[int, int]],
    shape: tuple[int, int],
    center_r2: int,
    center_c2: int,
) -> dict | None:
    if not blue:
        return None

    missing: set[tuple[int, int]] = set()
    preserved = 0
    for r, c in blue:
        rot = _rotate_cw_cell(r, c, center_r2, center_c2)
        if rot is None:
            return None
        rr, cc = rot
        if not _in_bounds(shape, rr, cc):
            return None
        if (rr, cc) in blue:
            preserved += 1
        else:
            missing.add((rr, cc))

    if not (3 <= len(missing) <= 5):
        return None
    if len(_components(missing)) != 1:
        return None
    if len(_components(blue)) != 1:
        return None

    complete = blue | missing
    if len(_components(complete)) != 1:
        return None

    seen: set[tuple[int, int]] = set()
    orbit_miss_counter: Counter[int] = Counter()
    fixed_blue_count = 0
    for start in complete:
        if start in seen:
            continue
        orbit = []
        cur = start
        for _ in range(4):
            if cur in orbit:
                break
            orbit.append(cur)
            seen.add(cur)
            nxt = _rotate_cw_cell(cur[0], cur[1], center_r2, center_c2)
            if nxt is None or not _in_bounds(shape, nxt[0], nxt[1]):
                return None
            cur = nxt
        orbit_set = set(orbit)
        if len(orbit_set) == 1:
            if start not in blue:
                return None
            fixed_blue_count += 1
            continue
        if len(orbit_set) != 4:
            return None
        miss_count = sum(cell in missing for cell in orbit_set)
        blue_count = sum(cell in blue for cell in orbit_set)
        if miss_count != 1 or blue_count != 3:
            return None
        orbit_miss_counter[miss_count] += 1

    if fixed_blue_count > 1:
        return None

    return {
        "center_r2": center_r2,
        "center_c2": center_c2,
        "missing": missing,
        "preserved": preserved,
    }


def _find_candidates(grid: np.ndarray) -> list[dict]:
    blue = {(int(r), int(c)) for r, c in np.argwhere(grid == BLUE)}
    h, w = grid.shape
    candidates: list[dict] = []
    for parity in (0, 1):
        for center_r2 in range(parity, 2 * h, 2):
            for center_c2 in range(parity, 2 * w, 2):
                info = _candidate_info(blue, (h, w), center_r2, center_c2)
                if info is not None:
                    candidates.append(info)
    candidates.sort(
        key=lambda item: (
            -item["preserved"],
            abs(item["center_r2"] - (h - 1)),
            abs(item["center_c2"] - (w - 1)),
            item["center_r2"],
            item["center_c2"],
        )
    )
    return candidates


def _ray_path(
    shape: tuple[int, int],
    blockers: set[tuple[int, int]],
    source: tuple[int, int],
    center_r2: int,
    center_c2: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    r, c = source
    dr = _sign(2 * r - center_r2)
    dc = _sign(2 * c - center_c2)
    assert dr != 0 or dc != 0
    path: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while _in_bounds(shape, rr, cc) and (rr, cc) not in blockers:
        path.append((rr, cc))
        rr += dr
        cc += dc
    return (dr, dc), path


def _solve_with_candidate(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    missing = set(info["missing"])
    blockers = {(int(r), int(c)) for r, c in np.argwhere(out == BLUE)} | missing

    for r, c in missing:
        out[r, c] = RED

    sources = sorted(
        missing,
        key=lambda cell: (
            max(abs(2 * cell[0] - info["center_r2"]), abs(2 * cell[1] - info["center_c2"])),
            abs(2 * cell[0] - info["center_r2"]) + abs(2 * cell[1] - info["center_c2"]),
            cell[0],
            cell[1],
        ),
        reverse=True,
    )
    for source in sources:
        _dir, path = _ray_path(out.shape, blockers, source, info["center_r2"], info["center_c2"])
        for r, c in path:
            if out[r, c] != BLACK:
                break
            out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing arm of a 4-fold blue pattern, then use the new red pixels as outward ray sources.

    Input structure:
    - The grid is mostly black background.
    - Near the center sits one connected blue pattern with 4-fold rotational structure around one center point.
    - Exactly one rotated arm of that pattern is missing, so the blue object contains three arms and one absent sector.

    Rule:
    - Find the rotation center that makes the blue pattern an almost 90-degree rotationally symmetric figure with one missing arm.
    - Rotate the blue pixels one quarter-turn clockwise; every rotated location that lands on black is a missing completion pixel and becomes red.
    - For each red completion pixel, take the sign of its row/column offset from the center as a step direction, then extend a red ray outward along that direction through black cells until the boundary or another non-black cell stops it.
    - Keep all original blue pixels unchanged.

    Key insight:
    - The same inferred completion pixels play two roles: they reveal the missing rotational arm and, through their radial position relative to the symmetry center, encode each ray's launch direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    candidates = _find_candidates(g)
    if not candidates:
        raise AssertionError("no valid rotational completion center found")
    return _solve_with_candidate(g, candidates[0])


ARM_TEMPLATES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (0, 2), (1, 2), (-1, 2)),
    ((0, 1), (1, 1), (0, 2), (-1, 2)),
    ((0, 1), (0, 2), (1, 2), (-1, 2), (0, 3)),
    ((0, 1), (1, 1), (-1, 1), (1, 2), (-1, 2)),
    ((0, 1), (0, 2), (1, 2), (-1, 2), (1, 3)),
    ((0, 1), (1, 1), (0, 2), (-1, 2), (-1, 3)),
)

GRID_SIDE_RANGE = (13, 19)
CENTER_MARGIN = 5
GEN_RETRIES = 800


def _build_input(
    h: int,
    w: int,
    center: tuple[int, int],
    template: tuple[tuple[int, int], ...],
    missing_arm: int,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    grid = np.zeros((h, w), dtype=int)
    cr, cc = center
    full_blue = {(cr, cc)}
    missing_cells: set[tuple[int, int]] = set()
    for arm_idx in range(4):
        rotated = {_rotate_vec(dr, dc, arm_idx) for dr, dc in template}
        arm_cells = {(cr + dr, cc + dc) for dr, dc in rotated}
        if arm_idx == missing_arm:
            missing_cells = arm_cells
        else:
            full_blue |= arm_cells
    for r, c in full_blue:
        grid[r, c] = BLUE
    return grid, missing_cells


def _good_example(grid: np.ndarray, chosen_center: tuple[int, int], missing_cells: set[tuple[int, int]]) -> bool:
    candidates = _find_candidates(grid)
    if len(candidates) != 1:
        return False
    info = candidates[0]
    if (info["center_r2"], info["center_c2"]) != (2 * chosen_center[0], 2 * chosen_center[1]):
        return False
    if info["missing"] != missing_cells:
        return False

    blockers = {(int(r), int(c)) for r, c in np.argwhere(grid == BLUE)} | missing_cells
    occupied_paths: set[tuple[int, int]] = set()
    nonempty_paths = 0
    dirs_seen: set[tuple[int, int]] = set()
    for source in missing_cells:
        direction, path = _ray_path(grid.shape, blockers, source, info["center_r2"], info["center_c2"])
        if path:
            nonempty_paths += 1
            dirs_seen.add(direction)
            if path[-1][0] not in {0, grid.shape[0] - 1} and path[-1][1] not in {0, grid.shape[1] - 1}:
                return False
        for cell in path:
            if cell in occupied_paths:
                return False
            occupied_paths.add(cell)

    if nonempty_paths < 2 or len(dirs_seen) < 2:
        return False

    out = main(grid)
    red_count = int(np.count_nonzero(out == RED))
    if red_count <= len(missing_cells):
        return False
    if not np.array_equal(out[grid == BLUE], np.full(int(np.count_nonzero(grid == BLUE)), BLUE)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(*GRID_SIDE_RANGE)
        w = random.randint(*GRID_SIDE_RANGE)
        if h < 2 * CENTER_MARGIN + 1 or w < 2 * CENTER_MARGIN + 1:
            continue

        center = (
            random.randint(CENTER_MARGIN, h - CENTER_MARGIN - 1),
            random.randint(CENTER_MARGIN, w - CENTER_MARGIN - 1),
        )
        template = random.choice(ARM_TEMPLATES)
        missing_arm = random.randint(0, 3)

        grid, missing_cells = _build_input(h, w, center, template, missing_arm)
        if _good_example(grid, center, missing_cells):
            return grid

    raise RuntimeError("failed to generate a non-degenerate unique example")
