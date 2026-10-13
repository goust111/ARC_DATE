from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, coords_where, in_bounds


def _unit_step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "red markers must be queen-aligned"
    sr = 0 if dr == 0 else dr // abs(dr)
    sc = 0 if dc == 0 else dc // abs(dc)
    steps = max(abs(dr), abs(dc))
    assert steps > 0, "expected two distinct red markers"
    return int(sr), int(sc), int(steps)


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc, steps = _unit_step(a, b)
    r0, c0 = a
    return [(int(r0 + i * sr), int(c0 + i * sc)) for i in range(steps + 1)]


def _orbit_cells(
    r: int, c: int, a2: int, b2: int, shape: tuple[int, int]
) -> tuple[tuple[int, int], ...] | None:
    pts = {
        (int(r), int(c)),
        (int(r), int(a2 - c)),
        (int(b2 - r), int(c)),
        (int(b2 - r), int(a2 - c)),
    }
    if not all(in_bounds(shape, rr, cc) for rr, cc in pts):
        return None
    return tuple(sorted((int(rr), int(cc)) for rr, cc in pts))


def _bbox_area(cells: set[tuple[int, int]]) -> int:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return int((max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1))


def _evaluate_axes(
    grid: np.ndarray, blue_set: set[tuple[int, int]], a2: int, b2: int
) -> dict | None:
    closure: set[tuple[int, int]] = set()
    support = 0
    for r, c in blue_set:
        orbit = _orbit_cells(r, c, a2, b2, grid.shape)
        if orbit is None:
            return None
        for rr, cc in orbit:
            if int(grid[rr, cc]) == Color.RED:
                return None
        closure.update(orbit)

        vr = (r, a2 - c)
        hr = (b2 - r, c)
        br = (b2 - r, a2 - c)
        support += int(vr in blue_set) + int(hr in blue_set) + int(br in blue_set)

    holes = {cell for cell in closure if int(grid[cell]) == Color.BLACK}
    score = (len(holes), -support, len(closure), _bbox_area(closure))
    return {
        "a2": int(a2),
        "b2": int(b2),
        "closure": frozenset(closure),
        "holes": frozenset(holes),
        "support": int(support),
        "score": score,
    }


def _infer_full_mask(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    blue_coords = coords_where(grid == Color.BLUE)
    assert blue_coords.size > 0, "expected a blue figure"
    blue_set = {tuple(map(int, rc)) for rc in blue_coords}

    candidates: list[dict] = []
    h, w = grid.shape
    for a2 in range(0, 2 * (w - 1) + 1):
        for b2 in range(0, 2 * (h - 1) + 1):
            cand = _evaluate_axes(grid, blue_set, a2, b2)
            if cand is not None:
                candidates.append(cand)

    assert candidates, "failed to infer a valid pair of symmetry axes"
    best_key = min(cand["score"] for cand in candidates)
    best = [cand for cand in candidates if cand["score"] == best_key]

    unique_closures = {cand["closure"] for cand in best}
    assert len(unique_closures) == 1, "symmetry inference is ambiguous"

    closure = next(iter(unique_closures))
    full_mask = np.zeros(grid.shape, dtype=bool)
    for r, c in closure:
        full_mask[r, c] = True

    return full_mask, best[0]


def _compose_output(
    grid: np.ndarray, full_mask: np.ndarray, path: set[tuple[int, int]]
) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    restore_mask = full_mask & (grid != Color.BLUE) & (grid != Color.RED)
    for r, c in np.argwhere(restore_mask):
        out[int(r), int(c)] = Color.BLUE

    for r, c in path:
        if bool(restore_mask[r, c]):
            out[r, c] = Color.ORANGE
        elif int(grid[r, c]) == Color.BLUE:
            out[r, c] = Color.GREEN
        else:
            out[r, c] = Color.RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simultaneously restore a doubly symmetric blue figure and draw the red segment, using orange where the two operations overlap.

    Input structure:
    - The grid has black background, one nearly complete blue figure, and exactly two isolated red marker pixels.
    - The blue figure is symmetric across one vertical and one horizontal axis, but several blue pixels were erased to black.
    - The two red markers lie on one horizontal, vertical, or 45-degree diagonal line and are not part of the blue figure.

    Rule:
    - Infer the blue figure's vertical and horizontal reflection axes from the surviving blue pixels, then close the figure under both reflections to recover the missing blue cells.
    - Trace the inclusive straight segment between the two red markers.
    - Off the segment, recovered cells become blue.
    - On the segment, original blue cells become green, original black or red cells become red, and cells that are both recovered-by-symmetry and on the segment become orange.
    - Red endpoints stay red because they are not recovery targets.

    Key insight:
    - The output is not a pipeline: symmetry restoration and line drawing are inferred independently from the same input, and their spatial overlap creates the orange evidence that both rules must be applied together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE, Color.RED}

    red_coords = coords_where(g == Color.RED)
    assert red_coords.shape == (2, 2), "expected exactly two red markers"
    start = tuple(map(int, red_coords[0]))
    end = tuple(map(int, red_coords[1]))

    full_mask, _info = _infer_full_mask(g)
    path = set(_path_cells(start, end))
    return _compose_output(g, full_mask, path)


_GRID_MIN = 11
_GRID_MAX = 17
_OBJ_SIZE_RANGE = (6, 10)
_AREA_RANGE = (20, 35)
_HOLE_RANGE = (4, 8)
_MAX_ATTEMPTS = 500


def _local_orbit(local_r: int, local_c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (int(local_r), int(local_c)),
                (int(local_r), int(w - 1 - local_c)),
                (int(h - 1 - local_r), int(local_c)),
                (int(h - 1 - local_r), int(w - 1 - local_c)),
            }
        )
    )


def _rep_weight(rep: tuple[int, int], h: int, w: int) -> int:
    return len(_local_orbit(rep[0], rep[1], h, w))


def _mask_from_reps(reps: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in reps:
        for rr, cc in _local_orbit(r, c, h, w):
            mask[rr, cc] = True
    return mask


def _connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return len(comps) == 1


def _frontier(reps: set[tuple[int, int]], qh: int, qw: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in reps:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < qh and 0 <= nc < qw and (nr, nc) not in reps:
                out.add((int(nr), int(nc)))
    return out


def _generate_full_local_mask() -> np.ndarray | None:
    h = random.randint(*_OBJ_SIZE_RANGE)
    w = random.randint(*_OBJ_SIZE_RANGE)
    target = random.randint(*_AREA_RANGE)
    if target > h * w:
        return None

    qh = (h + 1) // 2
    qw = (w + 1) // 2
    reps = {(random.randrange(qh), random.randrange(qw))}

    current = _rep_weight(next(iter(reps)), h, w)
    if current > target:
        return None

    while current < target:
        choices = [
            rep
            for rep in _frontier(reps, qh, qw)
            if current + _rep_weight(rep, h, w) <= target
        ]
        if not choices:
            return None
        random.shuffle(choices)
        rep = choices[0]
        reps.add(rep)
        current += _rep_weight(rep, h, w)

    mask = _mask_from_reps(reps, h, w)
    if current != target or not _connected(mask):
        return None

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 4 or len(cols) < 4:
        return None
    if np.all(mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]):
        return None
    return mask


def _line_sequences(h: int, w: int) -> list[list[tuple[int, int]]]:
    lines: list[list[tuple[int, int]]] = []
    for r in range(h):
        lines.append([(r, c) for c in range(w)])
    for c in range(w):
        lines.append([(r, c) for r in range(h)])
    for d in range(-(w - 1), h):
        line = [(r, r - d) for r in range(h) if 0 <= r - d < w]
        if len(line) >= 2:
            lines.append(line)
    for s in range(h + w - 1):
        line = [(r, s - r) for r in range(h) if 0 <= s - r < w]
        if len(line) >= 2:
            lines.append(line)
    return lines


def _global_mask(local_mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = local_mask.shape
    out[top : top + h, left : left + w] = local_mask
    return out


def _line_candidates(
    full_mask: np.ndarray, a2: int, b2: int
) -> list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]]:
    h, w = full_mask.shape
    candidates = []
    for line in _line_sequences(h, w):
        mask_indices = [i for i, cell in enumerate(line) if bool(full_mask[cell])]
        if len(mask_indices) < 2:
            continue
        left_options = []
        right_options = []
        for i in range(mask_indices[0]):
            r, c = line[i]
            if c * 2 != a2 and r * 2 != b2:
                left_options.append(i)
        for i in range(mask_indices[-1] + 1, len(line)):
            r, c = line[i]
            if c * 2 != a2 and r * 2 != b2:
                right_options.append(i)
        if not left_options or not right_options:
            continue
        li = random.choice(left_options)
        ri = random.choice(right_options)
        start = line[li]
        end = line[ri]
        path = line[li : ri + 1]
        if sum(bool(full_mask[cell]) for cell in path) >= 2:
            candidates.append((start, end, path))
    return candidates


def _orbit_groups(full_mask: np.ndarray, a2: int, b2: int) -> dict[tuple[tuple[int, int], ...], list[tuple[int, int]]]:
    groups: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(full_mask):
        orbit = _orbit_cells(int(r), int(c), a2, b2, full_mask.shape)
        assert orbit is not None
        groups.setdefault(orbit, []).append((int(r), int(c)))
    return groups


def _choose_holes(
    full_mask: np.ndarray, path: list[tuple[int, int]], a2: int, b2: int
) -> set[tuple[int, int]] | None:
    mask_cells = {tuple(map(int, rc)) for rc in np.argwhere(full_mask)}
    path_mask = [cell for cell in path if cell in mask_cells]
    off_path = [cell for cell in mask_cells if cell not in set(path)]
    if len(path_mask) < 2 or len(off_path) < 1:
        return None

    groups = _orbit_groups(full_mask, a2, b2)
    num_holes = random.randint(*_HOLE_RANGE)
    if num_holes >= len(mask_cells):
        return None

    max_line_holes = min(2, len(path_mask) - 1, num_holes - 1)
    if max_line_holes < 1:
        return None

    for _ in range(150):
        k_line = random.randint(1, max_line_holes)
        line_choice = set(random.sample(path_mask, k_line))
        off_choice = set(random.sample(off_path, num_holes - k_line))
        holes = line_choice | off_choice
        if len(holes) != num_holes:
            continue
        if len(set(path_mask) - holes) < 1:
            continue
        if not (set(off_path) & holes):
            continue
        if any(len(set(cells) - holes) == 0 for cells in groups.values()):
            continue
        return holes
    return None


def _expected_from_parts(
    inp: np.ndarray, full_mask: np.ndarray, endpoints: tuple[tuple[int, int], tuple[int, int]]
) -> np.ndarray:
    return _compose_output(inp, full_mask, set(_path_cells(endpoints[0], endpoints[1])))


def _line_only(inp: np.ndarray, endpoints: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for r, c in _path_cells(endpoints[0], endpoints[1]):
        out[r, c] = Color.GREEN if int(inp[r, c]) == Color.BLUE else Color.RED
    return out


def _symmetry_only(inp: np.ndarray, full_mask: np.ndarray) -> np.ndarray:
    out = inp.copy()
    restore = full_mask & (inp == Color.BLACK)
    out[restore] = Color.BLUE
    return out


def _attempt() -> np.ndarray | None:
    local_mask = _generate_full_local_mask()
    if local_mask is None:
        return None

    obj_h, obj_w = local_mask.shape
    grid_h = random.randint(max(_GRID_MIN, obj_h + 3), _GRID_MAX)
    grid_w = random.randint(max(_GRID_MIN, obj_w + 3), _GRID_MAX)
    top = random.randint(1, grid_h - obj_h - 1)
    left = random.randint(1, grid_w - obj_w - 1)

    full_mask = _global_mask(local_mask, top, left, (grid_h, grid_w))
    a2 = 2 * left + obj_w - 1
    b2 = 2 * top + obj_h - 1

    line_options = _line_candidates(full_mask, a2, b2)
    if not line_options:
        return None
    start, end, path = random.choice(line_options)

    holes = _choose_holes(full_mask, path, a2, b2)
    if holes is None:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    visible = full_mask.copy()
    for r, c in holes:
        visible[r, c] = False
    inp[visible] = Color.BLUE
    inp[start] = Color.RED
    inp[end] = Color.RED

    expected = _expected_from_parts(inp, full_mask, (start, end))
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    inferred_mask, info = _infer_full_mask(inp)
    if not np.array_equal(inferred_mask, full_mask):
        return None
    best_matches = [
        cand
        for cand in [
            _evaluate_axes(inp, {tuple(map(int, rc)) for rc in coords_where(inp == Color.BLUE)}, aa, bb)
            for aa in range(0, 2 * (grid_w - 1) + 1)
            for bb in range(0, 2 * (grid_h - 1) + 1)
        ]
        if cand is not None and cand["score"] == info["score"]
    ]
    if len({cand["closure"] for cand in best_matches}) != 1:
        return None

    path_set = set(path)
    orange_count = int(np.sum(expected == Color.ORANGE))
    green_count = int(np.sum(expected == Color.GREEN))
    blue_only_restored = int(np.sum((expected == Color.BLUE) & (inp == Color.BLACK)))
    red_count = int(np.sum(expected == Color.RED))
    if orange_count < 1 or green_count < 1 or blue_only_restored < 1 or red_count < 2:
        return None

    if np.array_equal(expected, _line_only(inp, (start, end))):
        return None
    if np.array_equal(expected, _symmetry_only(inp, full_mask)):
        return None

    if not set(map(int, np.unique(expected))) >= {Color.BLUE, Color.RED, Color.GREEN, Color.ORANGE}:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
