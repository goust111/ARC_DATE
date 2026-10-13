from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
ORANGE = 7

DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_NEIGHBORS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Mirror of arc_utils.in_bounds using canonical (r,c) coordinates."""
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    """Mirror of arc_utils.neighbors4 with explicit list output."""
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _axis_match_count(grid: np.ndarray, axis: str, index: int) -> int:
    """Count visible blue pixels whose mirror across the candidate axis is also blue."""
    h, w = grid.shape
    count = 0
    for r, c in np.argwhere(grid == BLUE):
        rr = int(r)
        cc = int(c)
        if axis == "h":
            mr, mc = 2 * index - rr, cc
        else:
            mr, mc = rr, 2 * index - cc
        if 0 <= mr < h and 0 <= mc < w and int(grid[mr, mc]) == BLUE:
            count += 1
    return count


def _best_axis(grid: np.ndarray, axis: str) -> tuple[int, list[int]]:
    n = grid.shape[0 if axis == "h" else 1]
    scores = [_axis_match_count(grid, axis, idx) for idx in range(n)]
    best = max(scores)
    winners = [idx for idx, score in enumerate(scores) if score == best]
    return winners[0], winners


def _recover_and_mark(
    grid: np.ndarray,
    horizontal_axis: int | None = None,
    vertical_axis: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, str]], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    if horizontal_axis is None:
        horizontal_axis, _ = _best_axis(g, "h")
    if vertical_axis is None:
        vertical_axis, _ = _best_axis(g, "v")

    restored = g.copy()
    recovered: list[tuple[int, int, str]] = []

    for r, c in np.argwhere(g == BLACK):
        rr = int(r)
        cc = int(c)
        hr, hc = 2 * horizontal_axis - rr, cc
        vr, vc = rr, 2 * vertical_axis - cc
        dr, dc = 2 * horizontal_axis - rr, 2 * vertical_axis - cc

        h_blue = _in_bounds(g.shape, hr, hc) and int(g[hr, hc]) == BLUE
        v_blue = _in_bounds(g.shape, vr, vc) and int(g[vr, vc]) == BLUE
        d_blue = _in_bounds(g.shape, dr, dc) and int(g[dr, dc]) == BLUE

        if not (h_blue or v_blue or d_blue):
            continue

        if h_blue and v_blue:
            kind = "B"
        elif h_blue:
            kind = "H"
        elif v_blue:
            kind = "V"
        else:
            kind = "D"

        restored[rr, cc] = BLUE
        recovered.append((rr, cc, kind))

    recovered.sort()
    paintable = restored == BLACK
    out = restored.copy()

    for r, c, kind in recovered:
        if kind == "H":
            offsets = DIAGONALS
            color = YELLOW
        elif kind == "V":
            offsets = ORTHOGONALS
            color = ORANGE
        elif kind == "B":
            offsets = EIGHT_NEIGHBORS
            color = GREEN
        else:
            offsets = []
            color = BLACK

        for dr, dc in offsets:
            nr = r + dr
            nc = c + dc
            if _in_bounds(out.shape, nr, nc) and bool(paintable[nr, nc]):
                out[nr, nc] = color

    return out, recovered, (horizontal_axis, vertical_axis)


def main(grid: np.ndarray) -> np.ndarray:
    """Restore a doubly symmetric blue shape, then mark each repaired hole by the axis that recovered it.

    Input structure:
    - A black background contains one central irregular blue object that is intended to be symmetric across one horizontal row axis and one vertical column axis.
    - Several blue cells are erased to black, creating missing pixels inside that symmetric orbit structure.

    Rule:
    - Infer the horizontal and vertical mirror axes by choosing the row index and column index with the most visible blue-to-blue mirror matches.
    - For each black cell, inspect its horizontal mirror, vertical mirror, and double-mirror orbit mates in the visible input; if any of those mates are still blue, restore the cell to blue.
    - Classify each restored cell by which single-axis mirrors are visible: horizontal-only -> H, vertical-only -> V, both -> B.
    - After restoring all holes, mark only the remaining black background around each restored cell: H paints the four diagonals yellow, V paints the four orthogonals orange, and B paints all eight neighbors green. Process restored cells in row-major order so later marks overwrite earlier marks on shared background cells.

    Key insight:
    - Every repaired pixel ends up blue, so the output geometry cannot be inferred from color alone; the neighborhood shape encodes which mirror axis was needed during the hidden symmetry-recovery step.
    """
    out, _recovered, _axes = _recover_and_mark(np.asarray(grid, dtype=int))
    return out


def _orbit4(horizontal_axis: int, vertical_axis: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r, c),
        (2 * horizontal_axis - r, c),
        (r, 2 * vertical_axis - c),
        (2 * horizontal_axis - r, 2 * vertical_axis - c),
    )


def _is_connected(shape: set[tuple[int, int]]) -> bool:
    start = next(iter(shape))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in shape and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(shape)


def _shape_orbit_reps(
    shape: set[tuple[int, int]],
    horizontal_axis: int,
    vertical_axis: int,
) -> list[tuple[int, int]]:
    reps: list[tuple[int, int]] = []
    for r in range(1, horizontal_axis):
        for c in range(1, vertical_axis):
            orbit = _orbit4(horizontal_axis, vertical_axis, r, c)
            if all(cell in shape for cell in orbit):
                reps.append((r, c))
    return reps


def _build_symmetric_shape(
    height: int,
    width: int,
    horizontal_axis: int,
    vertical_axis: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(250):
        shape: set[tuple[int, int]] = {(horizontal_axis, vertical_axis)}

        arm_up = random.randint(2, 3)
        arm_side = random.randint(2, 3)
        for d in range(1, arm_up + 1):
            shape.add((horizontal_axis - d, vertical_axis))
            shape.add((horizontal_axis + d, vertical_axis))
        for d in range(1, arm_side + 1):
            shape.add((horizontal_axis, vertical_axis - d))
            shape.add((horizontal_axis, vertical_axis + d))

        representatives = [
            (r, c)
            for r in range(1, horizontal_axis)
            for c in range(1, vertical_axis)
        ]
        random.shuffle(representatives)

        target_area = random.randint(32, 48)
        while representatives and len(shape) < target_area:
            r, c = representatives.pop()
            orbit = _orbit4(horizontal_axis, vertical_axis, r, c)
            if any(cell in shape for cell in orbit):
                continue
            touches_shape = any(
                any((nr, nc) in shape for nr, nc in _neighbors4(rr, cc))
                for rr, cc in orbit
            )
            if not touches_shape:
                continue
            shape.update(orbit)

        if not (30 <= len(shape) <= 50):
            continue
        if not _is_connected(shape):
            continue

        reps = _shape_orbit_reps(shape, horizontal_axis, vertical_axis)
        if len(reps) < 4:
            continue

        return shape, reps
    return None


def _choose_holes(
    reps: list[tuple[int, int]],
    horizontal_axis: int,
    vertical_axis: int,
) -> list[tuple[int, int]] | None:
    random.shuffle(reps)
    extra_kind = random.choice(["H", "V", "B", None])
    need = 3 + (1 if extra_kind is not None else 0)
    if len(reps) < need:
        return None

    pattern_kinds = ["H", "V", "B"] + ([extra_kind] if extra_kind is not None else [])
    holes: list[tuple[int, int]] = []

    for rep, kind in zip(reps[:need], pattern_kinds):
        r, c = rep
        a, b, c1, d = _orbit4(horizontal_axis, vertical_axis, r, c)

        if kind == "H":
            removed = random.choice([(a, c1), (b, d)])
        elif kind == "V":
            removed = random.choice([(a, b), (c1, d)])
        else:
            removed = random.choice([(a,), (b,), (c1,), (d,), (a, d), (b, c1)])

        holes.extend(removed)

    unique_holes = sorted(set(holes))
    if not (4 <= len(unique_holes) <= 8):
        return None
    return unique_holes


def _render_shape(
    height: int,
    width: int,
    shape: set[tuple[int, int]],
    holes: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for r, c in shape:
        grid[r, c] = BLUE
    for r, c in holes:
        grid[r, c] = BLACK
    return grid


def _expected_from_truth(
    inp: np.ndarray,
    full_shape: set[tuple[int, int]],
    horizontal_axis: int,
    vertical_axis: int,
) -> tuple[np.ndarray, list[tuple[int, int, str]]] | None:
    restored = inp.copy()
    visible_blue = {tuple(map(int, cell)) for cell in np.argwhere(inp == BLUE)}
    missing_blue = sorted(full_shape - visible_blue)
    recovered: list[tuple[int, int, str]] = []

    for r, c in missing_blue:
        hr, hc = 2 * horizontal_axis - r, c
        vr, vc = r, 2 * vertical_axis - c
        dr, dc = 2 * horizontal_axis - r, 2 * vertical_axis - c
        h_blue = int(inp[hr, hc]) == BLUE
        v_blue = int(inp[vr, vc]) == BLUE
        d_blue = int(inp[dr, dc]) == BLUE
        if not (h_blue or v_blue or d_blue):
            return None
        if h_blue and v_blue:
            kind = "B"
        elif h_blue:
            kind = "H"
        elif v_blue:
            kind = "V"
        else:
            kind = "D"
        recovered.append((r, c, kind))
        restored[r, c] = BLUE

    paintable = restored == BLACK
    out = restored.copy()
    for r, c, kind in recovered:
        if kind == "H":
            offsets = DIAGONALS
            color = YELLOW
        elif kind == "V":
            offsets = ORTHOGONALS
            color = ORANGE
        elif kind == "B":
            offsets = EIGHT_NEIGHBORS
            color = GREEN
        else:
            offsets = []
            color = BLACK
        for dr, dc in offsets:
            nr = r + dr
            nc = c + dc
            if _in_bounds(out.shape, nr, nc) and bool(paintable[nr, nc]):
                out[nr, nc] = color

    return out, recovered


def generate_input() -> np.ndarray:
    for _ in range(400):
        height = random.choice([13, 15, 17])
        width = random.choice([13, 15, 17])
        horizontal_axis = height // 2
        vertical_axis = width // 2

        built = _build_symmetric_shape(height, width, horizontal_axis, vertical_axis)
        if built is None:
            continue
        shape, reps = built

        holes = _choose_holes(reps[:], horizontal_axis, vertical_axis)
        if holes is None:
            continue

        inp = _render_shape(height, width, shape, holes)

        best_h, h_winners = _best_axis(inp, "h")
        best_v, v_winners = _best_axis(inp, "v")
        if best_h != horizontal_axis or best_v != vertical_axis:
            continue
        if len(h_winners) != 1 or len(v_winners) != 1:
            continue

        expected = _expected_from_truth(inp, shape, horizontal_axis, vertical_axis)
        if expected is None:
            continue
        expected_out, recovered_truth = expected
        if any(kind == "D" for _r, _c, kind in recovered_truth):
            continue
        if len(recovered_truth) != len(holes):
            continue

        counts = Counter(kind for _r, _c, kind in recovered_truth)
        if not all(counts[kind] >= 1 for kind in ("H", "V", "B")):
            continue

        solver_out, solver_recovered, solver_axes = _recover_and_mark(inp)
        if solver_axes != (horizontal_axis, vertical_axis):
            continue
        if solver_recovered != recovered_truth:
            continue
        if not np.array_equal(solver_out, expected_out):
            continue

        present_colors = set(map(int, solver_out.flat))
        if not {GREEN, YELLOW, ORANGE}.issubset(present_colors):
            continue

        if np.array_equal(inp, solver_out):
            continue

        return inp

    raise RuntimeError("generate_input failed to produce a valid sample")
