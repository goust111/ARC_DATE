# Combo variant of 1818057f (index 14)
# combo_spec: D3/1+D12/1
# Name: 不完整L型补全为完整十字

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Original task: detect an already-complete 5-cell '+' and recolor those existing 5 cells teal.
#    - New combo task: detect an incomplete cross whose visible part is an exact thin L:
#      the center is fg, exactly two orthogonal neighbors are fg, and they form a right angle.
#      Output turns that 3-cell elbow into a full 5-cell '+' by adding the two missing arms
#      and coloring the whole completed cross teal.
#    - Local example:
#        input              original solver           combo solver
#          .Y.                 unchanged                .8.
#          .YY             (not a full '+')             888
#          ...                                        .8.
#      The changed stage is both the motif test and the write step: we no longer require
#      four existing arms, and the output includes two newly created pixels.
# b) Complexity calibration:
#    - Requiring exactly two orthogonal fg neighbors keeps the target visibly distinct from
#      straight runs, T-junctions, and full pluses.
#    - The generator repeats all four elbow rotations across samples so the human-visible
#      rule is "complete corners into symmetric crosses", not a one-off orientation trick.
# c) Implicit-assumption audit against the original main():
#    - The old `all four neighbors are fg` predicate is invalid here because targets are
#      intentionally incomplete.
#    - The old recolor-only write path assumed every output cell already existed in input;
#      the combo task must also write teal into two bg cells.
#    - Reading from the original grid copy is now even more important: once one elbow is
#      completed to a plus, that new plus must not create secondary detections in the same pass.
# d) Placement legality:
#    - Centers are kept at least two cells from the border so the full repaired '+' and the
#      protected local neighborhood fit in bounds.
#    - Motif footprints, protected diagonals, and missing-arm slots are reserved so motifs do
#      not overlap and distractors cannot occupy cells that the solver is supposed to infer.
# e) Visual observability:
#    - Across examples, the only edited regions are bent 3-cell corners that become symmetric
#      teal crosses. Two new teal pixels appear exactly in the absent orthogonal directions.
#    - This is legible from a few samples without relying on color-id arithmetic.
# f) Visual neatness:
#    - Each motif stays thin near its center: diagonals and missing-arm positions remain empty
#      in the input, while optional extensions continue only beyond the two present arms.
#    - Distractors are mostly straight segments, so the grid remains readable instead of noisy.
# g) Color/change inference:
#    - The palette stays close to the original: red bg, yellow fg, teal output highlight.
#      The color change is tied to a structural repair event, not to absolute color values.
# h) Non-degeneration target:
#    - Valid samples must contain at least one exact L site, zero pre-existing plus centers,
#      and at least two bg->teal writes per repaired motif. That guarantees the task is not
#      collapsing back to the original "recolor existing plus" behavior.
# i) Generator assumptions required by the new main():
#    - `main()` assumes each intended site is an interior exact elbow and that the two missing
#      orthogonal cells are available to paint. `generate_input()` enforces this by protecting
#      those cells, rejecting accidental extra elbow sites, and re-validating `main(inp)`.

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ELBOW_DIRS = [
    ((-1, 0), (0, -1)),
    ((-1, 0), (0, 1)),
    ((1, 0), (0, -1)),
    ((1, 0), (0, 1)),
]
ELBOW_DIR_SET = {tuple(sorted(dirs)) for dirs in ELBOW_DIRS}
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SEGMENT_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
]


def _norm_dirs(dirs) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(dr), int(dc)) for dr, dc in dirs))


def _find_l_sites(grid: np.ndarray, fg: int) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    hits: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != fg:
                continue
            present = [(dr, dc) for dr, dc in ORTHO_DIRS if int(grid[r + dr, c + dc]) == fg]
            if len(present) != 2:
                continue
            key = _norm_dirs(present)
            if key in ELBOW_DIR_SET:
                hits[(r, c)] = key
    return hits


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_DIRS):
                out.append((r, c))
    return out


def _expected_output(inp: np.ndarray, sites: dict[tuple[int, int], tuple[tuple[int, int], ...]]) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    for r, c in sites:
        out[r, c] = 8
        for dr, dc in ORTHO_DIRS:
            out[r + dr, c + dc] = 8
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect exact thin L-corners in one foreground color and complete each one into a teal '+'.

    Input structure:
    - bg is the most common color.
    - One foreground color (for example yellow) forms thin segments and bends on the bg; the target motifs are 3-cell L-corners, not finished crosses.

    Rule:
    - A target center is any foreground pixel with exactly two foreground orthogonal neighbors, where one is horizontal and one is vertical.
    - Recolor the center plus all four orthogonal arm positions to teal (8), which adds teal pixels into the two missing directions and overwrites the existing elbow triplet.
    - Straight segments, isolated pixels, T-junctions, and already-complete pluses remain unchanged.

    Key insight: The solver must recognize a broken rotationally symmetric cross from its asymmetric L-shaped fragment. Detection is done on the original input only, and each detected elbow is expanded to the unique full '+' centered at the same pixel.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != fg:
                continue
            present = [(dr, dc) for dr, dc in ORTHO_DIRS if int(g[r + dr, c + dc]) == fg]
            if len(present) != 2 or _norm_dirs(present) not in ELBOW_DIR_SET:
                continue
            out[r, c] = 8
            for dr, dc in ORTHO_DIRS:
                out[r + dr, c + dc] = 8
    return out


def _orientation_schedule(rng: np.random.Generator, count: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    base = list(ELBOW_DIRS)
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(out) < count:
        rng.shuffle(base)
        take = min(len(base), count - len(out))
        out.extend(base[:take])
    return out


def _motif_cells(
    center: tuple[int, int],
    present_dirs: tuple[tuple[int, int], tuple[int, int]],
    extra_lengths: tuple[int, int],
) -> set[tuple[int, int]]:
    r, c = center
    cells = {(r, c)}
    for (dr, dc), extra in zip(present_dirs, extra_lengths):
        for step in range(1, 2 + int(extra)):
            cells.add((r + dr * step, c + dc * step))
    return cells


def _protect_zone(
    center: tuple[int, int],
    present_dirs: tuple[tuple[int, int], tuple[int, int]],
) -> set[tuple[int, int]]:
    r, c = center
    keep = {(r, c)}
    keep.update((r + dr, c + dc) for dr, dc in present_dirs)
    protected = {(r + dr, c + dc) for dr, dc in ORTHO_DIRS + DIAGS}
    protected.update((r + 2 * dr, c + 2 * dc) for dr, dc in ORTHO_DIRS if (dr, dc) not in present_dirs)
    return protected - keep


def _try_add_cells(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    fg: int,
    intended: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> np.ndarray | None:
    h, w = grid.shape
    if not cells:
        return None
    for r, c in cells:
        if r < 0 or r >= h or c < 0 or c >= w:
            return None
        if int(grid[r, c]) == fg:
            return None

    trial = grid.copy()
    for r, c in cells:
        trial[r, c] = fg

    if _find_l_sites(trial, fg) != intended:
        return None
    if _find_plus_centers(trial, fg):
        return None
    return trial


def _try_add_distractor(
    inp: np.ndarray,
    cells: list[tuple[int, int]],
    fg: int,
    blocked: set[tuple[int, int]],
    intended: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> bool:
    h, w = inp.shape
    if not cells:
        return False
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells):
        return False
    if any((r, c) in blocked for r, c in cells):
        return False
    if any(int(inp[r, c]) != 2 for r, c in cells):
        return False

    trial = inp.copy()
    for r, c in cells:
        trial[r, c] = fg

    if _find_l_sites(trial, fg) != intended:
        return False
    if _find_plus_centers(trial, fg):
        return False
    if not np.array_equal(main(trial), _expected_output(trial, intended)):
        return False

    inp[:, :] = trial
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(10, 23))
    w = int(rng.integers(10, 23))
    bg, fg = 2, 4

    inp = np.full((h, w), bg, dtype=int)
    intended: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    blocked: set[tuple[int, int]] = set()

    max_k = min(8, max(1, (h * w) // 42 + 2))
    min_k = 1 if max_k == 1 else 2
    k = int(rng.integers(min_k, max_k + 1))

    for dirs in _orientation_schedule(rng, k):
        wanted = _norm_dirs(dirs)
        placed = False
        for _ in range(240):
            center = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
            if any(max(abs(center[0] - rr), abs(center[1] - cc)) < 4 for rr, cc in intended):
                continue

            extra = [int(rng.integers(0, 4)), int(rng.integers(0, 4))]
            if extra[0] == 0 and extra[1] == 0:
                extra[int(rng.integers(0, 2))] = int(rng.integers(1, 4))

            cells = _motif_cells(center, dirs, (extra[0], extra[1]))
            protect = _protect_zone(center, dirs)
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells | protect):
                continue
            if any(int(inp[r, c]) != bg for r, c in cells | protect):
                continue

            intended_next = dict(intended)
            intended_next[center] = wanted
            trial = _try_add_cells(inp, cells, fg, intended_next)
            if trial is None:
                continue

            inp = trial
            intended = intended_next
            blocked.update(protect)
            placed = True
            break
        if not placed:
            return None

    if not intended:
        return None
    if _find_l_sites(inp, fg) != intended:
        return None
    if _find_plus_centers(inp, fg):
        return None
    if not np.array_equal(main(inp), _expected_output(inp, intended)):
        return None

    total = h * w
    target_density = float(rng.uniform(0.24, 0.37))
    target_fg = int(total * target_density)

    attempts = 0
    while int(np.sum(inp != bg)) < target_fg and attempts < total * 10:
        attempts += 1
        shape = SEGMENT_SHAPES[int(rng.integers(0, len(SEGMENT_SHAPES)))]
        base_r = int(rng.integers(0, h))
        base_c = int(rng.integers(0, w))
        cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
        _try_add_distractor(inp, cells, fg, blocked, intended)

    fg_count = int(np.sum(inp != bg))
    if fg_count < max(target_fg - 3, len(intended) * 4):
        return None
    if fg_count >= total // 2:
        return None
    if _find_l_sites(inp, fg) != intended:
        return None
    if _find_plus_centers(inp, fg):
        return None

    out = _expected_output(inp, intended)
    if not np.array_equal(main(inp), out):
        return None

    added_from_bg = int(np.sum((inp == bg) & (out == 8)))
    if added_from_bg != 2 * len(intended):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
