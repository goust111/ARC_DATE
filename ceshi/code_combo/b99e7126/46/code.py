# Combo variant of b99e7126 (index 46)
# combo_spec: D6/2+D7/2
# Name: 渐变铺贴 + 异常是多数

# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    The 29x29 canvas still holds a 7x7 lattice of 3x3 cell interiors separated by 1-pixel dividers.
#    The original task had one minority cell pattern M repeated on a subset of active coarse cells, and
#    one uniform majority cell pattern A everywhere else. Here A is still the global majority, but the
#    "correct" cells are no longer all identical. We pick one base 3x3 motif B whose dominant color forms
#    a 3x3 boolean mask. Inside one hidden 3x3 coarse block, mask=True cells should contain a gradient
#    family derived from B: along either rows or columns, the motif rotates by 90 degrees each step
#    (phase 0/1/2 inside the 3-cell span). Input shows only 3-4 such normal cells; every other cell in
#    the whole 7x7 board is the anomaly tile A. Output fills the missing mask=True cells with the rotated
#    normal motifs and leaves mask=False cells as A. Compared with the original rule, the macro mask still
#    comes from a cell's own interior, but the active coarse cells are no longer copies of one pattern:
#    they follow a directional gradient.
# b) Complexity calibration:
#    A literal reading of D7/2 ("exactly one normal cell") would make D6/2 unobservable, because one cell
#    cannot display a direction of gradual change. The fused design therefore treats the "normal" object as
#    a small minority family generated from one base motif B, while anomaly tile A remains the overwhelming
#    majority on the board. The gradient is constrained to one 3x3 coarse block and to only two directions
#    (row-gradient or column-gradient), so the task stays inferable from 2-4 examples without exploding the
#    search space.
# c) Broken assumptions from the original solver:
#    - `minority_pat = min(counts, ...)` no longer identifies the source, because the minority family now
#      contains up to three rotated patterns rather than one exact pattern.
#    - `existing[i, j] = cell_pat[i][j] == minority_pat` is invalid for the same reason.
#    - `target_pat = minority if mask else majority` also breaks: active coarse cells must be filled with
#      different rotations depending on row/column phase.
#    The new solver instead assumes: one exact majority anomaly tile A; all non-A cells lie inside one 3x3
#    coarse block; those non-A cells are rotations of one canonical base motif B; and their local row/col
#    phases determine which rotation appears. Generator explicitly constructs and verifies these assumptions.
# d) Placement legality:
#    The hidden 3x3 macro-block is placed at offsets 0..4 inside the 7x7 cell lattice, so it never crosses
#    the border. There is only one motif per cell interior, so no overlap can occur. The output-first
#    generator builds the full solved board first, then downgrades some active cells back to anomaly tile A
#    to create the input.
# e) Visual observability:
#    The gradient is made human-visible by forcing the base mask to have active cells in every row and every
#    column, and by forcing the input subset to span at least two gradient phases. Thus the few normal cells
#    already suggest either "same motif but rotated down the rows" or "same motif but rotated across the
#    columns", while the dominant color inside B still reveals the macro mask structure.
# f) Visual cleanliness:
#    The board uses only four colors: divider, shared background within motifs, normal-exclusive, and
#    anomaly-exclusive. The anomaly tile is uniform across the board, so the sparse normal family stands out
#    clearly rather than mixing many unrelated textures.
# g) Color inferability:
#    No rule depends on numeric color ids. The only color-based reading is "dominant color inside the normal
#    base motif is the active color", which is visible because generator restricts the active count to 6 or 7.
# h) Anti-degeneration guard:
#    Samples are rejected unless the input subset yields exactly one `(direction, placement, base motif)`
#    solution under the new solver. They are also rejected unless at least two gradient phases are visible in
#    the input, which prevents collapse into the old "copy one minority tile" behavior.
# i) New solver assumptions and generator guarantees:
#    - The anomaly tile is the unique most frequent exact cell pattern.
#    - Every non-anomaly cell belongs to the same 3x3 macro-block.
#    - All non-anomaly cells equal `rot90(base, phase)` where `phase` is local row or column index.
#    - The canonical base motif has four distinct quarter-turns and a unique dominant color.
#    `_gen_attempt()` checks all of these indirectly via `main(inp) == out` and directly via subset filters.

from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import paste


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """
    Infer a regular cell grid separated by full rows/cols of the same line color.
    Returns (step, row_divs, col_divs) where:
      - row_divs/col_divs are divider indices
      - step is the spacing between dividers (cell size including divider).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "divider step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be (k*step + 1)"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    return int(step), row_divs, col_divs


def _extract_cells(g: np.ndarray, step: int) -> tuple[int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    n = (g.shape[0] - 1) // step
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = tuple(arr.ravel().tolist())
            if key not in patterns:
                patterns[key] = arr
            row.append(key)
        cell_pat.append(row)
    return n, inner, patterns, cell_pat


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(np.asarray(arr, dtype=int).ravel().tolist())


def _dominant_mask(arr: np.ndarray) -> tuple[int, np.ndarray]:
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(-cnts)
    assert cnts[order[0]] > cnts[order[1]], "dominant color must be unique"
    color = int(vals[order[0]])
    return color, arr == color


def _phase_index(orientation: str, local_i: int, local_j: int) -> int:
    return local_i if orientation == "row" else local_j


def _candidate_solutions(
    cell_pat: list[list[tuple[int, ...]]],
    patterns: dict[tuple[int, ...], np.ndarray],
    majority_pat: tuple[int, ...],
) -> list[tuple[str, int, int, np.ndarray, np.ndarray]]:
    n = len(cell_pat)
    existing: list[tuple[int, int, tuple[int, ...]]] = [
        (i, j, cell_pat[i][j])
        for i in range(n)
        for j in range(n)
        if cell_pat[i][j] != majority_pat
    ]
    assert existing, "need at least one non-majority cell"

    candidates: list[tuple[str, int, int, np.ndarray, np.ndarray]] = []
    for orientation in ("row", "col"):
        for top in range(n - 3 + 1):
            for left in range(n - 3 + 1):
                if any(not (top <= i < top + 3 and left <= j < left + 3) for i, j, _ in existing):
                    continue
                base_arr: np.ndarray | None = None
                ok = True
                for i, j, pat in existing:
                    local_i = i - top
                    local_j = j - left
                    phase = _phase_index(orientation, local_i, local_j)
                    candidate = np.rot90(patterns[pat], -phase)
                    if base_arr is None:
                        base_arr = candidate
                    elif not np.array_equal(base_arr, candidate):
                        ok = False
                        break
                if not ok or base_arr is None:
                    continue
                _, mask = _dominant_mask(base_arr)
                if mask.shape != (3, 3):
                    continue
                for i, j, pat in existing:
                    local_i = i - top
                    local_j = j - left
                    phase = _phase_index(orientation, local_i, local_j)
                    if not mask[local_i, local_j]:
                        ok = False
                        break
                    expected = np.rot90(base_arr, phase)
                    if pat != _pattern_key(expected):
                        ok = False
                        break
                if ok:
                    candidates.append((orientation, top, left, base_arr.copy(), mask.copy()))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a sparse normal gradient that is buried inside a board dominated by anomalous cells.

    Input structure:
    - The board is split into equal 3x3 cell interiors by full divider rows and columns.
    - One anomaly cell pattern fills almost the entire 7x7 lattice.
    - A few minority cells are the only normal ones; they lie inside one hidden 3x3 coarse block and are
      rotations of the same base motif, changing gradually by row or by column.

    Rule:
    - Treat the dominant color of the canonical normal motif as the active color, giving a 3x3 mask.
    - Find the unique block placement and gradient direction such that every visible normal cell sits on a
      mask-active coarse position and matches the required 90-degree rotation for its local row/column phase.
    - Fill every remaining mask-active coarse cell with the corresponding rotated normal motif, and fill every
      mask-inactive coarse cell with the global anomaly tile.

    Key insight:
    - The source is no longer an exact repeated minority tile. Instead, the rare cells form a small
      direction-dependent family generated from one canonical motif.
    - Solving therefore requires two coupled inferences: anomaly is the majority background, and the few
      non-background cells reveal both the self-similar mask and the row/column gradient rule.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n, inner, patterns, cell_pat = _extract_cells(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) >= 2, "need anomaly background plus normal minority family"

    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    majority_pat = ordered[0][0]
    assert ordered[0][1] > ordered[1][1], "anomaly tile must be the unique majority"
    majority_arr = patterns[majority_pat]

    candidates = _candidate_solutions(cell_pat, patterns, majority_pat)
    unique: dict[tuple[str, int, int, tuple[int, ...]], tuple[str, int, int, np.ndarray, np.ndarray]] = {}
    for orientation, top, left, base_arr, mask in candidates:
        key = (orientation, top, left, _pattern_key(base_arr))
        unique[key] = (orientation, top, left, base_arr, mask)
    assert len(unique) == 1, "normal gradient family must determine a unique solution"
    orientation, top, left, base_arr, mask = next(iter(unique.values()))

    out = g.copy()
    for local_i in range(3):
        for local_j in range(3):
            abs_i = top + local_i
            abs_j = left + local_j
            if mask[local_i, local_j]:
                phase = _phase_index(orientation, local_i, local_j)
                target = np.rot90(base_arr, phase)
            else:
                target = majority_arr
            r0 = abs_i * step + 1
            c0 = abs_j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
    return out


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
ANOMALY_COUNTS = (4, 5)
GIVEN_CELL_SIZES = (3, 4)
COLOR_POOL = 10
MAX_RETRIES = 600


def _connected(mask: np.ndarray) -> bool:
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    seen = {coords[0]}
    frontier = [coords[0]]
    while frontier:
        r, c = frontier.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                frontier.append((nr, nc))
    return len(seen) == len(coords)


def _precompute_normal_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        active = int(mask.sum())
        if active not in ACTIVE_PIXEL_COUNTS:
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not _connected(mask):
            continue
        rots = {_pattern_key(np.rot90(mask.astype(int), k)) for k in range(4)}
        if len(rots) != 4:
            continue
        masks.append(mask)
    return masks


def _precompute_anomaly_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        active = int(mask.sum())
        if active not in ANOMALY_COUNTS:
            continue
        if not _connected(mask):
            continue
        if mask.all() or not mask.any():
            continue
        masks.append(mask)
    return masks


_NORMAL_MASKS = _precompute_normal_masks()
_ANOMALY_MASKS = _precompute_anomaly_masks()


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _pattern_from_mask(mask: np.ndarray, active_color: int, inactive_color: int) -> np.ndarray:
    return np.where(mask, active_color, inactive_color).astype(int)


def _phase_values(mask: np.ndarray, orientation: str) -> set[int]:
    values = set()
    for i, j in np.argwhere(mask):
        values.add(_phase_index(orientation, int(i), int(j)))
    return values


def _build_output(
    separator: int,
    anomaly_arr: np.ndarray,
    base_arr: np.ndarray,
    mask: np.ndarray,
    orientation: str,
    top: int,
    left: int,
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, anomaly_arr, *_cell_origin(i, j))
    for local_i in range(MASK_SIZE):
        for local_j in range(MASK_SIZE):
            if not mask[local_i, local_j]:
                continue
            phase = _phase_index(orientation, local_i, local_j)
            paste(grid, np.rot90(base_arr, phase), *_cell_origin(top + local_i, left + local_j))
    return grid


def _build_input(
    separator: int,
    anomaly_arr: np.ndarray,
    base_arr: np.ndarray,
    orientation: str,
    top: int,
    left: int,
    subset: tuple[tuple[int, int], ...],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, anomaly_arr, *_cell_origin(i, j))
    for local_i, local_j in subset:
        phase = _phase_index(orientation, local_i, local_j)
        paste(grid, np.rot90(base_arr, phase), *_cell_origin(top + local_i, left + local_j))
    return grid


def _choose_subset(mask: np.ndarray, orientation: str, top: int, left: int, base_arr: np.ndarray, anomaly_arr: np.ndarray, separator: int):
    active = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    needed_phases = _phase_values(mask, orientation)
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid: list[tuple[tuple[int, int], ...]] = []
        for subset in combinations(active, size):
            phases = {_phase_index(orientation, i, j) for i, j in subset}
            if len(phases) < 2:
                continue
            if len(needed_phases) >= 3 and len(phases) < 3:
                continue
            rows = {i for i, _ in subset}
            cols = {j for _, j in subset}
            if len(rows) < 2 or len(cols) < 2:
                continue
            inp = _build_input(separator, anomaly_arr, base_arr, orientation, top, left, subset)
            out = _build_output(separator, anomaly_arr, base_arr, mask, orientation, top, left)
            try:
                solved = main(inp)
            except Exception:
                continue
            if np.array_equal(solved, out):
                valid.append(subset)
        if valid:
            return random.choice(valid)
    return None


def _gen_attempt():
    separator, shared, normal_exclusive, anomaly_exclusive = random.sample(range(COLOR_POOL), 4)
    mask = random.choice(_NORMAL_MASKS)
    anomaly_mask = random.choice(_ANOMALY_MASKS)
    base_arr = _pattern_from_mask(mask, normal_exclusive, shared)
    anomaly_arr = _pattern_from_mask(anomaly_mask, anomaly_exclusive, shared)

    # Keep the anomaly tile visually separate from every normal rotation.
    normal_family = {_pattern_key(np.rot90(base_arr, k)) for k in range(4)}
    if _pattern_key(anomaly_arr) in normal_family:
        return None

    orientation = random.choice(("row", "col"))
    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    out = _build_output(separator, anomaly_arr, base_arr, mask, orientation, top, left)

    subset = _choose_subset(mask, orientation, top, left, base_arr, anomaly_arr, separator)
    if subset is None:
        return None
    inp = _build_input(separator, anomaly_arr, base_arr, orientation, top, left, subset)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    info = {
        "orientation": orientation,
        "top": top,
        "left": left,
        "active_count": int(mask.sum()),
        "shown_count": len(subset),
        "shown_phases": sorted({_phase_index(orientation, i, j) for i, j in subset}),
        "normal_family_size": len({_pattern_key(np.rot90(base_arr, _phase_index(orientation, i, j))) for i, j in np.argwhere(mask)}),
    }
    return inp, out, info


def generate_input():
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid fused b99e7126 input")
