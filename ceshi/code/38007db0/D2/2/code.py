from __future__ import annotations

# Variant of 38007db0
# Dimension: D2 — 源对象：行内唯一精灵单元格 (src)
# Change: 每行中模式只出现一次的方形单元格（内含独特精灵变体）... → 旋转异常项
#
# Design analysis
# ===============
# variant_spec:
#   task: Per-row rotation-anomaly extraction from a lattice grid into a single-column lattice.
#   unchanged_geometry:
#     lattice: same 1px gridlines, square cells, 3 colors total
#     output: still a single-column lattice with the selected source cell from each row
#   changed_source_logic:
#     old: pick the cell whose full bitmap appears once in that row
#     new: pick the cell whose 180-degree rotation class appears once in that row
#   concrete_pixel_behavior:
#     - Each cell still contains a bg-padded inner sprite in one sprite color.
#     - The sprite is intentionally asymmetric under both 90 and 180 degree rotation.
#     - In one row, most cells belong to one half-turn class:
#         class A = {0 deg, 180 deg}  or  class B = {90 deg, 270 deg}
#     - Exactly one cell belongs to the other class, and that cell is copied to output.
#   small_trace:
#     - Example row with three cells: [rot0(base), rot180(base), rot90(base)].
#     - Old solver compares raw bytes:
#         rot0 != rot180 != rot90, so every cell looks "unique"; choosing the first unique cell is wrong.
#     - New solver compares canonical signatures min(cell, rot180(cell)):
#         canon(rot0) == canon(rot180), canon(rot90) is different, so rot90 is the unique class member.
#   complexity_calibration:
#     - The variant stays local to the row; no extra side-channel objects are introduced.
#     - To avoid degeneration back to the original task, the generator forces the majority family to use both
#       of its exact orientations whenever possible, so raw byte-frequency is insufficient.
#     - The number of orientation families is only two ({0,180} vs {90,270}), which is visually learnable
#       from 2-4 ARC examples without exploding the hypothesis space.
#   implicit_assumption_review:
#     - Uniform-row / uniform-column detection still works because lattice geometry is unchanged.
#     - Using the most common non-lattice color as bg still works because the cell padding/background remains bg.
#     - The old row logic `Counter(cell.tobytes())` depended on "all normal cells are exact copies";
#       that assumption is intentionally false in this variant.
#     - The new logic assumes each row has exactly two half-turn classes and one of them has count 1.
#       The generator explicitly guarantees this.
#   placement_legality:
#     - Cells are still tiled by the same lattice helper, so no overlap or boundary issues are introduced.
#     - Every row has at least 3 columns, letting the generator place a singleton odd class and a repeated class.
#   visual_observability:
#     - The sprite is forced to be non-symmetric, so 90/180 degree changes are visible.
#     - Majority rows often look like "upright + upside-down + sideways", making the sideways cell the only
#       member of the other rotation family; this is legible by eye.
#   cleanliness:
#     - No extra markers or overlays are added; the grid remains as clean as the base task.

from collections import Counter
import math
import numpy as np
from arc_utils import paste


def _half_turn_signature(arr: np.ndarray) -> bytes:
    """Canonicalize a square sprite up to 180-degree rotation."""
    a = arr.tobytes()
    b = np.rot90(arr, 2).tobytes()
    return a if a <= b else b


def _raw_unique_index(cells: list[np.ndarray]) -> int | None:
    """Reference to the original task logic; used only for degeneration checks."""
    sigs = [cell.tobytes() for cell in cells]
    cnt = Counter(sigs)
    return next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the per-row rotation anomaly into a single-column lattice.

    The lattice parsing and output framing are unchanged from 38007db0.
    The only semantic change is the row selector:
    compare cells modulo 180-degree rotation instead of exact bitmap equality.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # Gridlines appear as fully-uniform rows/cols.
    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    # Lattice color = most common value among uniform rows/cols.
    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted([r for r in row_lines if int(g[r, 0]) == int(L)])
    col_lines = sorted([c for c in col_lines if int(g[0, c]) == int(L)])

    # Cell slices are the gaps between consecutive grid lines.
    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    cell_w = int(c_slices[0][1] - c_slices[0][0])

    # Background inside cells = most common non-lattice color.
    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        class_sigs = [_half_turn_signature(cell) for cell in cells]
        class_counts = Counter(class_sigs)

        uniq_classes = [sig for sig, n in class_counts.items() if int(n) == 1]
        assert len(uniq_classes) == 1 and len(class_counts) == 2
        uniq_idx = next(j for j, sig in enumerate(class_sigs) if sig == uniq_classes[0])
        chosen_cells.append(cells[int(uniq_idx)])

    # Output: same height, single lattice column => width = cell_w + 2 (outer gridlines).
    outW = int(cell_w + 2)
    out = np.full((H, outW), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = inner.shape[0] + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


def _is_good_base(inner: np.ndarray) -> bool:
    """Require visibly asymmetric rotations and two distinct half-turn families."""
    rots = [np.rot90(inner, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rots[i], rots[j]):
                return False

    area = int(inner.size)
    if int(np.count_nonzero(rots[0] != rots[1])) < max(3, area // 4):
        return False
    if int(np.count_nonzero(rots[0] != rots[2])) < max(2, area // 5):
        return False

    return _half_turn_signature(rots[0]) != _half_turn_signature(rots[1])


def _sample_base_inner(rng: np.random.Generator, inner_size: int, bg: int, sp: int) -> np.ndarray | None:
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.85 * inner_area)

    for _ in range(200):
        fill = int(rng.integers(min_fill, max_fill + 1))
        inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        on_idx = rng.choice(inner_area, size=fill, replace=False)
        inner.flat[on_idx] = int(sp)
        if _is_good_base(inner):
            return inner
    return None


def _row_orientations(rng: np.random.Generator, n_cols: int) -> tuple[list[int], int]:
    """
    Build a row that is ambiguous under exact-byte counting but clean under half-turn classes.

    Returns:
        orientations for the row, and the index of the anomalous cell.
    """
    if bool(rng.integers(0, 2)):
        majority_family = [0, 2]
        odd_choices = [1, 3]
    else:
        majority_family = [1, 3]
        odd_choices = [2]

    majority = [majority_family[0], majority_family[1]]
    while len(majority) < n_cols - 1:
        majority.append(int(rng.choice(majority_family)))
    rng.shuffle(majority)

    odd_orientation = int(rng.choice(odd_choices))
    odd_col = int(rng.integers(0, n_cols))

    orientations: list[int] = []
    m = 0
    for c in range(n_cols):
        if int(c) == int(odd_col):
            orientations.append(odd_orientation)
        else:
            orientations.append(int(majority[m]))
            m += 1
    return orientations, odd_col


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    base_inner = _sample_base_inner(rng, cell_size - 2, bg, sp)
    if base_inner is None:
        return None

    rotated_cells = [_make_cell(np.rot90(base_inner, k), bg) for k in range(4)]

    chosen_cells: list[np.ndarray] = []
    in_cells: list[list[np.ndarray]] = []
    for _row in range(n_rows):
        orientations, odd_col = _row_orientations(rng, n_cols)
        row_cells = [rotated_cells[k].copy() for k in orientations]
        chosen_cells.append(row_cells[int(odd_col)].copy())
        in_cells.append(row_cells)

        # Generator-side legality check: exactly one half-turn class singleton,
        # and the original raw-byte logic must not already solve the row.
        class_sigs = [_half_turn_signature(cell) for cell in row_cells]
        class_counts = Counter(class_sigs)
        if sorted(map(int, class_counts.values())) != [1, n_cols - 1]:
            return None
        raw_idx = _raw_unique_index(row_cells)
        if raw_idx is None or int(raw_idx) == int(odd_col):
            return None

    out = _build_lattice([[cell] for cell in chosen_cells], lattice, bg)
    inp = _build_lattice(in_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
