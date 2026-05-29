# Combo variant of 38007db0 (index 2)
# combo_spec: D2/2+D4/2
# Name: 旋转异常差异提取

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from arc_utils import paste

# Design analysis
# ===============
# a) Concrete behavior and trace
#    The lattice geometry stays the same as 38007db0: 1px grid lines, square
#    cells, one global sprite color on a cell background, and a single-column
#    output lattice.
#
#    The fused row logic is:
#    - A row is not "all exact copies plus one odd bitmap" anymore.
#    - Instead, each row contains:
#        repeated reference orientation twice,
#        one 180-degree mate of that reference (same half-turn class),
#        one quarter-turn anomaly from the other half-turn class.
#      Example with four cells:
#        [rot0(base), rot2(base), rot0(base), rot1(base)]
#    - Old exact-byte counting sees two singletons (rot2 and rot1), so it cannot
#      uniquely recover the intended source.
#    - New source logic groups cells by half-turn class using the canonical key
#      min(cell, rot180(cell)).
#        class A = {rot0, rot2} occurs 3 times
#        class B = {rot1} occurs 1 time
#      so rot1 is the rotation anomaly.
#    - The transform is also changed: we do not copy the whole anomaly cell.
#      We compare it against the row's repeated exact reference orientation and
#      keep only the anomaly's visible positive differences:
#        diff[r,c] = anomaly[r,c] if anomaly[r,c] != reference[r,c] else bg
#      Because all sprites use only {bg, sp}, this means "show the pixels that
#      the rotated anomaly has where the repeated reference does not".
#
# b) Complexity calibration
#    The D2 change would degenerate if a row were just [base, base, rotated]:
#    exact-byte uniqueness would already solve it. To prevent that, every row is
#    forced to contain one same-class half-turn distractor in addition to the
#    anomaly. That gives the old solver two singletons and breaks its assumption.
#    The D4 change would be visually messy if the rotation difference were too
#    dense, so the generator only accepts prototype sprites whose quarter-turn
#    positive difference count is small-but-visible (2-5-ish pixels).
#
# c) Implicit assumptions audit against the original main()
#    - Uniform lattice-line detection still works because geometry is unchanged.
#    - Inferring lattice color from uniform rows/cols still works.
#    - Inferring bg as the most common non-lattice color still works because bg
#      still dominates cell interiors and output diff cells remain sparse.
#    - The original row assumption "exactly one raw bitmap appears once" is now
#      false. The fused solver replaces it with:
#        1. exactly one half-turn class has count 1
#        2. exactly one exact bitmap has multiplicity >= 2, which is the row's
#           repeated reference orientation used for the diff transform
#      The generator enforces both.
#    - The original output assumption "copy the chosen cell verbatim" is false.
#      The solver now computes a sparse difference cell before writing output.
#
# d) Placement legality
#    Cells are still generated independently and pasted into a regular lattice,
#    so nothing overlaps or crosses grid boundaries. Rotations happen inside the
#    padded inner sprite area only.
#
# e) Visual observability
#    A human sees three supports per row:
#    - the dominant orientation repeated twice,
#    - its upside-down mate once,
#    - one sideways/other-family anomaly once.
#    Across 2-4 examples, the singleton from the "other rotation family" is
#    learnable by eye, and the output keeps only the few pixels that appear in
#    that anomaly but not in the row's repeated reference orientation.
#
# f) Visual cleanliness
#    No extra markers or colors are introduced. Output cells stay sparse and use
#    only the existing sprite color on bg, inside the same single-column frame.
#
# g) New assumptions introduced by the fused solver
#    - Each row must have exactly two half-turn classes with counts [3, 1].
#    - Each row must also have exactly one exact orientation repeated at least
#      twice, because that repeated orientation is the subtraction reference.
#    - The prototype sprite must be non-symmetric under 90 and 180 degrees so
#      both the anomaly selection and the diff output are visible.
#    - The generator checks all of these before accepting an instance.


def _half_turn_signature(arr: np.ndarray) -> bytes:
    a = arr.tobytes()
    b = np.rot90(arr, 2).tobytes()
    return a if a <= b else b


def _lattice_slices(
    g: np.ndarray,
) -> tuple[int, int, int, list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == int(lattice))
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == int(lattice))

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    return H, W, lattice, row_lines, r_slices, c_slices


def _positive_diff_cell(reference_cell: np.ndarray, anomaly_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(anomaly_cell.shape, int(bg), dtype=int)
    mask = (anomaly_cell != reference_cell) & (anomaly_cell != int(bg))
    diff[mask] = anomaly_cell[mask]
    return diff


def _raw_unique_index(cells: list[np.ndarray]) -> int | None:
    sigs = [cell.tobytes() for cell in cells]
    cnt = Counter(sigs)
    return next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the per-row rotation anomaly, then keep only the anomaly pixels that
    differ from the row's repeated exact reference orientation.
    """
    g = np.array(grid, dtype=int)
    H, _, lattice, row_lines, r_slices, c_slices = _lattice_slices(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_diffs: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]

        class_sigs = [_half_turn_signature(cell) for cell in cells]
        class_counts = Counter(class_sigs)
        uniq_classes = [sig for sig, count in class_counts.items() if int(count) == 1]
        assert len(class_counts) == 2 and len(uniq_classes) == 1
        anomaly_idx = next(i for i, sig in enumerate(class_sigs) if sig == uniq_classes[0])

        raw_sigs = [cell.tobytes() for cell in cells]
        raw_counts = Counter(raw_sigs)
        repeated_sigs = [sig for sig, count in raw_counts.items() if int(count) >= 2]
        assert len(repeated_sigs) == 1
        reference_idx = next(i for i, sig in enumerate(raw_sigs) if sig == repeated_sigs[0])

        reference_cell = cells[int(reference_idx)]
        anomaly_cell = cells[int(anomaly_idx)]
        chosen_diffs.append(_positive_diff_cell(reference_cell, anomaly_cell, bg))

    outW = int(cell_w + 2)
    out = np.full((H, outW), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for r in row_lines:
        out[int(r), :] = int(lattice)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = chosen_diffs[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = int(inner.shape[0]) + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = int(cells[0][0].shape[0])
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


def _positive_rotation_diff_count(a: np.ndarray, b: np.ndarray, bg: int) -> int:
    return int(np.count_nonzero((b != a) & (b != int(bg))))


def _rotation_overlap_count(a: np.ndarray, b: np.ndarray, sp: int) -> int:
    return int(np.count_nonzero((a == int(sp)) & (b == int(sp))))


def _is_good_base(inner: np.ndarray, bg: int) -> bool:
    rots = [np.rot90(inner, k) for k in range(4)]
    sp = next(v for v in map(int, np.unique(inner)) if int(v) != int(bg))
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rots[i], rots[j]):
                return False

    area = int(inner.size)
    max_sparse = max(5, area // 2)

    # We use row bases k in {0, 1} and anomalies k+1, so both diff views must
    # stay sparse-but-visible.
    for k in (0, 1):
        diff_count = _positive_rotation_diff_count(rots[k], rots[(k + 1) % 4], bg)
        if diff_count < 2 or diff_count > max_sparse:
            return False
        if _rotation_overlap_count(rots[k], rots[(k + 1) % 4], sp) < 1:
            return False

    # The same-class half-turn distractor must also be visibly different.
    for k in (0, 1):
        if _positive_rotation_diff_count(rots[k], rots[(k + 2) % 4], bg) < 2:
            return False

    return True


def _sample_base_inner(rng: np.random.Generator, inner_size: int, bg: int, sp: int) -> np.ndarray | None:
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.30 * inner_area)
    max_fill = math.floor(0.60 * inner_area)

    for _ in range(400):
        fill = int(rng.integers(min_fill, max_fill + 1))
        inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        on_idx = rng.choice(inner_area, size=fill, replace=False)
        inner.flat[on_idx] = int(sp)
        if _is_good_base(inner, bg):
            return inner
    return None


def _build_row_cells(
    rotated_cells: list[np.ndarray],
    base_k: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    reference_k = int(base_k)
    distractor_k = int((base_k + 2) % 4)
    anomaly_k = int((base_k + 1) % 4)

    orientations = [reference_k, reference_k, distractor_k, anomaly_k]
    for _ in range(30):
        perm = list(map(int, rng.permutation(len(orientations))))
        row_orientations = [orientations[i] for i in perm]
        row_cells = [rotated_cells[k].copy() for k in row_orientations]
        anomaly_idx = next(i for i, k in enumerate(row_orientations) if int(k) == int(anomaly_k))
        raw_idx = _raw_unique_index(row_cells)
        if raw_idx is None or int(raw_idx) == int(anomaly_idx):
            continue

        class_sigs = [_half_turn_signature(cell) for cell in row_cells]
        class_counts = Counter(class_sigs)
        if sorted(map(int, class_counts.values())) != [1, 3]:
            continue

        raw_counts = Counter(cell.tobytes() for cell in row_cells)
        if sorted(map(int, raw_counts.values())) != [1, 1, 2]:
            continue

        reference_idx = next(i for i, k in enumerate(row_orientations) if int(k) == int(reference_k))
        return row_cells, row_cells[int(reference_idx)], row_cells[int(anomaly_idx)], anomaly_idx

    raise AssertionError("failed to build non-degenerate row")


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = 4

    base_inner = _sample_base_inner(rng, cell_size - 2, bg, sp)
    if base_inner is None:
        return None

    rotated_cells = [_make_cell(np.rot90(base_inner, k), bg) for k in range(4)]

    row_base_choices = [0, 1]
    base_ks = [row_base_choices[int(rng.integers(0, len(row_base_choices)))] for _ in range(n_rows)]
    if n_rows >= 2 and len(set(map(int, base_ks))) == 1:
        base_ks[int(rng.integers(0, n_rows))] = int(1 - base_ks[0])

    in_cells: list[list[np.ndarray]] = []
    out_cells: list[list[np.ndarray]] = []
    out_signatures: set[bytes] = set()
    for base_k in base_ks:
        row_cells, reference_cell, anomaly_cell, anomaly_idx = _build_row_cells(rotated_cells, int(base_k), rng)
        diff_cell = _positive_diff_cell(reference_cell, anomaly_cell, bg)
        if int(np.count_nonzero(diff_cell != int(bg))) < 2:
            return None

        in_cells.append(row_cells)
        out_cells.append([diff_cell])
        out_signatures.add(diff_cell.tobytes())

        # Generator-side fused-rule checks.
        if int(_raw_unique_index(row_cells)) == int(anomaly_idx):
            return None

    if n_rows >= 2 and len(out_signatures) < 2:
        return None

    inp = _build_lattice(in_cells, lattice, bg)
    out = _build_lattice(out_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) and int(n_cols) == len(in_cells[0]) else None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 500 attempts")
