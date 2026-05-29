# Combo variant of 38007db0 (index 20)
# combo_spec: D2/0+D13/0
# Name: 颜色异常多干扰检测
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - The lattice geometry stays the same as the original task: 1-pixel lattice lines,
#   square cells, 3-4 lattice rows, and the output is still a single-column lattice.
# - Original rule per row:
#   one cell has a unique full sprite pattern and all others are exact copies, so
#   `Counter(cell_bytes)` is enough.
# - Fused rule per row:
#   1. Most cells are copies of one base sprite mask drawn in majority sprite color A.
#   2. One distractor cell keeps color A but flips 1-2 inner pixels.
#   3. One target cell keeps the base mask but uses anomaly sprite color B.
#   4. Output extracts the target cell, i.e. the only cell whose anomaly is in the
#      color dimension rather than the shape/pixel-flip dimension.
# - Small trace:
#   original row signatures: [base(A), base(A), base(A), odd_shape(A)] -> pick odd.
#   fused row signatures:    [base(A), flip(A), base(A), base(B)].
#   Here there are two singleton full cells (`flip(A)` and `base(B)`), so the old
#   singleton logic fails. The D13-style "pick the cell closest to base" also fails,
#   because `flip(A)` differs by only 1-2 pixels while `base(B)` differs on every
#   sprite pixel. The correct rule is to inspect sprite color and pick `base(B)`.
#
# b) Complexity calibration
# - Each row has exactly two anomalies, which is enough to create real interference
#   without making the grid visually messy.
# - The distractor is constrained to 1-2 pixel flips only. That keeps it clearly
#   readable as a near-copy of the base, and also guarantees that a shape-distance
#   heuristic prefers the distractor over the true answer.
# - The target uses a single global anomaly color B across the whole puzzle. This
#   gives repeated support across rows so a human can infer "pick the odd color".
# - We use 4-5 columns so the base signature still has a strict majority even after
#   inserting two anomalies.
#
# c) Implicit-assumption review
# - Original `main()` assumed exactly one singleton signature in each row. Broken:
#   now each row has two singleton full cells.
# - D13-only reasoning assumed the desired singleton is the one closest in shape to
#   the repeated base. Broken: in this combo the closest singleton is the distractor.
# - Grid parsing, lattice-color detection, row/column slicing, and output assembly
#   remain valid because the lattice structure itself is unchanged.
# - New solver assumption: each cell contains exactly one non-background sprite color,
#   and exactly one cell per row uses the minority sprite color. The generator
#   enforces both by construction.
#
# d) Placement legality
# - All cells are built at one fixed size and placed into non-overlapping lattice
#   slots separated by 1-pixel grid lines.
# - The target and distractor occupy distinct columns; the remaining columns are base
#   copies. No pasted object can overlap or leave the lattice bounds.
#
# e) Visual observability
# - A human sees rows with one slightly glitched majority-color cell and one clearly
#   different-color cell. The output always keeps the different-color cell.
# - No hidden ordering, color-number ranking, or off-screen metadata is used. The
#   selection rule is recoverable from visible color repetition across rows.
#
# f) Visual cleanliness
# - The fused change adds only one extra anomaly per row, not stacked markers or
#   extra overlays. The base sprite remains repeated and thus easy to visually anchor.
# - The target keeps the same shape as the base, so the color contrast is isolated
#   and clean instead of being confounded by a simultaneous large shape deformation.
#
# g) Change inferability
# - There are exactly four global colors: lattice L, background bg, majority sprite
#   color A, and anomaly sprite color B.
# - The target color change is supported by repeated examples across rows, while the
#   distractor always shares the majority color. That makes the color rule inferable
#   from 2-4 examples without requiring any numeric interpretation of color ids.
#
# h) Generator anti-degeneracy checks
# - The generator rejects samples unless the original solver's "first singleton"
#   heuristic would choose the distractor instead of the target.
# - It also rejects samples unless the D13-only "closest singleton to base" heuristic
#   would choose the distractor instead of the target.
#   This ensures the fusion behavior is actually present rather than collapsing back
#   to either source task.
#
# i) Main/generator contract
# - `main()` requires:
#   - recoverable lattice lines,
#   - a detectable background color,
#   - exactly one minority sprite color per row.
# - `generate_input()` guarantees:
#   - 4 distinct global colors,
#   - every cell has exactly one sprite color,
#   - each row has signature pattern counts `{base: >=2, distractor: 1, target: 1}`,
#   - the target is the unique minority-color cell.

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from arc_utils import paste


def _extract_lattice_structure(
    g: np.ndarray,
) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    row_lines = [r for r in range(h) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
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
    return lattice, row_lines, col_lines, r_slices, c_slices


def _sprite_color(cell: np.ndarray, bg: int) -> int:
    colors = {int(v) for v in cell.flatten() if int(v) != int(bg)}
    assert len(colors) == 1
    return next(iter(colors))


def _cell_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Each lattice row contains one majority sprite color, one same-color shape
      distractor, and one different-color target.
    - Extract the target cell: the only cell using the minority sprite color.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape

    lattice, row_lines, _col_lines, r_slices, c_slices = _extract_lattice_structure(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        sprite_colors = [_sprite_color(cell, bg) for cell in cells]
        color_counts = Counter(sprite_colors)
        top = color_counts.most_common()
        assert len(top) == 2
        assert int(top[0][1]) >= 3
        assert int(top[1][1]) == 1
        odd_color = int(top[1][0])
        odd_idx = next(i for i, color in enumerate(sprite_colors) if int(color) == odd_color)
        chosen_cells.append(cells[int(odd_idx)])

    out_w = int(cell_w + 2)
    out = np.full((h, out_w), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for r in row_lines:
        out[int(r), :] = int(lattice)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra):int(rb), 1:1 + int(cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(mask: np.ndarray, sprite: int, bg: int) -> np.ndarray:
    inner = np.full(mask.shape, int(bg), dtype=int)
    inner[mask] = int(sprite)
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


def _flip_mask(mask: np.ndarray, flip_idx: np.ndarray) -> np.ndarray:
    variant = mask.copy()
    flat = variant.reshape(-1)
    flat[flip_idx] = ~flat[flip_idx]
    return variant


def _gen_base_mask(rng: np.random.Generator, inner_size: int) -> np.ndarray:
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))
    mask = np.zeros((inner_size, inner_size), dtype=bool)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    mask.flat[on_idx] = True
    return mask


def _gen_distractor_mask(
    rng: np.random.Generator,
    base_mask: np.ndarray,
    used_sigs: set[bytes],
) -> np.ndarray | None:
    inner_area = base_mask.size
    for _ in range(200):
        flip_count = int(rng.choice([1, 2]))
        flip_idx = rng.choice(inner_area, size=flip_count, replace=False)
        variant = _flip_mask(base_mask, flip_idx)
        if not np.any(variant):
            continue
        if int(variant.sum()) == inner_area:
            continue
        sig = variant.tobytes()
        if sig in used_sigs:
            continue
        used_sigs.add(sig)
        return variant
    return None


def _row_analysis(cells: list[np.ndarray], bg: int) -> dict[str, int]:
    sigs = [cell.tobytes() for cell in cells]
    sig_counts = Counter(sigs)
    first_singleton_idx = next(i for i, sig in enumerate(sigs) if int(sig_counts[sig]) == 1)

    sprite_colors = [_sprite_color(cell, bg) for cell in cells]
    color_counts = Counter(sprite_colors)
    target_color = next(color for color, cnt in color_counts.items() if int(cnt) == 1)
    target_idx = next(i for i, color in enumerate(sprite_colors) if int(color) == int(target_color))

    base_sig, _base_count = sig_counts.most_common(1)[0]
    base_cell = next(cell for cell, sig in zip(cells, sigs) if sig == base_sig)
    other = []
    for idx, (cell, sig) in enumerate(zip(cells, sigs)):
        if sig != base_sig:
            other.append((_cell_distance(cell, base_cell), idx))
    other.sort()
    d13_pick_idx = int(other[0][1])

    return {
        "target_idx": int(target_idx),
        "first_singleton_idx": int(first_singleton_idx),
        "d13_pick_idx": int(d13_pick_idx),
        "target_distance": int(_cell_distance(cells[target_idx], base_cell)),
        "d13_distance": int(other[0][0]),
    }


def measure_fusion_behavior(grid: np.ndarray) -> dict[str, object]:
    """
    Quantify whether generated samples truly exhibit the combo behavior.

    Returns counts for:
    - rows where original first-singleton logic would fail,
    - rows where D13-only closest-to-base logic would fail.
    """
    g = np.array(grid, dtype=int)
    lattice, _row_lines, _col_lines, r_slices, c_slices = _extract_lattice_structure(g)
    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    rows: list[dict[str, int]] = []
    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        rows.append(_row_analysis(cells, bg))

    original_wrong = sum(int(row["first_singleton_idx"] != row["target_idx"]) for row in rows)
    d13_wrong = sum(int(row["d13_pick_idx"] != row["target_idx"]) for row in rows)
    return {
        "row_count": len(rows),
        "original_wrong_rows": int(original_wrong),
        "d13_wrong_rows": int(d13_wrong),
        "rows": rows,
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, majority_sp, anomaly_sp = map(int, rng.choice(10, size=4, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(4, 6))

    inner_size = cell_size - 2
    base_mask = _gen_base_mask(rng, inner_size)
    target_cell = _make_cell(base_mask, anomaly_sp, bg)
    base_cell = _make_cell(base_mask, majority_sp, bg)

    used_distractors = {base_mask.tobytes()}
    chosen_cells: list[np.ndarray] = []
    in_cells: list[list[np.ndarray]] = []

    for _ in range(n_rows):
        distractor_mask = _gen_distractor_mask(rng, base_mask, used_distractors)
        if distractor_mask is None:
            return None
        distractor_cell = _make_cell(distractor_mask, majority_sp, bg)

        row_cells = [base_cell.copy() for _ in range(n_cols)]
        distractor_col = int(rng.integers(0, n_cols - 1))
        target_col = int(rng.integers(distractor_col + 1, n_cols))
        row_cells[distractor_col] = distractor_cell
        row_cells[target_col] = target_cell.copy()

        analysis = _row_analysis(row_cells, bg)
        if analysis["first_singleton_idx"] == analysis["target_idx"]:
            return None
        if analysis["d13_pick_idx"] == analysis["target_idx"]:
            return None
        if analysis["d13_distance"] >= analysis["target_distance"]:
            return None

        chosen_cells.append(target_cell.copy())
        in_cells.append(row_cells)

    out = _build_lattice([[cell] for cell in chosen_cells], lattice, bg)
    inp = _build_lattice(in_cells, lattice, bg)
    if not np.array_equal(main(inp), out):
        return None

    stats = measure_fusion_behavior(inp)
    if int(stats["original_wrong_rows"]) != int(stats["row_count"]):
        return None
    if int(stats["d13_wrong_rows"]) != int(stats["row_count"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
