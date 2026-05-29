# Combo variant of 38007db0 (index 7)
# combo_spec: D1/0+D2/1
# Name: 列对齐镜像异常检测

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste


"""
Design analysis for combo D1/0 + D2/1
=====================================

spec:
  task: Scan each lattice column, detect the single mirror anomaly in that column,
    and extract those anomalous cells into a single-row lattice output.
  preserved_structure:
    - Exactly 3 colors: lattice L, cell background bg, sprite color sp.
    - Uniform 1px lattice lines, square cells of size 5 or 6.
    - 3-4 lattice rows and 3-4 lattice columns.
    - Each cell still contains 1px bg padding around a sprite placed in the inner area.
  fused_change:
    - D1/0 changes the alignment rule:
        original = one odd cell per row -> single-column output
        fused    = one odd cell per column -> single-row output
    - D2/1 changes what "odd" means:
        original = bytewise unique full cell
        fused    = after cropping away translation, one sprite is the horizontal or vertical
                   mirror of the repeated base sprite in that column.

Concrete behavior and trace:
  - Original 3x4-cell logic:
      row0: [base, base, odd, base] -> pick odd
      row1: [base, odd, base, base] -> pick odd
      row2: [odd, base, base, base] -> pick odd
      output is a 3x1 lattice.
  - Fused 3x4-cell logic:
      col0: [base@offA, base@offB, vflip(base)@offC] -> crop -> [base, base, vflip(base)] -> pick row2
      col1: [hflip(base)@offD, base@offE, base@offF] -> crop -> [hflip(base), base, base] -> pick row0
      col2: [base@offG, base@offH, base@offI] may happen in some row views, so row-scanning is invalid
      col3: [base@offJ, vflip(base)@offK, base@offL] -> pick row1
      output is a 1x4 lattice containing the three chosen anomaly cells in column order.
  - Key difference:
      full-cell byte equality is intentionally broken by varying offsets within each column,
      so the solver must normalize by cropping the sprite and then reason over mirror relation.

Complexity calibration:
  - Only two mirror families are allowed: horizontal and vertical flip.
  - A single asymmetric base sprite is reused everywhere so the task remains visually coherent.
  - Offsets vary within each column, which is enough to kill the original byte-count shortcut
    without adding extra colors or unrelated markers.
  - The generator forces at least one row to contain zero anomalies, so the dataset cannot
    silently collapse back to a per-row interpretation.

Implicit assumption audit against the original solver:
  - Gridline detection via fully uniform rows/cols still works; generator preserves that.
  - Background inference as the most common non-lattice color still works; sprites stay sparse.
  - The original assumption "one full-cell byte signature appears once in each row" is false now.
    Replacement assumptions are:
      1. each cell contains at least one sprite pixel,
      2. cropping the sprite removes translation,
      3. in each column exactly one cropped sprite is a mirror of the repeated cropped base,
      4. every column has exactly one anomaly, while some rows may have none.
  - The old output shape assumption (single column, original height) is also false now.
    The fused output must instead be a single row with the original column layout.

Placement legality:
  - Sprites are placed only inside the inner area of each cell.
  - Offsets within a column are sampled without replacement, ensuring distinct full cells there.
  - Lattice construction writes each cell into a disjoint slot, so nothing overlaps or leaves bounds.

Visual observability and cleanliness:
  - The sprite mask is forced to be asymmetric under both horizontal and vertical reflection,
    so mirror anomalies are visible to a human.
  - The sprite touches all four sides of its tight bbox, so cropping corresponds to a clear shape.
  - No extra annotations are added; the only cues are repetition-down-columns and a single mirror
    exception per column, producing a clean single-row output.

New solver assumptions checked against the generator:
  - `main()` assumes every column has one repeated cropped sprite and one mirrored exception.
  - `main()` assumes the exception is either `fliplr(base)` or `flipud(base)`.
  - `main()` assumes a unique sprite color distinct from lattice and background.
  - `generate_input()` constructs inputs to satisfy all of those conditions and also ensures
    at least one row has zero anomalies, which makes the D1 change operational rather than cosmetic.
"""


def _crop_sprite(cell: np.ndarray, sp: int) -> np.ndarray:
    coords = np.argwhere(cell == int(sp))
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return cell[int(r0) : int(r1), int(c0) : int(c1)].copy()


def _is_mirror_of(candidate: np.ndarray, base: np.ndarray) -> bool:
    return candidate.shape == base.shape and (
        np.array_equal(candidate, np.fliplr(base)) or np.array_equal(candidate, np.flipud(base))
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the mirror-anomaly cell from each lattice column into a single-row lattice.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == int(L))
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == int(L))

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    cell_h = int(r_slices[0][1] - r_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    sprite_colors = sorted(set(map(int, g.flatten())) - {int(L), int(bg)})
    assert len(sprite_colors) == 1
    sp = int(sprite_colors[0])

    chosen_cells: list[np.ndarray] = []
    for ca, cb in c_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ra, rb in r_slices]
        sprites = [_crop_sprite(cell, sp) for cell in cells]
        sigs = [sprite.tobytes() for sprite in sprites]
        counts = Counter(sigs)

        majority_sig, majority_count = counts.most_common(1)[0]
        assert int(majority_count) == len(cells) - 1
        base_sprite = next(sprite for sprite, sig in zip(sprites, sigs) if sig == majority_sig)

        mirror_idxs = [
            idx
            for idx, sprite in enumerate(sprites)
            if sigs[idx] != majority_sig and _is_mirror_of(sprite, base_sprite)
        ]
        assert len(mirror_idxs) == 1
        chosen_cells.append(cells[int(mirror_idxs[0])])

    out = np.full((int(cell_h) + 2, W), int(bg), dtype=int)
    out[0, :] = int(L)
    out[-1, :] = int(L)
    for c in col_lines:
        out[:, int(c)] = int(L)

    for j, (ca, cb) in enumerate(c_slices):
        out[1 : 1 + int(cell_h), int(ca) : int(cb)] = chosen_cells[int(j)]

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


def _all_offsets(inner_size: int, sprite_shape: tuple[int, int]) -> list[tuple[int, int]]:
    sh, sw = map(int, sprite_shape)
    return [
        (dr, dc)
        for dr in range(int(inner_size) - sh + 1)
        for dc in range(int(inner_size) - sw + 1)
    ]


def _place_mask(mask: np.ndarray, inner_size: int, offset: tuple[int, int], bg: int, sp: int) -> np.ndarray:
    inner = np.full((int(inner_size), int(inner_size)), int(bg), dtype=int)
    dr, dc = map(int, offset)
    h, w = mask.shape
    inner[dr : dr + int(h), dc : dc + int(w)][mask.astype(bool)] = int(sp)
    return inner


def _random_base_mask(rng: np.random.Generator, inner_size: int, min_offsets: int) -> np.ndarray | None:
    dims = [
        (h, w)
        for h in range(2, int(inner_size) + 1)
        for w in range(2, int(inner_size) + 1)
        if (int(inner_size) - h + 1) * (int(inner_size) - w + 1) >= int(min_offsets)
    ]
    if not dims:
        return None

    for _ in range(200):
        h, w = dims[int(rng.integers(len(dims)))]
        area = int(h * w)
        fill = int(rng.integers(max(3, area // 2), area))
        mask = np.zeros((int(h), int(w)), dtype=int)
        on_idx = rng.choice(area, size=fill, replace=False)
        mask.flat[on_idx] = 1

        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(np.fliplr(mask), np.flipud(mask)):
            continue
        return mask

    return None


def _choose_odd_rows(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray | None:
    for _ in range(80):
        odd_rows = rng.integers(0, n_rows, size=n_cols)
        counts = np.bincount(odd_rows, minlength=n_rows)
        if int(np.min(counts)) == 0:
            return odd_rows
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))
    inner_size = int(cell_size - 2)

    base_mask = _random_base_mask(rng, inner_size, max(n_rows, n_cols))
    if base_mask is None:
        return None

    hflip_mask = np.fliplr(base_mask)
    vflip_mask = np.flipud(base_mask)
    axis_masks = {"h": hflip_mask, "v": vflip_mask}
    axis_choices = [axis for axis, mask in axis_masks.items() if not np.array_equal(mask, base_mask)]
    if len(axis_choices) != 2:
        return None

    odd_rows = _choose_odd_rows(rng, n_rows, n_cols)
    if odd_rows is None:
        return None

    col_axes = [axis_choices[i % 2] for i in range(n_cols)]
    rng.shuffle(col_axes)

    offsets = _all_offsets(inner_size, base_mask.shape)
    if len(offsets) < max(n_rows, n_cols):
        return None

    in_cells: list[list[np.ndarray | None]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    chosen_cells: list[np.ndarray] = []
    used_output_cells: set[bytes] = set()

    for col in range(n_cols):
        built_col = False
        mirror_mask = axis_masks[col_axes[col]]
        odd_row = int(odd_rows[col])

        for _ in range(100):
            row_offset_ids = rng.permutation(len(offsets))[:n_rows]
            row_offsets = [offsets[int(idx)] for idx in row_offset_ids]

            column_cells: list[np.ndarray] = []
            for row, offset in enumerate(row_offsets):
                mask = mirror_mask if row == odd_row else base_mask
                inner = _place_mask(mask, inner_size, offset, bg, sp)
                column_cells.append(_make_cell(inner, bg))

            if len({cell.tobytes() for cell in column_cells}) != n_rows:
                continue

            chosen = column_cells[odd_row]
            key = chosen.tobytes()
            if key in used_output_cells:
                continue

            used_output_cells.add(key)
            chosen_cells.append(chosen.copy())
            for row in range(n_rows):
                in_cells[row][col] = column_cells[row]
            built_col = True
            break

        if not built_col:
            return None

    realized_cells = [[cell for cell in row if cell is not None] for row in in_cells]
    if any(len(row) != n_cols for row in realized_cells):
        return None

    out = _build_lattice([chosen_cells], lattice, bg)
    inp = _build_lattice(realized_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
