# Variant of 38007db0
# Dimension: D2 — 源对象：行内唯一精灵单元格 (src)
# Change: 每行中模式只出现一次的方形单元格（内含独特精灵变体）... → 镜像异常项

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste


"""
Design analysis for the D2 variant
==================================

spec:
  task: Per-row mirror-anomaly extraction from a lattice grid into a single-column lattice.
  preserved_structure:
    - Same 1px lattice gridlines and same single-column output frame as the original task.
    - Same 3 colors: lattice L, cell background bg, sprite color sp.
    - Same 3-4 lattice rows and 3-4 lattice columns.
    - Same square cells of size 5 or 6, with 1px bg padding around the inner area.
  core_change:
    - Original: one cell per row is bytewise unique; all others are exact copies.
    - Variant: all cells in a row use distinct whole-cell placements, so bytewise frequency is useless.
      After cropping away bg padding/translation, the majority cells share the same sprite, and the
      single anomaly is the horizontal or vertical mirror of that sprite.
  concrete_pixel_realization:
    - A small asymmetric sprite mask is embedded inside each cell's inner area at different offsets.
    - Majority cells in a row contain the same asymmetric sprite at different offsets.
    - The odd cell contains the sprite mirrored across one axis, also at an offset.
    - Because offsets differ, every full cell in the row is distinct; because cropping removes offsets,
      the majority cropped sprites become identical while the odd one is the mirrored anomaly.
  behavior_trace:
    - Original row logic on full cells:
        [base@left, base@mid, variant@right] -> byte counts identify variant directly.
    - Variant row logic on full cells:
        [base@left, base@mid, mirror(base)@right] with all three offsets different
        -> byte counts are {1,1,1}, so the old solver cannot identify the source object.
    - Variant row logic after sprite cropping:
        [base, base, mirror(base)] -> majority sprite is base, anomaly is the one equal to
        hflip(base) or vflip(base).
  complexity_calibration:
    - Offsets add enough variability that old bytewise matching fails on purpose.
    - Mirror choice is restricted to two axes only, avoiding an explosion of transform cases.
    - The sprite is forced to be asymmetric under both horizontal and vertical reflection so the
      transformation is visually observable and not confounded by accidental symmetry.
  implicit_assumption_audit:
    - Gridline detection still assumes uniform full rows/cols of color L; generator preserves this.
    - Background inference still assumes bg is the most common non-lattice color; generator keeps
      sprite fill sparse enough that bg dominates non-lattice pixels.
    - The original `cell.tobytes()` uniqueness assumption no longer holds and is replaced by:
        1. each cell has at least one sp pixel,
        2. cropping the sp bounding box removes translation,
        3. exactly one cropped sprite per row is a mirror of the repeated cropped base sprite.
    - `main()` now assumes the mirrored anomaly is one of {hflip(base), vflip(base)} and generator
      guarantees that.
  placement_legality:
    - Sprites are always placed inside the inner area using valid offsets.
    - Row offsets are sampled without replacement, so whole cells do not duplicate within a row.
    - Cell shapes remain square and aligned with the lattice; nothing overlaps gridlines.
  visual_observability:
    - The sprite touches all four sides of its own bbox, so the cropped shape is tight and readable.
    - The sprite is asymmetric under both flips, making the mirror relation obvious in small grids.
    - Using both horizontal and vertical anomalies across rows keeps the transformation family visible.
  cleanliness:
    - No extra markers, no extra colors, no overlapping annotations.
    - The only visual change from the original task is that the odd cell is a reflected sprite and
      all cells are translated within their local bg padding.
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
    Extract the mirror-anomaly cell from each lattice row into a single-column lattice.

    The original task could be solved by counting full-cell byte signatures. This variant deliberately
    breaks that shortcut by translating the sprite inside each cell, so `main()` must crop the sprite
    itself and reason over mirror relationships instead of raw cell equality.
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

    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    sprite_colors = sorted(set(map(int, g.flatten())) - {int(L), int(bg)})
    assert len(sprite_colors) == 1
    sp = int(sprite_colors[0])

    chosen_cells: list[np.ndarray] = []
    for (ra, rb) in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for (ca, cb) in c_slices]
        sprites = [_crop_sprite(cell, sp) for cell in cells]
        sigs = [sprite.tobytes() for sprite in sprites]
        counts = Counter(sigs)

        majority_sig, majority_count = counts.most_common(1)[0]
        assert int(majority_count) == len(cells) - 1
        base_sprite = next(sprite for sprite, sig in zip(sprites, sigs) if sig == majority_sig)

        mirror_idx = next(
            (idx for idx, sprite in enumerate(sprites) if sigs[idx] != majority_sig and _is_mirror_of(sprite, base_sprite)),
            None,
        )
        assert mirror_idx is not None
        chosen_cells.append(cells[int(mirror_idx)])

    out = np.full((H, int(cell_w) + 2), int(bg), dtype=int)
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


def _random_base_mask(rng: np.random.Generator, inner_size: int, n_cols: int) -> np.ndarray | None:
    dims = [
        (h, w)
        for h in range(2, int(inner_size) + 1)
        for w in range(2, int(inner_size) + 1)
        if (int(inner_size) - h + 1) * (int(inner_size) - w + 1) >= int(n_cols)
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

        # Force a tight, visually asymmetric sprite so mirror differences are observable.
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(np.fliplr(mask), np.flipud(mask)):
            continue
        return mask

    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))
    inner_size = int(cell_size - 2)

    base_mask = _random_base_mask(rng, inner_size, n_cols)
    if base_mask is None:
        return None

    hflip_mask = np.fliplr(base_mask)
    vflip_mask = np.flipud(base_mask)
    axis_masks = {"h": hflip_mask, "v": vflip_mask}
    axis_choices = [axis for axis, mask in axis_masks.items() if not np.array_equal(mask, base_mask)]
    if not axis_choices:
        return None

    if len(axis_choices) == 2:
        row_axes = [axis_choices[i % 2] for i in range(n_rows)]
        rng.shuffle(row_axes)
    else:
        row_axes = [axis_choices[0] for _ in range(n_rows)]

    offsets = _all_offsets(inner_size, base_mask.shape)
    if len(offsets) < n_cols:
        return None

    chosen_cells: list[np.ndarray] = []
    in_cells: list[list[np.ndarray]] = []
    used_output_cells: set[bytes] = set()

    for row in range(n_rows):
        built_row = False
        for _ in range(80):
            row_offset_ids = rng.permutation(len(offsets))[:n_cols]
            row_offsets = [offsets[int(idx)] for idx in row_offset_ids]
            odd_col = int(rng.integers(0, n_cols))
            mirror_mask = axis_masks[row_axes[row]]

            row_cells: list[np.ndarray] = []
            for col, offset in enumerate(row_offsets):
                mask = mirror_mask if col == odd_col else base_mask
                inner = _place_mask(mask, inner_size, offset, bg, sp)
                row_cells.append(_make_cell(inner, bg))

            if len({cell.tobytes() for cell in row_cells}) != n_cols:
                continue

            chosen = row_cells[odd_col]
            key = chosen.tobytes()
            if key in used_output_cells:
                continue

            used_output_cells.add(key)
            chosen_cells.append(chosen.copy())
            in_cells.append(row_cells)
            built_row = True
            break

        if not built_row:
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
