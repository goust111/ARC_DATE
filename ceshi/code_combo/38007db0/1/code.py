# Combo variant of 38007db0 (index 1)
# combo_spec: D2/1+D4/2
# Name: 镜像异常差异提取

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
Original row behavior:
  - A row contains exact full-cell copies of one base cell plus one bytewise-unique odd cell.
  - Old `main()` can identify the odd cell by counting full-cell byte signatures.
  - Output copies that whole odd cell into the single-column lattice.

Combo row behavior:
  - Each cell still contains one sprite on bg inside a lattice cell.
  - Majority cells in a row contain the same asymmetric sprite, but translated to different offsets.
    This makes all full cells distinct bytewise, so raw frequency counting is intentionally broken.
  - One cell contains the horizontal or vertical mirror of that sprite, also translated.
  - Solver must crop each cell down to its sprite bbox, recovering:
        [base, base, ..., mirror(base)]
    even though the full cells look like
        [base@offset1, base@offset2, ..., mirror(base)@offsetK].
  - Output is not the full mirror cell. After identifying the mirror anomaly, align its cropped
    sprite with the cropped base sprite and mark every differing pixel with `sp`, every matching
    pixel with `bg` (symmetric difference). Then place that sparse diff sprite back at the mirror
    cell's own offset inside an otherwise-bg cell.

Small concrete example on a 3x3 inner area:
  base bbox sprite      = [[sp, sp],
                           [bg, sp]]
  hflip(base)           = [[sp, sp],
                           [sp, bg]]
  aligned xor diff      = [[bg, bg],
                           [sp, sp]]
  If the mirror cell is placed at offset (1,0), the output row contains only those two bottom
  diff pixels at offset (1,0), not the full mirrored sprite.

b) Complexity calibration
-------------------------
  - Translation is preserved from D2 so the old source-object shortcut is invalidated.
  - Mirror choice is limited to horizontal or vertical flips only; this keeps the transform family
    small enough to infer from 2-4 examples.
  - The generator forces the base sprite to be asymmetric under both flips and to have at least
    two xor-difference pixels with each mirror, so the mirror relation and the sparse output are
    both visually obvious.
  - Output cells are sparse but not trivial: each diff mask has 2+ pixels, and row offsets vary.

c) Implicit-assumption audit against the original solver
--------------------------------------------------------
  - Uniform gridline rows/cols still exist and still use a single lattice color.
  - `bg` is still the most common non-lattice color because the sprite occupies only part of each
    cell interior and every cell also has a 1px bg padding border.
  - Original assumption "exactly one full-cell byte signature appears once" is false. The combo
    generator instead guarantees:
      1. every cell has at least one sprite pixel,
      2. every sprite touches all four sides of its own bbox, so cropping removes translation
         without changing the sprite identity,
      3. cropped row signatures are one majority sprite plus one mirror anomaly,
      4. the anomaly is either `fliplr(base)` or `flipud(base)`.
  - Original assumption "output wants the whole odd cell" is false. New solver assumes the desired
    object is the aligned xor-difference between the majority sprite and the mirrored anomaly,
    embedded back at the anomaly's offset. The generator constructs output exactly that way.

d) Placement legality
---------------------
  - All cells stay inside a regular 1px lattice.
  - Sprite masks are placed only at offsets that keep the entire bbox inside the `(cell_size-2)`
    square inner area.
  - Offsets within a row are sampled without replacement, so full cells do not duplicate.
  - The output diff sprite is pasted back into the mirror cell's bbox, which is guaranteed to fit.

e) Visual observability
-----------------------
  - A human sees repeated translated copies and one reflected copy per row; the odd cell is not
    discoverable by simple raw equality, but the mirror relation is visible after mentally aligning.
  - Output cells keep the odd cell's placement while stripping away all shared structure, leaving
    only the mirror-induced disagreement pixels. This makes the "mirror anomaly + difference
    extraction" composition readable from examples.

f) Visual cleanliness
---------------------
  - There are only three colors: lattice, bg, and sprite.
  - No overlapping annotations or extra markers are introduced.
  - Diff output is sparse and sits in the same cell geometry as the source grid, so the result
    remains tidy rather than cluttered.

g) New assumptions introduced by the combo solver
-------------------------------------------------
  - `_crop_sprite` assumes each cell contains exactly one contiguous bbox-worthy sprite color set;
    generator guarantees non-empty single-color sprites.
  - `_mirror_axis` assumes the anomaly matches either the horizontal or vertical mirror of the
    majority sprite and not both simultaneously; generator uses masks asymmetric under both flips.
  - `_embed_diff_at_anomaly_offset` assumes base and anomaly cropped sprites have identical shapes;
    this holds because mirroring preserves bbox dimensions.
  - The per-row majority is detected on cropped signatures, so generator must keep exactly one
    mirrored anomaly per row and all other cropped sprites identical to the same base sprite.
"""


def _lattice_slices(g: np.ndarray) -> tuple[int, int, list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape
    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == lattice)
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == lattice)

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices
    return H, lattice, row_lines, r_slices, c_slices


def _crop_sprite(cell: np.ndarray, sp: int) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(cell == int(sp))
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    sprite = cell[int(r0) : int(r1), int(c0) : int(c1)].copy()
    return sprite, (int(r0), int(c0))


def _mirror_axis(candidate: np.ndarray, base: np.ndarray) -> str | None:
    if candidate.shape != base.shape:
        return None
    if np.array_equal(candidate, np.fliplr(base)):
        return "h"
    if np.array_equal(candidate, np.flipud(base)):
        return "v"
    return None


def _xor_diff_sprite(base: np.ndarray, other: np.ndarray, bg: int, sp: int) -> np.ndarray:
    assert base.shape == other.shape
    diff = np.full(base.shape, int(bg), dtype=int)
    diff[base != other] = int(sp)
    return diff


def _embed_diff_at_anomaly_offset(
    cell_shape: tuple[int, int], diff_sprite: np.ndarray, offset: tuple[int, int], bg: int
) -> np.ndarray:
    out = np.full(cell_shape, int(bg), dtype=int)
    paste(out, diff_sprite, int(offset[0]), int(offset[1]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Per lattice row, find the translated mirror anomaly and output only the
    aligned sprite-difference pixels, embedded at the anomaly's own offset.
    """
    g = np.array(grid, dtype=int)
    H, lattice, row_lines, r_slices, c_slices = _lattice_slices(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    sprite_colors = sorted(set(map(int, g.flatten())) - {int(lattice), int(bg)})
    assert len(sprite_colors) == 1
    sp = int(sprite_colors[0])

    diff_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        cropped = [_crop_sprite(cell, sp) for cell in cells]
        sprites = [sprite for sprite, _ in cropped]
        sigs = [sprite.tobytes() for sprite in sprites]
        counts = Counter(sigs)

        majority_sig, majority_count = counts.most_common(1)[0]
        assert int(majority_count) == len(cells) - 1
        base_idx = next(i for i, sig in enumerate(sigs) if sig == majority_sig)
        base_sprite = sprites[int(base_idx)]

        odd_idx = next(i for i, sig in enumerate(sigs) if sig != majority_sig)
        odd_sprite, odd_offset = cropped[int(odd_idx)]
        assert _mirror_axis(odd_sprite, base_sprite) is not None

        diff_sprite = _xor_diff_sprite(base_sprite, odd_sprite, bg, sp)
        diff_cells.append(_embed_diff_at_anomaly_offset(cells[int(odd_idx)].shape, diff_sprite, odd_offset, bg))

    out = np.full((H, int(cell_w) + 2), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for r in row_lines:
        out[int(r), :] = int(lattice)
    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = diff_cells[int(i)]
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


def _diff_cell_from_masks(
    base_mask: np.ndarray,
    mirror_mask: np.ndarray,
    inner_size: int,
    odd_offset: tuple[int, int],
    bg: int,
    sp: int,
) -> np.ndarray:
    diff_mask = base_mask != mirror_mask
    diff_inner = _place_mask(diff_mask, inner_size, odd_offset, bg, sp)
    return _make_cell(diff_inner, bg)


def _random_base_mask(rng: np.random.Generator, inner_size: int, n_cols: int) -> np.ndarray | None:
    dims = [
        (h, w)
        for h in range(2, int(inner_size) + 1)
        for w in range(2, int(inner_size) + 1)
        if (int(inner_size) - h + 1) * (int(inner_size) - w + 1) >= int(n_cols)
    ]
    if not dims:
        return None

    for _ in range(300):
        h, w = dims[int(rng.integers(len(dims)))]
        area = int(h * w)
        min_fill = max(3, area // 2)
        max_fill = area - 1
        if min_fill > max_fill:
            continue

        mask = np.zeros((int(h), int(w)), dtype=int)
        on_idx = rng.choice(area, size=int(rng.integers(min_fill, max_fill + 1)), replace=False)
        mask.flat[on_idx] = 1

        # Tight bbox: translation should disappear after cropping.
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue

        hflip = np.fliplr(mask)
        vflip = np.flipud(mask)
        if np.array_equal(mask, hflip) or np.array_equal(mask, vflip):
            continue
        if np.array_equal(hflip, vflip):
            continue

        hdiff = int(np.sum(mask != hflip))
        vdiff = int(np.sum(mask != vflip))
        if min(hdiff, vdiff) < 2:
            continue
        return mask

    return None


def _row_axes(rng: np.random.Generator, n_rows: int) -> list[str]:
    axes = ["h", "v"] * ((int(n_rows) + 1) // 2)
    axes = axes[: int(n_rows)]
    rng.shuffle(axes)
    return axes


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

    axis_masks = {"h": np.fliplr(base_mask), "v": np.flipud(base_mask)}
    offsets = _all_offsets(inner_size, base_mask.shape)
    if len(offsets) < n_cols:
        return None

    in_cells: list[list[np.ndarray]] = []
    out_cells: list[np.ndarray] = []
    used_outputs: set[bytes] = set()
    used_rows: set[tuple[bytes, ...]] = set()

    for axis in _row_axes(rng, n_rows):
        mirror_mask = axis_masks[axis]
        built_row = False
        for _ in range(120):
            row_offset_ids = rng.permutation(len(offsets))[:n_cols]
            row_offsets = [offsets[int(idx)] for idx in row_offset_ids]
            odd_col = int(rng.integers(0, n_cols))

            row_cells: list[np.ndarray] = []
            for col, offset in enumerate(row_offsets):
                mask = mirror_mask if col == odd_col else base_mask
                inner = _place_mask(mask, inner_size, offset, bg, sp)
                row_cells.append(_make_cell(inner, bg))

            # Full-cell uniqueness is the anti-degeneration constraint for the source-object change.
            if len({cell.tobytes() for cell in row_cells}) != n_cols:
                continue

            odd_offset = row_offsets[int(odd_col)]
            diff_cell = _diff_cell_from_masks(base_mask, mirror_mask, inner_size, odd_offset, bg, sp)

            diff_key = diff_cell.tobytes()
            row_key = tuple(cell.tobytes() for cell in row_cells)
            if diff_key in used_outputs or row_key in used_rows:
                continue

            used_outputs.add(diff_key)
            used_rows.add(row_key)
            in_cells.append(row_cells)
            out_cells.append(diff_cell)
            built_row = True
            break

        if not built_row:
            return None

    inp = _build_lattice(in_cells, lattice, bg)
    out = _build_lattice([[cell] for cell in out_cells], lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
