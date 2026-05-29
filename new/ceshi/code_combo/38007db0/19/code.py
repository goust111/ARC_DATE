# Combo variant of 38007db0 (index 19)
# combo_spec: D2/1+D4/3
# Name: 镜像异常旋转输出

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np

from arc_utils import paste


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
   Original task:
   - Each lattice row contains repeated full cells plus one bytewise-unique cell.
   - The solver can count whole-cell signatures and copy the unique cell unchanged.
   Fusion task:
   - Each row still has one conceptual oddball, but all full cells are translated inside their local
     bg padding, so raw cell bytes are intentionally all different.
   - After cropping the sprite bbox, the row becomes [base, base, ..., mirror(base)] where the anomaly
     is either a horizontal or vertical mirror of the majority sprite.
   - The solver must identify the mirrored anomaly first, then rotate the entire square cell 90 degrees
     counterclockwise before writing it into the single-column output lattice.
   Example row trace:
   - Input full cells: [base@offset(0,0), mirror(base)@offset(1,0), base@offset(0,1)]
     -> whole-cell counts are {1,1,1}, so the original solver fails.
   - Crop sprites: [base, mirror(base), base]
     -> identify the middle cell as the mirror anomaly.
   - Output cell: rot90_ccw(full middle cell).

b) Complexity calibration
   - Mirror type is limited to {horizontal, vertical}; this is enough to force relational reasoning
     without creating too many transform classes.
   - Translation inside the cell destroys the old full-byte shortcut on purpose.
   - The generator forces the sprite mask to be asymmetric under both flips and under 90-degree rotation,
     so both the mirror anomaly and the final rotation are visually observable.
   - Offsets are reused only within the legal inner-area range and sampled without replacement per row,
     keeping rows readable while preserving the "all full cells distinct" property.

c) Hidden assumptions review against the original main()
   - Uniform gridline rows/cols still exist; generator preserves 1px lattice lines.
   - Background can still be inferred as the most common non-lattice color; generator keeps the sprite
     sparse enough that bg dominates the non-lattice area.
   - Original assumption "one full cell appears exactly once" is broken; the new solver replaces it with:
       1. every cell contains at least one sprite pixel,
       2. cropping to the sprite bbox removes translation noise,
       3. after cropping, exactly one sprite is a flip of the repeated majority sprite.
   - Rotation adds a new square-cell assumption. The original spec already constrains cells to 5x5 or 6x6,
     and the generator keeps them square.
   - The new loop that finds the majority cropped sprite assumes counts {n_cols-1, 1}; the generator
     explicitly constructs each row that way.

d) Placement legality
   - Each sprite mask is placed fully inside the inner `(cell_size-2) x (cell_size-2)` area.
   - Candidate mask dimensions are chosen so the inner area supports at least `n_cols` distinct offsets.
   - Row offsets are sampled without replacement, so cells in one row never overlap in interpretation and
     never duplicate as whole-cell arrays.
   - Rotating a square cell preserves shape, so the output lattice geometry remains valid.

e) Visual observability
   - The cropped sprite touches all four sides of its bbox, so the mirrored shape is easy to compare.
   - The mask is asymmetric under both flips and rotation, making the anomaly and post-extraction rotation
     both visible in 2-4 examples.
   - Rows mix horizontal and vertical mirror anomalies when possible, giving repeated evidence that the
     anomaly is "mirrored relative to the row majority" rather than an arbitrary edited pattern.

f) Visual cleanliness
   - No extra colors, markers, or overlays are added.
   - The only changes from the original visual language are local sprite translations, one mirrored anomaly
     per row, and the final rotated output column.

g) Inferability of changes
   - There is no color remapping, so the human only needs to infer two structural steps:
       1. find the cell whose sprite is the mirrored version of the row's repeated sprite,
       2. rotate that extracted cell 90 degrees counterclockwise.
   - Because the input rows repeat the same majority sprite several times and the outputs repeat the same
     fixed rotation rule across rows, the combined rule is recoverable from a few examples.

New solver assumptions introduced by the combo
----------------------------------------------
- All cells contain exactly one connected sprite color distinct from lattice and bg; cropping relies on at
  least one sprite pixel being present.
- The mirrored anomaly is always one of {fliplr(base), flipud(base)} after bbox cropping.
- The cropped majority sprite appears exactly `n_cols - 1` times in each row.
- The rotated anomaly cell is visually different from the unrotated anomaly cell; the generator enforces
  that so the D4 component does real work instead of collapsing to identity.
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


def _rot90_ccw(cell: np.ndarray) -> np.ndarray:
    return np.rot90(cell, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the mirror-anomaly cell from each lattice row, then rotate it 90 degrees counterclockwise
    into a single-column lattice.
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
    cell_w = int(c_slices[0][1] - c_slices[0][0])
    assert all(int(rb - ra) == int(cell_h) for ra, rb in r_slices)
    assert all(int(cb - ca) == int(cell_w) for ca, cb in c_slices)
    assert int(cell_h) == int(cell_w)

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    sprite_colors = sorted(set(map(int, g.flatten())) - {int(L), int(bg)})
    assert len(sprite_colors) == 1
    sp = int(sprite_colors[0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sprites = [_crop_sprite(cell, sp) for cell in cells]
        sigs = [sprite.tobytes() for sprite in sprites]
        counts = Counter(sigs)

        majority_sig, majority_count = counts.most_common(1)[0]
        assert int(majority_count) == len(cells) - 1
        base_sprite = next(sprite for sprite, sig in zip(sprites, sigs) if sig == majority_sig)

        mirror_idx = next(
            (
                idx
                for idx, sprite in enumerate(sprites)
                if sigs[idx] != majority_sig and _is_mirror_of(sprite, base_sprite)
            ),
            None,
        )
        assert mirror_idx is not None
        chosen_cells.append(_rot90_ccw(cells[int(mirror_idx)]))

    out = np.full((H, int(cell_w) + 2), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)
    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = chosen_cells[int(i)]
    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    size = inner.shape[0] + 2
    cell = np.full((size, size), int(bg), dtype=int)
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


@lru_cache(maxsize=None)
def _candidate_base_masks(inner_size: int, n_cols: int) -> tuple[tuple[int, int, bytes], ...]:
    candidates: list[tuple[int, int, bytes]] = []
    for h in range(2, int(inner_size) + 1):
        for w in range(2, int(inner_size) + 1):
            if (int(inner_size) - h + 1) * (int(inner_size) - w + 1) < int(n_cols):
                continue
            area = int(h * w)
            fill_low = max(3, (area + 1) // 2)
            fill_high = max(fill_low, area - 1)
            for bits in range(1 << area):
                fill = bits.bit_count()
                if fill < fill_low or fill > fill_high:
                    continue
                flat = np.array([(bits >> idx) & 1 for idx in range(area)], dtype=int)
                mask = flat.reshape((int(h), int(w)))
                if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
                    continue
                if np.array_equal(mask, np.rot90(mask, 1)):
                    continue
                if np.array_equal(np.fliplr(mask), np.flipud(mask)):
                    continue
                candidates.append((int(h), int(w), mask.astype(np.uint8).tobytes()))
    return tuple(candidates)


def _random_base_mask(rng: np.random.Generator, inner_size: int, n_cols: int) -> np.ndarray | None:
    candidates = _candidate_base_masks(int(inner_size), int(n_cols))
    if not candidates:
        return None
    h, w, raw = candidates[int(rng.integers(len(candidates)))]
    return np.frombuffer(raw, dtype=np.uint8).astype(int).reshape((int(h), int(w)))


def _row_is_valid(
    row_cells: list[np.ndarray],
    odd_idx: int,
    base_mask: np.ndarray,
    mirror_mask: np.ndarray,
    sp: int,
) -> bool:
    if len({cell.tobytes() for cell in row_cells}) != len(row_cells):
        return False

    sprites = [_crop_sprite(cell, sp) for cell in row_cells]
    sigs = [sprite.tobytes() for sprite in sprites]
    counts = Counter(sigs)
    majority_sig, majority_count = counts.most_common(1)[0]
    if int(majority_count) != len(row_cells) - 1:
        return False

    base_sprite = next(sprite for sprite, sig in zip(sprites, sigs) if sig == majority_sig)
    odd_sprite = sprites[int(odd_idx)]
    odd_cell = row_cells[int(odd_idx)]
    return (
        np.array_equal(base_sprite, base_mask)
        and np.array_equal(odd_sprite, mirror_mask)
        and _is_mirror_of(odd_sprite, base_sprite)
        and not np.array_equal(_rot90_ccw(odd_cell), odd_cell)
    )


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
    for row in range(n_rows):
        mirror_mask = axis_masks[row_axes[row]]
        built_row = False
        for _ in range(240):
            row_offset_ids = rng.permutation(len(offsets))[:n_cols]
            row_offsets = [offsets[int(idx)] for idx in row_offset_ids]
            odd_col = int(rng.integers(0, n_cols))

            row_cells: list[np.ndarray] = []
            for col, offset in enumerate(row_offsets):
                mask = mirror_mask if col == odd_col else base_mask
                inner = _place_mask(mask, inner_size, offset, bg, sp)
                row_cells.append(_make_cell(inner, bg))

            if not _row_is_valid(row_cells, odd_col, base_mask, mirror_mask, sp):
                continue

            rotated_odd = _rot90_ccw(row_cells[odd_col])
            chosen_cells.append(rotated_odd.copy())
            in_cells.append(row_cells)
            built_row = True
            break

        if not built_row:
            return None

    out = _build_lattice([[cell] for cell in chosen_cells], lattice, bg)
    inp = _build_lattice(in_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 1000 attempts")
