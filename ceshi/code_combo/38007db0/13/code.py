# Combo variant of 38007db0 (index 13)
# combo_spec: D2/1+D4/2+D13/0
# Name: 镜像异常差异提取含干扰

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from arc_utils import paste


"""
Design analysis for the fused variant
=====================================

a) Concretization and behavior trace
------------------------------------
Original task:
- Each lattice row contains one full-cell singleton and the rest are exact copies.
- Solver can count whole-cell byte signatures, pick the singleton, and copy that
  whole cell into the output column.

Fused task:
- Each row now contains:
  - `n_cols-2` majority cells that all use the same sprite mask, but each one is
    translated to a different offset inside its local bg padding.
  - 1 mirror anomaly whose cropped sprite is `hflip(base)` or `vflip(base)`.
  - 1 non-mirror pixel-flip distractor.
- Because offsets differ, all full cells in the row are bytewise distinct. The
  old "count whole-cell singleton" shortcut is intentionally broken.
- After cropping away local translation, the row becomes:
    [base, base, ..., mirror(base), distractor]
  so the majority cropped sprite still reveals the base object.
- Output no longer copies the whole chosen cell. It keeps only the pixels of the
  mirror anomaly that differ from the base sprite, placed back at the anomaly's
  own offset inside a blank cell.

Concrete small trace:
- Suppose a row has four cells:
    base@offsetA, base@offsetB, hflip(base)@offsetC, base+extra_pixels@offsetD
- Original solver sees four distinct full cells and cannot recover the target.
- New solver:
  1. crop each sprite to its tight bbox;
  2. count cropped signatures -> base appears twice, mirror and distractor once;
  3. among the two non-base sprites, keep the one that is a horizontal/vertical
     mirror of the base;
  4. compute `mirror minus base` pixelwise, leaving only visible differing
     sprite pixels in the output cell.

b) Complexity calibration
-------------------------
- D2/1 is made visually meaningful by forcing the base sprite to be asymmetric
  under at least one flip axis and by using multiple translated majority copies.
- D13/0 is realized as exactly two cropped singletons per row, so the solver
  must discriminate instead of picking an arbitrary oddball.
- D4/2 is visible because the chosen mirror is required to have at least two
  mirror-only sprite pixels, so the output is a sparse but non-empty highlight.
- To avoid an explosion of cases, the transform family is restricted to
  horizontal/vertical mirror only, and the distractor is restricted to compact
  bg->sp additions rather than arbitrary noisy edits.

c) Implicit assumption audit against the original main()
--------------------------------------------------------
- Original assumption: whole-cell byte counts identify the source object.
  Status: false. Translations make every full cell unique on purpose.
- Original assumption: there is exactly one singleton per row.
  Status: false. There are two cropped singletons: mirror + distractor.
- Original assumption: the output wants a full copied cell.
  Status: false. The output wants only the mirror/base delta pixels.
- Still valid:
  - lattice rows/cols remain fully uniform and detectable;
  - bg remains the most common non-lattice color;
  - all cells keep the same square size and 1px bg border.

New `main()` assumptions and generator guarantees:
- Each cell contains at least one sprite pixel, so cropping is defined.
- After cropping, each row has one majority sprite repeated `n_cols-2` times.
- Exactly one non-majority sprite is a horizontal or vertical mirror of that
  majority sprite; the other non-majority sprite is not.
- The mirror sprite has the same cropped shape as the base sprite, so the
  per-pixel diff is well-defined after translation normalization.
- The visible diff (`mirror` pixels where `base` is bg) is non-empty and has at
  least two pixels.

d) Placement legality
---------------------
- The lattice is still a regular 1px grid; no object crosses grid lines.
- Sprite masks are placed strictly inside the `(cell_size-2)x(cell_size-2)`
  inner area using legal offsets.
- Row offsets are sampled without replacement, so no two full cells in a row
  coincide accidentally.
- The distractor is created by adding sprite pixels only at bg locations inside
  the inner area, so it cannot erase the base completely or create emptiness.

e) Visual observability
-----------------------
- Across 2-4 examples, a human sees repeated translated copies of one sprite,
  one mirrored copy, and one other "wrong" modified copy per row.
- The output keeps only the sparse difference fragment belonging to the mirrored
  copy, which is directly relatable to the repeated base once the solver notices
  the mirror relation.
- No hidden ordering by color ids, coordinates, or numeric ranking is used.

f) Visual cleanliness
---------------------
- Only three colors remain: lattice, bg, sprite.
- No overlays or markers are added; output cells are blank except for a small
  difference fragment.
- The distractor uses compact local additions so the row remains readable rather
  than turning into clutter.

g) Color/change inferability
----------------------------
- There is no color remapping. The sprite color stays the same everywhere.
- The only transformation to infer is structural:
  "find the mirrored anomaly among multiple odd cells, then subtract the shared
  base sprite and keep only the anomaly's remaining visible pixels."

h) Anti-degeneracy checks to run after implementation
-----------------------------------------------------
- Full-cell uniqueness rate should be 100% per row, proving the original solver
  shortcut is broken.
- Each row should have exactly two cropped singletons and exactly one mirror
  candidate, proving the D13 and D2 controls are active.
- Output cells should be strict sparse subsets of the mirror anomaly rather than
  whole copied cells, proving D4 is active.

i) Main/generator contract
--------------------------
- `main()` identifies the base only from cropped sprite frequency.
- `generate_input()` constructs rows so that cropped-frequency majority exists,
  only one singleton is a mirror, and the mirror/base visible diff is non-empty.
"""


def _extract_lattice_structure(
    g: np.ndarray,
) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
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
    return lattice, row_lines, col_lines, r_slices, c_slices


def _crop_sprite(cell: np.ndarray, sp: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(cell == int(sp))
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return cell[int(r0) : int(r1), int(c0) : int(c1)].copy(), (int(r0), int(c0), int(r1), int(c1))


def _is_mirror_of(candidate: np.ndarray, base: np.ndarray) -> bool:
    return candidate.shape == base.shape and (
        np.array_equal(candidate, np.fliplr(base)) or np.array_equal(candidate, np.flipud(base))
    )


def _diff_cell(base_sprite: np.ndarray, mirror_sprite: np.ndarray, bbox: tuple[int, int, int, int], cell_shape: tuple[int, int], bg: int) -> np.ndarray:
    diff_sprite = np.full(mirror_sprite.shape, int(bg), dtype=int)
    mask = mirror_sprite != base_sprite
    diff_sprite[mask] = mirror_sprite[mask]
    out = np.full(cell_shape, int(bg), dtype=int)
    r0, c0, _r1, _c1 = bbox
    paste(out, diff_sprite, int(r0), int(c0))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Per row:
    - recover the repeated base sprite from cropped-sprite frequency,
    - choose the unique non-base cell whose cropped sprite is a mirror of that base,
    - output only the visible pixel difference between that mirror and the base.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    lattice, row_lines, _col_lines, r_slices, c_slices = _extract_lattice_structure(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    sprite_colors = sorted(set(map(int, g.flatten())) - {int(lattice), int(bg)})
    assert len(sprite_colors) == 1
    sp = int(sprite_colors[0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        cropped = [_crop_sprite(cell, sp) for cell in cells]
        sprites = [sprite for sprite, _bbox in cropped]
        bboxes = [bbox for _sprite, bbox in cropped]
        sigs = [sprite.tobytes() for sprite in sprites]
        counts = Counter(sigs)

        base_sig, base_count = counts.most_common(1)[0]
        assert int(base_count) == len(cells) - 2
        non_base_ids = [idx for idx, sig in enumerate(sigs) if sig != base_sig]
        assert len(non_base_ids) == 2

        base_sprite = next(sprite for sprite, sig in zip(sprites, sigs) if sig == base_sig)
        mirror_ids = [idx for idx in non_base_ids if _is_mirror_of(sprites[idx], base_sprite)]
        assert len(mirror_ids) == 1

        mirror_idx = int(mirror_ids[0])
        chosen_cells.append(_diff_cell(base_sprite, sprites[mirror_idx], bboxes[mirror_idx], cells[mirror_idx].shape, bg))

    out = np.full((H, int(cell_w) + 2), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for r in row_lines:
        out[int(r), :] = int(lattice)
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
    h, w = map(int, sprite_shape)
    return [
        (dr, dc)
        for dr in range(int(inner_size) - h + 1)
        for dc in range(int(inner_size) - w + 1)
    ]


def _place_mask(mask: np.ndarray, inner_size: int, offset: tuple[int, int], bg: int, sp: int) -> np.ndarray:
    inner = np.full((int(inner_size), int(inner_size)), int(bg), dtype=int)
    dr, dc = map(int, offset)
    h, w = mask.shape
    inner[dr : dr + int(h), dc : dc + int(w)][mask.astype(bool)] = int(sp)
    return inner


def _mask_to_sprite(mask: np.ndarray, bg: int, sp: int) -> np.ndarray:
    return np.where(mask.astype(bool), int(sp), int(bg)).astype(int)


def _mirror_visible_count(mask: np.ndarray, axis: str) -> int:
    mirrored = np.fliplr(mask) if axis == "h" else np.flipud(mask)
    return int(np.count_nonzero((mirrored == 1) & (mask == 0)))


def _mirror_overlap_count(mask: np.ndarray, axis: str) -> int:
    mirrored = np.fliplr(mask) if axis == "h" else np.flipud(mask)
    return int(np.count_nonzero((mirrored == 1) & (mask == 1)))


def _random_base_mask(
    rng: np.random.Generator, inner_size: int
) -> tuple[np.ndarray, list[str]] | None:
    dims = [
        (h, w)
        for h in range(2, int(inner_size) + 1)
        for w in range(2, int(inner_size) + 1)
        if (int(inner_size) - h + 1) * (int(inner_size) - w + 1) >= 4
    ]
    if not dims:
        return None

    for _ in range(400):
        h, w = dims[int(rng.integers(len(dims)))]
        area = int(h * w)
        min_fill = max(2, math.ceil(0.35 * area))
        max_fill = min(area - 1, math.floor(0.75 * area))
        if min_fill > max_fill:
            continue

        fill = int(rng.integers(min_fill, max_fill + 1))
        mask = np.zeros((int(h), int(w)), dtype=int)
        on_idx = rng.choice(area, size=fill, replace=False)
        mask.flat[on_idx] = 1

        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue

        axes = [
            axis
            for axis in ("h", "v")
            if _mirror_visible_count(mask, axis) >= 2 and _mirror_overlap_count(mask, axis) >= 1
        ]
        if not axes:
            continue
        return mask, axes

    return None


def _candidate_add_positions(
    inner_size: int, offset: tuple[int, int], sprite_shape: tuple[int, int], base_inner: np.ndarray, bg: int
) -> list[tuple[int, int]]:
    dr, dc = map(int, offset)
    h, w = map(int, sprite_shape)
    positions: list[tuple[int, int]] = []
    for r in range(max(0, dr - 1), min(int(inner_size), dr + h + 1)):
        for c in range(max(0, dc - 1), min(int(inner_size), dc + w + 1)):
            if int(base_inner[r, c]) == int(bg):
                positions.append((int(r), int(c)))
    if positions:
        return positions
    return [(int(r), int(c)) for r in range(int(inner_size)) for c in range(int(inner_size)) if int(base_inner[r, c]) == int(bg)]


def _build_distractor_inner(
    rng: np.random.Generator,
    base_mask: np.ndarray,
    inner_size: int,
    offset: tuple[int, int],
    bg: int,
    sp: int,
) -> np.ndarray | None:
    base_inner = _place_mask(base_mask, inner_size, offset, bg, sp)
    base_sprite = _mask_to_sprite(base_mask, bg, sp)
    positions = _candidate_add_positions(inner_size, offset, base_mask.shape, base_inner, bg)

    for _ in range(120):
        add_count = int(rng.choice([2, 3]))
        if len(positions) < add_count:
            continue
        chosen_ids = rng.choice(len(positions), size=add_count, replace=False)
        inner = base_inner.copy()
        for idx in chosen_ids:
            r, c = positions[int(idx)]
            inner[int(r), int(c)] = int(sp)

        sprite, _bbox = _crop_sprite(_make_cell(inner, bg), sp)
        if np.array_equal(sprite, base_sprite):
            continue
        if _is_mirror_of(sprite, base_sprite):
            continue
        return inner

    return None


def _row_output_cell(base_mask: np.ndarray, axis: str, offset: tuple[int, int], cell_size: int, bg: int, sp: int) -> np.ndarray:
    base_sprite = _mask_to_sprite(base_mask, bg, sp)
    mirror_mask = np.fliplr(base_mask) if axis == "h" else np.flipud(base_mask)
    mirror_sprite = _mask_to_sprite(mirror_mask, bg, sp)
    h, w = base_mask.shape
    dr, dc = map(int, offset)
    bbox = (1 + dr, 1 + dc, 1 + dr + int(h), 1 + dc + int(w))
    return _diff_cell(base_sprite, mirror_sprite, bbox, (int(cell_size), int(cell_size)), bg)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = 6
    n_rows = int(rng.integers(3, 5))
    inner_size = int(cell_size - 2)

    base_pick = _random_base_mask(rng, inner_size)
    if base_pick is None:
        return None
    base_mask, axis_choices = base_pick

    offsets = _all_offsets(inner_size, base_mask.shape)
    if len(offsets) < 4:
        return None
    n_cols = 5 if len(offsets) >= 5 and rng.random() < 0.5 else 4

    in_cells: list[list[np.ndarray]] = []
    out_cells: list[np.ndarray] = []
    used_output_sigs: set[bytes] = set()

    for _row in range(n_rows):
        built_row = False
        for _ in range(120):
            row_offset_ids = rng.permutation(len(offsets))[:n_cols]
            row_offsets = [offsets[int(idx)] for idx in row_offset_ids]
            mirror_col, distractor_col = map(int, rng.choice(n_cols, size=2, replace=False))
            axis = str(axis_choices[int(rng.integers(len(axis_choices)))])

            row_cells: list[np.ndarray] = []
            for col, offset in enumerate(row_offsets):
                if col == mirror_col:
                    mirror_mask = np.fliplr(base_mask) if axis == "h" else np.flipud(base_mask)
                    inner = _place_mask(mirror_mask, inner_size, offset, bg, sp)
                elif col == distractor_col:
                    inner = _build_distractor_inner(rng, base_mask, inner_size, offset, bg, sp)
                    if inner is None:
                        row_cells = []
                        break
                else:
                    inner = _place_mask(base_mask, inner_size, offset, bg, sp)
                row_cells.append(_make_cell(inner, bg))

            if len(row_cells) != n_cols:
                continue
            if len({cell.tobytes() for cell in row_cells}) != n_cols:
                continue

            output_cell = _row_output_cell(base_mask, axis, row_offsets[mirror_col], cell_size, bg, sp)
            output_key = output_cell.tobytes()
            if output_key in used_output_sigs:
                continue

            analysis = _analyze_grid(_build_lattice([row_cells], lattice, bg))
            row_info = analysis["rows"][0]
            if row_info["cropped_base_count"] != n_cols - 2:
                continue
            if row_info["cropped_singletons"] != 2:
                continue
            if row_info["mirror_candidates"] != 1:
                continue
            if row_info["visible_diff_pixels"] < 2:
                continue

            used_output_sigs.add(output_key)
            in_cells.append(row_cells)
            out_cells.append(output_cell)
            built_row = True
            break

        if not built_row:
            return None

    inp = _build_lattice(in_cells, lattice, bg)
    out = _build_lattice([[cell] for cell in out_cells], lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 400 attempts")


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    """
    Lightweight diagnostics used for post-generation auditing.
    """
    g = np.array(grid, dtype=int)
    lattice, _row_lines, _col_lines, r_slices, c_slices = _extract_lattice_structure(g)
    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    bg = int(Counter(bg_candidates).most_common(1)[0][0])
    sprite_colors = sorted(set(map(int, g.flatten())) - {int(lattice), int(bg)})
    sp = int(sprite_colors[0])

    rows: list[dict[str, int]] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        full_sigs = [cell.tobytes() for cell in cells]
        cropped = [_crop_sprite(cell, sp) for cell in cells]
        sprites = [sprite for sprite, _bbox in cropped]
        sigs = [sprite.tobytes() for sprite in sprites]
        counts = Counter(sigs)
        base_sig, base_count = counts.most_common(1)[0]
        base_sprite = next(sprite for sprite, sig in zip(sprites, sigs) if sig == base_sig)
        non_base_ids = [idx for idx, sig in enumerate(sigs) if sig != base_sig]
        mirror_ids = [idx for idx in non_base_ids if _is_mirror_of(sprites[idx], base_sprite)]

        visible_diff_pixels = 0
        if len(mirror_ids) == 1:
            mirror_sprite = sprites[int(mirror_ids[0])]
            visible_diff_pixels = int(np.count_nonzero((mirror_sprite == sp) & (base_sprite != sp)))

        rows.append(
            {
                "full_unique_count": int(len(set(full_sigs))),
                "cropped_distinct_count": int(len(counts)),
                "cropped_base_count": int(base_count),
                "cropped_singletons": int(sum(1 for v in counts.values() if int(v) == 1)),
                "mirror_candidates": int(len(mirror_ids)),
                "visible_diff_pixels": int(visible_diff_pixels),
            }
        )

    return {"rows": rows}
