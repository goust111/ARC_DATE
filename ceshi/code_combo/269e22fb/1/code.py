# Combo variant of 269e22fb (index 1)
# combo_spec: D1/0+D2/2
# Name: 多窗口分散裁剪联合匹配

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# ===============
# spec:
#   fused_rule:
#     preserved_core:
#       - output is still one dihedral variant of the same fixed 20x20 binary motif
#       - output colors are exactly the two colors visible in the input
#       - solver still has to infer orientation plus hidden crop offset
#     fusion:
#       - D2/2 contributes the visible form: the input is a sparse two-color crop made of
#         2-3 disconnected small rectangular blocks, with gaps painted in the motif
#         background color
#       - D1/0 contributes the match condition: every visible block is individually
#         non-unique across all motif variants/offsets, but the blocks together determine
#         a single global orientation+offset
#   concrete_pixel_picture:
#     example:
#       - imagine a 12x10 hidden crop of one motif orientation
#       - only three rectangles are shown, e.g. a 4x4 block at rows 0:3 cols 1:4,
#         a 3x5 block at rows 5:7 cols 3:7, and a 4x3 block at rows 8:11 cols 0:2
#       - all other cells are painted with the motif background color, so the eye sees
#         three separated patches rather than one solid crop
#     original_behavior:
#       - treat the whole HxW rectangle as observed evidence
#       - one crop alone uniquely identifies orientation and offset
#     fused_behavior:
#       - first recover 2-3 trustworthy rectangles from the candidate foreground polarity
#       - ignore the background-colored gaps as hidden area rather than evidence
#       - each rectangle may appear in multiple motif locations, but all rectangles must
#         align under one shared outer-crop offset
#     visual_difference:
#       - the task no longer looks like one contiguous cut-out; it looks like several
#         separated motif fragments whose spacing matters
#   complexity_calibration:
#     - each block is kept small (3-6 by 3-6) so standalone ambiguity is common
#     - each block must contain both colors and at least one foreground cell in every
#       row and column, which makes its rectangular extent recoverable from foreground
#       support
#     - generator rejects samples unless every block has >=2 standalone matches and the
#       whole block set has exactly 1 joint match
#     - generator also rejects samples if the entire sparse HxW canvas would itself be a
#       unique exact crop; this prevents quiet regression to the original single-window
#       puzzle despite the new appearance
#   implicit_assumptions_review:
#     - `len(colors) == 2` still holds because the fused variant uses only motif fg/bg
#     - `visible = g == fg` is no longer "the full crop"; it is only a cue for extracting
#       candidate revealed blocks
#     - `_extract_block_sets(...)` assumes the true blocks lie in 2-3 separated row-bands
#       or 2-3 separated column-bands, and each block has foreground support in every row
#       and column; generator constructs blocks to satisfy this
#     - the nested match loop assumes all blocks come from one single hidden HxW crop, so
#       one orientation and one outer offset must satisfy every block simultaneously
#     - `assert len(matches) == 1` is safe only because generation enforces joint
#       uniqueness after deduplicating repeated parses of the same candidate
#   placement_legality:
#     - all blocks stay inside the chosen HxW crop
#     - blocks never overlap and always have at least one empty separator row or column
#       along the chosen layout axis, preventing accidental merge in extraction
#     - blocks also overlap on the orthogonal axis, so humans and the solver recover one
#       clean layout axis instead of two competing decompositions
#     - visible area is capped so the input stays recognizably sparse instead of becoming
#       a nearly solid crop
#   visual_observability_and_neatness:
#     - the few rectangular patches are easy to spot by eye and do not overlap
#     - using only two colors avoids arbitrary color-role inference; the novel part is the
#       spatial sparsity and joint matching, which can be inferred from 2-4 examples
#     - the gaps are visually clean because they reuse the motif background color instead
#       of layering extra markers
#   new_solver_generator_contract:
#     - solver expects 2-3 recovered blocks from at least one foreground polarity
#     - generator guarantees that the correct polarity yields such a block set, every block
#       is locally ambiguous, the full set is jointly unique, and the sparse input is not
#       already solvable as one exact dense crop


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


_VARIANTS = _dihedral_variants(_pattern_mask())


def _build_single_window_counts() -> dict[tuple[int, int], dict[bytes, int]]:
    counts: dict[tuple[int, int], dict[bytes, int]] = {}
    for h in range(3, 7):
        for w in range(3, 7):
            bucket: dict[bytes, int] = {}
            for pattern in _VARIANTS:
                for r in range(20 - h + 1):
                    for c in range(20 - w + 1):
                        key = pattern[r : r + h, c : c + w].tobytes()
                        bucket[key] = bucket.get(key, 0) + 1
            counts[h, w] = bucket
    return counts


_SINGLE_WINDOW_COUNTS = _build_single_window_counts()


def _intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for value in map(int, idx[1:]):
        if value != prev + 1:
            runs.append((start, prev))
            start = value
        prev = value
    runs.append((start, prev))
    return runs


def _dedup_block_sets(block_sets: list[list[tuple[int, int, int, int]]]) -> list[list[tuple[int, int, int, int]]]:
    seen: set[tuple[tuple[int, int, int, int], ...]] = set()
    out: list[list[tuple[int, int, int, int]]] = []
    for blocks in block_sets:
        key = tuple(blocks)
        if key not in seen:
            seen.add(key)
            out.append(blocks)
    return out


def _extract_block_sets(mask: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    """
    Recover candidate visible rectangles from a foreground mask.

    The correct polarity is generated so that foreground support forms 2-3 separated row
    bands or column bands, and each band expands to one rectangular block.
    """
    candidates: list[list[tuple[int, int, int, int]]] = []

    row_runs = _intervals(mask.any(axis=1))
    if 2 <= len(row_runs) <= 3:
        blocks: list[tuple[int, int, int, int]] = []
        ok = True
        for r0, r1 in row_runs:
            cols = np.flatnonzero(mask[r0 : r1 + 1].any(axis=0))
            if len(cols) == 0 or np.any(np.diff(cols) > 1):
                ok = False
                break
            c0, c1 = int(cols[0]), int(cols[-1])
            sub = mask[r0 : r1 + 1, c0 : c1 + 1]
            if not sub.any(axis=1).all() or not sub.any(axis=0).all():
                ok = False
                break
            blocks.append((r0, c0, r1, c1))
        if ok:
            candidates.append(blocks)

    col_runs = _intervals(mask.any(axis=0))
    if 2 <= len(col_runs) <= 3:
        blocks = []
        ok = True
        for c0, c1 in col_runs:
            rows = np.flatnonzero(mask[:, c0 : c1 + 1].any(axis=1))
            if len(rows) == 0 or np.any(np.diff(rows) > 1):
                ok = False
                break
            r0, r1 = int(rows[0]), int(rows[-1])
            sub = mask[r0 : r1 + 1, c0 : c1 + 1]
            if not sub.any(axis=1).all() or not sub.any(axis=0).all():
                ok = False
                break
            blocks.append((r0, c0, r1, c1))
        if ok:
            candidates.append(blocks)

    return _dedup_block_sets(candidates)


def _count_single_window_matches(visible: np.ndarray) -> int:
    h, w = map(int, visible.shape)
    bucket = _SINGLE_WINDOW_COUNTS.get((h, w))
    if bucket is not None:
        return bucket.get(visible.tobytes(), 0)

    total = 0
    for pattern in _VARIANTS:
        for r in range(20 - h + 1):
            for c in range(20 - w + 1):
                total += int(np.array_equal(pattern[r : r + h, c : c + w], visible))
    return total


def _count_full_crop_matches(visible: np.ndarray) -> int:
    h, w = map(int, visible.shape)
    total = 0
    for pattern in _VARIANTS:
        for r in range(20 - h + 1):
            for c in range(20 - w + 1):
                total += int(np.array_equal(pattern[r : r + h, c : c + w], visible))
    return total


def _count_joint_matches(
    outer_shape: tuple[int, int],
    blocks: list[tuple[int, int, int, int]],
    visible_patches: list[np.ndarray],
) -> int:
    H, W = map(int, outer_shape)
    total = 0
    for pattern in _VARIANTS:
        for top in range(20 - H + 1):
            for left in range(20 - W + 1):
                ok = True
                for (r0, c0, r1, c1), patch in zip(blocks, visible_patches):
                    if not np.array_equal(pattern[top + r0 : top + r1 + 1, left + c0 : left + c1 + 1], patch):
                        ok = False
                        break
                total += int(ok)
    return total


def _all_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    matches: dict[tuple[int, int, int, int, int], tuple[int, int, np.ndarray]] = {}
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for blocks in _extract_block_sets(visible):
            for variant_idx, pattern in enumerate(_VARIANTS):
                for top in range(20 - H + 1):
                    for left in range(20 - W + 1):
                        ok = True
                        for r0, c0, r1, c1 in blocks:
                            if not np.array_equal(
                                pattern[top + r0 : top + r1 + 1, left + c0 : left + c1 + 1],
                                visible[r0 : r1 + 1, c0 : c1 + 1],
                            ):
                                ok = False
                                break
                        if ok:
                            matches[(int(fg), int(bg), variant_idx, top, left)] = (int(fg), int(bg), pattern)
    return list(matches.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from several scattered crop blocks that belong to one hidden window.

    Input structure:
    - The grid is a two-color HxW crop-sized canvas from one rotation/reflection of the fixed 20x20 motif.
    - Only 2-3 small rectangular pieces are genuinely revealed; the rest of the canvas is filled with the motif background color, so those gap cells are not trustworthy evidence.
    - The revealed pieces all come from one shared hidden crop and preserve their relative spacing inside that crop.

    Rule:
    - Try each input color as the motif foreground and recover candidate block rectangles from that polarity by finding 2-3 separated foreground bands.
    - For every dihedral motif variant and every possible HxW crop offset, require every recovered block to match exactly at its own coordinates while ignoring the background-colored gaps.
    - Accept the unique jointly consistent candidate and repaint the entire 20x20 motif with the recovered foreground/background colors.

    Key insight:
    - Any one small block can recur elsewhere in the motif, but the collection of blocks carries a shared spacing pattern that locks all of them onto one orientation and one global crop position.
    """
    matches = _all_matches(np.asarray(grid, dtype=int))
    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _visible_block_ok(block: np.ndarray) -> bool:
    return bool(
        block.any()
        and (~block).any()
        and block.any(axis=1).all()
        and block.any(axis=0).all()
    )


def _row_interval_options(crop: np.ndarray, r0: int, r1: int) -> list[tuple[int, int, int, int]]:
    _, W = crop.shape
    options: list[tuple[int, int, int, int]] = []
    for c0 in range(W):
        max_w = min(6, W - c0)
        for w in range(3, max_w + 1):
            c1 = c0 + w - 1
            if _visible_block_ok(crop[r0 : r1 + 1, c0 : c1 + 1]):
                options.append((r0, c0, r1, c1))
    return options


def _choose_with_shared_column(
    options_per_band: list[list[tuple[int, int, int, int]]],
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]] | None:
    shared_cols: set[int] | None = None
    for options in options_per_band:
        covered = {
            col
            for _, c0, _, c1 in options
            for col in range(c0, c1 + 1)
        }
        shared_cols = covered if shared_cols is None else shared_cols & covered
        if not shared_cols:
            return None

    anchor = int(rng.choice(sorted(shared_cols)))
    chosen = [
        band_options[int(rng.integers(len(band_options)))]
        for band_options in [
            [option for option in options if option[1] <= anchor <= option[3]]
            for options in options_per_band
        ]
    ]
    return chosen


def _col_interval_options(crop: np.ndarray, c0: int, c1: int) -> list[tuple[int, int, int, int]]:
    H, _ = crop.shape
    options: list[tuple[int, int, int, int]] = []
    for r0 in range(H):
        max_h = min(6, H - r0)
        for h in range(3, max_h + 1):
            r1 = r0 + h - 1
            if _visible_block_ok(crop[r0 : r1 + 1, c0 : c1 + 1]):
                options.append((r0, c0, r1, c1))
    return options


def _choose_with_shared_row(
    options_per_band: list[list[tuple[int, int, int, int]]],
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]] | None:
    shared_rows: set[int] | None = None
    for options in options_per_band:
        covered = {
            row
            for r0, _, r1, _ in options
            for row in range(r0, r1 + 1)
        }
        shared_rows = covered if shared_rows is None else shared_rows & covered
        if not shared_rows:
            return None

    anchor = int(rng.choice(sorted(shared_rows)))
    chosen = [
        band_options[int(rng.integers(len(band_options)))]
        for band_options in [
            [option for option in options if option[0] <= anchor <= option[2]]
            for options in options_per_band
        ]
    ]
    return chosen


def _sample_vertical_blocks(
    crop: np.ndarray, rng: np.random.Generator
) -> list[tuple[int, int, int, int]] | None:
    H, _ = crop.shape
    max_blocks = min(3, (H + 1) // 4)
    if max_blocks < 2:
        return None
    n_blocks = int(rng.integers(2, max_blocks + 1))

    for _ in range(120):
        used_rows: list[tuple[int, int]] = []
        for _ in range(n_blocks):
            span_h = int(rng.integers(3, min(6, H) + 1))
            starts = [
                r0
                for r0 in range(H - span_h + 1)
                if all(r0 + span_h - 1 < a - 1 or r0 > b + 1 for a, b in used_rows)
            ]
            if not starts:
                break
            r0 = int(rng.choice(starts))
            used_rows.append((r0, r0 + span_h - 1))
        if len(used_rows) != n_blocks:
            continue
        used_rows.sort()
        if any(used_rows[i][1] + 1 >= used_rows[i + 1][0] for i in range(len(used_rows) - 1)):
            continue

        options_per_band: list[list[tuple[int, int, int, int]]] = []
        for r0, r1 in used_rows:
            options = _row_interval_options(crop, r0, r1)
            if not options:
                options_per_band = []
                break
            options_per_band.append(options)
        if options_per_band:
            blocks = _choose_with_shared_column(options_per_band, rng)
            if blocks is not None:
                return blocks
    return None


def _sample_horizontal_blocks(
    crop: np.ndarray, rng: np.random.Generator
) -> list[tuple[int, int, int, int]] | None:
    _, W = crop.shape
    max_blocks = min(3, (W + 1) // 4)
    if max_blocks < 2:
        return None
    n_blocks = int(rng.integers(2, max_blocks + 1))

    for _ in range(120):
        used_cols: list[tuple[int, int]] = []
        for _ in range(n_blocks):
            span_w = int(rng.integers(3, min(6, W) + 1))
            starts = [
                c0
                for c0 in range(W - span_w + 1)
                if all(c0 + span_w - 1 < a - 1 or c0 > b + 1 for a, b in used_cols)
            ]
            if not starts:
                break
            c0 = int(rng.choice(starts))
            used_cols.append((c0, c0 + span_w - 1))
        if len(used_cols) != n_blocks:
            continue
        used_cols.sort()
        if any(used_cols[i][1] + 1 >= used_cols[i + 1][0] for i in range(len(used_cols) - 1)):
            continue

        options_per_band: list[list[tuple[int, int, int, int]]] = []
        for c0, c1 in used_cols:
            options = _col_interval_options(crop, c0, c1)
            if not options:
                options_per_band = []
                break
            options_per_band.append(options)
        if options_per_band:
            blocks = _choose_with_shared_row(options_per_band, rng)
            if blocks is not None:
                return blocks
    return None


def _sample_blocks(crop: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int, int, int]] | None:
    if int(rng.integers(2)) == 0:
        blocks = _sample_vertical_blocks(crop, rng)
        if blocks is None:
            blocks = _sample_horizontal_blocks(crop, rng)
        return blocks
    blocks = _sample_horizontal_blocks(crop, rng)
    if blocks is None:
        blocks = _sample_vertical_blocks(crop, rng)
    return blocks


def _compose_input(
    crop_shape: tuple[int, int],
    blocks: list[tuple[int, int, int, int]],
    full_output: np.ndarray,
    crop_top: int,
    crop_left: int,
    bg: int,
) -> np.ndarray:
    H, W = crop_shape
    inp = np.full((H, W), int(bg), dtype=int)
    for r0, c0, r1, c1 in blocks:
        inp[r0 : r1 + 1, c0 : c1 + 1] = full_output[
            crop_top + r0 : crop_top + r1 + 1,
            crop_left + c0 : crop_left + c1 + 1,
        ]
    return inp


def generate_input() -> np.ndarray:
    """
    Output-first generator for the fused scattered-block joint-matching variant.

    Enforced behavior:
    - Exactly 2-3 disconnected visible rectangles inside one hidden HxW crop.
    - Every rectangle is locally ambiguous across motif placements.
    - The rectangles jointly yield exactly one match.
    - Treating the whole sparse HxW canvas as an exact crop does not give a unique match,
      so the sample genuinely requires the fused sparse-window reasoning.
    """
    rng = np.random.default_rng()
    palette = np.array(Color.ALL_COLORS, dtype=int)

    for _ in range(500):
        fg, bg = map(int, rng.choice(palette, size=2, replace=False))
        pattern = _VARIANTS[int(rng.integers(len(_VARIANTS)))]
        full_output = np.where(pattern, int(fg), int(bg)).astype(int)

        H = int(rng.integers(8, 17))
        W = int(rng.integers(8, 13))
        crop_top = int(rng.integers(0, 21 - H))
        crop_left = int(rng.integers(0, 21 - W))
        crop = pattern[crop_top : crop_top + H, crop_left : crop_left + W]

        for _ in range(200):
            blocks = _sample_blocks(crop, rng)
            if blocks is None:
                continue

            total_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in blocks)
            if total_area < 24:
                continue
            if total_area * 10 > 6 * H * W:
                continue

            visible_patches = [crop[r0 : r1 + 1, c0 : c1 + 1] for r0, c0, r1, c1 in blocks]
            single_counts = [_count_single_window_matches(patch) for patch in visible_patches]
            if min(single_counts) < 2:
                continue
            if _count_joint_matches((H, W), blocks, visible_patches) != 1:
                continue

            inp = _compose_input((H, W), blocks, full_output, crop_top, crop_left, bg)
            if _count_full_crop_matches(inp == int(fg)) == 1:
                continue

            out = main(inp)
            if np.array_equal(out, full_output):
                return inp

    raise RuntimeError("failed to generate a fused jointly unique scattered-block crop")
