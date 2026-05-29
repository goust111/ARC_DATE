# Variant of 269e22fb
# Dimension: D2 — 源对象：裁剪窗口的形状 (src)
# Change: 矩形裁剪窗口，行数8-16，列数8-12，面积80-156... → 多个不连通的小裁剪块

from __future__ import annotations

import numpy as np

from arc_utils import colors_present


# Design analysis
# ===============
# spec:
#   preserved_core:
#     motif: the same fixed 20x20 binary pattern, still solved by matching one dihedral variant.
#     colors: exactly 2 colors in the input, and the output repaints the full 20x20 motif with
#       those same two colors.
#     inference_target: recover the motif orientation plus the hidden crop offset inside that
#       orientation.
#   D2_change:
#     original_src: one contiguous HxW crop window.
#     new_src: a hidden HxW crop window that reveals only 2-3 disconnected small rectangles.
#     new_input_pixels:
#       - The full input grid is still one HxW rectangle, with H in [8,16] and W in [8,12].
#       - Unrevealed cells are filled with the motif background color, so they look like blank gaps.
#       - Each revealed block is a true rectangular sub-window copied from the same hidden crop.
#       - Blocks are arranged in one clean axis only:
#         - vertical layout: row-disjoint bands with at least one all-background separator row
#         - horizontal layout: col-disjoint bands with at least one all-background separator col
#   concrete_behavior_trace:
#     original:
#       input = exact full rectangle crop; solver compares the whole HxW boolean window to every
#       possible offset/orientation.
#     variant:
#       input = same HxW outer canvas, but only a few local rectangles are trustworthy evidence.
#       solver first recovers those block rectangles from the candidate foreground color, then
#       compares only those revealed rectangles against every offset/orientation.
#     difference:
#       the unrevealed background-colored cells are no longer evidence about the motif; they are
#       masking gaps. The informative object is now a set of windows rather than one full window.
#   complexity_calibration:
#     - Each block is 3-7 by 3-6, contains both colors, and has at least one foreground cell in
#       every row and every column. This keeps the block visually textured and lets the solver
#       recover its rectangular extent from the correct foreground polarity.
#     - Using 2-3 blocks preserves enough total information for uniqueness without making the
#       puzzle degenerate back into one large crop.
#     - Generator enforces both unique reconstruction and a hidden-area ratio, so the new behavior
#       remains visibly distinct from the original full-rectangle crop.
#   implicit_assumptions_review:
#     - `colors_present(g)` and `assert len(colors) == 2` still hold because the generator uses
#       only the motif foreground color and motif background color.
#     - The old `visible = g == fg` over the whole grid is no longer valid as a full mask because
#       background-colored gaps are not real evidence. The new solver only trusts recovered block
#       rectangles, never the whole canvas.
#     - Matching every offset still assumes all observed pieces come from one single hidden crop in
#       one orientation. The generator guarantees that all blocks are sampled from one shared HxW
#       crop of one dihedral motif variant.
#     - Block extraction assumes the correct foreground polarity yields 2-3 row-bands or col-bands
#       of foreground support, with each block having foreground in every row/column. The generator
#       explicitly constructs blocks to satisfy this.
#   placement_legality:
#     - Blocks never overlap and stay inside the HxW input canvas.
#     - Vertical layouts use disjoint row intervals; horizontal layouts use disjoint col intervals.
#     - A one-cell separator gap prevents accidental merging in the extracted foreground support.
#   visual_observability:
#     - Humans see several separated “patches” of the motif instead of one solid crop, which is a
#       clear visual departure from the original task.
#     - The relative spacing of these patches inside the hidden crop is meaningful and repeated
#       across examples, so the multi-window rule is inferable from a few samples.


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
    Recover candidate revealed rectangles from a foreground mask.

    Assumption checked here: the correct polarity produces 2-3 separated row bands or column
    bands, and inside each block every row/column contains at least one foreground cell.
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


def _visible_block_ok(block: np.ndarray) -> bool:
    return bool(
        block.any()
        and (~block).any()
        and block.any(axis=1).all()
        and block.any(axis=0).all()
    )


def _row_interval_options(crop: np.ndarray, r0: int, r1: int) -> list[tuple[int, int, int, int]]:
    H, W = crop.shape
    del H
    options: list[tuple[int, int, int, int]] = []
    for c0 in range(W):
        max_w = min(6, W - c0)
        for w in range(3, max_w + 1):
            c1 = c0 + w - 1
            if _visible_block_ok(crop[r0 : r1 + 1, c0 : c1 + 1]):
                options.append((r0, c0, r1, c1))
    return options


def _col_interval_options(crop: np.ndarray, c0: int, c1: int) -> list[tuple[int, int, int, int]]:
    H, W = crop.shape
    del W
    options: list[tuple[int, int, int, int]] = []
    for r0 in range(H):
        max_h = min(7, H - r0)
        for h in range(3, max_h + 1):
            r1 = r0 + h - 1
            if _visible_block_ok(crop[r0 : r1 + 1, c0 : c1 + 1]):
                options.append((r0, c0, r1, c1))
    return options


def _sample_vertical_blocks(
    crop: np.ndarray, rng: np.random.Generator
) -> list[tuple[int, int, int, int]] | None:
    H, _ = crop.shape
    max_blocks = min(3, (H + 1) // 4)
    if max_blocks < 2:
        return None
    n_blocks = int(rng.integers(2, max_blocks + 1))

    for _ in range(100):
        used_rows: list[tuple[int, int]] = []
        for _ in range(n_blocks):
            span_h = int(rng.integers(3, min(7, H) + 1))
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

        blocks: list[tuple[int, int, int, int]] = []
        ok = True
        for r0, r1 in used_rows:
            options = _row_interval_options(crop, r0, r1)
            if not options:
                ok = False
                break
            blocks.append(options[int(rng.integers(len(options)))])
        if ok:
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

    for _ in range(100):
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

        blocks: list[tuple[int, int, int, int]] = []
        ok = True
        for c0, c1 in used_cols:
            options = _col_interval_options(crop, c0, c1)
            if not options:
                ok = False
                break
            blocks.append(options[int(rng.integers(len(options)))])
        if ok:
            return blocks
    return None


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from several disconnected crop fragments in one hidden window.

    Input structure:
    - The grid is a two-color HxW canvas cut from one dihedral variant of the shared 20x20 motif.
    - Only 2-3 small rectangular sub-windows are actually revealed; the rest of the canvas is filled
      with the motif background color, creating separated visible blocks.

    Rule:
    - Try each input color as the motif foreground, and recover the candidate visible rectangles from
      that polarity by looking for 2-3 separated row-bands or column-bands of foreground support.
    - For every dihedral orientation and every possible HxW offset, require each recovered block to
      match the motif exactly at the corresponding location, while ignoring masked gaps.
    - Repaint the uniquely matched full 20x20 motif with the discovered foreground/background colors.

    Key insight:
    - The input is no longer one continuous crop; it is a sparse set of local windows whose internal
      textures and relative spacing jointly identify the hidden crop position and motif orientation.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    base = _pattern_mask()
    matches: dict[tuple[int, int, int, int, int], tuple[int, int, np.ndarray]] = {}
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for blocks in _extract_block_sets(visible):
            for variant_idx, pattern in enumerate(_dihedral_variants(base)):
                for r in range(20 - H + 1):
                    for c in range(20 - W + 1):
                        ok = True
                        for r0, c0, r1, c1 in blocks:
                            if not np.array_equal(
                                pattern[r + r0 : r + r1 + 1, c + c0 : c + c1 + 1],
                                visible[r0 : r1 + 1, c0 : c1 + 1],
                            ):
                                ok = False
                                break
                        if ok:
                            matches[(int(fg), int(bg), variant_idx, r, c)] = (int(fg), int(bg), pattern)

    assert len(matches) == 1
    fg, bg, pattern = next(iter(matches.values()))
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    """
    Output-first generator for the disconnected-block variant.

    New assumptions enforced here:
    - Exactly 2-3 revealed rectangles.
    - Hidden gaps occupy a non-trivial fraction of the crop, so the variant stays visually distinct.
    - The generated sparse evidence yields exactly one (polarity, orientation, offset) match.
    """
    rng = np.random.default_rng()
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    for _ in range(2000):
        fg, bg = map(int, rng.choice(np.arange(10), size=2, replace=False))
        pattern = variants[int(rng.integers(len(variants)))]
        full_output = np.where(pattern, fg, bg).astype(int)

        H = int(rng.integers(8, 17))
        W = int(rng.integers(8, 13))
        crop_top = int(rng.integers(0, 21 - H))
        crop_left = int(rng.integers(0, 21 - W))
        crop = pattern[crop_top : crop_top + H, crop_left : crop_left + W]

        blocks = None
        if int(rng.integers(2)) == 0:
            blocks = _sample_vertical_blocks(crop, rng)
            if blocks is None:
                blocks = _sample_horizontal_blocks(crop, rng)
        else:
            blocks = _sample_horizontal_blocks(crop, rng)
            if blocks is None:
                blocks = _sample_vertical_blocks(crop, rng)
        if blocks is None:
            continue

        total_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in blocks)
        if total_area < 28:
            continue
        if total_area * 10 > 7 * H * W:
            continue

        inp = _compose_input((H, W), blocks, full_output, crop_top, crop_left, bg)
        try:
            out = main(inp)
        except Exception:
            continue
        if np.array_equal(out, full_output):
            return inp

    raise RuntimeError("failed to generate a unique disconnected-block crop")
