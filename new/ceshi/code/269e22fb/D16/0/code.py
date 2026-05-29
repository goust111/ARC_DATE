# Variant of 269e22fb
# Dimension: D16 — 输入-输出基数关系 (meta)
# Change: 1:1关系——一个裁剪窗口对应一个完整图案输出... → 多输入一输出（多个裁剪窗口拼成一个完整图案）

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
from arc_utils import colors_present


# Design analysis
# ===============
# spec:
#   preserved_core:
#     - Output is still one 20x20 dihedral variant of the same fixed binary motif.
#     - The motif is still recolored with exactly two task colors.
#     - Solving still fundamentally means matching observed local binary windows against
#       the eight dihedral orientations of the same hard-coded 20x20 mask.
#   D16_change:
#     - The input is no longer one crop.
#     - Instead it contains four separate crop-windows laid out side by side, with
#       full-height separator columns in a third "mask" color.
#     - Each frame is a 20-row carrier panel whose visible part is exactly one crop;
#       the crop is vertically padded with the mask color, so the panel can be split
#       cleanly but the crop's display y-position carries no semantic meaning.
#   concrete_pixel_structure:
#     - Choose one split row sr in [8,12] and one split col sc in [8,12].
#     - The true motif is partitioned into four rectangles:
#         (0:sr, 0:sc), (0:sr, sc:20), (sr:20, 0:sc), (sr:20, sc:20).
#     - These four windows jointly cover the entire 20x20 motif with no gaps.
#     - In the input, the four windows are shuffled horizontally. Each is embedded in a
#       20xW panel and separated by a 1-column full-height mask-colored divider.
#   behavior_trace:
#     - Original task: one crop -> enumerate (fg polarity, variant, row offset, col offset)
#       -> unique match -> repaint full motif.
#     - Variant: four crops -> split frames -> recover each visible rectangle ->
#       enumerate every placement for every frame in every polarity and variant ->
#       keep only assignments whose rectangle union covers all 400 cells ->
#       unique joint match -> repaint full motif.
#     - The key difference is that the solver must now satisfy a global coverage
#       constraint across multiple windows rather than accept a single local match.
#   complexity_calibration:
#     - Keeping four windows at 8..12 by 8..12 preserves the original visual scale.
#     - Using a clean 2x2 partition avoids accidental impossible layouts and keeps the
#       input visually interpretable: humans can see "four separated pieces of one image".
#     - Shuffling frame order and randomizing vertical panel padding prevents trivial
#       left-to-right concatenation and forces actual matching.
#   implicit_assumptions_review:
#     - Original `H <= 20 and W <= 20` assumed one crop only. No longer valid for the
#       whole input grid, so the new solver must split frames first.
#     - Original `len(colors) == 2` assumed no separator color. Now the whole input uses
#       exactly three colors: foreground, background, and mask/divider.
#     - Original loops over one `(r,c)` offset. Now each frame has its own unknown
#       `(r,c)` placement, so the search space becomes a product over frames.
#     - Original `assert len(matches) == 1` stayed conceptually correct, but a "match"
#       now means a full assignment of all frame placements whose union covers the motif.
#   placement_legality:
#     - The generator always uses the 2x2 partition, so windows never go out of bounds.
#     - Each panel width equals its crop width, so no panel contains a full-height
#       crop-colored column; only separator columns are solid mask columns.
#     - This guarantees `_split_frames` can recover frames unambiguously.
#   visual_observability:
#     - The multi-window change is obvious in every sample because the input always has
#       four separated panels instead of one crop.
#     - The output remains a single full 20x20 motif, making the many-to-one relation
#       directly visible from 2-3 examples.
#   cleanliness:
#     - The mask color appears only as panel padding and separator columns, so it is
#       visually distinct from motif content and never overlaps with the true crop data.
#   color_reasoning:
#     - Colors are only reused as motif fg/bg plus one neutral mask color.
#     - No ordering or transformation depends on color ids; the mask is identifiable by
#       its role as the only full-height solid separator color.


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


def _split_frames(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    h, w = map(int, grid.shape)
    assert h == 20

    solid_cols = [c for c in range(w) if len(set(map(int, grid[:, c]))) == 1]
    assert solid_cols, "expected at least one full-height separator column"
    sep = int(grid[0, solid_cols[0]])
    assert all(int(grid[0, c]) == sep for c in solid_cols)

    frames: list[np.ndarray] = []
    div_set = set(solid_cols)
    start = 0
    c = 0
    while c < w:
        if c in div_set:
            if start < c:
                frames.append(grid[:, start:c].copy())
            while c < w and c in div_set:
                c += 1
            start = c
        else:
            c += 1
    if start < w:
        frames.append(grid[:, start:w].copy())

    assert len(frames) >= 2
    return frames, sep


def _extract_visible_crop(frame: np.ndarray, sep: int) -> np.ndarray:
    rows = np.flatnonzero(np.any(frame != sep, axis=1))
    assert len(rows) > 0
    r0 = int(rows[0])
    r1 = int(rows[-1]) + 1
    assert np.all(frame[:r0, :] == sep)
    assert np.all(frame[r1:, :] == sep)
    crop = frame[r0:r1, :]
    assert np.all(crop != sep)
    return crop.copy()


def _candidate_positions(pattern: np.ndarray, crop_visible: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, crop_visible.shape)
    out: list[tuple[int, int]] = []
    for r in range(20 - h + 1):
        for c in range(20 - w + 1):
            if np.array_equal(pattern[r : r + h, c : c + w], crop_visible):
                out.append((r, c))
    return out


def _find_joint_matches(grid: np.ndarray, limit: int | None = None) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    frames, sep = _split_frames(g)
    crops = [_extract_visible_crop(frame, sep) for frame in frames]

    colors = colors_present(g, exclude=[sep])
    assert len(colors) == 2

    matches: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible_masks = [crop == int(fg) for crop in crops]
        sizes = [tuple(map(int, crop.shape)) for crop in crops]

        for variant_idx, pattern in enumerate(_VARIANTS):
            candidate_lists: list[list[tuple[int, int]]] = []
            possible_cover_masks: list[np.ndarray] = []
            failed = False
            for vis, (h, w) in zip(visible_masks, sizes):
                placements = _candidate_positions(pattern, vis)
                if not placements:
                    failed = True
                    break
                candidate_lists.append(placements)
                cover = np.zeros((20, 20), dtype=bool)
                for r, c in placements:
                    cover[r : r + h, c : c + w] = True
                possible_cover_masks.append(cover)
            if failed:
                continue

            order = sorted(range(len(crops)), key=lambda i: len(candidate_lists[i]))
            placed: list[tuple[int, int] | None] = [None] * len(crops)

            def dfs(k: int, covered: np.ndarray) -> bool:
                if limit is not None and len(matches) >= limit:
                    return True
                if k == len(order):
                    if covered.all():
                        matches.append(
                            (
                                int(fg),
                                int(bg),
                                int(variant_idx),
                                tuple(
                                    (int(item[0]), int(item[1]))
                                    for item in placed
                                    if item is not None
                                ),
                            )
                        )
                    return False

                i = order[k]
                h, w = sizes[i]
                for r, c in candidate_lists[i]:
                    placed[i] = (r, c)
                    new_covered = covered.copy()
                    new_covered[r : r + h, c : c + w] = True

                    optimistic = new_covered.copy()
                    for t in range(k + 1, len(order)):
                        optimistic |= possible_cover_masks[order[t]]
                    if optimistic.all():
                        stop = dfs(k + 1, new_covered)
                        if stop:
                            return True
                placed[i] = None
                return False

            dfs(0, np.zeros((20, 20), dtype=bool))
            if limit is not None and len(matches) >= limit:
                return matches

    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble several separated crop-windows into the shared 20x20 binary motif.

    Input structure:
    - The grid contains multiple 20-row panels concatenated horizontally and split by solid full-height separator columns in a third color.
    - Each panel hides all but one rectangular crop of the same underlying 20x20 motif; the crop order and its vertical panel padding are presentation-only.

    Rule:
    - Treat the separator color as non-data and recover the visible crop from each panel.
    - Try each remaining color as motif foreground, then test every crop against every position of every dihedral motif variant.
    - Keep the unique assignment where all crop placements come from one shared variant and their union covers the full 20x20 canvas.
    - Repaint that full motif with the matched foreground/background colors.

    Key insight:
    - The separated panels are not independent images; they are shuffled pieces of one oriented motif, so the correct solution is the only global stitching whose matched windows collectively cover all 400 cells.
    """
    g = np.asarray(grid, dtype=int)
    matches = _find_joint_matches(g, limit=2)
    assert len(matches) == 1
    fg, bg, variant_idx, _placements = matches[0]
    pattern = _VARIANTS[variant_idx]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _make_panel(crop: np.ndarray, sep: int, display_top: int) -> np.ndarray:
    h, w = map(int, crop.shape)
    panel = np.full((20, w), int(sep), dtype=int)
    panel[display_top : display_top + h, :] = crop
    return panel


def _compose_input(panels: list[np.ndarray], sep: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    bar = np.full((20, 1), int(sep), dtype=int)
    for i, panel in enumerate(panels):
        if i:
            pieces.append(bar)
        pieces.append(panel)
    return np.hstack(pieces)


def _leave_one_out_inputs(panels: list[np.ndarray], sep: int) -> list[np.ndarray]:
    out = []
    for i in range(len(panels)):
        subset = [panel for j, panel in enumerate(panels) if j != i]
        out.append(_compose_input(subset, sep))
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(200):
        fg, bg, sep = map(int, rng.choice(10, size=3, replace=False))
        pattern = _VARIANTS[int(rng.integers(len(_VARIANTS)))]
        output = np.where(pattern, fg, bg).astype(int)

        split_r = int(rng.integers(8, 13))
        split_c = int(rng.integers(8, 13))
        windows = [
            (0, 0, split_r, split_c),
            (0, split_c, split_r, 20 - split_c),
            (split_r, 0, 20 - split_r, split_c),
            (split_r, split_c, 20 - split_r, 20 - split_c),
        ]
        rng.shuffle(windows)

        panels: list[np.ndarray] = []
        for r, c, h, w in windows:
            crop = output[r : r + h, c : c + w]
            assert len(colors_present(crop)) == 2
            display_top = int(rng.integers(0, 20 - h + 1))
            panels.append(_make_panel(crop, sep, display_top))

        grid = _compose_input(panels, sep)
        matches = _find_joint_matches(grid, limit=2)
        if len(matches) != 1:
            continue
        if not np.array_equal(main(grid), output):
            continue

        # Variant check: each frame is structurally necessary under the joint-cover rule.
        leave_one_out = _leave_one_out_inputs(panels, sep)
        if any(len(_find_joint_matches(subgrid, limit=2)) == 1 for subgrid in leave_one_out):
            continue

        return grid

    raise RuntimeError("failed to generate a unique multi-window input")
