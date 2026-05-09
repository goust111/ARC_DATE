# Variant of 269e22fb
# Dimension: D1 — 匹配条件：裁剪窗口唯一性匹配方式 (match_cond)
# Change: 通过穷举8个二面体变体x2种前景/背景极性x所有偏移位置，找到唯一匹配裁剪窗口的(变体, 偏移, 极性)组合... → 多窗口联合匹配

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# spec:
#   core_change:
#     original_input: one fully observed two-color crop
#     variant_input: one masked outer crop that reveals exactly two smaller windows
#     uniqueness_shift: each window is allowed to be locally ambiguous, but the two
#       windows share the same outer-crop coordinates, so their joint offset/orientation
#       must be unique.
#   concrete_pixels:
#     outer_canvas:
#       size: 11-15 rows x 11-15 cols
#       colors: motif_fg, motif_bg, mask_color
#       mask_role: every cell outside the two windows is filled with a third color
#     visible_windows:
#       count: 2
#       size_each: 4-6 rows x 4-6 cols
#       geometry: solid rectangles, non-overlapping, separated by at least one fully
#         masked row or column for clean component parsing
#       source: both are copied from the same dihedral variant of the fixed 20x20 motif
#         and keep their true relative positions inside the outer crop
#   behavior_trace:
#     original:
#       - observe one HxW crop
#       - test all variant/offset/polarity candidates
#       - accept the single candidate whose whole crop matches
#     variant:
#       - observe one HxW masked crop containing two visible rectangles
#       - parse the two rectangles as windows inside a shared coordinate frame
#       - test all variant/offset/polarity candidates, but only require equality on
#         those two rectangles
#       - a single rectangle may recur elsewhere; the pair plus their spacing removes
#         the ambiguity
#   complexity_calibration:
#     - generator rejects samples unless both true windows have >=2 standalone matches
#       across all motif variants/offsets
#     - generator also rejects unless the pair has exactly 1 joint match
#     - this keeps the D1 change observable instead of quietly falling back to the
#       original single-window uniqueness regime
#   old_main_assumptions_that_break:
#     - `len(colors) == 2` no longer holds; input now uses exactly 3 colors
#     - `visible = g == fg` used to mean every cell is observed; now cells equal to
#       `mask_color` are unknown and must be ignored
#     - matching one rectangular crop is no longer sufficient; solver must recover two
#       rectangular components and keep their input-space offsets
#   new_solver_assumptions_and_generator_guarantees:
#     - non-mask cells form exactly 2 connected components under 4-neighborhood
#     - each connected component is a fully filled rectangle
#     - the two rectangles come from one shared outer crop, so one global offset applies
#       to both windows simultaneously
#     - output colors remain the two motif colors seen inside the windows; the mask color
#       never appears in the output
#   visual_readability:
#     - the mask color creates obvious hidden space, so the two-window structure is
#       immediately visible in a few examples
#     - only 3 colors are present, and color roles are structural rather than tied to
#       numeric ids


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
    for h in range(4, 7):
        for w in range(4, 7):
            bucket: dict[bytes, int] = {}
            for pattern in _VARIANTS:
                for r in range(20 - h + 1):
                    for c in range(20 - w + 1):
                        key = pattern[r : r + h, c : c + w].tobytes()
                        bucket[key] = bucket.get(key, 0) + 1
            counts[h, w] = bucket
    return counts


_SINGLE_WINDOW_COUNTS = _build_single_window_counts()


def _component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    H, W = map(int, mask.shape)
    seen = np.zeros((H, W), dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            area = 0
            while stack:
                cr, cc = stack.pop()
                area += 1
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            boxes.append((r0, c0, r1, c1, area))
    boxes.sort()

    rects: list[tuple[int, int, int, int]] = []
    for r0, c0, r1, c1, area in boxes:
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if area != h * w:
            return []
        rects.append((r0, c0, h, w))
    return rects


def _extract_windows(grid: np.ndarray, mask_color: int) -> list[tuple[int, int, int, int]] | None:
    rects = _component_boxes(grid != int(mask_color))
    if len(rects) != 2:
        return None
    return rects


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


def _count_joint_matches(
    outer_shape: tuple[int, int],
    windows: list[tuple[int, int, int, int]],
    visible_patches: list[np.ndarray],
) -> int:
    H, W = map(int, outer_shape)
    total = 0
    for pattern in _VARIANTS:
        for top in range(20 - H + 1):
            for left in range(20 - W + 1):
                ok = True
                for (r0, c0, h, w), patch in zip(windows, visible_patches):
                    if not np.array_equal(pattern[top + r0 : top + r0 + h, left + c0 : left + c0 + w], patch):
                        ok = False
                        break
                total += int(ok)
    return total


def _all_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 3

    matches: list[tuple[int, int, np.ndarray]] = []
    for mask_color in colors:
        windows = _extract_windows(g, int(mask_color))
        if windows is None:
            continue
        motif_colors = [color for color in colors if color != int(mask_color)]
        for fg in motif_colors:
            bg = int(motif_colors[0] if motif_colors[1] == fg else motif_colors[1])
            visible_patches = [
                g[r0 : r0 + h, c0 : c0 + w] == int(fg)
                for r0, c0, h, w in windows
            ]
            for pattern in _VARIANTS:
                for top in range(20 - H + 1):
                    for left in range(20 - W + 1):
                        ok = True
                        for (r0, c0, h, w), patch in zip(windows, visible_patches):
                            if not np.array_equal(
                                pattern[top + r0 : top + r0 + h, left + c0 : left + c0 + w],
                                patch,
                            ):
                                ok = False
                                break
                        if ok:
                            matches.append((int(fg), int(bg), pattern))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from two visible windows cut out of one masked crop.

    Input structure:
    - The grid is a three-color rectangular canvas: two motif colors plus one mask color.
    - Exactly two disjoint rectangular windows are visible; each window shows part of the same rotated/reflected 20x20 motif.
    - Cells in the mask color are hidden space inside the outer crop and do not constrain the match.

    Rule:
    - Treat each color in turn as the possible mask color, and treat the remaining two as foreground/background candidates.
    - Parse the non-mask cells into two rectangular windows and preserve their positions inside the shared outer canvas.
    - Search every dihedral motif variant and every outer-crop offset; keep the unique candidate whose two window patches both match.
    - Repaint the matched full 20x20 motif with the recovered foreground/background colors.

    Key insight:
    - A single small window may recur in multiple places, but the pair of windows carries a relative offset pattern that pinpoints one orientation and one global placement.
    """
    matches = _all_matches(np.asarray(grid, dtype=int))
    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _windows_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    r1, c1, h1, w1 = a
    r2, c2, h2, w2 = b
    return (
        r1 + h1 < r2
        or r2 + h2 < r1
        or c1 + w1 < c2
        or c2 + w2 < c1
    )


def _render_input(
    pattern: np.ndarray,
    fg: int,
    bg: int,
    mask_color: int,
    outer_top: int,
    outer_left: int,
    outer_shape: tuple[int, int],
    windows: list[tuple[int, int, int, int]],
) -> np.ndarray:
    H, W = map(int, outer_shape)
    grid = np.full((H, W), int(mask_color), dtype=int)
    for r0, c0, h, w in windows:
        patch = pattern[outer_top + r0 : outer_top + r0 + h, outer_left + c0 : outer_left + c0 + w]
        grid[r0 : r0 + h, c0 : c0 + w] = np.where(patch, int(fg), int(bg))
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = np.array(Color.ALL_COLORS, dtype=int)

    for _ in range(200):
        fg, bg, mask_color = map(int, rng.choice(palette, size=3, replace=False))
        pattern = _VARIANTS[int(rng.integers(len(_VARIANTS)))]
        H = int(rng.integers(11, 16))
        W = int(rng.integers(11, 16))
        outer_top = int(rng.integers(0, 20 - H + 1))
        outer_left = int(rng.integers(0, 20 - W + 1))
        expected = np.where(pattern, int(fg), int(bg)).astype(int)

        for _ in range(400):
            h1 = int(rng.integers(4, 7))
            w1 = int(rng.integers(4, 7))
            h2 = int(rng.integers(4, 7))
            w2 = int(rng.integers(4, 7))
            r1 = int(rng.integers(0, H - h1 + 1))
            c1 = int(rng.integers(0, W - w1 + 1))
            r2 = int(rng.integers(0, H - h2 + 1))
            c2 = int(rng.integers(0, W - w2 + 1))
            windows = sorted([(r1, c1, h1, w1), (r2, c2, h2, w2)])
            if not _windows_have_gap(windows[0], windows[1]):
                continue

            visible_patches = [
                pattern[outer_top + r0 : outer_top + r0 + h, outer_left + c0 : outer_left + c0 + w]
                for r0, c0, h, w in windows
            ]
            if any(_count_single_window_matches(patch) < 2 for patch in visible_patches):
                continue
            if _count_joint_matches((H, W), windows, visible_patches) != 1:
                continue

            grid = _render_input(pattern, fg, bg, mask_color, outer_top, outer_left, (H, W), windows)
            assert np.array_equal(main(grid), expected)
            return grid

    raise AssertionError("failed to generate a jointly unique two-window input")
