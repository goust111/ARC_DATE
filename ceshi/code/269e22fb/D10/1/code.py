# Variant of 269e22fb
# Dimension: D10 — 多步处理：匹配-重建流程 (multi_step)
# Change: 单步处理：匹配裁剪窗口到唯一的(变体, 偏移, 极性)，然后直接输出完整图案... → 迭代拼图补全

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


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

_CANVAS = 20
_SEP = 0
_GEN_RETRIES = 400


# Design analysis for the D10 variant
# -----------------------------------
# spec:
#   task: Assemble 2-4 rectangular crop blocks from the same dihedral motif instance.
#   input:
#     grid: larger black canvas containing multiple separated rectangles.
#     separator_color: fixed black (0); crop colors are two non-black ARC colors.
#     pieces: 2-4 solid rectangles; each rectangle is a crop from one shared 20x20 motif orientation.
#     placement_on_canvas: pieces are laid out as a tidy collage with 1+ cells of black gap.
#     source_windows:
#       - all from the same orientation and polarity
#       - windows jointly span the full 20x20 frame, but leave some cells uncovered
#       - at least one overlap between windows, so later placements can confirm earlier ones
#   output:
#     grid: 20x20
#     content: the assembled partial motif on black background, using the two piece colors
#   generator_strategy:
#     - build the full 20x20 motif first
#     - choose an overlapping 2/3/4-piece window template whose union spans all four borders but
#       still leaves some black cells in the final assembly
#     - crop those windows from the output and place the rectangles on a black staging canvas
#     - validate by calling main(input) and comparing with the known output
#
# behavior trace:
#   original:
#     one 10x10 crop -> try every orientation/offset/polarity -> unique match -> repaint full 20x20.
#   variant:
#     three separated rectangles -> each rectangle may have several candidate offsets on a shared
#     oriented motif -> search for one offset per rectangle whose placements form one connected
#     assembly spanning the whole 20x20 board -> overlay the rectangles onto a blank 20x20 canvas.
#   difference:
#     the solver can no longer terminate after a single local match; it must coordinate multiple
#     partial matches and use coverage/overlap consistency as a second stage.
#
# complexity calibration:
#   - A third separator color is necessary; otherwise disconnected crop pieces would be visually
#     indistinguishable from the motif background.
#   - Window templates keep the number of free choices small: 2, 3, or 4 pieces only, with one
#     shared orientation and with axis-aligned overlapping corner/edge rectangles.
#   - The assembled output intentionally leaves some black cells uncovered, so the task is visibly
#     different from the original "recover the entire hidden motif" behavior.
#
# hidden assumptions reviewed:
#   - The original `len(colors)==2` assumption no longer holds globally; now the grid has 3 colors
#     and the solver must exclude the separator before choosing foreground/background.
#   - The original single `(H,W)` crop scan no longer applies; each extracted rectangle has its own
#     size and candidate offset list.
#   - Directly returning `np.where(pattern, fg, bg)` would solve the old task but would skip the new
#     assembly stage and would incorrectly hallucinate unseen cells. The variant instead reconstructs
#     only the placed union on a blank 20x20 canvas after finding one common orientation and one
#     valid placement per block.
#   - The new search assumes every non-separator connected component is exactly one full rectangle.
#     The generator guarantees this by storing each crop as a solid non-separator rectangle and
#     separating rectangles with black gaps.
#
# legality / observability:
#   - Input rectangles never overlap on the staging canvas and never touch the canvas border.
#   - Source windows always stay within 20x20 bounds; their assembled bounding box is the full
#     20x20 frame even though their union area is strictly smaller than 400.
#   - Overlap area is always positive and the overlap graph is connected, so examples visibly behave
#     like jigsaw completion rather than a trivial disjoint tiling.


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _extract_piece_bboxes(g: np.ndarray, sep: int) -> list[tuple[int, int, int, int]]:
    mask = g != int(sep)
    H, W = map(int, g.shape)
    seen = np.zeros((H, W), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [rr for rr, _ in cells]
            cs = [cc for _, cc in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            rect = mask[r0 : r1 + 1, c0 : c1 + 1]
            assert rect.all()
            boxes.append((int(r0), int(c0), int(r1), int(c1)))
    boxes.sort()
    return boxes


def _candidate_positions(block_visible: np.ndarray, pattern: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, block_visible.shape)
    out: list[tuple[int, int]] = []
    for r in range(_CANVAS - h + 1):
        for c in range(_CANVAS - w + 1):
            if np.array_equal(pattern[r : r + h, c : c + w], block_visible):
                out.append((int(r), int(c)))
    return out


def _spans_canvas(placements: list[tuple[int, int]], blocks: list[np.ndarray]) -> bool:
    top = min(r for r, _ in placements)
    left = min(c for _, c in placements)
    bottom = max(r + block.shape[0] for block, (r, _) in zip(blocks, placements))
    right = max(c + block.shape[1] for block, (_, c) in zip(blocks, placements))
    return top == 0 and left == 0 and bottom == _CANVAS and right == _CANVAS


def _overlap_area(a: tuple[int, int], a_shape: tuple[int, int], b: tuple[int, int], b_shape: tuple[int, int]) -> int:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + int(a_shape[0]), ac0 + int(a_shape[1])
    br1, bc1 = br0 + int(b_shape[0]), bc0 + int(b_shape[1])
    h = min(ar1, br1) - max(ar0, br0)
    w = min(ac1, bc1) - max(ac0, bc0)
    return max(0, h) * max(0, w)


def _connected_by_overlap(placements: list[tuple[int, int]], blocks: list[np.ndarray]) -> bool:
    n = len(blocks)
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(n):
            if j in seen:
                continue
            if _overlap_area(placements[i], blocks[i].shape, placements[j], blocks[j].shape) > 0:
                seen.add(j)
                stack.append(j)
    return len(seen) == n


def _assemble_output(blocks: list[np.ndarray], placements: list[tuple[int, int]]) -> np.ndarray | None:
    out = np.full((_CANVAS, _CANVAS), _SEP, dtype=int)
    for block, (r, c) in zip(blocks, placements):
        h, w = map(int, block.shape)
        region = out[r : r + h, c : c + w]
        overlap = region != _SEP
        if overlap.any() and not np.array_equal(region[overlap], block[overlap]):
            return None
        region[:, :] = block
    return out


def _find_outputs(blocks: list[np.ndarray], visible_masks: list[np.ndarray], pattern: np.ndarray, limit: int = 2) -> list[np.ndarray]:
    options = [_candidate_positions(mask, pattern) for mask in visible_masks]
    if any(len(cands) == 0 for cands in options):
        return []

    order = sorted(range(len(blocks)), key=lambda i: len(options[i]))
    placements = [(-1, -1)] * len(blocks)
    outputs: list[np.ndarray] = []
    seen_outputs: set[bytes] = set()

    def dfs(idx: int) -> None:
        if len(outputs) >= limit:
            return
        if idx == len(order):
            if not _spans_canvas(placements, blocks):
                return
            if not _connected_by_overlap(placements, blocks):
                return
            out = _assemble_output(blocks, placements)
            if out is None:
                return
            key = out.tobytes()
            if key not in seen_outputs:
                seen_outputs.add(key)
                outputs.append(out)
            return

        piece_idx = order[idx]
        for r, c in options[piece_idx]:
            placements[piece_idx] = (int(r), int(c))
            dfs(idx + 1)
            if len(outputs) >= limit:
                return

    dfs(0)
    return outputs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble several detached crop blocks into one 20x20 partial reconstruction of the shared motif.

    Input structure:
    - The grid has a black staging background plus 2-4 separated rectangles.
    - Each rectangle is a two-color crop from the same rotation/reflection of one fixed 20x20 motif.
    - All rectangles use the same foreground/background pair, but come from different motif offsets.

    Rule:
    - Extract each non-black rectangle as one puzzle block.
    - Try each non-black color as the motif foreground and test every dihedral motif orientation.
    - For one shared orientation, place every block at a matching 20x20 offset so that the placed
      rectangles overlap into one connected assembly and collectively span the full 20x20 frame.
    - Overlay the blocks on a black 20x20 canvas; uncovered cells stay black.

    Key insight:
    - No single rectangle needs to solve the task alone; the correct orientation is the one for which
      all blocks can be embedded into one common 20x20 motif instance and jointly form one valid
      assembled picture.
    """
    g = np.asarray(grid, dtype=int)
    sep = int(g[0, 0])
    assert sep == _SEP

    boxes = _extract_piece_bboxes(g, sep)
    assert 2 <= len(boxes) <= 4
    blocks = [g[r0 : r1 + 1, c0 : c1 + 1].copy() for r0, c0, r1, c1 in boxes]

    colors = colors_present(g, exclude={sep})
    assert len(colors) == 2

    base = _pattern_mask()
    matches: dict[bytes, np.ndarray] = {}
    for fg in colors:
        visible_masks = [block == int(fg) for block in blocks]
        for pattern in _dihedral_variants(base):
            for assembled in _find_outputs(blocks, visible_masks, pattern):
                matches[assembled.tobytes()] = assembled

    assert len(matches) == 1
    return next(iter(matches.values()))


def _window_templates(piece_count: int) -> list[tuple[int, int, int, int]]:
    def small_anchor() -> tuple[int, int]:
        return random.randint(8, 10), random.randint(8, 10)

    def large_anchor() -> tuple[int, int]:
        return random.randint(11, 13), random.randint(11, 13)

    def overlap() -> int:
        return random.randint(2, 4)

    def center_band() -> tuple[int, int, int, int]:
        r0 = random.randint(8, 10)
        c0 = random.randint(8, 10)
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        return r0, c0, r0 + h, c0 + w

    if piece_count == 2:
        h1, w1 = large_anchor()
        h2, w2 = large_anchor()
        windows = [
            (0, 0, h1, w1),
            (_CANVAS - h2, _CANVAS - w2, h2, w2),
        ]
        while _overlap_area((0, 0), (h1, w1), (_CANVAS - h2, _CANVAS - w2), (h2, w2)) == 0:
            h1, w1 = large_anchor()
            h2, w2 = large_anchor()
            windows = [
                (0, 0, h1, w1),
                (_CANVAS - h2, _CANVAS - w2, h2, w2),
            ]
        return _transform_windows(windows)

    if piece_count == 3:
        h1, w1 = small_anchor()
        h3, w3 = small_anchor()
        ov1r, ov1c, ov2r, ov2c = overlap(), overlap(), overlap(), overlap()
        r0 = h1 - ov1r
        c0 = w1 - ov1c
        r1 = _CANVAS - h3 + ov2r
        c1 = _CANVAS - w3 + ov2c
        windows = [
            (0, 0, h1, w1),
            (r0, c0, r1 - r0, c1 - c0),
            (_CANVAS - h3, _CANVAS - w3, h3, w3),
        ]
        return _transform_windows(windows)

    h1, w1 = small_anchor()
    h4, w4 = small_anchor()
    mid_r0, mid_c0, mid_r1, mid_c1 = center_band()
    ov_a_r, ov_a_c, ov_d_r, ov_d_c = overlap(), overlap(), overlap(), overlap()
    b_r0 = h1 - ov_a_r
    b_c0 = w1 - ov_a_c
    c_r1 = _CANVAS - h4 + ov_d_r
    c_c1 = _CANVAS - w4 + ov_d_c
    windows = [
        (0, 0, h1, w1),
        (b_r0, b_c0, mid_r1 - b_r0, mid_c1 - b_c0),
        (mid_r0, mid_c0, c_r1 - mid_r0, c_c1 - mid_c0),
        (_CANVAS - h4, _CANVAS - w4, h4, w4),
    ]
    return _transform_windows(windows)


def _transform_windows(windows: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    k = random.randint(0, 3)
    do_flip = random.random() < 0.5
    transformed = []
    for r, c, h, w in windows:
        mask = np.zeros((_CANVAS, _CANVAS), dtype=bool)
        mask[r : r + h, c : c + w] = True
        moved = np.rot90(mask, k)
        if do_flip:
            moved = np.fliplr(moved)
        coords = np.argwhere(moved)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rect = moved[r0 : r1 + 1, c0 : c1 + 1]
        assert rect.all()
        transformed.append((int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1)))
    random.shuffle(transformed)
    return transformed


def _coverage_metrics(windows: list[tuple[int, int, int, int]]) -> tuple[int, int, int]:
    cover = np.zeros((_CANVAS, _CANVAS), dtype=bool)
    area_sum = 0
    for r, c, h, w in windows:
        cover[r : r + h, c : c + w] = True
        area_sum += int(h * w)
    overlap = area_sum - int(cover.sum())
    return int(cover.sum()), int(overlap), _CANVAS * _CANVAS - int(cover.sum())


def _layout_pieces(blocks: list[np.ndarray]) -> np.ndarray:
    n = len(blocks)
    if n <= 2:
        rows = [blocks]
    else:
        top = blocks[:2]
        bottom = blocks[2:]
        rows = [top, bottom]

    gap = 2
    margin = 1
    row_widths = [sum(block.shape[1] for block in row) + gap * max(0, len(row) - 1) for row in rows]
    row_heights = [max(block.shape[0] for block in row) for row in rows]
    H = margin * 2 + sum(row_heights) + gap * max(0, len(rows) - 1)
    W = margin * 2 + max(row_widths)
    canvas = np.full((H, W), _SEP, dtype=int)

    r = margin
    for row, row_h, row_w in zip(rows, row_heights, row_widths):
        c = margin + (W - 2 * margin - row_w) // 2
        for block in row:
            h, w = map(int, block.shape)
            canvas[r : r + h, c : c + w] = block
            c += w + gap
        r += row_h + gap
    return canvas


def _generate_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    base = _pattern_mask()
    pattern = random.choice(_dihedral_variants(base))
    fg, bg = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 2)
    full = np.where(pattern, int(fg), int(bg)).astype(int)

    piece_count = random.choices([2, 3, 4], weights=[1, 2, 2])[0]
    for _ in range(50):
        windows = _window_templates(int(piece_count))
        covered, overlap, uncovered = _coverage_metrics(windows)
        if 0 < covered < _CANVAS * _CANVAS and overlap > 0:
            break
    else:
        raise AssertionError("window template failed to leave uncovered cells")
    assert 0 < covered < _CANVAS * _CANVAS
    assert overlap > 0

    output = np.full((_CANVAS, _CANVAS), _SEP, dtype=int)
    blocks = [full[r : r + h, c : c + w].copy() for r, c, h, w in windows]
    for block, (r, c, h, w) in zip(blocks, windows):
        region = output[r : r + h, c : c + w]
        region[:, :] = block
    random.shuffle(blocks)
    grid = _layout_pieces(blocks)
    meta = {"piece_count": int(piece_count), "overlap_area": int(overlap), "uncovered_area": int(uncovered)}
    return grid, output, meta


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid, expected, _meta = _generate_attempt()
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid D10 jigsaw sample")
