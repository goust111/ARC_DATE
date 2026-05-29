# Variant of b10624e5
# Dimension: D2 — 源对象：左上象限锚块旁的彩色图案 (src)
# Change: 左上象限中与红色锚块相邻的彩色像素构成源图案... → 锚块内部嵌入的微型图案作为源

from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg, paste
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[5]))
    from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D2 variant
# ==================================
# a) Concrete behavior:
# - The divider layout and four-quadrant scaffold stay unchanged.
# - Every quadrant still contains one red square anchor.
# - The change is local to the source object: in TL, the anchor is no longer accompanied by
#   an outside motif. Instead, the anchor itself is a red square with a 1-pixel red frame and
#   a 2x2 colored miniature inside it.
# - The output fills the interiors of the other anchors with reflected+scaled copies of that
#   2x2 miniature. The copied pattern is placed strictly inside each target anchor, preserving
#   a 1-pixel red border there as well.
# - Small trace:
#   original path: find TL connected component containing red -> scale whole component ->
#   align its red bbox to each target anchor.
#   variant path: find TL red anchor bbox -> extract only the interior miniature ->
#   reflect/resize it to each target interior -> write it inside the target anchor frame.
#   The difference matters when target size != TL size: scaling the full 4x4 anchor would move
#   colors toward borders, while the variant keeps a fixed red frame and scales only the inner art.
#
# b) Complexity calibration:
# - The miniature is 2x2 with 3-4 visible colors and forced LR/UD asymmetry, so reflections are
#   observable in a few examples.
# - Target sizes are limited to 4..6. This keeps interiors at 2..4 pixels wide: large enough to
#   see the miniature, but still few enough that the relationship is easy to infer.
# - At least one non-TL anchor is larger than the TL anchor, and the three target sizes are not
#   all identical, so anchor size actually carries information.
#
# c) Old implicit assumptions that must change:
# - Original solver assumed the source pattern was the entire non-bg component touching the TL
#   anchor. That is now false: the informative part is the anchor interior only.
# - Original solver aligned by the scaled red bbox inside the copied patch. Here alignment by the
#   full patch would be wrong, because the target red frame must remain a fixed 1-pixel border.
# - Original roundtrip scaling assertion depended on a specially generated external motif. After
#   moving the motif into a fixed interior cavity, retaining that assertion would reject many
#   visually clear samples; generator-side oracle validation is sufficient instead.
#
# d) Placement legality:
# - All anchors are axis-aligned squares fully contained in one quadrant.
# - TL anchor size is fixed at 4, so the source interior is always exactly 2x2.
# - Target anchors have size 4..6, guaranteeing a non-empty interior and preventing the copied
#   pattern from touching the divider or leaving the anchor.
#
# e) Visual observability:
# - The only colored pixels in TL outside the divider are inside the anchor, so the embedded-source
#   change is visible immediately.
# - Because the 2x2 source is asymmetric, left/right and up/down reflections show up across the
#   other three quadrants.
#
# f) Visual cleanliness:
# - There is no extra decoration near anchors, no overlapping objects, and no ambiguous connected
#   components beyond the intended TL framed anchor.
# - The red border keeps anchors legible even after the interiors are filled.
#
# g) Color inference:
# - No hidden ordering or color-id arithmetic is used.
# - Colors are copied verbatim; only geometry changes.
#
# New solver assumptions audited against generator:
# - Each quadrant has exactly one red connected component whose bbox is a square anchor.
# - The TL anchor has size 4 and contains at least one non-red pixel in its 2x2 interior.
# - Non-TL anchors are at least 4x4, so their writable interior exists and is fully inside the
#   quadrant. The generator enforces all three conditions.


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 4
TARGET_SIZE_RANGE = (4, 6)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Return the unique full-row/full-col quadrant divider: (mid_row, mid_col).
    """
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate an anchor-interior miniature across quadrants using anchor-size scaling and reflections.

    Input structure:
    - A single full row and full column of one non-background color split the 27x27 grid into four quadrants.
    - Each quadrant contains one red square anchor; the top-left anchor is larger and has colored pixels embedded inside it.
    - The other three quadrants show only solid red anchors in the input.

    Rule:
    - Extract the miniature pattern from inside the top-left anchor, excluding its red border.
    - For every other quadrant, reflect that miniature left/right for right-side quadrants and up/down for bottom quadrants.
    - Resize the reflected miniature to the writable interior of the target anchor, then place it inside the red frame.

    Key insight: The source blueprint is hidden inside the top-left anchor itself; anchor size determines how large the interior copy becomes, while quadrant position determines the reflection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        assert side_r == side_c, "red anchors must have square bounding boxes"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), int(side_r))

    assert set(anchors) == set(QUADS), "must detect one red anchor in every quadrant"

    sr0, sc0, sL = anchors[(0, 0)]
    assert sL >= 4, "top-left anchor must be large enough to contain an interior pattern"
    src_inner = g[sr0 + 1 : sr0 + sL - 1, sc0 + 1 : sc0 + sL - 1].copy()
    assert src_inner.size > 0, "source interior must be non-empty"
    assert np.any(src_inner != RED), "source anchor interior must contain non-red pixels"

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        assert tL >= 4, "target anchor must have a writable interior"

        motif = src_inner
        if qr == 1:
            motif = np.fliplr(motif)
        if qb == 1:
            motif = np.flipud(motif)

        inner_size = tL - 2
        motif_s = _resize_nn(motif, inner_size, inner_size)
        out[tr0 + 1 : tr0 + tL - 1, tc0 + 1 : tc0 + tL - 1] = motif_s

    return out


def _random_inner_pattern() -> np.ndarray:
    while True:
        n_colors = random.randint(3, 4)
        picked = random.sample(COLOR_POOL, n_colors)
        cells = picked + random.choices(picked, k=4 - n_colors)
        random.shuffle(cells)
        arr = np.array(cells, dtype=int).reshape(2, 2)
        if np.array_equal(arr, np.fliplr(arr)):
            continue
        if np.array_equal(arr, np.flipud(arr)):
            continue
        return arr


def _build_source_patch(inner: np.ndarray) -> np.ndarray:
    patch = np.full((SOURCE_ANCHOR_SIZE, SOURCE_ANCHOR_SIZE), RED, dtype=int)
    patch[1:-1, 1:-1] = inner
    return patch


def _scaled_inner(inner: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    motif = inner
    if qr == 1:
        motif = np.fliplr(motif)
    if qb == 1:
        motif = np.flipud(motif)
    inner_size = target_size - 2
    return _resize_nn(motif, inner_size, inner_size)


def _choose_anchor_top_left(qb: int, qr: int, size: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    return (
        random.randint(q_r0, q_r1 - size + 1),
        random.randint(q_c0, q_c1 - size + 1),
    )


def _paint_inner(out: np.ndarray, top: int, left: int, inner: np.ndarray) -> None:
    out[top + 1 : top + 1 + inner.shape[0], left + 1 : left + 1 + inner.shape[1]] = inner


def _sample_target_sizes() -> dict[tuple[int, int], int]:
    while True:
        sizes = {
            (0, 0): SOURCE_ANCHOR_SIZE,
            (0, 1): random.randint(*TARGET_SIZE_RANGE),
            (1, 0): random.randint(*TARGET_SIZE_RANGE),
            (1, 1): random.randint(*TARGET_SIZE_RANGE),
        }
        non_tl = [sizes[(0, 1)], sizes[(1, 0)], sizes[(1, 1)]]
        if max(non_tl) == SOURCE_ANCHOR_SIZE:
            continue
        if len(set(non_tl)) == 1:
            continue
        return sizes


def _gen_attempt() -> np.ndarray | None:
    inner = _random_inner_pattern()
    source_patch = _build_source_patch(inner)
    sizes = _sample_target_sizes()

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    anchors[(0, 0)] = _choose_anchor_top_left(0, 0, SOURCE_ANCHOR_SIZE)
    for key, size in sizes.items():
        if key == (0, 0):
            continue
        anchors[key] = _choose_anchor_top_left(key[0], key[1], size)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, source_patch, *anchors[(0, 0)])

    for key, size in sizes.items():
        if key == (0, 0):
            continue
        paste(out, np.full((size, size), RED, dtype=int), *anchors[key])
        _paint_inner(out, anchors[key][0], anchors[key][1], _scaled_inner(inner, key[0], key[1], size))

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, source_patch, *anchors[(0, 0)])
    for key, size in sizes.items():
        if key == (0, 0):
            continue
        paste(inp, np.full((size, size), RED, dtype=int), *anchors[key])

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D2 input")
