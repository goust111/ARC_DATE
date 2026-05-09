# Variant of 269e22fb
# Dimension: D3 — 目标输出：完整图案的展示方式 (dst)
# Change: 输出完整的20x20图案，前景/背景用输入的两种颜色绘制... → 输出缩小的完整图案缩略图

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# - Source-task note:
#   The prompt names `b10624e5` as the scaling reference, but the allowed workspace
#   only contains its ARC json, not a solver file. The scaling baseline is therefore
#   implemented directly here as a fixed 2x2-to-1 majority compression step.
# - a) Concrete behavior:
#   The input stays a two-color rectangular crop cut from one dihedral orientation
#   of a shared 20x20 motif. The output changes from the full 20x20 reconstruction
#   to a 10x10 thumbnail obtained after reconstruction: each aligned 2x2 block of
#   the matched 20x20 motif is replaced by its majority color.
#   Original flow: crop -> identify orientation/offset -> repaint full 20x20.
#   Variant flow: crop -> identify orientation/offset -> compress 20x20 into 10x10.
#   Example micro-step: a foreground coarse cell is expanded as [[1,1],[1,0]], so
#   its 2x2 block still reads as foreground after majority voting.
# - b) Complexity calibration:
#   The original 20x20 mask has 42 aligned 2x2 blocks with a 2:2 split, so direct
#   majority downsampling would be undefined. To keep D3 meaningful without adding
#   extra hidden rules, the task uses a derived 20x20 motif built from a 10x10 coarse
#   mask plus asymmetric 3:1 micro-tiles. This preserves nontrivial local matching
#   while making every 2x2 vote unambiguous.
# - c) Implicit-assumption audit of `main()`:
#   `H <= 20`, `W <= 20`, and `len(colors)==2` still hold.
#   Trying both input colors as foreground is still valid because the crop remains
#   an exact recoloring of one binary motif window.
#   Exhaustive matching over dihedral variants and offsets is still valid.
#   The old final line `np.where(pattern, fg, bg)` is no longer correct because the
#   destination size is now 10x10, not 20x20.
#   Majority compression now relies on a new assumption: each aligned 2x2 block has
#   a strict 3:1 majority. The generator guarantees this by construction.
# - d) Placement legality:
#   There is still only one object family and one crop. Crops are taken with
#   0 <= r <= 20-H and 0 <= c <= 20-W, so nothing overlaps or goes out of bounds.
# - e) Visual observability:
#   Across examples, the input remains a local 20x20 fragment while the output is
#   visibly smaller and keeps the same two colors. The repeated 3:1 micro-texture
#   makes the "compress after matching" rule readable rather than invisible.
# - f) Visual cleanliness:
#   Only two colors are present, with no overlays or auxiliary markers.
# - g) Color inferability:
#   No color id drives behavior. The output simply reuses the input's foreground and
#   background colors after geometric matching and scale reduction.
# - New `main()` assumptions checked against `generate_input()`:
#   The crop may start at any cell, not necessarily on a 2x2 boundary, so `main()`
#   must recover the full 20x20 pose before downsampling. `generate_input()` always
#   samples from the same derived motif family, so the matched full pattern is always
#   majority-safe and the 10x10 thumbnail is deterministic.


_ORIGINAL_PATTERN = (
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

_FG_TILE = np.array([[1, 1], [1, 0]], dtype=bool)
_BG_TILE = np.array([[0, 0], [0, 1]], dtype=bool)


def _original_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _ORIGINAL_PATTERN], dtype=bool)


def _thumbnail_seed() -> np.ndarray:
    """
    Derive a 10x10 coarse mask from the original motif.

    Blocks with a strict 3:1 or 4:0 foreground majority stay foreground.
    The original motif has many 2:2 blocks, so ties are deterministically broken
    by the block's top-left cell. This tie handling is only used once here to
    define the coarse motif; runtime downsampling never sees ties.
    """
    original = _original_mask()
    coarse = np.zeros((10, 10), dtype=bool)
    for r in range(10):
        for c in range(10):
            block = original[2 * r : 2 * r + 2, 2 * c : 2 * c + 2]
            ones = int(block.sum())
            coarse[r, c] = ones > 2 or (ones == 2 and bool(block[0, 0]))
    return coarse


def _expand_coarse_mask(coarse: np.ndarray) -> np.ndarray:
    full = np.zeros((20, 20), dtype=bool)
    for r in range(10):
        for c in range(10):
            tile = _FG_TILE if bool(coarse[r, c]) else _BG_TILE
            full[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = tile
    return full


def _pattern_mask() -> np.ndarray:
    return _expand_coarse_mask(_thumbnail_seed())


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _downsample_majority(mask: np.ndarray) -> np.ndarray:
    assert mask.shape == (20, 20)
    coarse = np.zeros((10, 10), dtype=bool)
    for r in range(10):
        for c in range(10):
            block = mask[2 * r : 2 * r + 2, 2 * c : 2 * c + 2]
            ones = int(block.sum())
            assert ones != 2
            coarse[r, c] = ones >= 3
    return coarse


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden motif pose from a crop, then emit its 10x10 majority thumbnail.

    Input structure:
    - The grid is a two-color rectangular crop cut from one dihedral orientation of a shared 20x20 motif.
    - One input color marks the motif foreground texture and the other marks its background texture.

    Rule:
    - Try each input color as the candidate foreground mask.
    - Match that binary crop against every window in every dihedral orientation of the shared 20x20 motif.
    - After the unique full 20x20 pose is identified, compress it into a 10x10 image by taking the majority color in each aligned 2x2 block.

    Key insight:
    - The crop still identifies the same underlying motif and pose as in the original task, but the answer is no longer the full reconstruction; it is the matched motif's coarse thumbnail.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    base = _pattern_mask()
    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(base):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern))

    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    thumbnail = _downsample_majority(pattern)
    return np.where(thumbnail, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    """
    Output-first generator for the thumbnail variant.

    It samples a full 20x20 motif orientation and color pair, then chooses a crop
    whose unique match is verified by `main()`. The corresponding output would be
    the 10x10 majority thumbnail of that same oriented motif.
    """
    rng = random.Random()
    base = _pattern_mask()
    variants = _dihedral_variants(base)
    palette = list(range(10))

    for _ in range(400):
        fg, bg = rng.sample(palette, 2)
        pattern = variants[rng.randrange(len(variants))]
        expected = np.where(_downsample_majority(pattern), fg, bg).astype(int)

        H = rng.randint(8, 16)
        W = rng.randint(8, 12)
        r = rng.randint(0, 20 - H)
        c = rng.randint(0, 20 - W)

        full = np.where(pattern, fg, bg).astype(int)
        crop = full[r : r + H, c : c + W]

        if np.array_equal(main(crop), expected):
            return crop

    raise RuntimeError("failed to generate a uniquely solvable crop")
