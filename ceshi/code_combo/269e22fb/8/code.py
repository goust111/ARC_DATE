# Combo variant of 269e22fb (index 8)
# combo_spec: D3/2+D9/2
# Name: 四变体缩略图拼接输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# - a) Concrete behavior:
#   The input remains a two-color rectangular crop from one dihedral orientation
#   of a shared 20x20 motif, so the first half of the reasoning is unchanged:
#   infer foreground polarity, orientation, and crop offset by exhaustive window
#   matching. The fused output changes in two linked steps.
#   Step 1: reconstruct the matched full 20x20 pose and downsample it to a 10x10
#   thumbnail by aligned 2x2 majority vote.
#   Step 2: generate three more thumbnails by rotating that 10x10 image by
#   90/180/270 degrees, then tile the four 10x10 images into a 2x2 20x20 output.
#   Concrete micro-example: if the recovered 10x10 thumbnail has a foreground cell
#   at (1, 2), then the final output places that cell in the top-left quadrant,
#   and its rotated copies appear at the corresponding locations in the other
#   three quadrants. The original task stopped after "recover full motif"; the
#   fused task continues through "compress" and then "assemble four rotations".
# - b) Complexity calibration:
#   A direct majority vote on the original 20x20 motif would be ill-posed because
#   many aligned 2x2 blocks are tied 2:2. As in the D3/2 single variant, the
#   runtime motif is therefore derived from a 10x10 coarse mask expanded with
#   asymmetric 3:1 micro-tiles, making every 2x2 vote strict and observable.
#   The D9/2 tiling change is not allowed to collapse into four identical panels,
#   so the coarse seed is checked to be non-invariant under 90/180/270 rotation.
#   This keeps the extra rotational quadrants visually meaningful.
# - c) Implicit-assumption audit of `main()`:
#   `H <= 20`, `W <= 20`, and exactly two colors still hold because the input is
#   still one crop of one recolored motif window.
#   Trying both colors as candidate foreground still works because polarity is not
#   given explicitly.
#   Exhaustive matching over all dihedral variants and offsets still works because
#   the crop is still a literal subwindow of a 20x20 boolean mask.
#   The old output assumption "answer is the matched full 20x20 mask" no longer
#   holds. After a unique match is found, `main()` must instead:
#   recover a 10x10 thumbnail, rotate that thumbnail three times, and tile them.
#   Another new assumption is that 2x2 majority downsampling is deterministic;
#   `generate_input()` guarantees this by using the asymmetric micro-tile motif.
#   The rotation step assumes 10x10 square thumbnails so all four panels align
#   cleanly inside a 20x20 canvas; that is guaranteed by construction.
# - d) Placement legality:
#   The generator still places only one crop, so overlap is irrelevant. It must
#   choose `0 <= r <= 20-H` and `0 <= c <= 20-W`, which keeps the crop inside the
#   full motif. The output is assembled from four disjoint 10x10 quadrants inside
#   a fixed 20x20 grid, so no quadrant overlaps or goes out of bounds.
# - e) Visual observability:
#   Across examples, a human still sees the same kind of input crop, but the
#   outputs now show a clear 2x2 structure: the same small motif appears four
#   times with a consistent rotation progression. Because the quadrants share the
#   same colors and differ only by geometry, the new rule is inferable from a few
#   training pairs without relying on hidden numeric color identities.
# - f) Visual cleanliness:
#   The output uses only the two input colors and partitions the 20x20 canvas into
#   four clean 10x10 blocks. There are no separator lines, overlays, or auxiliary
#   glyphs competing with the motif itself.
# - g) Change inferability:
#   Color handling stays simple: the matched foreground/background colors are
#   reused everywhere. The only changes are structural and repeated in all four
#   quadrants, which makes the fusion readable. The ambiguous wording inherited
#   from standalone D9/2 mentions "40x40", but once D3/2 first shrinks the motif
#   to 10x10, a 2x2 tiling necessarily returns to 20x20; this implementation
#   follows the explicit fusion description rather than the standalone size.
# - New `main()` assumptions checked against `generate_input()`:
#   The crop may begin on any cell, so `main()` cannot infer the 10x10 thumbnail
#   directly from local 2x2 blocks; it must first recover the global 20x20 pose.
#   `generate_input()` always samples from the same derived motif family, verifies
#   the exact expected tiled output via `main()`, and only accepts examples whose
#   four quadrants truly realize the rotation-tiling rule.


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

    Tied 2x2 blocks are broken once here by the top-left cell so the runtime
    20x20 motif can be rebuilt from unambiguous 3:1 micro-tiles.
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


def _tile_rotating_thumbnails(thumbnail: np.ndarray) -> np.ndarray:
    assert thumbnail.shape == (10, 10)
    return np.block(
        [
            [thumbnail, np.rot90(thumbnail, 1)],
            [np.rot90(thumbnail, 2), np.rot90(thumbnail, 3)],
        ]
    )


def _paint(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(mask, int(fg), int(bg)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden motif pose from a crop, then emit a four-panel thumbnail rotation sheet.

    Input structure:
    - The grid is a two-color rectangular crop cut from one dihedral orientation of a shared 20x20 motif.
    - One input color marks the motif foreground micro-texture and the other marks its background micro-texture.

    Rule:
    - Try each input color as the candidate foreground mask.
    - Match that binary crop against every window in every dihedral orientation of the shared 20x20 motif.
    - After the unique 20x20 pose is identified, compress it into a 10x10 thumbnail by aligned 2x2 majority vote.
    - Rotate that thumbnail by 90, 180, and 270 degrees, then place the four thumbnails in a 2x2 20x20 layout using the same foreground/background colors.

    Key insight:
    - The crop still serves only to identify which hidden full motif pose is present; the final answer is not the pose itself but a tiled display of its coarse thumbnail under the four quarter-turn rotations.
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
    tiled = _tile_rotating_thumbnails(thumbnail)
    return _paint(tiled, fg, bg)


def generate_input() -> np.ndarray:
    """
    Output-first generator for the rotating-thumbnail combo variant.

    It samples a full 20x20 motif orientation and color pair, then searches for a
    crop whose unique match is verified by `main()`. The corresponding output is
    the 20x20 2x2 tiling of the matched 10x10 thumbnail and its 90/180/270-degree
    rotations.
    """
    rng = random.Random()
    base = _pattern_mask()
    variants = _dihedral_variants(base)
    palette = list(range(10))

    seed = _thumbnail_seed()
    assert all(not np.array_equal(seed, np.rot90(seed, k)) for k in (1, 2, 3))

    # Keep crop variety, but stay in the size regime that the source task spec
    # says is always uniquely identifying: min(H, W) >= 11 or max(H, W) >= 12.
    size_profiles = (
        ((11, 16), (8, 11)),
        ((8, 16), (12, 12)),
        ((11, 16), (12, 12)),
    )

    for _ in range(600):
        fg, bg = rng.sample(palette, 2)
        pattern = variants[rng.randrange(len(variants))]
        thumbnail = _downsample_majority(pattern)
        expected = _paint(_tile_rotating_thumbnails(thumbnail), fg, bg)

        (h_lo, h_hi), (w_lo, w_hi) = size_profiles[rng.randrange(len(size_profiles))]
        H = rng.randint(h_lo, h_hi)
        W = rng.randint(w_lo, w_hi)
        r = rng.randint(0, 20 - H)
        c = rng.randint(0, 20 - W)

        full = _paint(pattern, fg, bg)
        crop = full[r : r + H, c : c + W]

        try:
            if np.array_equal(main(crop), expected):
                return crop
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a uniquely solvable crop")
