# Variant of 269e22fb
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 固定20x20输出，与基础图案尺寸相同... → 输出为紧裁剪的前景区域

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, colors_present


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

_CROP_H_RANGE = range(8, 17)
_CROP_W_RANGE = range(8, 13)
_AMBIGUOUS = (-1, -1, -1)

# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - D9 changes the output from a fixed 20x20 repaint to the tight crop of the recovered
#       foreground scene.
#   concrete_behavior:
#     - In the original task, every motif orientation has foreground in every row and column,
#       so a naive tight crop would still be 20x20 and the variant would be invisible.
#     - The smallest compatible fix is to keep the same motif but only expose an oriented
#       19x18 support rectangle inside the 20x20 scene: canonical support = rows 1..19 and
#       cols 2..19, then the same dihedral transform is applied to both motif and support.
#     - The visible scene is therefore `motif_variant AND support_variant`. Inputs remain
#       two-color rectangular crops of a 20x20 binary scene; outputs are the tight crop of
#       that reconstructed scene, which now becomes 19x18 or 18x19 depending on orientation.
#   behavior_trace:
#     - Original flow: crop -> identify motif orientation -> repaint the full 20x20 motif.
#     - Variant flow: crop -> identify the oriented clipped scene -> repaint the 20x20 scene
#       -> crop to the foreground bbox.
#     - Difference: the matched full scene now contains two empty border bands, so returning
#       the whole 20x20 canvas would erase the D9 change. The final bbox slice is essential.
#   complexity_calibration:
#     - The hidden-state space stays small: 8 dihedral scenes, same crop sizes as the source.
#     - The two output shapes (19x18, 18x19) repeat across examples, making the rule visible
#       without introducing too many unconstrained cases.
#   implicit_assumptions_review:
#     - `H <= 20` and `W <= 20` still hold because inputs are still crops of a 20x20 scene.
#     - `len(colors) == 2` still holds because the motif is still recolored as foreground vs
#       background with no extra markers.
#     - The old solver assumed the matched motif mask itself was the final answer. That is no
#       longer safe because the scene intentionally includes blank margins around the visible
#       motif support.
#     - Uniqueness now depends on the crop distinguishing orientation under the clipped support.
#       The generator enforces that by using a precomputed uniqueness index.
#   placement_legality:
#     - There is only one binary object on one background, so overlap and collision logic are
#       unchanged.
#     - The support mask is always inside 20x20, so every generated scene is in bounds.
#     - The generator chooses windows that contain both colors and extend past at least one
#       support edge, making the crop smaller/larger relation visible to a human solver.
#   observability_and_cleanliness:
#     - Across examples, the same canonical motif reappears, but the output omits the two blank
#       support margins and therefore becomes a non-square tight crop.
#     - No color-id arithmetic is used; colors are preserved directly as foreground/background.
#     - The scene stays visually clean: one motif, one background, one rectangular crop.
#   generator_guarantees_for_main:
#     - Every emitted crop is checked against the same uniqueness table used by the solver.
#     - The support-clipped scene always has foreground, so bbox cropping is valid.
#     - The solver's unique-match assert is protected by construction.


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _support_mask() -> np.ndarray:
    support = np.zeros((20, 20), dtype=bool)
    support[1:, 2:] = True
    return support


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _build_scenes() -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    motif_variants = _dihedral_variants(_pattern_mask())
    support_variants = _dihedral_variants(_support_mask())
    scenes = []
    for variant_idx, (motif, support) in enumerate(zip(motif_variants, support_variants)):
        canvas = motif & support
        pts = np.argwhere(canvas)
        assert pts.size > 0
        scenes.append((canvas, bbox_coords(pts), variant_idx))
    return scenes


_SCENES = _build_scenes()


def _build_window_index() -> dict[tuple[int, int, bytes], tuple[int, int, int]]:
    index: dict[tuple[int, int, bytes], tuple[int, int, int]] = {}
    for scene_idx, (canvas, _bbox, _variant_idx) in enumerate(_SCENES):
        for H in _CROP_H_RANGE:
            for W in _CROP_W_RANGE:
                for r in range(20 - H + 1):
                    for c in range(20 - W + 1):
                        key = (H, W, canvas[r : r + H, c : c + W].tobytes())
                        match = (scene_idx, r, c)
                        prev = index.get(key)
                        if prev is None:
                            index[key] = match
                        elif prev != match:
                            index[key] = _AMBIGUOUS
    return index


_WINDOW_INDEX = _build_window_index()


def _tight_crop_repaint(canvas: np.ndarray, fg: int, bg: int) -> np.ndarray:
    pts = np.argwhere(canvas)
    assert pts.size > 0
    r0, c0, r1, c1 = bbox_coords(pts)
    cropped = canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return np.where(cropped, int(fg), int(bg)).astype(int)


def _lookup_unique_window(mask: np.ndarray) -> tuple[int, int, int] | None:
    H, W = map(int, mask.shape)
    if H in _CROP_H_RANGE and W in _CROP_W_RANGE:
        entry = _WINDOW_INDEX.get((H, W, mask.tobytes()))
        if entry is None or entry == _AMBIGUOUS:
            return None
        return entry

    matches: list[tuple[int, int, int]] = []
    for scene_idx, (canvas, _bbox, _variant_idx) in enumerate(_SCENES):
        for r in range(20 - H + 1):
            for c in range(20 - W + 1):
                if np.array_equal(canvas[r : r + H, c : c + W], mask):
                    matches.append((scene_idx, r, c))
                    if len(matches) > 1:
                        return None
    return matches[0] if matches else None


def _solve_match(
    grid: np.ndarray,
) -> tuple[int, int, np.ndarray, tuple[int, int, int, int], tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, np.ndarray, tuple[int, int, int, int], tuple[int, int, int]]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        entry = _lookup_unique_window(visible)
        if entry is None:
            continue
        scene_idx, r, c = entry
        canvas, bbox, variant_idx = _SCENES[int(scene_idx)]
        matches.append((int(fg), int(bg), canvas, bbox, (variant_idx, r, c)))

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the clipped motif scene, then return only its tight foreground crop.

    Input structure:
    - The grid is a two-color rectangular crop from a 20x20 scene built from one dihedral
      variant of the shared motif.
    - The visible motif is restricted to an oriented 19x18 support window, so the 20x20 scene
      includes empty border bands on two sides.

    Rule:
    - Try each input color as the motif foreground.
    - Match that binary crop against every possible window of every oriented clipped scene.
    - Repaint the uniquely matched 20x20 scene and then trim it to the foreground bbox.

    Key insight:
    - The local crop still identifies the same motif family, but the completed scene is no
      longer the full 20x20 motif. The output is the support-tight rectangle around the visible
      foreground after orientation has been determined.
    """
    fg, bg, canvas, _bbox, _meta = _solve_match(grid)
    return _tight_crop_repaint(canvas, fg, bg)


def _crosses_support_edge(r: int, c: int, H: int, W: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        (r < r0 <= r + H - 1)
        or (r <= r1 < r + H - 1)
        or (c < c0 <= c + W - 1)
        or (c <= c1 < c + W - 1)
    )


def _build_generator_candidates() -> list[list[tuple[int, int, int, int]]]:
    candidates: list[list[tuple[int, int, int, int]]] = [[] for _ in _SCENES]
    for scene_idx, (canvas, bbox, _variant_idx) in enumerate(_SCENES):
        for H in _CROP_H_RANGE:
            for W in _CROP_W_RANGE:
                for r in range(20 - H + 1):
                    for c in range(20 - W + 1):
                        if not _crosses_support_edge(r, c, H, W, bbox):
                            continue
                        crop = canvas[r : r + H, c : c + W]
                        if not crop.any() or crop.all():
                            continue
                        direct = _WINDOW_INDEX.get((H, W, crop.tobytes()))
                        inverse = _WINDOW_INDEX.get((H, W, (~crop).tobytes()))
                        if direct != (scene_idx, r, c):
                            continue
                        if inverse is not None and inverse != _AMBIGUOUS:
                            continue
                        candidates[scene_idx].append((H, W, r, c))
    return candidates


_GENERATOR_CANDIDATES = _build_generator_candidates()
_VIABLE_SCENE_INDICES = [idx for idx, items in enumerate(_GENERATOR_CANDIDATES) if items]


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D9 variant.

    1. Pick colors and one oriented clipped scene.
    2. Compute the expected tight-cropped output from that scene.
    3. Choose a pre-verified unique crop that crosses a support edge.
    4. Re-check with `main()` before returning.
    """
    assert _VIABLE_SCENE_INDICES

    for _attempt in range(200):
        fg, bg = random.sample(range(10), 2)
        scene_idx = random.choice(_VIABLE_SCENE_INDICES)
        canvas, _bbox, _variant_idx = _SCENES[int(scene_idx)]
        H, W, r, c = random.choice(_GENERATOR_CANDIDATES[int(scene_idx)])

        full_scene = np.where(canvas, int(fg), int(bg)).astype(int)
        inp = full_scene[r : r + H, c : c + W]
        expected = _tight_crop_repaint(canvas, fg, bg)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a uniquely solvable D9 crop")
