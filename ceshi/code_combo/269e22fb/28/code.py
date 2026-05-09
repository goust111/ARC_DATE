# Combo variant of 269e22fb (index 28)
# combo_spec: D5/2+D9/0
# Name: 可变尺寸图案前景紧裁剪输出

from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "solvers" / "arc_utils.py"
    if _candidate.exists():
        sys.path.append(str(_candidate.parent))
        break

from arc_utils import Color, bbox_coords, colors_present


_BASE_PATTERN_20 = (
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

_CANDIDATE_SIZES = (15, 20, 25)
_MIN_CROP = 8
_MAX_CROP_H = 16
_MAX_CROP_W = 12
_AMBIGUOUS = (-1, -1, -1)


# Design analysis
# - Concretization and behavior trace:
#   The original task hid one 20x20 motif and asked the solver to repaint that entire square.
#   D5/2 adds a hidden size choice, so the motif family now has three square scales:
#   15x15, 20x20, and 25x25, all produced by nearest-neighbor resizing of the same canonical
#   20x20 bitmap. D9/0 changes the output target from "that whole square" to "the smallest
#   rectangle containing the visible foreground scene". A direct tight crop of the resized
#   motif would still be square and would often collapse back to the full canvas, so the
#   fusion also needs the D9-style clipped scene: after resizing and orienting the motif, keep
#   only the cells inside a support rectangle with a 1-cell top margin and a 2-cell left
#   margin, then apply the same dihedral transform to that support. Example trace:
#   original: 10x10 crop -> match one 20x20 orientation -> repaint 20x20.
#   fused: 10x10 crop -> test both color polarities -> test 15/20/25 scenes -> recover one
#   oriented clipped scene -> crop that scene to its foreground bbox -> repaint only that bbox.
#   The final output shape is therefore size-dependent and orientation-dependent, e.g.
#   14x13 / 13x14 for size 15, 19x18 / 18x19 for size 20, and 24x23 / 23x24 for size 25.
# - Complexity calibration:
#   The free variables are intentionally kept to a small reusable product:
#   three sizes x eight dihedral orientations x legal crop offsets x foreground polarity.
#   This makes the task visibly different from the source while avoiding an open-ended scale
#   family. Keeping the support margins fixed at (top=1, left=2) preserves a consistent scene
#   identity across sizes, so examples repeat a small set of output aspect ratios instead of
#   producing arbitrary rectangles.
# - Implicit assumption audit:
#   The original `assert H <= 20 and W <= 20` assumed a single canvas size; the fused solver
#   instead only assumes the crop fits at least one candidate size. The original `_pattern_mask`
#   returned one hard-coded 20x20 array; now it depends on `size`. The original matching loops
#   scanned one 20x20 motif and treated the matched motif as the final answer. That last step is
#   no longer valid because the reconstructed scene includes deliberate blank bands, and the
#   answer is the scene's tight foreground crop rather than the full square canvas.
# - Placement legality:
#   Generation stays output-first. A scene is built by intersecting one resized motif variant
#   with the equally oriented support mask, so everything remains in-bounds by construction.
#   The crop window is always chosen inside that square scene, and there is still only one
#   foreground object plus one background, so overlap conflicts cannot occur.
# - Visual observability:
#   Across 2-4 examples, a human can see that the local two-color crop always expands into the
#   same family of motifs, but the final answer is not the full square anymore. Some examples
#   finish as 14x13, some as 19x18, some as 24x23, and their transposed counterparts under
#   rotation. That repeated low-cardinality shape family makes both hidden size and tight-crop
#   behavior inferable without using color ids or metadata.
# - Visual cleanliness:
#   The scene is still a single binary motif on a uniform background with one rectangular crop.
#   No overlays, counters, or extra markers are added, so the only new visible differences are
#   motif scale and the clipped-support margins that explain why the output is non-square.
# - Color inference:
#   Colors remain purely role-based: the two observed colors are foreground/background in some
#   order. No arithmetic or ordering on color ids is used, and polarity ambiguity is resolved
#   only by structural matching.
# - New solver assumptions guaranteed by the generator:
#   `main()` assumes exactly two colors, at least one candidate size large enough for the crop,
#   and exactly one match across size/orientation/offset/polarity. `generate_input()` enforces
#   those assumptions by sampling only from a precomputed unique-window table, filtering out
#   one-color windows, and rechecking `main(inp)` against the expected cropped output.
# - Fusion anti-degeneration check:
#   To avoid collapsing back to the source task, generator candidates are required to cross at
#   least one edge of the scene bbox. That means the input reveals one of the blank support
#   margins, and the fused output differs in shape from the matched full square motif.


def _canonical_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN_20], dtype=bool)


def _resize_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    row_idx = (np.arange(size) * mask.shape[0]) // size
    col_idx = (np.arange(size) * mask.shape[1]) // size
    return mask[row_idx][:, col_idx]


@lru_cache(maxsize=None)
def _pattern_mask(size: int) -> np.ndarray:
    assert size in _CANDIDATE_SIZES
    return _resize_binary_mask(_canonical_mask(), int(size))


@lru_cache(maxsize=None)
def _support_mask(size: int) -> np.ndarray:
    support = np.zeros((size, size), dtype=bool)
    support[1:, 2:] = True
    return support


def _dihedral_variants(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return tuple(variants)


@lru_cache(maxsize=None)
def _scenes() -> tuple[tuple[int, np.ndarray, tuple[int, int, int, int], int], ...]:
    out: list[tuple[int, np.ndarray, tuple[int, int, int, int], int]] = []
    for size in _CANDIDATE_SIZES:
        motif_variants = _dihedral_variants(_pattern_mask(size))
        support_variants = _dihedral_variants(_support_mask(size))
        for variant_idx, (motif, support) in enumerate(zip(motif_variants, support_variants)):
            canvas = motif & support
            pts = np.argwhere(canvas)
            assert pts.size > 0
            out.append((int(size), canvas, bbox_coords(pts), int(variant_idx)))
    return tuple(out)


def _crop_h_range(size: int) -> range:
    return range(_MIN_CROP, min(int(size) - 1, _MAX_CROP_H) + 1)


def _crop_w_range(size: int) -> range:
    return range(_MIN_CROP, min(int(size) - 1, _MAX_CROP_W) + 1)


def _build_window_index() -> dict[tuple[int, int, bytes], tuple[int, int, int]]:
    index: dict[tuple[int, int, bytes], tuple[int, int, int]] = {}
    for scene_idx, (size, canvas, _bbox, _variant_idx) in enumerate(_scenes()):
        for H in _crop_h_range(size):
            for W in _crop_w_range(size):
                for r in range(size - H + 1):
                    for c in range(size - W + 1):
                        key = (H, W, canvas[r : r + H, c : c + W].tobytes())
                        match = (int(scene_idx), int(r), int(c))
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
    if H >= _MIN_CROP and W >= _MIN_CROP and H <= _MAX_CROP_H and W <= _MAX_CROP_W:
        entry = _WINDOW_INDEX.get((H, W, mask.tobytes()))
        if entry is None or entry == _AMBIGUOUS:
            return None
        return entry

    matches: list[tuple[int, int, int]] = []
    for scene_idx, (size, canvas, _bbox, _variant_idx) in enumerate(_scenes()):
        if H > size or W > size:
            continue
        for r in range(size - H + 1):
            for c in range(size - W + 1):
                if np.array_equal(canvas[r : r + H, c : c + W], mask):
                    matches.append((int(scene_idx), int(r), int(c)))
                    if len(matches) > 1:
                        return None
    return matches[0] if matches else None


def _solve_match(
    grid: np.ndarray,
) -> tuple[int, int, int, np.ndarray, tuple[int, int, int, int], tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)

    colors = colors_present(g)
    assert len(colors) == 2
    assert any(H <= size and W <= size for size in _CANDIDATE_SIZES)

    matches: list[tuple[int, int, int, np.ndarray, tuple[int, int, int, int], tuple[int, int, int]]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        entry = _lookup_unique_window(visible)
        if entry is None:
            continue
        scene_idx, r, c = entry
        size, canvas, bbox, variant_idx = _scenes()[int(scene_idx)]
        matches.append((int(fg), int(bg), int(size), canvas, bbox, (int(variant_idx), int(r), int(c))))

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden resized scene, then return only its tight foreground crop.

    Input structure:
    - The grid is a two-color rectangular crop cut from one dihedral variant of a shared motif family.
    - The hidden full scene is square but not fixed-size: it may be 15x15, 20x20, or 25x25, and the visible motif is additionally clipped by an oriented support window that leaves blank border bands.

    Rule:
    - Try each observed input color as the motif foreground.
    - Match that boolean crop against every legal window of every candidate scene across all allowed sizes, orientations, and offsets.
    - Repaint the uniquely matched scene with the recovered foreground/background colors, then trim the result to the foreground's minimum bounding rectangle.

    Key insight:
    - The crop still comes directly from the final colored scene, but solving now requires two hidden inferences at once: which motif size generated the crop, and which oriented clipped scene it belongs to. The answer is the scene's tight foreground rectangle, not the whole square canvas.
    """
    fg, bg, _size, canvas, _bbox, _meta = _solve_match(grid)
    return _tight_crop_repaint(canvas, fg, bg)


def _crosses_scene_bbox_edge(r: int, c: int, H: int, W: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        (r < r0 <= r + H - 1)
        or (r <= r1 < r + H - 1)
        or (c < c0 <= c + W - 1)
        or (c <= c1 < c + W - 1)
    )


def _build_generator_candidates() -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    candidates: list[list[tuple[int, int, int, int]]] = [[] for _ in _scenes()]
    for scene_idx, (size, canvas, bbox, _variant_idx) in enumerate(_scenes()):
        for H in _crop_h_range(size):
            for W in _crop_w_range(size):
                for r in range(size - H + 1):
                    for c in range(size - W + 1):
                        if not _crosses_scene_bbox_edge(r, c, H, W, bbox):
                            continue
                        crop = canvas[r : r + H, c : c + W]
                        if not crop.any() or crop.all():
                            continue
                        direct = _WINDOW_INDEX.get((H, W, crop.tobytes()))
                        inverse = _WINDOW_INDEX.get((H, W, (~crop).tobytes()))
                        if direct != (int(scene_idx), int(r), int(c)):
                            continue
                        if inverse is not None and inverse != _AMBIGUOUS:
                            continue
                        candidates[scene_idx].append((int(H), int(W), int(r), int(c)))
    return tuple(tuple(items) for items in candidates)


_GENERATOR_CANDIDATES = _build_generator_candidates()
_VIABLE_SCENE_INDICES = tuple(idx for idx, items in enumerate(_GENERATOR_CANDIDATES) if items)


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D5/2 + D9/0 fusion.

    1. Pick one precomputed variable-size clipped scene and two distinct colors.
    2. Repaint that full hidden square scene.
    3. Choose a crop that uniquely identifies the scene and crosses a cropped-scene bbox edge.
    4. Re-check the crop with `main()` before returning it.
    """
    assert _VIABLE_SCENE_INDICES

    for _attempt in range(400):
        fg, bg = random.sample(Color.ALL_COLORS, 2)
        scene_idx = random.choice(_VIABLE_SCENE_INDICES)
        size, canvas, _bbox, _variant_idx = _scenes()[int(scene_idx)]
        H, W, r, c = random.choice(_GENERATOR_CANDIDATES[int(scene_idx)])

        full_scene = np.where(canvas, int(fg), int(bg)).astype(int)
        inp = full_scene[r : r + H, c : c + W].copy()
        expected = _tight_crop_repaint(canvas, int(fg), int(bg))
        solved = main(inp)
        if np.array_equal(solved, expected):
            return inp

    raise RuntimeError("failed to generate a uniquely solvable D5/2 + D9/0 combo crop")
