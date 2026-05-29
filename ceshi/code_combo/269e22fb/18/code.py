# Combo variant of 269e22fb (index 18)
# combo_spec: D4/1+D9/0
# Name: 前景紧裁剪加窗口标记

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

MARKER_COLOR = 2
_INPUT_COLOR_POOL = [0, 1, 3, 4, 5, 6, 7, 8, 9]
_CROP_H_RANGE = range(8, 17)
_CROP_W_RANGE = range(8, 13)
_AMBIGUOUS = (-1, -1, -1)
_SCENE_SHAPE_ORDER = ((19, 18), (18, 19))


# Design analysis
# ---------------
# combo_spec:
#   - D4/1 changes the repaint rule: the input window's foreground hits must become a fixed
#     third color in the output.
#   - D9/0 changes the output shape: the completed scene is returned as its tight foreground
#     crop rather than as the full 20x20 canvas.
# concrete_pixels:
#   - A naive tight crop of the original motif would still be 20x20 because the motif touches
#     every row and column. To make D9 visible without changing the matching problem too much,
#     the hidden 20x20 scene is `motif_variant AND support_variant`, where canonical support is
#     rows 1..19 and cols 2..19, then both masks receive the same dihedral transform.
#   - Input stays simple: one HxW two-color crop of that clipped 20x20 scene.
#   - Output is produced in three steps:
#       1. reconstruct the whole clipped 20x20 scene,
#       2. recolor only the crop-covered scene foreground to `MARKER_COLOR`,
#       3. slice the result to the scene foreground bbox.
#   - Therefore the final output is not 20x20 and uses exactly three semantic colors:
#     background, ordinary foreground, and marked foreground.
# behavior_trace:
#   - Original task: crop -> identify motif orientation/offset -> repaint full 20x20 motif.
#   - D4/1 alone: same reconstruction, but the matched window's foreground cells become a third
#     color on the 20x20 output.
#   - D9/0 alone: reconstruct the clipped scene, then drop the blank outer bands by tight crop.
#   - Combo: reconstruct the clipped scene, place the marker on the matched window's scene
#     foreground, then tight-crop. The same offset now controls both matching and the visible
#     marker location inside the trimmed output.
# complexity_calibration:
#   - Hidden state is still modest: 8 oriented scenes plus one window offset and one fg/bg
#     polarity choice.
#   - Output shapes repeat across examples (19x18 or 18x19), so the shape change is legible.
#   - The marker color is fixed across examples, and the generator enforces both marked and
#     unmarked foreground so the recolor rule is visible rather than degenerating into a plain
#     two-color tight crop.
# implicit_assumptions_audit:
#   - `len(colors)==2` still applies to the input because the marker exists only in outputs.
#   - The old assumption "matching the raw motif is enough" is broken by D9: the true hidden
#     canvas is the support-clipped scene, not the bare 20x20 motif.
#   - The old assumption "orientation alone determines the answer" is broken by D4: the exact
#     matched window `(top, left, H, W)` must be preserved to place the marker.
#   - The old assumption "output shape is always 20x20" is broken by D9: the final answer must
#     be cropped to the reconstructed scene's foreground bbox.
#   - New solver assumptions introduced here:
#       * `MARKER_COLOR` is absent from the input.
#       * the matched window hits some scene foreground, otherwise the third color disappears.
#       * the matched window does not consume all scene foreground, otherwise the ordinary
#         foreground color disappears and the three-way semantics become harder to infer.
# placement_legality:
#   - The support mask is fixed inside the 20x20 frame, so every oriented scene is in bounds.
#   - Generator windows satisfy `0 <= top <= 20-H` and `0 <= left <= 20-W`.
#   - The marker does not add a second object; it only recolors existing scene foreground, so
#     there is no overlap/collision logic beyond choosing a legal window.
# visual_observability:
#   - Inputs remain clean rectangular two-color crops, close to the source task.
#   - Outputs visibly differ from the source in two independent ways: they are non-20x20 tight
#     rectangles, and they contain a localized third-color subset anchored to the recovered crop.
#   - The generator requires crops that cross at least one final tight-crop edge, making the D9
#     effect visible in examples rather than latent.
# visual_cleanliness:
#   - Only one motif-shaped object is shown, with one background and one highlight overlay.
#   - The third color appears only on existing foreground cells, so the result reads as a marked
#     subregion of the same object instead of an unrelated decoration.
# change_inferability:
#   - The color change is structural rather than numeric: background stays the input background,
#     unmarked scene foreground stays the input foreground, and the crop-covered scene foreground
#     becomes the fixed marker color.
#   - Because input fg/bg vary but the marker color stays fixed, humans can infer that the third
#     color encodes a role, not a palette permutation.
# generator_guarantees_for_main:
#   - The uniqueness index is built over the clipped scenes, exactly matching the solver's search
#     space.
#   - Candidate windows are filtered for uniqueness, polarity safety, tight-crop-edge crossing, and
#     three-color output visibility before `generate_input()` samples them.


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
        for height in _CROP_H_RANGE:
            for width in _CROP_W_RANGE:
                for top in range(20 - height + 1):
                    for left in range(20 - width + 1):
                        key = (height, width, canvas[top : top + height, left : left + width].tobytes())
                        match = (scene_idx, top, left)
                        prev = index.get(key)
                        if prev is None:
                            index[key] = match
                        elif prev != match:
                            index[key] = _AMBIGUOUS
    return index


_WINDOW_INDEX = _build_window_index()


def _lookup_unique_window(mask: np.ndarray) -> tuple[int, int, int] | None:
    height, width = map(int, mask.shape)
    if height in _CROP_H_RANGE and width in _CROP_W_RANGE:
        entry = _WINDOW_INDEX.get((height, width, mask.tobytes()))
        if entry is None or entry == _AMBIGUOUS:
            return None
        return entry

    matches: list[tuple[int, int, int]] = []
    for scene_idx, (canvas, _bbox, _variant_idx) in enumerate(_SCENES):
        for top in range(20 - height + 1):
            for left in range(20 - width + 1):
                if np.array_equal(canvas[top : top + height, left : left + width], mask):
                    matches.append((scene_idx, top, left))
                    if len(matches) > 1:
                        return None
    return matches[0] if matches else None


def _render_tight_marked(
    canvas: np.ndarray,
    fg: int,
    bg: int,
    top: int,
    left: int,
    height: int,
    width: int,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    full = np.where(canvas, int(fg), int(bg)).astype(int)
    window = canvas[top : top + height, left : left + width]
    out_slice = full[top : top + height, left : left + width]
    out_slice[window] = int(MARKER_COLOR)
    r0, c0, r1, c1 = bbox
    return full[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _solve_match(
    grid: np.ndarray,
) -> tuple[int, int, np.ndarray, tuple[int, int, int, int], tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height <= 20 and width <= 20

    colors = colors_present(g)
    assert len(colors) == 2
    assert int(MARKER_COLOR) not in colors

    matches: list[tuple[int, int, np.ndarray, tuple[int, int, int, int], tuple[int, int, int]]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        entry = _lookup_unique_window(visible)
        if entry is None:
            continue
        scene_idx, top, left = entry
        canvas, bbox, variant_idx = _SCENES[int(scene_idx)]
        matches.append((int(fg), int(bg), canvas, bbox, (variant_idx, top, left)))

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the clipped motif scene, mark the observed window hits, and return the tight crop.

    Input structure:
    - The grid is a two-color rectangular crop from one dihedral orientation of the shared motif
      after that motif has been clipped by an oriented support mask inside a 20x20 canvas.
    - One input color stands for scene foreground and the other stands for scene background.

    Rule:
    - Try each input color as the candidate foreground mask and match that binary crop against
      every legal window of every oriented clipped scene.
    - Once the crop's orientation, fg/bg polarity, and offset are uniquely identified, repaint
      the whole clipped 20x20 scene with the input colors.
    - Recolor only the matched window's scene-foreground cells to a fixed third marker color,
      then trim the result to the reconstructed scene foreground bbox.

    Key insight:
    - The crop still identifies a hidden global scene, but the output now exposes two recovered
      facts at once: the support-clipped extent of that scene and the exact subset of foreground
      cells that the input window covered.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    fg, bg, canvas, bbox, (_variant_idx, top, left) = _solve_match(g)
    assert canvas[top : top + height, left : left + width].any()
    return _render_tight_marked(canvas, fg, bg, top, left, height, width, bbox)


def _crosses_tight_crop_edge(
    top: int, left: int, height: int, width: int, bbox: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        (top < r0 <= top + height - 1)
        or (top <= r1 < top + height - 1)
        or (left < c0 <= left + width - 1)
        or (left <= c1 < left + width - 1)
    )


def _build_generator_candidates() -> list[list[tuple[int, int, int, int]]]:
    candidates: list[list[tuple[int, int, int, int]]] = [[] for _ in _SCENES]
    for scene_idx, (canvas, bbox, _variant_idx) in enumerate(_SCENES):
        scene_fg = int(canvas.sum())
        for height in _CROP_H_RANGE:
            for width in _CROP_W_RANGE:
                area = int(height * width)
                for top in range(20 - height + 1):
                    for left in range(20 - width + 1):
                        if not _crosses_tight_crop_edge(top, left, height, width, bbox):
                            continue
                        crop = canvas[top : top + height, left : left + width]
                        if not crop.any() or crop.all():
                            continue
                        marker_count = int(crop.sum())
                        if marker_count < max(12, area // 5):
                            continue
                        if scene_fg - marker_count < max(12, scene_fg // 6):
                            continue
                        direct = _WINDOW_INDEX.get((height, width, crop.tobytes()))
                        inverse = _WINDOW_INDEX.get((height, width, (~crop).tobytes()))
                        if direct != (scene_idx, top, left):
                            continue
                        if inverse is not None and inverse != _AMBIGUOUS:
                            continue
                        candidates[scene_idx].append((height, width, top, left))
    return candidates


_GENERATOR_CANDIDATES = _build_generator_candidates()
_VIABLE_SCENE_INDICES = [idx for idx, items in enumerate(_GENERATOR_CANDIDATES) if items]
_SCENE_INDICES_BY_SHAPE: dict[tuple[int, int], list[int]] = {shape: [] for shape in _SCENE_SHAPE_ORDER}
for scene_idx in _VIABLE_SCENE_INDICES:
    canvas, bbox, _variant_idx = _SCENES[int(scene_idx)]
    shape = (int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1))
    _SCENE_INDICES_BY_SHAPE[shape].append(int(scene_idx))
_SCENE_BAGS: dict[tuple[int, int], list[int]] = {shape: [] for shape in _SCENE_SHAPE_ORDER}
_NEXT_SHAPE_INDEX = 0


def _random_colors() -> tuple[int, int]:
    fg, bg = random.sample(_INPUT_COLOR_POOL, 2)
    return int(fg), int(bg)


def _next_scene_index() -> int:
    global _NEXT_SHAPE_INDEX

    for _ in range(len(_SCENE_SHAPE_ORDER)):
        shape = _SCENE_SHAPE_ORDER[_NEXT_SHAPE_INDEX % len(_SCENE_SHAPE_ORDER)]
        _NEXT_SHAPE_INDEX += 1
        bag = _SCENE_BAGS[shape]
        if not bag:
            bag.extend(random.sample(_SCENE_INDICES_BY_SHAPE[shape], len(_SCENE_INDICES_BY_SHAPE[shape])))
        if bag:
            return int(bag.pop())

    assert _VIABLE_SCENE_INDICES
    return int(random.choice(_VIABLE_SCENE_INDICES))


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    assert _VIABLE_SCENE_INDICES

    for _attempt in range(64):
        fg, bg = _random_colors()
        scene_idx = _next_scene_index()
        canvas, bbox, variant_idx = _SCENES[int(scene_idx)]
        height, width, top, left = random.choice(_GENERATOR_CANDIDATES[int(scene_idx)])

        inp = np.where(canvas[top : top + height, left : left + width], int(fg), int(bg)).astype(int)
        expected = _render_tight_marked(canvas, fg, bg, top, left, height, width, bbox)
        output_colors = colors_present(expected)
        if len(output_colors) != 3:
            continue
        actual = main(inp)
        if np.array_equal(actual, expected):
            marker_count = int((expected == int(MARKER_COLOR)).sum())
            ordinary_fg_count = int((expected == int(fg)).sum())
            stats = {
                "variant_idx": int(variant_idx),
                "input_height": int(height),
                "input_width": int(width),
                "output_height": int(expected.shape[0]),
                "output_width": int(expected.shape[1]),
                "marker_count": int(marker_count),
                "ordinary_fg_count": int(ordinary_fg_count),
                "bbox_area": int(expected.shape[0] * expected.shape[1]),
            }
            return inp, expected, stats

    raise RuntimeError("failed to generate a uniquely solvable non-degenerate combo crop")


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D4/1 + D9/0 combo.

    1. Pick colors and one pre-balanced oriented support-clipped scene.
    2. Choose a pre-verified unique crop that crosses a final tight-crop edge and leaves visible marker
       and ordinary foreground in the final tight crop.
    3. Build the input from that crop and confirm the solver reproduces the expected output.
    """
    return _generate_case()[0]
