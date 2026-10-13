from __future__ import annotations

from collections import Counter
import random

import numpy as np


SIDE_ORDER = ("top", "right", "bottom", "left")
ADJACENT_SIDE_PAIRS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
TRANSFORMS = (
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
)
BACKGROUND = 0
PANEL_FILL = 5
MARKER_SIZE = 2
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(arr[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _apply_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(np.asarray(arr, dtype=int), k)
    return np.fliplr(out) if flip else out


def _apply_inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(np.asarray(arr, dtype=int)) if flip else np.asarray(arr, dtype=int)
    return np.rot90(out, -k)


def _detect_control_sides(grid: np.ndarray, bg: int) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    candidates: dict[str, int] = {}
    slices = {
        "top": g[0, 1:-1],
        "bottom": g[-1, 1:-1],
        "left": g[1:-1, 0],
        "right": g[1:-1, -1],
    }
    for side, line in slices.items():
        vals = list(map(int, line))
        if not vals:
            continue
        unique = set(vals)
        if len(unique) == 1:
            color = vals[0]
            if color != bg:
                candidates[side] = int(color)
    assert len(candidates) == 2, "expected exactly two colored border controls"
    sides = list(candidates)
    assert any(set(sides) == set(pair) for pair in ADJACENT_SIDE_PAIRS), "controls must occupy adjacent sides"
    return candidates


def _extract_object(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.shape[0] >= 3 and g.shape[1] >= 3
    inner_mask = g[1:-1, 1:-1] != bg
    box = _bbox(inner_mask)
    assert box is not None, "expected one interior panel object"
    r0, c0, r1, c1 = box
    return _crop(g, (r0 + 1, c0 + 1, r1 + 1, c1 + 1))


def _dominant_color(arr: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(arr).ravel())).most_common(1)[0][0])


def _detect_guide_sides(obj: np.ndarray) -> dict[str, int]:
    sprite = np.asarray(obj, dtype=int)
    fill = _dominant_color(sprite)
    sides = {
        "top": sprite[0, :],
        "bottom": sprite[-1, :],
        "left": sprite[:, 0],
        "right": sprite[:, -1],
    }
    guides: dict[str, int] = {}
    for side, line in sides.items():
        non_fill = [int(x) for x in line if int(x) != fill]
        if not non_fill:
            continue
        color, count = Counter(non_fill).most_common(1)[0]
        if count >= len(line) - 1:
            guides[side] = int(color)
    assert len(guides) == 2, "expected exactly two guide sides"
    assert any(set(guides) == set(pair) for pair in ADJACENT_SIDE_PAIRS), "guides must occupy adjacent sides"
    assert len(set(guides.values())) == 2, "guide colors must be distinct"
    return guides


def _strip_guides(obj: np.ndarray, guide_sides: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in guide_sides else 0
    bottom = obj.shape[0] - (1 if "bottom" in guide_sides else 0)
    left = 1 if "left" in guide_sides else 0
    right = obj.shape[1] - (1 if "right" in guide_sides else 0)
    panel = np.asarray(obj[top:bottom, left:right], dtype=int)
    assert panel.size > 0
    return panel


def _half_index(lo: int, hi: int, span: int) -> int:
    center2 = lo + hi
    split2 = span - 1
    assert center2 != split2, "marker center must not lie on the split line"
    return 0 if center2 < split2 else 1


def _compress_panel(panel: np.ndarray, bg: int) -> np.ndarray:
    arr = np.asarray(panel, dtype=int)
    fill = _dominant_color(arr)
    out = np.full((2, 2), bg, dtype=int)
    for color in sorted(set(map(int, arr.ravel())) - {bg, fill}):
        mask = arr == color
        box = _bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        assert int(mask.sum()) == 4, "each marker must have area 4"
        assert r1 - r0 == 1 and c1 - c0 == 1, "each marker must be a solid 2x2 block"
        assert np.all(arr[r0 : r1 + 1, c0 : c1 + 1] == color)
        qr = _half_index(r0, r1, arr.shape[0])
        qc = _half_index(c0, c1, arr.shape[1])
        assert out[qr, qc] == bg, "at most one marker may land in each output quadrant"
        out[qr, qc] = color
    return out


def _find_aligned_object(obj: np.ndarray, controls: dict[str, int]) -> np.ndarray:
    candidates = []
    for k, flip in TRANSFORMS:
        transformed = _apply_transform(obj, k, flip)
        try:
            guides = _detect_guide_sides(transformed)
        except AssertionError:
            continue
        if guides == controls:
            candidates.append(transformed)
    assert len(candidates) == 1, "guide colors must determine a unique dihedral alignment"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate or mirror the guide-edged panel into the border-indicated orientation, then compress its 2x2 markers into a 2x2 quadrant map.

    Input structure:
    - A black background contains exactly one interior rectangular panel whose dominant color is grey.
    - Two adjacent outer border sides are solid control colors, and the panel has two adjacent 1-pixel guide edges in the same two colors.
    - Inside the panel, after removing the guide edges, there are 1-3 distinct solid 2x2 colored markers on the grey fill.

    Rule:
    - Find the unique dihedral transform of the whole panel that makes each guide color face the matching colored border side.
    - Apply that transform to the panel, crop away the two guide edges, and keep the grey interior.
    - Split the cropped interior into top/bottom and left/right halves; each 2x2 marker colors the corresponding cell of a 2x2 black output.

    Key insight:
    - The guide edges are only orientation clues: they must be matched to the frame first, and only then discarded before the quadrant compression.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    controls = _detect_control_sides(g, bg)
    obj = _extract_object(g, bg)
    aligned = _find_aligned_object(obj, controls)
    panel = _strip_guides(aligned, controls)
    return _compress_panel(panel, bg)


def _attach_guides(panel: np.ndarray, guide_sides: dict[str, int]) -> np.ndarray:
    h, w = panel.shape
    top = 1 if "top" in guide_sides else 0
    bottom = 1 if "bottom" in guide_sides else 0
    left = 1 if "left" in guide_sides else 0
    right = 1 if "right" in guide_sides else 0
    obj = np.full((h + top + bottom, w + left + right), PANEL_FILL, dtype=int)
    r0, c0 = top, left
    _paste(obj, panel, r0, c0)
    if "top" in guide_sides:
        obj[0, c0 : c0 + w] = guide_sides["top"]
    if "bottom" in guide_sides:
        obj[-1, c0 : c0 + w] = guide_sides["bottom"]
    if "left" in guide_sides:
        obj[r0 : r0 + h, 0] = guide_sides["left"]
    if "right" in guide_sides:
        obj[r0 : r0 + h, -1] = guide_sides["right"]

    if "top" in guide_sides and "left" in guide_sides:
        obj[0, 0] = BACKGROUND
    if "top" in guide_sides and "right" in guide_sides:
        obj[0, -1] = BACKGROUND
    if "bottom" in guide_sides and "left" in guide_sides:
        obj[-1, 0] = BACKGROUND
    if "bottom" in guide_sides and "right" in guide_sides:
        obj[-1, -1] = BACKGROUND
    return obj


def _quadrant_starts(span: int, half: int) -> list[int]:
    mid = span // 2
    if half == 0:
        starts = list(range(1, mid - 1))
    else:
        starts = list(range(mid, span - 2))
    assert starts, "panel half must have room for a 2x2 marker"
    return starts


def _build_aligned_panel() -> tuple[np.ndarray, np.ndarray, set[int]]:
    h = random.choice((6, 8, 10))
    w = random.choice((6, 8, 10))
    panel = np.full((h, w), PANEL_FILL, dtype=int)
    out = np.full((2, 2), BACKGROUND, dtype=int)

    marker_count = random.randint(1, 3)
    cells = random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], marker_count)
    colors = random.sample(COLOR_POOL, marker_count)
    for color, (qr, qc) in zip(colors, cells):
        r = random.choice(_quadrant_starts(h, qr))
        c = random.choice(_quadrant_starts(w, qc))
        panel[r : r + MARKER_SIZE, c : c + MARKER_SIZE] = color
        out[qr, qc] = color
    return panel, out, set(colors)


def _paint_controls(grid: np.ndarray, controls: dict[str, int]) -> None:
    if "top" in controls:
        grid[0, :] = controls["top"]
    if "bottom" in controls:
        grid[-1, :] = controls["bottom"]
    if "left" in controls:
        grid[:, 0] = controls["left"]
    if "right" in controls:
        grid[:, -1] = controls["right"]

    sides = set(controls)
    if sides == {"top", "left"}:
        grid[0, 0] = random.choice([controls["top"], controls["left"]])
    elif sides == {"top", "right"}:
        grid[0, -1] = random.choice([controls["top"], controls["right"]])
    elif sides == {"bottom", "left"}:
        grid[-1, 0] = random.choice([controls["bottom"], controls["left"]])
    elif sides == {"bottom", "right"}:
        grid[-1, -1] = random.choice([controls["bottom"], controls["right"]])


def _naive_unrotated_output(grid: np.ndarray) -> np.ndarray:
    bg = _get_bg(grid)
    obj = _extract_object(grid, bg)
    guides = _detect_guide_sides(obj)
    panel = _strip_guides(obj, guides)
    return _compress_panel(panel, bg)


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        bg = _get_bg(grid)
        if bg != BACKGROUND:
            return False
        controls = _detect_control_sides(grid, bg)
        obj = _extract_object(grid, bg)
        input_guides = _detect_guide_sides(obj)
        if input_guides == controls:
            return False
        aligned = _find_aligned_object(obj, controls)
        aligned_panel = _strip_guides(aligned, controls)
        if not np.array_equal(_compress_panel(aligned_panel, bg), expected):
            return False
        if np.array_equal(_naive_unrotated_output(grid), expected):
            return False
        if np.array_equal(main(grid), _naive_unrotated_output(grid)):
            return False
        if not np.array_equal(main(grid), expected):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    panel, expected, marker_colors = _build_aligned_panel()

    available_controls = [c for c in COLOR_POOL if c not in marker_colors]
    if len(available_controls) < 2:
        return None
    control_colors = random.sample(available_controls, 2)
    target_sides = random.choice(ADJACENT_SIDE_PAIRS)
    controls = {target_sides[0]: control_colors[0], target_sides[1]: control_colors[1]}

    aligned_obj = _attach_guides(panel, controls)
    k, flip = random.choice(TRANSFORMS[1:])
    input_obj = _apply_inverse_transform(aligned_obj, k, flip)

    oh, ow = input_obj.shape
    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    grid = np.full((oh + top_margin + bottom_margin, ow + left_margin + right_margin), BACKGROUND, dtype=int)
    _paste(grid, input_obj, top_margin, left_margin)
    _paint_controls(grid, controls)

    if not _validate_generated(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
