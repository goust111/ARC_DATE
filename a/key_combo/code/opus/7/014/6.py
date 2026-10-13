import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
PANEL_COLOR = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# Every glyph spans a full 3x3 bbox and keeps (1,0)=0 so the mid-row label can
# sit directly to its left without touching the glyph component.
GLYPH_LIBRARY = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
)

GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = bbox(grid == PANEL_COLOR)
    assert box is not None, "expected grey panel"
    r0, c0, r1, c1 = box
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert panel.shape[0] >= 6 and panel.shape[1] >= 6, "panel must admit four quadrants"
    assert not np.any(panel == BACKGROUND), "panel bbox must be solid except for controls"
    return box


def _quadrant_boxes(height: int, width: int) -> dict[str, tuple[int, int, int, int]]:
    r_mid = (height - 1) // 2
    c_mid = (width - 1) // 2
    boxes = {
        "ul": (0, 0, r_mid, c_mid),
        "ur": (0, c_mid + 1, r_mid, width - 1),
        "ll": (r_mid + 1, 0, height - 1, c_mid),
        "lr": (r_mid + 1, c_mid + 1, height - 1, width - 1),
    }
    for box in boxes.values():
        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "quadrants must fit a 3x3 glyph"
    return boxes


def _quadrant_of(r: int, c: int, height: int, width: int) -> str:
    r_mid = (height - 1) // 2
    c_mid = (width - 1) // 2
    if r <= r_mid:
        return "ul" if c <= c_mid else "ur"
    return "ll" if c <= c_mid else "lr"


def _glyph_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _parse_dictionary(grid: np.ndarray, panel_box: tuple[int, int, int, int], bg: int) -> dict[int, np.ndarray]:
    left = np.asarray(grid[:, : panel_box[1]], dtype=int)
    colors = sorted(set(map(int, np.unique(left))) - {bg, PANEL_COLOR})
    assert 2 <= len(colors) <= 4, "expected 2-4 dictionary colors"

    glyphs: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(left == color)
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 4, "dictionary entry must be label + 3x3 glyph"
        region = left[r0 : r1 + 1, c0 : c1 + 1]
        first_col = region[:, 0] == color
        assert int(first_col.sum()) == 1, "entry must have exactly one key label"
        glyph = region[:, 1:] == color
        assert glyph.shape == (3, 3)
        assert np.all(glyph.sum(axis=0) >= 1) and np.all(glyph.sum(axis=1) >= 1), "glyph must span its 3x3 bbox"
        glyphs[color] = glyph

    return glyphs


def _extract_controls(panel: np.ndarray) -> dict[str, tuple[int, int, int]]:
    controls: dict[str, tuple[int, int, int]] = {}
    height, width = panel.shape
    coords = np.argwhere(panel != PANEL_COLOR)
    assert 2 <= len(coords) <= 4, "expected 2-4 controls"

    for r, c in coords:
        rr = int(r)
        cc = int(c)
        quadrant = _quadrant_of(rr, cc, height, width)
        assert quadrant not in controls, "at most one control per quadrant"
        controls[quadrant] = (rr, cc, int(panel[rr, cc]))
    return controls


def _stamp_from_controls(panel: np.ndarray, glyphs: dict[int, np.ndarray]) -> np.ndarray:
    height, width = panel.shape
    out = np.full((height, width), PANEL_COLOR, dtype=int)
    boxes = _quadrant_boxes(height, width)
    controls = _extract_controls(panel)

    for quadrant, (_r, _c, color) in controls.items():
        assert color in glyphs, "every control color must exist in the dictionary"
        center_r, center_c = _glyph_center(boxes[quadrant])
        patch = np.where(glyphs[color], color, PANEL_COLOR)
        paste(out, patch, center_r - 1, center_c - 1)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp dictionary glyphs into the quadrants indicated by same-colored control pixels inside a grey panel.

    Input structure:
    - The grid has a black background, a grey solid rectangular panel on the right,
      and a dictionary of 2-4 monochrome 3x3 glyphs on the left.
    - Each dictionary entry has a single same-colored key pixel immediately left of
      the glyph's 3x3 bbox.
    - The panel contains 2-4 non-grey single-pixel controls, with at most one
      control in each quadrant.

    Rule:
    - Find the grey panel bbox and split it into quadrants using the panel center;
      when a central row or column exists, it belongs to the upper-left side.
    - For each control pixel, use its position to choose the target quadrant and
      use its color to look up the matching 3x3 glyph in the left dictionary.
    - Remove the control pixel and stamp that glyph at the center of the chosen
      quadrant, recolored with the control's color.
    - Leave quadrants without controls unchanged and return only the cropped panel.

    Key insight:
    - The control pixel encodes two independent facts at once: its location selects
      where to stamp, and its color selects which glyph to stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == BACKGROUND, "expected black background"
    panel_box = _panel_bbox(g)
    glyphs = _parse_dictionary(g, panel_box, bg)
    panel = crop_bbox(g, panel_box)
    return _stamp_from_controls(panel, glyphs)


def _entry_patch(color: int, glyph: np.ndarray) -> np.ndarray:
    patch = np.full((3, 4), BACKGROUND, dtype=int)
    patch[1, 0] = color
    patch[:, 1:] = np.where(glyph, color, BACKGROUND)
    return patch


def _distinct_output_shape(glyph: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in glyph.flatten())


def _quadrant_stamp_box(height: int, width: int, quadrant: str) -> tuple[int, int, int, int]:
    center_r, center_c = _glyph_center(_quadrant_boxes(height, width)[quadrant])
    return center_r - 1, center_c - 1, center_r + 1, center_c + 1


def _control_candidates(height: int, width: int, quadrant: str) -> list[tuple[int, int]]:
    qbox = _quadrant_boxes(height, width)[quadrant]
    sr0, sc0, sr1, sc1 = _quadrant_stamp_box(height, width, quadrant)
    candidates = []
    for r in range(qbox[0], qbox[2] + 1):
        for c in range(qbox[1], qbox[3] + 1):
            if sr0 <= r <= sr1 and sc0 <= c <= sc1:
                continue
            candidates.append((r, c))
    return candidates


def _special_line_candidates(height: int, width: int, quadrant: str) -> list[tuple[int, int]]:
    r_mid = (height - 1) // 2
    c_mid = (width - 1) // 2
    want = set()
    for r, c in _control_candidates(height, width, quadrant):
        if height % 2 == 1 and r == r_mid:
            want.add((r, c))
        if width % 2 == 1 and c == c_mid:
            want.add((r, c))
    return sorted(want)


def _direct_expected(spec: dict) -> np.ndarray:
    panel = np.full((spec["panel_h"], spec["panel_w"]), PANEL_COLOR, dtype=int)
    for quadrant, color in spec["controls"].items():
        box = _quadrant_stamp_box(spec["panel_h"], spec["panel_w"], quadrant)
        r0, c0, r1, c1 = box
        glyph = spec["glyphs"][color]
        panel[r0 : r1 + 1, c0 : c1 + 1] = np.where(glyph, color, PANEL_COLOR)
    return panel


def _render_input(spec: dict) -> np.ndarray:
    grid = np.full((spec["grid_h"], spec["grid_w"]), BACKGROUND, dtype=int)

    row = spec["dict_top"]
    for color in spec["dict_colors"]:
        patch = _entry_patch(color, spec["glyphs"][color])
        paste(grid, patch, row, spec["dict_left"])
        row += 4

    pr0 = spec["panel_top"]
    pc0 = spec["panel_left"]
    grid[pr0 : pr0 + spec["panel_h"], pc0 : pc0 + spec["panel_w"]] = PANEL_COLOR

    for quadrant, color in spec["controls"].items():
        r, c = spec["control_positions"][quadrant]
        grid[pr0 + r, pc0 + c] = color

    return grid


def _all_shapes_distinct(glyphs: Iterable[np.ndarray]) -> bool:
    shapes = [_distinct_output_shape(glyph) for glyph in glyphs]
    return len(shapes) == len(set(shapes))


def _semantic_checks(spec: dict, grid: np.ndarray, out: np.ndarray) -> bool:
    expected = _direct_expected(spec)
    if not np.array_equal(out, expected):
        return False

    panel_h = spec["panel_h"]
    panel_w = spec["panel_w"]
    if not _all_shapes_distinct(spec["glyphs"].values()):
        return False

    used_quadrants = list(spec["controls"])
    used_colors = [spec["controls"][q] for q in used_quadrants]
    if len(set(used_quadrants)) != len(used_quadrants):
        return False
    if len(used_quadrants) < 2:
        return False

    # Position and color must each matter independently.
    first_q = used_quadrants[0]
    first_color = spec["controls"][first_q]

    other_q = next(q for q in used_quadrants if q != first_q)
    moved = dict(spec["controls"])
    moved[first_q], moved[other_q] = moved[other_q], moved[first_q]
    if moved[first_q] == first_color and moved[other_q] == spec["controls"][other_q]:
        return False
    moved_spec = dict(spec)
    moved_spec["controls"] = moved
    if np.array_equal(_direct_expected(moved_spec), expected):
        return False

    alt_colors = [c for c in spec["dict_colors"] if _distinct_output_shape(spec["glyphs"][c]) != _distinct_output_shape(spec["glyphs"][first_color])]
    if not alt_colors:
        return False
    recolored = dict(spec["controls"])
    recolored[first_q] = random.choice(alt_colors)
    recolored_spec = dict(spec)
    recolored_spec["controls"] = recolored
    if np.array_equal(_direct_expected(recolored_spec), expected):
        return False

    parsed = main(grid)
    if not np.array_equal(parsed, expected):
        return False

    # Ensure every used control lies outside its final stamp bbox, so the "remove
    # control then stamp" step is observable rather than incidental.
    for quadrant, (r, c) in spec["control_positions"].items():
        sr0, sc0, sr1, sc1 = _quadrant_stamp_box(panel_h, panel_w, quadrant)
        if sr0 <= r <= sr1 and sc0 <= c <= sc1:
            return False

    return True


def _sample_spec() -> dict | None:
    entry_count = random.randint(2, 4)
    dict_colors = random.sample(KEY_COLORS, entry_count)
    glyph_masks = random.sample(GLYPH_LIBRARY, entry_count)
    glyphs = {color: mask.copy() for color, mask in zip(dict_colors, glyph_masks)}

    panel_h = random.randint(8, 13)
    panel_w = random.randint(8, 13)
    boxes = _quadrant_boxes(panel_h, panel_w)

    quadrants = random.sample(list(boxes), random.randint(2, 4))
    controls: dict[str, int] = {}
    if random.random() < 0.4 and len(quadrants) >= 2:
        repeated_color = random.choice(dict_colors)
        pair = random.sample(quadrants, 2)
        controls[pair[0]] = repeated_color
        controls[pair[1]] = repeated_color
    for quadrant in quadrants:
        controls.setdefault(quadrant, random.choice(dict_colors))

    control_positions: dict[str, tuple[int, int]] = {}
    for quadrant in quadrants:
        special = _special_line_candidates(panel_h, panel_w, quadrant)
        candidates = _control_candidates(panel_h, panel_w, quadrant)
        if not candidates:
            return None
        if special and random.random() < 0.35:
            control_positions[quadrant] = random.choice(special)
        else:
            control_positions[quadrant] = random.choice(candidates)

    dict_h = entry_count * 3 + (entry_count - 1)
    dict_left = random.randint(1, 2)
    dict_top = random.randint(1, 4)

    panel_top = random.randint(1, 6)
    gap = random.randint(2, 4)
    panel_left = dict_left + 4 + gap

    grid_h = max(dict_top + dict_h + random.randint(1, 4), panel_top + panel_h + random.randint(1, 4))
    grid_w = panel_left + panel_w + random.randint(1, 4)
    if grid_h > 30 or grid_w > 30:
        return None

    spec = {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "dict_colors": dict_colors,
        "glyphs": glyphs,
        "dict_left": dict_left,
        "dict_top": dict_top,
        "panel_h": panel_h,
        "panel_w": panel_w,
        "panel_top": panel_top,
        "panel_left": panel_left,
        "controls": controls,
        "control_positions": control_positions,
    }
    return spec


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        spec = _sample_spec()
        if spec is None:
            continue
        grid = _render_input(spec)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if _semantic_checks(spec, grid, out):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
