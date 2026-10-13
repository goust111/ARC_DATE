from __future__ import annotations

import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, colors_present, get_bg
import itertools
import random


def _unique_large_foreground_color(g: np.ndarray, bg: int) -> int:
    colors = colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected a fill mask and a colored template"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    assert len(candidates) == 1, "large fill-mask color must be unique by area"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the enlarged mask with the small colored template rotated 90 degrees clockwise.

    Input structure
    - A background grid contains one small multicolor template and one much larger monochrome mask.
    - The monochrome mask is an integer block scale of the template foreground mask after a 90-degree clockwise rotation.

    Rule
    - Rotate the template bbox 90 degrees clockwise, scale each template pixel to the matching block size, and use that scaled template to recolor the large mask pixels.
    - Preserve the original template and all background pixels.

    Key insight
    - The large mask gives only the silhouette; the small template supplies the colors once it is rotated and scaled to that silhouette.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    mask_color = _unique_large_foreground_color(g, bg)

    target_mask = g == mask_color
    template_mask = (g != bg) & (g != mask_color)
    template_bbox = bbox(template_mask)
    target_bbox = bbox(target_mask)
    assert template_bbox is not None, "expected a colored template"
    assert target_bbox is not None, "expected a large fill mask"

    tr0, tc0, tr1, tc1 = template_bbox
    dr0, dc0, dr1, dc1 = target_bbox
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert not np.any(template == mask_color), "template must not use the mask color"

    rotated = np.rot90(template, k=-1)
    rot_h, rot_w = rotated.shape
    dest_h, dest_w = bbox_h(target_bbox), bbox_w(target_bbox)
    assert dest_h % rot_h == 0, "target height must be a scaled rotated-template height"
    assert dest_w % rot_w == 0, "target width must be a scaled rotated-template width"
    scale_h = dest_h // rot_h
    scale_w = dest_w // rot_w
    assert scale_h == scale_w and scale_h >= 1, "target must use one square block scale"

    scaled = np.kron(rotated, np.ones((scale_h, scale_w), dtype=int))
    expected_mask = scaled != bg
    actual_mask = target_mask[dr0 : dr1 + 1, dc0 : dc1 + 1]
    assert expected_mask.shape == actual_mask.shape
    assert np.array_equal(expected_mask, actual_mask), "target mask must match the rotated scaled template"

    out = g.copy()
    out_region = out[dr0 : dr1 + 1, dc0 : dc1 + 1]
    out_region[actual_mask] = scaled[actual_mask]
    return out

# --------------- generator (o5x) ---------------

from arc_utils import find_components, paste

# Template colors and structure.
_GEN_BG_COLOR, _GEN_MASK_COLOR = 0, 8
_GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
_GEN_TEMPLATE_MIN_FILL, _GEN_TEMPLATE_MAX_FILL = 0.5, 0.7
_GEN_TEMPLATE_MIN_COLORS, _GEN_TEMPLATE_MAX_COLORS = 3, 4
_GEN_TEMPLATE_COMPONENTS, _GEN_COMPONENT_CONNECTIVITY = 2, 4

# Template size and mask scale sampling.
_GEN_SIZE_SCALE_PAIRS = [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3)]
_GEN_SIZE_SCALE_WEIGHTS = [1, 1, 4, 3, 3]

# Canvas bounds and object spacing.
_GEN_CANVAS_MIN_H, _GEN_CANVAS_MAX_H = 16, 24
_GEN_CANVAS_MIN_W, _GEN_CANVAS_MAX_W = 12, 26
_GEN_EDGE_PADDING = 1
_GEN_VERTICAL_GAP, _GEN_HORIZONTAL_GAP = 4, 7

# Generation retry budget.
_GEN_RETRY_LIMIT = 300
_GEN_MASK_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _gen_template_masks(n: int, k: int) -> list[np.ndarray]:
    """Return full-bbox masks with the required 4-connected component count."""
    key = (int(n), int(k))
    if key in _GEN_MASK_CACHE:
        return _GEN_MASK_CACHE[key]

    masks: list[np.ndarray] = []
    for cells in itertools.combinations(range(n * n), k):
        flat = np.zeros(n * n, dtype=bool)
        flat[list(cells)] = True
        mask = flat.reshape(n, n)
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != n - 1:
            continue
        if cols.min() != 0 or cols.max() != n - 1:
            continue
        comps = find_components(mask.astype(int), bg=_GEN_BG_COLOR, connectivity=_GEN_COMPONENT_CONNECTIVITY, monochromatic=True)
        if len(comps) == _GEN_TEMPLATE_COMPONENTS:
            masks.append(mask.copy())

    assert masks, f"no valid template masks for n={n}, k={k}"
    _GEN_MASK_CACHE[key] = masks
    return masks


def _gen_choose_size_scale() -> tuple[int, int]:
    return random.choices(_GEN_SIZE_SCALE_PAIRS, weights=_GEN_SIZE_SCALE_WEIGHTS, k=1)[0]


def _gen_template(n: int) -> np.ndarray:
    valid_counts = [k for k in range(n * n + 1) if _GEN_TEMPLATE_MIN_FILL <= k / (n * n) <= _GEN_TEMPLATE_MAX_FILL]
    k = random.choice(valid_counts)
    fg_mask = random.choice(_gen_template_masks(n, k))

    num_colors = random.randint(_GEN_TEMPLATE_MIN_COLORS, _GEN_TEMPLATE_MAX_COLORS)
    colors = random.sample(_GEN_COLORS, num_colors)
    coords = [tuple(rc) for rc in np.argwhere(fg_mask)]
    assignments = colors + [random.choice(colors) for _ in range(len(coords) - num_colors)]
    random.shuffle(assignments)

    template = np.zeros((n, n), dtype=int)
    for (r, c), color in zip(coords, assignments):
        template[r, c] = color
    return template


def _gen_canvas_shape(min_h: int, min_w: int, mask_area: int, template_area: int) -> tuple[int, int] | None:
    h0 = max(_GEN_CANVAS_MIN_H, int(min_h))
    w0 = max(_GEN_CANVAS_MIN_W, int(min_w))
    choices = []
    for h in range(h0, _GEN_CANVAS_MAX_H + 1):
        for w in range(w0, _GEN_CANVAS_MAX_W + 1):
            bg_area = h * w - int(mask_area) - int(template_area)
            if bg_area > mask_area:
                choices.append((h, w))
    if not choices:
        return None
    return random.choice(choices)


def _gen_layout_min_shape(n: int, m: int, layout: str) -> tuple[int, int]:
    if layout == "vertical":
        return n + m + _GEN_VERTICAL_GAP + 2 * _GEN_EDGE_PADDING, max(n, m) + 2 * _GEN_EDGE_PADDING
    return max(n, m) + 2 * _GEN_EDGE_PADDING, n + m + _GEN_HORIZONTAL_GAP + 2 * _GEN_EDGE_PADDING


def _gen_layout_choices(n: int, m: int) -> list[tuple[str, tuple[int, int]]]:
    choices = []
    for layout in ["vertical", "horizontal"]:
        min_h, min_w = _gen_layout_min_shape(n, m, layout)
        if min_h <= _GEN_CANVAS_MAX_H and min_w <= _GEN_CANVAS_MAX_W:
            choices.append((layout, (min_h, min_w)))
    return choices


def _gen_place_pair(grid_shape: tuple[int, int], template: np.ndarray, target: np.ndarray, layout: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Sample non-overlapping top-left positions under the chosen layout."""
    h, w = grid_shape
    n = template.shape[0]
    m = target.shape[0]
    min_h, min_w = _gen_layout_min_shape(n, m, layout)
    if h < min_h or w < min_w:
        return None

    template_first = random.choice([True, False])
    first_size, second_size = (n, m) if template_first else (m, n)

    if layout == "vertical":
        first_row = random.randint(_GEN_EDGE_PADDING, h - first_size - _GEN_VERTICAL_GAP - second_size - _GEN_EDGE_PADDING)
        second_row = random.randint(first_row + first_size + _GEN_VERTICAL_GAP, h - second_size - _GEN_EDGE_PADDING)
        first_col = random.randint(_GEN_EDGE_PADDING, w - first_size - _GEN_EDGE_PADDING)
        second_col = random.randint(_GEN_EDGE_PADDING, w - second_size - _GEN_EDGE_PADDING)
        first_pos = (first_row, first_col)
        second_pos = (second_row, second_col)
    else:
        first_col = random.randint(_GEN_EDGE_PADDING, w - first_size - _GEN_HORIZONTAL_GAP - second_size - _GEN_EDGE_PADDING)
        second_col = random.randint(first_col + first_size + _GEN_HORIZONTAL_GAP, w - second_size - _GEN_EDGE_PADDING)
        first_row = random.randint(_GEN_EDGE_PADDING, h - first_size - _GEN_EDGE_PADDING)
        second_row = random.randint(_GEN_EDGE_PADDING, h - second_size - _GEN_EDGE_PADDING)
        first_pos = (first_row, first_col)
        second_pos = (second_row, second_col)

    if template_first:
        return first_pos, second_pos
    return second_pos, first_pos


def _gen_scaled_target(template: np.ndarray, scale: int) -> np.ndarray:
    rotated = np.rot90(template, k=-1)
    return np.kron(rotated, np.ones((scale, scale), dtype=int))


def _gen_render_pair(shape: tuple[int, int], template: np.ndarray, target: np.ndarray, positions: tuple[tuple[int, int], tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Render matching input/output grids for a placed template and target."""
    (tr, tc), (mr, mc) = positions
    m = target.shape[0]
    output = np.zeros(shape, dtype=int)
    paste(output, template, tr, tc, transparent=_GEN_BG_COLOR)
    paste(output, target, mr, mc, transparent=_GEN_BG_COLOR)
    inp = output.copy()
    region = inp[mr : mr + m, mc : mc + m]
    region[target != _GEN_BG_COLOR] = _GEN_MASK_COLOR
    return inp, output


def _gen_attempt() -> np.ndarray | None:
    """Generate one valid candidate or return None for retry."""
    n, scale = _gen_choose_size_scale()
    template = _gen_template(n)
    target = _gen_scaled_target(template, scale)

    mask_area = int(np.count_nonzero(target))
    template_area = int(np.count_nonzero(template))
    m = target.shape[0]

    layout_choices = _gen_layout_choices(n, m)
    if not layout_choices:
        return None

    layout, (min_h, min_w) = random.choice(layout_choices)
    shape = _gen_canvas_shape(min_h, min_w, mask_area, template_area)
    if shape is None:
        return None

    positions = _gen_place_pair(shape, template, target, layout)
    if positions is None:
        return None

    inp, output = _gen_render_pair(shape, template, target, positions)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 103eff5b input")
