from collections import Counter

import numpy as np

from arc_utils import components_mask, crop_bbox, find_components, get_bg, paste
import itertools
import random


def _shape_key(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.asarray(coords, dtype=int)
    assert coords.size > 0
    origin = coords.min(axis=0)
    norm = coords - origin
    height = int(norm[:, 0].max()) + 1
    width = int(norm[:, 1].max()) + 1
    cells = tuple(sorted((int(r), int(c)) for r, c in norm))
    return height, width, cells


def _inside_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        np.all(
            (coords[:, 0] >= r0)
            & (coords[:, 0] <= r1)
            & (coords[:, 1] >= c0)
            & (coords[:, 1] <= c1)
        )
    )


def _component_mask_in_bbox(
    shape: tuple[int, int], coords: np.ndarray, r0: int, c0: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _candidate_match(
    g: np.ndarray, comp: dict, bg: int, components_by_shape: dict
) -> tuple[list[np.ndarray], dict] | None:
    r0, c0, _, _ = comp["bbox"]
    crop = crop_bbox(g, comp["bbox"])
    obj_mask = _component_mask_in_bbox(crop.shape, comp["coords"], r0, c0)

    if not np.array_equal(crop != bg, obj_mask):
        return None

    gap_mask = ~obj_mask
    if not gap_mask.any():
        return None

    gaps = components_mask(gap_mask, connectivity=4)
    needed = Counter(_shape_key(gap) for gap in gaps)
    fill_by_shape = {}

    for key, count in needed.items():
        markers = [
            other
            for other in components_by_shape.get(key, [])
            if not _inside_bbox(other["coords"], comp["bbox"])
        ]
        colors = {int(marker["color"]) for marker in markers}
        if len(markers) != count or len(colors) != 1:
            return None
        fill_by_shape[key] = next(iter(colors))

    return gaps, fill_by_shape


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the foreground object whose missing cells match marker shapes, and fill those missing cells with the marker colors.

    Input structure
    A dominant background surrounds several monochromatic foreground objects. Exactly one object has background gaps inside its bounding box whose component shapes are matched by separate marker objects.
    Rule
    Find that object, crop its bounding box, and replace each background gap component with the color of the separate marker component or components having the same normalized shape.
    Key insight
    The marker objects do not mark absolute positions; their shapes identify which gaps in the selected object should be colored, while objects with unmatched gaps are distractors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"
    )
    assert components

    components_by_shape = {}
    for comp in components:
        components_by_shape.setdefault(_shape_key(comp["coords"]), []).append(comp)

    matches = []
    for comp in components:
        match = _candidate_match(g, comp, bg, components_by_shape)
        if match is not None:
            gaps, fill_by_shape = match
            matches.append((comp, gaps, fill_by_shape))

    assert len(matches) == 1
    target, gaps, fill_by_shape = matches[0]
    out = crop_bbox(g, target["bbox"])

    for gap in gaps:
        out[gap[:, 0], gap[:, 1]] = fill_by_shape[_shape_key(gap)]

    assert not np.any(out == bg)
    return out

# --------------- generator (att2, o5x) ---------------

from arc_utils import bbox


Rect = tuple[int, int, int, int]
BBox = tuple[int, int, int, int]
TargetConfigs = dict[tuple[int, int, bool], list[tuple[Rect, Rect]]]
DistractorConfig = tuple[int, int, np.ndarray, int, int]

# Canvas and palette.
_GRID_SIZE = 16
_COLORS = list(range(10))

# Target geometry and gap constraints.
_TARGET_HEIGHTS, _TARGET_WIDTHS = range(3, 8), range(4, 7)
_TARGET_FILL_RATIO_MIN, _TARGET_FILL_RATIO_MAX = 0.60, 0.75
_GAP_HEIGHTS, _GAP_WIDTHS = range(1, 4), range(1, 3)
_GAP_SHAPES = [(r, c) for r in _GAP_HEIGHTS for c in _GAP_WIDTHS]
_TARGET_SAME_SHAPE_CHOICES = [False, True]

# Distractor geometry and count constraints.
_DISTRACTOR_HEIGHTS, _DISTRACTOR_WIDTHS = range(3, 8), range(4, 8)
_DISTRACTOR_CELLS_RANGE = range(8, 33)
_DISTRACTOR_HOLE_MIN_AREA, _SNAKE_HOLE_WIDTH = 3, 3
_DISTRACTOR_COUNT_CHOICES = [1, 2]

# Placement and validation.
_BBOX_SEPARATION, _TARGET_MARGIN = 2, 2
_COMPONENT_CONNECTIVITY, _GENERATION_RETRIES = 4, 300


def _rect_mask(shape: tuple[int, int], rect: Rect) -> np.ndarray:
    r, c, h, w = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r : r + h, c : c + w] = True
    return mask


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_target_masks(gaps: np.ndarray, target: np.ndarray) -> bool:
    return (
        len(components_mask(gaps, connectivity=_COMPONENT_CONNECTIVITY)) == 2
        and len(components_mask(target, connectivity=_COMPONENT_CONNECTIVITY)) == 1
        and _touches_all_edges(target)
    )


def _make_target_configs() -> TargetConfigs:
    configs: TargetConfigs = {}
    for h in _TARGET_HEIGHTS:
        for w in _TARGET_WIDTHS:
            for s1, s2 in itertools.product(_GAP_SHAPES, repeat=2):
                gap_area = s1[0] * s1[1] + s2[0] * s2[1]
                fill_ratio = (h * w - gap_area) / (h * w)
                if not _TARGET_FILL_RATIO_MIN <= fill_ratio <= _TARGET_FILL_RATIO_MAX:
                    continue
                for r1 in range(h - s1[0] + 1):
                    for c1 in range(w - s1[1] + 1):
                        m1 = _rect_mask((h, w), (r1, c1, *s1))
                        for r2 in range(h - s2[0] + 1):
                            for c2 in range(w - s2[1] + 1):
                                m2 = _rect_mask((h, w), (r2, c2, *s2))
                                if np.any(m1 & m2):
                                    continue
                                gaps = m1 | m2
                                target = ~gaps
                                if not _valid_target_masks(gaps, target):
                                    continue
                                same_shape = s1 == s2
                                configs.setdefault((h, w, same_shape), []).append(
                                    ((r1, c1, *s1), (r2, c2, *s2))
                                )
    return configs


_TARGET_CONFIGS = _make_target_configs()
_TARGET_DIMS = sorted({(h, w) for h, w, _ in _TARGET_CONFIGS})


def _snake_hole(area: int) -> np.ndarray:
    q, rem = divmod(int(area), _SNAKE_HOLE_WIDTH)
    coords = []
    if rem == 0:
        for r in range(max(0, q - 1)):
            coords.extend((r, c) for c in range(_SNAKE_HOLE_WIDTH))
        base = max(0, q - 1)
        coords.extend([(base, 0), (base, 1), (base + 1, 0)])
    else:
        for r in range(q):
            coords.extend((r, c) for c in range(_SNAKE_HOLE_WIDTH))
        coords.extend((q, c) for c in range(rem))
    mask = np.zeros((max(r for r, _ in coords) + 1, _SNAKE_HOLE_WIDTH), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _make_distractor_configs() -> dict[int, list[DistractorConfig]]:
    configs: dict[int, list[DistractorConfig]] = {}
    for h in _DISTRACTOR_HEIGHTS:
        for w in _DISTRACTOR_WIDTHS:
            for cells in _DISTRACTOR_CELLS_RANGE:
                hole_area = h * w - cells
                if hole_area < _DISTRACTOR_HOLE_MIN_AREA:
                    continue
                hole = _snake_hole(hole_area)
                hh, hw = hole.shape
                if hh > h or hw > w:
                    continue
                for r in range(h - hh + 1):
                    for c in range(w - hw + 1):
                        gap = np.zeros((h, w), dtype=bool)
                        gap[r : r + hh, c : c + hw][hole] = True
                        obj = ~gap
                        if len(components_mask(obj, connectivity=_COMPONENT_CONNECTIVITY)) != 1:
                            continue
                        if not _touches_all_edges(obj):
                            continue
                        configs.setdefault(cells, []).append((h, w, hole, r, c))
    return configs


_DISTRACTOR_CONFIGS = _make_distractor_configs()
_DISTRACTOR_CELLS = sorted(_DISTRACTOR_CONFIGS)


def _bbox_far(a: BBox, b: BBox, pad: int = _BBOX_SEPARATION) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_patch(
    grid: np.ndarray, patch: np.ndarray, bg: int, placed: list[BBox], *, margin: int = 0
) -> BBox | None:
    """Paste one patch if a separated location exists; returns its foreground bbox."""
    h, w = patch.shape
    patch_mask = patch != bg
    patch_box = bbox(patch_mask)
    assert patch_box is not None
    rr0, cc0, rr1, cc1 = patch_box
    candidates = []
    for r in range(margin, _GRID_SIZE - h - margin + 1):
        for c in range(margin, _GRID_SIZE - w - margin + 1):
            box = (r + rr0, c + cc0, r + rr1, c + cc1)
            if any(not _bbox_far(box, old) for old in placed):
                continue
            if np.any(patch_mask & (grid[r : r + h, c : c + w] != bg)):
                continue
            candidates.append((r, c, box))
    if not candidates:
        return None
    r, c, box = random.choice(candidates)
    paste(grid, patch, r, c, transparent=bg)
    placed.append(box)
    return box


def _distractor_patch(color: int, bg: int) -> np.ndarray:
    cells = random.choice(_DISTRACTOR_CELLS)
    choices = _DISTRACTOR_CONFIGS[cells]
    h, w, hole, r, c = random.choice(choices)
    patch = np.full((h, w), color, dtype=int)
    patch[r : r + hole.shape[0], c : c + hole.shape[1]][hole] = bg
    return patch


def _sample_scene_parameters() -> tuple[int, bool, int, list[int], list[int]]:
    bg = random.choice(_COLORS)
    target_same_shape = random.choice(_TARGET_SAME_SHAPE_CHOICES)
    n_distractors = random.choice(_DISTRACTOR_COUNT_CHOICES)
    marker_color_count = 1 if target_same_shape else 2
    palette = random.sample([c for c in _COLORS if c != bg], 1 + marker_color_count + n_distractors)
    target_color = palette[0]
    marker_palette = palette[1 : 1 + marker_color_count]
    distractor_colors = palette[1 + marker_color_count :]
    return bg, target_same_shape, target_color, marker_palette, distractor_colors


def _sample_target_rects(target_same_shape: bool) -> tuple[int, int, list[Rect]] | None:
    h, w = random.choice(_TARGET_DIMS)
    pool = _TARGET_CONFIGS.get((h, w, target_same_shape))
    if not pool:
        return None
    return h, w, list(random.choice(pool))


def _build_target_and_markers(
    h: int, w: int, rects: list[Rect], target_color: int, marker_palette: list[int], bg: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return target-with-holes, expected solved crop, and matching marker patches."""
    target = np.full((h, w), target_color, dtype=int)
    expected = target.copy()
    marker_specs = []
    shape_to_color: dict[tuple[int, int], int] = {}
    next_marker = 0
    for r, c, gh, gw in rects:
        shape = (gh, gw)
        if shape not in shape_to_color:
            shape_to_color[shape] = marker_palette[next_marker]
            next_marker += 1
        color = shape_to_color[shape]
        target[r : r + gh, c : c + gw] = bg
        expected[r : r + gh, c : c + gw] = color
        marker_specs.append(np.full((gh, gw), color, dtype=int))
    return target, expected, marker_specs


def _place_scene_objects(
    grid: np.ndarray, target: np.ndarray, markers: list[np.ndarray], distractor_colors: list[int], bg: int
) -> bool:
    placed: list[BBox] = []
    if _place_patch(grid, target, bg, placed, margin=_TARGET_MARGIN) is None:
        return False

    for marker in markers:
        if _place_patch(grid, marker, bg, placed) is None:
            return False

    distractors = [_distractor_patch(color, bg) for color in distractor_colors]
    random.shuffle(distractors)
    for distractor in distractors:
        if _place_patch(grid, distractor, bg, placed) is None:
            return False
    return True


def _oracle_accepts(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    bg, target_same_shape, target_color, marker_palette, distractor_colors = _sample_scene_parameters()
    target_rects = _sample_target_rects(target_same_shape)
    if target_rects is None:
        return None
    h, w, rects = target_rects
    target, expected, markers = _build_target_and_markers(
        h, w, rects, target_color, marker_palette, bg
    )

    grid = np.full((_GRID_SIZE, _GRID_SIZE), bg, dtype=int)
    if not _place_scene_objects(grid, target, markers, distractor_colors, bg):
        return None
    if not _oracle_accepts(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GENERATION_RETRIES} attempts")
