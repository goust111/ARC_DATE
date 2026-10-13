from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_fable_1_034_3", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"unable to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
colors_present = _ARC_UTILS.colors_present
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


def _reading_key(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(box[0]), int(box[1])


def _rect_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _component_is_frame(coords: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
    if coords.size == 0:
        return False, None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False, None
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = _rect_border_mask((h, w), (0, 0, h - 1, w - 1))
    if int(local.sum()) != int(border.sum()):
        return False, None
    if not np.array_equal(local, border):
        return False, None
    return True, box


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in colors_present(grid, exclude={bg}):
        components = find_components(
            grid,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        )
        if len(components) != 2:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for component in components:
            valid, box = _component_is_frame(component["coords"])
            if not valid or box is None:
                ok = False
                break
            boxes.append(box)
        if ok:
            candidates.append((int(color), sorted(boxes, key=_reading_key)))

    assert len(candidates) == 1, "expected exactly one frame color"
    return candidates[0]


def _find_marker_color(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_boxes: list[tuple[int, int, int, int]],
) -> int:
    inside_mask = np.zeros(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        inside_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    candidates = [
        int(color)
        for color in colors_present(grid, exclude={bg, frame_color})
        if np.any((grid == int(color)) & inside_mask)
    ]
    assert len(candidates) == 1, "expected exactly one marker color"
    return candidates[0]


def _count_inside(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    return int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == int(color)))


def _extract_sprites(grid: np.ndarray, bg: int, frame_color: int, marker_color: int) -> list[dict]:
    sprite_grid = np.asarray(grid, dtype=int).copy()
    sprite_grid[sprite_grid == int(frame_color)] = int(bg)
    sprite_grid[sprite_grid == int(marker_color)] = int(bg)
    components = find_components(
        sprite_grid,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) >= 2, "expected at least two sprites"

    ordered = sorted(components, key=lambda comp: _reading_key(comp["bbox"]))
    sprite_shape = None
    sprites = []
    for component in ordered:
        box = tuple(map(int, component["bbox"]))
        crop = crop_bbox(sprite_grid, box)
        if sprite_shape is None:
            sprite_shape = crop.shape
        else:
            assert crop.shape == sprite_shape, "all sprite bboxes must share one shape"
        sprites.append({"bbox": box, "crop": crop})
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the two frame-internal marker counts as sprite indices and transparently overlay the selected sprites.

    Input structure
    - A dominant background contains exactly two same-colored one-cell-thick rectangular frames.
    - One marker color appears both strictly inside the frames and elsewhere as distractor pixels.
    - The remaining non-background pixels form several separated multicolor sprites whose bboxes all share the same size.

    Rule
    - Sort the two frames by reading order of their bbox top-left corners and count the marker-color pixels strictly inside each frame, giving N1 then N2.
    - Remove the frame color and marker color, then sort the remaining sprites by reading order of their bbox top-left corners and number them 1, 2, 3, ...
    - Take sprite N1 and sprite N2, align their bboxes at the same top-left corner, and overlay them using the background as transparent, with the first frame's sprite on top.

    Key insight
    - The marker counts are not rendered directly: each count is an address into the sprite list, and frame order determines the overlay layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    frame_color, frame_boxes = _find_frames(g, bg)
    marker_color = _find_marker_color(g, bg, frame_color, frame_boxes)
    counts = [_count_inside(g, box, marker_color) for box in frame_boxes]

    sprites = _extract_sprites(g, bg, frame_color, marker_color)
    assert all(1 <= count <= len(sprites) for count in counts), "frame counts must index existing sprites"

    lower = sprites[counts[1] - 1]["crop"]
    upper = sprites[counts[0] - 1]["crop"]
    out = np.full(lower.shape, bg, dtype=int)
    paste(out, lower, 0, 0, transparent=bg)
    paste(out, upper, 0, 0, transparent=bg)
    return out


_GEN_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400
_GEN_OVERLAP_BAG: list[bool] = []

_GEN_BG = Color.BLACK
_GEN_CANVAS_MIN = 20
_GEN_CANVAS_MAX = 30
_GEN_SPRITE_COUNT_RANGE = (4, 6)
_GEN_SPRITE_H_RANGE = (4, 6)
_GEN_SPRITE_W_RANGE = (4, 6)
_GEN_SPRITE_DENSITY_RANGE = (0.40, 0.72)
_GEN_FRAME_OUTER_H_RANGE = (5, 8)
_GEN_FRAME_OUTER_W_RANGE = (5, 9)
_GEN_BOX_PADDING = 1
_GEN_OUTSIDE_MARKER_RANGE = (3, 7)


def _gen_randint(lo: int, hi: int) -> int:
    return int(_GEN_RNG.integers(int(lo), int(hi) + 1))


def _gen_choice(seq):
    return seq[int(_GEN_RNG.integers(0, len(seq)))]


def _gen_overlap_mode() -> bool:
    global _GEN_OVERLAP_BAG
    if not _GEN_OVERLAP_BAG:
        _GEN_OVERLAP_BAG = [True, True, False]
        _GEN_RNG.shuffle(_GEN_OVERLAP_BAG)
    return bool(_GEN_OVERLAP_BAG.pop())


def _gen_line8_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, start)
    tr, tc = map(int, end)
    path = [(r, c)]
    while (r, c) != (tr, tc):
        dr = 0 if r == tr else (1 if tr > r else -1)
        dc = 0 if c == tc else (1 if tc > c else -1)
        if dr != 0 and dc != 0 and _GEN_RNG.random() < 0.35:
            if _GEN_RNG.random() < 0.5:
                dr = 0
            else:
                dc = 0
        r += dr
        c += dc
        path.append((r, c))
    return path


def _gen_mask_touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _gen_sprite_mask(h: int, w: int) -> np.ndarray | None:
    min_area = max(8, int(np.ceil(_GEN_SPRITE_DENSITY_RANGE[0] * h * w)))
    max_area = min(h * w - 1, int(np.floor(_GEN_SPRITE_DENSITY_RANGE[1] * h * w)))
    if min_area > max_area:
        return None
    target = _gen_randint(min_area, max_area)

    side_points = [
        (0, _gen_randint(0, w - 1)),
        (_gen_randint(0, h - 1), w - 1),
        (h - 1, _gen_randint(0, w - 1)),
        (_gen_randint(0, h - 1), 0),
    ]
    order = _GEN_RNG.permutation(len(side_points))
    cells = {side_points[int(order[0])]}
    prev = side_points[int(order[0])]
    for idx in order[1:]:
        point = side_points[int(idx)]
        cells.update(_gen_line8_path(prev, point))
        prev = point

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
        if not frontier:
            return None

        candidates = list(frontier)
        missing = []
        if not any(r == 0 for r, _ in cells):
            missing.extend([pt for pt in candidates if pt[0] == 0])
        if not any(r == h - 1 for r, _ in cells):
            missing.extend([pt for pt in candidates if pt[0] == h - 1])
        if not any(c == 0 for _, c in cells):
            missing.extend([pt for pt in candidates if pt[1] == 0])
        if not any(c == w - 1 for _, c in cells):
            missing.extend([pt for pt in candidates if pt[1] == w - 1])
        pick_from = missing if missing and _GEN_RNG.random() < 0.7 else candidates
        cells.add(pick_from[int(_GEN_RNG.integers(0, len(pick_from)))])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not _gen_mask_touches_all_sides(mask):
        return None
    border = _rect_border_mask(mask.shape, (0, 0, h - 1, w - 1))
    if np.array_equal(mask, border):
        return None
    return mask


def _gen_colorize_mask(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    cells = np.argwhere(mask)
    color_count = min(3, len(palette), len(cells))
    anchors_idx = _GEN_RNG.choice(len(cells), size=color_count, replace=False)
    anchors = cells[anchors_idx]
    colors = _GEN_RNG.choice(np.array(palette, dtype=int), size=color_count, replace=False)
    sprite = np.full(mask.shape, _GEN_BG, dtype=int)
    for r, c in cells:
        distances = np.abs(anchors[:, 0] - r) + np.abs(anchors[:, 1] - c)
        nearest = np.flatnonzero(distances == distances.min())
        idx = int(_GEN_RNG.choice(nearest))
        sprite[int(r), int(c)] = int(colors[idx])
    return sprite


def _gen_box_free(reserved: np.ndarray, r: int, c: int, h: int, w: int, pad: int, border: int) -> bool:
    rows, cols = reserved.shape
    if r < border or c < border or r + h > rows - border or c + w > cols - border:
        return False
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(rows, r + h + pad)
    cc1 = min(cols, c + w + pad)
    return not reserved[rr0:rr1, cc0:cc1].any()


def _gen_mark_box(reserved: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    rows, cols = reserved.shape
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(rows, r + h + pad)
    cc1 = min(cols, c + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _gen_place_boxes(
    reserved: np.ndarray,
    shapes: list[tuple[int, int]],
    *,
    pad: int = 1,
    border: int = 1,
) -> list[tuple[int, int]] | None:
    work = reserved.copy()
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    for idx in order:
        h, w = map(int, shapes[idx])
        candidates = [
            (r, c)
            for r in range(border, work.shape[0] - h - border + 1)
            for c in range(border, work.shape[1] - w - border + 1)
            if _gen_box_free(work, r, c, h, w, pad, border)
        ]
        if not candidates:
            return None
        r, c = candidates[int(_GEN_RNG.integers(0, len(candidates)))]
        placements[idx] = (int(r), int(c))
        _gen_mark_box(work, int(r), int(c), h, w, pad)
    reserved[:, :] = work
    return [placement for placement in placements if placement is not None]


def _gen_overlay_pair_kind(sprite_a: np.ndarray, sprite_b: np.ndarray) -> str:
    overlap = (sprite_a != _GEN_BG) & (sprite_b != _GEN_BG)
    if np.any(overlap & (sprite_a != sprite_b)):
        return "visible_overlap"
    if np.any(overlap):
        return "same_color_overlap"
    return "disjoint"


def _gen_frame_shape(max_count: int) -> tuple[int, int]:
    choices = []
    for h in range(_GEN_FRAME_OUTER_H_RANGE[0], _GEN_FRAME_OUTER_H_RANGE[1] + 1):
        for w in range(_GEN_FRAME_OUTER_W_RANGE[0], _GEN_FRAME_OUTER_W_RANGE[1] + 1):
            if (h - 2) * (w - 2) >= max_count:
                choices.append((h, w))
    return _gen_choice(choices)


def _gen_render_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _gen_expected_overlay(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    out = np.full(lower.shape, _GEN_BG, dtype=int)
    paste(out, lower, 0, 0, transparent=_GEN_BG)
    paste(out, upper, 0, 0, transparent=_GEN_BG)
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = _gen_randint(_GEN_CANVAS_MIN, _GEN_CANVAS_MAX)
    cols = _gen_randint(_GEN_CANVAS_MIN, _GEN_CANVAS_MAX)
    sprite_count = _gen_randint(*_GEN_SPRITE_COUNT_RANGE)
    sprite_h = _gen_randint(*_GEN_SPRITE_H_RANGE)
    sprite_w = _gen_randint(*_GEN_SPRITE_W_RANGE)

    palette = list(map(int, _GEN_RNG.permutation(np.arange(1, 10))))
    frame_color, marker_color = palette[:2]
    sprite_palette = palette[2:5]

    sprites = []
    for _ in range(sprite_count):
        mask = _gen_sprite_mask(sprite_h, sprite_w)
        if mask is None:
            return None
        sprites.append(_gen_colorize_mask(mask, sprite_palette))

    reserved = np.zeros((rows, cols), dtype=bool)
    sprite_shapes = [(sprite_h, sprite_w)] * sprite_count
    sprite_positions = _gen_place_boxes(reserved, sprite_shapes, pad=_GEN_BOX_PADDING, border=1)
    if sprite_positions is None:
        return None

    order = sorted(range(sprite_count), key=lambda idx: sprite_positions[idx])
    ordered_sprites = [sprites[idx] for idx in order]

    visible_overlap_pairs = []
    disjoint_pairs = []
    fallback_pairs = []
    for upper_idx in range(sprite_count):
        for lower_idx in range(sprite_count):
            if upper_idx == lower_idx:
                continue
            kind = _gen_overlay_pair_kind(ordered_sprites[upper_idx], ordered_sprites[lower_idx])
            if kind == "visible_overlap":
                visible_overlap_pairs.append((upper_idx, lower_idx))
            if kind == "disjoint":
                disjoint_pairs.append((upper_idx, lower_idx))
            if kind != "same_color_overlap":
                fallback_pairs.append((upper_idx, lower_idx))
    if not fallback_pairs:
        return None

    want_overlap = _gen_overlap_mode()
    if want_overlap and visible_overlap_pairs:
        upper_rank, lower_rank = _gen_choice(visible_overlap_pairs)
    elif not want_overlap and disjoint_pairs:
        upper_rank, lower_rank = _gen_choice(disjoint_pairs)
    elif visible_overlap_pairs:
        upper_rank, lower_rank = _gen_choice(visible_overlap_pairs)
    else:
        upper_rank, lower_rank = _gen_choice(fallback_pairs)

    max_count = max(upper_rank + 1, lower_rank + 1)
    frame_shapes = [_gen_frame_shape(max_count), _gen_frame_shape(max_count)]
    frame_positions = _gen_place_boxes(reserved, frame_shapes, pad=_GEN_BOX_PADDING, border=1)
    if frame_positions is None:
        return None

    frame_boxes = [
        (r, c, r + h - 1, c + w - 1)
        for (r, c), (h, w) in zip(frame_positions, frame_shapes)
    ]
    frame_boxes = sorted(frame_boxes, key=_reading_key)
    frame_counts = [upper_rank + 1, lower_rank + 1]

    grid = np.full((rows, cols), _GEN_BG, dtype=int)
    for sprite, (r, c) in zip(sprites, sprite_positions):
        paste(grid, sprite, r, c, transparent=_GEN_BG)

    for box, count in zip(frame_boxes, frame_counts):
        _gen_render_frame(grid, box, frame_color)
        r0, c0, r1, c1 = box
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        if len(interior) < count:
            return None
        chosen = _GEN_RNG.choice(len(interior), size=count, replace=False)
        for idx in np.atleast_1d(chosen):
            rr, cc = interior[int(idx)]
            grid[rr, cc] = marker_color

    outside_candidates = []
    frame_full_mask = np.zeros(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        frame_full_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != _GEN_BG:
                continue
            if frame_full_mask[r, c]:
                continue
            outside_candidates.append((r, c))
    outside_count = _gen_randint(*_GEN_OUTSIDE_MARKER_RANGE)
    if len(outside_candidates) < outside_count:
        return None
    chosen_outside = _GEN_RNG.choice(len(outside_candidates), size=outside_count, replace=False)
    for idx in np.atleast_1d(chosen_outside):
        rr, cc = outside_candidates[int(idx)]
        grid[rr, cc] = marker_color

    expected = _gen_expected_overlay(ordered_sprites[upper_rank], ordered_sprites[lower_rank])
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
