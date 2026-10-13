from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]
SIDE_PATTERNS = (
    frozenset({"top", "left"}),
    frozenset({"top", "right"}),
    frozenset({"bottom", "left"}),
    frozenset({"bottom", "right"}),
    frozenset({"top", "bottom"}),
    frozenset({"left", "right"}),
)
PALETTE = tuple(range(10))
MAX_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, bbox)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _intersection(a: BBox, b: BBox) -> BBox | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _control_specs(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    counts = Counter(map(int, g.ravel()))
    specs: list[dict[str, object]] = []
    for color, count in counts.items():
        if color == bg or count != 4:
            continue
        coords = np.argwhere(g == int(color))
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (
            int(np.sum(rows == rows.min())) != 1
            or int(np.sum(rows == rows.max())) != 1
            or int(np.sum(cols == cols.min())) != 1
            or int(np.sum(cols == cols.max())) != 1
        ):
            continue
        frame = (int(rows.min()) + 1, int(cols.min()) + 1, int(rows.max()) - 1, int(cols.max()) - 1)
        if frame[0] > frame[2] or frame[1] > frame[3]:
            continue
        specs.append({"color": int(color), "coords": coords, "frame": frame})
    specs.sort(key=lambda spec: int(spec["color"]))
    return bg, specs


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the overlap of two implied inset rectangles defined by the only two four-occurrence colors.

    Input structure:
    - A dominant background surrounds several colored pixels and small distractor objects.
    - Exactly two colors appear exactly four times, and within each such color the four pixels have unique
      top, bottom, left, and right extrema.
    - Each four-pixel color acts as control markers for an unpainted rectangle one cell inside those extrema.

    Rule:
    - Detect the two colors whose total occurrence count is exactly four.
    - For each one, take the row and column extrema of its four pixels and move every side inward by one cell
      to obtain an implied rectangle, including its border.
    - Intersect the two implied rectangles and return the input grid cropped to that overlap region.

    Key insight:
    - The four-pixel colors are not scene objects; they are hidden frame guides, and the answer comes from
      combining both hidden frames rather than using either one alone.
    """
    g = np.asarray(grid, dtype=int)
    _bg, specs = _control_specs(g)
    assert len(specs) == 2, "expected exactly two four-occurrence control colors"

    frame0 = specs[0]["frame"]
    frame1 = specs[1]["frame"]
    overlap = _intersection(frame0, frame1)
    assert overlap is not None, "control frames must overlap"
    return crop_bbox(g, overlap)


def _marker_positions(rng: random.Random, frame: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, frame)
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3
    return [
        (r0 - 1, rng.randint(c0 + 1, c1 - 1)),
        (r1 + 1, rng.randint(c0 + 1, c1 - 1)),
        (rng.randint(r0 + 1, r1 - 1), c0 - 1),
        (rng.randint(r0 + 1, r1 - 1), c1 + 1),
    ]


def _small_mask_library() -> list[np.ndarray]:
    raw = [
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
        np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    ]
    return raw


SHAPE_LIBRARY = _small_mask_library()


def _paint_scene(
    rng: random.Random,
    grid: np.ndarray,
    union_box: BBox,
    inner_box: BBox,
    scene_colors: list[int],
) -> None:
    ur0, uc0, ur1, uc1 = map(int, union_box)
    ir0, ic0, ir1, ic1 = map(int, inner_box)
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1

    h_thickness = 1 + int(inner_h >= 6 and rng.random() < 0.5)
    v_thickness = 1 + int(inner_w >= 6 and rng.random() < 0.5)

    h_start = rng.randint(ir0 + 1, ir1 - h_thickness)
    v_start = rng.randint(ic0 + 1, ic1 - v_thickness)
    draw_vertical_first = bool(rng.randint(0, 1))

    if draw_vertical_first:
        grid[ur0 : ur1 + 1, v_start : v_start + v_thickness] = scene_colors[1]
        grid[h_start : h_start + h_thickness, uc0 : uc1 + 1] = scene_colors[0]
    else:
        grid[h_start : h_start + h_thickness, uc0 : uc1 + 1] = scene_colors[0]
        grid[ur0 : ur1 + 1, v_start : v_start + v_thickness] = scene_colors[1]

    shape = rng.choice(SHAPE_LIBRARY)
    sh, sw = shape.shape
    sr0 = rng.randint(ir0, ir1 - sh + 1)
    sc0 = rng.randint(ic0, ic1 - sw + 1)
    patch = np.full(shape.shape, -1, dtype=int)
    patch[shape] = scene_colors[2]
    patch[~shape] = -1
    mask = patch != -1
    sub = grid[sr0 : sr0 + sh, sc0 : sc0 + sw]
    sub[mask] = patch[mask]

    if len(scene_colors) >= 4:
        anchor_choices = [
            (ir0, ic0),
            (ir0, ic1 - 1),
            (ir1 - 1, ic0),
            (ir1 - 1, ic1 - 1),
        ]
        ar0, ac0 = rng.choice(anchor_choices)
        grid[ar0 : ar0 + 2, ac0 : ac0 + 2] = scene_colors[3]
        cut_r = ar0 + rng.randint(0, 1)
        cut_c = ac0 + rng.randint(0, 1)
        grid[cut_r, cut_c] = scene_colors[rng.randint(0, 2)]


def _sample_free_cells(
    rng: random.Random,
    shape: tuple[int, int],
    forbidden: np.ndarray,
    count: int,
) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if not forbidden[r, c]]
    rng.shuffle(candidates)
    return candidates[:count]


def _add_distractors(
    rng: random.Random,
    grid: np.ndarray,
    bg: int,
    forbidden: np.ndarray,
    available_colors: list[int],
) -> None:
    n_groups = rng.randint(0, min(2, len(available_colors)))
    for color in available_colors[:n_groups]:
        count = rng.choice((1, 2, 3))
        cells = _sample_free_cells(rng, grid.shape, forbidden | (grid != bg), count)
        if len(cells) < count:
            continue
        for r, c in cells:
            grid[r, c] = color


def _describe_from_boxes(
    a_box: BBox,
    b_box: BBox,
) -> tuple[BBox, BBox, BBox]:
    overlap = _intersection(a_box, b_box)
    assert overlap is not None
    return a_box, b_box, overlap


def _semantic_ok(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg, specs = _control_specs(g)
    if len(specs) != 2:
        return False

    counts = Counter(map(int, g.ravel()))
    control_colors = {int(spec["color"]) for spec in specs}
    if {color for color, count in counts.items() if count == 4} != control_colors:
        return False

    a_box = specs[0]["frame"]
    b_box = specs[1]["frame"]
    overlap = _intersection(a_box, b_box)
    if overlap is None:
        return False
    if a_box == overlap or b_box == overlap:
        return False

    sides_from_a = sum(
        [
            int(a_box[0] == overlap[0]),
            int(a_box[1] == overlap[1]),
            int(a_box[2] == overlap[2]),
            int(a_box[3] == overlap[3]),
        ]
    )
    sides_from_b = sum(
        [
            int(b_box[0] == overlap[0]),
            int(b_box[1] == overlap[1]),
            int(b_box[2] == overlap[2]),
            int(b_box[3] == overlap[3]),
        ]
    )
    if sides_from_a == 0 or sides_from_b == 0:
        return False

    out = main(g)
    if not np.array_equal(out, crop_bbox(g, overlap)):
        return False
    if np.isin(out, list(control_colors)).any():
        return False

    a_mask = _bbox_mask(g.shape, a_box)
    b_mask = _bbox_mask(g.shape, b_box)
    overlap_mask = _bbox_mask(g.shape, overlap)
    scene_mask = (g != bg) & ~np.isin(g, list(control_colors))

    if not scene_mask[overlap_mask].any():
        return False
    if len(set(map(int, out.ravel()))) < 3:
        return False
    if not (scene_mask & a_mask & ~b_mask).any():
        return False
    if not (scene_mask & b_mask & ~a_mask).any():
        return False
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False
    return True


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(PALETTE)
    pool = [color for color in PALETTE if color != bg]
    rng.shuffle(pool)
    control_colors = pool[:2]
    scene_colors = pool[2:6]
    noise_colors = pool[6:]

    inner_h = rng.randint(5, 10)
    inner_w = rng.randint(5, 10)
    expand_top = rng.randint(1, 4)
    expand_bottom = rng.randint(1, 4)
    expand_left = rng.randint(1, 4)
    expand_right = rng.randint(1, 4)

    union_h = inner_h + expand_top + expand_bottom
    union_w = inner_w + expand_left + expand_right
    min_h = union_h + 2
    min_w = union_w + 2
    if min_h > 30 or min_w > 30:
        return None

    extra_h = rng.randint(0, 30 - min_h)
    extra_w = rng.randint(0, 30 - min_w)
    top_pad = rng.randint(0, extra_h)
    bottom_pad = extra_h - top_pad
    left_pad = rng.randint(0, extra_w)
    right_pad = extra_w - left_pad

    H = min_h + extra_h
    W = min_w + extra_w
    if H > 30 or W > 30:
        return None

    union_r0 = 1 + top_pad
    union_c0 = 1 + left_pad
    union_r1 = union_r0 + union_h - 1
    union_c1 = union_c0 + union_w - 1
    inner_r0 = union_r0 + expand_top
    inner_c0 = union_c0 + expand_left
    inner_r1 = inner_r0 + inner_h - 1
    inner_c1 = inner_c0 + inner_w - 1

    a_tight = set(rng.choice(SIDE_PATTERNS))
    a_box = (
        inner_r0 if "top" in a_tight else union_r0,
        inner_c0 if "left" in a_tight else union_c0,
        inner_r1 if "bottom" in a_tight else union_r1,
        inner_c1 if "right" in a_tight else union_c1,
    )
    b_box = (
        union_r0 if "top" in a_tight else inner_r0,
        union_c0 if "left" in a_tight else inner_c0,
        union_r1 if "bottom" in a_tight else inner_r1,
        union_c1 if "right" in a_tight else inner_c1,
    )
    a_box, b_box, overlap = _describe_from_boxes(a_box, b_box)
    if overlap != (inner_r0, inner_c0, inner_r1, inner_c1):
        return None

    a_markers = _marker_positions(rng, a_box)
    b_markers = _marker_positions(rng, b_box)
    marker_cells = a_markers + b_markers
    if len(set(marker_cells)) != 8:
        return None

    grid = np.full((H, W), bg, dtype=int)
    _paint_scene(rng, grid, (union_r0, union_c0, union_r1, union_c1), overlap, scene_colors)

    forbidden = _bbox_mask(grid.shape, (union_r0, union_c0, union_r1, union_c1))
    for r, c in marker_cells:
        forbidden[r, c] = True
    _add_distractors(rng, grid, bg, forbidden, noise_colors)

    for r, c in a_markers:
        grid[r, c] = control_colors[0]
    for r, c in b_markers:
        grid[r, c] = control_colors[1]

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
