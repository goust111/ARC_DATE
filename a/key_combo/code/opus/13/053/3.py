from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SPRITE_COLORS = [BLUE, RED, GREEN, YELLOW, ORANGE, TEAL, MAROON]
RNG = random.Random()


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = Counter(map(int, values))
    return int(counts.most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def find_components(
    grid: np.ndarray,
    *,
    colors: list[int] | set[int] | tuple[int, ...] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if colors is None:
        color_list = sorted(set(map(int, np.unique(g))))
    else:
        color_list = sorted(set(map(int, colors)))
    dirs = DIR4 if int(connectivity) == 4 else [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    comps: list[dict] = []
    for color in color_list:
        visited = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if visited[r, c] or int(g[r, c]) != int(color):
                    continue
                q = deque([(r, c)])
                visited[r, c] = True
                cells: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in dirs:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == int(color):
                            visited[nr, nc] = True
                            q.append((nr, nc))
                coords = np.array(cells, dtype=int)
                comps.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
    return comps


def normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def interior_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = map(int, box)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def bbox_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = map(int, box)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def is_rectangular_frame(component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected_area = 2 * (h + w) - 4
    if int(component["area"]) != expected_area:
        return False
    local = np.zeros((h, w), dtype=bool)
    coords = component["coords"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def region_masks_from_frames(
    shape: tuple[int, int],
    frame_boxes: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    outer, middle, inner = frame_boxes
    region_1 = interior_mask(shape, outer) & ~bbox_mask(shape, middle)
    region_2 = interior_mask(shape, middle) & ~bbox_mask(shape, inner)
    region_3 = interior_mask(shape, inner)
    return [region_1, region_2, region_3]


def extract_scene(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    grey_components = find_components(g, colors=[GREY], connectivity=4)
    frames = [comp for comp in grey_components if is_rectangular_frame(comp)]
    assert len(frames) == 3, "expected exactly three grey rectangular frames"

    frame_boxes = [tuple(map(int, comp["bbox"])) for comp in frames]
    frame_boxes.sort(key=lambda box: (-(box[2] - box[0] + 1) * (box[3] - box[1] + 1), box[0], box[1]))
    for outer, inner in zip(frame_boxes, frame_boxes[1:]):
        assert outer[0] < inner[0] < inner[2] < outer[2]
        assert outer[1] < inner[1] < inner[3] < outer[3]

    regions = region_masks_from_frames(g.shape, frame_boxes)

    marker_components = find_components(g, colors=[PINK], connectivity=4)
    assert len(marker_components) == 3, "expected exactly three magenta markers"
    region_markers: list[dict] = []
    used = set()
    for idx, region in enumerate(regions):
        matches = []
        for comp_id, comp in enumerate(marker_components):
            coords = comp["coords"]
            if all(bool(region[int(r), int(c)]) for r, c in coords):
                matches.append((comp_id, comp))
        assert len(matches) == 1, f"region {idx + 1} must contain exactly one marker"
        used.add(matches[0][0])
        region_markers.append(matches[0][1])
        region_values = g[region]
        assert set(map(int, np.unique(region_values))).issubset({bg, PINK}), "frame bands may only contain bg and marker pixels"
    assert len(used) == len(marker_components)

    sprite_colors = [color for color in np.unique(g) if int(color) not in {bg, GREY, PINK}]
    sprite_components = []
    for comp in find_components(g, colors=sprite_colors, connectivity=4):
        sprite_components.append(comp)
    assert 3 <= len(sprite_components) <= 4, "expected three or four colored sprites"

    return frame_boxes, regions, region_markers, sprite_components


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each nested frame band with the color of the corner sprite whose shape matches that band's magenta marker.

    Input structure:
    - The background is black, with three nested one-pixel-thick grey hollow rectangular frames near the center.
    - The band between each adjacent frame pair contains one magenta marker component, and the innermost frame interior contains one more magenta marker.
    - The upper-left or upper-right corner holds three or four separated colored sprites whose normalized shapes are compared against the markers.

    Rule:
    - Detect the three grey frames and sort them from outermost to innermost.
    - For each frame band, crop its magenta marker to a normalized binary mask and find the unique corner sprite with the same normalized mask.
    - Use the matched sprite's color to fill that entire band: outer band first, then middle band, then the innermost frame interior, while keeping all grey frame pixels unchanged.
    - The magenta marker pixels are overwritten by the fill, and the corner sprites remain unchanged.

    Key insight:
    - The markers are not palette labels; they are shape queries whose matched sprite colors become the outer-to-inner fill colors for the nested frame regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_boxes, regions, region_markers, sprite_components = extract_scene(g)

    sprite_by_shape: defaultdict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for sprite in sprite_components:
        sprite_by_shape[normalized_shape(sprite["coords"])].append(sprite)

    out = g.copy()
    for region, marker in zip(regions, region_markers):
        shape_key = normalized_shape(marker["coords"])
        matches = sprite_by_shape.get(shape_key, [])
        assert len(matches) == 1, "each marker must match exactly one sprite shape"
        color = int(matches[0]["color"])
        out[region] = color

    for box in frame_boxes:
        r0, c0, r1, c1 = box
        frame = g[r0 : r1 + 1, c0 : c1 + 1] == GREY
        out[r0 : r1 + 1, c0 : c1 + 1][frame] = GREY
    return out


SHAPES: list[tuple[tuple[int, int], ...]] = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
]


def shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    return int(h), int(w)


def shape_area(shape: tuple[tuple[int, int], ...]) -> int:
    return int(len(shape))


def paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for dr, dc in shape:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def candidate_positions(region: np.ndarray, shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    h, w = region.shape
    sh, sw = shape_bbox(shape)
    choices: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if all(bool(region[r0 + dr, c0 + dc]) for dr, dc in shape):
                choices.append((r0, c0))
    return choices


def choose_shapes(sprite_count: int) -> tuple[list[tuple[tuple[int, int], ...]], tuple[tuple[int, int], ...] | None]:
    by_area: defaultdict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    for shape in SHAPES:
        by_area[shape_area(shape)].append(shape)

    repeated_area = RNG.choice([area for area, items in by_area.items() if len(items) >= 2])
    pair = RNG.sample(by_area[repeated_area], 2)
    remaining = [shape for shape in SHAPES if shape not in pair]
    third = RNG.choice(remaining)
    matched = pair + [third]
    RNG.shuffle(matched)

    distractor = None
    if sprite_count == 4:
        preferred = [shape for shape in SHAPES if shape not in matched and shape_area(shape) in {shape_area(s) for s in matched}]
        pool = preferred if preferred else [shape for shape in SHAPES if shape not in matched]
        distractor = RNG.choice(pool)
    return matched, distractor


def choose_sprite_layout(grid_w: int, corner_right: bool, sprite_count: int) -> list[tuple[int, int]]:
    base_c = grid_w - 8 if corner_right else 0
    slots = [(0, base_c), (0, base_c + 5), (5, base_c), (5, base_c + 5)]
    chosen = RNG.sample(slots, sprite_count)
    RNG.shuffle(chosen)
    return chosen


def sprite_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    sh, sw = shape_bbox(shape)
    r_off = RNG.randint(0, 3 - sh)
    c_off = RNG.randint(0, 3 - sw)
    for dr, dc in shape:
        patch[r_off + dr, c_off + dc] = int(color)
    return patch


def expected_output_from_construction(
    inp: np.ndarray,
    frame_boxes: list[tuple[int, int, int, int]],
    layer_colors: list[int],
) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for region, color in zip(region_masks_from_frames(inp.shape, frame_boxes), layer_colors):
        out[region] = int(color)
    for box in frame_boxes:
        r0, c0, r1, c1 = box
        frame = inp[r0 : r1 + 1, c0 : c1 + 1] == GREY
        out[r0 : r1 + 1, c0 : c1 + 1][frame] = GREY
    return out


def build_instance() -> tuple[np.ndarray, np.ndarray]:
    inner_side = RNG.choice([5, 7])
    middle_side = inner_side + 4
    outer_side = inner_side + 8

    top_margin = RNG.choice([9, 10])
    bottom_margin = RNG.randint(1, 3)
    extra_cols = RNG.randint(0, 4)
    grid_h = top_margin + outer_side + bottom_margin
    grid_w = outer_side + extra_cols
    grid = np.zeros((grid_h, grid_w), dtype=int)

    outer_left = RNG.randint(0, extra_cols)
    outer_top = top_margin
    frame_boxes = [
        (outer_top, outer_left, outer_top + outer_side - 1, outer_left + outer_side - 1),
        (outer_top + 2, outer_left + 2, outer_top + 2 + middle_side - 1, outer_left + 2 + middle_side - 1),
        (outer_top + 4, outer_left + 4, outer_top + 4 + inner_side - 1, outer_left + 4 + inner_side - 1),
    ]
    for box in frame_boxes:
        draw_frame(grid, box, GREY)

    regions = region_masks_from_frames(grid.shape, frame_boxes)
    sprite_count = RNG.choice([3, 4])
    matched_shapes, distractor = choose_shapes(sprite_count)
    layer_shapes = matched_shapes[:]

    for region, shape in zip(regions, layer_shapes):
        choices = candidate_positions(region, shape)
        assert choices, "chosen marker shape must fit its region"
        r0, c0 = RNG.choice(choices)
        paint_shape(grid, shape, r0, c0, PINK)

    sprite_shapes = layer_shapes[:] + ([distractor] if distractor is not None else [])
    sprite_colors = RNG.sample(SPRITE_COLORS, len(sprite_shapes))
    layer_colors = sprite_colors[:3]
    corner_right = bool(RNG.randint(0, 1))
    sprite_slots = choose_sprite_layout(grid_w, corner_right, len(sprite_shapes))

    order = list(range(len(sprite_shapes)))
    RNG.shuffle(order)
    shape_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    for sprite_index, slot_index in enumerate(order):
        shape = sprite_shapes[sprite_index]
        color = sprite_colors[sprite_index]
        patch = sprite_patch(shape, color)
        sr, sc = sprite_slots[slot_index]
        paste(grid, patch, sr, sc, BLACK)
        shape_to_color[shape] = int(color)

    expected = expected_output_from_construction(
        grid,
        frame_boxes,
        [shape_to_color[shape] for shape in layer_shapes],
    )

    matched_areas = [shape_area(shape) for shape in layer_shapes]
    counts = Counter(matched_areas)
    assert max(counts.values()) >= 2, "generation must include same-area but different-shape markers"
    if distractor is not None:
        assert distractor not in layer_shapes
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp, expected = build_instance()
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(inp, expected):
            continue
        _, _, region_markers, sprite_components = extract_scene(inp)
        marker_keys = [normalized_shape(comp["coords"]) for comp in region_markers]
        sprite_keys = [normalized_shape(comp["coords"]) for comp in sprite_components]
        if any(sprite_keys.count(key) != 1 for key in marker_keys):
            continue
        return inp
    raise RuntimeError("failed to generate valid nested-frame shape-match input")
