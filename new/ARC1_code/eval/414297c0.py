import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location
import random


def _colors_at(g: np.ndarray, coords: np.ndarray) -> set[int]:
    return set(map(int, g[coords[:, 0], coords[:, 1]]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the large filled rectangle and stamp each external red sprite onto the matching marker inside it.

    Input structure
    - The background is the dominant color.
    - One large 8-connected foreground component is a filled rectangle whose fill color contains isolated marker pixels.
    - Each smaller 8-connected component is a sprite made from a shared stamp color plus exactly one marker color.

    Rule
    - Use the largest foreground component as the target rectangle and crop to its bounding box.
    - Infer the stamp color as the only color common to all smaller source sprites.
    - For every marker pixel inside the target rectangle, find the source sprite with the same marker color.
    - Copy that sprite's stamp-color offsets relative to its marker onto the cropped target, preserving the target marker.

    Key insight
    - The isolated marker color is a key: it tells which off-rectangle sprite mask should be aligned to that same-colored marker in the filled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert len(components) >= 2

    components = sorted(components, key=lambda comp: comp["area"], reverse=True)
    assert components[0]["area"] > components[1]["area"]
    target = components[0]
    source_components = components[1:]

    out = crop_bbox(g, target["bbox"])
    assert np.all(out != bg)

    colors, counts = np.unique(out, return_counts=True)
    assert len(colors) >= 2
    count_order = np.argsort(counts)[::-1]
    assert counts[count_order[0]] > counts[count_order[1]]
    fill_color = int(colors[count_order[0]])

    target_markers: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(out != fill_color):
        marker = int(out[r, c])
        assert marker not in target_markers
        target_markers[marker] = (int(r), int(c))
    assert target_markers

    source_color_sets = [_colors_at(g, comp["coords"]) for comp in source_components]
    common_source_colors = set.intersection(*source_color_sets)
    assert len(common_source_colors) == 1
    stamp_color = common_source_colors.pop()
    assert stamp_color not in target_markers

    templates: dict[int, list[tuple[int, int]]] = {}
    for component in source_components:
        coords = component["coords"]
        colors_in_component = _colors_at(g, coords)
        assert stamp_color in colors_in_component

        marker_colors = colors_in_component - {stamp_color}
        assert len(marker_colors) == 1
        marker_color = marker_colors.pop()
        assert marker_color not in templates

        marker_coords = coords[g[coords[:, 0], coords[:, 1]] == marker_color]
        assert len(marker_coords) == 1
        marker_r, marker_c = map(int, marker_coords[0])

        stamp_offsets: list[tuple[int, int]] = []
        stamp_coords = coords[g[coords[:, 0], coords[:, 1]] == stamp_color]
        assert len(stamp_coords) > 0
        for r, c in stamp_coords:
            stamp_offsets.append((int(r) - marker_r, int(c) - marker_c))
        templates[marker_color] = stamp_offsets

    H, W = out.shape
    for marker_color, (marker_r, marker_c) in target_markers.items():
        assert marker_color in templates
        for dr, dc in templates[marker_color]:
            r = marker_r + dr
            c = marker_c + dc
            assert 0 <= r < H and 0 <= c < W
            assert out[r, c] == fill_color
            out[r, c] = stamp_color
        out[marker_r, marker_c] = marker_color

    return out

# --------------- generator (o5x) ---------------

Pattern = tuple[tuple[int, int], ...]

# Color palette: red is the shared stamp color; all other non-bg colors are sampled.
BG_COLOR = 0
STAMP_COLOR = 2
COLOR_POOL = (1, 3, 4, 5, 6, 7, 8, 9)

# Target rectangle and outer canvas bounds are inclusive.
RECT_H_RANGE = (6, 12)
RECT_W_RANGE = (6, 15)
GRID_H_RANGE = (16, 26)
GRID_W_RANGE = (18, 24)
GRID_MIN_RECT_AREA_FACTOR = 2

# Marker and sprite sampling controls.
MARKER_COUNT_RANGE = (2, 5)
DISTRACTOR_PROB = 0.5
PATTERN_CELL_RANGE = (3, 7)
SPRITE_SHAPE = (3, 3)
SPRITE_CENTER = (1, 1)
SPRITE_OFFSETS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)

# Placement and retry controls.
MARKER_EDGE_MARGIN = 1
COMPONENT_BORDER = 1
COMPONENT_PADDING = 1
COMPONENT_PADDING_CONNECTIVITY = 8
FREE_LOCATION_KWARGS = dict(transparent=BG_COLOR, border=COMPONENT_BORDER, padding=COMPONENT_PADDING, padding_connectivity=COMPONENT_PADDING_CONNECTIVITY)
MAX_GENERATION_ATTEMPTS = 300


def _pattern_reaches_all_sides(pattern: Pattern) -> bool:
    return (
        any(dr == -1 for dr, _ in pattern)
        and any(dr == 1 for dr, _ in pattern)
        and any(dc == -1 for _, dc in pattern)
        and any(dc == 1 for _, dc in pattern)
    )


def _gen_pattern_pool() -> list[Pattern]:
    """Return shuffled valid 3x3 stamp masks."""
    pool = []
    min_cells, max_cells = PATTERN_CELL_RANGE
    for bits in range(1, 1 << len(SPRITE_OFFSETS)):
        pattern = tuple(SPRITE_OFFSETS[i] for i in range(len(SPRITE_OFFSETS)) if bits & (1 << i))
        if min_cells <= len(pattern) <= max_cells and _pattern_reaches_all_sides(pattern):
            pool.append(pattern)
    random.shuffle(pool)
    return pool


def _gen_sprite_patch(marker_color: int, pattern: Pattern) -> np.ndarray:
    patch = np.zeros(SPRITE_SHAPE, dtype=int)
    center_r, center_c = SPRITE_CENTER
    patch[center_r, center_c] = int(marker_color)
    rows = [center_r + dr for dr, _ in pattern]
    cols = [center_c + dc for _, dc in pattern]
    patch[rows, cols] = STAMP_COLOR
    return patch


def _gen_marker_positions(rect_h: int, rect_w: int, patterns: list[Pattern]) -> list[tuple[int, int]] | None:
    """Backtrack marker centers so all eventual stamps stay disjoint."""
    occupied = np.zeros((rect_h, rect_w), dtype=bool)
    centers = np.zeros((rect_h, rect_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patterns)
    order = list(range(len(patterns)))
    order.sort(key=lambda i: (-len(patterns[i]), random.random()))

    def search(j: int) -> bool:
        if j == len(order):
            return True

        i = order[j]
        candidates = []
        for r in range(MARKER_EDGE_MARGIN, rect_h - MARKER_EDGE_MARGIN):
            for c in range(MARKER_EDGE_MARGIN, rect_w - MARKER_EDGE_MARGIN):
                if centers[r - 1 : r + 2, c - 1 : c + 2].any():
                    continue
                cells = [(r, c)] + [(r + dr, c + dc) for dr, dc in patterns[i]]
                if all(not occupied[rr, cc] for rr, cc in cells):
                    candidates.append((r, c, cells))
        random.shuffle(candidates)

        for r, c, cells in candidates:
            for rr, cc in cells:
                occupied[rr, cc] = True
            centers[r, c] = True
            positions[i] = (r, c)
            if search(j + 1):
                return True
            positions[i] = None
            centers[r, c] = False
            for rr, cc in cells:
                occupied[rr, cc] = False
        return False

    if not search(0):
        return None
    return [pos for pos in positions if pos is not None]


def _gen_choose_grid_shape(
    rect_h: int, rect_w: int, marker_count: int, source_area: int
) -> tuple[int, int] | None:
    rect_area = rect_h * rect_w
    valid = []
    min_area = GRID_MIN_RECT_AREA_FACTOR * rect_area + source_area - marker_count
    for h in range(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1):
        for w in range(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1):
            if h < rect_h + 2 * COMPONENT_BORDER or w < rect_w + 2 * COMPONENT_BORDER:
                continue
            if h * w <= min_area:
                continue
            valid.append((h, w))
    if not valid:
        return None
    return random.choice(valid)


def _render_target(
    rect_h: int,
    rect_w: int,
    fill_color: int,
    marker_colors: list[int],
    marker_positions: list[tuple[int, int]],
    patterns: list[Pattern],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the clean input rectangle and expected stamped output crop."""
    clean_rect = np.full((rect_h, rect_w), fill_color, dtype=int)
    out = clean_rect.copy()
    for marker_color, (r, c), pattern in zip(marker_colors, marker_positions, patterns):
        clean_rect[r, c] = int(marker_color)
        out[r, c] = int(marker_color)
        rows = [r + dr for dr, _ in pattern]
        cols = [c + dc for _, dc in pattern]
        if np.any(out[rows, cols] != fill_color):
            return None
        out[rows, cols] = STAMP_COLOR
        out[r, c] = int(marker_color)
    return clean_rect, out


def _gen_attempt() -> np.ndarray | None:
    fill_color = random.choice(COLOR_POOL)
    marker_pool = [c for c in COLOR_POOL if c != fill_color]
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    has_distractor = random.random() < DISTRACTOR_PROB
    source_count = marker_count + int(has_distractor)
    if len(marker_pool) < source_count:
        return None

    source_marker_colors = random.sample(marker_pool, source_count)
    marker_colors = source_marker_colors[:marker_count]
    distractor_color = source_marker_colors[-1] if has_distractor else None

    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)

    pool = _gen_pattern_pool()
    if len(pool) < source_count:
        return None
    source_patterns = pool[:source_count]
    patterns = list(source_patterns[:marker_count])

    marker_positions = _gen_marker_positions(rect_h, rect_w, patterns)
    if marker_positions is None:
        return None

    rendered = _render_target(rect_h, rect_w, fill_color, marker_colors, marker_positions, patterns)
    if rendered is None:
        return None
    clean_rect, out = rendered

    source_area = sum(1 + len(pattern) for pattern in source_patterns)
    grid_shape = _gen_choose_grid_shape(rect_h, rect_w, marker_count, source_area)
    if grid_shape is None:
        return None

    inp = np.zeros(grid_shape, dtype=int)
    try:
        rr, cc = random_free_location(inp, clean_rect, **FREE_LOCATION_KWARGS)
        paste(inp, clean_rect, rr, cc, transparent=BG_COLOR)

        source_items = [(marker_color, pattern) for marker_color, pattern in zip(marker_colors, source_patterns[:marker_count])]
        if has_distractor and distractor_color is not None:
            source_items.append((distractor_color, source_patterns[-1]))
        random.shuffle(source_items)

        for marker_color, pattern in source_items:
            sprite = _gen_sprite_patch(marker_color, pattern)
            sr, sc = random_free_location(inp, sprite, **FREE_LOCATION_KWARGS)
            paste(inp, sprite, sr, sc, transparent=BG_COLOR)
    except ValueError:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
