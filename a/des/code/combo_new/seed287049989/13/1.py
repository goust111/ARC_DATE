import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
FRAME_COUNT_CHOICES = (2, 3)
PATTERN_LIBRARY = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool),
)


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _parse_color_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    top = grid[-2]
    bottom = grid[-1]
    mapping: dict[int, int] = {}
    for c in range(grid.shape[1]):
        marker = int(top[c])
        fill = int(bottom[c])
        if marker == bg and fill == bg:
            continue
        assert marker != bg and fill != bg, "legend entries must be aligned colored pixels"
        if marker in mapping:
            assert mapping[marker] == fill, "marker color legend must be unambiguous"
        mapping[marker] = fill
    assert mapping, "bottom legend must contain at least one mapping"
    return mapping


def _parse_pattern_legend(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    strip = grid[:-2, -3:]
    mapping: dict[int, np.ndarray] = {}
    r = 0
    while r <= strip.shape[0] - 3:
        if int(strip[r, 1]) == bg:
            r += 1
            continue

        if r > 0:
            assert not np.any(strip[r - 1] != bg), "pattern legend blocks must be separated by blank rows"

        block = strip[r : r + 3]
        key = int(block[0, 1])
        assert key != bg and key != GREY, "pattern key must be a border color"
        special = np.argwhere((block != bg) & (block != GREY))
        assert special.shape[0] == 1 and tuple(map(int, special[0])) == (0, 1), (
            "only the top-middle key pixel may use a non-bg non-grey legend color"
        )

        mask = block == GREY
        # The keyed pixel names the border color and also occupies one tile cell.
        mask[0, 1] = True
        mapping[key] = mask
        r += 4

    assert mapping, "right legend must contain at least one pattern"
    return mapping


def _frame_bbox(area: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(area == color)
    if coords.shape[0] < 16:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    patch = area[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not np.all(patch[border] == color):
        return None
    if np.any(patch[~border] == color):
        return None
    if int(coords.shape[0]) != int(border.sum()):
        return None
    return r0, c0, r1, c1


def _find_frames(area: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(area)):
        if color == bg:
            continue
        bbox = _frame_bbox(area, color)
        if bbox is not None:
            frames.append((color, bbox))
    frames.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    assert frames, "main area must contain at least one rectangular frame"
    return frames


def _tile_mask(mask: np.ndarray, fill_color: int, shape: tuple[int, int], bg: int) -> np.ndarray:
    tile = np.full(mask.shape, bg, dtype=int)
    tile[mask] = fill_color
    rep_r = (shape[0] + mask.shape[0] - 1) // mask.shape[0]
    rep_c = (shape[1] + mask.shape[1] - 1) // mask.shape[1]
    return np.tile(tile, (rep_r, rep_c))[: shape[0], : shape[1]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed room with a tiled pattern chosen by separate color and pattern legends.

    Input structure:
    - Black background with several single-pixel-thick rectangular frames of distinct border colors.
    - Each frame contains exactly one interior marker pixel whose color differs from the frame and the background.
    - The bottom two rows form a color legend: upper colored pixels are marker colors and the aligned lower pixels are fill colors.
    - The rightmost three columns form stacked 3x3 pattern legends separated by blank rows; each block's top-middle key names a border color and its grey cells define a binary tile.

    Rule:
    - Read the bottom legend as a marker-color -> fill-color mapping.
    - Read the right legend as a border-color -> 3x3 tile mapping, treating the keyed top-middle cell as the tile's corresponding active position.
    - For each rectangular frame, use its marker color to choose the fill color and its border color to choose the tile.
    - Remove the original marker and tile the frame interior from its top-left corner, replacing active tile cells with the mapped fill color and inactive tile cells with background.

    Key insight:
    Each room is solved only by composing two independent legends: the marker chooses the paint color, while the border chooses the repeating stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 7 and w >= 8, "grid is too small for the required legends and frames"

    bg = _background_color(g)
    color_map = _parse_color_legend(g, bg)
    pattern_map = _parse_pattern_legend(g, bg)
    main_area = g[:-2, :-3]
    frames = _find_frames(main_area, bg)

    out = g.copy()
    for border_color, (r0, c0, r1, c1) in frames:
        assert border_color in pattern_map, "every border color must have a pattern legend"
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_cells = np.argwhere(inner != bg)
        assert marker_cells.shape[0] == 1, "each frame interior must contain exactly one marker pixel"
        mr, mc = map(int, marker_cells[0])
        marker_color = int(inner[mr, mc])
        assert marker_color != border_color, "marker color must differ from frame color"
        assert marker_color in color_map, "every marker color must appear in the bottom legend"

        fill_color = color_map[marker_color]
        filled = _tile_mask(pattern_map[border_color], fill_color, inner.shape, bg)
        out[r0 + 1 : r1, c0 + 1 : c1] = filled

    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _draw_pattern_legend(grid: np.ndarray, top: int, left: int, border_color: int, mask: np.ndarray) -> None:
    block = np.full((3, 3), BG, dtype=int)
    block[mask] = GREY
    block[0, 1] = border_color
    grid[top : top + 3, left : left + 3] = block


def _place_frames(
    rng: random.Random, main_h: int, main_w: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((main_h, main_w), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []

    for height, width in sorted(sizes, key=lambda hw: hw[0] * hw[1], reverse=True):
        r_lo, r_hi = 1, main_h - height - 1
        c_lo, c_hi = 1, main_w - width - 1
        if r_lo > r_hi or c_lo > c_hi:
            return None

        for _ in range(200):
            top = rng.randint(r_lo, r_hi)
            left = rng.randint(c_lo, c_hi)
            rr0 = max(0, top - 1)
            rr1 = min(main_h, top + height + 1)
            cc0 = max(0, left - 1)
            cc1 = min(main_w, left + width + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placed.append((top, left, height, width))
            break
        else:
            return None

    rng.shuffle(placed)
    return placed


def _sample_frame_sizes(rng: random.Random, num_frames: int) -> list[tuple[int, int]]:
    sizes = [(rng.randint(5, 8), rng.randint(5, 9)) for _ in range(num_frames)]
    if all(h == 5 and w == 5 for h, w in sizes):
        idx = rng.randrange(num_frames)
        h, w = sizes[idx]
        sizes[idx] = (h + 1, w)
    return sizes


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.choice(FRAME_COUNT_CHOICES)
    sizes = _sample_frame_sizes(rng, num_frames)
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)

    main_h_min = max(4 * num_frames, max_h + 2, 13)
    main_w_min = max(max_w + 2, 15 if num_frames == 3 else 12)
    main_h = rng.randint(main_h_min, 25)
    main_w = rng.randint(main_w_min, 26)

    placements = _place_frames(rng, main_h, main_w, sizes)
    if placements is None:
        return None

    total_needed_colors = num_frames + 4
    color_pool = rng.sample(AVAILABLE_COLORS, total_needed_colors)
    border_colors = color_pool[:num_frames]
    marker_colors = color_pool[num_frames : num_frames + 2]
    fill_colors = color_pool[num_frames + 2 : num_frames + 4]
    color_mapping = dict(zip(marker_colors, fill_colors))

    if num_frames == 2:
        marker_assignment = marker_colors[:]
        rng.shuffle(marker_assignment)
    else:
        marker_assignment = [marker_colors[0], marker_colors[1], rng.choice(marker_colors)]
        rng.shuffle(marker_assignment)

    pattern_masks = rng.sample(list(PATTERN_LIBRARY), num_frames)

    grid = np.full((main_h + 2, main_w + 3), BG, dtype=int)

    for (top, left, height, width), border_color, marker_color in zip(
        placements, border_colors, marker_assignment
    ):
        _draw_frame(grid, top, left, height, width, border_color)
        inner_h = height - 2
        inner_w = width - 2
        marker_r = top + 1 + rng.randrange(inner_h)
        marker_c = left + 1 + rng.randrange(inner_w)
        grid[marker_r, marker_c] = marker_color

    legend_cols = sorted(rng.sample(range(1, main_w - 1), 2))
    for col, marker_color in zip(legend_cols, marker_colors):
        grid[main_h, col] = marker_color
        grid[main_h + 1, col] = color_mapping[marker_color]

    pattern_left = main_w
    top_padding_slack = main_h - 4 * num_frames
    legend_top = 1 + rng.randint(0, top_padding_slack)
    for idx, (border_color, mask) in enumerate(zip(border_colors, pattern_masks)):
        top = legend_top + 4 * idx
        _draw_pattern_legend(grid, top, pattern_left, border_color, mask)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
