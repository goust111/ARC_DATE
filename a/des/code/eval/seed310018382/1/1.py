import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    expected_size = 2 * (height + width) - 4
    if len(coords) != expected_size:
        return False

    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    return True


def _find_frame_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_count = -1

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        valid_count = 0
        for comp in _color_components(grid, color):
            if _is_rectangular_frame(comp):
                valid_count += 1
        if valid_count > best_count:
            best_count = valid_count
            best_color = color

    assert best_color is not None and best_count > 0, "expected at least one rectangular frame color"
    return int(best_color)


def _find_frames(grid: np.ndarray, frame_color: int) -> list[tuple[int, int, int, int]]:
    frames = []
    for comp in _color_components(grid, frame_color):
        if _is_rectangular_frame(comp):
            frames.append(_bbox(comp))
    assert frames, "expected at least one closed rectangular frame"
    frames.sort()
    return frames


def _extract_seed_color(grid: np.ndarray, frame: tuple[int, int, int, int], bg: int, frame_color: int) -> int:
    r0, c0, r1, c1 = frame
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (interior != bg) & (interior != frame_color)
    seed_coords = np.argwhere(seed_mask)
    assert len(seed_coords) == 1, "each frame interior must contain exactly one seed pixel"
    sr, sc = map(int, seed_coords[0])
    return int(interior[sr, sc])


def _legend_candidate_score(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    margins = 0
    if r == 0 or np.all(grid[r - 1, c : c + 3] == bg):
        margins += 1
    if c == 0 or np.all(grid[r : r + 3, c - 1] == bg):
        margins += 1
    if c + 3 >= w or np.all(grid[r : r + 3, c + 3] == bg):
        margins += 1
    if r + 3 >= h or np.all(grid[r + 3, c : c + 3] == bg):
        margins += 1
    patch = grid[r : r + 3, c : c + 3]
    non_bg = int(np.count_nonzero(patch != bg))
    return margins, non_bg, -r, -c


def _find_legend_tile(grid: np.ndarray, seed_color: int, min_row: int, bg: int, frame_color: int) -> np.ndarray:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], int, int]] = []

    for r in range(min_row, h - 2):
        for c in range(0, w - 2):
            patch = grid[r : r + 3, c : c + 3]
            if int(patch[0, 0]) != seed_color:
                continue
            colors = set(map(int, np.unique(patch)))
            if frame_color in colors:
                continue
            if not colors.issubset({bg, seed_color}):
                continue
            candidates.append((_legend_candidate_score(grid, r, c, bg), int(r), int(c)))

    assert candidates, f"missing legend block for seed color {seed_color}"
    _, r, c = max(candidates)
    return grid[r + 1 : r + 3, c + 1 : c + 3].copy()


def _tile_fill(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height) % 2
    cc = np.arange(width) % 2
    return tile[rr[:, None], cc[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed rectangular region using the 2x2 tile indexed by its seed color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color = _find_frame_color(g, bg)
    frames = _find_frames(g, frame_color)
    legend_min_row = max(r1 for _, _, r1, _ in frames) + 1

    frame_info = []
    seed_colors = set()
    for frame in frames:
        seed_color = _extract_seed_color(g, frame, bg, frame_color)
        assert seed_color not in seed_colors, "seed colors must be unique across frames"
        seed_colors.add(seed_color)
        frame_info.append((frame, seed_color))

    mapping = {
        seed_color: _find_legend_tile(g, seed_color, legend_min_row, bg, frame_color)
        for seed_color in seed_colors
    }

    out = g.copy()
    for (r0, c0, r1, c1), seed_color in frame_info:
        interior_h = int(r1 - r0 - 1)
        interior_w = int(c1 - c0 - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_fill(mapping[seed_color], interior_h, interior_w)
    return out


GEN_BG = 0
GEN_FRAME = 5
GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _sample_tile(color: int) -> np.ndarray:
    tile = np.full((2, 2), GEN_BG, dtype=int)
    mask = random.randint(1, 15)
    for idx in range(4):
        if (mask >> idx) & 1:
            tile[idx // 2, idx % 2] = int(color)
    return tile


def _legend_block(color: int, tile: np.ndarray) -> np.ndarray:
    block = np.full((3, 3), GEN_BG, dtype=int)
    block[0, 0] = int(color)
    block[1:, 1:] = tile
    return block


def _sample_region_count(legend_rows: int) -> int:
    low = 1 if legend_rows == 1 else 2
    return random.randint(low, 5)


def _sample_interior_size(region_count: int) -> int:
    limit = 7 if region_count <= 2 else 6 if region_count <= 4 else 5
    choices = list(range(1, limit + 1))
    return random.choice(choices)


def _build_expected(
    grid: np.ndarray,
    placements: list[tuple[int, int, int, int, np.ndarray]],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1, tile in placements:
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_fill(tile, r1 - r0 - 1, c1 - c0 - 1)
    return out


def _try_generate_input() -> np.ndarray | None:
    legend_rows = 1 if random.random() < 0.5 else 2
    region_count = _sample_region_count(legend_rows)
    colors = random.sample(list(GEN_COLORS), region_count)
    tiles = {color: _sample_tile(color) for color in colors}

    if legend_rows == 1:
        row_counts = [region_count]
    else:
        first_row = random.randint(1, region_count - 1)
        row_counts = [first_row, region_count - first_row]

    random.shuffle(colors)
    row_widths = [4 * count - 1 for count in row_counts]
    legend_height = 3 * legend_rows + (legend_rows - 1)
    min_h = max(legend_height + 6, 11 + 2 * region_count)
    min_w = max(max(row_widths) + 2, 12 + 2 * region_count)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.full((h, w), GEN_BG, dtype=int)

    legend_top = h - legend_height
    occupied = np.zeros((h, w), dtype=bool)
    occupied[max(0, legend_top - 1) : h, :] = True

    placements: list[tuple[int, int, int, int, np.ndarray]] = []
    specs = []
    for color in colors:
        inner_h = _sample_interior_size(region_count)
        inner_w = _sample_interior_size(region_count)
        specs.append(
            {
                "color": int(color),
                "tile": tiles[int(color)],
                "frame_h": int(inner_h + 2),
                "frame_w": int(inner_w + 2),
            }
        )
    specs.sort(key=lambda spec: spec["frame_h"] * spec["frame_w"], reverse=True)

    for spec in specs:
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        max_r0 = legend_top - frame_h - 1
        if max_r0 < 0 or frame_w > w:
            return None

        placed = False
        for _ in range(250):
            r0 = random.randint(0, max_r0)
            c0 = random.randint(0, w - frame_w)
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1

            pad_r0 = max(0, r0 - 1)
            pad_c0 = max(0, c0 - 1)
            pad_r1 = min(h, r1 + 2)
            pad_c1 = min(w, c1 + 2)
            if np.any(occupied[pad_r0:pad_r1, pad_c0:pad_c1]):
                continue

            grid[r0, c0 : c1 + 1] = GEN_FRAME
            grid[r1, c0 : c1 + 1] = GEN_FRAME
            grid[r0 : r1 + 1, c0] = GEN_FRAME
            grid[r0 : r1 + 1, c1] = GEN_FRAME

            seed_r = random.randint(r0 + 1, r1 - 1)
            seed_c = random.randint(c0 + 1, c1 - 1)
            grid[seed_r, seed_c] = int(spec["color"])

            occupied[pad_r0:pad_r1, pad_c0:pad_c1] = True
            placements.append((r0, c0, r1, c1, spec["tile"]))
            placed = True
            break
        if not placed:
            return None

    color_index = 0
    for row_idx, count in enumerate(row_counts):
        row_width = 4 * count - 1
        start_c = random.randint(0, w - row_width)
        row_r = legend_top + 4 * row_idx
        for j in range(count):
            color = int(colors[color_index])
            block = _legend_block(color, tiles[color])
            c0 = start_c + 4 * j
            grid[row_r : row_r + 3, c0 : c0 + 3] = block
            color_index += 1

    expected = _build_expected(grid, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _try_generate_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
