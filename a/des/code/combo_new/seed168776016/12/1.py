import random

import numpy as np


BLACK = 0
FRAME_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

LEGEND_COUNT_RANGE = (2, 4)
LEGEND_GAP_RANGE = (1, 2)
FRAME_INTERIOR_RANGE = (2, 7)
GEN_RETRIES = 500
PLACEMENT_RETRIES = 200

_TILE_MASKS = [
    np.array(
        [[(bits >> 0) & 1, (bits >> 1) & 1], [(bits >> 2) & 1, (bits >> 3) & 1]],
        dtype=bool,
    )
    for bits in range(1, 15)
]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(pts)
    return components


def _frame_bbox_if_valid(grid: np.ndarray, pts: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected_area = 2 * height + 2 * width - 4
    if len(pts) != expected_area:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not (
        np.all(sub[0, :] == FRAME_COLOR)
        and np.all(sub[-1, :] == FRAME_COLOR)
        and np.all(sub[:, 0] == FRAME_COLOR)
        and np.all(sub[:, -1] == FRAME_COLOR)
    ):
        return None
    if np.any(sub[1:-1, 1:-1] == FRAME_COLOR):
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components_of_color(grid, FRAME_COLOR):
        bbox = _frame_bbox_if_valid(grid, pts)
        if bbox is not None:
            frames.append(bbox)
    frames.sort()
    assert frames, "expected at least one grey rectangular frame"
    return frames


def _extract_legends(grid: np.ndarray) -> dict[int, np.ndarray]:
    h, w = grid.shape
    assert w >= 3, "grid must reserve the last three columns for legends"
    c0 = w - 3
    legends: dict[int, np.ndarray] = {}

    for r in range(h - 2):
        block = grid[r : r + 3, c0:w]
        if block.shape != (3, 3):
            continue
        if int(block[0, 0]) == BLACK:
            continue
        above_clear = r == 0 or np.all(grid[r - 1, c0:w] == BLACK)
        below_clear = r + 3 == h or np.all(grid[r + 3, c0:w] == BLACK)
        if not (above_clear and below_clear):
            continue

        tile = block[1:3, 1:3].copy()
        tile_colors = {int(v) for v in np.unique(tile)}
        assert BLACK in tile_colors and len(tile_colors) == 2, "legend tile must use black and one fill color"

        key_color = int(block[0, 0])
        assert key_color not in legends, "legend key colors must be unique"
        legends[key_color] = tile

    assert legends, "expected at least one legend in the rightmost 3 columns"
    return legends


def _repeat_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = (height + tile.shape[0] - 1) // tile.shape[0]
    reps_c = (width + tile.shape[1] - 1) // tile.shape[1]
    return np.tile(tile, (reps_r, reps_c))[:height, :width]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legends = _extract_legends(g)
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_pts = np.argwhere((interior != BLACK) & (interior != FRAME_COLOR))
        assert len(seed_pts) == 1, "each frame interior must contain exactly one seed pixel"
        sr, sc = map(int, seed_pts[0])
        seed_color = int(interior[sr, sc])
        assert seed_color in legends, "seed color must appear in the legend column"

        tile = legends[seed_color]
        out[r0 + 1 : r1, c0 + 1 : c1] = _repeat_tile(tile, interior.shape[0], interior.shape[1])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey rectangular enclosure by tiling the legend pattern indexed by its seed color.

    Input structure:
    - The background is black.
    - Several one-pixel-thick grey rectangular frames enclose black interiors, each with exactly one colored seed pixel.
    - The rightmost side contains a vertical column of 3x3 legend blocks whose top-left pixel is a seed color key.

    Rule:
    - For each framed region, read the unique seed color inside the frame.
    - Find the legend whose top-left pixel has the same color, and extract that legend's lower-right 2x2 tile.
    - Starting from the frame interior's top-left cell, repeat the 2x2 tile periodically until the full interior is filled.
    - The repeated tile overwrites the seed pixel while the grey frame remains unchanged.

    Key insight:
    - The seed pixel is only an index into the right-side legends; the actual fill is a spatial 2x2 pattern encoded in the matching legend block.
    """
    return _solve(grid)


def _build_legend_block(key_color: int, fill_color: int, tile_mask: np.ndarray) -> np.ndarray:
    block = np.full((3, 3), BLACK, dtype=int)
    for r in range(3):
        for c in range(3):
            if r == 0 and c == 0:
                continue
            if c == 0:
                continue
            if bool(tile_mask[(r - 1) % 2, (c - 1) % 2]):
                block[r, c] = fill_color
    block[0, 0] = key_color
    return block


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _legend_rows(rng: random.Random, height: int, count: int) -> list[int] | None:
    gaps = [rng.randint(*LEGEND_GAP_RANGE) for _ in range(count - 1)]
    total = 3 * count + sum(gaps)
    if total > height:
        return None
    top_pad = rng.randint(0, height - total)
    rows = []
    cur = top_pad
    for i in range(count):
        rows.append(cur)
        cur += 3
        if i < count - 1:
            cur += gaps[i]
    return rows


def _frame_specs(
    rng: random.Random,
    key_colors: list[int],
    legend_tiles: dict[int, np.ndarray],
) -> list[dict[str, object]]:
    frame_count = rng.randint(1, len(key_colors))
    used_keys = rng.sample(key_colors, frame_count)
    specs: list[dict[str, object]] = []
    for key_color in used_keys:
        interior_h = rng.randint(*FRAME_INTERIOR_RANGE)
        interior_w = rng.randint(*FRAME_INTERIOR_RANGE)
        seed_r = rng.randrange(interior_h)
        seed_c = rng.randrange(interior_w)
        specs.append(
            {
                "key_color": key_color,
                "tile": legend_tiles[key_color],
                "interior_h": interior_h,
                "interior_w": interior_w,
                "seed_pos": (seed_r, seed_c),
                "area": (interior_h + 2) * (interior_w + 2),
            }
        )
    specs.sort(key=lambda item: int(item["area"]), reverse=True)
    return specs


def _try_place_frames(
    rng: random.Random,
    height: int,
    width: int,
    frame_specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    placed_boxes: list[tuple[int, int, int, int]] = []
    usable_col_max = width - 6
    if usable_col_max < 1:
        return None

    for spec in frame_specs:
        inner_h = int(spec["interior_h"])
        inner_w = int(spec["interior_w"])
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if height - outer_h - 1 < 1 or usable_col_max - outer_w + 1 < 1:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, height - outer_h - 1)
            c0 = rng.randint(1, usable_col_max - outer_w + 1)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            padded = (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
            if any(_boxes_overlap(padded, other) for other in placed_boxes):
                continue
            spec["bbox"] = box
            placed_boxes.append(padded)
            placed = True
            break
        if not placed:
            return None

    return frame_specs


def _draw_sample(
    height: int,
    width: int,
    legend_rows: list[int],
    key_colors: list[int],
    fill_colors: list[int],
    tile_masks: list[np.ndarray],
    frame_specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((height, width), BLACK, dtype=int)
    expected = grid.copy()
    legend_tiles: dict[int, np.ndarray] = {}
    legend_col = width - 3

    for row, key_color, fill_color, mask in zip(legend_rows, key_colors, fill_colors, tile_masks, strict=True):
        block = _build_legend_block(key_color, fill_color, mask)
        grid[row : row + 3, legend_col:width] = block
        expected[row : row + 3, legend_col:width] = block
        legend_tiles[key_color] = block[1:3, 1:3].copy()

    for spec in frame_specs:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        key_color = int(spec["key_color"])
        tile = legend_tiles[key_color]
        inner_h = int(spec["interior_h"])
        inner_w = int(spec["interior_w"])
        seed_r, seed_c = map(int, spec["seed_pos"])

        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        expected[r0, c0 : c1 + 1] = FRAME_COLOR
        expected[r1, c0 : c1 + 1] = FRAME_COLOR
        expected[r0 : r1 + 1, c0] = FRAME_COLOR
        expected[r0 : r1 + 1, c1] = FRAME_COLOR

        grid[r0 + 1 + seed_r, c0 + 1 + seed_c] = key_color
        expected[r0 + 1 : r1, c0 + 1 : c1] = _repeat_tile(tile, inner_h, inner_w)

    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_count = rng.randint(*LEGEND_COUNT_RANGE)
    height = rng.randint(max(12, 4 * legend_count), 30)
    width = rng.randint(16, 30)

    rows = _legend_rows(rng, height, legend_count)
    if rows is None:
        return None

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    key_colors = colors[:legend_count]
    fill_colors = colors[legend_count : 2 * legend_count]
    tile_masks = [rng.choice(_TILE_MASKS) for _ in range(legend_count)]

    legend_tiles = {
        key_color: _build_legend_block(key_color, fill_color, mask)[1:3, 1:3].copy()
        for key_color, fill_color, mask in zip(key_colors, fill_colors, tile_masks, strict=True)
    }
    specs = _frame_specs(rng, key_colors, legend_tiles)
    specs = _try_place_frames(rng, height, width, specs)
    if specs is None:
        return None

    grid, expected = _draw_sample(height, width, rows, key_colors, fill_colors, tile_masks, specs)
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt(rng)
        if sample is not None:
            return sample
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
