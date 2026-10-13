import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_legend_entries(active: np.ndarray, bg: int) -> dict[int, np.ndarray] | None:
    h, w = map(int, active.shape)
    if h < 2 or w < 3 or (w - 1) % 2 != 0:
        return None

    entries: dict[int, np.ndarray] = {}
    top = 0
    while top < h:
        if top + 1 >= h:
            return None
        key = int(active[top, 0])
        if key == bg or int(active[top + 1, 0]) != bg:
            return None

        pattern = active[top : top + 2, 1:]
        if pattern.shape[1] < 2 or pattern.shape[1] % 2 != 0:
            return None
        if np.any(pattern == bg):
            return None

        tile = pattern[:, :2].copy()
        for c in range(0, pattern.shape[1], 2):
            if not np.array_equal(pattern[:, c : c + 2], tile):
                return None
        if key in entries:
            return None
        entries[key] = tile

        top += 2
        if top == h:
            break
        if not np.all(active[top] == bg):
            return None
        top += 1
    return entries if len(entries) >= 2 else None


def _parse_right_strip(strip: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], int] | None:
    if strip.shape[1] < 3:
        return None
    non_bg_rows = np.flatnonzero(np.any(strip != bg, axis=1))
    if len(non_bg_rows) == 0 or int(non_bg_rows[0]) != 0:
        return None

    active_h = int(non_bg_rows[-1]) + 1
    entries = _parse_legend_entries(strip[:active_h, :], bg)
    if entries is None:
        return None
    return entries, active_h


def _parse_bottom_strip(strip: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], int] | None:
    if strip.shape[0] < 2:
        return None
    non_bg_cols = np.flatnonzero(np.any(strip != bg, axis=0))
    if len(non_bg_cols) == 0 or int(non_bg_cols[0]) != 0:
        return None

    active_w = int(non_bg_cols[-1]) + 1
    entries = _parse_legend_entries(strip[:, :active_w], bg)
    if entries is None:
        return None
    return entries, active_w


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = map(int, grid.shape)

    for sep_col in range(w - 2, -1, -1):
        if not np.all(grid[:, sep_col] == bg):
            continue
        strip = grid[:, sep_col + 1 :]
        if not np.any(strip != bg):
            continue
        parsed = _parse_right_strip(strip, bg)
        if parsed is None:
            continue
        entries, active_h = parsed
        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[:active_h, sep_col + 1 : w] = True
        return entries, legend_mask

    for sep_row in range(0, h - 1):
        if not np.all(grid[sep_row, :] == bg):
            continue
        strip = grid[sep_row + 1 :, :]
        if not np.any(strip != bg):
            continue
        parsed = _parse_bottom_strip(strip, bg)
        if parsed is None:
            continue
        entries, active_w = parsed
        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[sep_row + 1 : h, :active_w] = True
        return entries, legend_mask

    raise AssertionError("legend area not found")


def _same_color_components(grid: np.ndarray, blocked: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or blocked[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pixels.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or blocked[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            components.append((color, pixels))
    return components


def _frame_bboxes(grid: np.ndarray, legend_mask: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for _color, pixels in _same_color_components(grid, legend_mask, bg):
        rows = [r for r, _ in pixels]
        cols = [c for _, c in pixels]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))

        if set(pixels) == expected:
            frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed color inside each framed room to look up a tiled fill pattern from the legend.

    Input structure:
    - Black background surrounds all objects.
    - A legend area sits on the right side or at the bottom, separated from the main area by one full black column or row.
    - Each legend entry occupies a 2-row band: its top-left cell is a single colored key pixel, and to the right a 2x2 tile is repeated horizontally; entries are separated by one black row.
    - The main area contains several non-overlapping 1-pixel-thick hollow rectangles of one frame color, each enclosing exactly one non-black seed pixel.

    Rule:
    - Parse the legend into a mapping from seed color to the repeated 2x2 tile shown beside that color.
    - For every hollow rectangle, find its unique interior seed pixel and use that seed color as the legend lookup key.
    - Tile the matched 2x2 pattern across the full interior of the rectangle from the interior's top-left corner, overwriting the seed pixel.
    - Leave the frame outlines, background, and legend area unchanged.

    Key insight:
    - The lone interior pixel is not part of the final picture; it is a lookup token that selects which visible tile should be repeated across the enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    legend, legend_mask = _find_legend(g, bg)
    frames = _frame_bboxes(g, legend_mask, bg)
    assert frames, "expected at least one framed region"

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere(interior != bg)
        assert len(seeds) == 1, "each frame interior must contain exactly one seed"
        sr, sc = map(int, seeds[0])
        seed_color = int(interior[sr, sc])
        assert seed_color in legend, "seed color must appear in the legend"

        tile = legend[seed_color]
        inner_h, inner_w = interior.shape
        for dr in range(inner_h):
            for dc in range(inner_w):
                out[r0 + 1 + dr, c0 + 1 + dc] = int(tile[dr % 2, dc % 2])
    return out


def _sample_tile(rng: random.Random, used: set[tuple[int, int, int, int]]) -> np.ndarray:
    while True:
        tile = np.array(
            [[rng.choice(COLOR_POOL), rng.choice(COLOR_POOL)], [rng.choice(COLOR_POOL), rng.choice(COLOR_POOL)]],
            dtype=int,
        )
        if FRAME in tile or BG in tile:
            continue
        sig = tuple(map(int, tile.reshape(-1)))
        if len(set(sig)) < 2 or sig in used:
            continue
        used.add(sig)
        return tile


def _target_frame_count(rng: random.Random, main_h: int, main_w: int) -> int:
    area = main_h * main_w
    max_count = min(4, max(1, area // 55))
    options = [count for count in (1, 2, 3, 4) if count <= max_count]
    return rng.choice(options)


def _place_frames(
    rng: random.Random,
    main_h: int,
    main_w: int,
    entry_keys: list[int],
) -> list[dict[str, int]] | None:
    target = _target_frame_count(rng, main_h, main_w)
    occupied = np.zeros((main_h, main_w), dtype=bool)
    frames: list[dict[str, int]] = []

    for _ in range(target):
        placed = False
        for _attempt in range(200):
            max_h = min(10, main_h - 2)
            max_w = min(10, main_w - 2)
            if max_h < 5 or max_w < 5:
                return None
            height = rng.randint(5, max_h)
            width = rng.randint(5, max_w)

            r0 = rng.randint(1, main_h - height - 1)
            c0 = rng.randint(1, main_w - width - 1)
            r1 = r0 + height
            c1 = c0 + width

            pad_r0 = r0 - 1
            pad_c0 = c0 - 1
            pad_r1 = r1 + 1
            pad_c1 = c1 + 1
            if occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
                continue

            occupied[pad_r0:pad_r1, pad_c0:pad_c1] = True
            frames.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "height": height,
                    "width": width,
                    "seed_r": rng.randint(r0 + 1, r1 - 2),
                    "seed_c": rng.randint(c0 + 1, c1 - 2),
                }
            )
            placed = True
            break
        if not placed:
            return None

    distinct_needed = min(len(frames), len(entry_keys), 2 if len(frames) > 1 else 1)
    chosen = rng.sample(entry_keys, distinct_needed)
    seed_colors = chosen[:]
    while len(seed_colors) < len(frames):
        seed_colors.append(rng.choice(entry_keys))
    rng.shuffle(seed_colors)
    for frame, color in zip(frames, seed_colors):
        frame["seed_color"] = int(color)
    return frames


def _render_grid(
    orientation: str,
    grid_h: int,
    grid_w: int,
    main_h: int,
    main_w: int,
    repeat_count: int,
    entry_keys: list[int],
    legend_tiles: dict[int, np.ndarray],
    frames: list[dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    pattern_w = 2 * repeat_count
    if orientation == "right":
        legend_col = main_w + 1
        for idx, key in enumerate(entry_keys):
            top = 3 * idx
            tile = legend_tiles[key]
            grid[top, legend_col] = key
            grid[top : top + 2, legend_col + 1 : legend_col + 1 + pattern_w] = np.tile(tile, (1, repeat_count))
    else:
        legend_row = main_h + 1
        for idx, key in enumerate(entry_keys):
            top = legend_row + 3 * idx
            tile = legend_tiles[key]
            grid[top, 0] = key
            grid[top : top + 2, 1 : 1 + pattern_w] = np.tile(tile, (1, repeat_count))

    for frame in frames:
        r0 = frame["r0"]
        c0 = frame["c0"]
        r1 = r0 + frame["height"] - 1
        c1 = c0 + frame["width"] - 1
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        grid[frame["seed_r"], frame["seed_c"]] = frame["seed_color"]

    out = grid.copy()
    for frame in frames:
        tile = legend_tiles[frame["seed_color"]]
        inner_h = frame["height"] - 2
        inner_w = frame["width"] - 2
        for dr in range(inner_h):
            for dc in range(inner_w):
                out[frame["r0"] + 1 + dr, frame["c0"] + 1 + dc] = int(tile[dr % 2, dc % 2])
    return grid, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(("right", "bottom"))
    entry_count = rng.randint(2, 4)
    repeat_count = rng.randint(2, 4)
    legend_h = 3 * entry_count - 1
    legend_w = 1 + 2 * repeat_count

    entry_keys = rng.sample(COLOR_POOL, entry_count)
    used_tiles: set[tuple[int, int, int, int]] = set()
    legend_tiles = {key: _sample_tile(rng, used_tiles) for key in entry_keys}

    if orientation == "right":
        main_w = rng.randint(12, 30 - 1 - legend_w)
        grid_h = rng.randint(max(12, legend_h), 30)
        grid_w = main_w + 1 + legend_w
        main_h = grid_h
    else:
        grid_w = rng.randint(max(12, legend_w), 30)
        main_h = rng.randint(12, 30 - 1 - legend_h)
        grid_h = main_h + 1 + legend_h
        main_w = grid_w

    frames = _place_frames(rng, main_h, main_w, entry_keys)
    if frames is None:
        return None

    grid, expected = _render_grid(
        orientation=orientation,
        grid_h=grid_h,
        grid_w=grid_w,
        main_h=main_h,
        main_w=main_w,
        repeat_count=repeat_count,
        entry_keys=entry_keys,
        legend_tiles=legend_tiles,
        frames=frames,
    )

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
