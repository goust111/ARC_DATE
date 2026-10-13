import random

import numpy as np


BG = 0
FRAME = 5
NON_FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
PANEL_WIDTH = 6
MAX_RETRIES = 400


def _panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == FRAME)
    assert pts.size > 0, "legend frame must be present"

    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    assert r0 > 0 and c0 > 0, "legend must sit in the bottom-right corner"
    assert r1 == grid.shape[0] - 1 and c1 == grid.shape[1] - 1, "legend frame must touch bottom and right edges"

    panel = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert panel.shape[0] >= 4 and panel.shape[1] >= 6, "legend panel is too small"
    assert np.all(panel[0, :] == FRAME)
    assert np.all(panel[-1, :] == FRAME)
    assert np.all(panel[:, 0] == FRAME)
    assert np.all(panel[:, -1] == FRAME)
    return r0, c0, r1, c1


def _legend_mapping(panel: np.ndarray) -> dict[frozenset[int], int]:
    interior = panel[1:-1, 1:-1]
    mapping: dict[frozenset[int], int] = {}

    for row in interior:
        nz = np.flatnonzero(row != BG)
        if nz.tolist() != [0, 1, 3]:
            continue
        a = int(row[0])
        b = int(row[1])
        mix = int(row[3])
        assert a != BG and b != BG and mix != BG
        key = frozenset((a, b))
        assert key not in mapping or mapping[key] == mix, "legend pair mapping must be unambiguous"
        mapping[key] = mix

    assert mapping, "legend mapping must be non-empty"
    return mapping


def _dominant_non_bg(values: np.ndarray) -> int:
    colors, counts = np.unique(values[values != BG], return_counts=True)
    assert len(colors) > 0
    return int(colors[np.argmax(counts)])


def _recolor_crossings(playfield: np.ndarray, mapping: dict[frozenset[int], int]) -> np.ndarray:
    row_ids = np.flatnonzero(np.all(playfield != BG, axis=1))
    col_ids = np.flatnonzero(np.all(playfield != BG, axis=0))
    assert len(row_ids) > 0 and len(col_ids) > 0, "playfield must contain both horizontal and vertical paths"

    row_colors = {int(r): _dominant_non_bg(playfield[int(r), :]) for r in row_ids}
    col_colors = {int(c): _dominant_non_bg(playfield[:, int(c)]) for c in col_ids}

    out = playfield.copy()
    for r in row_ids:
        rr = int(r)
        for c in col_ids:
            cc = int(c)
            key = frozenset((row_colors[rr], col_colors[cc]))
            if key in mapping:
                out[rr, cc] = mapping[key]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor horizontal/vertical path crossings using a framed legend.

    The grey frame in the bottom-right corner marks a legend panel.
    Each legend row encodes:
    - two adjacent input colors
    - one background gap
    - one output color

    After applying the pair -> mix lookup at every crossing, the framed legend
    area is removed from the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    r0, c0, r1, c1 = _panel_bbox(g)
    panel = g[r0 : r1 + 1, c0 : c1 + 1]
    mapping = _legend_mapping(panel)
    playfield = g[:r0, :c0]
    assert playfield.size > 0
    return _recolor_crossings(playfield, mapping)


def _sample_spaced_positions(length: int, count: int, rng: random.Random) -> list[int]:
    assert count >= 1
    assert length >= 2 * count - 1
    base = sorted(rng.sample(range(length - count + 1), count))
    return [b + i for i, b in enumerate(base)]


def _build_playfield(
    height: int,
    width: int,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    playfield = np.zeros((height, width), dtype=int)

    for row, color in zip(row_positions, row_colors):
        playfield[row, :] = color

    for col, color in zip(col_positions, col_colors):
        for row in range(height):
            if playfield[row, col] == BG or rng.random() < 0.5:
                playfield[row, col] = color

    return playfield


def _expected_output(
    playfield: np.ndarray,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
    mapping: dict[frozenset[int], int],
) -> np.ndarray:
    out = playfield.copy()
    for row, row_color in zip(row_positions, row_colors):
        for col, col_color in zip(col_positions, col_colors):
            key = frozenset((row_color, col_color))
            if key in mapping:
                out[row, col] = mapping[key]
    return out


def _assemble_input(
    playfield: np.ndarray,
    entries: list[tuple[frozenset[int], int]],
    rng: random.Random,
) -> np.ndarray:
    play_h, play_w = playfield.shape
    panel_h = len(entries) + 2
    grid = np.zeros((play_h + panel_h, play_w + PANEL_WIDTH), dtype=int)
    grid[:play_h, :play_w] = playfield

    r0 = play_h
    c0 = play_w
    grid[r0:, c0] = FRAME
    grid[r0:, -1] = FRAME
    grid[r0, c0:] = FRAME
    grid[-1, c0:] = FRAME

    for idx, (pair, mix) in enumerate(entries):
        a, b = tuple(pair)
        if rng.random() < 0.5:
            a, b = b, a
        rr = r0 + 1 + idx
        grid[rr, c0 + 1] = a
        grid[rr, c0 + 2] = b
        grid[rr, c0 + 4] = mix

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    total_paths = rng.randint(4, 6)
    horiz_count = rng.randint(2, total_paths - 2)
    vert_count = total_paths - horiz_count
    total_pairs = horiz_count * vert_count
    map_count = rng.randint(2, total_pairs - 1)

    path_colors = rng.sample(list(NON_FRAME_COLORS), total_paths)
    row_colors = path_colors[:horiz_count]
    col_colors = path_colors[horiz_count:]
    mix_pool = [c for c in NON_FRAME_COLORS if c not in path_colors]
    assert mix_pool

    mix_color_count = rng.randint(1, min(3, len(mix_pool)))
    mix_colors = rng.sample(mix_pool, mix_color_count)

    pair_ids = [(ri, ci) for ri in range(horiz_count) for ci in range(vert_count)]
    mapped_ids = set(rng.sample(pair_ids, map_count))
    mapping: dict[frozenset[int], int] = {}
    for ri, ci in pair_ids:
        if (ri, ci) not in mapped_ids:
            continue
        key = frozenset((row_colors[ri], col_colors[ci]))
        mapping[key] = rng.choice(mix_colors)

    min_h = max(8, 2 * horiz_count + 3)
    min_w = max(8, 2 * vert_count + 3)
    play_h = rng.randint(min_h, 16)
    play_w = rng.randint(min_w, 16)
    row_positions = _sample_spaced_positions(play_h, horiz_count, rng)
    col_positions = _sample_spaced_positions(play_w, vert_count, rng)

    playfield = _build_playfield(play_h, play_w, row_positions, row_colors, col_positions, col_colors, rng)
    expected = _expected_output(playfield, row_positions, row_colors, col_positions, col_colors, mapping)

    entries = list(mapping.items())
    rng.shuffle(entries)
    grid = _assemble_input(playfield, entries, rng)

    if np.count_nonzero(expected != playfield) != map_count:
        return None
    if np.count_nonzero(grid == FRAME) == 0:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
