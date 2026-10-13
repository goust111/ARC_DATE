import random
from collections import Counter

import numpy as np


BLACK = 0
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paired_palette_ok(row: np.ndarray) -> bool:
    return row.size % 2 == 0 and np.all(row[0::2] == row[1::2])


def _unique_most_frequent(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected at least one marker pixel"
    top = counts.most_common(2)
    if len(top) > 1:
        assert top[0][1] > top[1][1], "most frequent marker color must be unique"
    return int(top[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 5 and w >= 6, "expected room for palette row and framed board"
    assert _paired_palette_ok(g[0]), "palette row must contain adjacent equal-color pairs"

    palette = g[0].copy()
    frame_color = int(g[1, 0])
    assert frame_color != BLACK, "input frame color must be non-black"
    assert np.all(g[1, :] == frame_color), "top frame edge must be solid"
    assert np.all(g[-1, :] == frame_color), "bottom frame edge must be solid"
    assert np.all(g[1:, 0] == frame_color), "left frame edge must be solid"
    assert np.all(g[1:, -1] == frame_color), "right frame edge must be solid"

    interior = g[2:-1, 1:-1]
    marker_mask = (interior != BLACK) & (interior != frame_color)
    marker_values = interior[marker_mask]
    draw_color = _unique_most_frequent(marker_values)

    selected_mask = np.zeros_like(g, dtype=bool)
    selected_mask[2:-1, 1:-1] = interior == draw_color
    box = _bbox(selected_mask)
    assert box is not None, "selected color must appear in the interior"
    r0, c0, r1, c1 = box

    out = np.full_like(g, BLACK)
    out[0] = palette
    out[1, :] = draw_color
    out[-1, :] = draw_color
    out[1:, 0] = draw_color
    out[1:, -1] = draw_color
    out[r0 : r1 + 1, c0] = draw_color
    out[r0 : r1 + 1, c1] = draw_color
    out[r0, c0 : c1 + 1] = draw_color
    out[r1, c0 : c1 + 1] = draw_color
    return out


def _sample_positions(
    rng: random.Random,
    interior_h: int,
    interior_w: int,
    chosen_box: tuple[int, int, int, int],
    winner_count: int,
    other_specs: list[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    r0, c0, r1, c1 = chosen_box
    winner_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if len(winner_cells) < winner_count:
        return None

    placements: dict[tuple[int, int], int] = {}
    for cell in rng.sample(winner_cells, winner_count):
        placements[cell] = 0

    available = [
        (r, c)
        for r in range(interior_h)
        for c in range(interior_w)
        if (r, c) not in placements
    ]
    rng.shuffle(available)

    needed = sum(count for _, count in other_specs)
    if len(available) < needed:
        return None

    idx = 0
    for color, count in other_specs:
        for _ in range(count):
            placements[available[idx]] = color
            idx += 1
    return placements


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(3, 5)
    width = pair_count * 2
    height = rng.randint(7, 12)
    interior_h = height - 3
    interior_w = width - 2
    if interior_h < 2 or interior_w < 3:
        return None

    all_colors = list(range(1, 10))
    palette_colors = rng.sample(all_colors, pair_count)
    frame_color = rng.choice(palette_colors)
    marker_colors = [c for c in palette_colors if c != frame_color]
    if len(marker_colors) < 2:
        return None

    winner_color = rng.choice(marker_colors)
    loser_colors = [c for c in marker_colors if c != winner_color]
    rng.shuffle(loser_colors)
    loser_count = rng.randint(1, min(3, len(loser_colors)))
    loser_colors = loser_colors[:loser_count]

    max_box_h = min(interior_h, 5)
    max_box_w = min(interior_w, 6)
    bbox_h = rng.randint(1, max_box_h)
    bbox_w = rng.randint(1, max_box_w)
    box_r0 = rng.randint(0, interior_h - bbox_h)
    box_c0 = rng.randint(0, interior_w - bbox_w)
    chosen_box = (box_r0, box_c0, box_r0 + bbox_h - 1, box_c0 + bbox_w - 1)

    box_area = bbox_h * bbox_w
    winner_count = rng.randint(1, min(box_area, max(2, min(7, box_area))))
    if winner_count == 1 and box_area > 1 and rng.random() < 0.7:
        winner_count = 2

    other_specs: list[tuple[int, int]] = []
    remaining_capacity = interior_h * interior_w - winner_count
    for color in loser_colors:
        max_count = min(winner_count - 1, remaining_capacity)
        if max_count <= 0:
            return None
        count = rng.randint(1, max_count)
        other_specs.append((color, count))
        remaining_capacity -= count

    placements = _sample_positions(
        rng=rng,
        interior_h=interior_h,
        interior_w=interior_w,
        chosen_box=chosen_box,
        winner_count=winner_count,
        other_specs=other_specs,
    )
    if placements is None:
        return None

    palette = np.repeat(np.array(palette_colors, dtype=int), 2)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0] = palette
    grid[1, :] = frame_color
    grid[-1, :] = frame_color
    grid[1:, 0] = frame_color
    grid[1:, -1] = frame_color

    for (r, c), color in placements.items():
        grid[r + 2, c + 1] = winner_color if color == 0 else color

    # Reject accidental bbox shrinkage or frequency ambiguity.
    winner_mask = grid[2:-1, 1:-1] == winner_color
    if _bbox(winner_mask) != chosen_box:
        return None

    marker_values = grid[2:-1, 1:-1][(grid[2:-1, 1:-1] != BLACK) & (grid[2:-1, 1:-1] != frame_color)]
    counts = Counter(map(int, marker_values.reshape(-1)))
    if counts[winner_color] != winner_count:
        return None
    if any(color != winner_color and counts[color] >= winner_count for color in counts):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
