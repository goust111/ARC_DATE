from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_TO_MARKER = {
    (-1, 0): (0, 1),
    (1, 0): (2, 1),
    (0, -1): (1, 0),
    (0, 1): (1, 2),
}
MARKER_TO_DIR = {v: k for k, v in DIR_TO_MARKER.items()}
SIDE_CELLS = ((0, 1), (1, 0), (1, 2), (2, 1))

GEN_RETRIES = 500


def _glyph_patch(color: int, direction: tuple[int, int]) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    patch[0, 0] = color
    patch[0, 2] = color
    patch[2, 0] = color
    patch[2, 2] = color
    mr, mc = DIR_TO_MARKER[direction]
    patch[mr, mc] = color
    return patch


def _decode_patch(patch: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    if patch.shape != (3, 3):
        return None
    colors = {int(v) for v in patch.ravel() if int(v) not in (BLACK, GREY)}
    if len(colors) != 1:
        return None
    color = colors.pop()
    if patch[1, 1] != BLACK:
        return None
    if any(int(patch[r, c]) != color for r, c in ((0, 0), (0, 2), (2, 0), (2, 2))):
        return None
    marker_cells = [(r, c) for r, c in SIDE_CELLS if int(patch[r, c]) == color]
    if len(marker_cells) != 1:
        return None
    if any(int(patch[r, c]) != BLACK for r, c in SIDE_CELLS if (r, c) != marker_cells[0]):
        return None
    return color, MARKER_TO_DIR[marker_cells[0]]


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected one full-height grey separator column"
    return cols[0]


def _decode_glyphs(left: np.ndarray) -> dict[int, tuple[int, int]]:
    glyphs: list[tuple[int, int, int, tuple[int, int]]] = []
    for r in range(left.shape[0] - 2):
        for c in range(left.shape[1] - 2):
            decoded = _decode_patch(left[r : r + 3, c : c + 3])
            if decoded is not None:
                color, direction = decoded
                glyphs.append((r, c, color, direction))

    glyphs.sort()
    accepted: list[tuple[int, int, int, tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    for r, c, color, direction in glyphs:
        cells = {(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3)}
        if occupied.isdisjoint(cells):
            accepted.append((r, c, color, direction))
            occupied.update(cells)

    mapping: dict[int, tuple[int, int]] = {}
    for _r, _c, color, direction in accepted:
        assert color not in mapping, "glyph colors must be unique"
        mapping[color] = direction

    assert 2 <= len(mapping) <= 4, "expected 2-4 glyphs in the left panel"
    return mapping


def _sorted_seeds(right: np.ndarray) -> list[tuple[int, int, int]]:
    seeds = [(int(r), int(c), int(right[r, c])) for r, c in np.argwhere(right != BLACK)]
    seeds.sort()
    return seeds


def _cast_rays(
    right: np.ndarray, color_to_dir: dict[int, tuple[int, int]]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = right.copy()
    traces: list[dict[str, object]] = []
    for r, c, color in _sorted_seeds(right):
        assert color in color_to_dir, "every seed color must have a matching glyph"
        dr, dc = color_to_dir[color]
        rr, cc = r + dr, c + dc
        painted = 0
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == BLACK:
            out[rr, cc] = color
            painted += 1
            rr += dr
            cc += dc
        stop = "boundary" if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]) else "collision"
        traces.append(
            {
                "seed": (r, c, color),
                "direction": (dr, dc),
                "painted": painted,
                "stop": stop,
            }
        )
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode left-panel glyphs into directions, then emit right-panel color-bound rays and crop away the legend.

    Input structure:
    - The grid has a black background and one full-height grey separator column splitting it into left and right panels.
    - The left panel contains 2-4 colored 3x3 glyphs; each glyph uses one color, has colored corner anchors, and one colored side-center cell encoding a cardinal direction.
    - The right panel contains single-pixel seeds whose colors match left-panel glyph colors.

    Rule:
    - Read each left glyph, decode its side-center marker as up, down, left, or right, and build a color-to-direction map.
    - Process right-panel seeds in reading order (top to bottom, then left to right); from each seed, extend a ray cell by cell in its decoded direction.
    - Paint only black cells; stop before the first non-black cell or the panel boundary, then erase the entire left panel and separator by returning only the transformed right panel.

    Key insight:
    - The left glyphs do not map colors to colors; they symbolically map colors to directions, and that decoded direction controls each same-colored seed's ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    sep = _find_separator_col(g)
    color_to_dir = _decode_glyphs(g[:, :sep])
    right = g[:, sep + 1 :]
    out, _traces = _cast_rays(right, color_to_dir)
    return out


def _nonempty_sample(values: Iterable[int], k: int) -> list[int]:
    seq = list(values)
    return random.sample(seq, k)


def _slot_positions(height: int, left_cols: int) -> list[tuple[int, int]]:
    row_slots = [r for r in (1, 5, 9, 13, 17) if r + 2 < height]
    col_slots = [1] if left_cols == 1 else [1, 5]
    return [(r, c) for r in row_slots for c in col_slots]


def _choose_grid_shape(glyph_count: int) -> tuple[int, int, int]:
    for _ in range(100):
        height = random.randint(11, 18)
        left_cols = 1 if glyph_count == 2 and random.random() < 0.5 else 2
        if len(_slot_positions(height, left_cols)) < glyph_count:
            continue
        left_width = 5 if left_cols == 1 else 9
        max_right = min(16, 30 - left_width - 1)
        right_width = random.randint(8, max_right)
        return height, left_width, right_width
    raise RuntimeError("failed to choose a valid grid shape")


def _seed_counts(color_count: int) -> list[int]:
    counts = [1] * color_count
    extra_budget = random.randint(1, min(4, color_count + 2))
    while extra_budget > 0:
        idx = random.randrange(color_count)
        if counts[idx] < 3:
            counts[idx] += 1
            extra_budget -= 1
        elif all(count == 3 for count in counts):
            break
    return counts


def _place_seeds(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    right = np.zeros((height, width), dtype=int)
    taken: set[tuple[int, int]] = set()
    for color in colors:
        for _ in range(200):
            r = random.randrange(height)
            c = random.randrange(width)
            if (r, c) in taken:
                continue
            neighbors = sum((abs(r - rr) <= 1 and abs(c - cc) <= 1) for rr, cc in taken)
            if neighbors > 1:
                continue
            taken.add((r, c))
            right[r, c] = color
            break
        else:
            return None
    return right


def _oracle_accepts(
    full_grid: np.ndarray,
    assigned_dirs: dict[int, tuple[int, int]],
    expected_right: np.ndarray,
) -> bool:
    try:
        sep = _find_separator_col(full_grid)
        decoded_dirs = _decode_glyphs(full_grid[:, :sep])
        if decoded_dirs != assigned_dirs:
            return False
        if not np.array_equal(main(full_grid), expected_right):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    glyph_count = random.randint(2, 4)
    height, left_width, right_width = _choose_grid_shape(glyph_count)
    total_width = left_width + 1 + right_width
    grid = np.zeros((height, total_width), dtype=int)
    grid[:, left_width] = GREY

    positions = _slot_positions(height, 1 if left_width == 5 else 2)
    chosen_slots = random.sample(positions, glyph_count)
    colors = _nonempty_sample(PALETTE, glyph_count)
    directions = random.sample(list(DIR_TO_MARKER), glyph_count)
    color_to_dir = dict(zip(colors, directions))

    for (r, c), color in zip(chosen_slots, colors):
        patch = _glyph_patch(color, color_to_dir[color])
        grid[r : r + 3, c : c + 3] = patch

    counts = _seed_counts(glyph_count)
    seed_colors: list[int] = []
    for color, count in zip(colors, counts):
        seed_colors.extend([color] * count)
    random.shuffle(seed_colors)

    for _ in range(200):
        right = _place_seeds(height, right_width, seed_colors)
        if right is None:
            continue
        rendered_right, traces = _cast_rays(right, color_to_dir)
        if not traces:
            continue

        total_painted = sum(int(t["painted"]) for t in traces)
        if total_painted < glyph_count + 1 or total_painted > height * right_width // 2:
            continue

        per_color_painted = {color: 0 for color in colors}
        boundary_hits = 0
        collision_hits = 0
        for trace in traces:
            seed_color = int(trace["seed"][2])
            per_color_painted[seed_color] += int(trace["painted"])
            if trace["painted"] > 0 and trace["stop"] == "boundary":
                boundary_hits += 1
            if trace["painted"] > 0 and trace["stop"] == "collision":
                collision_hits += 1

        if any(per_color_painted[color] == 0 for color in colors):
            continue
        if boundary_hits == 0 or collision_hits == 0:
            continue
        if len({trace["direction"] for trace in traces}) < 2:
            continue

        grid[:, left_width + 1 :] = right
        if _oracle_accepts(grid, color_to_dir, rendered_right):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
