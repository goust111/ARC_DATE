import math
from collections import defaultdict

import numpy as np


BG_COLOR = 8
BASELINE_COLOR = 9
SPRITE_COLORS = (1, 2, 3, 4, 5, 6, 7)
MAX_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=max(10, int(values.max()) + 1))
    return int(np.argmax(counts))


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _extract_sprites(grid: np.ndarray) -> tuple[int, int, list[dict[str, int | float]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 2
    baseline_color = int(g[-1, 0])
    assert np.all(g[-1] == baseline_color), "bottom row must be a full-width baseline"

    above = g[:-1]
    bg = _most_common_color(above)
    colors = [int(color) for color in np.unique(above) if int(color) not in {bg, baseline_color}]
    assert colors, "expected at least one sprite above the baseline"

    sprites: list[dict[str, int | float]] = []
    for color in colors:
        mask = above == color
        assert _is_single_component(mask), f"color {color} must form exactly one 4-connected sprite"
        coords = np.argwhere(mask)
        area = int(len(coords))
        assert 1 <= area <= 6, f"sprite area must be in 1..6, got {area}"
        sprites.append(
            {
                "color": color,
                "area": area,
                "center_key": int(coords[:, 1].sum()),
                "min_c": int(coords[:, 1].min()),
                "min_r": int(coords[:, 0].min()),
            }
        )
    return bg, baseline_color, sprites


def _rounded_gap_sizes(total_gap: int, gap_count: int) -> list[int]:
    boundaries = [int(math.floor(i * total_gap / gap_count + 0.5)) for i in range(gap_count + 1)]
    return [boundaries[i + 1] - boundaries[i] for i in range(gap_count)]


def _spread_segments(width: int, lengths: list[int]) -> list[int]:
    total_len = int(sum(lengths))
    assert total_len <= width, "segments do not fit in the output width"
    gaps = _rounded_gap_sizes(width - total_len, len(lengths) + 1)
    starts: list[int] = []
    c = gaps[0]
    for idx, length in enumerate(lengths):
        starts.append(int(c))
        c += int(length) + int(gaps[idx + 1])
    return starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange scattered sprites into evenly spaced horizontal tree layers keyed by sprite area.

    Input structure:
    - The background is a dominant azure field.
    - The bottom row is a full-width maroon baseline.
    - Above the baseline sit several differently colored 4-connected monochrome sprites.
    - Sprite colors are unique, and each sprite has between 1 and 6 pixels.

    Rule:
    - Infer each sprite's depth from its pixel count: area 1 goes on the row just above the baseline, area 2 one row higher, and so on.
    - Within the same depth, preserve the left-to-right order of sprites by their original horizontal center.
    - Build an output whose height is the maximum sprite area plus the baseline row and whose width matches the input.
    - Replace every sprite by a horizontal segment in its original color whose length equals its area, and distribute segments on each depth row with gaps that are as even as possible across the width.

    Key insight:
    - A sprite's pixel count has a double role: it determines both the sprite's vertical layer in the output tree and the length of the simplified segment drawn there.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    assert height >= 2 and width >= 1

    bg, baseline_color, sprites = _extract_sprites(g)
    max_depth = max(int(sprite["area"]) for sprite in sprites)

    out = np.full((max_depth + 1, width), bg, dtype=int)
    out[-1, :] = baseline_color

    by_depth: dict[int, list[dict[str, int | float]]] = defaultdict(list)
    for sprite in sprites:
        by_depth[int(sprite["area"])].append(sprite)

    for depth, items in by_depth.items():
        ordered = sorted(items, key=lambda item: (int(item["center_key"]), int(item["min_c"]), int(item["min_r"])))
        starts = _spread_segments(width, [depth] * len(ordered))
        row = max_depth - depth
        for start, item in zip(starts, ordered):
            out[row, start : start + depth] = int(item["color"])

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    coords[:, 0] -= int(coords[:, 0].min())
    coords[:, 1] -= int(coords[:, 1].min())
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order]


def _transform_shape(coords: np.ndarray) -> np.ndarray:
    transformed = coords.copy()
    for _ in range(int(_RNG.integers(4))):
        transformed = np.column_stack((-transformed[:, 1], transformed[:, 0]))
    if bool(_RNG.integers(2)):
        transformed[:, 1] *= -1
    return _normalize_shape({(int(r), int(c)) for r, c in transformed})


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nbr = (r + dr, c + dc)
                if nbr not in cells:
                    frontier.add(nbr)
        frontier_list = list(frontier)
        pick = frontier_list[int(_RNG.integers(len(frontier_list)))]
        cells.add((int(pick[0]), int(pick[1])))
    return _transform_shape(_normalize_shape(cells))


def _can_place(coords: np.ndarray, top: int, left: int, occupied: np.ndarray) -> bool:
    rows = coords[:, 0] + int(top)
    cols = coords[:, 1] + int(left)
    max_r, max_c = occupied.shape
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= max_r or cols.max() >= max_c:
        return False
    for r, c in zip(rows, cols):
        r0 = max(0, int(r) - 1)
        r1 = min(max_r, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(max_c, int(c) + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _paint(coords: np.ndarray, top: int, left: int, grid: np.ndarray, occupied: np.ndarray, color: int) -> dict[str, int]:
    rows = coords[:, 0] + int(top)
    cols = coords[:, 1] + int(left)
    grid[rows, cols] = int(color)
    occupied[rows, cols] = True
    return {
        "area": int(len(coords)),
        "center_key": int(cols.sum()),
        "min_c": int(cols.min()),
    }


def _sample_areas() -> list[int] | None:
    for _ in range(100):
        if bool(_RNG.integers(2)):
            count = int(_RNG.integers(3, 7))
            areas = list(map(int, _RNG.choice(np.arange(1, 7), size=count, replace=False)))
        else:
            count = int(_RNG.integers(4, 8))
            areas = list(map(int, _RNG.integers(1, 7, size=count)))
            if len(set(areas)) == len(areas):
                areas[-1] = areas[0]
        if len(set(areas)) < 2:
            continue
        depth_counts = defaultdict(int)
        for area in areas:
            depth_counts[int(area)] += 1
        required_width = max(depth * depth_counts[depth] + max(2, depth_counts[depth] + 1) for depth in depth_counts)
        if required_width <= 30:
            return areas
    return None


def _centers_are_distinct(per_depth: dict[int, list[dict[str, int]]]) -> bool:
    for items in per_depth.values():
        keys = [int(item["center_key"]) for item in items]
        if len(keys) != len(set(keys)):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    areas = _sample_areas()
    if areas is None:
        return None

    depth_counts = defaultdict(int)
    for area in areas:
        depth_counts[int(area)] += 1
    min_width = max(12, max(depth * depth_counts[depth] + max(2, depth_counts[depth] + 1) for depth in depth_counts))
    width = int(_RNG.integers(min_width, 31))
    count = len(areas)
    min_height = max(10, 2 * count + 3)
    height = int(_RNG.integers(min_height, min(31, min_height + 9)))

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[-1, :] = BASELINE_COLOR
    occupied = np.zeros((height - 1, width), dtype=bool)

    colors = list(map(int, _RNG.choice(np.array(SPRITE_COLORS), size=count, replace=False)))
    order = np.argsort([-area for area in areas])
    placed: dict[int, list[dict[str, int]]] = defaultdict(list)

    for idx in order:
        area = int(areas[int(idx)])
        color = colors[int(idx)]
        shape = _random_polyomino(area)
        shape_h = int(shape[:, 0].max()) + 1
        shape_w = int(shape[:, 1].max()) + 1
        if shape_h >= height or shape_w > width:
            return None

        success = False
        for _ in range(250):
            top = int(_RNG.integers(0, height - 1 - shape_h + 1))
            left = int(_RNG.integers(0, width - shape_w + 1))
            if not _can_place(shape, top, left, occupied):
                continue
            sprite_info = _paint(shape, top, left, grid[:-1], occupied, color)
            placed[area].append(sprite_info)
            success = True
            break
        if not success:
            return None

    if not _centers_are_distinct(placed):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
