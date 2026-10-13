from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_uniform_non_bg(line: np.ndarray, bg: int) -> bool:
    first = int(line[0])
    return first != bg and bool(np.all(line == first))


def _collect_wall_candidates(grid: np.ndarray, bg: int) -> list[tuple[str, list[int]]]:
    h, w = grid.shape
    candidates: list[tuple[str, list[int]]] = []

    colors: list[int] = []
    for r in range(h):
        if _is_uniform_non_bg(grid[r, :], bg):
            colors.append(int(grid[r, 0]))
        else:
            break
    if 2 <= len(colors) <= 4 and len(set(colors)) == len(colors):
        candidates.append(("top", colors))

    colors = []
    for r in range(h - 1, -1, -1):
        if _is_uniform_non_bg(grid[r, :], bg):
            colors.append(int(grid[r, 0]))
        else:
            break
    if 2 <= len(colors) <= 4 and len(set(colors)) == len(colors):
        candidates.append(("bottom", colors))

    colors = []
    for c in range(w):
        if _is_uniform_non_bg(grid[:, c], bg):
            colors.append(int(grid[0, c]))
        else:
            break
    if 2 <= len(colors) <= 4 and len(set(colors)) == len(colors):
        candidates.append(("left", colors))

    colors = []
    for c in range(w - 1, -1, -1):
        if _is_uniform_non_bg(grid[:, c], bg):
            colors.append(int(grid[0, c]))
        else:
            break
    if 2 <= len(colors) <= 4 and len(set(colors)) == len(colors):
        candidates.append(("right", colors))

    return candidates


def _detect_wall(grid: np.ndarray) -> tuple[int, str, list[int]]:
    bg = _most_common_color(grid)
    candidates = _collect_wall_candidates(grid, bg)
    assert len(candidates) == 1, "expected exactly one edge wall"
    side, wall_colors = candidates[0]
    return bg, side, wall_colors


def _inner_stripe_lines(side: str, count: int, h: int, w: int) -> list[int]:
    if side == "top":
        return list(range(count - 1, -1, -1))
    if side == "bottom":
        return list(range(h - count, h))
    if side == "left":
        return list(range(count - 1, -1, -1))
    assert side == "right"
    return list(range(w - count, w))


def _line_color(grid: np.ndarray, side: str, line_index: int) -> int:
    if side in ("top", "bottom"):
        return int(grid[line_index, 0])
    return int(grid[0, line_index])


def _distance_to_wall(
    bbox: tuple[int, int, int, int], side: str, wall_depth: int, h: int, w: int
) -> int:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - wall_depth
    if side == "bottom":
        return h - wall_depth - r1 - 1
    if side == "left":
        return c0 - wall_depth
    assert side == "right"
    return w - wall_depth - c1 - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each scattered shape onto one wall stripe, ordered by distance to the wall.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, side, wall_colors = _detect_wall(g)
    wall_depth = len(wall_colors)

    other_colors = set(int(x) for x in np.unique(g)) - {bg} - set(wall_colors)
    assert len(other_colors) == 1, "expected one shared shape color"
    shape_color = other_colors.pop()

    components = _connected_components(g == shape_color)
    assert len(components) == wall_depth, "stripe count must match shape count"

    ranked = []
    for coords in components:
        box = _bbox(coords)
        dist = _distance_to_wall(box, side, wall_depth, h, w)
        ranked.append((dist, box, coords))
    ranked.sort(key=lambda item: (item[0], item[1]))

    out = np.full_like(g, bg)
    stripe_lines = _inner_stripe_lines(side, wall_depth, h, w)
    for rank, (_dist, _box, coords) in enumerate(ranked):
        stripe_line = stripe_lines[rank]
        stripe_color = _line_color(g, side, stripe_line)
        if side in ("top", "bottom"):
            for col in sorted({c for _, c in coords}):
                out[stripe_line, col] = stripe_color
        else:
            for row in sorted({r for r, _ in coords}):
                out[row, stripe_line] = stripe_color
    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    probs = np.full(parts, 1.0 / parts)
    return [int(x) for x in np.random.multinomial(total, probs)]


def _place_intervals(total: int, sizes: list[int], min_gap: int = 1, outer_margin: int = 1) -> list[int]:
    needed = 2 * outer_margin + sum(sizes) + min_gap * (len(sizes) - 1)
    extra = total - needed
    if extra < 0:
        raise ValueError("intervals do not fit")
    slack = _rand_composition(extra, len(sizes) + 1)
    starts: list[int] = []
    cur = outer_margin + slack[0]
    for i, size in enumerate(sizes):
        starts.append(cur)
        cur += size
        if i < len(sizes) - 1:
            cur += min_gap + slack[i + 1]
    return starts


def _paint_wall(grid: np.ndarray, side: str, wall_colors: list[int]) -> None:
    h, w = grid.shape
    _ = h, w
    if side == "top":
        for r, color in enumerate(wall_colors):
            grid[r, :] = int(color)
    elif side == "bottom":
        for offset, color in enumerate(wall_colors):
            grid[grid.shape[0] - 1 - offset, :] = int(color)
    elif side == "left":
        for c, color in enumerate(wall_colors):
            grid[:, c] = int(color)
    else:
        for offset, color in enumerate(wall_colors):
            grid[:, grid.shape[1] - 1 - offset] = int(color)


def _bg_is_dominant(grid: np.ndarray) -> bool:
    bg_count = int(np.count_nonzero(grid == BG))
    values, counts = np.unique(grid, return_counts=True)
    best_other = 0
    for value, count in zip(values, counts):
        if int(value) != BG:
            best_other = max(best_other, int(count))
    return bg_count > best_other


def _build_input(
    side: str,
    wall_colors: list[int],
    shape_color: int,
    h: int,
    w: int,
    specs: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    _paint_wall(grid, side, wall_colors)
    wall_depth = len(wall_colors)
    for spec in specs:
        height = spec["height"]
        width = spec["width"]
        distance = spec["distance"]
        if side == "top":
            top = wall_depth + distance
            left = spec["tangent_start"]
        elif side == "bottom":
            bottom = h - wall_depth - distance - 1
            top = bottom - height + 1
            left = spec["tangent_start"]
        elif side == "left":
            top = spec["tangent_start"]
            left = wall_depth + distance
        else:
            right = w - wall_depth - distance - 1
            left = right - width + 1
            top = spec["tangent_start"]
        grid[top : top + height, left : left + width] = shape_color
    return grid


def _expected_output(
    side: str,
    wall_colors: list[int],
    h: int,
    w: int,
    specs: list[dict[str, int]],
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    ranked = sorted(
        specs,
        key=lambda spec: (
            spec["distance"],
            spec["height"],
            spec["width"],
            spec["tangent_start"],
        ),
    )
    stripe_lines = _inner_stripe_lines(side, len(wall_colors), h, w)
    for rank, spec in enumerate(ranked):
        stripe_line = stripe_lines[rank]
        stripe_color = wall_colors[-1 - rank]
        if side in ("top", "bottom"):
            left = spec["tangent_start"]
            out[stripe_line, left : left + spec["width"]] = int(stripe_color)
        else:
            top = spec["tangent_start"]
            out[top : top + spec["height"], stripe_line] = int(stripe_color)
    return out


def _sample_horizontal_scene(side: str, wall_colors: list[int], shape_color: int) -> np.ndarray | None:
    count = len(wall_colors)
    widths = [random.randint(1, 4) for _ in range(count)]
    heights = [random.randint(1, 3) for _ in range(count)]
    distances = random.sample(range(1, 8), count)

    min_w = 2 + sum(widths) + (count - 1)
    if min_w > 20:
        return None
    w = random.randint(max(10, min_w), 20)

    max_depth = max(distance + height for distance, height in zip(distances, heights))
    min_h = count + max_depth + 1
    if min_h > 18:
        return None
    h = random.randint(max(10, min_h), 18)

    lefts = _place_intervals(w, widths)
    specs = []
    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        specs.append(
            {
                "distance": distances[idx],
                "height": heights[idx],
                "width": widths[idx],
                "tangent_start": lefts[idx],
            }
        )

    inp = _build_input(side, wall_colors, shape_color, h, w, specs)
    if not _bg_is_dominant(inp):
        return None
    expected = _expected_output(side, wall_colors, h, w, specs)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def _sample_vertical_scene(side: str, wall_colors: list[int], shape_color: int) -> np.ndarray | None:
    count = len(wall_colors)
    widths = [random.randint(1, 3) for _ in range(count)]
    heights = [random.randint(1, 4) for _ in range(count)]
    distances = random.sample(range(1, 8), count)

    min_h = 2 + sum(heights) + (count - 1)
    if min_h > 20:
        return None
    h = random.randint(max(10, min_h), 20)

    max_depth = max(distance + width for distance, width in zip(distances, widths))
    min_w = count + max_depth + 1
    if min_w > 18:
        return None
    w = random.randint(max(10, min_w), 18)

    tops = _place_intervals(h, heights)
    specs = []
    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        specs.append(
            {
                "distance": distances[idx],
                "height": heights[idx],
                "width": widths[idx],
                "tangent_start": tops[idx],
            }
        )

    inp = _build_input(side, wall_colors, shape_color, h, w, specs)
    if not _bg_is_dominant(inp):
        return None
    expected = _expected_output(side, wall_colors, h, w, specs)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def _generate_once() -> np.ndarray | None:
    side = random.choice(SIDES)
    count = random.randint(2, 4)
    picked = random.sample(PALETTE, count + 1)
    wall_colors = picked[:count]
    shape_color = picked[-1]

    if side in ("top", "bottom"):
        return _sample_horizontal_scene(side, wall_colors, shape_color)
    return _sample_vertical_scene(side, wall_colors, shape_color)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
