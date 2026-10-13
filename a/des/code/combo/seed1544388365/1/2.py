import random

import numpy as np


FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in FOUR_DIRS]


def _bbox(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]]
) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells), (r1 - r0 + 1, c1 - c0 + 1)


def _is_connected(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _is_irregular(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) < (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                cells.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            local_cells, shape = _normalize_cells(cells)
            components.append(
                {
                    "color": color,
                    "cells": local_cells,
                    "shape": shape,
                    "area": len(local_cells),
                }
            )
    return components


def _mask_signature(shape: tuple[int, int], cells: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    height, width = shape
    mask = [0] * (height * width)
    for r, c in cells:
        mask[r * width + c] = 1
    return tuple(mask)


def _merge_variants(comp_a: dict, comp_b: dict) -> list[dict]:
    cells_a = set(comp_a["cells"])
    cells_b = set(comp_b["cells"])
    height_a, width_a = comp_a["shape"]
    height_b, width_b = comp_b["shape"]

    variants: dict[tuple, dict] = {}
    for dr in range(-height_b, height_a + 1):
        for dc in range(-width_b, width_a + 1):
            shifted_b = {(r + dr, c + dc) for r, c in cells_b}
            if cells_a & shifted_b:
                continue
            union = cells_a | shifted_b
            min_r, min_c, max_r, max_c = _bbox(union)
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            if len(union) != height * width:
                continue

            norm_a = tuple(sorted((r - min_r, c - min_c) for r, c in cells_a))
            norm_b = tuple(sorted((r - min_r, c - min_c) for r, c in shifted_b))
            key = (height, width, norm_a, norm_b)
            variants[key] = {
                "shape": (height, width),
                "cells_a": norm_a,
                "cells_b": norm_b,
                "area": height * width,
            }

    return sorted(
        variants.values(),
        key=lambda item: (
            item["area"],
            item["shape"][0],
            item["shape"][1],
            _mask_signature(item["shape"], item["cells_a"]),
            _mask_signature(item["shape"], item["cells_b"]),
        ),
    )


def _perfect_matchings(
    count: int, edge_variants: dict[tuple[int, int], list[dict]], limit: int = 2
) -> list[tuple[tuple[int, int], ...]]:
    solutions: list[tuple[tuple[int, int], ...]] = []

    def rec(remaining: frozenset[int], picked: list[tuple[int, int]]) -> None:
        if len(solutions) >= limit:
            return
        if not remaining:
            solutions.append(tuple(picked))
            return
        i = min(remaining)
        for j in sorted(remaining):
            if j <= i or (i, j) not in edge_variants:
                continue
            picked.append((i, j))
            rec(remaining - {i, j}, picked)
            picked.pop()

    rec(frozenset(range(count)), [])
    return solutions


def _solve_components(components: list[dict]) -> tuple[list[dict], int, dict[tuple[int, int], list[dict]]]:
    assert components and len(components) % 2 == 0
    edge_variants: dict[tuple[int, int], list[dict]] = {}
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            variants = _merge_variants(components[i], components[j])
            if variants:
                edge_variants[(i, j)] = variants

    solutions = _perfect_matchings(len(components), edge_variants, limit=2)
    assert solutions, "no complementary pairing found"
    chosen = solutions[0]

    pair_infos: list[dict] = []
    for i, j in chosen:
        variant = edge_variants[(i, j)][0]
        height, width = variant["shape"]
        sprite = np.empty((height, width), dtype=int)
        for r, c in variant["cells_a"]:
            sprite[r, c] = int(components[i]["color"])
        for r, c in variant["cells_b"]:
            sprite[r, c] = int(components[j]["color"])
        pair_infos.append(
            {
                "shape": variant["shape"],
                "area": int(variant["area"]),
                "sprite": sprite,
                "variant_count": len(edge_variants[(i, j)]),
                "signature": (
                    _mask_signature(variant["shape"], variant["cells_a"]),
                    _mask_signature(variant["shape"], variant["cells_b"]),
                ),
            }
        )

    pair_infos.sort(
        key=lambda item: (
            item["area"],
            item["shape"][0],
            item["shape"][1],
            item["signature"][0],
            item["signature"][1],
        )
    )
    return pair_infos, len(solutions), edge_variants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary monochrome fragments, rebuild the full rectangles,
    sort the rectangles by area, and pack them left-to-right along the top.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components = _find_components(g, bg)
    pair_infos, _solution_count, _edge_variants = _solve_components(components)

    out_height = max(info["shape"][0] for info in pair_infos)
    out_width = sum(info["shape"][1] for info in pair_infos) + len(pair_infos) - 1
    out = np.full((out_height, out_width), bg, dtype=int)

    col = 0
    for info in pair_infos:
        height, width = info["shape"]
        out[:height, col : col + width] = info["sprite"]
        col += width + 1
    return out


RECT_SPECS = [(2, 3), (2, 4), (3, 3), (2, 5), (3, 4), (3, 5), (4, 4), (4, 5)]
PAIR_COUNT_OPTIONS = (2, 3, 4)
CANVAS_HEIGHT_RANGE = (18, 26)
CANVAS_WIDTH_RANGE = (18, 26)
SPLIT_TRIES = 300
PLACEMENT_TRIES = 300
GEN_RETRIES = 500


def _flip_cells(
    cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]], height: int, width: int, rng: random.Random
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    if rng.random() < 0.5:
        cells_a = {(r, width - 1 - c) for r, c in cells_a}
        cells_b = {(r, width - 1 - c) for r, c in cells_b}
    if rng.random() < 0.5:
        cells_a = {(height - 1 - r, c) for r, c in cells_a}
        cells_b = {(height - 1 - r, c) for r, c in cells_b}
    return cells_a, cells_b


def _sample_partition(height: int, width: int, rng: random.Random) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    full = {(r, c) for r in range(height) for c in range(width)}
    area = height * width
    for _ in range(SPLIT_TRIES):
        target = rng.randint(2, area - 2)
        cells_a = {(0, 0)}
        while len(cells_a) < target:
            frontier = set()
            for r, c in cells_a:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells_a:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells_a.add(rng.choice(tuple(frontier)))

        if len(cells_a) != target or (height - 1, width - 1) in cells_a:
            continue
        cells_b = full - cells_a
        if not _is_connected(cells_a) or not _is_connected(cells_b):
            continue
        if not _is_irregular(cells_a) or not _is_irregular(cells_b):
            continue

        cells_a, cells_b = _flip_cells(cells_a, cells_b, height, width, rng)
        return cells_a, cells_b
    return None


def _sprite_from_cells(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]], color: int, bg: int
) -> np.ndarray:
    norm_cells, (height, width) = _normalize_cells(cells)
    sprite = np.full((height, width), bg, dtype=int)
    for r, c in norm_cells:
        sprite[r, c] = color
    return sprite


def _rectangle_sprite(
    height: int, width: int, cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]], color_a: int, color_b: int
) -> np.ndarray:
    sprite = np.empty((height, width), dtype=int)
    for r, c in cells_a:
        sprite[r, c] = color_a
    for r, c in cells_b:
        sprite[r, c] = color_b
    return sprite


def _can_place(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> bool:
    height, width = sprite.shape
    g_height, g_width = grid.shape
    if top + height > g_height or left + width > g_width:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(g_height, top + height + 1)
    c1 = min(g_width, left + width + 1)
    return bool(np.all(grid[r0:r1, c0:c1] == bg))


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = sprite != bg
    window = grid[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    window[mask] = sprite[mask]


def _intended_output(rectangles: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: (item["area"], item["shape"][0], item["shape"][1]))
    out_height = max(item["shape"][0] for item in ordered)
    out_width = sum(item["shape"][1] for item in ordered) + len(ordered) - 1
    out = np.full((out_height, out_width), bg, dtype=int)
    col = 0
    for item in ordered:
        height, width = item["shape"]
        out[:height, col : col + width] = item["sprite"]
        col += width + 1
    return out


def _sample_rectangles(rng: random.Random, bg: int) -> tuple[list[dict], list[np.ndarray]] | None:
    n_pairs = rng.choice(PAIR_COUNT_OPTIONS)
    specs = rng.sample(RECT_SPECS, n_pairs)
    palette = rng.sample([c for c in range(10) if c != bg], 2 * n_pairs)

    rectangles: list[dict] = []
    piece_sprites: list[np.ndarray] = []

    for idx, (base_h, base_w) in enumerate(specs):
        height, width = (base_w, base_h) if base_h != base_w and rng.random() < 0.5 else (base_h, base_w)
        split = _sample_partition(height, width, rng)
        if split is None:
            return None
        cells_a, cells_b = split
        color_a = int(palette[2 * idx])
        color_b = int(palette[2 * idx + 1])
        piece_sprites.append(_sprite_from_cells(cells_a, color_a, bg))
        piece_sprites.append(_sprite_from_cells(cells_b, color_b, bg))
        rectangles.append(
            {
                "shape": (height, width),
                "area": height * width,
                "sprite": _rectangle_sprite(height, width, cells_a, cells_b, color_a, color_b),
            }
        )

    areas = [item["area"] for item in rectangles]
    if len(set(areas)) != len(areas):
        return None
    return rectangles, piece_sprites


def _place_pieces(piece_sprites: list[np.ndarray], bg: int, rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*CANVAS_HEIGHT_RANGE)
    width = rng.randint(*CANVAS_WIDTH_RANGE)
    grid = np.full((height, width), bg, dtype=int)

    sprites = piece_sprites[:]
    rng.shuffle(sprites)
    for sprite in sprites:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = rng.randint(0, height - sprite.shape[0])
            left = rng.randint(0, width - sprite.shape[1])
            if _can_place(grid, sprite, top, left, bg):
                _paste_sprite(grid, sprite, top, left, bg)
                placed = True
                break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randint(0, 9)
    sampled = _sample_rectangles(rng, bg)
    if sampled is None:
        return None
    rectangles, piece_sprites = sampled
    grid = _place_pieces(piece_sprites, bg, rng)
    if grid is None:
        return None
    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return None

    components = _find_components(grid, bg)
    try:
        pair_infos, solution_count, edge_variants = _solve_components(components)
    except AssertionError:
        return None
    if solution_count != 1:
        return None
    if any(info["variant_count"] != 1 for info in pair_infos):
        return None
    if not all(len(variants) == 1 for variants in edge_variants.values()):
        return None

    expected = _intended_output(rectangles, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
