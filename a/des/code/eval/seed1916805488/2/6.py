from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 9
PALETTE = tuple(range(1, 9))
GEN_RETRIES = 400


def _row_color(row: np.ndarray) -> int | None:
    vals = np.unique(row)
    if vals.size != 1:
        return None
    return int(vals[0])


def _interior_layers(interior: np.ndarray) -> list[int] | None:
    if interior.size == 0:
        return None

    row_colors: list[int] = []
    for row in interior:
        color = _row_color(row)
        if color is None or color == WALL:
            return None
        row_colors.append(color)

    first_filled = None
    for idx, color in enumerate(row_colors):
        if color != BG:
            first_filled = idx
            break
    if first_filled is None:
        return None

    filled = row_colors[first_filled:]
    if any(color == BG for color in filled):
        return None
    if not 2 <= len(filled) <= 4:
        return None
    if len(set(filled)) != len(filled):
        return None
    return filled


def _find_containers(grid: np.ndarray) -> list[tuple[int, int, int, int, list[int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    found: dict[tuple[int, int, int, int], tuple[int, int, int, int, list[int]]] = {}

    for r0 in range(height - 2):
        for c0 in range(width - 2):
            if g[r0, c0] == WALL:
                continue
            for r1 in range(r0 + 2, height):
                if g[r1, c0] != WALL:
                    continue
                for c1 in range(c0 + 2, width):
                    if g[r1, c1] != WALL:
                        continue

                    if np.any(g[r0, c0 : c1 + 1] == WALL):
                        continue
                    if not np.all(g[r0 + 1 : r1 + 1, c0] == WALL):
                        continue
                    if not np.all(g[r0 + 1 : r1 + 1, c1] == WALL):
                        continue
                    if not np.all(g[r1, c0 : c1 + 1] == WALL):
                        continue

                    interior = g[r0 + 1 : r1, c0 + 1 : c1]
                    layers = _interior_layers(interior)
                    if layers is None:
                        continue

                    found[(r0, c0, r1, c1)] = (r0, c0, r1, c1, layers)

    containers = list(found.values())
    assert containers, "expected at least one container"
    return containers


def _find_recipe_column(
    grid: np.ndarray, r0: int, c0: int, c1: int, colors: set[int]
) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, list[int]]] = []

    for c in range(c0 + 1, c1):
        if r0 - 1 < 0 or g[r0 - 1, c] != BG:
            continue
        top = r0 - 2
        if top < 0 or g[top, c] in (BG, WALL):
            continue

        seq = []
        rr = top
        while rr >= 0 and g[rr, c] not in (BG, WALL):
            seq.append(int(g[rr, c]))
            rr -= 1
        seq.reverse()
        if set(seq) == colors and len(seq) == len(colors):
            candidates.append((c, seq))

    assert len(candidates) == 1, "expected exactly one recipe card per container"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder sediment layers inside open-top containers using hanging recipe cards.

    Input structure
    - Background is black.
    - Maroon pixels form axis-aligned open-top rectangular containers.
    - Each container interior has 2-4 full-width colored rows stacked at the bottom.
    - One row above each container is a black gap, then a one-pixel-wide vertical
      recipe card whose colors are a permutation of the sediment colors.

    Rule
    - Pair each recipe card to the nearest container below via column overlap.
    - Read the card from top to bottom as the desired layer order from bottom to top.
    - Remove the card and restack the full-width sediment rows tightly against the bottom.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r0, c0, r1, c1, layers in _find_containers(g):
        recipe_col, recipe = _find_recipe_column(g, r0, c0, c1, set(layers))

        top = r0 - 2
        while top >= 0 and g[top, recipe_col] not in (BG, WALL):
            out[top, recipe_col] = BG
            top -= 1

        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        for depth, color in enumerate(recipe):
            out[r1 - 1 - depth, c0 + 1 : c1] = int(color)

    return out


def _sample_recipe(colors: list[int]) -> list[int]:
    if len(colors) == 2:
        return [colors[1], colors[0]]
    recipe = colors[:]
    while True:
        random.shuffle(recipe)
        if recipe != colors:
            return recipe[:]


def _draw_case(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    sediment: list[int],
    recipe: list[int],
    recipe_col_offset: int,
) -> None:
    layers = len(sediment)
    r0 = top + layers + 1
    r1 = r0 + inner_h
    c0 = left
    c1 = c0 + inner_w + 1

    grid[r0 + 1 : r1 + 1, c0] = WALL
    grid[r0 + 1 : r1 + 1, c1] = WALL
    grid[r1, c0 : c1 + 1] = WALL

    for idx, color in enumerate(reversed(sediment)):
        grid[r1 - 1 - idx, c0 + 1 : c1] = int(color)

    card_col = c0 + 1 + recipe_col_offset
    card_top = top
    for idx, color in enumerate(recipe):
        grid[card_top + idx, card_col] = int(color)


def _occupied(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 < or0 or or1 < r0 or c1 < oc0 or oc1 < c0):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 28)
    width = random.randint(16, 30)
    grid = np.full((height, width), BG, dtype=int)

    count_choices = [1, 2, 3, 4]
    weights = [1, 3, 3, 1]
    n_containers = random.choices(count_choices, weights=weights, k=1)[0]

    specs = []
    for _ in range(n_containers):
        layer_count = random.randint(2, 4)
        extra_space = random.randint(0, 3)
        inner_h = layer_count + extra_space
        inner_w = random.randint(2, 5)
        colors = random.sample(PALETTE, layer_count)
        recipe = _sample_recipe(colors)
        recipe_col_offset = random.randint(0, inner_w - 1)
        total_h = layer_count + inner_h + 3
        total_w = inner_w + 2
        specs.append(
            {
                "layers": layer_count,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "sediment": colors,
                "recipe": recipe,
                "recipe_col_offset": recipe_col_offset,
                "total_h": total_h,
                "total_w": total_w,
            }
        )

    # Place larger items first to reduce rejection.
    specs.sort(key=lambda item: (item["total_h"] * item["total_w"]), reverse=True)
    inflated_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int, dict]] = []

    for spec in specs:
        placed = False
        for _ in range(200):
            top = random.randint(0, height - spec["total_h"])
            left = random.randint(0, width - spec["total_w"])
            box = (
                max(0, top - 1),
                max(0, left - 1),
                min(height - 1, top + spec["total_h"]),
                min(width - 1, left + spec["total_w"]),
            )
            if _occupied(box, inflated_boxes):
                continue
            inflated_boxes.append(box)
            placements.append((top, left, spec))
            placed = True
            break
        if not placed:
            return None

    for top, left, spec in placements:
        _draw_case(
            grid,
            top=top,
            left=left,
            inner_h=int(spec["inner_h"]),
            inner_w=int(spec["inner_w"]),
            sediment=list(spec["sediment"]),
            recipe=list(spec["recipe"]),
            recipe_col_offset=int(spec["recipe_col_offset"]),
        )

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
