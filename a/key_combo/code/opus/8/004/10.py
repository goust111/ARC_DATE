from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

SEED_COLORS = (RED, GREEN, YELLOW, PINK)
SHAPE = (8, 8)

CORNER_POSITIONS = {
    "tl": (0, 0),
    "tr": (0, SHAPE[1] - 1),
    "bl": (SHAPE[0] - 1, 0),
    "br": (SHAPE[0] - 1, SHAPE[1] - 1),
}


def _template_cells(name: str, shape: tuple[int, int]) -> set[tuple[int, int]]:
    rows, cols = shape
    if name == "tl":
        return {(1, 1), (1, 2), (2, 1), (2, 2)}
    if name == "tr":
        return {(1, cols - 3), (1, cols - 2), (2, cols - 3), (2, cols - 2)}
    if name == "bl":
        return {(rows - 2, 0), (rows - 2, 1), (rows - 3, 0)}
    if name == "br":
        return {(rows - 2, cols - 1), (rows - 2, cols - 2), (rows - 3, cols - 1)}
    raise ValueError(f"unknown corner {name}")


def _neighbors8(r: int, c: int, shape: tuple[int, int]):
    rows, cols = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Expand corner seeds into fixed templates, then recolor large 8-connected foreground components teal.

    Input structure:
    - The input is an 8x8 grid with black background.
    - One to three corners contain colored seed pixels, and the interior contains isolated gray pixels.
    - Corner seeds use non-black colors, while gray pixels may later connect to the expanded corner templates.

    Rule:
    - Remove each original corner seed and replace it with the fixed template assigned to that corner, using the seed color.
    - Treat all non-background pixels after expansion as 8-connected foreground components.
    - Recolor every component with more than four pixels to teal; keep components of size four or less unchanged.

    Key insight:
    - The output depends on simulating the corner template expansion first, because that expansion can absorb nearby gray pixels and change component sizes before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for name, pos in CORNER_POSITIONS.items():
        color = int(out[pos])
        if color == BG:
            continue
        out[pos] = BG
        for r, c in _template_cells(name, out.shape):
            out[r, c] = color

    for component in _connected_components(out != BG):
        if len(component) > 4:
            for r, c in component:
                out[r, c] = TEAL

    return out


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _interior_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = shape
    return [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]


def _attachment_candidates(corner: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    template = _template_cells(corner, shape)
    candidates = []
    for r, c in _interior_cells(shape):
        if (r, c) in template:
            continue
        if any(_chebyshev((r, c), cell) == 1 for cell in template):
            candidates.append((r, c))
    return candidates


def _far_candidates(corners: list[str], shape: tuple[int, int]) -> list[tuple[int, int]]:
    templates = set().union(*(_template_cells(name, shape) for name in corners))
    candidates = []
    for r, c in _interior_cells(shape):
        if any(_chebyshev((r, c), cell) <= 1 for cell in templates):
            continue
        candidates.append((r, c))
    return candidates


def _is_input_gray_legal(
    pos: tuple[int, int],
    chosen: list[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
) -> bool:
    if any(_chebyshev(pos, other) <= 1 for other in chosen):
        return False
    if any(_chebyshev(pos, seed) <= 1 for seed in seed_positions):
        return False
    return True


def _choose_positions(
    candidates: list[tuple[int, int]],
    count: int,
    chosen: list[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    shuffled = list(candidates)
    random.shuffle(shuffled)
    result = list(chosen)
    picked: list[tuple[int, int]] = []

    def backtrack(index: int) -> bool:
        if len(picked) == count:
            return True
        if index >= len(shuffled):
            return False
        remaining = len(shuffled) - index
        if len(picked) + remaining < count:
            return False

        pos = shuffled[index]
        if _is_input_gray_legal(pos, result, seed_positions):
            result.append(pos)
            picked.append(pos)
            if backtrack(index + 1):
                return True
            picked.pop()
            result.pop()
        return backtrack(index + 1)

    if backtrack(0):
        return picked
    return None


def _input_constraints_hold(grid: np.ndarray, corners: list[str]) -> bool:
    gray_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == GREY)]
    if not 3 <= len(gray_cells) <= 6:
        return False
    if not 1 <= len(corners) <= 3:
        return False
    for corner in corners:
        if int(grid[CORNER_POSITIONS[corner]]) not in SEED_COLORS:
            return False
    for r, c in gray_cells:
        for nr, nc in _neighbors8(r, c, grid.shape):
            if grid[nr, nc] != BG:
                return False
    return True


def _output_has_both_size_branches(output: np.ndarray) -> bool:
    sizes = [len(component) for component in _connected_components(output != BG)]
    return any(size > 4 for size in sizes) and any(size <= 4 for size in sizes)


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = np.zeros(SHAPE, dtype=int)
        corner_count = random.choice((1, 2, 2, 3, 3))
        corners = random.sample(list(CORNER_POSITIONS), corner_count)
        corner_colors = random.sample(SEED_COLORS, corner_count)
        seed_positions = [CORNER_POSITIONS[name] for name in corners]

        for name, color in zip(corners, corner_colors):
            grid[CORNER_POSITIONS[name]] = color

        large_corner = random.choice(corners)
        large_size_needed = 1 if len(_template_cells(large_corner, SHAPE)) == 4 else 2
        chosen_grays: list[tuple[int, int]] = []

        attached = _choose_positions(
            _attachment_candidates(large_corner, SHAPE),
            large_size_needed,
            chosen_grays,
            seed_positions,
        )
        if attached is None:
            continue
        chosen_grays.extend(attached)

        far = _choose_positions(
            _far_candidates(corners, SHAPE),
            1,
            chosen_grays,
            seed_positions,
        )
        if far is None:
            continue
        chosen_grays.extend(far)

        gray_total = random.randint(max(3, len(chosen_grays)), 6)
        attach_pool = sorted(set().union(*(_attachment_candidates(name, SHAPE) for name in corners)))
        far_pool = _far_candidates(corners, SHAPE)
        while len(chosen_grays) < gray_total:
            use_attach = random.random() < 0.6
            pool = attach_pool if use_attach else far_pool
            options = [
                pos
                for pos in pool
                if pos not in chosen_grays and _is_input_gray_legal(pos, chosen_grays, seed_positions)
            ]
            if not options:
                other_pool = far_pool if use_attach else attach_pool
                options = [
                    pos
                    for pos in other_pool
                    if pos not in chosen_grays and _is_input_gray_legal(pos, chosen_grays, seed_positions)
                ]
            if not options:
                break
            chosen_grays.append(random.choice(options))

        if len(chosen_grays) != gray_total:
            continue

        for r, c in chosen_grays:
            grid[r, c] = GREY

        if not _input_constraints_hold(grid, corners):
            continue

        output = main(grid)
        if not _output_has_both_size_branches(output):
            continue

        return grid

    raise RuntimeError("failed to generate a valid corner-template component-size puzzle")
