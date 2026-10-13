from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
BLUE = 1
RED = 2
GREY = 5

AREA_CHOICES = {
    0: (2, 4, 6, 8),
    1: (1, 3, 5, 7, 9),
}
MAX_SHAPE_SPAN = 5
MIN_FILL_RATIO = 0.35
MAX_SHAPE_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, np.ndarray | int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, np.ndarray | int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                }
            )

    return components


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    h, w = padded.shape
    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if padded[nr, nc] or seen[nr, nc]:
                continue
            seen[nr, nc] = True
            queue.append((nr, nc))

    return seen[1:-1, 1:-1]


def _component_border(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())

    rr0 = max(0, r0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w - 1, c1 + 1)

    local = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
    local[coords[:, 0] - rr0, coords[:, 1] - cc0] = True

    exterior = _exterior_background(local)
    border = _dilate8(local) & exterior & ~local
    return np.argwhere(border) + np.array([rr0, cc0], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a one-pixel outer ring around each foreground object, colored by the object's area parity.

    Input structure:
    - The background is Azure.
    - The foreground contains several separated 4-connected Grey objects.
    - Objects have varied shapes and sizes, with enough spacing that their outer rings do not overlap.

    Rule:
    - Identify each foreground object independently and count its pixels.
    - If an object's area is odd, color its one-pixel 8-neighborhood outer ring Red.
    - If an object's area is even, color its one-pixel 8-neighborhood outer ring Blue.
    - Keep every original object cell unchanged.

    Key insight:
    - The border color is not determined by visible shape class, but by an implicit per-object pixel count whose parity selects the ring color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    out = g.copy()

    for comp in _find_components(g, bg):
        coords = np.asarray(comp["coords"], dtype=int)
        border_color = RED if int(coords.shape[0]) % 2 == 1 else BLUE
        border = _component_border(coords, g.shape)
        for r, c in border:
            if int(g[int(r), int(c)]) == bg:
                out[int(r), int(c)] = border_color

    return out


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(r - min_r, c - min_c) for r, c in cells}
    max_r = max(r for r, _ in shifted)
    max_c = max(c for _, c in shifted)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _has_hole(mask: np.ndarray) -> bool:
    exterior = _exterior_background(mask)
    return bool((~mask & ~exterior).any())


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    frontier.add((r + dr, c + dc))
            frontier.difference_update(cells)
            cells.add(random.choice(tuple(frontier)))

        mask = _mask_from_cells(cells)
        mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.array(mask, dtype=bool, copy=True)

        if max(mask.shape) > MAX_SHAPE_SPAN:
            continue
        if area / float(mask.shape[0] * mask.shape[1]) < MIN_FILL_RATIO:
            continue
        if _has_hole(mask):
            continue
        return mask
    return None


def _sample_shapes(count: int) -> list[np.ndarray] | None:
    parities = [1, 0] + [random.randint(0, 1) for _ in range(count - 2)]
    random.shuffle(parities)
    shapes: list[np.ndarray] = []

    for parity in parities:
        area = random.choice(AREA_CHOICES[int(parity)])
        shape = _random_polyomino(int(area))
        if shape is None:
            return None
        shapes.append(shape)

    random.shuffle(shapes)
    shapes.sort(key=lambda mask: (mask.shape[0] + 2) * (mask.shape[1] + 2), reverse=True)
    return shapes


def _choose_grid_shape(shapes: list[np.ndarray]) -> tuple[int, int]:
    halo_areas = [(mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in shapes]
    max_halo_h = max(int(mask.shape[0]) + 2 for mask in shapes)
    max_halo_w = max(int(mask.shape[1]) + 2 for mask in shapes)
    base = max(
        12,
        max_halo_h + random.randint(1, 4),
        max_halo_w + random.randint(1, 4),
        int(np.ceil(np.sqrt(sum(halo_areas) * random.uniform(1.8, 2.4)))),
    )
    h = random.randint(base, min(30, base + 6))
    w = random.randint(base, min(30, base + 8))
    return h, w


def _place_shapes(shapes: list[np.ndarray], h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), AZURE, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for shape in shapes:
        halo = _dilate8(np.pad(shape, 1, constant_values=False))
        hh, hw = halo.shape
        positions = [(r, c) for r in range(h - hh + 1) for c in range(w - hw + 1)]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            window = reserved[r0 : r0 + hh, c0 : c0 + hw]
            if bool((window & halo).any()):
                continue

            reserved[r0 : r0 + hh, c0 : c0 + hw] |= halo
            obj = grid[r0 + 1 : r0 + 1 + shape.shape[0], c0 + 1 : c0 + 1 + shape.shape[1]]
            obj[shape] = GREY
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 5)
    shapes = _sample_shapes(count)
    if shapes is None:
        return None

    h, w = _choose_grid_shape(shapes)
    inp = _place_shapes(shapes, h, w)
    if inp is None:
        return None

    out = main(inp)
    if not (out == RED).any() or not (out == BLUE).any():
        return None
    if int((inp == GREY).sum()) != sum(int(shape.sum()) for shape in shapes):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
