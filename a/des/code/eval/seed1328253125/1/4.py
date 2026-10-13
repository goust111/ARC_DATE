from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
RING_COLORS = [1, 3, 4, 6, 7, 8, 9]

# --------------- generator config ---------------

GRID_LIMIT = 30
OBJECT_COUNT_RANGE = (2, 4)  # inclusive
RING_SIZE_RANGE = (7, 13)    # inclusive
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _draw_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _max_inner_layers(height: int, width: int) -> int:
    layers = 0
    while height - 2 * (layers + 1) >= 3 and width - 2 * (layers + 1) >= 3:
        layers += 1
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red seeds inside each rectangular ring, then draw inward-shrunk
    concentric rings with alternating original/grey colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    ring_mask = (g != BLACK) & (g != RED)

    for comp in _components(ring_mask):
        color = int(g[comp[0]])
        assert all(int(g[r, c]) == color for r, c in comp), "each ring must be monochrome"

        r0, c0, r1, c1 = _bbox(comp)
        expected = set(_border_cells(r0, c0, r1, c1))
        assert set(comp) == expected, "foreground components must be hollow rectangle borders"

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_count = int(np.count_nonzero(inner == RED))

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        max_inner = _max_inner_layers(height, width)
        for layer in range(min(seed_count, max_inner) + 1):
            draw_color = color if layer % 2 == 0 else GREY
            _draw_border(out, r0 + layer, c0 + layer, r1 - layer, c1 - layer, draw_color)

    return out


def _render_output(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        height = int(spec["height"])
        width = int(spec["width"])
        color = int(spec["color"])
        seed_count = int(spec["seed_count"])
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        max_inner = _max_inner_layers(height, width)
        for layer in range(min(seed_count, max_inner) + 1):
            draw_color = color if layer % 2 == 0 else GREY
            _draw_border(out, r0 + layer, c0 + layer, r1 - layer, c1 - layer, draw_color)
    return out


def _build_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        height = int(spec["height"])
        width = int(spec["width"])
        color = int(spec["color"])
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        _draw_border(inp, r0, c0, r1, c1, color)
        for r, c in spec["seeds"]:
            inp[int(r), int(c)] = RED
    return inp


def _sample_seed_counts(num_objects: int) -> list[int]:
    counts = [1, 2, 3]
    random.shuffle(counts)
    counts = counts[: min(num_objects, 3)]
    while len(counts) < num_objects:
        counts.append(random.randint(1, 3))
    random.shuffle(counts)
    return counts


def _sample_ring_size(seed_count: int) -> tuple[int, int]:
    if seed_count == 3 and random.random() < 0.5:
        if random.random() < 0.5:
            return random.choice([7, 8]), random.randint(*RING_SIZE_RANGE)
        return random.randint(*RING_SIZE_RANGE), random.choice([7, 8])
    return random.randint(*RING_SIZE_RANGE), random.randint(*RING_SIZE_RANGE)


def _grid_bounds(specs: list[dict[str, object]]) -> tuple[int, int]:
    num_objects = len(specs)
    max_h = max(int(spec["height"]) for spec in specs)
    max_w = max(int(spec["width"]) for spec in specs)
    if num_objects == 2:
        lo = 20
    elif num_objects == 3:
        lo = 24
    else:
        lo = 28
    h_lo = max(lo, max_h + 4)
    w_lo = max(lo, max_w + 4)
    return h_lo, w_lo


def _place_specs(specs: list[dict[str, object]], shape: tuple[int, int]) -> bool:
    height, width = shape
    occupied = np.zeros(shape, dtype=bool)

    for spec in sorted(specs, key=lambda s: int(s["height"]) * int(s["width"]), reverse=True):
        ring_h = int(spec["height"])
        ring_w = int(spec["width"])
        if height - ring_h - 1 < 1 or width - ring_w - 1 < 1:
            return False

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, height - ring_h - 1)
            c0 = random.randint(1, width - ring_w - 1)
            if bool(occupied[r0 - 1 : r0 + ring_h + 1, c0 - 1 : c0 + ring_w + 1].any()):
                continue
            occupied[r0 - 1 : r0 + ring_h + 1, c0 - 1 : c0 + ring_w + 1] = True
            spec["r0"] = r0
            spec["c0"] = c0
            placed = True
            break
        if not placed:
            return False
    return True


def _choose_seed_positions(r0: int, c0: int, height: int, width: int, seed_count: int) -> list[tuple[int, int]]:
    interior = [(r, c) for r in range(r0 + 1, r0 + height - 1) for c in range(c0 + 1, c0 + width - 1)]
    random.shuffle(interior)
    return interior[:seed_count]


def _gen_attempt() -> np.ndarray | None:
    num_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(RING_COLORS, num_objects)
    seed_counts = _sample_seed_counts(num_objects)

    specs: list[dict[str, object]] = []
    for color, seed_count in zip(colors, seed_counts, strict=False):
        ring_h, ring_w = _sample_ring_size(seed_count)
        specs.append(
            {
                "color": color,
                "seed_count": seed_count,
                "height": ring_h,
                "width": ring_w,
            }
        )

    h_lo, w_lo = _grid_bounds(specs)
    if h_lo > GRID_LIMIT or w_lo > GRID_LIMIT:
        return None

    grid_h = random.randint(h_lo, GRID_LIMIT)
    grid_w = random.randint(w_lo, GRID_LIMIT)
    if not _place_specs(specs, (grid_h, grid_w)):
        return None

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        spec["seeds"] = _choose_seed_positions(r0, c0, int(spec["height"]), int(spec["width"]), int(spec["seed_count"]))

    inp = _build_input((grid_h, grid_w), specs)
    out = _render_output((grid_h, grid_w), specs)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
