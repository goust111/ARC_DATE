from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
SEED_COLORS = (2, 3, 4)  # red, green, yellow

CANVAS_LIMIT = 30
GEN_RETRIES = 400
PLACE_RETRIES = 250
RING_COUNT_OPTIONS = (1, 2, 3, 4)
RING_COUNT_WEIGHTS = (1, 3, 3, 1)
INNER_SIZE_RANGE = range(1, 8)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _clockwise_border(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    border: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        border.append((r0, c))
    for r in range(r0 + 1, r1):
        border.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        border.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        border.append((r, c0))
    return border


def _ring_from_component(grid: np.ndarray, component: list[tuple[int, int]]) -> dict[str, object]:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("blue component is too small to be a rectangular ring")

    border = _clockwise_border(r0, c0, r1, c1)
    if set(component) != set(border):
        raise ValueError("blue component is not a single-pixel rectangle ring")

    seeds: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            value = int(grid[r, c])
            if value == BLACK:
                continue
            if value not in SEED_COLORS:
                raise ValueError("ring interior contains an unexpected color")
            seeds.append((r, c))
    if len(seeds) != 1:
        raise ValueError("each ring interior must contain exactly one seed pixel")

    sr, sc = seeds[0]
    return {
        "bbox": (r0, c0, r1, c1),
        "border": border,
        "seed": (sr, sc),
        "seed_color": int(grid[sr, sc]),
        "area": (r1 - r0 - 1) * (c1 - c0 - 1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each blue rectangle ring clockwise from the seed's nearest border point by a length equal to the ring's interior area.

    Input structure:
    - The background is black.
    - One or more non-touching blue single-pixel rectangular rings appear on the grid.
    - Each ring contains exactly one interior seed pixel colored red, green, or yellow.

    Rule:
    - For each ring, compute the area A of its interior empty rectangle, excluding the blue border itself.
    - Find the border pixel with minimum Manhattan distance to the seed; break ties by choosing the topmost such pixel, then the leftmost.
    - Starting from that border pixel, walk clockwise around the ring and recolor A consecutive border pixels with the seed color.
    - If A is at least the ring's border length, recolor the entire border; keep the seed pixel unchanged and leave all other cells as they are.

    Key insight:
    - The interior size is a hidden numeric control for how much of the border changes, while the seed position determines the unique clockwise starting point through nearest-boundary matching.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rings = [_ring_from_component(g, comp) for comp in _components_of_color(g, BLUE)]
    for ring in sorted(rings, key=lambda item: item["bbox"]):  # deterministic only
        border = ring["border"]
        sr, sc = ring["seed"]
        seed_color = ring["seed_color"]
        area = int(ring["area"])

        start = min(border, key=lambda pos: (abs(pos[0] - sr) + abs(pos[1] - sc), pos[0], pos[1]))
        start_idx = border.index(start)
        count = min(area, len(border))

        for offset in range(count):
            r, c = border[(start_idx + offset) % len(border)]
            out[r, c] = seed_color

    return out


PARTIAL_SIZES = [
    (h, w)
    for h in INNER_SIZE_RANGE
    for w in INNER_SIZE_RANGE
    if h * w < 2 * (h + w) + 4
]
FULL_SIZES = [
    (h, w)
    for h in INNER_SIZE_RANGE
    for w in INNER_SIZE_RANGE
    if h * w >= 2 * (h + w) + 4
]


def _seed_mode_buckets(inner_h: int, inner_w: int) -> dict[str, list[tuple[int, int]]]:
    buckets = {"unique": [], "double_tie": [], "multi_tie": []}
    for rr in range(1, inner_h + 1):
        for cc in range(1, inner_w + 1):
            distances = (
                rr,
                inner_h + 1 - rr,
                cc,
                inner_w + 1 - cc,
            )
            tie_count = sum(dist == min(distances) for dist in distances)
            if tie_count == 1:
                buckets["unique"].append((rr, cc))
            elif tie_count == 2:
                buckets["double_tie"].append((rr, cc))
            else:
                buckets["multi_tie"].append((rr, cc))
    return buckets


def _choose_seed_cell(inner_h: int, inner_w: int, rng: random.Random) -> tuple[int, int]:
    buckets = _seed_mode_buckets(inner_h, inner_w)
    available = [mode for mode, cells in buckets.items() if cells]
    chosen_mode = rng.choice(available)
    return rng.choice(buckets[chosen_mode])


def _sample_branch_kinds(count: int, rng: random.Random) -> list[str]:
    if count == 1:
        return [rng.choice(("partial", "full"))]

    kinds: list[str] = []
    first = rng.choice(("partial", "full"))
    second = "full" if first == "partial" else "partial"
    for idx in range(count):
        kinds.append(first if idx % 2 == 0 else second)
    rng.shuffle(kinds)
    return kinds


def _sample_ring_specs(rng: random.Random) -> list[dict[str, object]]:
    ring_count = rng.choices(RING_COUNT_OPTIONS, weights=RING_COUNT_WEIGHTS, k=1)[0]
    branch_kinds = _sample_branch_kinds(ring_count, rng)
    used_sizes: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []

    for kind in branch_kinds:
        pool = FULL_SIZES if kind == "full" else PARTIAL_SIZES
        candidates = [size for size in pool if size not in used_sizes]
        if not candidates:
            raise RuntimeError("not enough distinct sizes for generation")
        inner_h, inner_w = rng.choice(candidates)
        used_sizes.add((inner_h, inner_w))
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_color": rng.choice(SEED_COLORS),
            }
        )

    rng.shuffle(specs)
    return specs


def _place_specs(specs: list[dict[str, object]], rng: random.Random) -> list[dict[str, object]] | None:
    occupied = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=bool)
    placed: list[dict[str, object]] = []

    for spec in specs:
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if outer_h > CANVAS_LIMIT - 2 or outer_w > CANVAS_LIMIT - 2:
            return None

        for _ in range(PLACE_RETRIES):
            r0 = rng.randint(1, CANVAS_LIMIT - outer_h - 1)
            c0 = rng.randint(1, CANVAS_LIMIT - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue

            seed_rr, seed_cc = _choose_seed_cell(inner_h, inner_w, rng)
            placed.append(
                {
                    **spec,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "seed_r": r0 + seed_rr,
                    "seed_c": c0 + seed_cc,
                }
            )
            occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            break
        else:
            return None

    return placed


def _render_grid(placed_specs: list[dict[str, object]], rng: random.Random) -> np.ndarray:
    grid = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=int)

    for spec in placed_specs:
        r0, c0, r1, c1 = int(spec["r0"]), int(spec["c0"]), int(spec["r1"]), int(spec["c1"])
        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
        grid[int(spec["seed_r"]), int(spec["seed_c"])] = int(spec["seed_color"])

    coords = np.argwhere(grid != BLACK)
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    top_margin = rng.randint(0, min(3, int(rmin)))
    left_margin = rng.randint(0, min(3, int(cmin)))
    bottom_margin = rng.randint(0, min(3, CANVAS_LIMIT - 1 - int(rmax)))
    right_margin = rng.randint(0, min(3, CANVAS_LIMIT - 1 - int(cmax)))

    return grid[
        rmin - top_margin : rmax + bottom_margin + 1,
        cmin - left_margin : cmax + right_margin + 1,
    ].copy()


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > CANVAS_LIMIT:
        return False

    if not set(np.unique(grid)).issubset({BLACK, BLUE, *SEED_COLORS}):
        return False

    try:
        rings = [_ring_from_component(grid, comp) for comp in _components_of_color(grid, BLUE)]
    except ValueError:
        return False

    if not rings:
        return False

    seen_sizes: set[tuple[int, int]] = set()
    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        size = (r1 - r0 + 1, c1 - c0 + 1)
        if size in seen_sizes:
            return False
        seen_sizes.add(size)

    output = main(grid)
    return isinstance(output, np.ndarray) and output.shape == grid.shape and max(output.shape) <= CANVAS_LIMIT


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _sample_ring_specs(rng)
    placed_specs = _place_specs(specs, rng)
    if placed_specs is None:
        return None

    grid = _render_grid(placed_specs, rng)
    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
