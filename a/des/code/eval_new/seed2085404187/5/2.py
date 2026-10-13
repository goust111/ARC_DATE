from __future__ import annotations

import numpy as np


BG_COLOR = 0
FENCE_COLOR = 5
SEED_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_ring_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0 + 1, r1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _draw_rect_outline(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw inward concentric rectangular outlines inside each fenced region according to its seed count.

    Input structure:
    - bg = black. The grid contains one or more closed grey rectangular fences,
      each drawn as a 1-pixel-thick hollow ring.
    - Every fence encloses 1-4 seed pixels of a single non-grey color; different
      fences may use different seed colors.
    - Fence exteriors are background-only, and fences do not overlap or nest.

    Rule:
    - For each grey fence, count the enclosed seed pixels; that count N requests
      N inward-shrinking rectangular outlines.
    - All outlines inside a fence use the fence's seed color. The first outline
      hugs the inner wall, and each later outline shifts one cell further inward.
    - Keep the grey fence unchanged. Stop drawing when the next shrink step can
      no longer form a complete rectangular ring.

    Key insight:
    - The number of same-color seeds is an instruction, not decoration: it sets
      the depth of the concentric geometry that must be drawn inside that fence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _components(g == FENCE_COLOR):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "fence must enclose an interior"
        assert set(comp) == _rect_ring_coords(r0, c0, r1, c1), "grey component must be a rectangular fence"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != BG_COLOR) & (interior != FENCE_COLOR)
        seed_values = interior[seed_mask]
        assert 1 <= seed_values.size <= 4, "each fence must contain 1-4 seed pixels"

        colors = np.unique(seed_values)
        assert colors.size == 1, "all seeds in one fence must share a color"
        seed_color = int(colors[0])

        for depth in range(1, int(seed_values.size) + 1):
            top = r0 + depth
            left = c0 + depth
            bottom = r1 - depth
            right = c1 - depth
            if top >= bottom or left >= right:
                break
            _draw_rect_outline(out, top, left, bottom, right, seed_color)

    return out


def _sample_branch(rng: np.random.Generator, seed_count: int) -> int:
    """Choose how many requested rings are geometrically drawable."""
    options = ["full", "none"]
    if seed_count >= 2:
        options.append("partial")
    branch = options[int(rng.integers(0, len(options)))]
    if branch == "full":
        return seed_count
    if branch == "none":
        return 0
    return int(rng.integers(1, seed_count))


def _sample_inner_dims(
    rng: np.random.Generator, seed_count: int, drawable_layers: int
) -> tuple[int, int]:
    """Sample an interior size whose max drawable ring count matches the requested branch."""
    if drawable_layers == seed_count:
        small = int(rng.integers(2 * seed_count, 2 * seed_count + 4))
        large = int(rng.integers(small, min(12, small + 4) + 1))
    elif drawable_layers == 0:
        small = 1
        large = int(rng.integers(seed_count, min(8, seed_count + 4) + 1))
    else:
        small_choices = [2 * drawable_layers, 2 * drawable_layers + 1]
        small = int(small_choices[int(rng.integers(0, len(small_choices)))])
        min_large = max(small, (seed_count + small - 1) // small)
        large = int(rng.integers(min_large, min(10, min_large + 4) + 1))

    if rng.random() < 0.5:
        return small, large
    return large, small


def _choose_seed_positions(
    rng: np.random.Generator, inner_h: int, inner_w: int, seed_count: int
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    picks = rng.choice(len(cells), size=seed_count, replace=False)
    return [cells[int(i)] for i in np.asarray(picks, dtype=int)]


def _place_boxes(
    rng: np.random.Generator, height: int, width: int, boxes: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    """Place outer fence boxes with a 1-cell gap and a 1-cell border margin."""
    occupied = np.zeros((height, width), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True)

    for idx in order:
        box_h, box_w = boxes[idx]
        if box_h > height - 2 or box_w > width - 2:
            return None

        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - box_h):
            for c0 in range(1, width - box_w):
                rr0 = r0 - 1
                cc0 = c0 - 1
                rr1 = r0 + box_h + 1
                cc1 = c0 + box_w + 1
                if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        placements[idx] = (r0, c0)
        occupied[r0 - 1 : r0 + box_h + 1, c0 - 1 : c0 + box_w + 1] = True

    return [placements[i] for i in range(len(boxes))]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(12, 31))
    width = int(rng.integers(12, 31))
    n_fences = int(rng.integers(1, 5))
    colors = rng.choice(SEED_COLORS, size=n_fences, replace=False)

    specs: list[dict[str, int]] = []
    any_drawable = False
    for color in np.asarray(colors, dtype=int):
        seed_count = int(rng.integers(1, 5))
        drawable_layers = _sample_branch(rng, seed_count)
        inner_h, inner_w = _sample_inner_dims(rng, seed_count, drawable_layers)
        specs.append(
            {
                "color": int(color),
                "seed_count": seed_count,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )
        any_drawable |= drawable_layers > 0

    if not any_drawable:
        return None

    placements = _place_boxes(rng, height, width, [(s["outer_h"], s["outer_w"]) for s in specs])
    if placements is None:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    for spec, (r0, c0) in zip(specs, placements):
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        color = spec["color"]

        _draw_rect_outline(grid, r0, c0, r0 + outer_h - 1, c0 + outer_w - 1, FENCE_COLOR)
        for dr, dc in _choose_seed_positions(rng, inner_h, inner_w, spec["seed_count"]):
            grid[r0 + 1 + dr, c0 + 1 + dc] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-rings input")
