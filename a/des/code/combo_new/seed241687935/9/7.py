from __future__ import annotations

import random
from typing import Iterable

import numpy as np


AZURE = 8
MAROON = 9
ARC_MAX = 30

DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
ORTH_DIRS = frozenset({(-1, 0), (0, -1), (0, 1), (1, 0)})
DIAG_DIRS = frozenset({(-1, -1), (-1, 1), (1, -1), (1, 1)})

COLOR_POOL = tuple(color for color in range(10) if color not in (AZURE, MAROON))
GEN_ATTEMPTS = 500
OBJECT_COUNT_CHOICES = (2, 3, 4)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _get_bg(grid: np.ndarray) -> int:
    if np.any(grid == AZURE):
        return AZURE
    return _mode_color(grid)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS8:
        yield r + dr, c + dc


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neighbors = _neighbors8 if conn8 else _neighbors4
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in neighbors(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords)


def _boundary_mask(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros_like(filled, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            if (
                r == 0
                or r == h - 1
                or c == 0
                or c == w - 1
                or not filled[r - 1, c]
                or not filled[r + 1, c]
                or not filled[r, c - 1]
                or not filled[r, c + 1]
            ):
                boundary[r, c] = True
    return boundary


def _direction_component_center(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    candidates: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for cr, cc in coords:
        rels = []
        ok = True
        for r, c in coords:
            dr = r - cr
            dc = c - cc
            if max(abs(dr), abs(dc)) > 1:
                ok = False
                break
            rels.append((dr, dc))
        if ok:
            rels_tuple = tuple(sorted(rels))
            candidates.append(((cr, cc), rels_tuple))
    assert len(candidates) == 1, "legend template center must be uniquely recoverable"
    center, rels = candidates[0]
    dirs = tuple(rel for rel in rels if rel != (0, 0))
    assert dirs, "legend template must contain at least one growth direction"
    return center, dirs


def _find_fenced_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    jobs: list[dict[str, object]] = []
    for comp in _components(g != MAROON, conn8=False):
        if _touches_border(comp, h, w):
            continue
        colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != bg}
        if len(colors) != 1:
            continue
        color = colors.pop()
        seed_pixels = [(r, c) for r, c in comp if int(g[r, c]) == color]
        if len(seed_pixels) != 1:
            continue
        allowed = frozenset(comp)
        steps = sum(1 for r, c in comp if int(g[r, c]) == bg)
        jobs.append(
            {
                "color": color,
                "seed": seed_pixels[0],
                "steps": steps,
                "allowed": allowed,
            }
        )
    return jobs


def _read_templates(grid: np.ndarray, jobs: list[dict[str, object]]) -> dict[int, tuple[tuple[int, int], ...]]:
    g = np.asarray(grid, dtype=int)
    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    seed_by_color = {int(job["color"]): tuple(job["seed"]) for job in jobs}
    for color, seed in seed_by_color.items():
        comps = _components(g == color, conn8=True)
        template_comp: list[tuple[int, int]] | None = None
        for comp in comps:
            if seed not in comp:
                assert template_comp is None, f"expected one legend template for color {color}"
                template_comp = comp
        assert template_comp is not None, f"missing legend template for color {color}"
        _, dirs = _direction_component_center(template_comp)
        templates[color] = dirs
    return templates


def _grow(seed: tuple[int, int], dirs: tuple[tuple[int, int], ...], allowed: frozenset[tuple[int, int]], steps: int) -> set[tuple[int, int]]:
    visited = {seed}
    frontier = {seed}
    for _ in range(steps):
        next_frontier: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (nr, nc) in allowed and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    next_frontier.add((nr, nc))
        if not next_frontier:
            break
        frontier = next_frontier
    return visited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each enclosed seed according to its matching bottom legend template for as many steps as its enclosure area allows.

    Input structure:
    - The background is azure, and one-pixel-thick maroon closed fences form several separate enclosed regions.
    - Every fenced region contains exactly one colored seed pixel and otherwise azure interior cells.
    - A legend sits at the bottom: each seed color appears again as a small 3×3 same-color template whose center is the seed color and whose surrounding same-color neighbors indicate growth directions.

    Rule:
    - Find every enclosed non-maroon region that does not touch the grid border, and read its unique seed color and position.
    - Count the azure cells inside that region; this count is the number of growth iterations for that seed.
    - Locate the legend template with the same color, recover the neighbor offsets around its center, and use those offsets as the seed's growth directions.
    - Starting from the seed, repeatedly extend the current frontier by one cell along every template direction, stopping any branch that would leave the enclosed region or cross the maroon fence.
    - Paint all reached cells with the seed color while leaving the fences, legend, and background unchanged.

    Key insight:
    - The color only selects a direction template; the hidden quantitative clue is that the enclosure's interior area determines how many times that template-driven growth is iterated.
    """
    g = np.asarray(grid, dtype=int)
    jobs = _find_fenced_regions(g)
    assert jobs, "expected at least one fenced region"

    templates = _read_templates(g, jobs)
    assert set(templates) == {int(job["color"]) for job in jobs}

    out = g.copy()
    for job in jobs:
        color = int(job["color"])
        seed = tuple(job["seed"])
        steps = int(job["steps"])
        allowed = job["allowed"]
        grown = _grow(seed, templates[color], allowed, steps)
        for r, c in grown:
            out[r, c] = color
    return out


def _direction_sets() -> tuple[tuple[tuple[int, int], ...], ...]:
    valid: list[tuple[tuple[int, int], ...]] = []
    for mask in range(1, 1 << len(DIRS8)):
        dirs = tuple(DIRS8[i] for i in range(len(DIRS8)) if (mask >> i) & 1)
        if len(dirs) < 2 or len(dirs) > 5:
            continue
        coords = [(0, 0), *dirs]
        centers = 0
        for cr, cc in coords:
            if all(max(abs(r - cr), abs(c - cc)) <= 1 for r, c in coords):
                centers += 1
        if centers == 1:
            valid.append(dirs)
    return tuple(valid)


VALID_DIRECTION_SETS = _direction_sets()
ORTHOGONAL_TEMPLATES = tuple(dirs for dirs in VALID_DIRECTION_SETS if all(d in ORTH_DIRS for d in dirs))
DIAGONAL_TEMPLATES = tuple(dirs for dirs in VALID_DIRECTION_SETS if all(d in DIAG_DIRS for d in dirs))
MIXED_TEMPLATES = tuple(
    dirs for dirs in VALID_DIRECTION_SETS if any(d in ORTH_DIRS for d in dirs) and any(d in DIAG_DIRS for d in dirs)
)


def _randchoice(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def _choose_directions(rng: random.Random) -> tuple[tuple[int, int], ...]:
    buckets = [ORTHOGONAL_TEMPLATES, DIAGONAL_TEMPLATES, MIXED_TEMPLATES]
    bucket = buckets[rng.randrange(len(buckets))]
    return tuple(_randchoice(rng, bucket))


def _carve_notch(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    side = _randchoice(rng, ("top", "bottom", "left", "right"))
    carved = mask.copy()
    if side in ("top", "bottom"):
        if h < 6 or w < 4:
            return carved
        depth = rng.randint(1, h - 3)
        span = rng.randint(1, w - 2)
        start = rng.randint(1, w - span - 1)
        if side == "top":
            carved[:depth, start : start + span] = False
        else:
            carved[h - depth :, start : start + span] = False
    else:
        if w < 6 or h < 4:
            return carved
        depth = rng.randint(1, w - 3)
        span = rng.randint(1, h - 2)
        start = rng.randint(1, h - span - 1)
        if side == "left":
            carved[start : start + span, :depth] = False
        else:
            carved[start : start + span, w - depth :] = False
    return carved


def _generate_region_patch(
    rng: random.Random, color: int, dirs: tuple[tuple[int, int], ...]
) -> dict[str, object] | None:
    h = rng.randint(6, 11)
    w = rng.randint(6, 11)
    filled = np.ones((h, w), dtype=bool)
    notch_count = rng.choices([0, 1, 2, 3], weights=[1, 3, 3, 2], k=1)[0]
    for _ in range(notch_count):
        filled = _carve_notch(filled, rng)

    if not np.any(filled):
        return None
    if len(_components(filled, conn8=False)) != 1:
        return None

    boundary = _boundary_mask(filled)
    if len(_components(boundary, conn8=False)) != 1:
        return None

    interior = filled & ~boundary
    interior_comps = _components(interior, conn8=False)
    if len(interior_comps) != 1:
        return None
    interior_cells = interior_comps[0]
    if len(interior_cells) < 4:
        return None

    seed = _randchoice(rng, interior_cells)
    grown = _grow(seed, dirs, frozenset(interior_cells), len(interior_cells) - 1)
    if len(grown) < min(3, len(interior_cells)):
        return None

    patch = np.full((h, w), AZURE, dtype=int)
    patch[boundary] = MAROON
    patch[seed] = color
    return {
        "patch": patch,
        "shape": patch.shape,
        "color": color,
        "dirs": dirs,
        "seed": seed,
        "interior": frozenset(interior_cells),
        "grown": grown,
    }


def _place_objects(
    rng: random.Random, shapes: list[tuple[int, int]], top_h: int, width: int
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((top_h, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)

    for idx in order:
        h, w = shapes[idx]
        if h + 2 > top_h or w + 2 > width:
            return None
        candidates: list[tuple[int, int]] = []
        for r in range(1, top_h - h):
            for c in range(1, width - w):
                if occupied[r - 1 : r + h + 1, c - 1 : c + w + 1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        pos = _randchoice(rng, candidates)
        placements[idx] = pos
        r, c = pos
        occupied[r : r + h, c : c + w] = True

    return [pos for pos in placements if pos is not None]


def _paint_template(grid: np.ndarray, top: int, left: int, color: int, dirs: tuple[tuple[int, int], ...]) -> None:
    grid[top + 1, left + 1] = color
    for dr, dc in dirs:
        grid[top + 1 + dr, left + 1 + dc] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = _randchoice(rng, OBJECT_COUNT_CHOICES)
    colors = rng.sample(COLOR_POOL, count)

    objects: list[dict[str, object]] = []
    for color in colors:
        dirs = _choose_directions(rng)
        obj = _generate_region_patch(rng, color, dirs)
        if obj is None:
            return None
        objects.append(obj)

    legend_width = 4 * count - 1
    min_width = max(legend_width + 2, max(int(obj["shape"][1]) for obj in objects) + 4)
    if min_width > ARC_MAX:
        return None
    width = rng.randint(min_width, ARC_MAX)

    max_patch_h = max(int(obj["shape"][0]) for obj in objects)
    top_h_min = max(max_patch_h + 3, 10)
    top_h_max = ARC_MAX - 4
    if top_h_min > top_h_max:
        return None
    top_h = rng.randint(top_h_min, top_h_max)
    height = top_h + 4

    placements = _place_objects(rng, [tuple(obj["shape"]) for obj in objects], top_h, width)
    if placements is None:
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    for obj, (r0, c0) in zip(objects, placements):
        patch = obj["patch"]
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = np.where(patch != AZURE, patch, grid[r0 : r0 + h, c0 : c0 + w])

    legend_order = colors[:]
    rng.shuffle(legend_order)
    legend_left = rng.randint(1, width - legend_width - 1)
    legend_top = height - 3
    for idx, color in enumerate(legend_order):
        left = legend_left + 4 * idx
        dirs = next(tuple(obj["dirs"]) for obj in objects if int(obj["color"]) == color)
        _paint_template(grid, legend_top, left, color, dirs)

    jobs = _find_fenced_regions(grid)
    if len(jobs) != count:
        return None
    if {int(job["color"]) for job in jobs} != set(colors):
        return None
    templates = _read_templates(grid, jobs)
    if set(templates) != set(colors):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
