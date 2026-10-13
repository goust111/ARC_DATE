from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_ROWS = (18, 26)
GRID_COLS = (18, 26)
GROUP_COUNT_OPTIONS = (3, 4)
STEP_RANGE = (2, 4)
CENTER_MANHATTAN_MIN = 4
MIN_CHANGED_CELLS = 20
MIN_VISIBLE_RING_CELLS = 4
GEN_ATTEMPTS = 4000


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _seed_triplet(center: tuple[int, int], step: int, diag: int) -> set[tuple[int, int]]:
    r, c = center
    d = int(step)
    s = int(diag)
    return {
        (r, c),
        (r - d, c - s * d),
        (r + d, c + s * d),
    }


def _infer_group(color: int, coords: np.ndarray) -> dict:
    pts = [tuple(map(int, rc)) for rc in coords]
    assert len(pts) == 3, "each color must appear exactly three times"

    center = None
    endpoints: list[tuple[int, int]] | None = None
    for idx, candidate in enumerate(pts):
        other = [pts[j] for j in range(3) if j != idx]
        if (
            2 * candidate[0] == other[0][0] + other[1][0]
            and 2 * candidate[1] == other[0][1] + other[1][1]
        ):
            center = candidate
            endpoints = other
            break

    assert center is not None and endpoints is not None, "three seeds must have a unique midpoint"
    dr = endpoints[0][0] - center[0]
    dc = endpoints[0][1] - center[1]
    assert abs(dr) == abs(dc) and dr != 0, "seed triplet must lie on a 45-degree diagonal"
    assert (endpoints[1][0] - center[0], endpoints[1][1] - center[1]) == (-dr, -dc)
    diag = 1 if dr * dc > 0 else -1
    step = abs(dr)
    seeds = _seed_triplet(center, step, diag)
    assert seeds == set(pts), "seed spacing must be equal on both sides of the center"
    return {
        "color": int(color),
        "center": center,
        "d": int(step),
        "diag": int(diag),
        "seeds": seeds,
    }


def _extract_groups(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _get_bg(grid)
    colors = sorted(set(map(int, grid.flat)) - {bg})
    assert len(colors) in GROUP_COUNT_OPTIONS, "expected 3 or 4 seed groups"

    groups = []
    occupied: set[tuple[int, int]] = set()
    centers: list[tuple[int, int]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        group = _infer_group(color, coords)
        assert occupied.isdisjoint(group["seeds"]), "seed groups must not overlap"
        for r0, c0 in centers:
            r1, c1 = group["center"]
            assert abs(r0 - r1) + abs(c0 - c1) >= CENTER_MANHATTAN_MIN
        occupied |= group["seeds"]
        centers.append(group["center"])
        groups.append(group)
    return bg, groups


def _ring_cells_at_radius(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0 = center
    q = int(radius)
    cells: set[tuple[int, int]] = set()

    for c in range(c0 - q, c0 + q + 1):
        if 0 <= r0 - q < h and 0 <= c < w:
            cells.add((r0 - q, c))
        if 0 <= r0 + q < h and 0 <= c < w:
            cells.add((r0 + q, c))
    for r in range(r0 - q, r0 + q + 1):
        if 0 <= r < h and 0 <= c0 - q < w:
            cells.add((r, c0 - q))
        if 0 <= r < h and 0 <= c0 + q < w:
            cells.add((r, c0 + q))
    return cells


def _all_ring_cells(shape: tuple[int, int], center: tuple[int, int], step: int) -> tuple[set[tuple[int, int]], list[int]]:
    h, w = shape
    r, c = center
    limit = max(r, h - 1 - r, c, w - 1 - c)
    cells: set[tuple[int, int]] = set()
    radii: list[int] = []
    radius = int(step)
    while radius <= limit:
        ring = _ring_cells_at_radius(shape, center, radius)
        if ring:
            cells |= ring
            radii.append(radius)
        radius += step
    return cells, radii


def _prepare_groups(groups: list[dict], shape: tuple[int, int]) -> list[dict]:
    prepared = []
    for group in groups:
        copied = dict(group)
        ring_cells, radii = _all_ring_cells(shape, copied["center"], copied["d"])
        assert radii, "each group must generate at least one square border"
        copied["ring_cells"] = ring_cells
        copied["radii"] = radii
        prepared.append(copied)
    return prepared


def _occlusion_edges(groups: list[dict]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for lower in groups:
        for upper in groups:
            if lower["color"] == upper["color"]:
                continue
            if any(seed in lower["ring_cells"] for seed in upper["seeds"]):
                edges.add((lower["color"], upper["color"]))
    return edges


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int] | None:
    outgoing = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for src, dst in edges:
        if dst not in outgoing[src]:
            outgoing[src].add(dst)
            indegree[dst] += 1

    order: list[int] = []
    frontier = sorted(color for color in colors if indegree[color] == 0)
    while frontier:
        if len(frontier) != 1:
            return None
        color = frontier.pop()
        order.append(color)
        for nxt in sorted(outgoing[color]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(nxt)
                frontier.sort()

    if len(order) != len(colors):
        return None
    return order


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _render_from_order(
    shape: tuple[int, int],
    bg: int,
    groups: list[dict],
    order: list[int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    by_color = {group["color"]: group for group in groups}

    for color in order:
        _paint_cells(out, by_color[color]["ring_cells"], color)
    for group in groups:
        _paint_cells(out, group["seeds"], group["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow concentric square borders from diagonal seed triplets, infer the unique occlusion chain from seed invasions, and render the visible layered result.

    Input structure:
    - The grid has a black dominant background with 3 or 4 foreground colors.
    - Each foreground color marks exactly three pixels on one 45-degree diagonal, equally spaced, with the middle pixel as that group's center seed.
    - Different colors are different groups; seed pixels do not overlap.

    Rule:
    - For each color, recover the center seed and spacing d from its three equally spaced diagonal pixels.
    - Draw every hollow axis-aligned square border centered on that seed whose Chebyshev radius is a positive multiple of d, stopping once an entire border would miss the grid.
    - If any border of group A passes through any original seed pixel of group B, then A is below B; the generated inputs guarantee these relations form one unique bottom-to-top total order.
    - Render all groups from bottom to top so later groups cover earlier ones on overlaps, then restore every original seed pixel in its own color.

    Key insight:
    - The same seed coordinates serve two roles at once: they set each group's square-growth rhythm and they act as probes that reveal which other groups must sit above it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, groups = _extract_groups(g)
    groups = _prepare_groups(groups, g.shape)
    colors = [group["color"] for group in groups]
    edges = _occlusion_edges(groups)
    order = _unique_topological_order(colors, edges)
    assert order is not None, "occlusion relations must define a unique total order"
    return _render_from_order(g.shape, bg, groups, order)


def _random_center(shape: tuple[int, int], step: int) -> tuple[int, int]:
    h, w = shape
    margin = 2 * step
    return (
        random.randint(margin, h - 1 - margin),
        random.randint(margin, w - 1 - margin),
    )


def _make_random_group(shape: tuple[int, int], color: int) -> dict:
    step = random.randint(*STEP_RANGE)
    diag = random.choice((-1, 1))
    center = _random_center(shape, step)
    return {
        "color": int(color),
        "center": center,
        "d": int(step),
        "diag": int(diag),
        "seeds": _seed_triplet(center, step, diag),
    }


def _chain_edges(order: list[int]) -> set[tuple[int, int]]:
    return {(order[i], order[i + 1]) for i in range(len(order) - 1)}


def _visible_ring_cells(out: np.ndarray, groups: list[dict], color: int) -> int:
    group = next(group for group in groups if group["color"] == color)
    return sum(1 for cell in group["ring_cells"] if cell not in group["seeds"] and int(out[cell]) == color)


def _has_non_seed_overlap(groups: list[dict], order: list[int]) -> bool:
    by_color = {group["color"]: group for group in groups}
    seeds = set().union(*(group["seeds"] for group in groups))
    for lower_color, upper_color in _chain_edges(order):
        overlap = (by_color[lower_color]["ring_cells"] & by_color[upper_color]["ring_cells"]) - seeds
        if overlap:
            return True
    return False


def _simple_orders(groups: list[dict]) -> list[list[int]]:
    keyed = [
        ("d", lambda g: (g["d"], g["center"][0], g["center"][1], g["color"])),
        ("row", lambda g: (g["center"][0], g["center"][1], g["d"], g["color"])),
        ("col", lambda g: (g["center"][1], g["center"][0], g["d"], g["color"])),
        ("area", lambda g: (len(g["ring_cells"]), g["d"], g["center"][0], g["color"])),
    ]
    orders: list[list[int]] = []
    for _name, key in keyed:
        asc = [group["color"] for group in sorted(groups, key=key)]
        desc = list(reversed(asc))
        orders.append(asc)
        orders.append(desc)
    return orders


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        bg, groups = _extract_groups(grid)
        if bg != BACKGROUND:
            return False
        groups = _prepare_groups(groups, grid.shape)
        colors = [group["color"] for group in groups]
        edges = _occlusion_edges(groups)
        order = _unique_topological_order(colors, edges)
        if order is None:
            return False
        if edges != _chain_edges(order):
            return False
        if len({group["d"] for group in groups}) == 1:
            return False
        if len({group["diag"] for group in groups}) == 1 and random.random() < 0.7:
            return False
        if any(len(group["radii"]) < 2 for group in groups):
            return False
        if not _has_non_seed_overlap(groups, order):
            return False
        if order in _simple_orders(groups):
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if int(np.count_nonzero(out != grid)) < MIN_CHANGED_CELLS:
            return False
        if np.array_equal(out, _render_from_order(grid.shape, bg, groups, list(reversed(order)))):
            return False
        if any(_visible_ring_cells(out, groups, color) < MIN_VISIBLE_RING_CELLS for color in colors):
            return False
        return True
    except Exception:
        return False


def _attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_ROWS),
        random.randint(*GRID_COLS),
    )
    count = random.choice(GROUP_COUNT_OPTIONS)
    colors = random.sample(PALETTE, count)

    for _ in range(300):
        groups: list[dict] = []
        used_seeds: set[tuple[int, int]] = set()
        centers: list[tuple[int, int]] = []

        ok = True
        for color in colors:
            placed = None
            for _ in range(80):
                candidate = _make_random_group(shape, color)
                if not used_seeds.isdisjoint(candidate["seeds"]):
                    continue
                if any(
                    abs(candidate["center"][0] - r) + abs(candidate["center"][1] - c) < CENTER_MANHATTAN_MIN
                    for r, c in centers
                ):
                    continue
                placed = candidate
                break
            if placed is None:
                ok = False
                break
            groups.append(placed)
            used_seeds |= placed["seeds"]
            centers.append(placed["center"])

        if not ok:
            continue

        grid = np.full(shape, BACKGROUND, dtype=int)
        for group in groups:
            _paint_cells(grid, group["seeds"], group["color"])

        if _semantic_ok(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-occlusion ARC input")
