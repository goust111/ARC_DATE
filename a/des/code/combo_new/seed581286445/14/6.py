from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_SIZE = 30
GEN_RETRIES = 400
SHAPE_RETRIES = 200

BASE_TEMPLATES: dict[str, list[tuple[int, int]]] = {
    "chain3_iso": [(0, 0), (0, 1), (0, 2), (2, 0)],
    "chain4_iso": [(0, 0), (0, 1), (0, 2), (0, 3), (2, 1)],
    "plus": [(1, 1), (1, 0), (1, 2), (0, 1), (2, 1)],
    "tee_iso": [(1, 0), (1, 1), (1, 2), (2, 1), (0, 4)],
}

DIM_OPTIONS = {
    4: [(2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (2, 5), (5, 2)],
    5: [(3, 3), (3, 4), (4, 3), (2, 5), (5, 2)],
}


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _are_bbox_neighbors(a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    horizontal = (
        (ac1 + 2 == bc0 or bc1 + 2 == ac0)
        and _intervals_overlap(ar0, ar1, br0, br1)
    )
    vertical = (
        (ar1 + 2 == br0 or br1 + 2 == ar0)
        and _intervals_overlap(ac0, ac1, bc0, bc1)
    )
    return horizontal or vertical


def _neighbor_counts(components: list[dict]) -> list[int]:
    n = len(components)
    counts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _are_bbox_neighbors(components[i], components[j]):
                counts[i] += 1
                counts[j] += 1
    return counts


def _anchor_of(group: list[dict]) -> dict:
    return max(
        group,
        key=lambda comp: (
            int(comp["area"]),
            -int(comp["bbox"][0]),
            -int(comp["bbox"][1]),
        ),
    )


def _paint_translated(out: np.ndarray, grid: np.ndarray, comp: dict, dst_r0: int, dst_c0: int) -> None:
    src_r0, src_c0, _src_r1, _src_c1 = comp["bbox"]
    dr = int(dst_r0 - src_r0)
    dc = int(dst_c0 - src_c0)
    for r, c in comp["coords"]:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = int(grid[r, c])


def main(grid: np.ndarray) -> np.ndarray:
    """Group objects by neighbor count, then stack each repeated-count group onto its largest member.

    Input structure:
    - The background is black.
    - The foreground contains several separate monochrome 4-connected objects with varied shapes.
    - Objects never share an edge, but some bounding boxes are separated from others by exactly one cell horizontally or vertically.

    Rule:
    - Count, for each object, how many other objects have a bounding box that is orthogonally one cell away from its own while overlapping on the other axis.
    - Group objects that have the same neighbor count.
    - For every group with at least two members, choose the largest-area object as the anchor (breaking ties by smaller top-row index), align every group member's bounding-box top-left corner to the anchor's, and draw them in descending area order so smaller objects remain on top.
    - Objects whose neighbor count is unique stay in their original positions.

    Key insight:
    - The hidden grouping key is topological proximity measured through one-cell bounding-box gaps, not color or shape; once that grouping is found, each repeated-count group collapses onto its largest member and overlays there.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = BG if (g == BG).any() else int(Counter(map(int, g.flat)).most_common(1)[0][0])

    components = _find_components(g, bg=bg)
    if not components:
        return g.copy()

    counts = _neighbor_counts(components)
    by_count: dict[int, list[dict]] = defaultdict(list)
    for comp, count in zip(components, counts):
        by_count[int(count)].append(comp)

    out = np.full((h, w), bg, dtype=int)

    singletons: list[dict] = []
    merged_groups: list[tuple[tuple[int, int], list[dict], dict]] = []
    for count, group in by_count.items():
        if len(group) == 1:
            singletons.append(group[0])
            continue
        anchor = _anchor_of(group)
        merged_groups.append((anchor["bbox"][:2], group, anchor))

    for comp in sorted(singletons, key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"])):
        _paint_translated(out, g, comp, comp["bbox"][0], comp["bbox"][1])

    for _anchor_pos, group, anchor in sorted(merged_groups, key=lambda item: (item[0][0], item[0][1])):
        ordered = sorted(
            group,
            key=lambda comp: (
                -int(comp["area"]),
                int(comp["bbox"][0]),
                int(comp["bbox"][1]),
                int(comp["color"]),
            ),
        )
        dst_r0, dst_c0, _dst_r1, _dst_c1 = anchor["bbox"]
        for comp in ordered:
            _paint_translated(out, g, comp, dst_r0, dst_c0)

    return out


def _transform_lattice(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = list(points)
    if random.random() < 0.5:
        out = [(c, r) for r, c in out]
    if random.random() < 0.5:
        out = [(r, -c) for r, c in out]
    if random.random() < 0.5:
        out = [(-r, c) for r, c in out]
    min_r = min(r for r, _c in out)
    min_c = min(c for _r, c in out)
    return [(r - min_r, c - min_c) for r, c in out]


def _possible_areas(h: int, w: int) -> list[int]:
    min_area = h + w - 1
    max_area = min(10, h * w)
    return list(range(min_area, max_area + 1))


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    raise RuntimeError(f"failed to build a {h}x{w} shape of area {area}")


def _lattice_degrees(points: list[tuple[int, int]]) -> list[int]:
    n = len(points)
    deg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1]) == 1:
                deg[i] += 1
                deg[j] += 1
    return deg


def _render_objects(
    lattice_points: list[tuple[int, int]],
    h: int,
    w: int,
    colors: list[int],
    areas: list[int],
    grid_shape: tuple[int, int],
    origin: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    base_r, base_c = origin
    for (lr, lc), color, area in zip(lattice_points, colors, areas):
        mask = _random_connected_mask(h, w, area)
        r0 = int(base_r + lr * (h + 1))
        c0 = int(base_c + lc * (w + 1))
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = int(color)
        grid[r0 : r0 + h, c0 : c0 + w] = np.where(patch != 0, patch, grid[r0 : r0 + h, c0 : c0 + w])
    return grid


def _has_cross_color_contact(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v == BG:
                continue
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nv = int(grid[nr, nc])
                    if nv != BG and nv != v:
                        return True
    return False


def _input_is_valid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > MAX_SIZE:
        return False
    if not (grid == BG).any():
        return False
    if _has_cross_color_contact(grid):
        return False

    components = _find_components(grid, bg=BG)
    if len(components) < 4:
        return False
    if any(not (3 <= int(comp["area"]) <= 10) for comp in components):
        return False
    if len({int(comp["color"]) for comp in components}) != len(components):
        return False

    counts = _neighbor_counts(components)
    multiplicities = Counter(counts)
    if max(multiplicities.values()) < 2:
        return False
    if not any(v == 1 for v in multiplicities.values()):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape or max(out.shape) > MAX_SIZE:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    template_name = random.choice(list(BASE_TEMPLATES))
    base_points = BASE_TEMPLATES[template_name]
    points = _transform_lattice(base_points)
    n = len(points)

    dims = random.choice(DIM_OPTIONS[n])
    h, w = dims
    areas_pool = _possible_areas(h, w)
    if len(areas_pool) < n:
        return None
    areas = random.sample(areas_pool, n)
    random.shuffle(areas)

    degrees = _lattice_degrees(points)
    if max(Counter(degrees).values()) < 2 or not any(v == 1 for v in Counter(degrees).values()):
        return None

    colors = random.sample(list(range(1, 10)), n)

    max_lr = max(r for r, _c in points)
    max_lc = max(c for _r, c in points)
    content_h = max_lr * (h + 1) + h
    content_w = max_lc * (w + 1) + w
    if content_h > MAX_SIZE or content_w > MAX_SIZE:
        return None

    max_h = min(MAX_SIZE, content_h + 8)
    max_w = min(MAX_SIZE, content_w + 8)
    grid_h = random.randint(content_h, max_h)
    grid_w = random.randint(content_w, max_w)
    origin_r = random.randint(0, grid_h - content_h)
    origin_c = random.randint(0, grid_w - content_w)

    try:
        grid = _render_objects(points, h, w, colors, areas, (grid_h, grid_w), (origin_r, origin_c))
    except RuntimeError:
        return None

    if not _input_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
