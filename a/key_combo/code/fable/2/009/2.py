from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from solvers.arc_utils import colors_present, get_bg, paste


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _coord_set(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _normalize_coords(coords: Iterable[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    assert pts
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = {(r - min_r, c - min_c) for r, c in pts}
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return norm, (height, width)


def _translate(coords: Iterable[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in coords}


def _footprint(template_coords: set[tuple[int, int]], top: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = map(int, top)
    return _translate(template_coords, tr, tc)


def _candidate_placements(
    grid: np.ndarray,
    color: int,
    template_coords: set[tuple[int, int]],
    template_shape: tuple[int, int],
    bg: int,
) -> list[tuple[int, int]]:
    """Return all top-left placements whose full footprint explains one color residue."""
    h, w = template_shape
    H, W = grid.shape
    visible = _coord_set(grid == color)
    candidates: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            footprint = _footprint(template_coords, (r0, c0))
            if not visible.issubset(footprint):
                continue
            if any(int(grid[r, c]) == int(bg) for r, c in footprint):
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _recover_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert len(colors) >= 3, "expected several overlapping copies"

    visible_counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    max_area = max(visible_counts.values())
    template_colors = [color for color, area in visible_counts.items() if area == max_area]
    assert len(template_colors) == 1, "expected exactly one fully visible template copy"
    template_color = int(template_colors[0])

    template_coords, template_shape = _normalize_coords(_coord_set(g == template_color))
    area = len(template_coords)

    placements: dict[int, tuple[int, int]] = {}
    footprints: dict[int, set[tuple[int, int]]] = {}
    for color in colors:
        vis_area = visible_counts[color]
        assert 0 < vis_area <= area
        if color != template_color:
            assert vis_area < area, "only one copy may be fully visible"
        candidates = _candidate_placements(g, color, template_coords, template_shape, bg)
        if color == template_color:
            assert len(candidates) == 1, "template placement must be unique"
        else:
            assert len(candidates) == 1, f"ambiguous placement for color {color}"
        placements[color] = candidates[0]
        footprints[color] = _footprint(template_coords, candidates[0])

    return {
        "bg": bg,
        "colors": colors,
        "template_color": template_color,
        "template_coords": template_coords,
        "template_shape": template_shape,
        "area": area,
        "visible_counts": visible_counts,
        "placements": placements,
        "footprints": footprints,
    }


def _occlusion_edges(grid: np.ndarray, footprints: dict[int, set[tuple[int, int]]]) -> dict[int, set[int]]:
    g = np.asarray(grid, dtype=int)
    colors = list(footprints)
    edges = {color: set() for color in colors}
    for i, left in enumerate(colors):
        for right in colors[i + 1 :]:
            overlap = footprints[left] & footprints[right]
            if not overlap:
                continue
            left_front = any(int(g[r, c]) == int(left) for r, c in overlap)
            right_front = any(int(g[r, c]) == int(right) for r, c in overlap)
            assert not (left_front and right_front), "pairwise overlap cannot show both colors"
            if left_front:
                edges[left].add(right)
            elif right_front:
                edges[right].add(left)
    return edges


def _unique_topological_order(edges: dict[int, set[int]]) -> list[int]:
    indeg = {node: 0 for node in edges}
    for src, dsts in edges.items():
        for dst in dsts:
            assert dst in indeg and dst != src
            indeg[dst] += 1

    order: list[int] = []
    ready = [node for node, deg in indeg.items() if deg == 0]
    while ready:
        assert len(ready) == 1, "occlusion chain must define a unique total order"
        node = ready.pop()
        order.append(node)
        for dst in sorted(edges[node]):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)

    assert len(order) == len(edges), "occlusion relation must be acyclic"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the occlusion depth order of congruent overlapping sprites and output their colors as a front-to-back column.

    Input structure:
    - A background grid contains several solid monochrome copies of one untranslated shape family, all with distinct colors.
    - The copies share the same orientation and partially overlap, so deeper copies may be split into visible fragments.
    - Exactly one copy is fully visible; every other color is a cropped residue of that template at some translated location.

    Rule:
    - Use the unique fully visible copy as the template shape.
    - For each other color, find the unique translated template footprint whose visible cells match that color and whose hidden cells are occupied by front colors instead of background.
    - From the recovered full footprints, read every visible overlap witness to determine who covers whom, then topologically sort those relations and return the front-to-back color order as an n×1 column.

    Key insight:
    - The missing cells of a deeper residue are not absent geometry; they are hidden template cells, and the colors showing on those hidden positions reveal the unique depth chain.
    """
    scene = _recover_scene(np.asarray(grid, dtype=int))
    edges = _occlusion_edges(grid, scene["footprints"])
    order = _unique_topological_order(edges)
    return np.asarray(order, dtype=int).reshape(-1, 1)


def _solid_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _overlap_count(template_coords: set[tuple[int, int]], delta: tuple[int, int]) -> int:
    dr, dc = map(int, delta)
    return sum((r - dr, c - dc) in template_coords for r, c in template_coords)


def _random_connected_mask(h: int, w: int) -> np.ndarray | None:
    area_min = max(7, int(np.ceil(0.45 * h * w)))
    area_max = max(area_min, int(np.floor(0.75 * h * w)))
    target = random.randint(area_min, area_max)

    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier = []
        for r, c in list(cells):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
        return None
    if np.array_equal(mask, mask[:, ::-1]) or np.array_equal(mask, mask[::-1, :]) or np.array_equal(mask, mask[::-1, ::-1]):
        return None
    return mask


def _shift_pair(template_coords: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    area = len(template_coords)
    buckets: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {}
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            overlap = _overlap_count(template_coords, (dr, dc))
            if 0 < overlap < area:
                buckets.setdefault((_sign(dr), _sign(dc)), []).append(((dr, dc), overlap))

    bucket_items = list(buckets.items())
    random.shuffle(bucket_items)
    for _bucket, items in bucket_items:
        heavy = sorted(items, key=lambda item: item[1], reverse=True)
        light = sorted(items, key=lambda item: item[1])
        pairs: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        for d1, ov1 in heavy:
            for d2, ov2 in light:
                if ov1 <= ov2:
                    continue
                total = (d1[0] + d2[0], d1[1] + d2[1])
                if _overlap_count(template_coords, total) != 0:
                    continue
                pairs.append((d1, d2, ov1 - ov2))
        if pairs:
            pairs.sort(key=lambda item: item[2], reverse=True)
            choice_pool = pairs[: min(6, len(pairs))]
            d1, d2, _gap = random.choice(choice_pool)
            return d1, d2
    return None


def _build_relative_positions(mask: np.ndarray, n_objects: int) -> list[tuple[int, int]] | None:
    template_coords = _coord_set(mask)
    pair = _shift_pair(template_coords, mask.shape)
    if pair is None:
        return None
    d_heavy, d_light = pair
    steps = [d_heavy if i % 2 == 0 else d_light for i in range(n_objects - 1)]

    positions = [(0, 0)]
    for dr, dc in steps:
        pr, pc = positions[-1]
        positions.append((pr + dr, pc + dc))

    area = int(mask.sum())
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            rel = (positions[j][0] - positions[i][0], positions[j][1] - positions[i][1])
            overlap = _overlap_count(template_coords, rel)
            if j == i + 1:
                if not (0 < overlap < area):
                    return None
            elif overlap != 0:
                return None
    return positions


def _absolute_positions(mask: np.ndarray, relative_positions: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    h, w = mask.shape
    rows = [r for r, _ in relative_positions]
    cols = [c for _, c in relative_positions]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    extent_h = max_r - min_r + h
    extent_w = max_c - min_c + w
    if extent_h + 2 > 30 or extent_w + 2 > 30:
        return None

    extra_h = random.randint(2, min(12, 30 - extent_h))
    extra_w = random.randint(2, min(12, 30 - extent_w))
    top = random.randint(1, extra_h - 1)
    left = random.randint(1, extra_w - 1)
    H = extent_h + extra_h
    W = extent_w + extra_w
    shift_r = top - min_r
    shift_c = left - min_c
    absolute = [(r + shift_r, c + shift_c) for r, c in relative_positions]
    return absolute, (H, W)


def _render_scene(
    mask: np.ndarray,
    order: list[int],
    positions: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    canvas = np.zeros(shape, dtype=int)
    patch_by_color = {color: _solid_patch(mask, color) for color in order}
    footprints: dict[int, set[tuple[int, int]]] = {}
    mask_coords = _coord_set(mask)

    for color, (r0, c0) in zip(order, positions):
        footprints[color] = _translate(mask_coords, r0, c0)

    for color, (r0, c0) in reversed(list(zip(order, positions))):
        paste(canvas, patch_by_color[color], r0, c0, transparent=BG)
    return canvas, footprints


def _topo_from_hidden(grid: np.ndarray, footprints: dict[int, set[tuple[int, int]]]) -> list[int]:
    return _unique_topological_order(_occlusion_edges(grid, footprints))


def _semantic_ok(mask: np.ndarray, order: list[int], grid: np.ndarray, footprints: dict[int, set[tuple[int, int]]]) -> bool:
    try:
        area = int(mask.sum())
        visible_counts = [int(np.count_nonzero(grid == color)) for color in order]
        if visible_counts[0] != area:
            return False
        if any(count <= 0 or count >= area for count in visible_counts[1:]):
            return False
        if not any(visible_counts[j] > visible_counts[i] for i in range(1, len(order) - 1) for j in range(i + 1, len(order))):
            return False
        if _topo_from_hidden(grid, footprints) != order:
            return False

        recovered = _recover_scene(grid)
        for color in order:
            if recovered["footprints"][color] != footprints[color]:
                return False
        return np.array_equal(main(grid), np.asarray(order, dtype=int).reshape(-1, 1))
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    mask = _random_connected_mask(h, w)
    if mask is None:
        return None

    n_objects = random.randint(3, 5)
    relative_positions = _build_relative_positions(mask, n_objects)
    if relative_positions is None:
        return None

    placed = _absolute_positions(mask, relative_positions)
    if placed is None:
        return None
    positions, canvas_shape = placed

    order = random.sample(PALETTE, n_objects)
    grid, footprints = _render_scene(mask, order, positions, canvas_shape)

    if int(np.count_nonzero(grid == BG)) <= max(int(mask.sum()), grid.size // 3):
        return None
    if not _semantic_ok(mask, order, grid, footprints):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-chain input")
