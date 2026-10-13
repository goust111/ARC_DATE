from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
COLOR_ATTEMPTS = 600
LATTICE_H_RANGE = (4, 6)
LATTICE_W_RANGE = (4, 6)
CELL_SIZE_CHOICES = (2, 3)
MARGIN_CHOICES = (1, 2)
MULTI_REGION_ATTEMPTS = 40


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_regions(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    mask = (grid != BG) & (grid != GREY)
    labels = np.zeros((h, w), dtype=int)
    regions: list[dict[str, object]] = []

    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue

            color = int(grid[r, c])
            queue = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            region_colors = {int(grid[rr, cc]) for rr, cc in coords}
            assert len(region_colors) == 1, "each region must be monochrome"
            assert color in region_colors
            regions.append({"id": next_id, "color": color, "coords": arr})
            next_id += 1

    return labels, regions


def _region_adjacency(labels: np.ndarray, grid: np.ndarray) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {
        int(region_id): set() for region_id in np.unique(labels) if int(region_id) != 0
    }
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY:
                continue

            if 0 < c < w - 1:
                left = int(labels[r, c - 1])
                right = int(labels[r, c + 1])
                if left != 0 and right != 0 and left != right:
                    adjacency[left].add(right)
                    adjacency[right].add(left)

            if 0 < r < h - 1:
                up = int(labels[r - 1, c])
                down = int(labels[r + 1, c])
                if up != 0 and down != 0 and up != down:
                    adjacency[up].add(down)
                    adjacency[down].add(up)

    return adjacency


def _is_rectangular(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    area = int(coords.shape[0])
    return area == (r1 - r0 + 1) * (c1 - c0 + 1)


def _majority_color(region_id: int, regions_by_id: dict[int, dict[str, object]], adjacency: dict[int, set[int]]) -> int | None:
    neighbors = adjacency.get(region_id, set())
    if not neighbors:
        return None

    counts = Counter(int(regions_by_id[nid]["color"]) for nid in neighbors)
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    if len(winners) != 1:
        return None
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor rectangular regions according to the majority color among grey-separated neighbors.

    Input structure:
    - bg = Black (0). Grey (5) one-cell-thick lines partition the canvas into closed colored regions.
    - Every region is a 4-connected monochrome area whose pixels are neither black nor grey.
    - Regions may be rectangular or non-rectangular (for example L- or T-shaped).

    Rule:
    - Identify every colored region and the adjacency graph induced by grey boundary segments.
    - For each region, count the colors of all adjacent regions, with each adjacent region contributing one vote.
    - If the region itself is a filled rectangle and there is a unique majority neighbor color, repaint the whole region to that color.
    - Non-rectangular regions, and rectangular regions whose neighbor-color vote is tied, keep their original color. Black background and grey dividers stay unchanged.

    Key insight:
    - The output depends on two hidden properties at once: the majority color among neighboring regions and whether the current region is a true rectangle rather than a concave shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    labels, regions = _component_regions(g)
    adjacency = _region_adjacency(labels, g)
    regions_by_id = {int(region["id"]): region for region in regions}

    out = g.copy()
    for region in regions:
        region_id = int(region["id"])
        coords = np.asarray(region["coords"], dtype=int)
        if not _is_rectangular(coords):
            continue
        winner = _majority_color(region_id, regions_by_id, adjacency)
        if winner is None:
            continue
        out[coords[:, 0], coords[:, 1]] = winner

    return out


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((c, -r) for r, c in cells)


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((r, -c) for r, c in cells)


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    current = cells
    for _ in range(4):
        seen.add(_normalize_shape(current))
        seen.add(_normalize_shape(_flip_shape(current)))
        current = _rotate_shape(current)
    return list(seen)


CONCAVE_SHAPES = [
    variant
    for base in (
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    )
    for variant in _shape_variants(base)
]

RECTANGULAR_MERGES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
]


def _shape_fits(
    anchor_r: int,
    anchor_c: int,
    shape: tuple[tuple[int, int], ...],
    used: set[tuple[int, int]],
    lattice_h: int,
    lattice_w: int,
) -> list[tuple[int, int]] | None:
    placed = []
    for dr, dc in shape:
        rr = anchor_r + dr
        cc = anchor_c + dc
        if not (0 <= rr < lattice_h and 0 <= cc < lattice_w):
            return None
        if (rr, cc) in used:
            return None
        placed.append((rr, cc))
    return placed


def _build_groups(lattice_h: int, lattice_w: int) -> list[list[tuple[int, int]]] | None:
    used: set[tuple[int, int]] = set()
    groups: list[list[tuple[int, int]]] = []

    mandatory_concave = 1 + (random.random() < 0.45)
    for _ in range(mandatory_concave):
        placed = None
        for _ in range(MULTI_REGION_ATTEMPTS):
            shape = random.choice(CONCAVE_SHAPES)
            max_r = lattice_h - max(r for r, _ in shape)
            max_c = lattice_w - max(c for _, c in shape)
            anchor_r = random.randrange(max_r)
            anchor_c = random.randrange(max_c)
            placed = _shape_fits(anchor_r, anchor_c, shape, used, lattice_h, lattice_w)
            if placed is not None:
                break
        if placed is None:
            return None
        groups.append(placed)
        used.update(placed)

    extra_merges = random.randint(1, 4)
    for _ in range(extra_merges):
        if random.random() < 0.7:
            shape = random.choice(RECTANGULAR_MERGES)
        else:
            shape = random.choice(CONCAVE_SHAPES)
        for _ in range(MULTI_REGION_ATTEMPTS):
            max_r = lattice_h - max(r for r, _ in shape)
            max_c = lattice_w - max(c for _, c in shape)
            anchor_r = random.randrange(max_r)
            anchor_c = random.randrange(max_c)
            placed = _shape_fits(anchor_r, anchor_c, shape, used, lattice_h, lattice_w)
            if placed is None:
                continue
            groups.append(placed)
            used.update(placed)
            break

    for r in range(lattice_h):
        for c in range(lattice_w):
            if (r, c) not in used:
                groups.append([(r, c)])

    if len(groups) < 6:
        return None
    return groups


def _owner_grid(groups: list[list[tuple[int, int]]], lattice_h: int, lattice_w: int) -> np.ndarray:
    owner = np.full((lattice_h, lattice_w), -1, dtype=int)
    for region_id, group in enumerate(groups):
        for r, c in group:
            owner[r, c] = region_id
    assert np.all(owner >= 0)
    return owner


def _group_is_rectangle(group: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in group]
    cols = [c for _, c in group]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return len(group) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _group_adjacency(owner: np.ndarray, group_count: int) -> dict[int, set[int]]:
    lattice_h, lattice_w = owner.shape
    adjacency = {region_id: set() for region_id in range(group_count)}
    for r in range(lattice_h):
        for c in range(lattice_w):
            current = int(owner[r, c])
            if r + 1 < lattice_h:
                other = int(owner[r + 1, c])
                if current != other:
                    adjacency[current].add(other)
                    adjacency[other].add(current)
            if c + 1 < lattice_w:
                other = int(owner[r, c + 1])
                if current != other:
                    adjacency[current].add(other)
                    adjacency[other].add(current)
    return adjacency


def _group_vote(color_by_region: dict[int, int], adjacency: dict[int, set[int]], region_id: int) -> tuple[int | None, bool]:
    neighbors = adjacency[region_id]
    if not neighbors:
        return None, False
    counts = Counter(color_by_region[nid] for nid in neighbors)
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    if len(winners) != 1:
        return None, True
    return int(winners[0]), False


def _assign_colors(adjacency: dict[int, set[int]], rectangular: dict[int, bool]) -> dict[int, int] | None:
    region_ids = sorted(adjacency)
    concave_ids = [rid for rid in region_ids if not rectangular[rid]]
    if not concave_ids:
        return None

    for _ in range(COLOR_ATTEMPTS):
        palette_size = random.choice((3, 4, 4, 5))
        palette = random.sample(FILL_COLORS, palette_size)
        color_by_region = {rid: random.choice(palette) for rid in region_ids}

        changed_rectangles = 0
        tied_rectangles = 0
        guarded_concaves = 0
        for rid in region_ids:
            winner, tied = _group_vote(color_by_region, adjacency, rid)
            if tied and rectangular[rid]:
                tied_rectangles += 1
            if winner is None:
                continue
            if rectangular[rid] and winner != color_by_region[rid]:
                changed_rectangles += 1
            if (not rectangular[rid]) and winner != color_by_region[rid]:
                guarded_concaves += 1

        if changed_rectangles and tied_rectangles and guarded_concaves and len(set(color_by_region.values())) >= 3:
            return color_by_region

    return None


def _render_grid(
    owner: np.ndarray,
    color_by_region: dict[int, int],
    cell_h: int,
    cell_w: int,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
) -> np.ndarray:
    lattice_h, lattice_w = owner.shape
    patch_h = lattice_h * (cell_h + 1) + 1
    patch_w = lattice_w * (cell_w + 1) + 1
    patch = np.full((patch_h, patch_w), GREY, dtype=int)

    for r in range(lattice_h):
        for c in range(lattice_w):
            region_id = int(owner[r, c])
            color = color_by_region[region_id]
            r0 = 1 + r * (cell_h + 1)
            c0 = 1 + c * (cell_w + 1)
            patch[r0 : r0 + cell_h, c0 : c0 + cell_w] = color

    for r in range(lattice_h):
        for c in range(lattice_w - 1):
            left = int(owner[r, c])
            right = int(owner[r, c + 1])
            if left != right:
                continue
            color = color_by_region[left]
            r0 = 1 + r * (cell_h + 1)
            wall_c = 1 + c * (cell_w + 1) + cell_w
            patch[r0 : r0 + cell_h, wall_c] = color

    for r in range(lattice_h - 1):
        for c in range(lattice_w):
            top = int(owner[r, c])
            bottom = int(owner[r + 1, c])
            if top != bottom:
                continue
            color = color_by_region[top]
            wall_r = 1 + r * (cell_h + 1) + cell_h
            c0 = 1 + c * (cell_w + 1)
            patch[wall_r, c0 : c0 + cell_w] = color

    for vr in range(1, lattice_h):
        for vc in range(1, lattice_w):
            around = [
                int(owner[vr - 1, vc - 1]),
                int(owner[vr - 1, vc]),
                int(owner[vr, vc - 1]),
                int(owner[vr, vc]),
            ]
            counts = Counter(around)
            region_id, count = counts.most_common(1)[0]
            if count >= 3:
                patch[vr * (cell_h + 1), vc * (cell_w + 1)] = color_by_region[region_id]

    canvas_h = patch_h + margin_top + margin_bottom
    canvas_w = patch_w + margin_left + margin_right
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    canvas[
        margin_top : margin_top + patch_h,
        margin_left : margin_left + patch_w,
    ] = patch
    return canvas


def _gen_attempt() -> np.ndarray | None:
    lattice_h = random.randint(*LATTICE_H_RANGE)
    lattice_w = random.randint(*LATTICE_W_RANGE)
    cell_h = random.choice(CELL_SIZE_CHOICES)
    cell_w = random.choice(CELL_SIZE_CHOICES)

    groups = _build_groups(lattice_h, lattice_w)
    if groups is None:
        return None

    owner = _owner_grid(groups, lattice_h, lattice_w)
    adjacency = _group_adjacency(owner, len(groups))
    rectangular = {region_id: _group_is_rectangle(group) for region_id, group in enumerate(groups)}

    if not any(not is_rect for is_rect in rectangular.values()):
        return None
    if not any(is_rect and len(adjacency[rid]) >= 2 for rid, is_rect in rectangular.items()):
        return None

    color_by_region = _assign_colors(adjacency, rectangular)
    if color_by_region is None:
        return None

    margin_top = random.choice(MARGIN_CHOICES)
    margin_bottom = random.choice(MARGIN_CHOICES)
    margin_left = random.choice(MARGIN_CHOICES)
    margin_right = random.choice(MARGIN_CHOICES)
    grid = _render_grid(owner, color_by_region, cell_h, cell_w, margin_top, margin_bottom, margin_left, margin_right)

    if max(grid.shape) > 30:
        return None

    # Confirm the generated sample actually exhibits the intended branches under the real solver.
    labels, regions = _component_regions(grid)
    adjacency_px = _region_adjacency(labels, grid)
    regions_by_id = {int(region["id"]): region for region in regions}
    changed_rectangles = 0
    tied_rectangles = 0
    guarded_concaves = 0
    for region in regions:
        region_id = int(region["id"])
        coords = np.asarray(region["coords"], dtype=int)
        winner = _majority_color(region_id, regions_by_id, adjacency_px)
        if winner is None and _is_rectangular(coords) and adjacency_px.get(region_id):
            neighbor_counts = Counter(int(regions_by_id[nid]["color"]) for nid in adjacency_px[region_id])
            if neighbor_counts:
                top = max(neighbor_counts.values())
                if sum(count == top for count in neighbor_counts.values()) > 1:
                    tied_rectangles += 1
        elif winner is not None:
            own = int(region["color"])
            if _is_rectangular(coords) and own != winner:
                changed_rectangles += 1
            if (not _is_rectangular(coords)) and own != winner:
                guarded_concaves += 1

    if not (changed_rectangles and tied_rectangles and guarded_concaves):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
