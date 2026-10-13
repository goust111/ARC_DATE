from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
COLOR_POOL = (1, 2, 3, 4)
GRID_SIDE_RANGE = (14, 18)
FRAME_COUNT_CHOICES = (3, 4)
FRAME_SIZE_RANGE = (4, 8)
PLACE_TRIES = 400
GEN_TRIES = 800


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _bboxes_share_row_or_col(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _frame_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _frame_cells(bbox: BBox) -> set[Coord]:
    return {(int(r), int(c)) for r, c in np.argwhere(_frame_mask((bbox[2] + 1, bbox[3] + 1), bbox))}


def _frame_corners(bbox: BBox) -> dict[str, Coord]:
    r0, c0, r1, c1 = bbox
    return {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"coords": coords, "bbox": _bbox_from_coords(coords), "area": len(cells)})
    return comps


def _merged_fragment_groups(components: list[dict]) -> list[dict]:
    groups = [{"coords": comp["coords"].copy(), "bbox": comp["bbox"]} for comp in components]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if _bboxes_share_row_or_col(groups[i]["bbox"], groups[j]["bbox"]):
                    coords = np.vstack([groups[i]["coords"], groups[j]["coords"]])
                    groups[i] = {"coords": coords, "bbox": _bbox_from_coords(coords)}
                    del groups[j]
                    changed = True
                    break
    return groups


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lower, higher in edges:
        if lower == higher:
            raise AssertionError("self-overlap cannot define layer order")
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        if len(available) != 1:
            raise AssertionError("overlap constraints must determine a unique total order")
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def _reconstruct_frames(grid: np.ndarray) -> tuple[list[int], dict[int, BBox], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != BG)
    assert 3 <= len(colors) <= 4, "expected 3-4 foreground colors"

    bboxes: dict[int, BBox] = {}
    frames: dict[int, np.ndarray] = {}
    for color in colors:
        components = _components_of_color(g, color)
        assert 3 <= len(components) <= 6, "each color must have 3-6 fragments"
        assert all(1 <= comp["area"] <= 3 for comp in components), "each fragment must have 1-3 pixels"
        groups = _merged_fragment_groups(components)
        assert len(groups) == 1, "same-color fragments must merge to one group"
        bbox = groups[0]["bbox"]
        r0, c0, r1, c1 = bbox
        assert 4 <= (r1 - r0 + 1) <= 8 and 4 <= (c1 - c0 + 1) <= 8, "reconstructed frame size out of range"
        frame = _frame_mask(g.shape, bbox)
        visible = g == color
        assert np.all(frame[visible]), "visible fragment pixels must lie on the reconstructed frame"
        bboxes[color] = bbox
        frames[color] = frame
    return colors, bboxes, frames


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct fragmented color frames and output their unique bottom-to-top stacking order.

    Input structure:
    - The grid is mostly black background.
    - There are 3-4 foreground colors, each split into 3-6 disconnected 4-connected fragments of 1-3 pixels.
    - Fragments of the same color belong to one hidden rectangular frame, but they may only align by shared row-span or shared column-span rather than by direct contact.

    Rule:
    - For each color independently, find its 4-connected fragments and merge them transitively whenever two fragment bounding boxes overlap in row span or in column span.
    - Reconstruct the color's object as the hollow rectangular frame given by the merged bounding box.
    - At any cell covered by multiple reconstructed frames, the visible input color is the top frame there, so every other covering frame must lie below it.
    - Topologically sort those overlap relations into the unique bottom-to-top order, then output that color order as an N x 1 column.

    Key insight:
    - The decisive step is not component connectivity but transitive row/column-span merging: only the merged bounding boxes reveal the full frames, and only those rebuilt frames reveal the hidden overlaps that determine stack order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors, _bboxes, frames = _reconstruct_frames(g)

    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frames[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            if top == BG:
                continue
            assert top in covering, "visible overlap pixel must belong to one covering frame"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    assert edges, "need overlap evidence to determine order"
    order = _unique_layer_order(colors, edges)
    return np.array(order, dtype=int).reshape(-1, 1)


def _randint_pair(low: int, high: int) -> int:
    return random.randint(low, high)


def _random_bbox(h: int, w: int) -> BBox:
    fh = _randint_pair(*FRAME_SIZE_RANGE)
    fw = _randint_pair(*FRAME_SIZE_RANGE)
    r0 = random.randint(0, h - fh)
    c0 = random.randint(0, w - fw)
    return (r0, c0, r0 + fh - 1, c0 + fw - 1)


def _place_next_bbox(shape: tuple[int, int], prev_bbox: BBox, older_cell_sets: list[set[Coord]]) -> tuple[BBox, set[Coord]] | None:
    h, w = shape
    pr0, pc0, pr1, pc1 = prev_bbox
    prev_cells = older_cell_sets[-1]
    older = older_cell_sets[:-1]

    for _ in range(PLACE_TRIES):
        fh = _randint_pair(*FRAME_SIZE_RANGE)
        fw = _randint_pair(*FRAME_SIZE_RANGE)
        rmin = max(0, pr0 - fh + 1)
        rmax = min(h - fh, pr1)
        cmin = max(0, pc0 - fw + 1)
        cmax = min(w - fw, pc1)
        if rmin > rmax or cmin > cmax:
            continue
        r0 = random.randint(rmin, rmax)
        c0 = random.randint(cmin, cmax)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        cells = _frame_cells(bbox)
        if len(cells & prev_cells) not in (2, 4):
            continue
        if any(cells & prior for prior in older):
            continue
        return bbox, cells
    return None


def _render_full_stack(shape: tuple[int, int], order: list[int], bboxes: list[BBox]) -> tuple[np.ndarray, list[set[Coord]], dict[int, set[Coord]]]:
    grid = np.zeros(shape, dtype=int)
    frame_cell_sets = [_frame_cells(bbox) for bbox in bboxes]
    for color, cells in zip(order, frame_cell_sets):
        for r, c in cells:
            grid[r, c] = color
    visible = {color: {(int(r), int(c)) for r, c in np.argwhere(grid == color)} for color in order}
    return grid, frame_cell_sets, visible


def _cell_has_other_neighbor(cell: Coord, groups: list[set[Coord]], idx: int) -> bool:
    r, c = cell
    for j, group in enumerate(groups):
        if j == idx:
            continue
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in group:
                return True
    return False


def _try_grow_group(group: set[Coord], groups: list[set[Coord]], idx: int, visible_cells: set[Coord], used: set[Coord]) -> bool:
    candidates: list[Coord] = []
    for r, c in group:
        for nr, nc in _neighbors4(r, c):
            cell = (nr, nc)
            if cell in used or cell not in visible_cells:
                continue
            if _cell_has_other_neighbor(cell, groups, idx):
                continue
            candidates.append(cell)
    if not candidates:
        return False
    random.shuffle(candidates)
    group.add(candidates[0])
    used.add(candidates[0])
    return True


def _rasterize_groups(shape: tuple[int, int], color: int, groups: list[set[Coord]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for group in groups:
        for r, c in group:
            grid[r, c] = color
    return grid


def _groups_valid(shape: tuple[int, int], color: int, groups: list[set[Coord]], target_bbox: BBox, visible_cells: set[Coord], required_sets: list[set[Coord]]) -> bool:
    if not (3 <= len(groups) <= 6):
        return False
    if any(not (1 <= len(group) <= 3) for group in groups):
        return False
    used = set().union(*groups) if groups else set()
    if not used <= visible_cells:
        return False
    for required in required_sets:
        if not (used & required):
            return False

    grid = _rasterize_groups(shape, color, groups)
    comps = _components_of_color(grid, color)
    if len(comps) != len(groups):
        return False
    if any(not (1 <= comp["area"] <= 3) for comp in comps):
        return False
    merged = _merged_fragment_groups(comps)
    if len(merged) != 1 or merged[0]["bbox"] != target_bbox:
        return False
    return True


def _build_fragment_groups(shape: tuple[int, int], bbox: BBox, visible_cells: set[Coord], required_sets: list[set[Coord]]) -> list[set[Coord]] | None:
    corners = _frame_corners(bbox)
    corner_templates = [
        ("tl", "tr", "bl", "br"),
        ("tl", "tr", "bl"),
        ("tl", "tr", "br"),
        ("tl", "bl", "br"),
        ("tr", "bl", "br"),
    ]

    for _ in range(120):
        templates = corner_templates[:]
        random.shuffle(templates)
        for template in templates:
            if not all(corners[name] in visible_cells for name in template):
                continue
            groups = [{corners[name]} for name in template]
            used = set().union(*groups)
            ok = True

            for required in required_sets:
                options = [cell for cell in required if cell in visible_cells]
                random.shuffle(options)
                placed = False
                for cell in options:
                    if cell in used:
                        placed = True
                        break

                    attachable = []
                    for idx, group in enumerate(groups):
                        if len(group) >= 3:
                            continue
                        if any(cell in _neighbors4(r, c) for r, c in group):
                            if not _cell_has_other_neighbor(cell, groups, idx):
                                attachable.append(idx)
                    if attachable:
                        idx = random.choice(attachable)
                        groups[idx].add(cell)
                        used.add(cell)
                        placed = True
                        break

                    if any(cell in _neighbors4(r, c) for r, c in used):
                        continue
                    groups.append({cell})
                    used.add(cell)
                    placed = True
                    break

                if not placed:
                    ok = False
                    break

            if not ok:
                continue

            order = list(range(len(groups)))
            random.shuffle(order)
            for idx in order:
                target_size = random.randint(len(groups[idx]), 3)
                while len(groups[idx]) < target_size:
                    if not _try_grow_group(groups[idx], groups, idx, visible_cells, used):
                        break

            if _groups_valid(shape, 1, groups, bbox, visible_cells, required_sets):
                return groups
    return None


def _explicit_semantic_checks(
    grid: np.ndarray,
    order: list[int],
    bboxes: list[BBox],
    witness_cells: dict[int, Coord],
) -> bool:
    out = main(grid)
    expected = np.array(order, dtype=int).reshape(-1, 1)
    if not np.array_equal(out, expected):
        return False

    color_to_bbox = {color: bbox for color, bbox in zip(order, bboxes)}
    frame_masks = {color: _frame_mask(grid.shape, bbox) for color, bbox in color_to_bbox.items()}
    for index in range(1, len(order)):
        lower = order[index - 1]
        higher = order[index]
        witness = witness_cells[higher]
        if int(grid[witness]) != higher:
            return False
        if not frame_masks[lower][witness] or not frame_masks[higher][witness]:
            return False
        if any(frame_masks[order[j]][witness] for j in range(index + 1, len(order))):
            return False
    return True


def _generate_one() -> np.ndarray | None:
    h = _randint_pair(*GRID_SIDE_RANGE)
    w = _randint_pair(*GRID_SIDE_RANGE)
    n = random.choice(FRAME_COUNT_CHOICES)
    order = random.sample(list(COLOR_POOL), n)

    first_bbox = _random_bbox(h, w)
    bboxes = [first_bbox]
    cell_sets = [_frame_cells(first_bbox)]
    for _ in range(1, n):
        placed = _place_next_bbox((h, w), bboxes[-1], cell_sets)
        if placed is None:
            return None
        bbox, cells = placed
        bboxes.append(bbox)
        cell_sets.append(cells)

    full_grid, frame_cell_sets, visible_cells = _render_full_stack((h, w), order, bboxes)

    for color, bbox in zip(order, bboxes):
        if _bbox_from_mask(full_grid == color) != bbox:
            return None
        if not set(_frame_corners(bbox).values()) <= visible_cells[color]:
            return None

    required_sets: dict[int, list[set[Coord]]] = {color: [] for color in order}
    witness_cells: dict[int, Coord] = {}
    for index in range(1, n):
        higher = order[index]
        overlap = frame_cell_sets[index - 1] & frame_cell_sets[index]
        if len(overlap) < 2:
            return None
        visible_overlap = [cell for cell in overlap if cell in visible_cells[higher]]
        if not visible_overlap:
            return None
        required_sets[higher].append(set(visible_overlap))
        witness_cells[higher] = random.choice(visible_overlap)

    inp = np.zeros((h, w), dtype=int)
    for index, color in enumerate(order):
        groups = _build_fragment_groups((h, w), bboxes[index], visible_cells[color], required_sets[color])
        if groups is None:
            return None
        for group in groups:
            for r, c in group:
                inp[r, c] = color

    for color in order:
        comps = _components_of_color(inp, color)
        if not (3 <= len(comps) <= 6):
            return None
        if any(not (1 <= comp["area"] <= 3) for comp in comps):
            return None
        merged = _merged_fragment_groups(comps)
        if len(merged) != 1:
            return None

    if not _explicit_semantic_checks(inp, order, bboxes, witness_cells):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragmented-frame stacking example")
