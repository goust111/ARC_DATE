from __future__ import annotations

from collections import deque
import random

import numpy as np


GRAY = 5
SIDE_LABELS = ("top", "bottom", "left", "right")
ALL_LABELS = SIDE_LABELS + ("tie",)
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _interior_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _wall_distances(shape: tuple[int, int], cell: tuple[int, int]) -> dict[str, int]:
    h, w = shape
    r, c = cell
    return {
        "top": int(r),
        "bottom": int(h - 1 - r),
        "left": int(c),
        "right": int(w - 1 - c),
    }


def _cell_label(shape: tuple[int, int], cell: tuple[int, int]) -> str:
    distances = _wall_distances(shape, cell)
    nearest = min(distances.values())
    winners = [side for side, dist in distances.items() if dist == nearest]
    return winners[0] if len(winners) == 1 else "tie"


def _classify_cells(shape: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    classes = {label: [] for label in ALL_LABELS}
    h, w = shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            classes[_cell_label(shape, (r, c))].append((r, c))
    return classes


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for cell in cells:
        if _interior_bounds(shape, cell):
            blocked.add(cell)
        for nb in _neighbors8(cell):
            if _interior_bounds(shape, nb):
                blocked.add(nb)
    return blocked


def _can_place(component: set[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    if not component:
        return False
    blocked = _halo(occupied, shape)
    return all(_interior_bounds(shape, cell) and cell not in blocked for cell in component)


def _find_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for nb in _neighbors4(cell):
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def _bfs_path(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell in goals:
            path: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = cell
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for nb in _neighbors4(cell):
            if not _interior_bounds(shape, nb):
                continue
            if nb in blocked or nb in parent:
                continue
            parent[nb] = cell
            queue.append(nb)
    return None


def _connect_anchors(
    anchors: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    if not anchors:
        return None
    component = {anchors[0]}
    blocked_base = _halo(occupied, shape)
    for anchor in anchors[1:]:
        blocked = set(blocked_base)
        blocked.difference_update(component)
        blocked.discard(anchor)
        path = _bfs_path(anchor, component, blocked, shape)
        if path is None:
            return None
        component.update(path)
    return component


def _grow_component(
    component: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    max_extra: int,
) -> set[tuple[int, int]]:
    out = set(component)
    for _ in range(max_extra):
        blocked = _halo(occupied, shape) | out
        frontier = []
        for cell in out:
            for nb in _neighbors4(cell):
                if not _interior_bounds(shape, nb):
                    continue
                if nb in blocked:
                    continue
                frontier.append(nb)
        if not frontier:
            break
        out.add(random.choice(frontier))
    return out


def _pick_tie_bridge(shape: tuple[int, int], occupied: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    tie_cells = _classify_cells(shape)["tie"][:]
    random.shuffle(tie_cells)
    for tie in tie_cells:
        if not _can_place({tie}, occupied, shape):
            continue
        side_neighbors = [nb for nb in _neighbors4(tie) if _interior_bounds(shape, nb) and _cell_label(shape, nb) != "tie"]
        if not side_neighbors:
            continue
        random.shuffle(side_neighbors)
        distinct_labels: list[tuple[str, tuple[int, int]]] = []
        seen: set[str] = set()
        for nb in side_neighbors:
            label = _cell_label(shape, nb)
            if label not in seen:
                distinct_labels.append((label, nb))
                seen.add(label)
        anchors = [tie, distinct_labels[0][1]]
        if len(distinct_labels) >= 2 and random.random() < 0.7:
            anchors.append(distinct_labels[1][1])
        component = set(anchors)
        for _label, cell in distinct_labels:
            for nb in _neighbors4(cell):
                if _interior_bounds(shape, nb) and _cell_label(shape, nb) == _cell_label(shape, cell):
                    component.add(nb)
                    break
        if _can_place(component, occupied, shape):
            return component
    return None


def _vertical_strip(shape: tuple[int, int], occupied: set[tuple[int, int]], side: str) -> set[tuple[int, int]] | None:
    h, w = shape
    assert side in {"top", "bottom"}
    columns = list(range(1, w - 1))
    random.shuffle(columns)
    for c in columns:
        cells: list[tuple[int, int]] = []
        if side == "top":
            r_iter = range(1, h - 1)
        else:
            r_iter = range(h - 2, 0, -1)
        for r in r_iter:
            cell = (r, c)
            if _cell_label(shape, cell) != side:
                break
            cells.append(cell)
        if len(cells) < 2:
            continue
        max_len = min(4, len(cells))
        lengths = list(range(2, max_len + 1))
        random.shuffle(lengths)
        for length in lengths:
            component = set(cells[:length])
            if _can_place(component, occupied, shape):
                return component
    return None


def _horizontal_strip(shape: tuple[int, int], occupied: set[tuple[int, int]], side: str) -> set[tuple[int, int]] | None:
    h, w = shape
    assert side in {"left", "right"}
    rows = list(range(1, h - 1))
    random.shuffle(rows)
    for r in rows:
        cells: list[tuple[int, int]] = []
        if side == "left":
            c_iter = range(1, w - 1)
        else:
            c_iter = range(w - 2, 0, -1)
        for c in c_iter:
            cell = (r, c)
            if _cell_label(shape, cell) != side:
                break
            cells.append(cell)
        if len(cells) < 2:
            continue
        max_len = min(4, len(cells))
        lengths = list(range(2, max_len + 1))
        random.shuffle(lengths)
        for length in lengths:
            component = set(cells[:length])
            if _can_place(component, occupied, shape):
                return component
    return None


def _pick_free_anchor(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    label: str,
) -> tuple[int, int] | None:
    candidates = _classify_cells(shape)[label][:]
    random.shuffle(candidates)
    blocked = _halo(occupied, shape)
    for cell in candidates:
        if cell not in blocked:
            return cell
    return None


def _small_blob(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    label: str,
) -> set[tuple[int, int]] | None:
    anchor = _pick_free_anchor(shape, occupied, label)
    if anchor is None:
        return None
    component = {anchor}
    growth_budget = random.randint(0, 2)
    for _ in range(growth_budget):
        blocked = _halo(occupied, shape) | component
        frontier = []
        for cell in component:
            for nb in _neighbors4(cell):
                if not _interior_bounds(shape, nb):
                    continue
                if nb in blocked:
                    continue
                if _cell_label(shape, nb) == label:
                    frontier.append(nb)
        if not frontier:
            break
        component.add(random.choice(frontier))
    return component if _can_place(component, occupied, shape) else None


def _pack_pixels(
    pixels: list[tuple[int, int]],
    shape: tuple[int, int],
) -> dict[str, dict[int, list[tuple[int, int]]]]:
    groups = {
        "top": {},
        "bottom": {},
        "left": {},
        "right": {},
    }
    for r, c in pixels:
        label = _cell_label(shape, (r, c))
        if label == "tie":
            continue
        if label in {"top", "bottom"}:
            groups[label].setdefault(c, []).append((r, c))
        else:
            groups[label].setdefault(r, []).append((r, c))
    return groups


def _solve_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5

    h, w = map(int, g.shape)
    bg = int(g[0, 0])
    assert int(g[0, -1]) == bg and int(g[-1, 0]) == bg and int(g[-1, -1]) == bg

    top_color = int(g[0, 1])
    bottom_color = int(g[-1, 1])
    left_color = int(g[1, 0])
    right_color = int(g[1, -1])
    assert len({bg, GRAY, top_color, bottom_color, left_color, right_color}) == 6
    assert np.all(g[0, 1:-1] == top_color)
    assert np.all(g[-1, 1:-1] == bottom_color)
    assert np.all(g[1:-1, 0] == left_color)
    assert np.all(g[1:-1, -1] == right_color)

    interior = g[1:-1, 1:-1]
    assert set(np.unique(interior)).issubset({bg, GRAY})

    gray_pixels = [tuple(map(int, cell)) for cell in np.argwhere(g == GRAY)]
    out = g.copy()
    out[1:-1, 1:-1] = bg

    for r, c in gray_pixels:
        if _cell_label(g.shape, (r, c)) == "tie":
            out[r, c] = GRAY

    groups = _pack_pixels(gray_pixels, g.shape)
    for c, coords in groups["top"].items():
        coords.sort(key=lambda rc: rc[0])
        for idx, _cell in enumerate(coords, start=1):
            assert out[idx, c] == bg
            out[idx, c] = top_color
    for c, coords in groups["bottom"].items():
        coords.sort(key=lambda rc: rc[0], reverse=True)
        for idx, _cell in enumerate(coords, start=1):
            rr = h - 1 - idx
            assert out[rr, c] == bg
            out[rr, c] = bottom_color
    for r, coords in groups["left"].items():
        coords.sort(key=lambda rc: rc[1])
        for idx, _cell in enumerate(coords, start=1):
            assert out[r, idx] == bg
            out[r, idx] = left_color
    for r, coords in groups["right"].items():
        coords.sort(key=lambda rc: rc[1], reverse=True)
        for idx, _cell in enumerate(coords, start=1):
            cc = w - 1 - idx
            assert out[r, cc] == bg
            out[r, cc] = right_color

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Split each gray fragment pixel among the four colored walls and pack it against its unique nearest wall.

    Input structure:
    - The grid border is a rectangular four-color frame: the top, bottom, left, and right walls are
      solid strips in four distinct colors, while the four corners stay background.
    - Inside the frame there are only background cells and several gray 4-connected fragments.

    Rule:
    - For each gray pixel, compare its perpendicular distances to the top, bottom, left, and right walls.
    - If exactly one wall is nearest, the pixel flies straight toward that wall, recolors to that wall's
      color, and packs tightly with other pixels assigned to the same wall in the same row or column,
      preserving their original order by distance to that wall.
    - If two or more walls are tied for nearest, that pixel stays in place and remains gray.

    Key insight:
    - The fragments do not move as whole objects: ownership is decided pixel by pixel, so one connected
      gray fragment can split across multiple walls while tie-line pixels remain as gray residue.
    """
    return _solve_from_grid(input_grid)


def _naive_projection(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(g[0, 0])
    out = g.copy()
    out[1:-1, 1:-1] = bg
    top_color = int(g[0, 1])
    bottom_color = int(g[-1, 1])
    left_color = int(g[1, 0])
    right_color = int(g[1, -1])
    colors = {
        "top": top_color,
        "bottom": bottom_color,
        "left": left_color,
        "right": right_color,
    }
    for r, c in np.argwhere(g == GRAY):
        rr = int(r)
        cc = int(c)
        label = _cell_label(g.shape, (rr, cc))
        if label == "tie":
            out[rr, cc] = GRAY
        elif label == "top":
            out[1, cc] = colors[label]
        elif label == "bottom":
            out[g.shape[0] - 2, cc] = colors[label]
        elif label == "left":
            out[rr, 1] = colors[label]
        else:
            out[rr, g.shape[1] - 2] = colors[label]
    return out


def _build_case(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    occupied: set[tuple[int, int]] = set()
    gray_cells: set[tuple[int, int]] = set()

    bridge = _pick_tie_bridge(shape, occupied)
    if bridge is None:
        return None
    gray_cells.update(bridge)
    occupied.update(bridge)

    vertical_side = random.choice(["top", "bottom"])
    vertical = _vertical_strip(shape, occupied, vertical_side)
    if vertical is None:
        return None
    gray_cells.update(vertical)
    occupied.update(vertical)

    horizontal_side = random.choice(["left", "right"])
    horizontal = _horizontal_strip(shape, occupied, horizontal_side)
    if horizontal is None:
        return None
    gray_cells.update(horizontal)
    occupied.update(horizontal)

    covered = {_cell_label(shape, cell) for cell in gray_cells}
    for label in SIDE_LABELS:
        if label in covered:
            continue
        blob = _small_blob(shape, occupied, label)
        if blob is None:
            return None
        gray_cells.update(blob)
        occupied.update(blob)

    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        label = random.choice(ALL_LABELS)
        if label == "tie":
            anchor = _pick_free_anchor(shape, occupied, "tie")
            if anchor is None:
                continue
            options = [
                nb
                for nb in _neighbors4(anchor)
                if _interior_bounds(shape, nb) and _cell_label(shape, nb) != "tie"
            ]
            random.shuffle(options)
            anchors = [anchor]
            if options:
                anchors.append(options[0])
            component = _connect_anchors(anchors, occupied, shape)
            if component is None:
                continue
        else:
            blob = _small_blob(shape, occupied, label)
            if blob is None:
                continue
            component = blob
        component = _grow_component(component, occupied, shape, max_extra=random.randint(0, 2))
        if _can_place(component, occupied, shape):
            gray_cells.update(component)
            occupied.update(component)

    components = _find_components(gray_cells)
    if not (3 <= len(components) <= 7):
        return None
    if not any(("tie" in {_cell_label(shape, cell) for cell in comp}) and len({_cell_label(shape, cell) for cell in comp}) >= 2 for comp in components):
        return None

    labels = {_cell_label(shape, cell) for cell in gray_cells}
    if labels != set(ALL_LABELS):
        return None

    packed = _pack_pixels(list(gray_cells), shape)
    has_stacking = any(len(coords) >= 2 for groups in packed.values() for coords in groups.values())
    if not has_stacking:
        return None

    total_interior = (h - 2) * (w - 2)
    if not (10 <= len(gray_cells) <= max(12, total_interior // 3)):
        return None

    bg, top, bottom, left, right = random.sample(PALETTE, 5)
    grid = np.full((h, w), bg, dtype=int)
    grid[0, 1:-1] = top
    grid[-1, 1:-1] = bottom
    grid[1:-1, 0] = left
    grid[1:-1, -1] = right
    for r, c in gray_cells:
        grid[r, c] = GRAY

    if np.array_equal(_solve_from_grid(grid), _naive_projection(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = (random.randint(9, 24), random.randint(9, 24))
        grid = _build_case(shape)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
