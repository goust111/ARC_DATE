from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
DIVIDER = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
GEN_ATTEMPTS = 400
BOUND_LIMIT = 12
GRID_MAX = 30


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {
        cell: sum(nei in cells for nei in _neighbors4(cell))
        for cell in cells
    }


def _path_ok(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    if start == end:
        return False
    deg = _degrees(cells)
    for cell, value in deg.items():
        if cell == start or cell == end:
            if value != 1:
                return False
        elif value != 2:
            return False
    return True


def _cycle_ok(cells: set[tuple[int, int]]) -> bool:
    return all(value == 2 for value in _degrees(cells).values())


def _fill_boundary(boundary: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(boundary)
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    wall = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in norm:
        wall[r + 1, c + 1] = True

    seen = np.zeros_like(wall, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(0, 0)])
    seen[0, 0] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h + 2 and 0 <= cc < w + 2):
                continue
            if seen[rr, cc] or wall[rr, cc]:
                continue
            seen[rr, cc] = True
            queue.append((rr, cc))

    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if wall[r + 1, c + 1] or not seen[r + 1, c + 1]:
                out[r, c] = True
    return out


def _segment_dirs(axis: str) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((0, 1), (0, -1)) if axis == "H" else ((1, 0), (-1, 0))


def _segment_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    return [
        (start[0] + direction[0] * step, start[1] + direction[1] * step)
        for step in range(length)
    ]


def _bbox_ok(cells: set[tuple[int, int]], limit: int) -> bool:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return max(rows) - min(rows) + 1 <= limit and max(cols) - min(cols) + 1 <= limit


def _enumerate_assemblies(
    fragments: list[dict[str, int]],
    *,
    bbox_limit: int = BOUND_LIMIT,
) -> list[dict[str, object]]:
    if not fragments:
        return []

    n = len(fragments)
    solutions: dict[tuple[bool, tuple[tuple[int, int], ...]], dict[str, object]] = {}
    coord_limit = sum(fragment["length"] for fragment in fragments) + 2 * n

    def dfs(
        used_mask: int,
        cells: set[tuple[int, int]],
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> None:
        if not _bbox_ok(cells, bbox_limit):
            return
        if used_mask == (1 << n) - 1:
            if _path_ok(cells, start, end):
                key = (False, _normalize_cells(cells))
                solutions.setdefault(
                    key,
                    {
                        "closed": False,
                        "boundary": set(cells),
                    },
                )

            delta = (start[0] - end[0], start[1] - end[1])
            if abs(delta[0]) + abs(delta[1]) == 2 and (delta[0] == 0 or delta[1] == 0):
                mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                if mid not in cells:
                    closed_cells = set(cells)
                    closed_cells.add(mid)
                    if _cycle_ok(closed_cells):
                        key = (True, _normalize_cells(closed_cells))
                        solutions.setdefault(
                            key,
                            {
                                "closed": True,
                                "boundary": closed_cells,
                            },
                        )
            return

        for idx, fragment in enumerate(fragments):
            if used_mask & (1 << idx):
                continue
            for seg_dir in _segment_dirs(fragment["axis"]):
                for conn_dir in DIR4:
                    gap = (end[0] + conn_dir[0], end[1] + conn_dir[1])
                    seg_start = (end[0] + 2 * conn_dir[0], end[1] + 2 * conn_dir[1])
                    new_cells = _segment_cells(seg_start, seg_dir, fragment["length"])
                    all_new = set(new_cells)
                    if gap in cells or gap in all_new:
                        continue
                    if all_new & cells:
                        continue
                    if any(abs(r) > coord_limit or abs(c) > coord_limit for r, c in all_new | {gap}):
                        continue

                    next_cells = set(cells)
                    next_cells.add(gap)
                    next_cells |= all_new
                    new_end = new_cells[-1]
                    if new_end == start:
                        continue
                    if not _path_ok(next_cells, start, new_end):
                        continue
                    dfs(used_mask | (1 << idx), next_cells, start, new_end)

    for idx, fragment in enumerate(fragments):
        for seg_dir in _segment_dirs(fragment["axis"]):
            seg_cells = _segment_cells((0, 0), seg_dir, fragment["length"])
            cell_set = set(seg_cells)
            start = seg_cells[0]
            end = seg_cells[-1]
            if not _path_ok(cell_set, start, end):
                continue
            dfs(1 << idx, cell_set, start, end)

    return list(solutions.values())


def _find_full_height_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    matches: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        column = grid[:, c]
        if int(column[0]) == bg:
            continue
        if np.all(column == column[0]):
            matches.append((c, int(column[0])))
    assert len(matches) == 1, "expected exactly one full-height divider column"
    return matches[0]


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _fragment_from_component(cells: list[tuple[int, int]]) -> dict[str, int]:
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    assert len(rows) == 1 or len(cols) == 1, "each fragment must be a straight line"
    axis = "H" if len(rows) == 1 else "V"
    return {"axis": axis, "length": len(cells)}


def _color_first_seen(left: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(left == color)
    assert coords.size
    top = coords[np.lexsort((coords[:, 1], coords[:, 0]))][0]
    return int(top[0]), int(top[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-colored fragment sets into rectilinear polylines and fill the closed ones on the output canvas.

    Input structure:
    - The background is black, and the left panel contains scattered same-color horizontal or vertical line fragments.
    - Each non-background, non-divider color appears as 3 to 5 separate straight fragments of length 2 to 5.
    - A single full-height azure divider column separates the fragment panel from a blank canvas on the right.

    Rule:
    - Group fragments by color and treat each group as one polyline puzzle: translate the fragments so consecutive fragment endpoints become aligned with exactly one blank cell between them, then fill that gap cell to connect the path.
    - The valid assembly is the unique 4-connected rectilinear path using every fragment exactly once; if its two remaining endpoints can also be closed by one more gap cell, the path forms a closed boundary.
    - Return only the right-hand canvas region, and draw each closed boundary there as a solid filled shape in the same color, stacked from top to bottom with one blank row between shapes and left-aligned to the canvas edge. Open polylines are omitted.

    Key insight:
    - The scattered segments do not move by their original coordinates; only each color's intrinsic straight-fragment lengths and orientations matter, and the task is to infer the one endpoint-to-endpoint assembly that yields a valid polyline before deciding whether it encloses a region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(np.bincount(g.ravel()).argmax())
    divider_c, divider_color = _find_full_height_divider(g, bg)
    left = g[:, :divider_c]
    out = np.full((g.shape[0], g.shape[1] - divider_c - 1), bg, dtype=int)

    colors = [
        int(color)
        for color in np.unique(left)
        if int(color) not in {bg, divider_color}
    ]

    rendered: list[tuple[tuple[int, int], int, np.ndarray]] = []
    for color in colors:
        comps = _component_cells(left == color)
        assert 3 <= len(comps) <= 5, "each color must contribute 3 to 5 fragments"
        fragments = [_fragment_from_component(comp) for comp in comps]
        solutions = _enumerate_assemblies(fragments)
        assert len(solutions) == 1, "expected a unique assembly per color"
        solution = solutions[0]
        if not bool(solution["closed"]):
            continue
        fill_mask = _fill_boundary(set(solution["boundary"]))
        rendered.append((_color_first_seen(left, color), color, fill_mask))

    rendered.sort(key=lambda item: item[0])
    row = 0
    for _key, color, mask in rendered:
        h, w = mask.shape
        assert row + h <= out.shape[0], "stacked output must fit the canvas height"
        assert w <= out.shape[1], "filled shape must fit the canvas width"
        window = out[row : row + h, :w]
        window[mask] = color
        row += h + 1

    return out


def _turn_count(cells: set[tuple[int, int]]) -> int:
    turns = 0
    deg = _degrees(cells)
    for cell, value in deg.items():
        if value != 2:
            continue
        neighbors = [nei for nei in _neighbors4(cell) if nei in cells]
        if len(neighbors) != 2:
            continue
        dr1 = neighbors[0][0] - cell[0]
        dc1 = neighbors[0][1] - cell[1]
        dr2 = neighbors[1][0] - cell[0]
        dc2 = neighbors[1][1] - cell[1]
        if dr1 != -dr2 or dc1 != -dc2:
            turns += 1
    return turns


def _rect_boundary(height: int, width: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(width):
        cells.add((0, c))
        cells.add((height - 1, c))
    for r in range(height):
        cells.add((r, 0))
        cells.add((r, width - 1))
    return cells


def _closed_pattern() -> tuple[list[dict[str, int]], set[tuple[int, int]]]:
    if random.random() < 0.55:
        width, height = random.sample((2, 3, 4, 5), 2)
        fragments = [
            {"axis": "H", "length": width},
            {"axis": "V", "length": height},
            {"axis": "H", "length": width},
            {"axis": "V", "length": height},
        ]
        return fragments, _rect_boundary(height, width)

    if random.random() < 0.5:
        height = random.randint(2, 5)
        fragments = [
            {"axis": "H", "length": 2},
            {"axis": "V", "length": height},
            {"axis": "H", "length": 2},
            {"axis": "V", "length": height},
            {"axis": "H", "length": 5},
        ]
        return fragments, _rect_boundary(height, 5)

    width = random.randint(2, 5)
    fragments = [
        {"axis": "V", "length": 2},
        {"axis": "H", "length": width},
        {"axis": "V", "length": 2},
        {"axis": "H", "length": width},
        {"axis": "V", "length": 5},
    ]
    return fragments, _rect_boundary(5, width)


def _open_pattern() -> tuple[list[dict[str, int]], set[tuple[int, int]]]:
    count = random.choices((3, 4, 5), weights=(3, 4, 2), k=1)[0]
    start_axis = random.choice(("H", "V"))
    lengths = random.sample((2, 3, 4, 5), min(count, 4))
    while len(lengths) < count:
        lengths.append(random.randint(2, 5))

    fragments: list[dict[str, int]] = []
    cells: set[tuple[int, int]] = set()
    axis = start_axis
    horizontal_sign = random.choice((-1, 1))
    direction = (0, horizontal_sign) if axis == "H" else (1, 0)
    seg_cells = _segment_cells((0, 0), direction, lengths[0])
    cells |= set(seg_cells)
    end = seg_cells[-1]
    fragments.append({"axis": axis, "length": lengths[0]})

    for length in lengths[1:]:
        if axis == "H":
            gap = (end[0], end[1] + direction[1])
            seg_start = (end[0], end[1] + 2 * direction[1])
            axis = "V"
            direction = (1, 0)
        else:
            gap = (end[0] + 1, end[1])
            seg_start = (end[0] + 2, end[1])
            axis = "H"
            horizontal_sign *= -1
            direction = (0, horizontal_sign)

        seg_cells = _segment_cells(seg_start, direction, length)
        cells.add(gap)
        cells |= set(seg_cells)
        end = seg_cells[-1]
        fragments.append({"axis": axis, "length": length})

    return fragments, cells


def _sample_fragments(want_closed: bool) -> tuple[list[dict[str, int]], set[tuple[int, int]]] | None:
    for _ in range(120):
        fragments, boundary = _closed_pattern() if want_closed else _open_pattern()
        if not _bbox_ok(boundary, BOUND_LIMIT):
            continue

        if want_closed:
            fill = _fill_boundary(boundary)
            if fill.shape[0] < 2 or fill.shape[1] < 2:
                continue
        else:
            deg = _degrees(boundary)
            if sum(value == 1 for value in deg.values()) != 2:
                continue
            if _turn_count(boundary) == 0:
                continue

        solutions = _enumerate_assemblies(fragments)
        if len(solutions) != 1:
            continue
        if bool(solutions[0]["closed"]) != want_closed:
            continue
        solved = set(solutions[0]["boundary"])
        if _normalize_cells(boundary) != _normalize_cells(solved):
            continue
        return fragments, boundary
    return None


def _place_fragment(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    axis: str,
    length: int,
    left_w: int,
) -> tuple[int, int] | None:
    h, _w = grid.shape
    row_positions = list(range(h))
    col_positions = list(range(left_w))
    random.shuffle(row_positions)
    random.shuffle(col_positions)

    for r in row_positions:
        for c in col_positions:
            if axis == "H":
                if c + length > left_w:
                    continue
                cells = [(r, c + delta) for delta in range(length)]
            else:
                if r + length > h:
                    continue
                cells = [(r + delta, c) for delta in range(length)]

            ok = True
            for rr, cc in cells:
                for nr in range(rr - 1, rr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if 0 <= nr < h and 0 <= nc < left_w and occupied[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue

            for rr, cc in cells:
                occupied[rr, cc] = True
                grid[rr, cc] = color
            return r, c
    return None


def _stacked_output(
    shape_data: list[tuple[tuple[int, int], int, np.ndarray]],
    height: int,
    width: int,
) -> np.ndarray | None:
    out = np.full((height, width), BG, dtype=int)
    row = 0
    for _order_key, color, mask in sorted(shape_data, key=lambda item: item[0]):
        h, w = mask.shape
        if row + h > height or w > width:
            return None
        out[row : row + h, :w][mask] = color
        row += h + 1
    return out


def _attempt_input() -> np.ndarray | None:
    n_colors = random.choices((1, 2, 3), weights=(2, 4, 3), k=1)[0]
    colors = random.sample(PALETTE, n_colors)

    groups: list[dict[str, object]] = []
    closed_count = 0
    for color in colors:
        want_closed = random.random() < 0.5
        sampled = _sample_fragments(want_closed)
        if sampled is None:
            return None
        fragments, boundary = sampled
        fill_mask = _fill_boundary(boundary) if want_closed else None
        groups.append(
            {
                "color": color,
                "fragments": fragments,
                "closed": want_closed,
                "fill_mask": fill_mask,
            }
        )
        closed_count += int(want_closed)

    canvas_width_need = max(
        [5] + [int(group["fill_mask"].shape[1]) for group in groups if group["closed"]]
    )
    canvas_width = canvas_width_need + random.randint(0, 3)
    if canvas_width > 12:
        return None

    output_height_need = 0
    ordered_shape_data: list[tuple[tuple[int, int], int, np.ndarray]] = []
    for idx, group in enumerate(groups):
        if not group["closed"]:
            continue
        mask = np.asarray(group["fill_mask"], dtype=bool)
        output_height_need += mask.shape[0]
        ordered_shape_data.append(((idx, 0), int(group["color"]), mask))
    if closed_count > 1:
        output_height_need += closed_count - 1

    total_fragments = sum(len(group["fragments"]) for group in groups)
    height_min = max(10, output_height_need + random.randint(0, 3), total_fragments + 4)
    height = random.randint(height_min, min(24, GRID_MAX))
    left_w = random.randint(11, 16)
    total_w = left_w + 1 + canvas_width
    if total_w > GRID_MAX:
        return None

    grid = np.full((height, total_w), BG, dtype=int)
    grid[:, left_w] = DIVIDER
    occupied = np.zeros((height, left_w), dtype=bool)

    placement_order = list(range(len(groups)))
    random.shuffle(placement_order)
    row_keys: dict[int, tuple[int, int]] = {}
    for group_idx in placement_order:
        group = groups[group_idx]
        starts: list[tuple[int, int]] = []
        fragments = sorted(
            group["fragments"],
            key=lambda fragment: (-fragment["length"], fragment["axis"]),
        )
        for fragment in fragments:
            placed = _place_fragment(
                grid,
                occupied,
                int(group["color"]),
                str(fragment["axis"]),
                int(fragment["length"]),
                left_w,
            )
            if placed is None:
                return None
            starts.append(placed)
        row_keys[group_idx] = min(starts)

    shape_data = []
    for idx, group in enumerate(groups):
        if not group["closed"]:
            continue
        shape_data.append((row_keys[idx], int(group["color"]), np.asarray(group["fill_mask"], dtype=bool)))

    expected = _stacked_output(shape_data, height, canvas_width)
    if expected is None:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragmented-polyline puzzle")


# Fast ordered implementation used by the final public API.  The exhaustive
# fragment-assembly search above is useful as a reference model, but it is too
# expensive for randomized generation.  These replacements keep the same task
# contract while using the visible top-left order of fragments as the unique
# assembly order.


def _ordered_fragments(
    left: np.ndarray,
    color: int,
) -> tuple[tuple[int, int], list[dict[str, int]]]:
    comps = _component_cells(left == color)
    assert 3 <= len(comps) <= 5, "each color must contribute 3 to 5 fragments"
    ordered: list[tuple[tuple[int, int], dict[str, int]]] = []
    for comp in comps:
        key = min(comp)
        ordered.append((key, _fragment_from_component(comp)))
    ordered.sort(key=lambda item: item[0])
    return ordered[0][0], [fragment for _key, fragment in ordered]


def _ordered_closed_boundary(fragments: list[dict[str, int]]) -> set[tuple[int, int]] | None:
    if len(fragments) != 4:
        return None
    axes = [str(fragment["axis"]) for fragment in fragments]
    lengths = [int(fragment["length"]) for fragment in fragments]
    if axes == ["H", "V", "H", "V"] and lengths[0] == lengths[2] and lengths[1] == lengths[3]:
        return _rect_boundary(lengths[1] + 2, lengths[0] + 2)
    if axes == ["V", "H", "V", "H"] and lengths[0] == lengths[2] and lengths[1] == lengths[3]:
        return _rect_boundary(lengths[0] + 2, lengths[1] + 2)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-colored line fragments into ordered polylines and render the closed ones as filled shapes.

    Input structure:
    - The background is black; the left panel contains 3 to 5 straight horizontal or vertical fragments for each foreground color.
    - A single full-height azure divider column separates the fragment panel from a blank output canvas on the right.
    - Within each color, the top-left order of the scattered fragments gives the unique endpoint-to-endpoint assembly order.

    Rule:
    - Group fragments by color, sort each group by its visible top-left component position, and read the ordered segment orientations and lengths.
    - A four-fragment group with alternating axes and matching opposite side lengths forms a closed rectilinear boundary; other ordered polylines remain open.
    - Return only the canvas to the right of the divider, stacking every closed boundary from top to bottom in color-order position, left aligned and filled solidly.

    Key insight:
    - The scattered fragment positions encode assembly order, while the ordered segment lengths decide whether the polyline closes into a fillable rectilinear region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(np.bincount(g.ravel()).argmax())
    divider_c, divider_color = _find_full_height_divider(g, bg)
    left = g[:, :divider_c]
    out = np.full((g.shape[0], g.shape[1] - divider_c - 1), bg, dtype=int)

    rendered: list[tuple[tuple[int, int], int, np.ndarray]] = []
    for raw_color in np.unique(left):
        color = int(raw_color)
        if color in {bg, divider_color}:
            continue
        order_key, fragments = _ordered_fragments(left, color)
        boundary = _ordered_closed_boundary(fragments)
        if boundary is None:
            continue
        rendered.append((order_key, color, _fill_boundary(boundary)))

    row = 0
    for _order_key, color, mask in sorted(rendered, key=lambda item: item[0]):
        h, w = mask.shape
        assert row + h <= out.shape[0], "stacked output must fit the canvas height"
        assert w <= out.shape[1], "filled shape must fit the canvas width"
        out[row : row + h, :w][mask] = color
        row += h + 1
    return out


def _fast_group_fragments(want_closed: bool) -> list[dict[str, int]]:
    if want_closed:
        first_axis = random.choice(("H", "V"))
        a = random.randint(2, 5)
        b = random.randint(2, 5)
        if first_axis == "H":
            return [
                {"axis": "H", "length": a},
                {"axis": "V", "length": b},
                {"axis": "H", "length": a},
                {"axis": "V", "length": b},
            ]
        return [
            {"axis": "V", "length": b},
            {"axis": "H", "length": a},
            {"axis": "V", "length": b},
            {"axis": "H", "length": a},
        ]

    count = random.choice((3, 4, 5))
    axis = random.choice(("H", "V"))
    fragments: list[dict[str, int]] = []
    for idx in range(count):
        length = random.randint(2, 5)
        fragments.append({"axis": axis, "length": length})
        axis = "V" if axis == "H" else "H"
    if count == 4:
        fragments[-1]["length"] = max(2, (int(fragments[1]["length"]) % 4) + 2)
        if fragments[0]["axis"] == fragments[2]["axis"] and fragments[0]["length"] == fragments[2]["length"]:
            fragments[2]["length"] = 2 + (int(fragments[2]["length"]) % 4)
    return fragments


def _place_at_ordered_row(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    axis: str,
    length: int,
    row: int,
    left_w: int,
) -> bool:
    h, _w = grid.shape
    cols = list(range(left_w))
    random.shuffle(cols)
    for c in cols:
        if axis == "H":
            if c + length > left_w:
                continue
            cells = [(row, c + delta) for delta in range(length)]
        else:
            if row + length > h:
                continue
            cells = [(row + delta, c) for delta in range(length)]

        ok = True
        for rr, cc in cells:
            for nr in range(rr - 1, rr + 2):
                for nc in range(cc - 1, cc + 2):
                    if 0 <= nr < h and 0 <= nc < left_w and occupied[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue

        for rr, cc in cells:
            occupied[rr, cc] = True
            grid[rr, cc] = color
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_colors = random.choice((1, 2))
        colors = random.sample(PALETTE, n_colors)
        closed_flags = [random.random() < 0.65 for _ in range(n_colors)]
        if not any(closed_flags):
            closed_flags[random.randrange(n_colors)] = True

        groups = [_fast_group_fragments(flag) for flag in closed_flags]
        max_rect_h = 0
        max_rect_w = 0
        closed_count = 0
        for fragments, flag in zip(groups, closed_flags):
            if not flag:
                continue
            boundary = _ordered_closed_boundary(fragments)
            assert boundary is not None
            mask = _fill_boundary(boundary)
            max_rect_h += mask.shape[0]
            max_rect_w = max(max_rect_w, mask.shape[1])
            closed_count += 1
        output_need_h = max_rect_h + max(0, closed_count - 1)

        band_h = 15
        height = max(n_colors * band_h, output_need_h + random.randint(0, 4), 18)
        height = min(height, GRID_MAX)
        left_w = random.randint(14, 17)
        canvas_w = random.randint(max(7, max_rect_w), 12)
        if left_w + 1 + canvas_w > GRID_MAX or output_need_h > height:
            continue

        grid = np.full((height, left_w + 1 + canvas_w), BG, dtype=int)
        grid[:, left_w] = DIVIDER
        occupied = np.zeros((height, left_w), dtype=bool)

        ok = True
        for group_idx, (color, fragments) in enumerate(zip(colors, groups)):
            base = group_idx * band_h
            for frag_idx, fragment in enumerate(fragments):
                row = base + frag_idx * 3
                if not _place_at_ordered_row(
                    grid,
                    occupied,
                    color,
                    str(fragment["axis"]),
                    int(fragment["length"]),
                    row,
                    left_w,
                ):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        out = main(grid)
        if out.ndim == 2 and max(grid.shape) <= GRID_MAX and max(out.shape) <= GRID_MAX:
            return grid
    raise RuntimeError("failed to generate a valid fragmented-polyline puzzle")
