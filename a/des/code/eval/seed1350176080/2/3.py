from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRY_LIMIT = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ORTHOGONAL_STEPS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _is_square_frame(grid: np.ndarray, points: list[tuple[int, int]], bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 4:
        return False

    expected = 4 * side_h - 4
    if len(points) != expected:
        return False

    if not np.all(grid[r0, c0 : c1 + 1] == BLACK):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == BLACK):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == BLACK):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == BLACK):
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg))


def _find_containers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for comp in _components(grid == BLACK):
        if not _is_square_frame(grid, comp, bg):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )
    containers.sort(key=lambda item: item["bbox"])
    return containers


def _find_legend_entries(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    mask = (grid != bg) & (grid != BLACK)
    entries: list[dict[str, object]] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h not in (2, 3) or w != 2 * h:
            continue
        if len(comp) != h * w:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        split = h
        left = patch[:, :split]
        right = patch[:, split:]
        left_colors = {int(v) for v in np.unique(left)}
        right_colors = {int(v) for v in np.unique(right)}
        if len(left_colors) != 1 or len(right_colors) != 2:
            continue
        key_color = next(iter(left_colors))
        if key_color in {bg, BLACK} or right_colors & {bg, BLACK}:
            continue
        entries.append(
            {
                "bbox": (r0, c0, r1, c1),
                "key_color": key_color,
                "pattern": right.copy(),
            }
        )

    entries.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert entries, "expected at least one legend entry"
    return entries


def _build_legend_mask(shape: tuple[int, int], entries: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for entry in entries:
        r0, c0, r1, c1 = entry["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _border_owner_map(shape: tuple[int, int], containers: list[dict[str, object]]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, container in enumerate(containers):
        r0, c0, r1, c1 = container["bbox"]
        owner[r0, c0 : c1 + 1] = idx
        owner[r1, c0 : c1 + 1] = idx
        owner[r0 : r1 + 1, c0] = idx
        owner[r0 : r1 + 1, c1] = idx
    return owner


def _find_lines(
    grid: np.ndarray,
    bg: int,
    legend_mask: np.ndarray,
    border_owner: np.ndarray,
) -> list[dict[str, object]]:
    mask = (grid != bg) & (grid != BLACK) & (~legend_mask)
    lines: list[dict[str, object]] = []
    h, w = grid.shape
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        if not (r0 == r1 or c0 == c1):
            continue
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        touched: set[int] = set()
        for r, c in comp:
            for dr, dc in ORTHOGONAL_STEPS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and border_owner[nr, nc] >= 0:
                    touched.add(int(border_owner[nr, nc]))
        if len(touched) != 2:
            continue
        lines.append({"color": next(iter(colors)), "containers": tuple(sorted(touched))})
    assert lines, "expected at least one colored connecting line"
    return lines


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ph, pw = pattern.shape
    rows = np.arange(h)[:, None] % ph
    cols = np.arange(w)[None, :] % pw
    return pattern[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each connecting line's color as a key into the corner legend, then tile the matched
    pattern into the two endpoint containers. If several lines reach the same container, the
    pattern from the top-most legend entry wins.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    legend_entries = _find_legend_entries(g, bg)
    legend_mask = _build_legend_mask(g.shape, legend_entries)
    containers = _find_containers(g, bg)
    assert containers, "expected at least one square container"
    border_owner = _border_owner_map(g.shape, containers)
    lines = _find_lines(g, bg, legend_mask, border_owner)

    priorities = {int(entry["key_color"]): idx for idx, entry in enumerate(legend_entries)}
    patterns = {int(entry["key_color"]): np.asarray(entry["pattern"], dtype=int) for entry in legend_entries}

    chosen: dict[int, int] = {}
    for line in lines:
        color = int(line["color"])
        assert color in priorities, "every line color must appear in the legend"
        for container_idx in line["containers"]:
            prev = chosen.get(container_idx)
            if prev is None or priorities[color] < priorities[prev]:
                chosen[container_idx] = color

    out = g.copy()
    for container_idx, color in chosen.items():
        ir0, ic0, ir1, ic1 = containers[container_idx]["interior"]
        tile = _tile_pattern(patterns[color], (ir1 - ir0 + 1, ic1 - ic0 + 1))
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = tile
    return out


def _draw_container(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = BLACK
    grid[r1, c0 : c1 + 1] = BLACK
    grid[r0 : r1 + 1, c0] = BLACK
    grid[r0 : r1 + 1, c1] = BLACK


def _pattern_choices(size: int, a: int, b: int) -> list[np.ndarray]:
    if size == 2:
        return [
            np.array([[a, b], [b, a]], dtype=int),
            np.array([[a, b], [a, b]], dtype=int),
            np.array([[a, a], [b, b]], dtype=int),
            np.array([[a, a], [a, b]], dtype=int),
            np.array([[a, b], [b, b]], dtype=int),
        ]

    base_l = np.array([[a, a, a], [a, b, b], [a, b, b]], dtype=int)
    return [
        np.array([[a, b, a], [b, a, b], [a, b, a]], dtype=int),
        np.array([[a, b, a], [a, b, a], [a, b, a]], dtype=int),
        np.array([[a, a, a], [b, b, b], [a, a, a]], dtype=int),
        base_l,
        np.rot90(base_l, 1),
        np.rot90(base_l, 2),
        np.rot90(base_l, 3),
    ]


def _all_slots(nrows: int, ncols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(nrows) for c in range(ncols)]


def _slot_neighbors(slot: tuple[int, int], nrows: int, ncols: int) -> list[tuple[int, int]]:
    r, c = slot
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < nrows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < ncols:
        out.append((r, c + 1))
    return out


def _adjacent_slot_edges(nrows: int, ncols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(nrows):
        for c in range(ncols):
            if r + 1 < nrows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < ncols:
                edges.append(((r, c), (r, c + 1)))
    return edges


def _choose_graph(
    nrows: int,
    ncols: int,
    priority_case: bool,
    isolated_case: bool,
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]], tuple[int, int] | None]:
    slots = _all_slots(nrows, ncols)
    if priority_case:
        hubs = [slot for slot in slots if len(_slot_neighbors(slot, nrows, ncols)) >= 2]
        if not hubs:
            raise ValueError("no valid hub slots")
        hub = random.choice(hubs)
        neighbors = _slot_neighbors(hub, nrows, ncols)
        degree = random.choice([k for k in (2, 3) if k <= len(neighbors)])
        chosen_neighbors = random.sample(neighbors, degree)
        edges = [(hub, neighbor) for neighbor in chosen_neighbors]
        occupied = {hub, *chosen_neighbors}
        if isolated_case:
            extras = [slot for slot in slots if slot not in occupied]
            if not extras:
                raise ValueError("no room for isolated container")
            occupied.add(random.choice(extras))
        return sorted(occupied), edges, hub

    all_edges = _adjacent_slot_edges(nrows, ncols)
    random.shuffle(all_edges)
    chosen_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()
    for a, b in all_edges:
        if a in used or b in used:
            continue
        chosen_edges.append((a, b))
        used.add(a)
        used.add(b)
        if len(chosen_edges) == 2:
            break
    if len(chosen_edges) < 2:
        raise ValueError("failed to build a disjoint matching")

    occupied = set(used)
    if isolated_case:
        extras = [slot for slot in slots if slot not in occupied]
        if not extras:
            raise ValueError("no room for isolated container")
        occupied.add(random.choice(extras))
    return sorted(occupied), chosen_edges, None


def _line_cells(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    if ar0 == br0:
        row = ar0 + (ar1 - ar0) // 2
        if ac0 < bc0:
            return [(row, c) for c in range(ac1 + 1, bc0)]
        return [(row, c) for c in range(bc1 + 1, ac0)]

    col = ac0 + (ac1 - ac0) // 2
    if ar0 < br0:
        return [(r, col) for r in range(ar1 + 1, br0)]
    return [(r, col) for r in range(br1 + 1, ar0)]


def _render_legend(
    grid: np.ndarray,
    corner: str,
    entries: list[dict[str, object]],
) -> None:
    heights = [int(entry["pattern"].shape[0]) for entry in entries]
    widths = [2 * h for h in heights]
    total_h = sum(heights) + max(0, len(entries) - 1)
    h, w = grid.shape

    if corner.startswith("top"):
        start_r = 0
    else:
        start_r = h - total_h

    current_r = start_r
    for entry, entry_h, entry_w in zip(entries, heights, widths):
        if corner.endswith("left"):
            start_c = 0
        else:
            start_c = w - entry_w
        key_color = int(entry["key_color"])
        pattern = np.asarray(entry["pattern"], dtype=int)
        grid[current_r : current_r + entry_h, start_c : start_c + entry_h] = key_color
        grid[current_r : current_r + entry_h, start_c + entry_h : start_c + entry_w] = pattern
        current_r += entry_h + 1


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    bg = random.choice([color for color in range(10) if color != BLACK])
    priority_case = random.choice([False, True])
    isolated_case = random.choice([False, True])
    corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    side = random.choice([5, 7])
    nrows = random.choice([2, 3])
    ncols = random.choice([2, 3])
    row_gap = random.randint(2, 5)
    col_gap = random.randint(2, 5)

    occupied_slots, edge_slots, hub_slot = _choose_graph(nrows, ncols, priority_case, isolated_case)
    line_colors = random.sample([color for color in range(10) if color not in {bg, BLACK}], len(edge_slots))

    edge_data: list[dict[str, object]] = []
    color_pool = [color for color in range(10) if color not in {bg, BLACK}]
    for slot_pair, line_color in zip(edge_slots, line_colors):
        pattern_size = random.choice([2, 3])
        pattern_colors = random.sample([color for color in color_pool if color != line_color], 2)
        pattern = random.choice(_pattern_choices(pattern_size, pattern_colors[0], pattern_colors[1]))
        edge_data.append({"slots": slot_pair, "line_color": line_color, "pattern": pattern})

    if priority_case and hub_slot is not None:
        hub_patterns = []
        for edge in edge_data:
            if hub_slot in edge["slots"]:
                tiled = _tile_pattern(np.asarray(edge["pattern"], dtype=int), (side - 2, side - 2))
                hub_patterns.append(tiled.tobytes())
        if len(set(hub_patterns)) < len(hub_patterns):
            raise ValueError("priority case must offer distinct competing fills")

    legend_order = edge_data[:]
    random.shuffle(legend_order)
    legend_entries = [
        {"key_color": int(edge["line_color"]), "pattern": np.asarray(edge["pattern"], dtype=int).copy()}
        for edge in legend_order
    ]

    legend_h = sum(entry["pattern"].shape[0] for entry in legend_entries) + max(0, len(legend_entries) - 1)
    legend_w = max(2 * entry["pattern"].shape[0] for entry in legend_entries)

    board_h = nrows * side + (nrows - 1) * row_gap
    board_w = ncols * side + (ncols - 1) * col_gap

    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)

    reserve_axis = random.choice(["vertical", "horizontal"])
    if corner.startswith("top") and reserve_axis == "vertical":
        top_pad += legend_h + 1
    if corner.startswith("bottom") and reserve_axis == "vertical":
        bottom_pad += legend_h + 1
    if corner.endswith("left") and reserve_axis == "horizontal":
        left_pad += legend_w + 1
    if corner.endswith("right") and reserve_axis == "horizontal":
        right_pad += legend_w + 1

    grid_h = top_pad + board_h + bottom_pad
    grid_w = left_pad + board_w + right_pad
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    slot_positions: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for sr, sc in occupied_slots:
        r0 = top_pad + sr * (side + row_gap)
        c0 = left_pad + sc * (side + col_gap)
        bbox = (r0, c0, r0 + side - 1, c0 + side - 1)
        slot_positions[(sr, sc)] = bbox
        _draw_container(grid, bbox)

    for edge in edge_data:
        bbox_a = slot_positions[edge["slots"][0]]
        bbox_b = slot_positions[edge["slots"][1]]
        cells = _line_cells(bbox_a, bbox_b)
        if not cells:
            raise ValueError("edge gap must leave at least one line cell")
        for r, c in cells:
            grid[r, c] = int(edge["line_color"])

    _render_legend(grid, corner, legend_entries)

    if _most_common_color(grid) != bg:
        raise ValueError("background must stay dominant")

    legend_rank = {int(entry["key_color"]): idx for idx, entry in enumerate(legend_entries)}
    pattern_by_color = {int(entry["key_color"]): np.asarray(entry["pattern"], dtype=int) for entry in legend_entries}
    output = grid.copy()
    incident: dict[tuple[int, int], list[int]] = {slot: [] for slot in occupied_slots}
    for edge in edge_data:
        color = int(edge["line_color"])
        a, b = edge["slots"]
        incident[a].append(color)
        incident[b].append(color)

    for slot, colors in incident.items():
        if not colors:
            continue
        chosen_color = min(colors, key=lambda color: legend_rank[color])
        ir0, ic0, ir1, ic1 = (
            slot_positions[slot][0] + 1,
            slot_positions[slot][1] + 1,
            slot_positions[slot][2] - 1,
            slot_positions[slot][3] - 1,
        )
        output[ir0 : ir1 + 1, ic0 : ic1 + 1] = _tile_pattern(
            pattern_by_color[chosen_color],
            (ir1 - ir0 + 1, ic1 - ic0 + 1),
        )

    return grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            inp, expected = _build_case()
        except (AssertionError, ValueError):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRY_LIMIT} attempts")
