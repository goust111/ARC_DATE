from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, in_bounds, neighbors4


BG = Color.BLACK
FRAME = Color.GREY
TIE = Color.RED
SEED_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.MAROON]
GEN_ATTEMPTS = 400
GRAPH_ATTEMPTS = 120
SEED_ATTEMPTS = 100
GAP_ATTEMPTS = 50


def _norm_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _grid_size(slot_rows: int, slot_cols: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return slot_rows * (cell_h + 1) + 1, slot_cols * (cell_w + 1) + 1


def _line_positions_from_layout(slot_count: int, cell_size: int) -> list[int]:
    return [i * (cell_size + 1) for i in range(slot_count + 1)]


def _slot_bounds(
    row_lines: list[int], col_lines: list[int], slot_r: int, slot_c: int
) -> tuple[int, int, int, int]:
    return (
        row_lines[slot_r] + 1,
        row_lines[slot_r + 1] - 1,
        col_lines[slot_c] + 1,
        col_lines[slot_c + 1] - 1,
    )


def _slot_index(slot_r: int, slot_c: int, slot_cols: int) -> int:
    return slot_r * slot_cols + slot_c


def _slot_rc(slot_idx: int, slot_cols: int) -> tuple[int, int]:
    return divmod(slot_idx, slot_cols)


def _label_from_colors(colors: list[int]) -> int:
    uniq = sorted(set(int(c) for c in colors))
    return uniq[0] if len(uniq) == 1 else TIE


def _boundary_frame_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    grey = np.asarray(grid, dtype=int) == FRAME
    frame_mask = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grey[r, c] and not frame_mask[r, c]:
                frame_mask[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grey[r, c] and not frame_mask[r, c]:
                frame_mask[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds((h, w), nr, nc) and grey[nr, nc] and not frame_mask[nr, nc]:
                frame_mask[nr, nc] = True
                queue.append((nr, nc))
    return frame_mask


def _erase_decorations(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cleaned = np.asarray(grid, dtype=int).copy()
    frame_mask = _boundary_frame_mask(cleaned)
    cleaned[(cleaned == FRAME) & (~frame_mask)] = BG
    return cleaned, frame_mask


def _recover_grid_lines(frame_mask: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = frame_mask.shape
    row_lines = [r for r in range(h) if int(frame_mask[r, :].sum()) > w // 2]
    col_lines = [c for c in range(w) if int(frame_mask[:, c].sum()) > h // 2]
    assert len(row_lines) >= 2 and len(col_lines) >= 2
    assert all(b - a >= 2 for a, b in zip(row_lines, row_lines[1:]))
    assert all(b - a >= 2 for a, b in zip(col_lines, col_lines[1:]))
    return row_lines, col_lines


def _coord_to_slot(r: int, c: int, row_lines: list[int], col_lines: list[int]) -> int:
    slot_r = slot_c = None
    for i in range(len(row_lines) - 1):
        if row_lines[i] < r < row_lines[i + 1]:
            slot_r = i
            break
    for j in range(len(col_lines) - 1):
        if col_lines[j] < c < col_lines[j + 1]:
            slot_c = j
            break
    assert slot_r is not None and slot_c is not None, "colored pixel must lie inside a slot"
    return _slot_index(slot_r, slot_c, len(col_lines) - 1)


def _build_slot_graph(
    cleaned: np.ndarray, row_lines: list[int], col_lines: list[int]
) -> list[list[int]]:
    slot_rows = len(row_lines) - 1
    slot_cols = len(col_lines) - 1
    graph = [set() for _ in range(slot_rows * slot_cols)]

    for slot_r in range(slot_rows):
        for slot_c in range(slot_cols - 1):
            wall_c = col_lines[slot_c + 1]
            r0 = row_lines[slot_r] + 1
            r1 = row_lines[slot_r + 1] - 1
            if np.any(cleaned[r0 : r1 + 1, wall_c] == BG):
                u = _slot_index(slot_r, slot_c, slot_cols)
                v = _slot_index(slot_r, slot_c + 1, slot_cols)
                graph[u].add(v)
                graph[v].add(u)

    for slot_r in range(slot_rows - 1):
        for slot_c in range(slot_cols):
            wall_r = row_lines[slot_r + 1]
            c0 = col_lines[slot_c] + 1
            c1 = col_lines[slot_c + 1] - 1
            if np.any(cleaned[wall_r, c0 : c1 + 1] == BG):
                u = _slot_index(slot_r, slot_c, slot_cols)
                v = _slot_index(slot_r + 1, slot_c, slot_cols)
                graph[u].add(v)
                graph[v].add(u)

    return [sorted(neigh) for neigh in graph]


def _extract_seeds(
    cleaned: np.ndarray, row_lines: list[int], col_lines: list[int]
) -> list[dict[str, int | tuple[int, int]]]:
    seeds = []
    for r, c in np.argwhere((cleaned != BG) & (cleaned != FRAME)):
        seeds.append(
            {
                "coord": (int(r), int(c)),
                "slot": _coord_to_slot(int(r), int(c), row_lines, col_lines),
                "color": int(cleaned[r, c]),
            }
        )
    seeds.sort(key=lambda item: (int(item["slot"]), int(item["color"])))
    assert 2 <= len(seeds) <= 4
    assert len({int(seed["slot"]) for seed in seeds}) == len(seeds)
    return seeds


def _bfs_distances(graph: list[list[int]], start: int) -> dict[int, int]:
    dist = {int(start): 0}
    queue: deque[int] = deque([int(start)])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def _slot_labels(
    graph: list[list[int]], seeds: list[dict[str, int | tuple[int, int]]]
) -> tuple[list[int | None], list[dict[int, int]]]:
    dist_maps = [_bfs_distances(graph, int(seed["slot"])) for seed in seeds]
    labels: list[int | None] = []
    for node in range(len(graph)):
        best = None
        best_colors: list[int] = []
        for seed, dist_map in zip(seeds, dist_maps):
            if node not in dist_map:
                continue
            dist = dist_map[node]
            color = int(seed["color"])
            if best is None or dist < best:
                best = dist
                best_colors = [color]
            elif dist == best:
                best_colors.append(color)
        labels.append(None if best is None else _label_from_colors(best_colors))
    return labels, dist_maps


def _render_solver_output(
    cleaned: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    labels: list[int | None],
) -> np.ndarray:
    out = cleaned.copy()
    slot_cols = len(col_lines) - 1
    for slot_idx, color in enumerate(labels):
        if color is None:
            continue
        slot_r, slot_c = _slot_rc(slot_idx, slot_cols)
        r0, r1, c0, c1 = _slot_bounds(row_lines, col_lines, slot_r, slot_c)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = int(color)
    return out


def _solve_competition(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[dict]]:
    cleaned, frame_mask = _erase_decorations(grid)
    row_lines, col_lines = _recover_grid_lines(frame_mask)
    graph = _build_slot_graph(cleaned, row_lines, col_lines)
    seeds = _extract_seeds(cleaned, row_lines, col_lines)
    labels, _ = _slot_labels(graph, seeds)
    out = _render_solver_output(cleaned, row_lines, col_lines, labels)
    return out, cleaned, row_lines, col_lines, seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill lattice slots through wall gaps, letting colored seeds compete for each slot.

    Input structure:
    - Black background with a one-pixel-thick gray lattice whose lines reach the grid border.
    - The lattice partitions the interior into regular rectangular slots; some wall segments contain a single black gap.
    - Two to four non-gray seed pixels of distinct colors sit inside different slots, and extra gray fragments may float inside slots.

    Rule:
    - Treat gray components that touch the border as the real lattice frame, and erase all other gray fragments to black.
    - Recover the slot grid from the frame lines, then connect adjacent slots exactly when their shared wall segment contains a gap.
    - Run breadth-first search on that slot graph from every seed slot, recording each slot's shortest graph distance to each seed.
    - Fill each slot interior: unique nearest seed color wins, equal shortest distances from multiple seed colors produce red, while frame lines stay gray and gap cells stay black.

    Key insight:
    - The visible lattice is really an implicit graph: wall gaps define edges between slot nodes, so ownership depends on shortest paths between slots rather than on raw pixel distance through the canvas.
    """
    out, _cleaned, _row_lines, _col_lines, _seeds = _solve_competition(np.asarray(grid, dtype=int))
    return out


def _all_lattice_edges(slot_rows: int, slot_cols: int) -> list[tuple[int, int]]:
    edges = []
    for slot_r in range(slot_rows):
        for slot_c in range(slot_cols):
            u = _slot_index(slot_r, slot_c, slot_cols)
            if slot_c + 1 < slot_cols:
                edges.append((u, _slot_index(slot_r, slot_c + 1, slot_cols)))
            if slot_r + 1 < slot_rows:
                edges.append((u, _slot_index(slot_r + 1, slot_c, slot_cols)))
    return edges


def _adj_from_edges(node_count: int, edges: set[tuple[int, int]]) -> list[list[int]]:
    graph = [set() for _ in range(node_count)]
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    return [sorted(neigh) for neigh in graph]


def _random_connected_edges(
    slot_rows: int, slot_cols: int, rng: random.Random
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]]:
    node_count = slot_rows * slot_cols
    all_edges = [_norm_edge(u, v) for u, v in _all_lattice_edges(slot_rows, slot_cols)]
    shuffled = all_edges[:]
    rng.shuffle(shuffled)

    parent = list(range(node_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    tree_edges: set[tuple[int, int]] = set()
    for edge in shuffled:
        u, v = edge
        if find(u) != find(v):
            union(u, v)
            tree_edges.add(edge)

    open_edges = set(tree_edges)
    extra_prob = rng.uniform(0.18, 0.5)
    for edge in shuffled:
        if edge not in open_edges and rng.random() < extra_prob:
            open_edges.add(edge)

    if len(open_edges) == len(all_edges):
        removable = [edge for edge in open_edges if edge not in tree_edges]
        if removable:
            open_edges.remove(rng.choice(removable))

    return open_edges, tree_edges, all_edges


def _structured_open_edges(slot_rows: int, slot_cols: int, rng: random.Random) -> set[tuple[int, int]]:
    mid_c = slot_cols // 2
    open_edges: set[tuple[int, int]] = set()

    for slot_c in range(slot_cols - 1):
        open_edges.add(_norm_edge(_slot_index(0, slot_c, slot_cols), _slot_index(0, slot_c + 1, slot_cols)))
        open_edges.add(
            _norm_edge(
                _slot_index(slot_rows - 1, slot_c, slot_cols),
                _slot_index(slot_rows - 1, slot_c + 1, slot_cols),
            )
        )

    for slot_r in range(slot_rows - 1):
        open_edges.add(_norm_edge(_slot_index(slot_r, mid_c, slot_cols), _slot_index(slot_r + 1, mid_c, slot_cols)))

    for slot_c in range(slot_cols):
        if slot_c == mid_c:
            continue
        split = rng.randint(0, slot_rows - 2)
        for slot_r in range(split):
            open_edges.add(
                _norm_edge(_slot_index(slot_r, slot_c, slot_cols), _slot_index(slot_r + 1, slot_c, slot_cols))
            )
        for slot_r in range(split + 1, slot_rows - 1):
            open_edges.add(
                _norm_edge(_slot_index(slot_r, slot_c, slot_cols), _slot_index(slot_r + 1, slot_c, slot_cols))
            )

    return open_edges


def _slot_center_choices(start: int, size: int) -> list[int]:
    if size % 2 == 1:
        return [start + size // 2]
    return [start + size // 2 - 1, start + size // 2]


def _make_seed_pixels(meta: dict, rng: random.Random) -> dict[int, tuple[int, int]]:
    row_lines = meta["row_lines"]
    col_lines = meta["col_lines"]
    cell_h = meta["cell_h"]
    cell_w = meta["cell_w"]
    seed_pixels: dict[int, tuple[int, int]] = {}
    for slot in meta["seed_slots"]:
        slot_r, slot_c = _slot_rc(slot, meta["slot_cols"])
        r0, _r1, c0, _c1 = _slot_bounds(row_lines, col_lines, slot_r, slot_c)
        seed_pixels[slot] = (
            rng.choice(_slot_center_choices(r0, cell_h)),
            rng.choice(_slot_center_choices(c0, cell_w)),
        )
    return seed_pixels


def _slot_safe_cells(
    row_lines: list[int], col_lines: list[int], slot_r: int, slot_c: int
) -> tuple[list[int], list[int]]:
    r0, r1, c0, c1 = _slot_bounds(row_lines, col_lines, slot_r, slot_c)
    safe_rows = list(range(r0 + 1, r1))
    safe_cols = list(range(c0 + 1, c1))
    return safe_rows, safe_cols


def _make_decorations(meta: dict, rng: random.Random) -> set[tuple[int, int]] | None:
    slot_rows = meta["slot_rows"]
    slot_cols = meta["slot_cols"]
    row_lines = meta["row_lines"]
    col_lines = meta["col_lines"]
    seed_slots = set(meta["seed_slots"])
    decorations: set[tuple[int, int]] = set()

    orientation = rng.choice(["h", "v"])
    family_slots: list[int] = []

    if orientation == "h":
        candidate_rows = [
            slot_r
            for slot_r in range(slot_rows)
            if sum(_slot_index(slot_r, slot_c, slot_cols) not in seed_slots for slot_c in range(slot_cols)) >= 2
        ]
        if candidate_rows:
            slot_r = rng.choice(candidate_rows)
            cols = [slot_c for slot_c in range(slot_cols) if _slot_index(slot_r, slot_c, slot_cols) not in seed_slots]
            rng.shuffle(cols)
            family_count = rng.randint(2, min(len(cols), 3))
            for slot_c in cols[:family_count]:
                safe_rows, safe_cols = _slot_safe_cells(row_lines, col_lines, slot_r, slot_c)
                if not safe_rows or len(safe_cols) < 2:
                    return None
                rr = rng.choice(safe_rows)
                for cc in safe_cols:
                    decorations.add((rr, cc))
                family_slots.append(_slot_index(slot_r, slot_c, slot_cols))

    if not family_slots:
        candidate_cols = [
            slot_c
            for slot_c in range(slot_cols)
            if sum(_slot_index(slot_r, slot_c, slot_cols) not in seed_slots for slot_r in range(slot_rows)) >= 2
        ]
        if not candidate_cols:
            return None
        slot_c = rng.choice(candidate_cols)
        rows = [slot_r for slot_r in range(slot_rows) if _slot_index(slot_r, slot_c, slot_cols) not in seed_slots]
        rng.shuffle(rows)
        family_count = rng.randint(2, min(len(rows), 3))
        for slot_r in rows[:family_count]:
            safe_rows, safe_cols = _slot_safe_cells(row_lines, col_lines, slot_r, slot_c)
            if len(safe_rows) < 2 or not safe_cols:
                return None
            cc = rng.choice(safe_cols)
            for rr in safe_rows:
                decorations.add((rr, cc))
            family_slots.append(_slot_index(slot_r, slot_c, slot_cols))

    extra_slots = [
        slot
        for slot in range(slot_rows * slot_cols)
        if slot not in seed_slots and slot not in family_slots
    ]
    rng.shuffle(extra_slots)
    for slot in extra_slots[: rng.randint(0, min(2, len(extra_slots))) if extra_slots else 0]:
        slot_r, slot_c = _slot_rc(slot, slot_cols)
        safe_rows, safe_cols = _slot_safe_cells(row_lines, col_lines, slot_r, slot_c)
        if len(safe_rows) >= 2 and len(safe_cols) >= 2:
            rr = rng.choice(safe_rows)
            cc = rng.choice(safe_cols)
            coords = [(rr, cc), (rr, safe_cols[0]), (safe_rows[0], cc)]
            decorations.update(coords[: rng.randint(1, len(coords))])

    return decorations if decorations else None


def _make_gap_positions(meta: dict, rng: random.Random) -> dict[tuple[int, int], tuple[int, int]]:
    row_lines = meta["row_lines"]
    col_lines = meta["col_lines"]
    slot_cols = meta["slot_cols"]
    gaps: dict[tuple[int, int], tuple[int, int]] = {}
    for edge in meta["open_edges"]:
        u, v = edge
        ru, cu = _slot_rc(u, slot_cols)
        rv, cv = _slot_rc(v, slot_cols)
        if ru == rv:
            wall_c = col_lines[max(cu, cv)]
            candidates = list(range(row_lines[ru] + 1, row_lines[ru + 1]))
            gaps[edge] = (rng.choice(candidates), wall_c)
        else:
            wall_r = row_lines[max(ru, rv)]
            candidates = list(range(col_lines[cu] + 1, col_lines[cu + 1]))
            gaps[edge] = (wall_r, rng.choice(candidates))
    return gaps


def _meta_seeds(meta: dict) -> list[dict[str, int | tuple[int, int]]]:
    return [
        {"slot": slot, "color": color, "coord": meta["seed_pixels"][slot]}
        for slot, color in zip(meta["seed_slots"], meta["seed_colors"])
    ]


def _graph_labels_from_meta(meta: dict) -> tuple[list[int | None], list[dict[int, int]]]:
    graph = _adj_from_edges(meta["slot_rows"] * meta["slot_cols"], meta["open_edges"])
    return _slot_labels(graph, _meta_seeds(meta))


def _open_grid_labels(meta: dict) -> list[int]:
    labels = []
    seeds = list(zip(meta["seed_slots"], meta["seed_colors"]))
    for node in range(meta["slot_rows"] * meta["slot_cols"]):
        slot_r, slot_c = _slot_rc(node, meta["slot_cols"])
        best = None
        best_colors: list[int] = []
        for seed_slot, color in seeds:
            seed_r, seed_c = _slot_rc(seed_slot, meta["slot_cols"])
            dist = abs(slot_r - seed_r) + abs(slot_c - seed_c)
            if best is None or dist < best:
                best = dist
                best_colors = [color]
            elif dist == best:
                best_colors.append(color)
        labels.append(_label_from_colors(best_colors))
    return labels


def _render_base(meta: dict) -> np.ndarray:
    h, w = meta["shape"]
    grid = np.full((h, w), BG, dtype=int)
    for r in meta["row_lines"]:
        grid[r, :] = FRAME
    for c in meta["col_lines"]:
        grid[:, c] = FRAME
    for r, c in meta["gaps"].values():
        grid[r, c] = BG
    return grid


def _render_expected_output(meta: dict) -> np.ndarray:
    labels, _ = _graph_labels_from_meta(meta)
    out = _render_base(meta)
    for slot in range(meta["slot_rows"] * meta["slot_cols"]):
        slot_r, slot_c = _slot_rc(slot, meta["slot_cols"])
        r0, r1, c0, c1 = _slot_bounds(meta["row_lines"], meta["col_lines"], slot_r, slot_c)
        out[r0 : r1 + 1, c0 : c1 + 1] = labels[slot]
    for slot, color in zip(meta["seed_slots"], meta["seed_colors"]):
        rr, cc = meta["seed_pixels"][slot]
        out[rr, cc] = color
    return out


def _render_input(meta: dict) -> np.ndarray:
    grid = _render_base(meta)
    for r, c in meta["decorations"]:
        grid[r, c] = FRAME
    for slot, color in zip(meta["seed_slots"], meta["seed_colors"]):
        rr, cc = meta["seed_pixels"][slot]
        grid[rr, cc] = color
    return grid


def _slot_interior_mask(meta: dict) -> np.ndarray:
    mask = np.zeros(meta["shape"], dtype=bool)
    for slot in range(meta["slot_rows"] * meta["slot_cols"]):
        slot_r, slot_c = _slot_rc(slot, meta["slot_cols"])
        r0, r1, c0, c1 = _slot_bounds(meta["row_lines"], meta["col_lines"], slot_r, slot_c)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _pixel_competition_output(grid: np.ndarray) -> np.ndarray:
    cleaned, frame_mask = _erase_decorations(grid)
    row_lines, col_lines = _recover_grid_lines(frame_mask)
    seeds = _extract_seeds(cleaned, row_lines, col_lines)
    passable = cleaned != FRAME
    dist_maps = []
    h, w = cleaned.shape

    for seed in seeds:
        start_r, start_c = seed["coord"]
        dist = np.full((h, w), 10**9, dtype=int)
        dist[start_r, start_c] = 0
        queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            for nr, nc in neighbors4(r, c):
                if in_bounds(cleaned.shape, nr, nc) and passable[nr, nc] and dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
        dist_maps.append(dist)

    out = cleaned.copy()
    slot_cols = len(col_lines) - 1
    for slot in range((len(row_lines) - 1) * slot_cols):
        slot_r, slot_c = _slot_rc(slot, slot_cols)
        r0, r1, c0, c1 = _slot_bounds(row_lines, col_lines, slot_r, slot_c)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r, c] != BG:
                    continue
                best = min(int(dist[r, c]) for dist in dist_maps)
                winner_colors = [
                    int(seed["color"])
                    for seed, dist in zip(seeds, dist_maps)
                    if int(dist[r, c]) == best
                ]
                out[r, c] = _label_from_colors(winner_colors)
    return out


def _layout_is_semantic(meta: dict) -> bool:
    labels, dist_maps = _graph_labels_from_meta(meta)
    seed_slots = set(meta["seed_slots"])
    nonseed_nodes = [node for node in range(len(labels)) if node not in seed_slots]
    if not nonseed_nodes:
        return False

    if any(label is None for label in labels):
        return False
    if not any(labels[node] == TIE for node in nonseed_nodes):
        return False
    if not any(labels[node] not in (None, TIE) for node in nonseed_nodes):
        return False
    if len(set(labels)) < 3:
        return False

    open_labels = _open_grid_labels(meta)
    if not any(labels[node] != open_labels[node] for node in nonseed_nodes):
        return False

    graph = _adj_from_edges(meta["slot_rows"] * meta["slot_cols"], meta["open_edges"])
    if any(len(graph[node]) == 0 for node in range(len(graph))):
        return False
    if not meta["decorations"]:
        return False

    expected = _render_expected_output(meta)
    inp = _render_input(meta)
    if not np.array_equal(main(inp), expected):
        return False

    pixel_out = _pixel_competition_output(inp)
    interior = _slot_interior_mask(meta)
    if not np.any(
        (pixel_out != expected)
        & interior
        & (inp != FRAME)
    ):
        return False

    return len(dist_maps) == len(meta["seed_slots"])


def _try_make_meta(rng: random.Random) -> dict | None:
    for _ in range(GRAPH_ATTEMPTS):
        slot_rows = rng.randint(3, 5)
        slot_cols = rng.choice([3, 5])
        cell_h_choices = [size for size in (4, 5) if _grid_size(slot_rows, slot_cols, size, 4)[0] <= 30]
        cell_w_choices = [size for size in (4, 5) if _grid_size(slot_rows, slot_cols, 4, size)[1] <= 30]
        if not cell_h_choices or not cell_w_choices:
            continue
        cell_h = rng.choice(cell_h_choices)
        cell_w = rng.choice(cell_w_choices)
        h, w = _grid_size(slot_rows, slot_cols, cell_h, cell_w)
        if h > 30 or w > 30:
            continue

        open_edges = _structured_open_edges(slot_rows, slot_cols, rng)

        meta = {
            "slot_rows": slot_rows,
            "slot_cols": slot_cols,
            "cell_h": cell_h,
            "cell_w": cell_w,
            "shape": (h, w),
            "row_lines": _line_positions_from_layout(slot_rows, cell_h),
            "col_lines": _line_positions_from_layout(slot_cols, cell_w),
            "open_edges": open_edges,
        }
        slot_count = slot_rows * slot_cols
        base_seed_slots = [0, slot_cols - 1]
        extra_candidates = [
            slot
            for slot in range(slot_count)
            if slot not in base_seed_slots
            and _slot_rc(slot, slot_cols)[0] not in (0, slot_rows - 1)
        ]
        seed_count_choices = [2]
        if extra_candidates:
            seed_count_choices.append(3)
        if len(extra_candidates) >= 2:
            seed_count_choices.append(4)

        for _ in range(SEED_ATTEMPTS):
            seed_count = rng.choice(seed_count_choices)
            extra_needed = seed_count - 2
            seed_slots = base_seed_slots + (rng.sample(extra_candidates, extra_needed) if extra_needed else [])
            seed_slots = sorted(seed_slots)
            seed_colors = rng.sample(SEED_COLORS, seed_count)
            meta["seed_slots"] = seed_slots
            meta["seed_colors"] = seed_colors
            meta["seed_pixels"] = _make_seed_pixels(meta, rng)
            decorations = _make_decorations(meta, rng)
            if decorations is None:
                continue
            meta["decorations"] = decorations

            for _ in range(GAP_ATTEMPTS):
                meta["gaps"] = _make_gap_positions(meta, rng)
                if _layout_is_semantic(meta):
                    return meta.copy()
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        meta = _try_make_meta(random.Random())
        if meta is None:
            continue
        grid = _render_input(meta)
        if grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
