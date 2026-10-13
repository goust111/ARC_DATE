from __future__ import annotations

import random
from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_2_049_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
label_mask = _ARC_UTILS.label_mask
neighbors4 = _ARC_UTILS.neighbors4
in_bounds = _ARC_UTILS.in_bounds


BG = Color.BLACK
GREY = Color.GREY
CELL = 3
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MARKER_PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
MAIN_LENGTHS = [5, 7]
BRANCH_LENGTHS = [0, 1, 2]
MAX_GENERATION_ATTEMPTS = 500
MAX_PATH_ATTEMPTS = 200
MAX_BRANCH_ATTEMPTS = 80


Coord = tuple[int, int]


def _diag_adj(a: Coord, b: Coord) -> bool:
    return abs(int(a[0]) - int(b[0])) == 1 and abs(int(a[1]) - int(b[1])) == 1


def _orth_adj(a: Coord, b: Coord) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _graph_adjacency(nodes: set[Coord]) -> dict[Coord, set[Coord]]:
    adj = {node: set() for node in nodes}
    for r, c in nodes:
        for dr, dc in DIAG_DIRS:
            nb = (r + dr, c + dc)
            if nb in nodes:
                adj[(r, c)].add(nb)
    return adj


def _multi_source_distances(
    adjacency: dict[Coord | int, set[Coord | int]],
    sources: set[Coord | int],
) -> dict[Coord | int, int]:
    dist: dict[Coord | int, int] = {}
    q: deque[Coord | int] = deque()
    for src in sources:
        dist[src] = 0
        q.append(src)
    while q:
        node = q.popleft()
        for nb in adjacency[node]:
            if nb in dist:
                continue
            dist[nb] = dist[node] + 1
            q.append(nb)
    return dist


def _resolve_ownership(
    adjacency: dict[Coord | int, set[Coord | int]],
    sources_by_color: dict[int, set[Coord | int]],
) -> tuple[dict[Coord | int, int], dict[Coord | int, tuple[str, int | None, tuple[int, ...]]]]:
    distances_by_color = {
        int(color): _multi_source_distances(adjacency, set(nodes))
        for color, nodes in sources_by_color.items()
    }
    owners: dict[Coord | int, int] = {}
    meta: dict[Coord | int, tuple[str, int | None, tuple[int, ...]]] = {}

    for node in adjacency:
        best_dist: int | None = None
        best_colors: list[int] = []
        for color, dist in distances_by_color.items():
            if node not in dist:
                continue
            d = int(dist[node])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_colors = [int(color)]
            elif d == best_dist:
                best_colors.append(int(color))

        if best_dist is None:
            owners[node] = GREY
            meta[node] = ("unreachable", None, ())
        elif len(best_colors) == 1:
            owners[node] = int(best_colors[0])
            meta[node] = ("unique", int(best_dist), tuple(best_colors))
        else:
            owners[node] = GREY
            meta[node] = ("tie", int(best_dist), tuple(sorted(best_colors)))

    return owners, meta


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    grey_mask = g == GREY
    labels, n_labels = label_mask(grey_mask, connectivity=4)
    assert n_labels > 0, "expected at least one gray component"

    adjacency: dict[int, set[int]] = {label_id: set() for label_id in range(1, n_labels + 1)}
    for r, c in np.argwhere(grey_mask):
        comp = int(labels[r, c])
        for dr, dc in DIAG_DIRS:
            rr, cc = int(r) + dr, int(c) + dc
            if not in_bounds(g.shape, rr, cc):
                continue
            other = int(labels[rr, cc])
            if other and other != comp:
                adjacency[comp].add(other)

    marker_mask = (g != bg) & (g != GREY)
    marker_positions = [tuple(map(int, p)) for p in np.argwhere(marker_mask)]
    marker_colors = sorted({int(g[r, c]) for r, c in marker_positions})
    assert len(marker_colors) >= 2, "expected at least two marker colors"

    sources_by_color: dict[int, set[int]] = {}
    for r, c in marker_positions:
        color = int(g[r, c])
        touched = {
            int(labels[rr, cc])
            for rr, cc in neighbors4(r, c)
            if in_bounds(g.shape, rr, cc) and int(labels[rr, cc]) != 0
        }
        assert touched, "each marker pixel must edge-touch a gray component"
        sources_by_color.setdefault(color, set()).update(touched)

    return {
        "bg": bg,
        "labels": labels,
        "adjacency": adjacency,
        "marker_mask": marker_mask,
        "marker_positions": marker_positions,
        "sources_by_color": sources_by_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay marker colors through the gray-component diagonal-contact graph, then erase the markers.

    Input structure:
    - The background is the most common color.
    - Gray pixels form several 4-connected components, and distinct gray components never share an edge.
    - Colored marker pixels lie on the background and are edge-adjacent to gray components.

    Rule:
    - A gray component touched by a marker is a 0-hop source for that marker color.
    - Build a graph whose nodes are the gray 4-connected components and whose edges connect components that touch only diagonally.
    - Recolor each gray component with the marker color that reaches it in the fewest graph hops; equal best hop counts stay gray, and unreachable components stay gray.
    - Recolor every marker pixel to the background.

    Key insight:
    - The real connectivity is not the gray 4-neighborhood itself but the hidden graph of gray components linked by corner contacts, so ownership is a shortest-path race between components rather than a one-step local transfer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _analyze_grid(g)
    owners, _meta = _resolve_ownership(info["adjacency"], info["sources_by_color"])

    out = np.array(g, copy=True)
    out[info["marker_mask"]] = int(info["bg"])
    labels = info["labels"]
    for comp, color in owners.items():
        if int(color) != GREY:
            out[labels == int(comp)] = int(color)
    return out


def _extend_simple_diag_path(path: list[Coord], used: set[Coord], target_len: int) -> bool:
    if len(path) == target_len:
        return True

    current = path[-1]
    moves = list(DIAG_DIRS)
    random.shuffle(moves)
    for dr, dc in moves:
        nxt = (int(current[0]) + dr, int(current[1]) + dc)
        if nxt in used:
            continue
        if any(_diag_adj(nxt, other) for other in used if other != current):
            continue
        path.append(nxt)
        used.add(nxt)
        if _extend_simple_diag_path(path, used, target_len):
            return True
        used.remove(nxt)
        path.pop()
    return False


def _generate_main_path() -> list[Coord]:
    target_len = int(random.choice(MAIN_LENGTHS))
    for _ in range(MAX_PATH_ATTEMPTS):
        path = [(0, 0)]
        used = {(0, 0)}
        if _extend_simple_diag_path(path, used, target_len):
            return path
    raise RuntimeError("failed to generate main diagonal-contact path")


def _try_add_branch(occupied: set[Coord], main_path: list[Coord]) -> list[Coord]:
    desired_len = int(random.choices(BRANCH_LENGTHS, weights=[2, 4, 3])[0])
    if desired_len == 0:
        return []

    center = main_path[len(main_path) // 2]
    roots = [node for node in main_path[1:-1] if node != center]
    random.shuffle(roots)

    for _ in range(MAX_BRANCH_ATTEMPTS):
        for root in roots:
            branch: list[Coord] = []
            used = set(occupied)
            current = root
            ok = True
            for _step in range(desired_len):
                candidates = []
                moves = list(DIAG_DIRS)
                random.shuffle(moves)
                for dr, dc in moves:
                    nxt = (int(current[0]) + dr, int(current[1]) + dc)
                    if nxt in used:
                        continue
                    if any(_diag_adj(nxt, other) for other in used if other != current):
                        continue
                    candidates.append(nxt)
                if not candidates:
                    ok = False
                    break
                nxt = random.choice(candidates)
                branch.append(nxt)
                used.add(nxt)
                current = nxt
            if ok:
                return branch
    return []


def _span_ok(nodes: set[Coord]) -> bool:
    rows = [r for r, _c in nodes]
    cols = [c for _r, c in nodes]
    return (max(rows) - min(rows) + 1) <= 9 and (max(cols) - min(cols) + 1) <= 9


def _choose_isolated_node(occupied: set[Coord]) -> Coord | None:
    rows = [r for r, _c in occupied]
    cols = [c for _r, c in occupied]
    candidates: list[Coord] = []
    for r in range(min(rows) - 4, max(rows) + 5):
        for c in range(min(cols) - 4, max(cols) + 5):
            node = (r, c)
            if (r + c) % 2 != 0:
                continue
            if node in occupied:
                continue
            if any(_diag_adj(node, other) for other in occupied):
                continue
            if not _span_ok(set(occupied) | {node}):
                continue
            candidates.append(node)
    if not candidates:
        return None

    candidates.sort(key=lambda p: max(abs(p[0]), abs(p[1])), reverse=True)
    top_band = candidates[: max(6, len(candidates) // 3)]
    return random.choice(top_band)


_CORNER_PATHS = {
    (-1, -1): [[(0, 1), (0, 0)], [(1, 0), (0, 0)]],
    (-1, 1): [[(0, 1), (0, 2)], [(1, 2), (0, 2)]],
    (1, -1): [[(2, 1), (2, 0)], [(1, 0), (2, 0)]],
    (1, 1): [[(2, 1), (2, 2)], [(1, 2), (2, 2)]],
}
_LOCAL_CELLS = [(r, c) for r in range(CELL) for c in range(CELL)]
_LOCAL_CORNERS = {(0, 0), (0, 2), (2, 0), (2, 2)}


def _shape_for_node(required_dirs: set[Coord]) -> set[Coord]:
    cells: set[Coord] = {(1, 1)}
    for direction in required_dirs:
        for cell in random.choice(_CORNER_PATHS[direction]):
            cells.add(cell)

    min_area = max(3, len(cells))
    target_area = random.randint(min_area, 9)
    while len(cells) < target_area:
        frontier = [
            cell
            for cell in _LOCAL_CELLS
            if cell not in cells and any(_orth_adj(cell, other) for other in cells)
        ]
        if not frontier:
            break
        non_corner = [cell for cell in frontier if cell not in _LOCAL_CORNERS]
        choices = non_corner or frontier
        cells.add(random.choice(choices))
    return cells


def _pad_split(base: int, limit: int = 30) -> tuple[int, int]:
    rem = int(limit - base)
    assert rem >= 3
    extra = random.randint(2, min(6, rem))
    left = random.randint(1, extra - 1)
    right = extra - left
    return left, right


def _draw_instance(
    nodes: set[Coord],
    sources_by_color: dict[int, set[Coord]],
) -> tuple[np.ndarray, dict[Coord, set[Coord]], dict[Coord, set[Coord]], dict[int, Coord]]:
    rows = [r for r, _c in nodes]
    cols = [c for _r, c in nodes]
    r_shift = -min(rows)
    c_shift = -min(cols)
    norm_nodes = {(r + r_shift, c + c_shift) for r, c in nodes}
    norm_sources = {
        int(color): {(r + r_shift, c + c_shift) for r, c in srcs}
        for color, srcs in sources_by_color.items()
    }
    assert _span_ok(norm_nodes)

    adjacency = _graph_adjacency(norm_nodes)
    node_shapes = {
        node: _shape_for_node(
            {(nb[0] - node[0], nb[1] - node[1]) for nb in adjacency[node]}
        )
        for node in norm_nodes
    }

    max_r = max(node[0] for node in norm_nodes)
    max_c = max(node[1] for node in norm_nodes)
    base_h = CELL * (max_r + 1)
    base_w = CELL * (max_c + 1)
    top_pad, bottom_pad = _pad_split(base_h)
    left_pad, right_pad = _pad_split(base_w)
    grid = np.full(
        (base_h + top_pad + bottom_pad, base_w + left_pad + right_pad),
        BG,
        dtype=int,
    )

    node_cells: dict[Coord, set[Coord]] = {}
    for node, shape in node_shapes.items():
        base_r = top_pad + CELL * int(node[0])
        base_c = left_pad + CELL * int(node[1])
        cells = {(base_r + r, base_c + c) for r, c in shape}
        node_cells[node] = cells
        for r, c in cells:
            grid[r, c] = GREY

    labels, n_labels = label_mask(grid == GREY, connectivity=4)
    assert int(n_labels) == len(norm_nodes)

    node_to_label: dict[Coord, int] = {}
    for node, cells in node_cells.items():
        anchor = next(iter(cells))
        label_id = int(labels[anchor])
        assert label_id != 0
        node_to_label[node] = label_id

    placed_markers: dict[int, Coord] = {}
    used_marker_cells: set[Coord] = set()

    def marker_candidates(label_id: int) -> list[Coord]:
        out: list[Coord] = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if int(grid[r, c]) != BG:
                    continue
                touched = {
                    int(labels[rr, cc])
                    for rr, cc in neighbors4(r, c)
                    if in_bounds(grid.shape, rr, cc) and int(labels[rr, cc]) != 0
                }
                if touched == {label_id}:
                    out.append((int(r), int(c)))
        return out

    for color, src_nodes in norm_sources.items():
        assert len(src_nodes) == 1
        src_node = next(iter(src_nodes))
        label_id = node_to_label[src_node]
        candidates = marker_candidates(label_id)
        random.shuffle(candidates)
        chosen = None
        for cell in candidates:
            if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in used_marker_cells):
                continue
            chosen = cell
            break
        assert chosen is not None, "failed to place marker"
        grid[chosen] = int(color)
        placed_markers[int(color)] = chosen
        used_marker_cells.add(chosen)

    return grid, adjacency, node_cells, placed_markers


def _naive_marker_owner(
    node_cells: set[Coord],
    markers: dict[int, Coord],
) -> int:
    best_dist: int | None = None
    best_colors: list[int] = []
    for color, marker in markers.items():
        d = min(abs(r - marker[0]) + abs(c - marker[1]) for r, c in node_cells)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_colors = [int(color)]
        elif d == best_dist:
            best_colors.append(int(color))
    return GREY if len(best_colors) != 1 else int(best_colors[0])


def _validate_instance(
    grid: np.ndarray,
    abstract_adjacency: dict[Coord, set[Coord]],
    node_cells: dict[Coord, set[Coord]],
    sources_by_color: dict[int, set[Coord]],
    markers: dict[int, Coord],
) -> None:
    info = _analyze_grid(grid)
    labels = info["labels"]
    recovered_adjacency = info["adjacency"]
    recovered_sources = info["sources_by_color"]

    assert len(recovered_adjacency) == len(abstract_adjacency)
    assert {int(color): len(nodes) for color, nodes in recovered_sources.items()} == {
        int(color): len(nodes) for color, nodes in sources_by_color.items()
    }

    node_to_label: dict[Coord, int] = {}
    for node, cells in node_cells.items():
        anchor = next(iter(cells))
        label_id = int(labels[anchor])
        assert label_id != 0
        node_to_label[node] = label_id

    for node, neighbors in abstract_adjacency.items():
        expected = {node_to_label[nb] for nb in neighbors}
        assert recovered_adjacency[node_to_label[node]] == expected

    for color, src_nodes in sources_by_color.items():
        expected = {node_to_label[node] for node in src_nodes}
        assert recovered_sources[int(color)] == expected

    owners, meta = _resolve_ownership(abstract_adjacency, sources_by_color)
    out = main(grid)
    assert np.all(out[(grid != BG) & (grid != GREY)] == BG)

    saw_unique_multihop = False
    saw_tie = False
    saw_unreachable = False
    saw_shortcut_mismatch = False

    for node, cells in node_cells.items():
        expected_color = int(owners[node])
        actual_colors = {int(out[r, c]) for r, c in cells}
        assert len(actual_colors) == 1
        assert actual_colors == {expected_color}

        kind, dist, _colors = meta[node]
        if kind == "unique" and dist is not None and dist > 0:
            saw_unique_multihop = True
        elif kind == "tie":
            saw_tie = True
        elif kind == "unreachable":
            saw_unreachable = True

        naive = _naive_marker_owner(cells, markers)
        if naive != expected_color:
            saw_shortcut_mismatch = True

    assert saw_unique_multihop, "generator must expose multi-hop unique ownership"
    assert saw_tie, "generator must expose tie-freeze behavior"
    assert saw_unreachable, "generator must expose unreachable gray behavior"
    assert saw_shortcut_mismatch, "generator must reject purely geometric nearest-marker shortcuts"


def _gen_attempt() -> np.ndarray | None:
    try:
        main_path = _generate_main_path()
        occupied = set(main_path)
        branch = _try_add_branch(occupied, main_path)
        occupied.update(branch)
        isolated = _choose_isolated_node(occupied)
        assert isolated is not None
        occupied.add(isolated)
        assert _span_ok(occupied)

        colors = random.sample(MARKER_PALETTE, 2)
        sources_by_color = {
            int(colors[0]): {main_path[0]},
            int(colors[1]): {main_path[-1]},
        }

        adjacency = _graph_adjacency(occupied)
        owners, meta = _resolve_ownership(adjacency, sources_by_color)
        center = main_path[len(main_path) // 2]
        assert meta[center][0] == "tie"
        assert meta[isolated][0] == "unreachable"
        assert any(
            meta[node][0] == "unique" and meta[node][1] is not None and int(meta[node][1]) > 0
            for node in occupied
        )
        assert any(int(color) != GREY for color in owners.values())

        grid, norm_adjacency, node_cells, markers = _draw_instance(occupied, sources_by_color)
        _validate_instance(grid, norm_adjacency, node_cells, {
            color: {(
                node[0] - min(r for r, _c in occupied),
                node[1] - min(c for _r, c in occupied),
            ) for node in srcs}
            for color, srcs in sources_by_color.items()
        }, markers)

        assert int(get_bg(grid)) == BG
        assert int((grid == GREY).sum()) > sum(int((grid == color).sum()) for color in colors)
        return grid
    except AssertionError:
        return None
    except RuntimeError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid diagonal-relay recoloring input")
