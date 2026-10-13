from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


FRAME = 8
BG = 0
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_ATTEMPTS = 1000


def _detect_line_indices(grid: np.ndarray, frame: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_hits = np.sum(grid == frame, axis=1)
    col_hits = np.sum(grid == frame, axis=0)
    line_rows = [int(r) for r in range(h) if int(row_hits[r]) >= max(3, w // 2)]
    line_cols = [int(c) for c in range(w) if int(col_hits[c]) >= max(3, h // 2)]
    return line_rows, line_cols


def _find_bucket(lines: list[int], x: int) -> int:
    for i in range(len(lines) - 1):
        if lines[i] < x < lines[i + 1]:
            return i
    raise ValueError(f"coordinate {x} not inside any chamber interval")


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    line_rows, line_cols = _detect_line_indices(g, FRAME)
    if len(line_rows) < 4 or len(line_cols) < 4:
        raise ValueError("failed to detect lattice rows/cols")
    if line_rows[0] != 0 or line_rows[-1] != g.shape[0] - 1:
        raise ValueError("outer horizontal frame missing")
    if line_cols[0] != 0 or line_cols[-1] != g.shape[1] - 1:
        raise ValueError("outer vertical frame missing")

    row_spans = [int(line_rows[i + 1] - line_rows[i] - 1) for i in range(len(line_rows) - 1)]
    col_spans = [int(line_cols[i + 1] - line_cols[i] - 1) for i in range(len(line_cols) - 1)]
    if not all(2 <= span <= 5 for span in row_spans + col_spans):
        raise ValueError("unexpected chamber size")

    n_rows = len(line_rows) - 1
    n_cols = len(line_cols) - 1
    chambers = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    markers = []
    marker_chambers: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere((g != BG) & (g != FRAME)):
        rr = int(r)
        cc = int(c)
        chamber = (_find_bucket(line_rows, rr), _find_bucket(line_cols, cc))
        if chamber in marker_chambers:
            raise ValueError("multiple markers in one chamber")
        marker_chambers[chamber] = len(markers)
        markers.append({"pos": (rr, cc), "color": int(g[rr, cc]), "chamber": chamber})
    if len(markers) < 2:
        raise ValueError("expected 2+ markers")

    graph = {ch: set() for ch in chambers}
    gaps: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    for r in range(n_rows - 1):
        wall_r = line_rows[r + 1]
        for c in range(n_cols):
            c0 = line_cols[c] + 1
            c1 = line_cols[c + 1]
            if np.any(g[wall_r, c0:c1] != FRAME):
                a = (r, c)
                b = (r + 1, c)
                graph[a].add(b)
                graph[b].add(a)
                gaps.append(("h", a, b))

    for r in range(n_rows):
        for c in range(n_cols - 1):
            wall_c = line_cols[c + 1]
            r0 = line_rows[r] + 1
            r1 = line_rows[r + 1]
            if np.any(g[r0:r1, wall_c] != FRAME):
                a = (r, c)
                b = (r, c + 1)
                graph[a].add(b)
                graph[b].add(a)
                gaps.append(("v", a, b))

    return {
        "grid": g,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "chambers": chambers,
        "markers": markers,
        "marker_chambers": marker_chambers,
        "graph": graph,
        "gaps": gaps,
    }


def _reachable_chambers(parsed: dict, start_index: int) -> set[tuple[int, int]]:
    markers = parsed["markers"]
    graph = parsed["graph"]
    start = markers[start_index]["chamber"]
    blocked = {m["chamber"] for i, m in enumerate(markers) if i != start_index}
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur != start and cur in blocked:
            continue
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    parsed = _parse_grid(grid)
    g = parsed["grid"]
    line_rows = parsed["line_rows"]
    line_cols = parsed["line_cols"]
    markers = parsed["markers"]

    reaches: list[set[tuple[int, int]]] = []
    counts: list[int] = []
    for i in range(len(markers)):
        reach = _reachable_chambers(parsed, i)
        reaches.append(reach)
        counts.append(len(reach))

    chamber_owner: dict[tuple[int, int], int] = {}
    chamber_reachers: dict[tuple[int, int], list[int]] = {}
    for chamber in parsed["chambers"]:
        reachers = [i for i, reach in enumerate(reaches) if chamber in reach]
        chamber_reachers[chamber] = reachers
        if not reachers:
            continue
        winner = min(
            reachers,
            key=lambda idx: (counts[idx], markers[idx]["chamber"][0], markers[idx]["chamber"][1], markers[idx]["color"]),
        )
        chamber_owner[chamber] = winner

    out = g.copy()
    for r in range(parsed["n_rows"]):
        for c in range(parsed["n_cols"]):
            r0 = line_rows[r] + 1
            r1 = line_rows[r + 1]
            c0 = line_cols[c] + 1
            c1 = line_cols[c + 1]
            chamber = (r, c)
            if chamber in chamber_owner:
                color = markers[chamber_owner[chamber]]["color"]
            else:
                color = BG
            out[r0:r1, c0:c1] = int(color)

    details = {
        "parsed": parsed,
        "reaches": reaches,
        "counts": counts,
        "chamber_owner": chamber_owner,
        "chamber_reachers": chamber_reachers,
    }
    return out, details


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fill lattice chambers by door-gap reachability, breaking conflicts in favor of the most constrained marker.

    Input structure:
    - Black cells form chamber interiors; cyan frame rows and columns partition the grid into a regular lattice of enclosed chambers.
    - A few cyan frame pixels are missing, creating black one-cell door gaps between adjacent chambers.
    - Two or three single-pixel colored markers occupy distinct chambers.

    Rule:
    - For each marker, start from its chamber and propagate through chamber-to-chamber door gaps while cyan frame cells remain blocked.
    - Count how many chambers each marker can reach; chambers containing other markers are terminal reachable chambers rather than transit chambers because their colored pixels block onward propagation.
    - Recolor every reachable chamber with the color of its unique reaching marker, or with the color of the reaching marker whose total reachable-chamber count is smallest when multiple markers can reach it.
    - Recolor the marker pixel together with its chamber interior, keep unreachable chambers black, and leave the cyan frame unchanged.

    Key insight:
    - Local flooding is not enough: conflicts can only be resolved after computing every marker's full reachable chamber set and ranking markers globally by how constrained their range is.
    """
    out, _details = _solve_details(np.asarray(input_grid, dtype=int))
    return out


def _candidate_edges(n_rows: int, n_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for r in range(n_rows):
        for c in range(n_cols):
            if r + 1 < n_rows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < n_cols:
                edges.append(((r, c), (r, c + 1)))
    return edges


def _graph_from_edges(
    n_rows: int,
    n_cols: int,
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    graph = {(r, c): set() for r in range(n_rows) for c in range(n_cols)}
    for a, b in edges:
        a = tuple(a)
        b = tuple(b)
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _layout_reachable(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    marker_chambers: tuple[tuple[int, int], ...],
    start_index: int,
) -> set[tuple[int, int]]:
    start = marker_chambers[start_index]
    blocked = set(marker_chambers)
    blocked.remove(start)
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur != start and cur in blocked:
            continue
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _layout_valid(
    n_rows: int,
    n_cols: int,
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    marker_chambers: tuple[tuple[int, int], ...],
) -> bool:
    chambers = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    reaches = [_layout_reachable(graph, marker_chambers, i) for i in range(len(marker_chambers))]
    counts = [len(reach) for reach in reaches]
    chamber_reachers = {ch: [i for i, reach in enumerate(reaches) if ch in reach] for ch in chambers}
    overlaps = [ch for ch, ids in chamber_reachers.items() if len(ids) >= 2]

    if len(set(counts)) != len(counts):
        return False
    if min(counts) < 2:
        return False
    if not overlaps:
        return False
    if not any(len(ids) == 1 for ids in chamber_reachers.values()):
        return False
    if not any(len(ids) == 0 for ids in chamber_reachers.values()):
        return False
    if len({tuple(sorted(reach)) for reach in reaches}) != len(reaches):
        return False
    for chamber in overlaps:
        ids = chamber_reachers[chamber]
        winner = min(ids, key=lambda idx: counts[idx])
        if any(counts[winner] < counts[other] for other in ids if other != winner):
            return True
    return False


def _choose_marker_layout(
    n_rows: int,
    n_cols: int,
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    marker_count: int,
) -> tuple[tuple[int, int], ...] | None:
    chambers = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    combos = list(itertools.combinations(chambers, marker_count))
    random.shuffle(combos)
    for combo in combos:
        if _layout_valid(n_rows, n_cols, graph, combo):
            return combo
    return None


def _three_marker_template(
    n_rows: int,
    n_cols: int,
) -> tuple[set[tuple[tuple[int, int], tuple[int, int]]], tuple[tuple[int, int], ...]]:
    base_markers = [(0, 0), (0, 1), (0, 2)]
    base_edges = [
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
        ((0, 1), (0, 2)),
    ]

    transforms = [
        ((2, 3), lambda r, c: (r, c)),
        ((2, 3), lambda r, c: (r, 2 - c)),
        ((3, 2), lambda r, c: (c, 1 - r)),
        ((3, 2), lambda r, c: (2 - c, r)),
    ]
    (h, w), transform = random.choice(transforms)
    r0 = random.randint(0, n_rows - h)
    c0 = random.randint(0, n_cols - w)

    def place(cell: tuple[int, int]) -> tuple[int, int]:
        rr, cc = transform(cell[0], cell[1])
        return (r0 + rr, c0 + cc)

    markers = tuple(place(cell) for cell in base_markers)
    edges = {tuple(sorted((place(a), place(b)))) for a, b in base_edges}
    return edges, markers


def _build_grid(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    markers: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    h = n_rows * cell_h + (n_rows + 1)
    w = n_cols * cell_w + (n_cols + 1)
    grid = np.full((h, w), BG, dtype=int)
    line_rows = [r * (cell_h + 1) for r in range(n_rows + 1)]
    line_cols = [c * (cell_w + 1) for c in range(n_cols + 1)]
    grid[line_rows, :] = FRAME
    grid[:, line_cols] = FRAME

    for a, b in edges:
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:
            row = line_rows[r1] + 1 + random.randrange(cell_h)
            col = line_cols[min(c1, c2) + 1]
            grid[row, col] = BG
        else:
            row = line_rows[min(r1, r2) + 1]
            col = line_cols[c1] + 1 + random.randrange(cell_w)
            grid[row, col] = BG

    for (r, c), color in markers:
        rr = line_rows[r] + 1 + random.randrange(cell_h)
        cc = line_cols[c] + 1 + random.randrange(cell_w)
        grid[rr, cc] = int(color)

    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, details = _solve_details(grid)
    except Exception:
        return False

    parsed = details["parsed"]
    reaches = details["reaches"]
    counts = details["counts"]
    chamber_reachers = details["chamber_reachers"]
    markers = parsed["markers"]

    if not (3 <= parsed["n_rows"] <= 4 and 3 <= parsed["n_cols"] <= 4):
        return False
    if not (2 <= len(parsed["gaps"]) <= 4):
        return False
    if not (2 <= len(markers) <= 3):
        return False
    if len({m["color"] for m in markers}) != len(markers):
        return False
    if len(set(counts)) != len(counts):
        return False
    if min(counts) < 2:
        return False

    overlaps = [ch for ch, ids in chamber_reachers.items() if len(ids) >= 2]
    if not overlaps:
        return False
    if not any(len(ids) == 1 for ids in chamber_reachers.values()):
        return False
    if not any(len(ids) == 0 for ids in chamber_reachers.values()):
        return False

    if not any(
        len(chamber_reachers[ch]) >= 2
        and len({markers[i]["color"] for i in chamber_reachers[ch]}) >= 2
        for ch in overlaps
    ):
        return False

    if len({tuple(sorted(reach)) for reach in reaches}) != len(reaches):
        return False

    winner_changes_conflict = False
    for chamber in overlaps:
        ids = chamber_reachers[chamber]
        winner = min(ids, key=lambda idx: counts[idx])
        if any(counts[winner] < counts[other] for other in ids if other != winner):
            winner_changes_conflict = True
            break
    if not winner_changes_conflict:
        return False

    if out.shape != grid.shape or max(out.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.choice((3, 4))
    n_cols = random.choice((3, 4))
    cell_h = random.choice((3, 4))
    cell_w = random.choice((3, 4))

    chamber_count = n_rows * n_cols
    marker_count = 3 if chamber_count >= 9 and random.random() < 0.35 else 2

    if marker_count == 3:
        edges, marker_chambers = _three_marker_template(n_rows, n_cols)
    else:
        candidate_edges = _candidate_edges(n_rows, n_cols)
        random.shuffle(candidate_edges)
        edge_count = random.randint(2, 4)
        edges = {tuple(sorted(edge)) for edge in candidate_edges[:edge_count]}
        graph = _graph_from_edges(n_rows, n_cols, edges)
        marker_chambers = _choose_marker_layout(n_rows, n_cols, graph, marker_count)
        if marker_chambers is None:
            return None

    marker_colors = random.sample(MARKER_COLORS, marker_count)
    markers = list(zip(marker_chambers, marker_colors))

    grid = _build_grid(n_rows, n_cols, cell_h, cell_w, edges, markers)
    if _semantic_ok(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
