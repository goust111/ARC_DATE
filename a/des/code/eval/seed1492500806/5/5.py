import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
WHITE = 9

INPUT_COLORS = [BLUE, RED, YELLOW]
GATE_MARKERS = {ORANGE: "mix", AZURE: "select"}
SELECT_ORDER = [RED, ORANGE, YELLOW, GREEN, AZURE, BLUE, MAGENTA]
BASE_PARTS = {
    BLUE: frozenset({"b"}),
    RED: frozenset({"r"}),
    YELLOW: frozenset({"y"}),
    GREEN: frozenset({"b", "y"}),
    MAGENTA: frozenset({"r", "b"}),
    ORANGE: frozenset({"r", "y"}),
    AZURE: frozenset({"r", "b", "y"}),
}
PART_COLOR = {v: k for k, v in BASE_PARTS.items()}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _side_cells(r: int, c: int, h: int, w: int) -> dict[str, list[tuple[int, int]]]:
    sides = {
        "top": [(r - 1, c), (r - 1, c + 1)],
        "bottom": [(r + 2, c), (r + 2, c + 1)],
        "left": [(r, c - 1), (r + 1, c - 1)],
        "right": [(r, c + 2), (r + 1, c + 2)],
    }
    return {
        name: [(rr, cc) for rr, cc in cells if 0 <= rr < h and 0 <= cc < w]
        for name, cells in sides.items()
    }


def _marker_on_side(g: np.ndarray, r: int, c: int, colors: set[int]) -> int | None:
    h, w = g.shape
    for cells in _side_cells(r, c, h, w).values():
        for rr, cc in cells:
            color = int(g[rr, cc])
            if color in colors:
                return color
    return None


def _overlaps(candidate: tuple[int, int, int, int], accepted: list[dict]) -> bool:
    r0, c0, r1, c1 = candidate
    for node in accepted:
        a0, b0, a1, b1 = node["bbox"]
        if not (r1 < a0 or a1 < r0 or c1 < b0 or b1 < c0):
            return True
    return False


def _find_nodes(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    raw = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = g[r : r + 2, c : c + 2]
            if not np.all(block == block[0, 0]):
                continue
            color = int(block[0, 0])
            bbox = (r, c, r + 1, c + 1)
            if color in INPUT_COLORS and _marker_on_side(g, r, c, {GREEN}) == GREEN:
                raw.append({"type": "input", "color": color, "bbox": bbox})
            elif color == WHITE:
                marker = _marker_on_side(g, r, c, set(GATE_MARKERS))
                if marker is not None:
                    raw.append({"type": "gate", "gate": GATE_MARKERS[marker], "bbox": bbox})
            elif color == GREY and _marker_on_side(g, r, c, {RED}) == RED:
                raw.append({"type": "output", "bbox": bbox})

    priority = {"gate": 0, "input": 1, "output": 2}
    raw.sort(key=lambda n: (priority[n["type"]], n["bbox"][0], n["bbox"][1]))
    nodes: list[dict] = []
    for node in raw:
        if not _overlaps(node["bbox"], nodes):
            node["id"] = len(nodes)
            nodes.append(node)
    assert nodes and any(n["type"] == "input" for n in nodes)
    assert any(n["type"] == "output" for n in nodes)
    return nodes


def _node_cells(nodes: list[dict]) -> dict[tuple[int, int], int]:
    out = {}
    for node in nodes:
        r0, c0, r1, c1 = node["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[(r, c)] = node["id"]
    return out


def _center(node: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = node["bbox"]
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _edge_direction(a: dict, b: dict) -> tuple[int, int]:
    if a["type"] == "input" or b["type"] == "output":
        return a["id"], b["id"]
    if b["type"] == "input" or a["type"] == "output":
        return b["id"], a["id"]
    ar, ac = _center(a)
    br, bc = _center(b)
    if ac != bc:
        return (a["id"], b["id"]) if ac < bc else (b["id"], a["id"])
    return (a["id"], b["id"]) if ar < br else (b["id"], a["id"])


def _build_edges(g: np.ndarray, nodes: list[dict]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    node_at = _node_cells(nodes)
    node_mask = np.zeros_like(g, dtype=bool)
    for r, c in node_at:
        node_mask[r, c] = True
    wire_mask = (g == GREY) & ~node_mask

    undirected = set()
    for comp in _components(wire_mask):
        adjacent = set()
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nid = node_at.get((r + dr, c + dc))
                if nid is not None:
                    adjacent.add(nid)
        if len(adjacent) >= 2:
            for a, b in combinations(sorted(adjacent), 2):
                undirected.add((a, b))

    by_id = {node["id"]: node for node in nodes}
    incoming = {node["id"]: [] for node in nodes}
    outgoing = {node["id"]: [] for node in nodes}
    for a, b in undirected:
        src, dst = _edge_direction(by_id[a], by_id[b])
        outgoing[src].append(dst)
        incoming[dst].append(src)
    return incoming, outgoing


def _mix(a: int, b: int) -> int:
    if a == b:
        return int(a)
    parts = BASE_PARTS[int(a)] | BASE_PARTS[int(b)]
    return PART_COLOR[parts]


def _select(a: int, b: int) -> int:
    rank = {color: i for i, color in enumerate(SELECT_ORDER)}
    return int(a) if rank[int(a)] <= rank[int(b)] else int(b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Evaluate a small color-signal circuit drawn with 2x2 nodes and grey wires.
    """
    g = np.asarray(grid, dtype=int)
    nodes = _find_nodes(g)
    incoming, _outgoing = _build_edges(g, nodes)
    nodes_by_id = {node["id"]: node for node in nodes}

    values: dict[int, int] = {}
    for node in nodes:
        if node["type"] == "input":
            values[node["id"]] = int(node["color"])

    for node in sorted(nodes, key=lambda n: (_center(n)[1], _center(n)[0])):
        nid = node["id"]
        if node["type"] == "input":
            continue
        src_values = [values[src] for src in incoming[nid] if src in values]
        if node["type"] == "gate":
            assert len(src_values) == 2, "logic gates must have two evaluated inputs"
            if node["gate"] == "mix":
                values[nid] = _mix(src_values[0], src_values[1])
            else:
                values[nid] = _select(src_values[0], src_values[1])
        elif node["type"] == "output":
            assert len(src_values) == 1, "outputs must have one evaluated input"
            values[nid] = int(src_values[0])

    out = g.copy()
    for nid, value in values.items():
        node = nodes_by_id[nid]
        if node["type"] == "input":
            continue
        r0, c0, r1, c1 = node["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(value)
    return out


def _draw_segment(g: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        g[r0, lo : hi + 1] = GREY
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        g[lo : hi + 1, c0] = GREY
    else:
        raise ValueError("segments must be orthogonal")


def _draw_path(g: np.ndarray, points: list[tuple[int, int]]) -> None:
    for a, b in zip(points, points[1:]):
        _draw_segment(g, a, b)


def _place_node(g: np.ndarray, r: int, c: int, color: int) -> None:
    g[r : r + 2, c : c + 2] = int(color)


def generate_input() -> np.ndarray:
    yoff = random.randint(0, 1)
    width = random.randint(32, 36)
    height = 29 + yoff
    out_c = width - 3
    grid = np.full((height, width), BLACK, dtype=int)

    # Four left-edge inputs feed two first-layer gates, then a final gate and one output.
    input_rows = [2 + yoff, 9 + yoff, 14 + yoff, 23 + yoff]
    colors = [BLUE, RED, YELLOW, random.choice(INPUT_COLORS)]
    random.shuffle(colors)
    for r, color in zip(input_rows, colors):
        _place_node(grid, r, 1, color)
        grid[r, 0] = GREEN

    g1 = (6 + yoff, 10)
    g2 = (18 + yoff, 10)
    g3 = (12 + yoff, 20)
    gate_types = [ORANGE, AZURE, random.choice([ORANGE, AZURE])]
    random.shuffle(gate_types)
    for (r, c), marker in zip([g1, g2, g3], gate_types):
        _place_node(grid, r, c, WHITE)
        if (r, c) == g3:
            grid[r, c - 1] = marker
        else:
            grid[r + 2, c] = marker

    output = (12 + yoff, out_c)
    _place_node(grid, output[0], output[1], GREY)
    grid[output[0], width - 1] = RED

    # Inputs to first-layer gates.
    _draw_path(grid, [(input_rows[0], 3), (input_rows[0], 10), (g1[0] - 1, 10)])
    _draw_path(grid, [(input_rows[1] + 1, 3), (input_rows[1] + 1, 9), (g1[0], 9)])
    _draw_path(grid, [(input_rows[2], 3), (input_rows[2], 10), (g2[0] - 1, 10)])
    _draw_path(grid, [(input_rows[3] + 1, 3), (input_rows[3] + 1, 9), (g2[0], 9)])

    # First-layer outputs into the final gate, then final output.
    _draw_path(grid, [(g1[0] + 1, 12), (g1[0] + 1, 20), (g3[0] - 1, 20)])
    _draw_path(grid, [(g2[0] + 1, 12), (g2[0] + 1, 20), (g3[0] + 2, 20)])
    _draw_path(grid, [(g3[0], 22), (g3[0], out_c - 1)])

    # Construction-time validation catches accidental wire contacts after random sizing.
    result = main(grid)
    assert isinstance(result, np.ndarray)
    return grid
