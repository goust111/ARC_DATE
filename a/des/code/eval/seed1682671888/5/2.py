from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(int(n)))

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    parts: list[list[tuple[int, int]]] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or labels[r, c] >= 0:
                continue
            q = deque([(int(r), int(c))])
            labels[r, c] = cid
            pts = [(int(r), int(c))]
            while q:
                rr, cc = q.popleft()
                for dr, dc in _DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and labels[nr, nc] < 0:
                        labels[nr, nc] = cid
                        q.append((int(nr), int(nc)))
                        pts.append((int(nr), int(nc)))
            parts.append(pts)
            cid += 1
    return labels, parts


def _border_color(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _background_color(g: np.ndarray, wall: int) -> int:
    vals, cnts = np.unique(g[g != int(wall)], return_counts=True)
    assert len(vals) > 0, "expected non-wall cells"
    return int(vals[int(np.argmax(cnts))])


def _detect_doors(g: np.ndarray, wall: int, bg: int) -> tuple[np.ndarray, dict[tuple[int, int], str]]:
    h, w = g.shape
    door_mask = np.zeros((h, w), dtype=bool)
    door_kind: dict[tuple[int, int], str] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != int(bg):
                continue
            vertical = (
                int(g[r - 1, c]) == int(wall)
                and int(g[r + 1, c]) == int(wall)
                and int(g[r, c - 1]) != int(wall)
                and int(g[r, c + 1]) != int(wall)
            )
            horizontal = (
                int(g[r, c - 1]) == int(wall)
                and int(g[r, c + 1]) == int(wall)
                and int(g[r - 1, c]) != int(wall)
                and int(g[r + 1, c]) != int(wall)
            )
            if vertical ^ horizontal:
                door_mask[r, c] = True
                door_kind[(int(r), int(c))] = "vertical" if vertical else "horizontal"
    return door_mask, door_kind


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use door color-matching to decide which rooms connect, then fill each connected set
    with the marker color common to all rooms in that set.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall = _border_color(g)
    bg = _background_color(g, wall)
    door_mask, door_kind = _detect_doors(g, wall, bg)

    room_mask = (g != int(wall)) & (~door_mask)
    room_labels, rooms = _components(room_mask)
    assert rooms, "expected at least one room"

    room_color_sets: list[set[int]] = []
    room_color_counts: list[Counter[int]] = []
    for pts in rooms:
        counts: Counter[int] = Counter()
        for r, c in pts:
            v = int(g[r, c])
            if v != int(bg):
                counts[v] += 1
        assert counts, "every room must contain at least one marker"
        room_color_sets.append(set(counts))
        room_color_counts.append(counts)

    dsu = _DSU(len(rooms))
    for (r, c), kind in door_kind.items():
        if kind == "vertical":
            a = int(room_labels[r, c - 1])
            b = int(room_labels[r, c + 1])
        else:
            a = int(room_labels[r - 1, c])
            b = int(room_labels[r + 1, c])
        assert a >= 0 and b >= 0 and a != b, "door must separate two rooms"
        if room_color_sets[a] & room_color_sets[b]:
            dsu.union(a, b)

    groups: dict[int, list[int]] = {}
    for rid in range(len(rooms)):
        groups.setdefault(dsu.find(rid), []).append(rid)

    fill_color: dict[int, int] = {}
    for root, members in groups.items():
        common = set(room_color_sets[members[0]])
        total_counts: Counter[int] = Counter()
        for rid in members:
            common &= room_color_sets[rid]
            total_counts.update(room_color_counts[rid])
        assert common, "connected rooms must share at least one marker color"
        fill = max(sorted(common), key=lambda col: (int(total_counts[col]), -int(col)))
        fill_color[root] = int(fill)

    out = g.copy()
    for rid, pts in enumerate(rooms):
        color = fill_color[dsu.find(rid)]
        for r, c in pts:
            if int(g[r, c]) == int(bg):
                out[r, c] = int(color)
    return out


# --------------- generator ---------------

_GEN_BG = 0
_GEN_WALL = 5
_GEN_MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_ROOM_ROWS = [2, 3, 4]
_GEN_ROOM_COLS = [2, 3, 4]
_GEN_ROOM_SIZE_RANGE = (3, 7)
_GEN_DOOR_PROB_RANGE = (0.45, 0.85)
_GEN_SECOND_MARKER_PROB = 0.45
_GEN_MAX_ATTEMPTS = 400


def _room_id(n_cols: int, rr: int, cc: int) -> int:
    return int(rr * n_cols + cc)


def _choose_statuses(n_edges: int) -> list[bool] | None:
    if n_edges <= 0:
        return None
    mode = random.choice(["mix", "mix", "open_heavy", "locked_heavy"])
    for _ in range(80):
        if mode == "mix":
            prob = 0.5
        elif mode == "open_heavy":
            prob = 0.75
        else:
            prob = 0.25
        states = [bool(random.random() < prob) for _ in range(n_edges)]
        if n_edges == 1:
            return states
        if mode == "mix" and any(states) and not all(states):
            return states
        if mode == "open_heavy" and any(states):
            return states
        if mode == "locked_heavy" and not all(states):
            return states
    return None


def _assign_component_colors(
    n_components: int, adjacency: dict[int, set[int]]
) -> dict[int, int] | None:
    order = list(range(int(n_components)))
    random.shuffle(order)
    order.sort(key=lambda x: len(adjacency.get(int(x), set())), reverse=True)

    color_of: dict[int, int] = {}
    for comp in order:
        blocked = {color_of[nbr] for nbr in adjacency.get(int(comp), set()) if nbr in color_of}
        choices = [c for c in _GEN_MARKER_COLORS if c not in blocked]
        if not choices:
            return None
        color_of[int(comp)] = int(random.choice(choices))
    return color_of


def _pick_marker_positions(r0: int, r1: int, c0: int, c1: int, n_markers: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(cells)
    picked: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in picked):
            picked.append(cell)
            if len(picked) == n_markers:
                return picked
    assert False, "room should be large enough for isolated markers"


def _generate_candidate() -> np.ndarray | None:
    n_room_rows = random.choice(_GEN_ROOM_ROWS)
    n_room_cols = random.choice(_GEN_ROOM_COLS)
    room_heights = [random.randrange(*_GEN_ROOM_SIZE_RANGE) for _ in range(n_room_rows)]
    room_widths = [random.randrange(*_GEN_ROOM_SIZE_RANGE) for _ in range(n_room_cols)]

    row_lines = [0]
    for h in room_heights:
        row_lines.append(row_lines[-1] + h + 1)
    col_lines = [0]
    for w in room_widths:
        col_lines.append(col_lines[-1] + w + 1)

    H = row_lines[-1] + 1
    W = col_lines[-1] + 1
    grid = np.full((H, W), _GEN_BG, dtype=int)
    grid[row_lines, :] = _GEN_WALL
    grid[:, col_lines] = _GEN_WALL

    room_boxes: dict[int, tuple[int, int, int, int]] = {}
    for rr in range(n_room_rows):
        for cc in range(n_room_cols):
            rid = _room_id(n_room_cols, rr, cc)
            room_boxes[rid] = (
                row_lines[rr] + 1,
                row_lines[rr + 1] - 1,
                col_lines[cc] + 1,
                col_lines[cc + 1] - 1,
            )

    edges: list[dict[str, object]] = []
    for rr in range(n_room_rows):
        for cc in range(n_room_cols - 1):
            if random.random() >= random.uniform(*_GEN_DOOR_PROB_RANGE):
                continue
            a = _room_id(n_room_cols, rr, cc)
            b = _room_id(n_room_cols, rr, cc + 1)
            r0, r1, _, _ = room_boxes[a]
            edges.append(
                {
                    "a": a,
                    "b": b,
                    "kind": "vertical",
                    "cell": (random.randint(r0, r1), col_lines[cc + 1]),
                }
            )
    for rr in range(n_room_rows - 1):
        for cc in range(n_room_cols):
            if random.random() >= random.uniform(*_GEN_DOOR_PROB_RANGE):
                continue
            a = _room_id(n_room_cols, rr, cc)
            b = _room_id(n_room_cols, rr + 1, cc)
            _, _, c0, c1 = room_boxes[a]
            edges.append(
                {
                    "a": a,
                    "b": b,
                    "kind": "horizontal",
                    "cell": (row_lines[rr + 1], random.randint(c0, c1)),
                }
            )

    if not edges:
        return None
    states = _choose_statuses(len(edges))
    if states is None:
        return None

    dsu = _DSU(n_room_rows * n_room_cols)
    n_open = 0
    n_locked = 0
    for edge, is_open in zip(edges, states, strict=False):
        edge["open"] = bool(is_open)
        if is_open:
            dsu.union(int(edge["a"]), int(edge["b"]))
            n_open += 1
        else:
            n_locked += 1

    comp_index: dict[int, int] = {}
    room_to_comp: dict[int, int] = {}
    for rid in range(n_room_rows * n_room_cols):
        root = dsu.find(rid)
        if root not in comp_index:
            comp_index[root] = len(comp_index)
        room_to_comp[rid] = comp_index[root]

    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(comp_index))}
    for edge in edges:
        if bool(edge["open"]):
            continue
        ca = room_to_comp[int(edge["a"])]
        cb = room_to_comp[int(edge["b"])]
        if ca != cb:
            adjacency[ca].add(cb)
            adjacency[cb].add(ca)

    comp_colors = _assign_component_colors(len(comp_index), adjacency)
    if comp_colors is None:
        return None

    for edge in edges:
        r, c = edge["cell"]
        grid[int(r), int(c)] = _GEN_BG

    for rid, (r0, r1, c0, c1) in room_boxes.items():
        color = comp_colors[room_to_comp[rid]]
        n_markers = 2 if random.random() < _GEN_SECOND_MARKER_PROB else 1
        for r, c in _pick_marker_positions(r0, r1, c0, c1, n_markers):
            grid[int(r), int(c)] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if n_open == 0 and n_locked == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
