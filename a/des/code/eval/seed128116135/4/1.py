import numpy as np


BG = 0
PIPE = 5
CONTAINER_SHAPES = (
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 2),
    (2, 1),
    (2, 3),
    (3, 2),
)
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return tuple(out)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _color_counts(g: np.ndarray) -> dict[int, int]:
    vals, cnts = np.unique(g, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts, strict=False)}


def _background_color(g: np.ndarray) -> int:
    counts = _color_counts(g)
    return max(counts, key=counts.get)


def _pipe_color(g: np.ndarray, bg: int) -> int:
    counts = _color_counts(g)
    non_bg = [(cnt, col) for col, cnt in counts.items() if col != bg]
    assert non_bg, "grid must contain non-background pixels"
    return max(non_bg)[1]


def _background_components(g: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != bg or int(labels[r, c]) >= 0:
                continue
            q = [(r, c)]
            labels[r, c] = cid
            cells = []
            touches_border = False
            r0 = r1 = r
            c0 = c1 = c
            head = 0
            while head < len(q):
                rr, cc = q[head]
                head += 1
                cells.append((rr, cc))
                touches_border |= rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == bg and int(labels[nr, nc]) < 0:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            rectangular = len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
            comps.append(
                {
                    "cells": cells,
                    "touches_border": touches_border,
                    "rectangular": rectangular,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
            cid += 1
    return labels, comps


def _find_seed(g: np.ndarray, bg: int, pipe: int) -> tuple[int, int]:
    h, w = g.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v == bg or v == pipe:
                continue
            if any(_in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == pipe for nr, nc in _neighbors4(r, c)):
                candidates.append((c, r))
    assert candidates, "a colored seed pixel adjacent to the pipe is required"
    c, r = min(candidates)
    return int(r), int(c)


def _branch_marker_colors(
    g: np.ndarray, cell: tuple[int, int], bg: int, pipe: int, seed_pos: tuple[int, int]
) -> set[int]:
    r, c = cell
    colors = set()
    for nr, nc in _neighbors8(r, c):
        if not _in_bounds(g.shape, nr, nc):
            continue
        if (nr, nc) == seed_pos:
            continue
        v = int(g[nr, nc])
        if v != bg and v != pipe:
            colors.add(v)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the seed color through a grey branch network and fill the reached terminal container.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    pipe = _pipe_color(g, bg)
    seed_pos = _find_seed(g, bg, pipe)
    seed_color = int(g[seed_pos])

    bg_labels, bg_comps = _background_components(g, bg)
    start_neighbors = [p for p in _neighbors4(*seed_pos) if _in_bounds(g.shape, p[0], p[1]) and int(g[p]) == pipe]
    assert start_neighbors, "seed must touch the pipe network"
    cur = min(start_neighbors, key=lambda p: (p[1], p[0]))
    prev = seed_pos

    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    terminal_comp = None

    while True:
        if cur in seen:
            break
        seen.add(cur)
        path.append(cur)

        found_terminal = None
        for nr, nc in _neighbors4(*cur):
            if not _in_bounds(g.shape, nr, nc) or int(g[nr, nc]) != bg:
                continue
            cid = int(bg_labels[nr, nc])
            if cid < 0:
                continue
            comp = bg_comps[cid]
            if (
                not comp["touches_border"]
                and comp["rectangular"]
                and comp["height"] <= 3
                and comp["width"] <= 3
            ):
                found_terminal = cid
                break
        if found_terminal is not None:
            terminal_comp = found_terminal
            break

        nxts = []
        for nr, nc in _neighbors4(*cur):
            if not _in_bounds(g.shape, nr, nc):
                continue
            if (nr, nc) == prev or int(g[nr, nc]) != pipe or (nr, nc) in seen:
                continue
            nxts.append((nr, nc))

        if not nxts:
            break
        if len(nxts) == 1:
            prev, cur = cur, nxts[0]
            continue

        chosen = None
        for nb in nxts:
            if seed_color in _branch_marker_colors(g, nb, bg, pipe, seed_pos):
                chosen = nb
                break
        assert chosen is not None, "branch selection must be determined by marker colors"
        prev, cur = cur, chosen

    out = g.copy()
    for r, c in path:
        out[r, c] = seed_color
    if terminal_comp is not None:
        for r, c in bg_comps[terminal_comp]["cells"]:
            out[r, c] = seed_color
    return out


def _draw_frame(g: np.ndarray, row: int, left: int, inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    top = row - 1
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    g[top, left : right + 1] = PIPE
    g[bottom, left : right + 1] = PIPE
    g[top : bottom + 1, left] = PIPE
    g[top : bottom + 1, right] = PIPE
    return [(rr, cc) for rr in range(top + 1, bottom) for cc in range(left + 1, right)]


def _connect(parent_row: int, parent_col: int, child_row: int, child_col: int) -> list[tuple[int, int]]:
    cells = []
    r0, r1 = sorted((parent_row, child_row))
    for rr in range(r0, r1 + 1):
        cells.append((rr, parent_col))
    for cc in range(parent_col + 1, child_col + 1):
        cells.append((child_row, cc))
    return cells


def _build_tree(leaf_rows: list[int], branch_cols: list[int], frame_col: int, rng: np.random.Generator, seed_color: int):
    spare_colors = [c for c in MARKER_COLORS if c != seed_color]

    def rec(lo: int, hi: int, level: int):
        if hi - lo == 1:
            return {"kind": "leaf", "leaf_idx": lo, "row": leaf_rows[lo], "col": frame_col}
        mid = (lo + hi) // 2
        upper = rec(lo, mid, level + 1)
        lower = rec(mid, hi, level + 1)
        row = (upper["row"] + lower["row"]) // 2
        return {
            "kind": "node",
            "row": row,
            "col": branch_cols[level],
            "upper": upper,
            "lower": lower,
            "take_upper": bool(rng.integers(0, 2)),
            "other_color": int(rng.choice(spare_colors)),
        }

    return rec(0, len(leaf_rows), 0)


def _paint_network(g: np.ndarray, node: dict, seed_color: int) -> None:
    if node["kind"] == "leaf":
        return
    _paint_network(g, node["upper"], seed_color)
    _paint_network(g, node["lower"], seed_color)
    for child in (node["upper"], node["lower"]):
        for r, c in _connect(node["row"], node["col"], child["row"], child["col"]):
            g[r, c] = PIPE

    other = node["other_color"]
    upper_marker = seed_color if node["take_upper"] else other
    lower_marker = other if node["take_upper"] else seed_color
    g[node["row"] - 1, node["col"] - 1] = upper_marker
    g[node["row"] + 1, node["col"] - 1] = lower_marker


def _selected_path(root: dict) -> tuple[set[tuple[int, int]], int]:
    cells = set()
    for c in range(1, root["col"] + 1):
        cells.add((root["row"], c))

    node = root
    chosen_leaf = -1
    while node["kind"] != "leaf":
        child = node["upper"] if node["take_upper"] else node["lower"]
        for cell in _connect(node["row"], node["col"], child["row"], child["col"]):
            cells.add(cell)
        node = child
    chosen_leaf = int(node["leaf_idx"])
    return cells, chosen_leaf


def _expected_output(inp: np.ndarray, path_cells: set[tuple[int, int]], interior: list[tuple[int, int]], seed_color: int) -> np.ndarray:
    out = inp.copy()
    for r, c in path_cells:
        out[r, c] = seed_color
    for r, c in interior:
        out[r, c] = seed_color
    return out


def _make_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    depth = int(rng.choice([1, 2, 3]))
    leaf_count = 2 ** depth
    seed_color = int(rng.choice(MARKER_COLORS))

    shapes = [CONTAINER_SHAPES[int(rng.integers(0, len(CONTAINER_SHAPES)))] for _ in range(leaf_count)]
    leaf_rows = []
    cursor = int(rng.integers(2, 5))
    for inner_h, _inner_w in shapes:
        leaf_rows.append(cursor)
        cursor += inner_h + int(rng.integers(2, 5))

    branch_cols = []
    x = int(rng.integers(3, 6))
    for _ in range(depth):
        branch_cols.append(x)
        x += int(rng.integers(4, 7))
    frame_col = x + int(rng.integers(2, 5))

    max_inner_w = max(w for _, w in shapes)
    height = leaf_rows[-1] + shapes[-1][0] + 3 + int(rng.integers(0, 3))
    width = frame_col + max_inner_w + 4 + int(rng.integers(0, 3))
    g = np.full((height, width), BG, dtype=int)

    root = _build_tree(leaf_rows, branch_cols, frame_col, rng, seed_color)
    _paint_network(g, root, seed_color)
    g[root["row"], 1 : root["col"] + 1] = PIPE

    interiors = []
    for row, (inner_h, inner_w) in zip(leaf_rows, shapes, strict=False):
        interiors.append(_draw_frame(g, row, frame_col, inner_h, inner_w))

    g[root["row"], 0] = seed_color

    path_cells, chosen_leaf = _selected_path(root)
    out = _expected_output(g, path_cells, interiors[chosen_leaf], seed_color)
    return g, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        inp, expected = _make_instance(rng)
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
