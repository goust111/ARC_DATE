from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
PIPE = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


Cell = tuple[int, int]


def _neighbors(shape: tuple[int, int], r: int, c: int) -> list[Cell]:
    out: list[Cell] = []
    h, w = shape
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _terminal_component(grid: np.ndarray, start: Cell, blocked: Cell | None) -> set[Cell]:
    stack = [start]
    seen = {blocked} if blocked is not None else set()
    cells: set[Cell] = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        if int(grid[r, c]) != PIPE:
            continue
        cells.add(cell)
        for nxt in _neighbors(grid.shape, r, c):
            if nxt not in seen and int(grid[nxt]) == PIPE:
                stack.append(nxt)
    assert cells, "terminal component must be non-empty"
    return cells


def _find_roots(grid: np.ndarray) -> dict[int, Cell]:
    roots: dict[int, Cell] = {}
    colors = sorted(int(x) for x in np.unique(grid) if int(x) not in (BG, PIPE))
    assert colors, "expected at least one colored seed tree"
    for color in colors:
        mask = (grid == PIPE) | (grid == color)
        colored = np.argwhere(grid == color)
        candidates: list[Cell] = []
        for r0, c0 in colored:
            r = int(r0)
            c = int(c0)
            deg = sum(bool(mask[rr, cc]) for rr, cc in _neighbors(grid.shape, r, c))
            if deg == 1:
                candidates.append((r, c))
            else:
                assert deg == 2, "non-root colored pixels must be in-line markers"
        assert len(candidates) == 1, "each seed color must have one root pixel"
        roots[color] = candidates[0]
    return roots


def _trace_terminal(grid: np.ndarray, color: int, root: Cell) -> set[Cell]:
    mask = (grid == PIPE) | (grid == color)
    prev: Cell | None = None
    cur = root

    while True:
        r, c = cur
        nbrs = [nxt for nxt in _neighbors(grid.shape, r, c) if nxt != prev and bool(mask[nxt])]

        if len(nbrs) == 0:
            assert int(grid[r, c]) == PIPE, "path should end at a grey terminal entry"
            return _terminal_component(grid, cur, prev)

        if len(nbrs) == 1:
            prev, cur = cur, nbrs[0]
            continue

        marked = [nxt for nxt in nbrs if int(grid[nxt]) == color]
        if len(marked) == 1:
            prev, cur = cur, marked[0]
            continue

        assert int(grid[r, c]) == PIPE, "only grey cells can open into a terminal"
        assert not marked, "branch choice must be unique"
        return _terminal_component(grid, cur, prev)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each seed through the grey pipe tree, use same-colored markers to
    choose branches, erase the markers, and recolor only the reached terminal.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    roots = _find_roots(g)

    for color, root in roots.items():
        for r0, c0 in np.argwhere(g == color):
            cell = (int(r0), int(c0))
            if cell != root:
                out[cell] = BG

        for r, c in _trace_terminal(g, color, root):
            out[r, c] = color

    return out


@dataclass
class Node:
    kind: str
    length: int
    upper: "Node | None" = None
    lower: "Node | None" = None
    choose_upper: bool = True
    shape: tuple[Cell, ...] | None = None
    leaf_id: int = -1


TERMINAL_SHAPES: tuple[tuple[Cell, ...], ...] = (
    ((-1, 0), (0, 0), (1, 0)),
    ((-1, 0), (0, 0), (-1, 1), (0, 1)),
    ((0, 0), (1, 0), (0, 1), (1, 1)),
    ((-1, 0), (0, 0), (-1, 1), (0, 1), (-1, 2), (0, 2)),
    ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)),
)


def _leaf_count(node: Node) -> int:
    if node.kind == "leaf":
        return 1
    assert node.upper is not None and node.lower is not None
    return _leaf_count(node.upper) + _leaf_count(node.lower)


def _make_tree(leaves: int) -> Node:
    if leaves == 1:
        return Node(
            kind="leaf",
            length=random.randint(2, 5),
            shape=random.choice(TERMINAL_SHAPES),
        )
    upper_leaves = random.randint(1, leaves - 1)
    return Node(
        kind="branch",
        length=random.randint(2, 4),
        upper=_make_tree(upper_leaves),
        lower=_make_tree(leaves - upper_leaves),
        choose_upper=random.choice([True, False]),
    )


def _assign_leaf_ids(node: Node, next_id: int = 0) -> int:
    if node.kind == "leaf":
        node.leaf_id = next_id
        return next_id + 1
    assert node.upper is not None and node.lower is not None
    next_id = _assign_leaf_ids(node.upper, next_id)
    next_id = _assign_leaf_ids(node.lower, next_id)
    return next_id


def _chosen_leaf(node: Node) -> int:
    if node.kind == "leaf":
        return node.leaf_id
    assert node.upper is not None and node.lower is not None
    return _chosen_leaf(node.upper if node.choose_upper else node.lower)


def _add_line(cells: set[Cell], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            cells.add((r0, c))
        return
    assert c0 == c1
    for r in range(min(r0, r1), max(r0, r1) + 1):
        cells.add((r, c0))


def _center_row(leaf_rows: list[int], lo: int, hi: int) -> int:
    return (leaf_rows[lo] + leaf_rows[hi - 1]) // 2


def _render_node(
    node: Node,
    x0: int,
    lo: int,
    hi: int,
    leaf_rows: list[int],
    pipe_cells: set[Cell],
    markers: set[Cell],
    terminals: dict[int, set[Cell]],
) -> None:
    y = _center_row(leaf_rows, lo, hi)
    x1 = x0 + node.length - 1
    _add_line(pipe_cells, y, x0, y, x1)

    if node.kind == "leaf":
        assert node.shape is not None
        anchor_c = x1 + 1
        terminal = {(y + dr, anchor_c + dc) for dr, dc in node.shape}
        pipe_cells.update(terminal)
        terminals[node.leaf_id] = terminal
        return

    assert node.upper is not None and node.lower is not None
    mid = lo + _leaf_count(node.upper)
    upper_y = _center_row(leaf_rows, lo, mid)
    lower_y = _center_row(leaf_rows, mid, hi)
    assert upper_y < y < lower_y

    _add_line(pipe_cells, upper_y, x1, lower_y, x1)
    markers.add((y - 1, x1) if node.choose_upper else (y + 1, x1))

    _render_node(node.upper, x1 + 1, lo, mid, leaf_rows, pipe_cells, markers, terminals)
    _render_node(node.lower, x1 + 1, mid, hi, leaf_rows, pipe_cells, markers, terminals)


def _make_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    leaves = random.randint(2, 4)
    lane_gap = random.choice([4, 5])
    leaf_rows = [i * lane_gap for i in range(leaves)]
    tree = _make_tree(leaves)
    _assign_leaf_ids(tree)

    pipe_cells: set[Cell] = set()
    markers: set[Cell] = set()
    terminals: dict[int, set[Cell]] = {}
    root = (_center_row(leaf_rows, 0, leaves), 0)

    _render_node(tree, 1, 0, leaves, leaf_rows, pipe_cells, markers, terminals)
    all_cells = pipe_cells | markers | {root}
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    def norm(cell: Cell) -> Cell:
        return (cell[0] - min_r, cell[1] - min_c)

    norm_root = norm(root)
    inp[norm_root] = color
    out[norm_root] = color

    for cell in pipe_cells:
        inp[norm(cell)] = PIPE
        out[norm(cell)] = PIPE

    for cell in markers:
        inp[norm(cell)] = color
        out[norm(cell)] = BG

    for cell in terminals[_chosen_leaf(tree)]:
        out[norm(cell)] = color

    rot = random.randint(0, 3)
    return np.rot90(inp, rot), np.rot90(out, rot)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _place_patches(
    in_patches: list[np.ndarray],
    out_patches: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    dims = [patch.shape for patch in in_patches]
    bbox_area = sum(h * w for h, w in dims)
    base = max(
        max(h for h, _ in dims) + 4,
        max(w for _, w in dims) + 4,
        int(np.ceil(np.sqrt(bbox_area * 1.4))) + 4,
    )
    h = random.randint(base, base + 4)
    w = random.randint(base, base + 4)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = list(range(len(in_patches)))
    random.shuffle(order)
    for idx in order:
        patch = in_patches[idx]
        target = out_patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(200):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if bool(blocked[rr0:rr1, cc0:cc1].any()):
                continue
            _paste(inp, patch, r0, c0)
            _paste(out, target, r0, c0)
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return inp, out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    tree_count = random.randint(1, 3)
    colors = random.sample(SEED_COLORS, tree_count)
    patches = [_make_patch(color) for color in colors]
    inp = [patch for patch, _ in patches]
    out = [target for _, target in patches]
    return _place_patches(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _generate_case()
        if case is None:
            continue
        inp, out = case
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid pipe-tree puzzle")
