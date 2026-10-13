from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
SIDE_STEP = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}

BG_COLOR = 0
MIN_PIECES = 3
MAX_PIECES = 5
MAX_DEGREE = 3
GEN_MAX_ATTEMPTS = 400
INPUT_SIDE_RANGE = (20, 30)
ROOT_SHAPES = ((4, 5), (5, 4), (5, 5))
OTHER_SHAPES = ((3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4))


@dataclass(frozen=True)
class Piece:
    idx: int
    patch: np.ndarray
    bg: int
    body: int
    area: int
    ports: dict[int, tuple[str, tuple[int, int]]]


def _side_from_coord(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = map(int, shape)
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "ports must sit on exactly one bbox side"
    return sides[0]


def _piece_from_patch(idx: int, patch: np.ndarray, bg: int, body: int | None = None) -> Piece:
    g = np.asarray(patch, dtype=int)
    mask = g != int(bg)
    assert mask.any(), "piece patch must contain foreground"
    counts = Counter(map(int, g[mask].tolist()))
    if body is None:
        common = counts.most_common()
        assert len(common) == 1 or common[0][1] > common[1][1], "body color must be the unique majority in every piece"
        body = int(common[0][0])
    else:
        body = int(body)
        assert body in counts, "declared body color missing from piece"

    ports: dict[int, tuple[str, tuple[int, int]]] = {}
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        if color == body:
            continue
        side = _side_from_coord(g.shape, rr, cc)
        assert color not in ports, "each port color must appear at most once per piece"
        ports[color] = (side, (rr, cc))

    assert ports, "each piece must have at least one colored port"
    return Piece(idx=idx, patch=g.copy(), bg=int(bg), body=body, area=int(mask.sum()), ports=ports)


def _flip_piece(piece: Piece, axis: str) -> Piece:
    if axis == "horizontal":
        patch = piece.patch[:, ::-1]
    elif axis == "vertical":
        patch = piece.patch[::-1, :]
    else:
        raise ValueError(f"unknown axis: {axis}")
    return _piece_from_patch(piece.idx, patch, piece.bg, piece.body)


def _place_piece_on_cells(
    cells: dict[tuple[int, int], int],
    piece: Piece,
    top_left: tuple[int, int],
) -> None:
    r0, c0 = map(int, top_left)
    for r, c in np.argwhere(piece.patch != piece.bg):
        rr = r0 + int(r)
        cc = c0 + int(c)
        key = (rr, cc)
        assert key not in cells, "assembled pieces must not overlap"
        cells[key] = int(piece.patch[int(r), int(c)])


def _anchor_index(pieces: list[Piece]) -> int:
    areas = [int(piece.area) for piece in pieces]
    max_area = max(areas)
    winners = [piece.idx for piece in pieces if int(piece.area) == max_area]
    assert len(winners) == 1, "largest piece anchor must be unique"
    return int(winners[0])


def _port_pairs(pieces: list[Piece]) -> tuple[dict[int, list[tuple[int, str, tuple[int, int]]]], dict[int, list[int]]]:
    pairs: dict[int, list[tuple[int, str, tuple[int, int]]]] = {}
    by_piece: dict[int, list[int]] = {piece.idx: [] for piece in pieces}
    for piece in pieces:
        for color, (side, coord) in piece.ports.items():
            pairs.setdefault(int(color), []).append((piece.idx, str(side), tuple(map(int, coord))))
            by_piece[piece.idx].append(int(color))

    for color, endpoints in pairs.items():
        assert len(endpoints) == 2, f"port color {color} must appear exactly twice"
        assert endpoints[0][0] != endpoints[1][0], "a paired port color must belong to two different pieces"
    return pairs, by_piece


def _assert_tree(anchor: int, pieces: list[Piece], port_pairs: dict[int, list[tuple[int, str, tuple[int, int]]]]) -> None:
    graph = {piece.idx: set() for piece in pieces}
    for endpoints in port_pairs.values():
        a = int(endpoints[0][0])
        b = int(endpoints[1][0])
        graph[a].add(b)
        graph[b].add(a)

    assert len(port_pairs) == len(pieces) - 1, "paired ports must form a tree"
    seen = set()
    queue = deque([int(anchor)])
    while queue:
        node = int(queue.popleft())
        if node in seen:
            continue
        seen.add(node)
        queue.extend(sorted(graph[node] - seen))
    assert len(seen) == len(pieces), "port tree must connect every piece to the anchor"


def _assemble_from_pieces(pieces: list[Piece]) -> np.ndarray:
    anchor = _anchor_index(pieces)
    port_pairs, by_piece = _port_pairs(pieces)
    _assert_tree(anchor, pieces, port_pairs)

    base_by_idx = {piece.idx: piece for piece in pieces}
    placed_piece: dict[int, Piece] = {anchor: base_by_idx[anchor]}
    placed_pos: dict[int, tuple[int, int]] = {anchor: (0, 0)}
    cells: dict[tuple[int, int], int] = {}
    _place_piece_on_cells(cells, placed_piece[anchor], placed_pos[anchor])

    queue = deque([anchor])
    while queue:
        parent_idx = int(queue.popleft())
        parent_piece = placed_piece[parent_idx]
        parent_r0, parent_c0 = map(int, placed_pos[parent_idx])

        for color in by_piece[parent_idx]:
            endpoints = port_pairs[int(color)]
            if int(endpoints[0][0]) == parent_idx:
                _src, parent_side, parent_local = endpoints[0]
                child_idx, _child_side, _child_local = endpoints[1]
            else:
                _src, parent_side, parent_local = endpoints[1]
                child_idx, _child_side, _child_local = endpoints[0]
            child_idx = int(child_idx)
            if child_idx in placed_piece:
                continue

            step_r, step_c = SIDE_STEP[str(parent_side)]
            parent_pr = parent_r0 + int(parent_local[0])
            parent_pc = parent_c0 + int(parent_local[1])
            target_child_port = (parent_pr + int(step_r), parent_pc + int(step_c))

            child_piece = base_by_idx[child_idx]
            child_side = str(child_piece.ports[int(color)][0])
            needed_side = OPPOSITE[str(parent_side)]
            if child_side != needed_side:
                if str(parent_side) in {"left", "right"}:
                    assert child_side == str(parent_side), "horizontal ports must be same-side or opposite-side"
                    child_piece = _flip_piece(child_piece, "horizontal")
                else:
                    assert child_side == str(parent_side), "vertical ports must be same-side or opposite-side"
                    child_piece = _flip_piece(child_piece, "vertical")
                assert child_piece.ports[int(color)][0] == needed_side

            child_port = child_piece.ports[int(color)][1]
            child_pos = (
                int(target_child_port[0]) - int(child_port[0]),
                int(target_child_port[1]) - int(child_port[1]),
            )

            _place_piece_on_cells(cells, child_piece, child_pos)
            placed_piece[child_idx] = child_piece
            placed_pos[child_idx] = child_pos
            queue.append(child_idx)

    assert len(placed_piece) == len(pieces), "every piece must be assembled"
    rows = [coord[0] for coord in cells]
    cols = [coord[1] for coord in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), pieces[0].bg, dtype=int)
    for (r, c), color in cells.items():
        out[r - r0, c - c0] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stitch same-body puzzle pieces together by matching same-color ports, mirroring a piece first when its paired port sits on the wrong side.

    Input structure:
    - The grid has a black-dominant background and several separated multicolor pieces.
    - Every piece shares one body color and contains one or more single-cell colored ports on its bbox sides.
    - Each port color appears exactly twice on two different pieces, and those pairings form a tree.

    Rule:
    - Identify the unique largest piece as the anchor and keep its orientation unchanged.
    - Traverse the port-pair tree outward from the anchor; for each paired color, the two ports must become edge-adjacent with opposite facing sides.
    - If the unattached piece shows that paired port on the same side as the already placed port, mirror the whole piece horizontally or vertically, then translate it so the two port pixels touch.

    Key insight:
    - Port colors tell which pieces connect, but the port side itself determines whether translation alone works or whether the whole piece must be mirrored before docking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(comps) >= 2, "expected multiple disconnected pieces"

    pieces = []
    for idx, comp in enumerate(comps):
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, bbox)
        pieces.append(_piece_from_patch(idx, patch, bg))

    body_colors = {piece.body for piece in pieces}
    assert len(body_colors) == 1, "all pieces must share one body color"
    return _assemble_from_pieces(pieces)


def _random_tree(n: int) -> tuple[dict[int, int | None], dict[int, list[int]], list[int]]:
    parent: dict[int, int | None] = {0: None}
    children = {i: [] for i in range(n)}
    degree = [0] * n
    for child in range(1, n):
        choices = [node for node in range(child) if degree[node] < MAX_DEGREE]
        assert choices
        chosen_parent = int(random.choice(choices))
        parent[child] = chosen_parent
        children[chosen_parent].append(child)
        degree[chosen_parent] += 1
        degree[child] += 1
    for node_children in children.values():
        random.shuffle(node_children)
    return parent, children, degree


def _choose_shapes(n: int) -> list[tuple[int, int]]:
    shapes = [tuple(map(int, random.choice(ROOT_SHAPES)))]
    for _ in range(1, n):
        shapes.append(tuple(map(int, random.choice(OTHER_SHAPES))))
    return shapes


def _assign_sides(children: dict[int, list[int]], n: int) -> dict[tuple[int, int], str] | None:
    available = {idx: set(SIDES) for idx in range(n)}
    edge_side: dict[tuple[int, int], str] = {}
    queue = deque([0])
    while queue:
        parent = int(queue.popleft())
        kids = list(children[parent])
        if len(kids) > len(available[parent]):
            return None
        chosen_sides = random.sample(sorted(available[parent]), len(kids))
        for child, side in zip(kids, chosen_sides):
            edge_side[(parent, child)] = str(side)
            available[parent].remove(side)
            child_parent_side = OPPOSITE[str(side)]
            if child_parent_side not in available[child]:
                return None
            available[child].remove(child_parent_side)
            queue.append(child)
    return edge_side


def _side_slot(h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, int(random.randint(1, w - 2))
    if side == "bottom":
        return h - 1, int(random.randint(1, w - 2))
    if side == "left":
        return int(random.randint(1, h - 2)), 0
    if side == "right":
        return int(random.randint(1, h - 2)), w - 1
    raise ValueError(f"unknown side: {side}")


def _build_final_pieces(
    bg: int,
    body: int,
    shapes: list[tuple[int, int]],
    parent: dict[int, int | None],
    children: dict[int, list[int]],
    edge_side: dict[tuple[int, int], str],
    edge_color_by_child: dict[int, int],
) -> dict[int, Piece]:
    ports_by_piece: dict[int, dict[int, str]] = {idx: {} for idx in range(len(shapes))}
    for child, color in edge_color_by_child.items():
        parent_idx = int(parent[child])
        parent_side = str(edge_side[(parent_idx, child)])
        ports_by_piece[parent_idx][int(color)] = parent_side
        ports_by_piece[child][int(color)] = OPPOSITE[parent_side]

    pieces: dict[int, Piece] = {}
    for idx, (h, w) in enumerate(shapes):
        patch = np.full((int(h), int(w)), int(body), dtype=int)
        for color, side in ports_by_piece[idx].items():
            r, c = _side_slot(int(h), int(w), str(side))
            patch[int(r), int(c)] = int(color)
        pieces[idx] = _piece_from_patch(idx, patch, bg, body)

    assert _anchor_index(list(pieces.values())) == 0
    return pieces


def _placement_for_edge(parent_piece: Piece, parent_pos: tuple[int, int], child_piece: Piece, color: int) -> tuple[int, int]:
    parent_side, parent_port = parent_piece.ports[int(color)]
    child_side, child_port = child_piece.ports[int(color)]
    assert child_side == OPPOSITE[parent_side]
    step_r, step_c = SIDE_STEP[str(parent_side)]
    parent_gr = int(parent_pos[0]) + int(parent_port[0])
    parent_gc = int(parent_pos[1]) + int(parent_port[1])
    target = (parent_gr + int(step_r), parent_gc + int(step_c))
    return int(target[0]) - int(child_port[0]), int(target[1]) - int(child_port[1])


def _try_place_final(pieces: dict[int, Piece], parent: dict[int, int | None], children: dict[int, list[int]], edge_color_by_child: dict[int, int]) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], int]] | None:
    positions: dict[int, tuple[int, int]] = {0: (0, 0)}
    cells: dict[tuple[int, int], int] = {}
    _place_piece_on_cells(cells, pieces[0], positions[0])

    queue = deque([0])
    while queue:
        current = int(queue.popleft())
        for child in children[current]:
            color = int(edge_color_by_child[child])
            child_pos = _placement_for_edge(pieces[current], positions[current], pieces[child], color)
            new_cells: dict[tuple[int, int], int] = {}
            for r, c in np.argwhere(pieces[child].patch != pieces[child].bg):
                rr = int(child_pos[0]) + int(r)
                cc = int(child_pos[1]) + int(c)
                key = (rr, cc)
                if key in cells or key in new_cells:
                    return None
                new_cells[key] = int(pieces[child].patch[int(r), int(c)])
            cells.update(new_cells)
            positions[child] = child_pos
            queue.append(child)
    return positions, cells


def _cells_to_grid(cells: dict[tuple[int, int], int], bg: int) -> np.ndarray:
    rows = [coord[0] for coord in cells]
    cols = [coord[1] for coord in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for (r, c), color in cells.items():
        out[int(r) - r0, int(c) - c0] = int(color)
    return out


def _scramble_pieces(
    final_pieces: dict[int, Piece],
    parent: dict[int, int | None],
    edge_side: dict[tuple[int, int], str],
    mirror_by_child: dict[int, bool],
) -> dict[int, Piece]:
    scrambled = {0: final_pieces[0]}
    for child in range(1, len(final_pieces)):
        piece = final_pieces[child]
        if mirror_by_child[child]:
            axis = "horizontal" if edge_side[(int(parent[child]), child)] in {"left", "right"} else "vertical"
            piece = _flip_piece(piece, axis)
        scrambled[child] = piece
    return scrambled


def _scatter_input(pieces: dict[int, Piece], bg: int) -> np.ndarray | None:
    h = int(random.randint(*INPUT_SIDE_RANGE))
    w = int(random.randint(*INPUT_SIDE_RANGE))
    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    order = list(pieces)
    random.shuffle(order)

    for idx in order:
        piece = pieces[idx]
        ph, pw = piece.patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r0 + ph + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c0 + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        paste(grid, piece.patch, int(r0), int(c0))
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= piece.patch != int(bg)
    return grid


def _mirror_plan(n: int) -> dict[int, bool]:
    children = list(range(1, n))
    mirror_count = random.randint(1, len(children) - 1)
    mirrored = set(random.sample(children, mirror_count))
    return {child: child in mirrored for child in children}


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    final_pieces: dict[int, Piece],
    scrambled_pieces: dict[int, Piece],
    mirror_by_child: dict[int, bool],
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(final_pieces) < 3:
        return False

    root_area = int(final_pieces[0].area)
    other_areas = [int(final_pieces[idx].area) for idx in range(1, len(final_pieces))]
    if not all(root_area > area for area in other_areas):
        return False

    if not any(mirror_by_child.values()) or all(mirror_by_child.values()):
        return False

    for child in range(1, len(final_pieces)):
        if mirror_by_child[child]:
            if np.array_equal(scrambled_pieces[child].patch, final_pieces[child].patch):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = int(random.randint(MIN_PIECES, MAX_PIECES))
    parent, children, degree = _random_tree(n)
    if max(degree) > MAX_DEGREE:
        return None

    edge_side = _assign_sides(children, n)
    if edge_side is None:
        return None

    body = int(random.choice([color for color in range(1, 10)]))
    port_palette = [color for color in range(1, 10) if color != body]
    random.shuffle(port_palette)
    edge_color_by_child = {child: int(port_palette[child - 1]) for child in range(1, n)}

    shapes = _choose_shapes(n)
    final_pieces = _build_final_pieces(BG_COLOR, body, shapes, parent, children, edge_side, edge_color_by_child)
    final_placement = _try_place_final(final_pieces, parent, children, edge_color_by_child)
    if final_placement is None:
        return None
    _positions, final_cells = final_placement
    expected = _cells_to_grid(final_cells, BG_COLOR)

    mirror_by_child = _mirror_plan(n)
    scrambled_pieces = _scramble_pieces(final_pieces, parent, edge_side, mirror_by_child)
    inp = _scatter_input(scrambled_pieces, BG_COLOR)
    if inp is None:
        return None

    if not _semantic_checks(inp, expected, final_pieces, scrambled_pieces, mirror_by_child):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-port docking input")
