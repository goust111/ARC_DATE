from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
LEAF_COLOR = 9

ROOT_COLORS = (BLUE, RED, YELLOW)
MARKER_COLORS = (ORANGE, AZURE)
COLOR_RING = (RED, ORANGE, YELLOW, GREEN, AZURE, BLUE, MAGENTA)
RING_INDEX = {color: idx for idx, color in enumerate(COLOR_RING)}

MAX_INTERNAL_DEPTH = 4
TRUNK_LEN_RANGE = (2, 4)
TOP_PAD_RANGE = (1, 4)
SIDE_PAD_RANGE = (1, 4)
BOTTOM_PAD_RANGE = (1, 3)
GEN_RETRIES = 400


@dataclass
class ParsedLeaf:
    bbox: tuple[int, int, int, int]


@dataclass
class ParsedNode:
    pos: tuple[int, int]
    marker: int
    left: ParsedNode | ParsedLeaf
    right: ParsedNode | ParsedLeaf


@dataclass
class DesignLeaf:
    endpoint: tuple[int, int]
    bbox: tuple[int, int, int, int]


@dataclass
class DesignNode:
    marker: int
    left: DesignNode | DesignLeaf
    right: DesignNode | DesignLeaf


def _is_uniform_block(g: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 1 >= h or c0 + 1 >= w:
        return False
    return bool(np.all(g[r0 : r0 + 2, c0 : c0 + 2] == color))


def _find_blocks(g: np.ndarray, colors: set[int]) -> list[tuple[int, int, int, int, int]]:
    blocks: list[tuple[int, int, int, int, int]] = []
    h, w = g.shape
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(g[r, c])
            if color not in colors:
                continue
            if np.all(g[r : r + 2, c : c + 2] == color):
                blocks.append((r, c, r + 1, c + 1, color))
    return blocks


def _color_shift(color: int, delta: int) -> int:
    return COLOR_RING[(RING_INDEX[int(color)] + delta) % len(COLOR_RING)]


def _candidate_leaf_bbox(g: np.ndarray, endpoint: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = endpoint
    candidates = []
    for dr in (0, 1):
        for dc in (0, 1):
            r0 = r - dr
            c0 = c - dc
            if _is_uniform_block(g, r0, c0, LEAF_COLOR):
                candidates.append((r0, c0, r0 + 1, c0 + 1))
    assert len(candidates) == 1, f"expected one leaf block containing {endpoint}, got {candidates}"
    return candidates[0]


def _follow_vertical_to_marker(g: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    assert r >= 0
    while r >= 0 and int(g[r, c]) == GRAY:
        r -= 1
    assert r >= 0 and int(g[r, c]) in MARKER_COLORS, "trunk must end at a marker"
    return r, c


def _follow_branch(
    g: np.ndarray,
    marker_pos: tuple[int, int],
    direction: int,
) -> tuple[tuple[int, int], str]:
    r, c = marker_pos
    k = 1
    saw_gray = False
    h, w = g.shape
    while True:
        rr = r - k
        cc = c + direction * k
        assert 0 <= rr < h and 0 <= cc < w, "branch left the grid"
        value = int(g[rr, cc])
        if value == GRAY:
            saw_gray = True
            k += 1
            continue
        assert saw_gray, "branch must contain at least one gray segment"
        if value in MARKER_COLORS:
            return (rr, cc), "node"
        assert value == LEAF_COLOR, f"branch must end at a marker or leaf block, got {value}"
        return (rr, cc), "leaf"


def _parse_node(
    g: np.ndarray,
    pos: tuple[int, int],
    seen_nodes: set[tuple[int, int]],
    seen_leaves: set[tuple[int, int, int, int]],
) -> ParsedNode:
    assert pos not in seen_nodes, "tree must be acyclic"
    seen_nodes.add(pos)
    marker = int(g[pos])
    assert marker in MARKER_COLORS

    children: list[ParsedNode | ParsedLeaf] = []
    for direction in (-1, 1):
        endpoint, kind = _follow_branch(g, pos, direction)
        if kind == "node":
            children.append(_parse_node(g, endpoint, seen_nodes, seen_leaves))
        else:
            bbox = _candidate_leaf_bbox(g, endpoint)
            seen_leaves.add(bbox)
            children.append(ParsedLeaf(bbox=bbox))

    return ParsedNode(pos=pos, marker=marker, left=children[0], right=children[1])


def _paint_leaf(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _propagate(node: ParsedNode, incoming: int, out: np.ndarray) -> None:
    if node.marker == ORANGE:
        left_color = _color_shift(incoming, 1)
        right_color = _color_shift(incoming, 2)
    else:
        assert node.marker == AZURE
        left_color = _color_shift(incoming, -1)
        right_color = _color_shift(incoming, -2)

    for child, color in ((node.left, left_color), (node.right, right_color)):
        if isinstance(child, ParsedLeaf):
            _paint_leaf(out, child.bbox, color)
        else:
            _propagate(child, color, out)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    root_blocks = _find_blocks(g, set(ROOT_COLORS))
    assert len(root_blocks) == 1, f"expected one root block, got {root_blocks}"
    root_r0, root_c0, _root_r1, _root_c1, root_color = root_blocks[0]

    trunk_starts = []
    for c in (root_c0, root_c0 + 1):
        rr = root_r0 - 1
        if rr >= 0 and int(g[rr, c]) in (GRAY,) + MARKER_COLORS:
            trunk_starts.append((rr, c))
    assert len(trunk_starts) == 1, f"expected one trunk attachment, got {trunk_starts}"
    first_marker = _follow_vertical_to_marker(g, trunk_starts[0])

    seen_nodes: set[tuple[int, int]] = set()
    seen_leaves: set[tuple[int, int, int, int]] = set()
    tree = _parse_node(g, first_marker, seen_nodes, seen_leaves)

    all_markers = {
        tuple(map(int, coord))
        for coord in np.argwhere(np.isin(g, MARKER_COLORS))
    }
    assert seen_nodes == all_markers, "all marker pixels must belong to the tree"

    all_leaves = {(r0, c0, r1, c1) for r0, c0, r1, c1, _color in _find_blocks(g, {LEAF_COLOR})}
    assert seen_leaves == all_leaves, "all leaf blocks must be reachable from the root"

    out = g.copy()
    _propagate(tree, int(root_color), out)
    return out


def _sample_shifts(depth: int) -> list[int]:
    shifts = [random.randint(2, 3)]
    while len(shifts) < depth:
        shifts.insert(0, sum(shifts) + 2)
    return shifts


def _build_design(depth: int, forced_path: tuple[int, ...] | None) -> DesignNode:
    marker = random.choice(MARKER_COLORS)
    children: list[DesignNode | DesignLeaf] = []
    next_level_is_leaf = depth == 1
    forced_side = forced_path[0] if forced_path else None
    next_forced = forced_path[1:] if forced_path and len(forced_path) > 1 else ()
    for side_index in (0, 1):
        if next_level_is_leaf:
            children.append(DesignLeaf(endpoint=(-1, -1), bbox=(-1, -1, -1, -1)))
            continue

        must_continue = forced_side == side_index
        continue_down = must_continue or bool(random.getrandbits(1))
        if continue_down:
            child_forced = next_forced if must_continue else None
            children.append(_build_design(depth - 1, child_forced))
        else:
            children.append(DesignLeaf(endpoint=(-1, -1), bbox=(-1, -1, -1, -1)))

    return DesignNode(marker=marker, left=children[0], right=children[1])


def _assign_layout(
    node: DesignNode,
    level: int,
    pos: tuple[int, int],
    shifts: list[int],
    leaves: list[DesignLeaf],
    markers: list[tuple[int, int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    r, c = pos
    markers.append((r, c, node.marker))
    shift = shifts[level]
    for direction, side_name in ((-1, "left"), (1, "right")):
        child = getattr(node, side_name)
        child_pos = (r - shift, c + direction * shift)
        edges.append((pos, child_pos))
        if isinstance(child, DesignNode):
            _assign_layout(child, level + 1, child_pos, shifts, leaves, markers, edges)
        else:
            end_r, end_c = child_pos
            if direction == -1:
                bbox = (end_r - 1, end_c - 1, end_r, end_c)
            else:
                bbox = (end_r - 1, end_c, end_r, end_c + 1)
            child.endpoint = child_pos
            child.bbox = bbox
            leaves.append(child)


def _render_design(
    root_color: int,
    root_attach_side: str,
    trunk_len: int,
    root: DesignNode,
    shifts: list[int],
) -> tuple[np.ndarray, dict[tuple[int, int, int, int], int]]:
    leaves: list[DesignLeaf] = []
    markers: list[tuple[int, int, int]] = []
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    _assign_layout(root, 0, (0, 0), shifts, leaves, markers, edges)

    if root_attach_side == "left":
        root_bbox = (trunk_len + 1, 0, trunk_len + 2, 1)
        trunk_col = 0
    else:
        root_bbox = (trunk_len + 1, -1, trunk_len + 2, 0)
        trunk_col = 0

    occupied: list[tuple[int, int]] = []
    occupied.extend((r, c) for r, c, _color in markers)
    for (r0, c0, r1, c1) in [root_bbox] + [leaf.bbox for leaf in leaves]:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                occupied.append((r, c))
    for (r0, c0), (r1, c1) in edges:
        step_c = 1 if c1 > c0 else -1
        for k in range(1, abs(c1 - c0)):
            occupied.append((r0 - k, c0 + step_c * k))
    for r in range(1, trunk_len + 1):
        occupied.append((r, trunk_col))

    rows = [r for r, _c in occupied]
    cols = [c for _r, c in occupied]
    top_pad = random.randint(*TOP_PAD_RANGE)
    left_pad = random.randint(*SIDE_PAD_RANGE)
    right_pad = random.randint(*SIDE_PAD_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    row_shift = top_pad - min(rows)
    col_shift = left_pad - min(cols)

    height = max(rows) - min(rows) + 1 + top_pad + bottom_pad
    width = max(cols) - min(cols) + 1 + left_pad + right_pad
    grid = np.full((height, width), BG, dtype=int)

    rr0, rc0, rr1, rc1 = root_bbox
    grid[rr0 + row_shift : rr1 + 1 + row_shift, rc0 + col_shift : rc1 + 1 + col_shift] = root_color

    for r in range(1, trunk_len + 1):
        grid[r + row_shift, trunk_col + col_shift] = GRAY

    for r, c, color in markers:
        grid[r + row_shift, c + col_shift] = color

    for (r0, c0), (r1, c1) in edges:
        step_c = 1 if c1 > c0 else -1
        for k in range(1, abs(c1 - c0)):
            rr = r0 - k + row_shift
            cc = c0 + step_c * k + col_shift
            assert int(grid[rr, cc]) == BG
            grid[rr, cc] = GRAY

    leaf_bboxes: dict[tuple[int, int, int, int], int] = {}
    for leaf in leaves:
        r0, c0, r1, c1 = leaf.bbox
        sr0, sc0, sr1, sc1 = r0 + row_shift, c0 + col_shift, r1 + row_shift, c1 + col_shift
        assert np.all(grid[sr0 : sr1 + 1, sc0 : sc1 + 1] == BG)
        grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = LEAF_COLOR
        leaf.bbox = (sr0, sc0, sr1, sc1)
        leaf_bboxes[leaf.bbox] = LEAF_COLOR

    return grid, leaf_bboxes


def _expected_leaf_colors(node: DesignNode, incoming: int, colors: dict[tuple[int, int, int, int], int]) -> None:
    if node.marker == ORANGE:
        left_color = _color_shift(incoming, 1)
        right_color = _color_shift(incoming, 2)
    else:
        left_color = _color_shift(incoming, -1)
        right_color = _color_shift(incoming, -2)

    for child, color in ((node.left, left_color), (node.right, right_color)):
        if isinstance(child, DesignLeaf):
            colors[child.bbox] = color
        else:
            _expected_leaf_colors(child, color, colors)


def _paint_expected(inp: np.ndarray, root: DesignNode, root_color: int, row_shifted_leaf_bboxes: dict[tuple[int, int, int, int], int]) -> np.ndarray:
    out = inp.copy()
    target_colors: dict[tuple[int, int, int, int], int] = {}
    _expected_leaf_colors(root, root_color, target_colors)
    assert set(target_colors) == set(row_shifted_leaf_bboxes)
    for bbox, color in target_colors.items():
        _paint_leaf(out, bbox, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    depth = random.randint(1, MAX_INTERNAL_DEPTH)
    forced_path = tuple(random.randint(0, 1) for _ in range(max(depth - 1, 0)))
    design = _build_design(depth, forced_path)
    shifts = _sample_shifts(depth)
    root_color = random.choice(ROOT_COLORS)
    trunk_len = random.randint(*TRUNK_LEN_RANGE)
    root_attach_side = random.choice(("left", "right"))

    try:
        inp, leaf_bboxes = _render_design(root_color, root_attach_side, trunk_len, design, shifts)
        expected = _paint_expected(inp, design, root_color, leaf_bboxes)
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid tree-color input")
