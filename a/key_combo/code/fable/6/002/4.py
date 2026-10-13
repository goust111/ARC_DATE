from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import random

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_ALL_COLORS = tuple(range(10))
_SLOT_GRIDS = ((1, 1), (1, 2), (2, 1), (2, 2))

_HOLE_SIZE = 2
_GRID_MIN = 20
_GRID_MAX = 30
_MAX_GEN_ATTEMPTS = 600


@dataclass
class ParsedSprite:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    total_holes: int
    pointers: list[tuple[float, float, int]] = field(default_factory=list)


@dataclass
class GenNode:
    idx: int
    children: list[int] = field(default_factory=list)
    orientation: str | None = None
    color: int | None = None
    total_holes: int = 0
    slot_shape: tuple[int, int] | None = None
    hole_slots: list[tuple[int, int]] = field(default_factory=list)
    pointer_slots: list[tuple[int, int]] = field(default_factory=list)
    patch: np.ndarray | None = None


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    """4-neighborhood of (r,c)."""
    for dr, dc in _DIR4:
        yield int(r + dr), int(c + dc)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array, returned as (r0,c0,r1,c1)."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to bbox (inclusive)."""
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    """Paste a patch into dst at top-left (r0,c0)."""
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    dst_view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        dst_view[:, :] = patch
    else:
        mask = patch != int(transparent)
        dst_view[mask] = patch[mask]
    return dst


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask, each as an (N,2) coords array."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _largest_body_per_color(grid: np.ndarray, color: int) -> np.ndarray:
    comps = _mask_components(np.asarray(grid) == int(color))
    assert comps, f"missing component for color {color}"
    comps.sort(key=lambda comp: int(comp.shape[0]), reverse=True)
    assert comps[0].shape[0] >= 4, "actual sprite bodies must dominate pointer pixels"
    for comp in comps[1:]:
        assert comp.shape[0] == 1, "non-body components should be single-pixel pointers"
    return comps[0]


def _interior_holes(body_patch: np.ndarray) -> list[np.ndarray]:
    """Hole components inside a local sprite mask, returned in local coordinates."""
    body_patch = np.asarray(body_patch, dtype=bool)
    h, w = body_patch.shape
    void = ~body_patch
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if void[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if void[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and void[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((int(nr), int(nc)))

    holes = void & (~exterior)
    return _mask_components(holes)


def _parse_input(grid: np.ndarray) -> tuple[int, dict[int, ParsedSprite], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) >= 2, "need at least two sprites so the pointer tree is nontrivial"

    sprites: dict[int, ParsedSprite] = {}
    for color in colors:
        coords = _largest_body_per_color(g, int(color))
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = crop_bbox(g, bbox)
        local_body = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([[r0, c0]], dtype=int)
        local_body[rel[:, 0], rel[:, 1]] = True

        pointers: list[tuple[float, float, int]] = []
        hole_rel_parts = _interior_holes(local_body)
        for hole_rel in hole_rel_parts:
            hole_global = hole_rel + np.array([[r0, c0]], dtype=int)
            colors_inside = []
            for rr, cc in hole_global:
                v = int(g[int(rr), int(cc)])
                if v != bg:
                    colors_inside.append((int(rr), int(cc), int(v)))
            if not colors_inside:
                continue
            hole_colors = {v for _, _, v in colors_inside}
            assert len(colors_inside) == 1 and len(hole_colors) == 1, "pointer holes must contain exactly one colored pixel"
            child_color = int(colors_inside[0][2])
            assert child_color != int(color), "a sprite cannot point to itself"
            center = hole_global.mean(axis=0)
            pointers.append((float(center[0]), float(center[1]), child_color))

        assert len(pointers) <= 2, "each sprite may have at most two pointer holes"
        sprites[int(color)] = ParsedSprite(
            color=int(color),
            coords=coords,
            bbox=bbox,
            total_holes=len(hole_rel_parts),
            pointers=pointers,
        )

    incoming = {color: 0 for color in sprites}
    for sprite in sprites.values():
        seen_children: set[int] = set()
        for _rr, _cc, child in sprite.pointers:
            assert child in sprites, f"pointer to missing sprite color {child}"
            assert child not in seen_children, "duplicate child pointer in one parent"
            seen_children.add(int(child))
            incoming[int(child)] += 1
            assert incoming[int(child)] <= 1, "every sprite must have at most one parent"

    roots = [color for color, deg in incoming.items() if deg == 0]
    assert len(roots) == 1, "pointer graph must have a unique root"
    return bg, sprites, int(roots[0])


def _ordered_pointer_children(sprite: ParsedSprite) -> tuple[list[int], str | None]:
    if not sprite.pointers:
        return [], None
    if len(sprite.pointers) == 1:
        return [int(sprite.pointers[0][2])], None

    a, b = sprite.pointers
    if abs(a[1] - b[1]) >= abs(a[0] - b[0]):
        ordered = sorted(sprite.pointers, key=lambda item: item[1])
        return [int(item[2]) for item in ordered], "h"
    ordered = sorted(sprite.pointers, key=lambda item: item[0])
    return [int(item[2]) for item in ordered], "v"


def _pack_children(children: list[np.ndarray], axis: str, bg: int) -> np.ndarray:
    if axis == "h":
        out_h = max(int(child.shape[0]) for child in children)
        out_w = sum(int(child.shape[1]) for child in children) + len(children) - 1
        out = np.full((out_h, out_w), int(bg), dtype=int)
        c = 0
        for child in children:
            r = (out_h - int(child.shape[0])) // 2
            paste(out, child, r, c, transparent=bg)
            c += int(child.shape[1]) + 1
        return out

    out_h = sum(int(child.shape[0]) for child in children) + len(children) - 1
    out_w = max(int(child.shape[1]) for child in children)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    r = 0
    for child in children:
        c = (out_w - int(child.shape[1])) // 2
        paste(out, child, r, c, transparent=bg)
        r += int(child.shape[0]) + 1
    return out


def _wrap_frame(color: int, inner: np.ndarray, bg: int) -> np.ndarray:
    out = np.full((inner.shape[0] + 4, inner.shape[1] + 4), int(bg), dtype=int)
    out[0, :] = int(color)
    out[-1, :] = int(color)
    out[:, 0] = int(color)
    out[:, -1] = int(color)
    paste(out, inner, 2, 2, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a canonical nested frame diagram from color pointers hidden inside sprite holes.

    Input structure:
    - A dominant background color surrounds several pairwise-disjoint monochrome holey sprites, each with its own color.
    - For each sprite color, the largest 4-connected component of that color is the actual sprite body; any smaller component of that color is just a single pointer pixel inside another sprite's hole.
    - Some enclosed holes are empty, while some holes contain exactly one non-background pixel whose color matches a different sprite; each sprite has at most two such pointer holes.

    Rule:
    - Detect every sprite body, find its enclosed hole regions, and read a hole's lone colored pixel as a direct pointer to the child sprite of that color.
    - The unique sprite with no incoming pointer is the outer root. A sprite with no outgoing pointers becomes one solid pixel in the output.
    - A sprite with one or two outgoing pointers becomes a one-pixel-thick rectangular frame in its own color, with its rendered direct children placed inside and separated from the frame by a one-cell background ring.
    - If a parent has two children, place them left-right or top-bottom according to the relative positions of their pointer holes in the parent sprite, with one background cell between siblings, then crop to the outermost frame.

    Key insight: Hole count is a decoy; the real containment tree and sibling arrangement come only from which child color is written inside which hole and where those pointer holes sit.
    """
    bg, sprites, root_color = _parse_input(np.asarray(grid, dtype=int))
    memo: dict[int, np.ndarray] = {}

    def render(color: int) -> np.ndarray:
        if color in memo:
            return memo[color]
        sprite = sprites[int(color)]
        ordered_children, axis = _ordered_pointer_children(sprite)
        if not ordered_children:
            patch = np.array([[int(color)]], dtype=int)
        elif len(ordered_children) == 1:
            patch = _wrap_frame(int(color), render(int(ordered_children[0])), bg)
        else:
            group = _pack_children([render(int(child)) for child in ordered_children], str(axis), bg)
            patch = _wrap_frame(int(color), group, bg)
        memo[int(color)] = patch
        return patch

    return render(int(root_color))


def _shape_size(nodes: dict[int, GenNode], idx: int, memo: dict[int, tuple[int, int]]) -> tuple[int, int]:
    if idx in memo:
        return memo[idx]
    node = nodes[int(idx)]
    if not node.children:
        memo[int(idx)] = (1, 1)
        return memo[int(idx)]
    if len(node.children) == 1:
        ch, cw = _shape_size(nodes, int(node.children[0]), memo)
        memo[int(idx)] = (ch + 4, cw + 4)
        return memo[int(idx)]
    a_h, a_w = _shape_size(nodes, int(node.children[0]), memo)
    b_h, b_w = _shape_size(nodes, int(node.children[1]), memo)
    if node.orientation == "h":
        memo[int(idx)] = (max(a_h, b_h) + 4, a_w + b_w + 1 + 4)
    else:
        memo[int(idx)] = (a_h + b_h + 1 + 4, max(a_w, b_w) + 4)
    return memo[int(idx)]


def _build_tree() -> tuple[dict[int, GenNode], int]:
    kind = random.choices(
        ["root_two", "root_one_then_two", "root_two_plus_chain", "root_two_plus_two"],
        weights=[0.22, 0.24, 0.34, 0.20],
        k=1,
    )[0]

    node_count = {"root_two": 3, "root_one_then_two": 4, "root_two_plus_chain": 4, "root_two_plus_two": 5}[kind]
    nodes = {idx: GenNode(idx=idx) for idx in range(node_count)}
    root = 0

    if kind == "root_two":
        nodes[0].children = [1, 2]
    elif kind == "root_one_then_two":
        nodes[0].children = [1]
        nodes[1].children = [2, 3]
    elif kind == "root_two_plus_chain":
        nodes[0].children = [1, 2]
        internal = random.choice([1, 2])
        nodes[internal].children = [3]
    else:
        nodes[0].children = [1, 2]
        internal = random.choice([1, 2])
        nodes[internal].children = [3, 4]

    two_child_nodes = [idx for idx, node in nodes.items() if len(node.children) == 2]
    if len(two_child_nodes) == 1:
        nodes[two_child_nodes[0]].orientation = random.choice(["h", "v"])
    elif len(two_child_nodes) == 2:
        first = random.choice(["h", "v"])
        second = "v" if first == "h" else "h"
        nodes[two_child_nodes[0]].orientation = first
        nodes[two_child_nodes[1]].orientation = second

    size_memo: dict[int, tuple[int, int]] = {}
    for idx in sorted(nodes):
        _shape_size(nodes, idx, size_memo)

    focus_asym = None
    for idx in two_child_nodes:
        a, b = nodes[idx].children
        a_area = int(np.prod(size_memo[a]))
        b_area = int(np.prod(size_memo[b]))
        if a_area != b_area:
            focus_asym = idx
            break

    for idx in two_child_nodes:
        children = list(nodes[idx].children)
        if idx == focus_asym:
            a, b = children
            a_area = int(np.prod(size_memo[a]))
            b_area = int(np.prod(size_memo[b]))
            if a_area < b_area:
                children = [b, a]
        elif random.random() < 0.5:
            children.reverse()
        nodes[idx].children = children

    return nodes, root


def _assign_colors(nodes: dict[int, GenNode], bg: int) -> None:
    palette = random.sample([c for c in _ALL_COLORS if c != int(bg)], len(nodes))
    for idx, color in zip(sorted(nodes), palette):
        nodes[idx].color = int(color)

    focus = next((idx for idx in sorted(nodes) if len(nodes[idx].children) == 2), None)
    if focus is not None:
        first, second = nodes[focus].children
        if int(nodes[first].color) < int(nodes[second].color):
            nodes[first].color, nodes[second].color = nodes[second].color, nodes[first].color


def _assign_hole_counts(nodes: dict[int, GenNode]) -> None:
    leaves = [node for node in nodes.values() if not node.children]
    internals = [node for node in nodes.values() if node.children]
    for node in nodes.values():
        child_count = len(node.children)
        if child_count == 0:
            node.total_holes = random.choices([1, 2, 3, 4], weights=[0.08, 0.22, 0.38, 0.32], k=1)[0]
        elif child_count == 1:
            node.total_holes = 1 + random.choices([0, 1, 2], weights=[0.30, 0.45, 0.25], k=1)[0]
        else:
            node.total_holes = 2 + random.choices([0, 1, 2], weights=[0.42, 0.38, 0.20], k=1)[0]

    if leaves and internals:
        decoy_leaf = max(leaves, key=lambda node: (node.total_holes, -node.idx))
        smallest_parent = min(internals, key=lambda node: (len(node.children), node.total_holes, node.idx))
        smallest_parent.total_holes = len(smallest_parent.children)
        decoy_leaf.total_holes = max(decoy_leaf.total_holes, len(smallest_parent.children) + 2)
        decoy_leaf.total_holes = min(decoy_leaf.total_holes, 4)
        assert decoy_leaf.total_holes > smallest_parent.total_holes


def _choose_layout(node: GenNode) -> None:
    child_count = len(node.children)
    feasible = []
    for rows, cols in _SLOT_GRIDS:
        if rows * cols < int(node.total_holes):
            continue
        if child_count == 2 and node.orientation == "h" and cols < 2:
            continue
        if child_count == 2 and node.orientation == "v" and rows < 2:
            continue
        feasible.append((rows, cols))
    assert feasible, "no feasible slot grid for requested hole count"

    area_min = min(rows * cols for rows, cols in feasible)
    compact = [shape for shape in feasible if shape[0] * shape[1] == area_min]
    node.slot_shape = random.choice(compact)
    rows, cols = node.slot_shape
    all_slots = [(r, c) for r in range(rows) for c in range(cols)]

    pointer_slots: list[tuple[int, int]] = []
    if child_count == 2:
        if node.orientation == "h":
            row = random.randrange(rows)
            pointer_slots = [(row, 0), (row, cols - 1)]
        else:
            col = random.randrange(cols)
            pointer_slots = [(0, col), (rows - 1, col)]
    elif child_count == 1:
        pointer_slots = [random.choice(all_slots)]

    remaining = [slot for slot in all_slots if slot not in pointer_slots]
    extra_needed = int(node.total_holes) - child_count
    extras = random.sample(remaining, extra_needed)
    node.pointer_slots = pointer_slots
    node.hole_slots = pointer_slots + extras


def _slot_origin(slot: tuple[int, int]) -> tuple[int, int]:
    sr, sc = map(int, slot)
    return 1 + sr * (_HOLE_SIZE + 1), 1 + sc * (_HOLE_SIZE + 1)


def _build_patch(node: GenNode, nodes: dict[int, GenNode], bg: int) -> None:
    assert node.color is not None and node.slot_shape is not None
    rows, cols = node.slot_shape
    h = rows * (_HOLE_SIZE + 1) + 1
    w = cols * (_HOLE_SIZE + 1) + 1
    patch = np.full((h, w), int(node.color), dtype=int)

    for slot in node.hole_slots:
        r0, c0 = _slot_origin(slot)
        patch[r0 : r0 + _HOLE_SIZE, c0 : c0 + _HOLE_SIZE] = int(bg)

    for child_idx, slot in zip(node.children, node.pointer_slots):
        r0, c0 = _slot_origin(slot)
        rr = r0 + random.randrange(_HOLE_SIZE)
        cc = c0 + random.randrange(_HOLE_SIZE)
        patch[rr, cc] = int(nodes[int(child_idx)].color)

    node.patch = patch


def _render_spec(nodes: dict[int, GenNode], idx: int, bg: int) -> np.ndarray:
    node = nodes[int(idx)]
    assert node.color is not None
    if not node.children:
        return np.array([[int(node.color)]], dtype=int)
    if len(node.children) == 1:
        return _wrap_frame(int(node.color), _render_spec(nodes, int(node.children[0]), bg), bg)
    group = _pack_children([_render_spec(nodes, int(ch), bg) for ch in node.children], str(node.orientation), bg)
    return _wrap_frame(int(node.color), group, bg)


def _scatter_patches(nodes: dict[int, GenNode], bg: int) -> np.ndarray | None:
    patches = [node.patch for node in nodes.values()]
    assert all(patch is not None for patch in patches)
    dims = [patch.shape for patch in patches if patch is not None]
    area_need = sum((int(h) + 2) * (int(w) + 2) for h, w in dims)
    min_side = max(
        _GRID_MIN,
        max(int(h) for h, _ in dims) + 4,
        max(int(w) for _, w in dims) + 4,
        int(np.ceil(np.sqrt(area_need))) + 2,
    )
    if min_side > _GRID_MAX:
        return None

    h = random.randint(min_side, min(_GRID_MAX, min_side + 4))
    w = random.randint(min_side, min(_GRID_MAX, min_side + 4))
    grid = np.full((h, w), int(bg), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = sorted(nodes.values(), key=lambda node: int(node.patch.shape[0] * node.patch.shape[1]), reverse=True)
    for node in order:
        patch = node.patch
        assert patch is not None
        ph, pw = map(int, patch.shape)
        candidates = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                rr0 = r0 - 1
                cc0 = c0 - 1
                rr1 = r0 + ph + 1
                cc1 = c0 + pw + 1
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        paste(grid, patch, r0, c0, transparent=bg)
        reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True

    return grid


def _semantic_ok(grid: np.ndarray, nodes: dict[int, GenNode], root: int, bg: int, expected: np.ndarray) -> bool:
    if int(get_bg(grid)) != int(bg):
        return False
    if np.array_equal(grid, expected):
        return False

    try:
        parsed_bg, sprites, root_color = _parse_input(grid)
    except Exception:
        return False
    if parsed_bg != int(bg):
        return False
    if root_color != int(nodes[root].color):
        return False

    color_to_idx = {int(node.color): idx for idx, node in nodes.items()}
    if set(color_to_idx) != set(sprites):
        return False

    for color, sprite in sprites.items():
        node = nodes[color_to_idx[int(color)]]
        if sprite.total_holes != int(node.total_holes):
            return False
        ordered_children, axis = _ordered_pointer_children(sprite)
        expected_children = [int(nodes[ch].color) for ch in node.children]
        if ordered_children != expected_children:
            return False
        if len(expected_children) <= 1 and axis is not None:
            return False
        if len(expected_children) == 2 and axis != str(node.orientation):
            return False

    if max(expected.shape) > 30:
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    leaves = [node for node in nodes.values() if not node.children]
    internals = [node for node in nodes.values() if node.children]
    if not leaves or not internals:
        return False
    if max(node.total_holes for node in leaves) <= min(node.total_holes for node in internals):
        return False

    two_child_nodes = [node for node in nodes.values() if len(node.children) == 2]
    if not two_child_nodes:
        return False
    if not any(
        [int(nodes[ch].color) for ch in node.children] != sorted(int(nodes[ch].color) for ch in node.children)
        for node in two_child_nodes
    ):
        return False

    render_areas = {idx: int(np.prod(_render_spec(nodes, idx, bg).shape)) for idx in nodes}
    asym_nodes = [
        node
        for node in two_child_nodes
        if render_areas[node.children[0]] != render_areas[node.children[1]]
    ]
    if asym_nodes and not any(
        node.children != sorted(node.children, key=lambda child: render_areas[child])
        for node in asym_nodes
    ):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_ALL_COLORS)
    nodes, root = _build_tree()
    _assign_colors(nodes, bg)
    _assign_hole_counts(nodes)
    for node in nodes.values():
        _choose_layout(node)
    for node in nodes.values():
        _build_patch(node, nodes, bg)

    expected = _render_spec(nodes, root, bg)
    if max(expected.shape) > 30 or len(nodes) < 3:
        return None

    grid = _scatter_patches(nodes, bg)
    if grid is None:
        return None
    if not _semantic_ok(grid, nodes, root, bg, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-hole-pointer ARC instance")
