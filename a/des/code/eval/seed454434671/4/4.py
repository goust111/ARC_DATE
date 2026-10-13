from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DEPTH_COLORS = (RED, BLUE, GREEN, YELLOW)
INPUT_FRAME_COLORS = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]


@dataclass
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    depth: int = 0
    anchored: bool = False


@dataclass
class Node:
    children: list["Node"] = field(default_factory=list)
    anchor: bool = False
    layout: str = "h"
    size: tuple[int, int] = (0, 0)
    child_offsets: list[tuple[int, int]] = field(default_factory=list)
    depth: int = 0
    color: int = 0
    bbox: tuple[int, int, int, int] | None = None
    anchor_rel: tuple[int, int, int, int] | None = None


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask_from_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _is_rect_frame(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    r0, c0, r1, c1 = _bbox(mask)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    want = np.zeros((h, w), dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return bool(np.array_equal(sub, want))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _extract_frames(grid: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, GREY)):
        mask = grid == color
        assert _is_rect_frame(mask), f"color {color} is not a rectangular frame"
        frames.append(Frame(color=color, bbox=_bbox(mask)))

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            a = frames[i].bbox
            b = frames[j].bbox
            if _overlap(a, b):
                assert _contains(a, b) or _contains(b, a), "frames partially overlap"

    for i, frame in enumerate(frames):
        parents = [j for j, other in enumerate(frames) if _contains(other.bbox, frame.bbox)]
        if parents:
            parent = min(parents, key=lambda j: _area(frames[j].bbox))
            frame.parent = parent
            frames[parent].children.append(i)

    def set_depth(idx: int, depth: int) -> None:
        frames[idx].depth = depth
        for child in frames[idx].children:
            set_depth(child, depth + 1)

    roots = [i for i, frame in enumerate(frames) if frame.parent is None]
    for idx in roots:
        set_depth(idx, 0)

    for frame in frames:
        if frame.children:
            continue
        r0, c0, r1, c1 = frame.bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        frame.anchored = bool(np.any(inner == GREY))

    return frames


def _area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular frames by containment depth and shrink anchored leaf frames inward.

    Input structure:
    - Black background.
    - Several one-pixel-thick rectangular frames of distinct colors.
    - Frames are either disjoint or strictly nested, forming a forest of containment trees.
    - Some leaf frames contain a small solid grey rectangle anchor.

    Rule:
    - Compute each frame's depth in the containment tree.
    - Recolor depth 0/1/2/3 frames to red/blue/green/yellow, cycling if deeper levels appear.
    - If a leaf frame contains a grey anchor, move each side of that frame inward by one cell.
    - Keep grey anchor pixels fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    frames = _extract_frames(g)
    out = np.full_like(g, BG)

    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        bbox = frame.bbox
        if not frame.children and frame.anchored:
            assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5, "anchored leaf is too small to shrink"
            bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        color = DEPTH_COLORS[frame.depth % len(DEPTH_COLORS)]
        _draw_frame(out, bbox, color)

    out[g == GREY] = GREY
    return out


# --------------- generator ---------------

MIN_GRID = 18
MAX_GRID = 30
MAX_FRAMES = 7
MAX_TREE_RETRIES = 400
MAX_LAYOUT_RETRIES = 120


def _all_nodes(roots: list[Node]) -> list[Node]:
    out: list[Node] = []

    def visit(node: Node) -> None:
        out.append(node)
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return out


def _leaf_nodes(roots: list[Node]) -> list[Node]:
    return [node for node in _all_nodes(roots) if not node.children]


def _make_tree(rng: random.Random, depth: int, max_depth: int) -> Node:
    if depth >= max_depth or (depth >= 1 and rng.random() < 0.4):
        return Node()

    if depth == 0:
        child_count = rng.choice([1, 2, 2, 3])
    else:
        child_count = rng.choice([1, 1, 2])

    children = [_make_tree(rng, depth + 1, max_depth) for _ in range(child_count)]
    return Node(children=children, layout=rng.choice(["h", "v"]))


def _generate_forest(rng: random.Random) -> list[Node]:
    for _ in range(MAX_TREE_RETRIES):
        root_count = 1 if rng.random() < 0.7 else 2
        max_depth = rng.choice([2, 3])
        roots = [_make_tree(rng, 0, max_depth) for _ in range(root_count)]
        nodes = _all_nodes(roots)
        leaves = _leaf_nodes(roots)
        if not (3 <= len(nodes) <= MAX_FRAMES and len(leaves) >= 2):
            continue

        anchored_count = rng.randint(1, len(leaves) - 1)
        rng.shuffle(leaves)
        for idx, leaf in enumerate(leaves):
            leaf.anchor = idx < anchored_count
        return roots
    raise RuntimeError("failed to build a valid forest")


def _choose_anchor_rel(rng: random.Random, h: int, w: int) -> tuple[int, int, int, int]:
    max_ah = min(2, h - 4)
    max_aw = min(2, w - 4)
    ah = rng.randint(1, max_ah)
    aw = rng.randint(1, max_aw)
    ar0 = rng.randint(2, h - 2 - ah)
    ac0 = rng.randint(2, w - 2 - aw)
    return ar0, ac0, ar0 + ah - 1, ac0 + aw - 1


def _size_tree(node: Node, rng: random.Random) -> None:
    for child in node.children:
        _size_tree(child, rng)

    if not node.children:
        if node.anchor:
            h = rng.randint(7, 10)
            w = rng.randint(7, 10)
            node.anchor_rel = _choose_anchor_rel(rng, h, w)
        else:
            h = rng.randint(5, 9)
            w = rng.randint(5, 9)
        node.size = (h, w)
        node.child_offsets = []
        return

    node.layout = rng.choice(["h", "v"])
    child_sizes = [child.size for child in node.children]
    gaps = [rng.randint(1, 3) for _ in range(max(0, len(node.children) - 1))]
    offsets: list[tuple[int, int]] = []

    if node.layout == "h":
        track_h = max(h for h, _ in child_sizes) + rng.randint(0, 2)
        cursor = 2
        for idx, child in enumerate(node.children):
            ch, cw = child.size
            top = 2 + rng.randint(0, track_h - ch)
            offsets.append((top, cursor))
            cursor += cw
            if idx < len(gaps):
                cursor += gaps[idx]
        node.size = (track_h + 4, cursor + 2)
    else:
        track_w = max(w for _, w in child_sizes) + rng.randint(0, 2)
        cursor = 2
        for idx, child in enumerate(node.children):
            ch, cw = child.size
            left = 2 + rng.randint(0, track_w - cw)
            offsets.append((cursor, left))
            cursor += ch
            if idx < len(gaps):
                cursor += gaps[idx]
        node.size = (cursor + 2, track_w + 4)

    node.child_offsets = offsets


def _set_depths(node: Node, depth: int) -> None:
    node.depth = depth
    for child in node.children:
        _set_depths(child, depth + 1)


def _assign_colors(roots: list[Node], rng: random.Random) -> None:
    nodes = _all_nodes(roots)
    ordered = sorted(nodes, key=lambda node: (node.depth, len(node.children)), reverse=True)

    def backtrack(i: int, available: list[int]) -> bool:
        if i == len(ordered):
            return True
        node = ordered[i]
        bad = DEPTH_COLORS[node.depth % len(DEPTH_COLORS)]
        choices = [color for color in available if color != bad]
        rng.shuffle(choices)
        for color in choices:
            node.color = color
            nxt = [c for c in available if c != color]
            if backtrack(i + 1, nxt):
                return True
        return False

    pool = INPUT_FRAME_COLORS[:]
    if not backtrack(0, pool):
        raise RuntimeError("failed to assign distinct frame colors")


def _place_subtree(node: Node, top: int, left: int) -> None:
    h, w = node.size
    node.bbox = (top, left, top + h - 1, left + w - 1)
    for child, (dr, dc) in zip(node.children, node.child_offsets):
        _place_subtree(child, top + dr, left + dc)


def _place_roots(roots: list[Node], rng: random.Random) -> tuple[int, int]:
    sizes = [root.size for root in roots]

    for _ in range(MAX_LAYOUT_RETRIES):
        if len(roots) == 1:
            req_h, req_w = sizes[0]
            if req_h > MAX_GRID - 2 or req_w > MAX_GRID - 2:
                continue
            grid_h = rng.randint(max(MIN_GRID, req_h + 2), MAX_GRID)
            grid_w = rng.randint(max(MIN_GRID, req_w + 2), MAX_GRID)
            top = rng.randint(1, grid_h - req_h - 1)
            left = rng.randint(1, grid_w - req_w - 1)
            _place_subtree(roots[0], top, left)
            return grid_h, grid_w

        layout = rng.choice(["h", "v"])
        gap = rng.randint(2, 4)
        if layout == "h":
            req_h = max(h for h, _ in sizes)
            req_w = sum(w for _, w in sizes) + gap
            if req_h > MAX_GRID - 2 or req_w > MAX_GRID - 2:
                continue
            grid_h = rng.randint(max(MIN_GRID, req_h + 2), MAX_GRID)
            grid_w = rng.randint(max(MIN_GRID, req_w + 2), MAX_GRID)
            top_base = rng.randint(1, grid_h - req_h - 1)
            left = rng.randint(1, grid_w - req_w - 1)
            cursor = left
            for root, (rh, rw) in zip(roots, sizes):
                top = top_base + rng.randint(0, req_h - rh)
                _place_subtree(root, top, cursor)
                cursor += rw + gap
            return grid_h, grid_w

        req_h = sum(h for h, _ in sizes) + gap
        req_w = max(w for _, w in sizes)
        if req_h > MAX_GRID - 2 or req_w > MAX_GRID - 2:
            continue
        grid_h = rng.randint(max(MIN_GRID, req_h + 2), MAX_GRID)
        grid_w = rng.randint(max(MIN_GRID, req_w + 2), MAX_GRID)
        top = rng.randint(1, grid_h - req_h - 1)
        left_base = rng.randint(1, grid_w - req_w - 1)
        cursor = top
        for root, (rh, rw) in zip(roots, sizes):
            left = left_base + rng.randint(0, req_w - rw)
            _place_subtree(root, cursor, left)
            cursor += rh + gap
        return grid_h, grid_w

    raise RuntimeError("failed to place roots")


def _draw_input(grid: np.ndarray, roots: list[Node]) -> None:
    for node in _all_nodes(roots):
        assert node.bbox is not None
        _draw_frame(grid, node.bbox, node.color)
        if node.anchor_rel is not None:
            r0, c0, _, _ = node.bbox
            ar0, ac0, ar1, ac1 = node.anchor_rel
            grid[r0 + ar0 : r0 + ar1 + 1, c0 + ac0 : c0 + ac1 + 1] = GREY


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        frames = _extract_frames(grid)
        leaves = [frame for frame in frames if not frame.children]
        anchored = [frame for frame in leaves if frame.anchored]
        unanchored = [frame for frame in leaves if not frame.anchored]
        if not anchored or not unanchored:
            return False
        if max(frame.depth for frame in frames) < 2:
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    roots = _generate_forest(rng)
    for root in roots:
        _set_depths(root, 0)
        _size_tree(root, rng)

    nodes = _all_nodes(roots)
    if any(h > MAX_GRID - 2 or w > MAX_GRID - 2 for h, w in (node.size for node in nodes)):
        return None

    _assign_colors(roots, rng)
    try:
        grid_h, grid_w = _place_roots(roots, rng)
    except RuntimeError:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_input(grid, roots)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_TREE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
