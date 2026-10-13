from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Frame:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    parent: int | None = None
    depth: int = 0

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width


@dataclass
class _Node:
    depth: int
    children: list["_Node"] = field(default_factory=list)
    outer_h: int = 0
    outer_w: int = 0
    bbox: tuple[int, int, int, int] | None = None
    layout: str = "leaf"


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _frame_from_component(coords: np.ndarray, color: int) -> _Frame | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if coords.shape[0] != 2 * (h + w) - 4:
        return None

    patch = np.zeros((h, w), dtype=bool)
    patch[rows - r0, cols - c0] = True
    border = np.zeros_like(patch)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(patch, border):
        return None
    return _Frame(color=color, r0=r0, c0=c0, r1=r1, c1=c1)


def _contains(outer: _Frame, inner: _Frame) -> bool:
    return (
        outer.r0 < inner.r0
        and outer.c0 < inner.c0
        and outer.r1 > inner.r1
        and outer.c1 > inner.c1
    )


def _annotate_depths(frames: list[_Frame]) -> None:
    for i, frame in enumerate(frames):
        parents = [j for j, other in enumerate(frames) if i != j and _contains(other, frame)]
        if parents:
            frame.parent = min(parents, key=lambda j: frames[j].area)

    def depth_of(idx: int) -> int:
        frame = frames[idx]
        if frame.depth:
            return frame.depth
        if frame.parent is None:
            frame.depth = 1
        else:
            frame.depth = depth_of(frame.parent) + 1
        return frame.depth

    for i in range(len(frames)):
        depth_of(i)


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[_Frame]]:
    candidates: list[tuple[int, list[_Frame]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        if len(comps) < 2:
            continue
        frames: list[_Frame] = []
        ok = True
        for comp in comps:
            frame = _frame_from_component(comp, color)
            if frame is None:
                ok = False
                break
            frames.append(frame)
        if ok:
            candidates.append((color, frames))
    assert len(candidates) == 1, "expected a unique border color"
    border_color, frames = candidates[0]
    _annotate_depths(frames)
    return border_color, frames


def _interior(grid: np.ndarray, frame: _Frame) -> np.ndarray:
    return grid[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1]


def _analyze_grid(
    grid: np.ndarray,
) -> tuple[int, int, list[_Frame], int, list[tuple[int, int, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    border_color, frames = _find_frames(g, bg)

    holders: list[int] = []
    for i, frame in enumerate(frames):
        inside = _interior(g, frame)
        if np.any((inside != bg) & (inside != border_color)):
            holders.append(i)
    assert holders, "expected one patterned container"

    deepest_depth = max(frames[i].depth for i in holders)
    source_candidates = [i for i in holders if frames[i].depth == deepest_depth]
    assert len(source_candidates) == 1, "expected a unique deepest patterned frame"
    source_idx = source_candidates[0]
    assert deepest_depth == max(frame.depth for frame in frames), "pattern must live at the deepest level"

    source_inside = _interior(g, frames[source_idx])
    payload_cells = np.argwhere((source_inside != bg) & (source_inside != border_color))
    assert payload_cells.size > 0, "pattern must be non-empty"
    payload = [(int(r), int(c), int(source_inside[r, c])) for r, c in payload_cells]

    peer_count = sum(frame.depth == deepest_depth for frame in frames)
    assert peer_count >= 2, "need at least two same-level deepest containers"
    for i, frame in enumerate(frames):
        if frame.depth != deepest_depth or i == source_idx:
            continue
        peer_inside = _interior(g, frame)
        assert not np.any((peer_inside != bg) & (peer_inside != border_color)), "peer containers must start empty"
        ih, iw = peer_inside.shape
        for dr, dc, _color in payload:
            assert dr < ih and dc < iw, "pattern must fit all peer containers"

    return bg, border_color, frames, source_idx, payload, deepest_depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the deepest container's payload into every other container at the same nesting level.

    Input structure:
    - The grid has a dominant background color, typically black.
    - One foreground color forms several disjoint one-pixel-thick rectangular frames.
    - Frames may be nested inside other frames, and exactly one deepest-level frame contains a non-background, non-frame pattern while its peers are empty.

    Rule:
    - Recover every rectangular frame and infer its nesting depth from strict containment.
    - Find the unique deepest frame whose interior contains the payload pattern.
    - Record all non-background, non-frame cells in that frame by their positions relative to the frame interior's top-left corner.
    - Paste those same relative cells into every other frame at the same depth, leaving every other pixel unchanged.

    Key insight:
    The visible pattern is not enough by itself; the target set is determined by the hidden containment tree of same-colored rectangular containers.
    """
    g = np.asarray(grid, dtype=int)
    bg, _border_color, frames, _source_idx, payload, target_depth = _analyze_grid(g)
    out = g.copy()
    for frame in frames:
        if frame.depth != target_depth:
            continue
        for dr, dc, color in payload:
            out[frame.r0 + 1 + dr, frame.c0 + 1 + dc] = color
    return out


_MAX_GENERATION_ATTEMPTS = 300
_MAX_LEAVES = 6
_MIN_LEAF_INTERIOR = 4
_MAX_LEAF_INTERIOR = 8


def _iter_nodes(roots: list[_Node]) -> list[_Node]:
    out: list[_Node] = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children)
    return out


def _build_tree(rng: np.random.Generator) -> tuple[list[_Node], list[_Node]] | None:
    max_depth = int(rng.integers(1, 4))
    root_count = 2 if max_depth == 1 else int(rng.integers(1, 3))
    roots = [_Node(depth=1) for _ in range(root_count)]

    current = roots
    branched = root_count > 1
    for depth in range(1, max_depth):
        next_nodes: list[_Node] = []
        for node in current:
            if depth == max_depth - 1 and len(current) == 1 and not branched:
                child_count = int(rng.integers(2, 4))
            else:
                child_count = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
            if child_count > 1:
                branched = True
            node.children = [_Node(depth=depth + 1) for _ in range(child_count)]
            next_nodes.extend(node.children)
        current = next_nodes

    leaves = current
    all_nodes = _iter_nodes(roots)
    if len(leaves) < 2 or len(leaves) > _MAX_LEAVES:
        return None
    if len(all_nodes) > 12:
        return None
    return roots, leaves


def _size_node(node: _Node, rng: np.random.Generator) -> None:
    for child in node.children:
        _size_node(child, rng)

    if not node.children:
        inner_h = int(rng.integers(_MIN_LEAF_INTERIOR, _MAX_LEAF_INTERIOR + 1))
        inner_w = int(rng.integers(_MIN_LEAF_INTERIOR, _MAX_LEAF_INTERIOR + 1))
        node.outer_h = inner_h + 2
        node.outer_w = inner_w + 2
        node.layout = "leaf"
        return

    if len(node.children) == 1:
        child = node.children[0]
        inner_h = child.outer_h + 2 + int(rng.integers(0, 3))
        inner_w = child.outer_w + 2 + int(rng.integers(0, 3))
        node.layout = "single"
    else:
        node.layout = str(rng.choice(["h", "v"]))
        if node.layout == "h":
            sum_w = sum(child.outer_w for child in node.children)
            max_h = max(child.outer_h for child in node.children)
            inner_h = max_h + 2 + int(rng.integers(0, 3))
            inner_w = sum_w + (len(node.children) - 1) + 2 + int(rng.integers(0, 4))
        else:
            sum_h = sum(child.outer_h for child in node.children)
            max_w = max(child.outer_w for child in node.children)
            inner_h = sum_h + (len(node.children) - 1) + 2 + int(rng.integers(0, 4))
            inner_w = max_w + 2 + int(rng.integers(0, 3))

    node.outer_h = inner_h + 2
    node.outer_w = inner_w + 2


def _split_extra(rng: np.random.Generator, total: int, slots: int) -> list[int]:
    if slots == 1:
        return [total]
    if total == 0:
        return [0] * slots
    probs = np.full(slots, 1.0 / slots)
    return rng.multinomial(total, probs).tolist()


def _place_node(node: _Node, r0: int, c0: int, rng: np.random.Generator) -> None:
    node.bbox = (r0, c0, r0 + node.outer_h - 1, c0 + node.outer_w - 1)
    if not node.children:
        return

    inner_h = node.outer_h - 2
    inner_w = node.outer_w - 2

    if len(node.children) == 1:
        child = node.children[0]
        row_offset = int(rng.integers(1, inner_h - child.outer_h))
        col_offset = int(rng.integers(1, inner_w - child.outer_w))
        _place_node(child, r0 + 1 + row_offset, c0 + 1 + col_offset, rng)
        return

    if node.layout == "h":
        total_w = sum(child.outer_w for child in node.children)
        base_slots = [1] * (len(node.children) + 1)
        extra = inner_w - (total_w + sum(base_slots))
        slots = [base + add for base, add in zip(base_slots, _split_extra(rng, extra, len(base_slots)))]
        col = slots[0]
        for idx, child in enumerate(node.children):
            row_offset = int(rng.integers(1, inner_h - child.outer_h))
            _place_node(child, r0 + 1 + row_offset, c0 + 1 + col, rng)
            col += child.outer_w + slots[idx + 1]
        return

    total_h = sum(child.outer_h for child in node.children)
    base_slots = [1] * (len(node.children) + 1)
    extra = inner_h - (total_h + sum(base_slots))
    slots = [base + add for base, add in zip(base_slots, _split_extra(rng, extra, len(base_slots)))]
    row = slots[0]
    for idx, child in enumerate(node.children):
        col_offset = int(rng.integers(1, inner_w - child.outer_w))
        _place_node(child, r0 + 1 + row, c0 + 1 + col_offset, rng)
        row += child.outer_h + slots[idx + 1]


def _choose_padding(rng: np.random.Generator, required: int, limit: int = 30) -> tuple[int, int] | None:
    budget = limit - required
    if budget < 2:
        return None
    first_hi = min(4, budget - 1)
    if first_hi < 1:
        return None
    first = int(rng.integers(1, first_hi + 1))
    second_hi = min(4, budget - first)
    if second_hi < 1:
        return None
    second = int(rng.integers(1, second_hi + 1))
    return first, second


def _place_forest(roots: list[_Node], rng: np.random.Generator) -> tuple[int, int] | None:
    roots = list(roots)
    if len(roots) == 2 and rng.random() < 0.5:
        roots.reverse()

    if len(roots) == 1:
        required_h = roots[0].outer_h
        required_w = roots[0].outer_w
        pads_h = _choose_padding(rng, required_h)
        pads_w = _choose_padding(rng, required_w)
        if pads_h is None or pads_w is None:
            return None
        top, bottom = pads_h
        left, right = pads_w
        _place_node(roots[0], top, left, rng)
        return required_h + top + bottom, required_w + left + right

    gap = int(rng.integers(2, 5))
    options = ["h", "v"]
    if rng.random() < 0.5:
        options.reverse()

    for layout in options:
        if layout == "h":
            required_h = max(root.outer_h for root in roots)
            required_w = roots[0].outer_w + roots[1].outer_w + gap
        else:
            required_h = roots[0].outer_h + roots[1].outer_h + gap
            required_w = max(root.outer_w for root in roots)

        pads_h = _choose_padding(rng, required_h)
        pads_w = _choose_padding(rng, required_w)
        if pads_h is None or pads_w is None:
            continue
        top, bottom = pads_h
        left, right = pads_w

        if layout == "h":
            r0 = top + int(rng.integers(0, required_h - roots[0].outer_h + 1))
            r1 = top + int(rng.integers(0, required_h - roots[1].outer_h + 1))
            _place_node(roots[0], r0, left, rng)
            _place_node(roots[1], r1, left + roots[0].outer_w + gap, rng)
        else:
            c0 = left + int(rng.integers(0, required_w - roots[0].outer_w + 1))
            c1 = left + int(rng.integers(0, required_w - roots[1].outer_w + 1))
            _place_node(roots[0], top, c0, rng)
            _place_node(roots[1], top + roots[0].outer_h + gap, c1, rng)
        return required_h + top + bottom, required_w + left + right

    return None


def _draw_frames(grid: np.ndarray, node: _Node, color: int) -> None:
    assert node.bbox is not None
    r0, c0, r1, c1 = node.bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    for child in node.children:
        _draw_frames(grid, child, color)


def _random_shape(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        choice = list(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add(choice)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_tree(rng)
    if built is None:
        return None
    roots, leaves = built

    for root in roots:
        _size_node(root, rng)
    shape = _place_forest(roots, rng)
    if shape is None:
        return None

    grid_h, grid_w = shape
    grid = np.zeros((grid_h, grid_w), dtype=int)

    colors = [int(c) for c in rng.permutation(np.arange(1, 10))]
    border_color = colors[0]
    pattern_color = colors[1]

    for root in roots:
        _draw_frames(grid, root, border_color)

    min_inner_h = min(leaf.outer_h - 2 for leaf in leaves)
    min_inner_w = min(leaf.outer_w - 2 for leaf in leaves)
    pattern_h = int(rng.integers(1, min(4, min_inner_h) + 1))
    pattern_w = int(rng.integers(1, min(4, min_inner_w) + 1))
    max_area = min(pattern_h * pattern_w, 6)
    area = int(rng.integers(2 if max_area >= 2 else 1, max_area + 1))
    pattern = _random_shape(rng, pattern_h, pattern_w, area)
    offset_r = int(rng.integers(0, min_inner_h - pattern_h + 1))
    offset_c = int(rng.integers(0, min_inner_w - pattern_w + 1))

    source_leaf = leaves[int(rng.integers(0, len(leaves)))]
    assert source_leaf.bbox is not None
    r0, c0, _r1, _c1 = source_leaf.bbox
    for pr, pc in np.argwhere(pattern):
        grid[r0 + 1 + offset_r + int(pr), c0 + 1 + offset_c + int(pc)] = pattern_color

    try:
        _bg, _border, frames, _source, payload, depth = _analyze_grid(grid)
        if sum(frame.depth == depth for frame in frames) != len(leaves):
            return None
        out = main(grid)
        if not np.any(out != grid):
            return None
        if len(payload) != int(pattern.sum()):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_GENERATION_ATTEMPTS} attempts")
