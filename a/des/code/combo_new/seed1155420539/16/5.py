from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_DIM = 30
MAX_GENERATION_ATTEMPTS = 600
MIN_DEPTH = 2
MAX_DEPTH = 5
MIN_FRAME_MARGIN = 2
MAX_FRAME_MARGIN = 4


@dataclass
class FrameNode:
    color: int
    bbox: tuple[int, int, int, int]
    children: list["FrameNode"]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _mirror_bbox(
    inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return (
        or0 + or1 - ir1,
        oc0 + oc1 - ic1,
        or0 + or1 - ir0,
        oc0 + oc1 - ic0,
    )


def _shift_subtree(node: FrameNode, dr: int, dc: int) -> FrameNode:
    return FrameNode(
        color=node.color,
        bbox=_shift_bbox(node.bbox, dr, dc),
        children=[_shift_subtree(child, dr, dc) for child in node.children],
    )


def _mirror_subtree(node: FrameNode, outer_bbox: tuple[int, int, int, int]) -> FrameNode:
    return FrameNode(
        color=node.color,
        bbox=_mirror_bbox(node.bbox, outer_bbox),
        children=[_mirror_subtree(child, outer_bbox) for child in node.children],
    )


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    rows = [pt[0] for pt in coords]
    cols = [pt[1] for pt in coords]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected_area = 2 * (h + w) - 4
    if len(coords) != expected_area:
        return False
    coord_set = set(coords)
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    return True


def _contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _build_frame_forest(frames: list[FrameNode]) -> list[FrameNode]:
    parents = [-1] * len(frames)
    for i, frame in enumerate(frames):
        containers = [j for j, outer in enumerate(frames) if _contains_bbox(outer.bbox, frame.bbox)]
        if containers:
            parents[i] = min(containers, key=lambda j: _bbox_area(frames[j].bbox))

    children_of: dict[int, list[int]] = {i: [] for i in range(len(frames))}
    roots: list[int] = []
    for i, parent in enumerate(parents):
        if parent == -1:
            roots.append(i)
        else:
            children_of[parent].append(i)

    def build(idx: int) -> FrameNode:
        child_indices = sorted(
            children_of[idx],
            key=lambda j: (frames[j].bbox[0], frames[j].bbox[1], _bbox_area(frames[j].bbox)),
        )
        return FrameNode(
            color=frames[idx].color,
            bbox=frames[idx].bbox,
            children=[build(child_idx) for child_idx in child_indices],
        )

    return [build(idx) for idx in sorted(roots, key=lambda j: (frames[j].bbox[0], frames[j].bbox[1]))]


def _fill_odd_layer(out: np.ndarray, node: FrameNode, bg: int) -> None:
    r0, c0, r1, c1 = node.bbox
    if r1 - r0 <= 1 or c1 - c0 <= 1:
        return
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    allowed = np.ones(inner.shape, dtype=bool)
    for child in node.children:
        cr0, cc0, cr1, cc1 = child.bbox
        allowed[cr0 - r0 - 1 : cr1 - r0, cc0 - c0 - 1 : cc1 - c0] = False
    inner[allowed & (inner == bg)] = node.color


def _apply_frame_node(out: np.ndarray, node: FrameNode, depth: int, bg: int) -> None:
    r0, c0, r1, c1 = node.bbox
    if depth % 2 == 1:
        _fill_odd_layer(out, node, bg)
        child_nodes = node.children
    else:
        out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(out[r0 : r1 + 1, c0 : c1 + 1], 2)
        child_nodes = [_mirror_subtree(child, node.bbox) for child in node.children]
    for child in child_nodes:
        _apply_frame_node(out, child, depth + 1, bg)


def _apply_known_tree(grid: np.ndarray, roots: list[FrameNode], bg: int) -> np.ndarray:
    out = grid.copy()
    for root in roots:
        _apply_frame_node(out, root, 1, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested rectangular frames and alternate fill-versus-rotate actions by depth parity.

    Input structure:
    - The background is black. The foreground contains several distinct-color hollow rectangular frames, each one pixel thick.
    - Frames can be nested, so some frame bounding boxes lie strictly inside others.
    - The innermost frame contains one monochrome 4-connected sprite that is not itself a rectangle.

    Rule:
    - Recover the nesting depth of every frame, with each outermost frame at depth 1 and each directly enclosed child one level deeper.
    - On odd depths, fill every background cell inside the frame but outside any immediately nested child frame with the frame's color.
    - On even depths, rotate that frame's entire bounding box, including its border and all deeper contents, by 180 degrees.
    - Continue from outer layers to inner layers so that deeper actions operate on the already transformed arrangement.

    Key insight:
    - Frame colors are only labels; the real control signal is the hidden nesting depth, whose odd/even parity determines whether a layer fills space or flips its whole subscene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames = [
        FrameNode(color=int(comp["color"]), bbox=comp["bbox"], children=[])
        for comp in _same_color_components(g, bg)
        if _is_rect_frame(comp["coords"])
    ]
    assert frames, "expected at least one rectangular frame"

    forest = _build_frame_forest(frames)
    return _apply_known_tree(g, forest, bg)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _grow_polyomino(rng: random.Random, h: int, w: int, area: int) -> np.ndarray:
    seed = (rng.randrange(h), rng.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        touches_edge |= nr in (0, h - 1) or nc in (0, w - 1)
                        stack.append((nr, nc))
            if not touches_edge:
                return True
    return False


def _sample_sprite(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(200):
        box_h = rng.randint(2, max_h)
        box_w = rng.randint(2, max_w)
        max_area = box_h * box_w - 1
        if max_area < 3:
            continue
        area = rng.randint(3, max_area)
        sprite = _grow_polyomino(rng, box_h, box_w, area)
        if np.all(sprite):
            continue
        if np.array_equal(sprite, np.rot90(sprite, 2)):
            continue
        if _has_enclosed_hole(sprite):
            continue
        return sprite
    return None


def _sample_margins(rng: random.Random) -> tuple[int, int, int, int]:
    for _ in range(100):
        margins = tuple(rng.randint(MIN_FRAME_MARGIN, MAX_FRAME_MARGIN) for _ in range(4))
        if margins[0] != margins[1] or margins[2] != margins[3]:
            return margins
    raise RuntimeError("failed to sample asymmetric margins")


def _build_chain_attempt(rng: random.Random) -> tuple[np.ndarray, FrameNode] | None:
    depth = rng.randint(MIN_DEPTH, MAX_DEPTH)
    colors = rng.sample(COLOR_POOL, depth + 1)
    frame_colors = colors[:depth]
    sprite_color = colors[-1]

    leaf_h = rng.randint(5, 8)
    leaf_w = rng.randint(5, 8)
    sprite = _sample_sprite(rng, leaf_h - 2, leaf_w - 2)
    if sprite is None:
        return None

    sprite_h, sprite_w = sprite.shape
    interior_h = leaf_h - 2
    interior_w = leaf_w - 2
    if sprite_h > interior_h or sprite_w > interior_w:
        return None

    sprite_r = rng.randint(0, interior_h - sprite_h)
    sprite_c = rng.randint(0, interior_w - sprite_w)

    local = np.zeros((leaf_h, leaf_w), dtype=int)
    leaf = FrameNode(color=frame_colors[-1], bbox=(0, 0, leaf_h - 1, leaf_w - 1), children=[])
    _draw_frame(local, leaf.bbox, leaf.color)
    sprite_region = local[1 + sprite_r : 1 + sprite_r + sprite_h, 1 + sprite_c : 1 + sprite_c + sprite_w]
    sprite_region[sprite] = sprite_color
    node = leaf

    for color in reversed(frame_colors[:-1]):
        top, bottom, left, right = _sample_margins(rng)
        old_h, old_w = local.shape
        new_h = old_h + top + bottom
        new_w = old_w + left + right
        if new_h > 26 or new_w > 26:
            return None
        new_local = np.zeros((new_h, new_w), dtype=int)
        new_local[top : top + old_h, left : left + old_w] = local
        node = FrameNode(
            color=color,
            bbox=(0, 0, new_h - 1, new_w - 1),
            children=[_shift_subtree(node, top, left)],
        )
        _draw_frame(new_local, node.bbox, node.color)
        local = new_local

    return local, node


def _embed_object(
    rng: random.Random, local: np.ndarray, root: FrameNode
) -> tuple[np.ndarray, list[FrameNode], np.ndarray]:
    obj_h, obj_w = local.shape
    grid_h = rng.randint(obj_h + 2, MAX_DIM)
    grid_w = rng.randint(obj_w + 2, MAX_DIM)
    top = rng.randint(1, grid_h - obj_h - 1)
    left = rng.randint(1, grid_w - obj_w - 1)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    inp[top : top + obj_h, left : left + obj_w] = local
    roots = [_shift_subtree(root, top, left)]

    expected = np.zeros((grid_h, grid_w), dtype=int)
    transformed_local = _apply_known_tree(local, [root], BLACK)
    expected[top : top + obj_h, left : left + obj_w] = transformed_local
    return inp, roots, expected


def _has_even_rotation_effect(inp: np.ndarray, out: np.ndarray, roots: list[FrameNode]) -> bool:
    changed = False

    def visit(node: FrameNode, depth: int) -> None:
        nonlocal changed
        r0, c0, r1, c1 = node.bbox
        if depth % 2 == 0 and not np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
            changed = True
        for child in node.children:
            visit(child, depth + 1)

    for root in roots:
        visit(root, 1)
    return changed


def _has_odd_fill_effect(inp: np.ndarray, out: np.ndarray, roots: list[FrameNode]) -> bool:
    found = False

    def visit(node: FrameNode, depth: int) -> None:
        nonlocal found
        if depth % 2 == 1:
            r0, c0, r1, c1 = node.bbox
            inner_in = inp[r0 + 1 : r1, c0 + 1 : c1]
            inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
            if np.any((inner_in == BLACK) & (inner_out == node.color)):
                found = True
        for child in node.children:
            visit(child, depth + 1)

    for root in roots:
        visit(root, 1)
    return found


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    built = _build_chain_attempt(rng)
    if built is None:
        return None
    local, root = built
    inp, roots, expected = _embed_object(rng, local, root)

    try:
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if not _has_even_rotation_effect(inp, expected, roots):
        return None
    if not _has_odd_fill_effect(inp, expected, roots):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
