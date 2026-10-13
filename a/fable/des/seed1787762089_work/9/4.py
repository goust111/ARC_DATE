from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np


BG = 0
GREY = 5
LEGEND_PALETTE = (2, 4, 8, 3, 7)

GRID_MIN = 20
GRID_MAX = 26
GEN_RETRIES = 500

_BLOB_MASKS = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
)


@dataclass
class Node:
    kind: str
    depth: int
    children: list["Node"] = field(default_factory=list)
    mask: np.ndarray | None = None
    size: tuple[int, int] = (0, 0)
    offsets: list[tuple[int, int]] = field(default_factory=list)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(patch, border))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = outer
    ir0, ic0, ir1, ic1 = inner
    return r0 < ir0 and c0 < ic0 and ir1 < r1 and ic1 < c1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    h, _w = g.shape

    legend_rows = [r for r in range(h) if int(g[r, 0]) != bg]
    legend = [int(g[r, 0]) for r in legend_rows]
    assert legend, "expected a left-edge legend"

    out = g.copy()
    out[legend_rows, 0] = bg

    objects = []
    frames = []
    for coords in _components(g == GREY):
        bbox = _bbox(coords)
        is_frame = _is_rect_frame(coords)
        item = {"coords": coords, "bbox": bbox, "is_frame": is_frame}
        objects.append(item)
        if is_frame:
            frames.append(item)

    assert objects, "expected at least one grey object"
    assert frames, "expected at least one grey frame"

    for item in objects:
        depth = 1 + sum(
            1 for frame in frames if frame is not item and _strictly_contains(frame["bbox"], item["bbox"])
        )
        assert 1 <= depth <= len(legend), "legend length must match the maximum object depth"
        color = legend[depth - 1]
        out[tuple(item["coords"].T)] = color

    return out


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.fliplr(out)
    k = rng.randrange(4)
    out = np.rot90(out, k)
    return out.copy()


def _make_blob(rng: random.Random, depth: int) -> Node:
    return Node(kind="blob", depth=depth, mask=_transform_mask(rng.choice(_BLOB_MASKS), rng))


def _make_frame_tree(rng: random.Random, depth: int, target_depth: int, blob_path: bool) -> Node:
    node = Node(kind="frame", depth=depth)
    if depth == target_depth:
        if blob_path:
            node.children = [_make_blob(rng, depth + 1) for _ in range(rng.choice([1, 2]))]
        return node

    primary = _make_frame_tree(rng, depth + 1, target_depth, blob_path)
    children = [primary]
    extra_prob = 0.5 if depth < target_depth - 1 else 0.35
    if rng.random() < extra_prob:
        extra_target = depth + 1
        if depth + 1 < target_depth and rng.random() < 0.4:
            extra_target = depth + 2
        children.append(_make_frame_tree(rng, depth + 1, min(target_depth, extra_target), False))
    rng.shuffle(children)
    node.children = children
    return node


def _measure(node: Node, rng: random.Random) -> tuple[int, int]:
    if node.kind == "blob":
        assert node.mask is not None
        node.size = tuple(int(x) for x in node.mask.shape)
        return node.size

    for child in node.children:
        _measure(child, rng)

    if not node.children:
        inner_h = rng.randint(1, 3)
        inner_w = rng.randint(1, 4)
        node.size = (inner_h + 2, inner_w + 2)
        node.offsets = []
        return node.size

    top = 1 + int(rng.random() < 0.2)
    bottom = 1 + int(rng.random() < 0.2)
    left = 1 + int(rng.random() < 0.2)
    right = 1 + int(rng.random() < 0.2)
    gap = 0 if len(node.children) == 1 else rng.choice([1, 2])
    orientation = rng.choice(["h", "v"])
    child_sizes = [child.size for child in node.children]

    if orientation == "h":
        base_h = max(h for h, _ in child_sizes)
        base_w = sum(w for _, w in child_sizes) + gap * (len(child_sizes) - 1)
        content_h = base_h + rng.choice([0, 0, 1])
        content_w = base_w
        offsets = []
        x = 0
        for child in node.children:
            ch, cw = child.size
            y = rng.randint(0, content_h - ch)
            offsets.append((1 + top + y, 1 + left + x))
            x += cw + gap
    else:
        base_h = sum(h for h, _ in child_sizes) + gap * (len(child_sizes) - 1)
        base_w = max(w for _, w in child_sizes)
        content_h = base_h
        content_w = base_w + rng.choice([0, 0, 1])
        offsets = []
        y = 0
        for child in node.children:
            ch, cw = child.size
            x = rng.randint(0, content_w - cw)
            offsets.append((1 + top + y, 1 + left + x))
            y += ch + gap

    node.size = (2 + top + bottom + content_h, 2 + left + right + content_w)
    node.offsets = offsets
    return node.size


def _forest_layout(rng: random.Random, roots: list[Node]) -> tuple[int, int, list[tuple[int, int]]] | None:
    if len(roots) == 1:
        return roots[0].size[0], roots[0].size[1], [(0, 0)]

    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    for orientation in ("h", "v"):
        gap = rng.choice([2, 3])
        if orientation == "h":
            base_h = max(root.size[0] for root in roots)
            base_w = sum(root.size[1] for root in roots) + gap
            forest_h = base_h + rng.choice([0, 1, 2])
            forest_w = base_w
            offsets = []
            x = 0
            for root in roots:
                y = rng.randint(0, forest_h - root.size[0])
                offsets.append((y, x))
                x += root.size[1] + gap
        else:
            base_h = sum(root.size[0] for root in roots) + gap
            base_w = max(root.size[1] for root in roots)
            forest_h = base_h
            forest_w = base_w + rng.choice([0, 1, 2])
            offsets = []
            y = 0
            for root in roots:
                x = rng.randint(0, forest_w - root.size[1])
                offsets.append((y, x))
                y += root.size[0] + gap
        if forest_h <= GRID_MAX and forest_w <= GRID_MAX - 2:
            candidates.append((forest_h, forest_w, offsets))

    if not candidates:
        return None
    return rng.choice(candidates)


def _legend_rows(rng: random.Random, height: int, count: int) -> list[int] | None:
    if 2 * count - 1 > height:
        return None
    bases = sorted(rng.sample(range(height - count + 1), count))
    return [base + i for i, base in enumerate(bases)]


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _render_node(inp: np.ndarray, out: np.ndarray, node: Node, r0: int, c0: int, colors: list[int]) -> None:
    if node.kind == "blob":
        assert node.mask is not None
        h, w = node.mask.shape
        inp[r0 : r0 + h, c0 : c0 + w][node.mask] = GREY
        out[r0 : r0 + h, c0 : c0 + w][node.mask] = colors[node.depth - 1]
        return

    h, w = node.size
    _paint_frame(inp, r0, c0, h, w, GREY)
    _paint_frame(out, r0, c0, h, w, colors[node.depth - 1])
    for child, (dr, dc) in zip(node.children, node.offsets):
        _render_node(inp, out, child, r0 + dr, c0 + dc, colors)


def _max_depth(node: Node) -> int:
    if node.kind == "blob":
        return node.depth
    if not node.children:
        return node.depth
    return max(_max_depth(child) for child in node.children)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    max_frame_depth = rng.choice([3, 4])
    with_blobs = rng.choice([False, True])
    root_count = rng.choice([1, 2])

    primary = _make_frame_tree(rng, 1, max_frame_depth, with_blobs)
    roots = [primary]
    if root_count == 2:
        secondary_depth = rng.randint(1, max_frame_depth - 1)
        roots.append(_make_frame_tree(rng, 1, secondary_depth, False))
        rng.shuffle(roots)

    for root in roots:
        _measure(root, rng)

    forest = _forest_layout(rng, roots)
    if forest is None:
        return None
    forest_h, forest_w, root_offsets = forest

    legend_len = max_frame_depth + int(with_blobs)
    min_h = max(forest_h, 2 * legend_len - 1, GRID_MIN)
    min_w = max(2 + forest_w, GRID_MIN)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)
    legend_rows = _legend_rows(rng, h, legend_len)
    if legend_rows is None:
        return None

    max_root_r = h - forest_h
    max_root_c = w - forest_w
    if max_root_c < 2:
        return None
    root_r = rng.randint(0, max_root_r)
    root_c = rng.randint(2, max_root_c)

    colors = rng.sample(list(LEGEND_PALETTE), legend_len)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for row, color in zip(legend_rows, colors):
        inp[row, 0] = color

    for root, (dr, dc) in zip(roots, root_offsets):
        _render_node(inp, out, root, root_r + dr, root_c + dc, colors)

    max_depth = max(_max_depth(root) for root in roots)
    if max_depth != legend_len:
        return None
    if int(np.sum(inp == BG)) <= int(np.sum(inp != BG)):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
