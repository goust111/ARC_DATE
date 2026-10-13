from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


BG = 0
FRAME_COLOR = 5
LEGEND_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


@dataclass
class SolverNode:
    coords: np.ndarray
    center: tuple[float, float]
    area: int
    holes: list[np.ndarray]
    is_frame: bool
    parent: SolverNode | None = None
    children: list[SolverNode] = field(default_factory=list)


@dataclass
class GenNode:
    kind: str
    children: list["GenNode"] = field(default_factory=list)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return np.zeros((1, 1), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    outside = np.zeros_like(local, dtype=bool)
    q = deque()
    h, w = local.shape
    for r in range(h):
        for c in (0, w - 1):
            if not local[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not local[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not local[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes_local = (~local) & (~outside)
    holes: list[np.ndarray] = []
    for hole_coords in _components(holes_local):
        full = np.zeros(shape, dtype=bool)
        full[hole_coords[:, 0] + r0, hole_coords[:, 1] + c0] = True
        holes.append(full)
    return holes


def _legend_colors(grid: np.ndarray) -> list[int]:
    legend_coords = np.argwhere((grid != BG) & (grid != FRAME_COLOR))
    assert legend_coords.size > 0, "expected a vertical legend"
    order = sorted((int(r), int(c)) for r, c in legend_coords.tolist())
    return [int(grid[r, c]) for r, c in order]


def _node_sort_key(node: SolverNode) -> tuple[float, float]:
    return float(node.center[0]), float(node.center[1])


def _paste_non_bg(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    mask = src != BG
    dst[top : top + src.shape[0], left : left + src.shape[1]][mask] = src[mask]


def _stack_horizontally(sprites: list[np.ndarray], gap: int) -> np.ndarray:
    assert sprites
    h = max(sprite.shape[0] for sprite in sprites)
    w = sum(sprite.shape[1] for sprite in sprites) + gap * (len(sprites) - 1)
    out = np.zeros((h, w), dtype=int)
    c = 0
    for sprite in sprites:
        _paste_non_bg(out, sprite, 0, c)
        c += sprite.shape[1] + gap
    return out


def _render_canonical(node: SolverNode, depth: int, legend: list[int]) -> np.ndarray:
    assert depth < len(legend), "legend shorter than nesting depth"
    color = int(legend[depth])
    if not node.is_frame:
        return np.array([[color]], dtype=int)

    children = sorted(node.children, key=_node_sort_key)
    if not children:
        out = np.zeros((5, 5), dtype=int)
        out[0, :] = color
        out[-1, :] = color
        out[:, 0] = color
        out[:, -1] = color
        return out

    child_sprites = [_render_canonical(child, depth + 1, legend) for child in children]
    content = _stack_horizontally(child_sprites, gap=1)
    h, w = content.shape
    out = np.zeros((h + 6, w + 6), dtype=int)
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color
    _paste_non_bg(out, content, 3, 3)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Canonicalize a rough nested grey drawing into a colorized rectangular nesting diagram.

    Input structure:
    - The grid has black background, grey 4-connected foreground objects, and a vertical color legend.
    - Grey objects are either closed frames with enclosed holes or solid blobs without holes.
    - The legend is a single vertical stack of non-black, non-grey cells whose top-to-bottom order gives depth colors.

    Rule:
    - Detect every grey connected object and classify it by whether it encloses one or more holes.
    - Build the containment tree: an object is a child of the smallest frame whose hole contains all of its cells.
    - Replace each blob by one colored pixel and each frame by a 1-cell-thick rectangle that leaves 2 cells of inner padding around its direct children; siblings are sorted by original centroid row-major order and laid out left-to-right with a 1-cell gap.
    - Color each canonical object with the legend color indexed by its nesting depth, then crop to the minimal bounding box.

    Key insight:
    - The output keeps only topology: nesting depth controls color through the legend, and containment alone controls the canonical geometry.
    """
    g = np.asarray(grid, dtype=int)
    fg_mask = g == FRAME_COLOR
    comps = _components(fg_mask)
    assert comps, "expected at least one grey object"

    nodes: list[SolverNode] = []
    for coords in comps:
        holes = _component_holes(g.shape, coords)
        nodes.append(
            SolverNode(
                coords=coords,
                center=(float(coords[:, 0].mean()), float(coords[:, 1].mean())),
                area=int(coords.shape[0]),
                holes=holes,
                is_frame=bool(holes),
            )
        )

    assert any(node.is_frame for node in nodes), "expected at least one frame"

    for node in nodes:
        containers = []
        for frame in nodes:
            if frame is node or not frame.is_frame:
                continue
            if any(np.all(hole[tuple(node.coords.T)]) for hole in frame.holes):
                containers.append(frame)
        if containers:
            node.parent = min(containers, key=lambda item: item.area)
            node.parent.children.append(node)

    legend = _legend_colors(g)
    roots = sorted([node for node in nodes if node.parent is None], key=_node_sort_key)
    sprites = [_render_canonical(node, 0, legend) for node in roots]
    out = _stack_horizontally(sprites, gap=1) if len(sprites) > 1 else sprites[0]
    return _crop_non_bg(out, bg=BG)


def _random_blob(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(40):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        target = int(rng.integers(3, min(7, h * w) + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            idx = int(rng.integers(len(frontier)))
            cells.add(frontier[idx])
        if len(cells) != target:
            continue
        sprite = np.zeros((h, w), dtype=int)
        for r, c in cells:
            sprite[r, c] = FRAME_COLOR
        sprite = _crop_non_bg(sprite, bg=BG)
        coords = np.argwhere(sprite == FRAME_COLOR)
        if coords.shape[0] < 3:
            continue
        if _component_holes(sprite.shape, coords):
            continue
        return sprite
    return None


def _render_input_node(node: GenNode, rng: np.random.Generator) -> np.ndarray | None:
    if node.kind == "blob":
        return _random_blob(rng)

    child_sprites: list[np.ndarray] = []
    for child in node.children:
        sprite = _render_input_node(child, rng)
        if sprite is None:
            return None
        child_sprites.append(sprite)

    if not child_sprites:
        return None

    gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(child_sprites) - 1))]
    top_offsets = sorted(int(v) for v in rng.integers(0, 3, size=len(child_sprites)))

    positions = []
    cur_c = 0
    content_h = 0
    for i, sprite in enumerate(child_sprites):
        top = top_offsets[i]
        positions.append((top, cur_c))
        content_h = max(content_h, top + sprite.shape[0])
        cur_c += sprite.shape[1]
        if i < len(gaps):
            cur_c += gaps[i]
    content_w = cur_c

    pad_top = int(rng.integers(2, 5))
    pad_bottom = int(rng.integers(2, 5))
    pad_left = int(rng.integers(2, 5))
    pad_right = int(rng.integers(2, 5))

    base = np.zeros((content_h + pad_top + pad_bottom + 2, content_w + pad_left + pad_right + 2), dtype=int)
    base[0, :] = FRAME_COLOR
    base[-1, :] = FRAME_COLOR
    base[:, 0] = FRAME_COLOR
    base[:, -1] = FRAME_COLOR
    for sprite, (top, left) in zip(child_sprites, positions):
        _paste_non_bg(base, sprite, 1 + pad_top + top, 1 + pad_left + left)

    out = np.zeros((base.shape[0] + 2, base.shape[1] + 2), dtype=int)
    out[1:-1, 1:-1] = base

    if out.shape[1] > 6 and rng.random() < 0.8:
        out[0, int(rng.integers(2, out.shape[1] - 2))] = FRAME_COLOR
    if out.shape[1] > 6 and rng.random() < 0.8:
        out[-1, int(rng.integers(2, out.shape[1] - 2))] = FRAME_COLOR
    if out.shape[0] > 6 and rng.random() < 0.8:
        out[int(rng.integers(2, out.shape[0] - 2)), 0] = FRAME_COLOR
    if out.shape[0] > 6 and rng.random() < 0.8:
        out[int(rng.integers(2, out.shape[0] - 2)), -1] = FRAME_COLOR

    out = _crop_non_bg(out, bg=BG)

    border_pixels = []
    for c in range(out.shape[1]):
        if out[0, c] == FRAME_COLOR:
            border_pixels.append((0, c))
        if out[-1, c] == FRAME_COLOR:
            border_pixels.append((out.shape[0] - 1, c))
    for r in range(out.shape[0]):
        if out[r, 0] == FRAME_COLOR:
            border_pixels.append((r, 0))
        if out[r, -1] == FRAME_COLOR:
            border_pixels.append((r, out.shape[1] - 1))
    if not border_pixels:
        return None

    frame_mask = np.zeros_like(out, dtype=bool)
    q = deque([border_pixels[0]])
    frame_mask[border_pixels[0]] = True
    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if out[nr, nc] == FRAME_COLOR and not frame_mask[nr, nc]:
                    frame_mask[nr, nc] = True
                    q.append((nr, nc))

    frame_coords = np.argwhere(frame_mask)
    if not _component_holes(out.shape, frame_coords):
        return None
    return out


def _make_any_subtree(
    rng: np.random.Generator, depth: int, target_depth: int, force_deep: bool
) -> GenNode:
    if depth >= target_depth:
        return GenNode("blob")
    if depth > 0 and not force_deep and rng.random() < 0.45:
        return GenNode("blob")
    return _make_frame_subtree(rng, depth, target_depth, force_deep)


def _make_frame_subtree(
    rng: np.random.Generator, depth: int, target_depth: int, force_deep: bool
) -> GenNode:
    child_count = int(rng.integers(1, 4))
    deep_index = int(rng.integers(child_count)) if force_deep and depth < target_depth else -1
    children: list[GenNode] = []
    for i in range(child_count):
        child_force = i == deep_index
        if child_force:
            child = _make_any_subtree(rng, depth + 1, target_depth, True)
        else:
            child = _make_any_subtree(rng, depth + 1, target_depth, False)
        children.append(child)
    if all(child.kind == "blob" for child in children) and depth + 1 < target_depth:
        children[int(rng.integers(child_count))] = _make_frame_subtree(rng, depth + 1, target_depth, True)
    return GenNode("frame", children)


def _tree_max_depth(node: GenNode, depth: int = 0) -> int:
    if not node.children:
        return depth
    return max(_tree_max_depth(child, depth + 1) for child in node.children)


def _compose_subject(rng: np.random.Generator, sprites: list[np.ndarray]) -> np.ndarray:
    gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(sprites) - 1))]
    top_offsets = sorted(int(v) for v in rng.integers(0, 3, size=len(sprites)))
    positions = []
    cur_c = 0
    h = 0
    for i, sprite in enumerate(sprites):
        top = top_offsets[i]
        positions.append((top, cur_c))
        h = max(h, top + sprite.shape[0])
        cur_c += sprite.shape[1]
        if i < len(gaps):
            cur_c += gaps[i]
    out = np.zeros((h, cur_c), dtype=int)
    for sprite, (top, left) in zip(sprites, positions):
        _paste_non_bg(out, sprite, top, left)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    target_depth = int(rng.choice([1, 2, 2, 3]))
    root_frame = _make_frame_subtree(rng, 0, target_depth, True)
    roots = [root_frame]
    if rng.random() < 0.4:
        roots.append(GenNode("blob"))

    root_sprites: list[np.ndarray] = []
    for root in roots:
        sprite = _render_input_node(root, rng)
        if sprite is None:
            return None
        root_sprites.append(sprite)

    subject = _compose_subject(rng, root_sprites)
    max_depth = max(_tree_max_depth(root) for root in roots)
    legend_colors = rng.choice(LEGEND_PALETTE, size=max_depth + 1, replace=False).tolist()

    legend_col = int(rng.integers(0, 2))
    subject_col = int(rng.integers(legend_col + 3, legend_col + 7))
    top_pad = int(rng.integers(0, 4))
    bottom_pad = int(rng.integers(0, 4))
    right_pad = int(rng.integers(0, 5))
    legend_bottom_offset = int(rng.integers(0, 2))

    grid_h = max(subject.shape[0] + top_pad + bottom_pad, len(legend_colors) + legend_bottom_offset + 1)
    grid_w = subject_col + subject.shape[1] + right_pad
    if max(grid_h, grid_w) > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _paste_non_bg(grid, subject, top_pad, subject_col)

    legend_top = grid_h - legend_bottom_offset - len(legend_colors)
    for i, color in enumerate(legend_colors):
        grid[legend_top + i, legend_col] = int(color)

    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if len(np.unique(grid)) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
