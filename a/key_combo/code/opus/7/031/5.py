import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
LAYER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _rect_ring_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_layers(source_patch: np.ndarray) -> list[dict]:
    h, w = source_patch.shape
    top, left, bottom, right = 0, 0, h - 1, w - 1
    layers = []
    while True:
        sub = source_patch[top : bottom + 1, left : right + 1]
        color = int(sub[0, 0])
        if np.all(sub == color):
            layers.append(
                {
                    "shape": tuple(map(int, sub.shape)),
                    "color": color,
                    "mask": np.ones(sub.shape, dtype=bool),
                }
            )
            break
        ring = _rect_ring_mask(sub.shape)
        assert np.all(sub[ring] == color), "expected a uniform one-cell ring"
        layers.append({"shape": tuple(map(int, sub.shape)), "color": color, "mask": ring})
        top += 1
        left += 1
        bottom -= 1
        right -= 1
        assert top <= bottom and left <= right, "invalid nested rectangle structure"
    return layers


def _paint_nested_source(grid: np.ndarray, r0: int, c0: int, shapes: list[tuple[int, int]], colors: list[int]) -> None:
    for idx, ((h, w), color) in enumerate(zip(shapes, colors)):
        rr = r0 + idx
        cc = c0 + idx
        if idx == len(shapes) - 1:
            grid[rr : rr + h, cc : cc + w] = color
        else:
            grid[rr, cc : cc + w] = color
            grid[rr + h - 1, cc : cc + w] = color
            grid[rr : rr + h, cc] = color
            grid[rr : rr + h, cc + w - 1] = color


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_shape: tuple[int, int]) -> None:
    h, w = inner_shape
    outer_h, outer_w = h + 2, w + 2
    grid[r0, c0 : c0 + outer_w] = FRAME
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = FRAME
    grid[r0 : r0 + outer_h, c0] = FRAME
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = FRAME


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h_lim, w_lim = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > h_lim or c0 + w > w_lim:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h_lim, r0 + h + pad)
    cc1 = min(w_lim, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    h_lim, w_lim = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h_lim, r0 + h + pad)
    cc1 = min(w_lim, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the nested source rectangle into layer stamps and place each stamp into the gray frame with matching interior size.

    Input structure:
    - The background is black.
    - One multicolor filled rectangle is composed of 3 or 4 nested one-cell-thick rectangular layers, with a solid innermost rectangle.
    - Several disjoint gray hollow rectangular frames appear elsewhere; each frame has a black interior.

    Rule:
    - Locate the nested multicolor rectangle and peel it from outermost to innermost.
    - Record each layer's outer size and its color; non-innermost layers are rectangular rings, and the innermost layer is a solid rectangle.
    - For every gray frame, measure its interior size and find the unique source layer with the same size.
    - Paint that layer's ring-or-solid mask in its color inside the frame while keeping the gray border and the original source rectangle unchanged.

    Key insight:
    - The left object is not recolored in place; it is decomposed into size-indexed stamp templates, and frame interiors act as slots keyed by those exact layer sizes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    source_mask = (g != BG) & (g != FRAME)
    source_coords = np.argwhere(source_mask)
    assert source_coords.size > 0, "expected a colored nested source rectangle"
    sr0, sc0, sr1, sc1 = _bbox_from_coords(source_coords)
    assert np.all(source_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]), "source must fill its bounding box"
    source_patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1]

    layers = _extract_layers(source_patch)
    layer_by_shape = {}
    for layer in layers:
        shape = layer["shape"]
        assert shape not in layer_by_shape, "layer sizes must be unique"
        layer_by_shape[shape] = layer

    frame_components = _components(g == FRAME)
    assert len(frame_components) == len(layers), "frame count must equal layer count"

    out = g.copy()
    for comp in frame_components:
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        fh, fw = patch.shape
        assert fh >= 3 and fw >= 3, "gray frames must have an interior"
        assert np.all(patch[0, :] == FRAME) and np.all(patch[-1, :] == FRAME)
        assert np.all(patch[:, 0] == FRAME) and np.all(patch[:, -1] == FRAME)
        assert np.all(patch[1:-1, 1:-1] == BG), "frame interior must be black"

        inner_shape = (fh - 2, fw - 2)
        assert inner_shape in layer_by_shape, "every frame interior must match a source layer"
        layer = layer_by_shape[inner_shape]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[layer["mask"]] = layer["color"]

    return out


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.choice([3, 4])
    max_inner = 5 if layer_count == 3 else 3
    inner_h = random.randint(1, max_inner)
    inner_w = random.randint(1, max_inner)
    outer_shapes = [
        (inner_h + 2 * d, inner_w + 2 * d)
        for d in range(layer_count - 1, -1, -1)
    ]
    colors = random.sample(LAYER_COLORS, layer_count)

    source_h, source_w = outer_shapes[0]
    frame_outer_shapes = [(h + 2, w + 2) for h, w in outer_shapes]
    max_frame_h = max(h for h, _ in frame_outer_shapes)
    max_frame_w = max(w for _, w in frame_outer_shapes)

    min_h = max(18, max(source_h, max_frame_h) + 4)
    min_w = max(20, source_w + max_frame_w + 7)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    source_c_max = max(1, min(grid_w // 3, grid_w - source_w - max_frame_w - 4))
    if source_c_max < 1:
        return None
    source_r = random.randint(1, grid_h - source_h - 1)
    source_c = random.randint(1, source_c_max)
    _paint_nested_source(grid, source_r, source_c, outer_shapes, colors)
    _mark_occupied(occupied, source_r, source_c, source_h, source_w, pad=1)

    right_start = source_c + source_w + random.randint(2, 4)
    if right_start + max_frame_w > grid_w:
        return None

    frame_indices = list(range(layer_count))
    random.shuffle(frame_indices)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(frame_indices, key=lambda idx: frame_outer_shapes[idx][0] * frame_outer_shapes[idx][1], reverse=True)
    for idx in order:
        fh, fw = frame_outer_shapes[idx]
        candidates = []
        for r0 in range(1, grid_h - fh):
            for c0 in range(right_start, grid_w - fw):
                if _can_place(occupied, r0, c0, fh, fw, pad=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        _draw_frame(grid, r0, c0, outer_shapes[idx])
        _mark_occupied(occupied, r0, c0, fh, fw, pad=1)

    if len(placements) != layer_count:
        return None

    gray_components = _components(grid == FRAME)
    if len(gray_components) != layer_count:
        return None
    if main(grid).shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-layer stamp sample")
