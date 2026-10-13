from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _connected_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = _neighbors4 if connectivity == 4 else _neighbors8
    components: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            coords = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for nr, nc in neigh(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _bbox_from_coords(coords)


def _crop_true(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    crop = _crop_true(mask)
    return crop.shape, crop.tobytes()


def _hole_masks(local_mask: np.ndarray) -> list[np.ndarray]:
    bg = ~local_mask
    holes = []
    h, w = local_mask.shape
    for comp in _connected_components(bg, connectivity=4):
        rs = comp[:, 0]
        cs = comp[:, 1]
        if np.any(rs == 0) or np.any(rs == h - 1) or np.any(cs == 0) or np.any(cs == w - 1):
            continue
        hole = np.zeros_like(local_mask, dtype=bool)
        hole[rs, cs] = True
        holes.append(hole)
    return holes


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert colors, "expected at least one foreground object"

    objects = []
    for color in colors:
        mask = g == color
        comps = _connected_components(mask, connectivity=8)
        assert len(comps) == 1, "each object color must form one 8-connected component"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox_from_coords(coords)

        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local = coords - np.array([r0, c0], dtype=int)
        local_mask[local[:, 0], local[:, 1]] = True

        holes = _hole_masks(local_mask)
        assert len(holes) <= 1, "each object may have at most one enclosed hole"
        hole_mask = holes[0] if holes else None

        silhouette = local_mask.copy()
        if hole_mask is not None:
            silhouette |= hole_mask

        patch = np.full(local_mask.shape, bg, dtype=int)
        patch[local_mask] = color

        objects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "silhouette_key": _shape_key(silhouette),
                "hole_key": None if hole_mask is None else _shape_key(hole_mask),
                "hole_bbox": None if hole_mask is None else _bbox_from_mask(hole_mask),
            }
        )

    return bg, objects


def _build_chain(objects: list[dict]) -> tuple[int, dict[int, int]]:
    n = len(objects)
    key_to_idx: dict[tuple[tuple[int, int], bytes], int] = {}
    for idx, obj in enumerate(objects):
        key = obj["silhouette_key"]
        assert key not in key_to_idx, "solid silhouettes must be unique"
        key_to_idx[key] = idx

    parent_to_child: dict[int, int] = {}
    indegree = [0] * n
    for idx, obj in enumerate(objects):
        hole_key = obj["hole_key"]
        if hole_key is None:
            continue
        assert hole_key in key_to_idx, "every hole must match exactly one other silhouette"
        child = key_to_idx[hole_key]
        assert child != idx, "an object cannot contain itself"
        parent_to_child[idx] = child
        indegree[child] += 1
        assert indegree[child] <= 1, "a child silhouette cannot fit multiple parents"

    assert len(parent_to_child) == max(0, n - 1), "valid nesting requires one chain edge per holed object"
    roots = [idx for idx, deg in enumerate(indegree) if deg == 0]
    tails = [idx for idx in range(n) if idx not in parent_to_child]
    assert len(roots) == 1, "expected exactly one outermost object"
    assert len(tails) == 1, "expected exactly one solid tail object"

    outer = roots[0]
    seen = set()
    cur = outer
    while True:
        assert cur not in seen, "containment relation must be acyclic"
        seen.add(cur)
        if cur not in parent_to_child:
            break
        cur = parent_to_child[cur]
    assert len(seen) == n, "containment relation must be a single chain"
    return outer, parent_to_child


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    assert view.shape == patch.shape
    mask = patch != bg
    assert not np.any((view != bg) & mask), "assembled objects must only overlap holes"
    view[mask] = patch[mask]


def _assemble(objects: list[dict], outer: int, parent_to_child: dict[int, int], bg: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)

    def place(idx: int, r0: int, c0: int) -> None:
        obj = objects[idx]
        _paste_patch(out, obj["patch"], r0, c0, bg)
        child = parent_to_child.get(idx)
        if child is None:
            return
        hole_r0, hole_c0, _hole_r1, _hole_c1 = obj["hole_bbox"]
        place(child, r0 + hole_r0, c0 + hole_c0)

    outer_r0, outer_c0, _outer_r1, _outer_c1 = objects[outer]["bbox"]
    place(outer, outer_r0, outer_c0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered holed objects into one nested doll and leave only the outer shell in place.

    Input structure:
    - The grid has one background color and several uniquely colored monochrome objects, each color forming one 8-connected component.
    - Every object except the chain tail contains exactly one enclosed 4-connected background hole; the tail is solid.
    - Each hole shape matches, up to translation only, exactly one other object's solid silhouette obtained by filling that object's own hole.

    Rule:
    - Recover the unique containment chain by matching every holed object's hole shape to the filled silhouette of its child object.
    - Identify the object that is not fitted into any other hole as the outermost shell and keep its current position.
    - Recursively translate each child so its filled silhouette exactly occupies the current hole, then continue with that child's own hole until the solid tail fills the last cavity.
    - Clear the original locations of every non-outer object; the output keeps only the assembled nested structure at the outer object's position.

    Key insight:
    - The decisive signal is the hidden correspondence between negative space and filled silhouettes: a hole never matches an object's visible annulus directly, it matches that object's hole-filled outline.
    """
    bg, objects = _extract_objects(grid)
    outer, parent_to_child = _build_chain(objects)
    return _assemble(objects, outer, parent_to_child, bg, np.asarray(grid).shape)


GEN_BG = 0
GEN_MAX_CANVAS = 30
GEN_PLACEMENT_GAP = 2
GEN_RETRIES = 400


def _random_connected_shape(min_cells: int = 3, max_cells: int = 7) -> np.ndarray | None:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    target = random.randint(min_cells, min(max_cells, h * w))

    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < target:
        frontier = []
        filled = np.argwhere(mask)
        for r, c in filled:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True

    crop = _crop_true(mask)
    if min(crop.shape) < 2:
        return None
    if len(_connected_components(crop, connectivity=4)) != 1:
        return None
    if _hole_masks(crop):
        return None
    return crop


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _random_wrapper(child: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    ch, cw = child.shape
    pad = 3
    canvas = np.zeros((ch + 2 * pad, cw + 2 * pad), dtype=bool)
    child_mask = np.zeros_like(canvas, dtype=bool)
    child_mask[pad : pad + ch, pad : pad + cw] = child

    silhouette = _dilate4(child_mask)
    extra_steps = random.randint(max(1, (ch + cw) // 3), max(2, ch + cw))
    for _ in range(extra_steps):
        frontier = []
        for r, c in np.argwhere(silhouette):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and not silhouette[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        silhouette[rr, cc] = True

    sil_r0, sil_c0, sil_r1, sil_c1 = _bbox_from_mask(silhouette)
    silhouette = silhouette[sil_r0 : sil_r1 + 1, sil_c0 : sil_c1 + 1].copy()
    child_mask = child_mask[sil_r0 : sil_r1 + 1, sil_c0 : sil_c1 + 1].copy()

    actual = silhouette & ~child_mask
    if len(_connected_components(silhouette, connectivity=4)) != 1:
        return None
    if len(_connected_components(actual, connectivity=8)) != 1:
        return None

    holes = _hole_masks(actual)
    if len(holes) != 1:
        return None
    if _shape_key(holes[0]) != _shape_key(child_mask):
        return None

    hole_r0, hole_c0, _hole_r1, _hole_c1 = _bbox_from_mask(child_mask)
    return actual, (hole_r0, hole_c0)


def _make_chain() -> tuple[list[np.ndarray], list[tuple[int, int] | None]] | None:
    count = random.randint(2, 5)
    tail = _random_connected_shape()
    if tail is None:
        return None

    patches = [tail]
    offsets: list[tuple[int, int] | None] = [None]
    current_silhouette = tail

    for _ in range(1, count):
        wrapped = _random_wrapper(current_silhouette)
        if wrapped is None:
            return None
        patch, offset = wrapped
        if max(patch.shape) > 16:
            return None
        patches.append(patch)
        offsets.append(offset)
        current_silhouette = patch | np.pad(
            current_silhouette,
            (
                (offset[0], patch.shape[0] - offset[0] - current_silhouette.shape[0]),
                (offset[1], patch.shape[1] - offset[1] - current_silhouette.shape[1]),
            ),
            constant_values=False,
        )

    return patches, offsets


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, GEN_BG, dtype=int)
    patch[mask] = color
    return patch


def _place_patches(patches: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    canvas = np.full((GEN_MAX_CANVAS, GEN_MAX_CANVAS), GEN_BG, dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    order = list(range(len(patches)))
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    for idx in order:
        ph, pw = patches[idx].shape
        locations = []
        for r0 in range(GEN_MAX_CANVAS - ph + 1):
            for c0 in range(GEN_MAX_CANVAS - pw + 1):
                if np.any(occupied[r0 : r0 + ph, c0 : c0 + pw]):
                    continue
                locations.append((r0, c0))
        if not locations:
            return None
        r0, c0 = random.choice(locations)
        _paste_patch(canvas, patches[idx], r0, c0, GEN_BG)
        positions[idx] = (r0, c0)

        rr0 = max(0, r0 - GEN_PLACEMENT_GAP)
        cc0 = max(0, c0 - GEN_PLACEMENT_GAP)
        rr1 = min(GEN_MAX_CANVAS, r0 + ph + GEN_PLACEMENT_GAP)
        cc1 = min(GEN_MAX_CANVAS, c0 + pw + GEN_PLACEMENT_GAP)
        occupied[rr0:rr1, cc0:cc1] = True

    assert all(pos is not None for pos in positions)
    return canvas, positions


def _assemble_generated(
    patches: list[np.ndarray],
    offsets: list[tuple[int, int] | None],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((GEN_MAX_CANVAS, GEN_MAX_CANVAS), GEN_BG, dtype=int)

    def place(idx: int, r0: int, c0: int) -> None:
        _paste_patch(out, patches[idx], r0, c0, GEN_BG)
        if idx == 0:
            return
        hole_r0, hole_c0 = offsets[idx]
        place(idx - 1, r0 + hole_r0, c0 + hole_c0)

    outer = len(patches) - 1
    outer_r0, outer_c0 = positions[outer]
    place(outer, outer_r0, outer_c0)
    return out


def _gen_attempt() -> np.ndarray | None:
    chain = _make_chain()
    if chain is None:
        return None
    masks, offsets = chain

    colors = random.sample(list(range(1, 10)), len(masks))
    colored = [_colorize(mask, color) for mask, color in zip(masks, colors)]

    placed = _place_patches(colored)
    if placed is None:
        return None
    inp_full, positions = placed
    expected_full = _assemble_generated(colored, offsets, positions)

    coords = np.argwhere(inp_full != GEN_BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = random.randint(0, min(3, int(r0)))
    left_margin = random.randint(0, min(3, int(c0)))
    bottom_margin = random.randint(0, min(3, int(inp_full.shape[0] - 1 - r1)))
    right_margin = random.randint(0, min(3, int(inp_full.shape[1] - 1 - c1)))
    rs = slice(r0 - top_margin, r1 + bottom_margin + 1)
    cs = slice(c0 - left_margin, c1 + right_margin + 1)
    inp = inp_full[rs, cs].copy()
    expected = expected_full[rs, cs].copy()

    if _get_bg(inp) != GEN_BG:
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-assembly input")
