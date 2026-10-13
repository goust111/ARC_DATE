from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG_COLOR = 7
BLUEPRINT_COLOR = 5
FRAGMENT_COLORS = [0, 1, 2, 3, 4, 6, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, m.shape)), m.tobytes()


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = [_mask_key(np.rot90(mask, k)) for k in range(4)]
    return min(keys)


def _infer_background(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    border = np.concatenate([g[0], g[-1], g[1:-1, 0], g[1:-1, -1]])
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[np.argmax(counts)])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(g == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _extract_holes(subgrid: np.ndarray, comp_mask: np.ndarray, bg: int) -> list[dict]:
    holes: list[dict] = []
    h, w = comp_mask.shape
    for coords in _components(~comp_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        assert np.all(subgrid[coords[:, 0], coords[:, 1]] == bg)
        hole_mask, (r0, c0) = _coords_to_mask(coords)
        holes.append(
            {
                "coords": coords,
                "mask": hole_mask,
                "top_left": (int(r0), int(c0)),
                "key": _rotation_key(hole_mask),
            }
        )
    return holes


def _find_blueprint(grid: np.ndarray, bg: int) -> tuple[dict, list[dict], np.ndarray]:
    candidates: list[tuple[dict, list[dict], np.ndarray]] = []
    for comp in _monochrome_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]
        comp_mask = np.zeros(subgrid.shape, dtype=bool)
        rel = comp["coords"] - np.array([r0, c0], dtype=int)
        comp_mask[rel[:, 0], rel[:, 1]] = True
        holes = _extract_holes(subgrid, comp_mask, bg)
        if holes:
            candidates.append((comp, holes, subgrid))
    assert candidates, "expected a blueprint object with enclosed holes"
    max_area = max(len(comp["coords"]) for comp, _holes, _sub in candidates)
    finalists = [item for item in candidates if len(item[0]["coords"]) == max_area]
    assert len(finalists) == 1, "expected a unique blueprint object"
    return finalists[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the blueprint object and fill each enclosed hole with the color of the
    uniquely shape-matching outside fragment, allowing rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _infer_background(g)
    blueprint, holes, subgrid = _find_blueprint(g, bg)

    fragment_colors: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in _monochrome_components(g, bg):
        if comp["bbox"] == blueprint["bbox"] and np.array_equal(comp["coords"], blueprint["coords"]):
            continue
        frag_mask, _ = _coords_to_mask(comp["coords"])
        key = _rotation_key(frag_mask)
        assert key not in fragment_colors, "expected unique fragment shapes up to rotation"
        fragment_colors[key] = int(comp["color"])

    assert len(fragment_colors) == len(holes), "expected one fragment per hole"

    out = subgrid.copy()
    for hole in holes:
        color = fragment_colors.get(hole["key"])
        assert color is not None, "every hole must have a matching fragment"
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def _random_connected_mask() -> np.ndarray | None:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    max_area = min(h * w, 6)
    if max_area < 3:
        return None
    target_area = random.randint(3, max_area)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in NEIGHBORS4:
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != target_area:
        return None

    coords = np.asarray(sorted(cells), dtype=int)
    mask, _ = _coords_to_mask(coords)
    return mask


def _sample_unique_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    keys: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(200):
        if len(shapes) == count:
            return shapes
        mask = _random_connected_mask()
        if mask is None:
            continue
        key = _rotation_key(mask)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(mask)
    return None


def _random_oriented(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, random.randrange(4))


def _place_bbox_nonoverlap(
    blocked: np.ndarray,
    patch_shape: tuple[int, int],
    *,
    margin: int = 1,
    keep_off_border: bool = True,
) -> tuple[int, int] | None:
    h, w = patch_shape
    H, W = blocked.shape
    r_start = margin if keep_off_border else 0
    c_start = margin if keep_off_border else 0
    r_end = H - h - margin + 1 if keep_off_border else H - h + 1
    c_end = W - w - margin + 1 if keep_off_border else W - w + 1
    if r_end <= r_start or c_end <= c_start:
        return None

    positions = []
    for r0 in range(r_start, r_end):
        for c0 in range(c_start, c_end):
            rr0 = max(0, r0 - margin)
            cc0 = max(0, c0 - margin)
            rr1 = min(H, r0 + h + margin)
            cc1 = min(W, c0 + w + margin)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    if not positions:
        return None
    return random.choice(positions)


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, patch_shape: tuple[int, int], margin: int = 1) -> None:
    h, w = patch_shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(blocked.shape[0], r0 + h + margin)
    cc1 = min(blocked.shape[1], c0 + w + margin)
    blocked[rr0:rr1, cc0:cc1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = int(color)


def _build_blueprint(
    height: int,
    width: int,
    hole_shapes: Iterable[np.ndarray],
) -> tuple[np.ndarray, list[dict]] | None:
    patch = np.full((height, width), BLUEPRINT_COLOR, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    holes: list[dict] = []

    for base_shape in hole_shapes:
        hole_mask = _random_oriented(base_shape)
        pos = _place_bbox_nonoverlap(blocked, hole_mask.shape, margin=1, keep_off_border=True)
        if pos is None:
            return None
        r0, c0 = pos
        _paint_mask(patch, hole_mask, r0, c0, BG_COLOR)
        _mark_blocked(blocked, r0, c0, hole_mask.shape, margin=1)
        coords = np.argwhere(hole_mask) + np.array([r0, c0], dtype=int)
        holes.append(
            {
                "mask": hole_mask,
                "coords": coords,
                "key": _rotation_key(hole_mask),
            }
        )

    return patch, holes


def _expected_output(blueprint_patch: np.ndarray, holes: list[dict], colors: list[int]) -> np.ndarray:
    out = blueprint_patch.copy()
    for hole, color in zip(holes, colors):
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _place_fragments(
    grid: np.ndarray,
    blueprint_bbox: tuple[int, int, int, int],
    base_shapes: list[np.ndarray],
    colors: list[int],
) -> bool:
    H, W = grid.shape
    blocked = np.zeros((H, W), dtype=bool)
    br0, bc0, br1, bc1 = blueprint_bbox
    blocked[max(0, br0 - 1) : min(H, br1 + 2), max(0, bc0 - 1) : min(W, bc1 + 2)] = True

    order = list(range(len(base_shapes)))
    random.shuffle(order)
    for idx in order:
        frag_mask = _random_oriented(base_shapes[idx])
        pos = _place_bbox_nonoverlap(blocked, frag_mask.shape, margin=1, keep_off_border=True)
        if pos is None:
            return False
        r0, c0 = pos
        _paint_mask(grid, frag_mask, r0, c0, colors[idx])
        _mark_blocked(blocked, r0, c0, frag_mask.shape, margin=1)
    return True


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.randint(2, 5)
    shapes = _sample_unique_shapes(hole_count)
    if shapes is None:
        return None

    min_side = max(9, hole_count * 2 + 5)
    blueprint_h = random.randint(min_side, min_side + 5)
    blueprint_w = random.randint(min_side, min_side + 6)

    built = _build_blueprint(blueprint_h, blueprint_w, shapes)
    if built is None:
        return None
    blueprint_patch, holes = built

    colors = random.sample(FRAGMENT_COLORS, hole_count)
    expected = _expected_output(blueprint_patch, holes, colors)

    canvas_h = random.randint(blueprint_h + 6, blueprint_h + 14)
    canvas_w = random.randint(blueprint_w + 6, blueprint_w + 14)
    grid = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)

    br0 = random.randint(1, canvas_h - blueprint_h - 1)
    bc0 = random.randint(1, canvas_w - blueprint_w - 1)
    grid[br0 : br0 + blueprint_h, bc0 : bc0 + blueprint_w] = blueprint_patch
    blueprint_bbox = (br0, bc0, br0 + blueprint_h - 1, bc0 + blueprint_w - 1)

    if not _place_fragments(grid, blueprint_bbox, shapes, colors):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
