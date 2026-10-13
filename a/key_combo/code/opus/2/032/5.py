from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 5
BODY_COLORS = [1, 3, 4, 6]
MARKER_COLORS = [2, 7, 8, 9]
SHAPES = [(3, 3), (3, 4)]
MAX_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def find_components(grid: np.ndarray, *, bg: int, connectivity: int = 4) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    if int(connectivity) == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    out: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == int(bg):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            out.append({"coords": coords, "bbox": bbox_coords(coords), "area": len(coords)})
    return out


def _touched_edge(box: BBox, shape: tuple[int, int]) -> str:
    H, W = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    edges = []
    if int(r0) == 0:
        edges.append("top")
    if int(r1) == H - 1:
        edges.append("bottom")
    if int(c0) == 0:
        edges.append("left")
    if int(c1) == W - 1:
        edges.append("right")
    assert len(edges) == 1, f"expected exactly one touched edge, got {edges}"
    return edges[0]


def _marker_color(patch: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in patch.flatten() if int(v) != int(bg))
    singletons = [color for color, count in counts.items() if int(count) == 1]
    assert len(singletons) == 1, "expected exactly one embedded marker color"
    return int(singletons[0])


def _path_slice(box: BBox, target_r: int, target_c: int) -> tuple[slice, slice]:
    r0, c0, r1, c1 = map(int, box)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if int(target_r) > int(r0):
        return slice(r1 + 1, target_r), slice(c0, c1 + 1)
    if int(target_r) < int(r0):
        return slice(target_r + h, r0), slice(c0, c1 + 1)
    if int(target_c) > int(c0):
        return slice(r0, r1 + 1), slice(c1 + 1, target_c)
    return slice(r0, r1 + 1), slice(target_c + w, c0)


def _target_origin(box: BBox, shape: tuple[int, int], edge: str) -> Coord:
    H, W = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if edge == "top":
        return H - h, c0
    if edge == "bottom":
        return 0, c0
    if edge == "left":
        return r0, W - w
    return r0, 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each edge-touching marked block to the opposite edge, rotate it 180 degrees, and color its trail with the marker color.

    Input structure:
    - The background is the dominant gray color.
    - Foreground consists of several small multicolor rectangular blocks, each touching exactly one outer grid edge.
    - Every block has one repeated body color plus exactly one different embedded marker pixel.

    Rule:
    - Determine the touched edge of each block and slide the whole block straight toward the opposite edge.
    - Remove the source block, place the moved block at the opposite edge, and rotate its full bbox patch by 180 degrees.
    - Fill the strip strictly between source and destination with the block's embedded marker color.

    Key insight:
    - The touched border gives the travel direction, while the unique embedded marker simultaneously identifies the trail color and the pixel that visibly changes position under the 180-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4)
    assert objects, "expected at least one block"

    out = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for obj in objects:
        box = obj["bbox"]
        patch = crop_bbox(g, box)
        edge = _touched_edge(box, g.shape)
        marker = _marker_color(patch, bg)
        target_r, target_c = _target_origin(box, g.shape, edge)
        path_rs, path_cs = _path_slice(box, target_r, target_c)
        rotated = np.rot90(patch, 2)

        if path_rs.start < path_rs.stop and path_cs.start < path_cs.stop:
            assert not bool(occupied[path_rs, path_cs].any()), "paths must not overlap"
            out[path_rs, path_cs] = int(marker)
            occupied[path_rs, path_cs] = True

        h, w = rotated.shape
        target_mask = rotated != int(bg)
        target_occ = occupied[target_r : target_r + h, target_c : target_c + w]
        assert not bool((target_occ & target_mask).any()), "targets must not overlap"
        paste(out, rotated, target_r, target_c, transparent=int(bg))
        target_occ[target_mask] = True

    return out


def _marker_positions(h: int, w: int) -> list[Coord]:
    coords = [(r, c) for r in range(h) for c in range(w)]
    if (h, w) == (3, 3):
        coords.remove((1, 1))
    return coords


def _make_patch(h: int, w: int, body: int, marker: int) -> np.ndarray:
    patch = np.full((h, w), int(body), dtype=int)
    rr, cc = random.choice(_marker_positions(h, w))
    patch[rr, cc] = int(marker)
    return patch


def _choose_disjoint_starts(length: int, widths: list[int]) -> tuple[int, int] | None:
    starts_a = list(range(1, length - widths[0]))
    starts_b = list(range(1, length - widths[1]))
    pairs = [
        (a, b)
        for a in starts_a
        for b in starts_b
        if a + widths[0] < b or b + widths[1] < a
    ]
    if not pairs:
        return None
    return random.choice(pairs)


def _build_vertical_case() -> np.ndarray | None:
    (h0, w0), (h1, w1) = random.choice(SHAPES), random.choice(SHAPES)
    H = random.randint(9, 20)
    W = random.randint(max(12, w0 + w1 + 4), 24)
    starts = _choose_disjoint_starts(W, [w0, w1])
    if starts is None:
        return None

    body0, body1 = random.sample(BODY_COLORS, 2)
    marker0, marker1 = random.sample(MARKER_COLORS, 2)
    c0, c1 = starts
    grid = np.full((H, W), BG, dtype=int)
    paste(grid, _make_patch(h0, w0, body0, marker0), 0, c0)
    paste(grid, _make_patch(h1, w1, body1, marker1), H - h1, c1)
    return grid


def _build_horizontal_case() -> np.ndarray | None:
    (h0, w0), (h1, w1) = random.choice(SHAPES), random.choice(SHAPES)
    H = random.randint(12, 24)
    W = random.randint(9, 20)
    starts = _choose_disjoint_starts(H, [h0, h1])
    if starts is None:
        return None

    body0, body1 = random.sample(BODY_COLORS, 2)
    marker0, marker1 = random.sample(MARKER_COLORS, 2)
    r0, r1 = starts
    grid = np.full((H, W), BG, dtype=int)
    paste(grid, _make_patch(h0, w0, body0, marker0), r0, 0)
    paste(grid, _make_patch(h1, w1, body1, marker1), r1, W - w1)
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    bg = get_bg(grid)
    objects = find_components(grid, bg=bg, connectivity=4)
    if len(objects) != 2:
        return False
    edges = {_touched_edge(obj["bbox"], grid.shape) for obj in objects}
    if edges not in ({"top", "bottom"}, {"left", "right"}):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    # Under the unique-edge and non-crossing-strip constraints, the always-valid
    # family consists of opposite-edge pairs with a shared motion orientation.
    for _ in range(MAX_RETRIES):
        if random.random() < 0.5:
            grid = _build_vertical_case()
        else:
            grid = _build_horizontal_case()
        if grid is not None and _valid_input(grid):
            return grid
    raise RuntimeError("generate_input failed")
