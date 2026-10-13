import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
PALETTE = tuple(color for color in range(10) if color not in {BLACK, GREY, AZURE})
CORNERS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 300
QUADRANT_SIZE_RANGE = (8, 14)
FRAME_OUTER_MIN = 4
FRAME_OUTER_MAX = 9
OBJECT_MARGIN = 1


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rectangle_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not np.any(mask):
        return None
    comps = _components(mask)
    if len(comps) != 1:
        return None
    r0, c0, r1, c1 = _bbox(comps[0])
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    if np.array_equal(mask, expected):
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _parse_frame(quadrant: np.ndarray, row_off: int, col_off: int) -> dict:
    frame_mask = quadrant == AZURE
    bbox = _rectangle_frame_bbox(frame_mask)
    assert bbox is not None, "expected one azure rectangular frame"
    r0, c0, r1, c1 = bbox

    candidates = {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
        "br": (r1 + 1, c1 + 1),
    }
    marker_hits: list[tuple[str, int, int, int]] = []
    for corner, (mr, mc) in candidates.items():
        if 0 <= mr < quadrant.shape[0] and 0 <= mc < quadrant.shape[1]:
            color = int(quadrant[mr, mc])
            if color not in {BLACK, GREY, AZURE}:
                marker_hits.append((corner, mr, mc, color))

    assert len(marker_hits) == 1, "expected exactly one frame marker"
    corner, mr, mc, marker_color = marker_hits[0]

    other_foreground = np.argwhere((quadrant != BLACK) & (quadrant != AZURE))
    assert len(other_foreground) == 1
    assert tuple(map(int, other_foreground[0])) == (mr, mc)

    return {
        "kind": "frame",
        "corner": corner,
        "marker_color": marker_color,
        "bbox": (row_off + r0, col_off + c0, row_off + r1, col_off + c1),
    }


def _parse_object(quadrant: np.ndarray, row_off: int, col_off: int) -> dict:
    object_mask = (quadrant != BLACK) & (quadrant != GREY) & (quadrant != AZURE)
    comps = _components(object_mask)
    assert len(comps) == 1, "expected exactly one colored object"
    coords = comps[0]
    colors = {int(quadrant[r, c]) for r, c in coords}
    assert len(colors) == 1, "object must be monochrome"
    color = next(iter(colors))
    r0, c0, r1, c1 = _bbox(coords)
    shape = quadrant[r0 : r1 + 1, c0 : c1 + 1] == color

    return {
        "kind": "object",
        "color": color,
        "bbox": (row_off + r0, col_off + c0, row_off + r1, col_off + c1),
        "shape": shape,
        "coords": [(row_off + r, col_off + c) for r, c in coords],
    }


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored object to the frame with the same-colored corner marker,
    scale the object by the largest integer factor that fits the frame interior,
    and anchor it to the marker's corner inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(row_lines) == 1 and len(col_lines) == 1, "expected one full grey row and column"
    split_r = row_lines[0]
    split_c = col_lines[0]

    quadrants = {
        "tl": (0, split_r, 0, split_c),
        "tr": (0, split_r, split_c + 1, g.shape[1]),
        "bl": (split_r + 1, g.shape[0], 0, split_c),
        "br": (split_r + 1, g.shape[0], split_c + 1, g.shape[1]),
    }

    frames: list[dict] = []
    objects: list[dict] = []
    for r0, r1, c0, c1 in quadrants.values():
        quad = g[r0:r1, c0:c1]
        if np.any(quad == AZURE):
            frames.append(_parse_frame(quad, r0, c0))
        else:
            objects.append(_parse_object(quad, r0, c0))

    assert len(frames) == 2 and len(objects) == 2
    frame_by_color = {frame["marker_color"]: frame for frame in frames}
    assert len(frame_by_color) == 2
    assert {obj["color"] for obj in objects} == set(frame_by_color)

    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BLACK

    for obj in objects:
        frame = frame_by_color[obj["color"]]
        fr0, fc0, fr1, fc1 = frame["bbox"]
        inner_r0, inner_c0 = fr0 + 1, fc0 + 1
        inner_r1, inner_c1 = fr1 - 1, fc1 - 1
        frame_h = inner_r1 - inner_r0 + 1
        frame_w = inner_c1 - inner_c0 + 1
        obj_h, obj_w = obj["shape"].shape
        factor = min(frame_h // obj_h, frame_w // obj_w)
        assert factor >= 1, "object must fit its target frame"

        scaled = _scale_mask(obj["shape"], factor)
        scaled_h, scaled_w = scaled.shape
        corner = frame["corner"]
        top = inner_r0 if "t" in corner else inner_r1 - scaled_h + 1
        left = inner_c0 if "l" in corner else inner_c1 - scaled_w + 1
        patch = out[top : top + scaled_h, left : left + scaled_w]
        patch[scaled] = obj["color"]

    return out


def _random_shape(max_h: int, max_w: int) -> np.ndarray:
    mode = random.choice(("blob", "line", "rect", "el", "tee"))

    if mode == "line":
        if max_h == 1 or (max_w > 1 and random.random() < 0.5):
            length = random.randint(1, min(5, max_w))
            return np.ones((1, length), dtype=bool)
        length = random.randint(1, min(5, max_h))
        return np.ones((length, 1), dtype=bool)

    if mode == "rect":
        h = random.randint(1, min(4, max_h))
        w = random.randint(1, min(4, max_w))
        return np.ones((h, w), dtype=bool)

    if mode == "el" and max_h >= 2 and max_w >= 2:
        h = random.randint(2, min(5, max_h))
        w = random.randint(2, min(5, max_w))
        mask = np.zeros((h, w), dtype=bool)
        orientation = random.choice(CORNERS)
        if orientation == "tl":
            mask[0, :] = True
            mask[:, 0] = True
        elif orientation == "tr":
            mask[0, :] = True
            mask[:, -1] = True
        elif orientation == "bl":
            mask[-1, :] = True
            mask[:, 0] = True
        else:
            mask[-1, :] = True
            mask[:, -1] = True
        return mask

    if mode == "tee" and max_h >= 3 and max_w >= 3:
        h = random.randint(3, min(5, max_h))
        w = random.randint(3, min(5, max_w))
        mask = np.zeros((h, w), dtype=bool)
        orientation = random.choice(CORNERS)
        if orientation == "tl":
            mask[0, :] = True
            mask[:, 0] = True
            mask[:, w // 2] = True
        elif orientation == "tr":
            mask[0, :] = True
            mask[:, -1] = True
            mask[:, w // 2] = True
        elif orientation == "bl":
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, w // 2] = True
        else:
            mask[-1, :] = True
            mask[:, -1] = True
            mask[:, w // 2] = True
        return _tight_mask(mask)

    h = random.randint(1, min(5, max_h))
    w = random.randint(1, min(5, max_w))
    area_cap = h * w
    target = random.randint(1, min(area_cap, max(1, h + w + 2)))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = {start}
    while np.count_nonzero(mask) < target:
        candidates: list[tuple[int, int]] = []
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c, h, w):
                if not mask[nr, nc]:
                    candidates.append((nr, nc))
        if not candidates:
            break
        chosen = random.choice(candidates)
        mask[chosen] = True
        frontier.add(chosen)
    return _tight_mask(mask)


def _draw_frame(
    grid: np.ndarray,
    quad_bounds: tuple[int, int, int, int],
    marker_color: int,
) -> dict | None:
    qr0, qr1, qc0, qc1 = quad_bounds
    qh = qr1 - qr0
    qw = qc1 - qc0
    max_outer_h = min(FRAME_OUTER_MAX, qh - 2)
    max_outer_w = min(FRAME_OUTER_MAX, qw - 2)
    if max_outer_h < FRAME_OUTER_MIN or max_outer_w < FRAME_OUTER_MIN:
        return None

    outer_h = random.randint(FRAME_OUTER_MIN, max_outer_h)
    outer_w = random.randint(FRAME_OUTER_MIN, max_outer_w)
    local_r0 = random.randint(1, qh - outer_h - 1)
    local_c0 = random.randint(1, qw - outer_w - 1)
    local_r1 = local_r0 + outer_h - 1
    local_c1 = local_c0 + outer_w - 1

    abs_r0 = qr0 + local_r0
    abs_c0 = qc0 + local_c0
    abs_r1 = qr0 + local_r1
    abs_c1 = qc0 + local_c1

    grid[abs_r0, abs_c0 : abs_c1 + 1] = AZURE
    grid[abs_r1, abs_c0 : abs_c1 + 1] = AZURE
    grid[abs_r0 : abs_r1 + 1, abs_c0] = AZURE
    grid[abs_r0 : abs_r1 + 1, abs_c1] = AZURE

    corner = random.choice(CORNERS)
    marker_positions = {
        "tl": (abs_r0 - 1, abs_c0 - 1),
        "tr": (abs_r0 - 1, abs_c1 + 1),
        "bl": (abs_r1 + 1, abs_c0 - 1),
        "br": (abs_r1 + 1, abs_c1 + 1),
    }
    mr, mc = marker_positions[corner]
    grid[mr, mc] = marker_color

    return {
        "corner": corner,
        "marker_color": marker_color,
        "bbox": (abs_r0, abs_c0, abs_r1, abs_c1),
        "interior_h": outer_h - 2,
        "interior_w": outer_w - 2,
    }


def _place_object(
    grid: np.ndarray,
    quad_bounds: tuple[int, int, int, int],
    color: int,
    max_h: int,
    max_w: int,
) -> dict | None:
    qr0, qr1, qc0, qc1 = quad_bounds
    qh = qr1 - qr0
    qw = qc1 - qc0
    if max_h < 1 or max_w < 1 or qh < 2 * OBJECT_MARGIN + 1 or qw < 2 * OBJECT_MARGIN + 1:
        return None

    shape = _random_shape(max_h, max_w)
    obj_h, obj_w = shape.shape
    if obj_h > qh - 2 * OBJECT_MARGIN or obj_w > qw - 2 * OBJECT_MARGIN:
        return None

    top = random.randint(OBJECT_MARGIN, qh - obj_h - OBJECT_MARGIN)
    left = random.randint(OBJECT_MARGIN, qw - obj_w - OBJECT_MARGIN)
    patch = grid[qr0 + top : qr0 + top + obj_h, qc0 + left : qc0 + left + obj_w]
    patch[shape] = color

    return {"shape": shape, "bbox": (qr0 + top, qc0 + left, qr0 + top + obj_h - 1, qc0 + left + obj_w - 1)}


def _quadrant_bounds(grid: np.ndarray, split_r: int, split_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (0, split_r, 0, split_c),
        "tr": (0, split_r, split_c + 1, grid.shape[1]),
        "bl": (split_r + 1, grid.shape[0], 0, split_c),
        "br": (split_r + 1, grid.shape[0], split_c + 1, grid.shape[1]),
    }


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(*QUADRANT_SIZE_RANGE)
    bottom_h = random.randint(*QUADRANT_SIZE_RANGE)
    left_w = random.randint(*QUADRANT_SIZE_RANGE)
    right_w = random.randint(*QUADRANT_SIZE_RANGE)

    grid = np.zeros((top_h + 1 + bottom_h, left_w + 1 + right_w), dtype=int)
    split_r = top_h
    split_c = left_w
    grid[split_r, :] = GREY
    grid[:, split_c] = GREY

    bounds = _quadrant_bounds(grid, split_r, split_c)
    all_quads = list(bounds)
    frame_quads = random.sample(all_quads, 2)
    object_quads = [name for name in all_quads if name not in frame_quads]
    random.shuffle(object_quads)
    colors = random.sample(PALETTE, 2)

    frames: dict[int, dict] = {}
    for frame_quad, color in zip(frame_quads, colors):
        frame = _draw_frame(grid, bounds[frame_quad], color)
        if frame is None:
            return None
        frames[color] = frame

    for object_quad, color in zip(object_quads, colors):
        frame = frames[color]
        qr0, qr1, qc0, qc1 = bounds[object_quad]
        max_h = min(frame["interior_h"], qr1 - qr0 - 2 * OBJECT_MARGIN)
        max_w = min(frame["interior_w"], qc1 - qc0 - 2 * OBJECT_MARGIN)
        placed = _place_object(grid, bounds[object_quad], color, max_h, max_w)
        if placed is None:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
