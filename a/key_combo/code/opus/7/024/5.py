from __future__ import annotations

from collections import deque
import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

TOP_COLOR = RED
RIGHT_COLOR = BLUE
BOTTOM_COLOR = GREEN
LEFT_COLOR = YELLOW
FRAME_SIDE_COLORS = (TOP_COLOR, RIGHT_COLOR, BOTTOM_COLOR, LEFT_COLOR)

GRID_RANGE = (15, 19)
INTERIOR_RANGE = (7, 9)
GEN_ATTEMPTS = 800

_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_cells(
    cells: tuple[tuple[int, int], ...], rotate_k: int, flip: bool
) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    for _ in range(rotate_k % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    return _normalize_shape(set(pts))


def _shape_has_turn_or_branch(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    for r, c in cell_set:
        nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in cell_set]
        if len(nbrs) >= 3:
            return True
        if len(nbrs) == 2:
            dr1, dc1 = nbrs[0][0] - r, nbrs[0][1] - c
            dr2, dc2 = nbrs[1][0] - r, nbrs[1][1] - c
            if dr1 != -dr2 or dc1 != -dc2:
                return True
    return False


def _build_sprite_library() -> list[tuple[tuple[int, int], ...]]:
    base_shapes = [
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)),
    ]
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for shape in base_shapes:
        for k in range(4):
            for flip in (False, True):
                transformed = _transform_cells(shape, k, flip)
                arr = np.array(transformed, dtype=int)
                if len(transformed) < 3 or len(transformed) > 6:
                    continue
                if arr[:, 0].ptp() < 1 or arr[:, 1].ptp() < 1:
                    continue
                if not _shape_has_turn_or_branch(transformed):
                    continue
                shapes.add(transformed)
    return sorted(shapes)


SPRITE_LIBRARY = _build_sprite_library()


def _draw_segment(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        canvas[r0, lo : hi + 1] = True
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        canvas[lo : hi + 1, c0] = True
        return
    lo, hi = sorted((c0, c1))
    canvas[r0, lo : hi + 1] = True
    lo, hi = sorted((r0, r1))
    canvas[lo : hi + 1, c1] = True


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not stroke[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not stroke[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not stroke[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))
    holes = (~stroke) & (~reachable)
    return stroke | holes


def _extreme_anchors(sprite: np.ndarray) -> list[tuple[float, float]]:
    coords = np.argwhere(sprite)
    rows = coords[:, 0]
    cols = coords[:, 1]
    top_r = int(rows.min())
    bottom_r = int(rows.max())
    left_c = int(cols.min())
    right_c = int(cols.max())
    top_cols = cols[rows == top_r]
    bottom_cols = cols[rows == bottom_r]
    left_rows = rows[cols == left_c]
    right_rows = rows[cols == right_c]
    return [
        (float(np.mean(top_cols)), float(top_r)),
        (float(right_c), float(np.mean(right_rows))),
        (float(np.mean(bottom_cols)), float(bottom_r)),
        (float(left_c), float(np.mean(left_rows))),
    ]


def _homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> np.ndarray:
    a_rows = []
    b_vals = []
    for (x, y), (u, v) in zip(src, dst):
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        b_vals.append(u)
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b_vals.append(v)
    params = np.linalg.solve(np.array(a_rows, dtype=float), np.array(b_vals, dtype=float))
    return np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=float,
    )


def _apply_homography(hmat: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    vec = hmat @ np.array([x, y, 1.0], dtype=float)
    if abs(vec[2]) < 1e-9:
        raise AssertionError("degenerate point mapping")
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


def _map_sprite_pixels(
    sprite: np.ndarray, interior_shape: tuple[int, int]
) -> tuple[dict[tuple[int, int], tuple[int, int]], list[tuple[float, float]]]:
    ih, iw = interior_shape
    src = _extreme_anchors(sprite)
    dst = [
        ((iw - 1) / 2.0, 0.0),
        (float(iw - 1), (ih - 1) / 2.0),
        ((iw - 1) / 2.0, float(ih - 1)),
        (0.0, (ih - 1) / 2.0),
    ]
    hmat = _homography(src, dst)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in np.argwhere(sprite):
        x, y = _apply_homography(hmat, (float(c), float(r)))
        rr = int(np.clip(np.rint(y), 0, ih - 1))
        cc = int(np.clip(np.rint(x), 0, iw - 1))
        mapped[(int(r), int(c))] = (rr, cc)
    return mapped, dst


def _recolor_by_nearest_side(filled: np.ndarray) -> np.ndarray:
    ih, iw = filled.shape
    out = np.zeros((ih, iw), dtype=int)
    for r, c in np.argwhere(filled):
        top = r
        right = iw - 1 - c
        bottom = ih - 1 - r
        left = c
        distances = [top, right, bottom, left]
        nearest = min(distances)
        if distances.count(nearest) == 1:
            out[r, c] = FRAME_SIDE_COLORS[distances.index(nearest)]
        else:
            out[r, c] = GRAY
    return out


def _locate_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    if grid[r0, c0] != BLACK or grid[r0, c1] != BLACK:
                        continue
                    if grid[r1, c0] != BLACK or grid[r1, c1] != BLACK:
                        continue
                    if not np.all(grid[r0, c0 + 1 : c1] == TOP_COLOR):
                        continue
                    if not np.all(grid[r1, c0 + 1 : c1] == BOTTOM_COLOR):
                        continue
                    if not np.all(grid[r0 + 1 : r1, c0] == LEFT_COLOR):
                        continue
                    if not np.all(grid[r0 + 1 : r1, c1] == RIGHT_COLOR):
                        continue
                    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
                        continue
                    candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected a unique four-colored hollow frame"
    return candidates[0]


def _validate_sprite(coords: np.ndarray) -> None:
    assert 3 <= len(coords) <= 6, "sprite must have 3-6 pixels"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 > r0 and c1 > c0, "sprite bbox must be at least 2x2"
    centered = coords - coords.mean(axis=0, keepdims=True)
    rank = np.linalg.matrix_rank(centered)
    assert rank >= 2, "sprite pixels must not be collinear"
    normalized = _normalize_shape({(int(r - r0), int(c - c0)) for r, c in coords})
    assert _shape_has_turn_or_branch(normalized), "sprite needs a turn or branch"


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    frame = _locate_frame(g)
    fr0, fc0, fr1, fc1 = frame
    gray_mask = g == GRAY
    sprite_mask = gray_mask.copy()
    sprite_mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = False
    sprite_mask[fr0: fr1 + 1, fc0: fc1 + 1] &= False

    comps = _find_components(sprite_mask)
    assert len(comps) == 1, "expected a single external gray sprite"
    sprite_coords = comps[0]
    _validate_sprite(sprite_coords)

    sr0, sc0, sr1, sc1 = _bbox(sprite_coords)
    sprite = g[sr0 : sr1 + 1, sc0 : sc1 + 1] == GRAY

    ih = fr1 - fr0 - 1
    iw = fc1 - fc0 - 1
    assert ih >= 2 and iw >= 2

    mapped, dst_anchors = _map_sprite_pixels(sprite, (ih, iw))
    stroke = np.zeros((ih, iw), dtype=bool)
    for rr, cc in mapped.values():
        stroke[rr, cc] = True

    seed_count = int(stroke.sum())
    coords_set = {tuple(map(int, rc)) for rc in np.argwhere(sprite)}
    for r, c in coords_set:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) not in coords_set:
                continue
            _draw_segment(stroke, mapped[(r, c)], mapped[(nr, nc)])

    line_count = int(stroke.sum())
    filled = _fill_enclosed(stroke)
    filled_count = int(filled.sum())
    recolored = _recolor_by_nearest_side(filled)

    out = g.copy()
    out[g == GRAY] = BLACK
    interior = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
    interior[:] = np.where(recolored != BLACK, recolored, BLACK)

    if not return_meta:
        return out

    side_counts = {color: int(np.count_nonzero(recolored == color)) for color in FRAME_SIDE_COLORS}
    meta = {
        "frame": frame,
        "sprite_bbox": (sr0, sc0, sr1, sc1),
        "seed_count": seed_count,
        "line_count": line_count,
        "filled_count": filled_count,
        "gray_count": int(np.count_nonzero(recolored == GRAY)),
        "side_counts": side_counts,
        "mapped_points": mapped,
        "dst_anchors": dst_anchors,
        "filled": filled,
        "recolored": recolored,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the outside gray sprite into the colored frame, fill the enclosed shape, and recolor it by nearest side.

    Input structure:
    - A black background contains one hollow rectangular frame whose corners stay black while the four sides have fixed colors: top red, right blue, bottom green, left yellow.
    - The frame interior is entirely black in the input.
    - Outside the frame there is exactly one 4-connected gray sprite with 3-6 pixels, bbox at least 2x2, and a non-linear turn or branch.

    Rule:
    - Remove the original outside sprite and crop it to its bbox.
    - Use the sprite's top, right, bottom, and left extreme pixels (centroid if an extreme is shared) as anchor points, map those anchors to the corresponding midpoints of the frame interior edges, and map the remaining sprite pixels through the same continuous warp, rounding to integer cells.
    - Inside the frame, connect mapped positions of 4-neighbor sprite pairs with gray segments; if the mapped endpoints are neither in one row nor one column, draw the required L path horizontally first and then vertically.
    - Fill black regions enclosed by those gray segments, then recolor every filled gray cell by the uniquely nearest interior edge color; cells tied between two or more nearest edges remain gray.

    Key insight:
    - The sprite first defines a stretched filled region inside the frame, and only then does each filled cell inherit color from its nearest frame side, so both geometry and distance-to-edge are necessary.
    """
    return _solve(grid, return_meta=False)


def _grid_with_frame(
    h: int, w: int, frame_top: int, frame_left: int, interior_h: int, interior_w: int
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    r0 = frame_top
    c0 = frame_left
    r1 = r0 + interior_h + 1
    c1 = c0 + interior_w + 1
    grid[r0, c0 + 1 : c1] = TOP_COLOR
    grid[r1, c0 + 1 : c1] = BOTTOM_COLOR
    grid[r0 + 1 : r1, c0] = LEFT_COLOR
    grid[r0 + 1 : r1, c1] = RIGHT_COLOR
    return grid


def _sprite_positions(
    h: int, w: int, frame: tuple[int, int, int, int], sprite_h: int, sprite_w: int
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame
    positions: list[tuple[int, int]] = []
    gap_r0 = max(0, fr0 - 1)
    gap_c0 = max(0, fc0 - 1)
    gap_r1 = min(h - 1, fr1 + 1)
    gap_c1 = min(w - 1, fc1 + 1)
    for r0 in range(h - sprite_h + 1):
        for c0 in range(w - sprite_w + 1):
            r1 = r0 + sprite_h - 1
            c1 = c0 + sprite_w - 1
            separated = r1 < gap_r0 or r0 > gap_r1 or c1 < gap_c0 or c0 > gap_c1
            if separated:
                positions.append((r0, c0))
    return positions


def _interesting_case(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    if np.array_equal(inp, out):
        return False
    if meta["line_count"] <= meta["seed_count"]:
        return False
    if meta["filled_count"] <= meta["line_count"]:
        return False
    if meta["gray_count"] < 1:
        return False
    colored_sides = sum(1 for count in meta["side_counts"].values() if count > 0)
    if colored_sides < 2:
        return False
    if meta["filled_count"] < 6:
        return False
    frame = meta["frame"]
    fr0, fc0, fr1, fc1 = frame
    interior = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
    if np.count_nonzero(interior) == interior.size:
        return False
    if len(_find_components(interior != BLACK)) != 1:
        return False
    output_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if output_sig in _SEEN_OUTPUT_SIGS:
        return False
    return True


def _sample_case() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    interior_h = random.randint(*INTERIOR_RANGE)
    interior_w = random.randint(*INTERIOR_RANGE)
    outer_h = interior_h + 2
    outer_w = interior_w + 2

    shape = random.choice(SPRITE_LIBRARY)
    sprite_h = max(r for r, _ in shape) + 1
    sprite_w = max(c for _, c in shape) + 1

    frame_positions = [
        (r0, c0)
        for r0 in range(h - outer_h + 1)
        for c0 in range(w - outer_w + 1)
        if _sprite_positions(h, w, (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1), sprite_h, sprite_w)
    ]
    if not frame_positions:
        return None

    frame_top, frame_left = random.choice(frame_positions)
    frame = (frame_top, frame_left, frame_top + outer_h - 1, frame_left + outer_w - 1)
    grid = _grid_with_frame(h, w, frame_top, frame_left, interior_h, interior_w)

    positions = _sprite_positions(h, w, frame, sprite_h, sprite_w)
    if not positions:
        return None
    sprite_top, sprite_left = random.choice(positions)
    for r, c in shape:
        grid[sprite_top + r, sprite_left + c] = GRAY

    try:
        out, meta = _solve(grid, return_meta=True)
    except Exception:
        return None

    if not np.array_equal(main(grid), out):
        return None
    if not _interesting_case(grid, out, meta):
        return None

    output_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
