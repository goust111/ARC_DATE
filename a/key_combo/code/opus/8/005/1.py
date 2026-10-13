from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import Color, bbox_coords, components_mask


ORANGE = Color.ORANGE
GREY = Color.GREY
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
TEAL = Color.TEAL
BLACK = Color.BLACK

FRAME_COLORS = {BLUE, RED, GREEN, YELLOW}
STRIPE_COLORS = {BLUE, TEAL}

SHAPES = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    comps = components_mask(mask, connectivity=4)
    assert comps, "expected at least one component"
    return max(comps, key=lambda coords: int(coords.shape[0]))


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    left = _largest_component(grid == RED)
    right = _largest_component(grid == YELLOW)
    lr0, lc0, lr1, lc1 = bbox_coords(left)
    rr0, rc0, rr1, rc1 = bbox_coords(right)
    assert lc0 == lc1 and rc0 == rc1, "frame sides must be vertical"
    assert lr0 == rr0 and lr1 == rr1, "frame sides must span the same rows"
    r0 = int(lr0 - 1)
    r1 = int(lr1 + 1)
    c0 = int(lc0)
    c1 = int(rc0)
    assert r0 >= 0 and c0 >= 0 and r1 < grid.shape[0] and c1 < grid.shape[1]
    assert np.all(grid[r0, c0 : c1 + 1] == BLUE), "top frame edge must be blue"
    assert np.all(grid[r1, c0 : c1 + 1] == GREEN), "bottom frame edge must be green"
    return r0, c0, r1, c1


def _scatter_pixels(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame
    pts = []
    for r, c in np.argwhere(grid != BLACK):
        rr = int(r)
        cc = int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        pts.append((rr, cc, int(grid[rr, cc])))
    pts.sort()
    return pts


def _carrier_color(crop: np.ndarray, coords: np.ndarray) -> int:
    seen = set()
    h, w = crop.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(crop[nr, nc]) in STRIPE_COLORS:
                seen.add(int(crop[nr, nc]))
    assert len(seen) == 1, "each orange object must sit in one stripe color"
    return next(iter(seen))


def _draw_obstacles(crop: np.ndarray, scatters: list[tuple[int, int, int]], frame: tuple[int, int, int, int]) -> np.ndarray:
    out = crop.copy()
    h, w = out.shape
    r0, c0, _r1, _c1 = frame
    for sr, sc, color in scatters:
        if color == BLUE:
            rr, cc, dr, dc = 1, sc - c0, 1, 0
        elif color == GREEN:
            rr, cc, dr, dc = h - 2, sc - c0, -1, 0
        elif color == RED:
            rr, cc, dr, dc = sr - r0, 1, 0, 1
        elif color == YELLOW:
            rr, cc, dr, dc = sr - r0, w - 2, 0, -1
        else:
            continue

        while 1 <= rr <= h - 2 and 1 <= cc <= w - 2 and int(out[rr, cc]) in STRIPE_COLORS:
            out[rr, cc] = GREY
            rr += dr
            cc += dc
    return out


def _slide_distance(base: np.ndarray, coords: np.ndarray, direction: int) -> int:
    h, w = base.shape
    shift = 0
    while True:
        next_cols = coords[:, 1] + direction * (shift + 1)
        if np.any(next_cols < 1) or np.any(next_cols > w - 2):
            break
        if np.any(base[coords[:, 0], next_cols] == GREY):
            break
        shift += 1
    return int(direction * shift)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the colored frame, turn outer side-mark pixels into interior gray blockers, then slide each orange shape horizontally.

    Input structure:
    - A black background contains one colored rectangular frame whose top, left, bottom, and right edges are blue, red, green, and yellow.
    - Inside the frame, horizontal full-width stripes of blue or teal carry one orange 4-connected shape each.
    - Outside the frame, 2 to 4 isolated pixels reuse a frame-edge color and lie on that edge's row or column extension.

    Rule:
    - Crop to the frame's bbox, keeping the frame and its interior only.
    - For each outer pixel, project inward from the matching frame side and paint a gray one-cell-wide segment through stripe cells until the next cell would be orange or gray, or until the opposite inner edge.
    - Clear each orange object back to its stripe color, then slide blue-carried objects right and teal-carried objects left as far as possible without crossing a gray blocker or the frame interior boundary.
    - Paint all orange objects at their final translated positions simultaneously.

    Key insight:
    - The outside pixels do not survive the crop; they only encode where gray blocker segments are inserted, and those blockers jointly determine each stripe object's stopping point.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _frame_bbox(g)
    r0, c0, r1, c1 = frame
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    scatters = _scatter_pixels(g, frame)
    with_obstacles = _draw_obstacles(crop, scatters, frame)

    orange_objects = components_mask(crop == ORANGE, connectivity=4)
    assert orange_objects, "expected orange objects inside the frame"

    base = with_obstacles.copy()
    plans: list[tuple[np.ndarray, int]] = []
    for coords in orange_objects:
        carrier = _carrier_color(crop, coords)
        direction = 1 if carrier == BLUE else -1
        base[coords[:, 0], coords[:, 1]] = carrier
        plans.append((coords, direction))

    out = base.copy()
    for coords, direction in plans:
        delta = _slide_distance(base, coords, direction)
        moved = coords.copy()
        moved[:, 1] += delta
        out[moved[:, 0], moved[:, 1]] = ORANGE
    return out


def _shape_cells(shape: np.ndarray) -> np.ndarray:
    return np.argwhere(shape == 1)


def _assign_stripe_colors(count: int) -> list[int]:
    colors = [BLUE, TEAL]
    while len(colors) < count:
        colors.append(random.choice([BLUE, TEAL]))
    random.shuffle(colors)
    return colors


def _choose_shapes(count: int) -> list[np.ndarray]:
    return [random.choice(SHAPES).copy() for _ in range(count)]


def _place_object(
    crop: np.ndarray,
    stripe_start: int,
    stripe_height: int,
    shape: np.ndarray,
    direction: int,
    blocked: bool,
    mode: str,
    blocker_col: int | None,
    forbidden_cols: set[int],
) -> tuple[np.ndarray, tuple[int, int]]:
    cells = _shape_cells(shape)
    shape_h, shape_w = shape.shape
    h, w = crop.shape
    stripe_end = stripe_start + stripe_height - 1

    row_lo = stripe_start + 1
    row_hi = stripe_end - shape_h + 1
    if stripe_end == h - 2:
        row_hi -= 1
    assert row_lo <= row_hi, "shape must fit the stripe"

    col_lo = 2
    col_hi = w - 3 - shape_w + 1
    assert col_lo <= col_hi, "shape must stay away from frame sides"

    candidates = []
    for top in range(row_lo, row_hi + 1):
        for left in range(col_lo, col_hi + 1):
            placed = cells + np.array([top, left])
            cols = {int(c) for c in placed[:, 1]}
            if mode == "vertical":
                if cols & forbidden_cols:
                    continue
                assert blocker_col is not None or not blocked
                if blocked:
                    if direction == 1 and int(placed[:, 1].max()) >= int(blocker_col):
                        continue
                    if direction == -1 and int(placed[:, 1].min()) <= int(blocker_col):
                        continue
                else:
                    if direction == 1 and int(placed[:, 1].max()) >= w - 3:
                        continue
                    if direction == -1 and int(placed[:, 1].min()) <= 2:
                        continue
            else:
                if not blocked:
                    if direction == 1 and int(placed[:, 1].max()) >= w - 3:
                        continue
                    if direction == -1 and int(placed[:, 1].min()) <= 2:
                        continue
            candidates.append((placed, (top, left)))

    assert candidates, "expected at least one valid object placement"
    placed, anchor = random.choice(candidates)
    crop[placed[:, 0], placed[:, 1]] = ORANGE
    return placed, anchor


def _build_crop(
    stripe_colors: list[int],
    shapes: list[np.ndarray],
    mode: str,
    blocked_indices: set[int],
) -> tuple[np.ndarray, list[dict]]:
    stripe_count = len(stripe_colors)
    stripe_height = 4
    inner_h = stripe_count * stripe_height
    inner_w = random.randint(max(8, stripe_count + 5), 12)
    crop = np.zeros((inner_h + 2, inner_w + 2), dtype=int)

    crop[0, :] = BLUE
    crop[-1, :] = GREEN
    crop[1:-1, 0] = RED
    crop[1:-1, -1] = YELLOW

    stripe_info: list[dict] = []
    for idx, color in enumerate(stripe_colors):
        start = 1 + idx * stripe_height
        end = start + stripe_height
        crop[start:end, 1:-1] = color
        stripe_info.append(
            {
                "color": int(color),
                "start": int(start),
                "height": stripe_height,
                "direction": 1 if color == BLUE else -1,
            }
        )

    forbidden_cols: set[int] = set()
    blocker_cols_by_idx: dict[int, int] = {}
    if mode == "vertical":
        free_cols = list(range(2, crop.shape[1] - 2))
        random.shuffle(free_cols)
        assert len(free_cols) >= len(blocked_indices)
        for idx in sorted(blocked_indices):
            blocker_cols_by_idx[idx] = int(free_cols.pop())
            forbidden_cols.add(blocker_cols_by_idx[idx])

    objects = []
    for idx, info in enumerate(stripe_info):
        blocked = idx in blocked_indices
        placed, anchor = _place_object(
            crop,
            info["start"],
            info["height"],
            shapes[idx],
            info["direction"],
            blocked,
            mode,
            blocker_cols_by_idx.get(idx),
            forbidden_cols,
        )
        objects.append(
            {
                "coords": placed,
                "anchor": anchor,
                "blocked": blocked,
                "blocker_col": blocker_cols_by_idx.get(idx),
                "direction": info["direction"],
                "color": info["color"],
            }
        )
    return crop, objects


def _scatter_for_object(
    obj: dict,
    frame_offset: tuple[int, int],
    outer_shape: tuple[int, int],
    frame_shape: tuple[int, int],
    mode: str,
) -> tuple[int, int, int]:
    fr0, fc0 = frame_offset
    outer_h, outer_w = outer_shape
    frame_h, frame_w = frame_shape
    coords = obj["coords"]
    if mode == "horizontal":
        if obj["direction"] == 1:
            lead_col = int(coords[:, 1].max())
            rows = sorted(int(r) for r in coords[coords[:, 1] == lead_col][:, 0])
            row = random.choice(rows)
            return fr0 + row, random.randint(fc0 + frame_w + 1, outer_w - 1), YELLOW
        lead_col = int(coords[:, 1].min())
        rows = sorted(int(r) for r in coords[coords[:, 1] == lead_col][:, 0])
        row = random.choice(rows)
        return fr0 + row, random.randint(0, fc0 - 2), RED

    col = int(obj["blocker_col"])
    if random.choice([True, False]):
        return random.randint(0, fr0 - 2), fc0 + col, BLUE
    return random.randint(fr0 + frame_h + 1, outer_h - 1), fc0 + col, GREEN


def _build_sample() -> np.ndarray:
    stripe_count = random.randint(2, 4)
    stripe_colors = _assign_stripe_colors(stripe_count)
    shapes = _choose_shapes(stripe_count)
    mode = random.choice(["horizontal", "vertical"])

    if stripe_count == 2:
        blocked_count = 2
    else:
        blocked_count = random.randint(2, stripe_count)
    blocked_indices = set(random.sample(range(stripe_count), blocked_count))

    crop, objects = _build_crop(stripe_colors, shapes, mode, blocked_indices)
    frame_h, frame_w = crop.shape

    top_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    left_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)

    outer_h = frame_h + top_margin + bottom_margin
    outer_w = frame_w + left_margin + right_margin
    grid = np.zeros((outer_h, outer_w), dtype=int)
    fr0, fc0 = top_margin, left_margin
    grid[fr0 : fr0 + frame_h, fc0 : fc0 + frame_w] = crop

    scatters = []
    for idx in sorted(blocked_indices):
        scatters.append(_scatter_for_object(objects[idx], (fr0, fc0), grid.shape, crop.shape, mode))

    for r, c, color in scatters:
        grid[int(r), int(c)] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    frame = _frame_bbox(grid)
    scatters = _scatter_pixels(grid, frame)
    assert 2 <= len(scatters) <= 4
    assert all(color in FRAME_COLORS for _, _, color in scatters)
    crop = grid[frame[0] : frame[2] + 1, frame[1] : frame[3] + 1]
    assert len(components_mask(crop == ORANGE, connectivity=4)) >= 2
    out = main(grid)
    assert out.ndim == 2 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid = _build_sample()
            _validate_generated(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
