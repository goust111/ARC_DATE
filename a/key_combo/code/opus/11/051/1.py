from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, bbox_h, components_mask, in_bounds, neighbors4, paste


BLACK = Color.BLACK
BLUE = Color.BLUE
GREEN = Color.GREEN
GREY = Color.GREY

GEN_ATTEMPTS = 400
BIG_THRESHOLD = 4


def _green_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for coords in components_mask(grid == GREEN, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        h = bbox_h((r0, c0, r1, c1))
        w = c1 - c0 + 1
        area = int(coords.shape[0])
        assert area == h * w, "green objects must be solid rectangles"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN), "green objects must fill their bbox"
        rects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "h": h,
                "w": w,
                "area": area,
            }
        )
    rects.sort(key=lambda item: item["bbox"])
    return rects


def _seed_position(grid: np.ndarray) -> tuple[int, int]:
    seeds = np.argwhere(grid == BLUE)
    assert seeds.shape[0] == 1, "expected exactly one blue seed"
    return int(seeds[0, 0]), int(seeds[0, 1])


def _flood_reached_black(grid: np.ndarray) -> np.ndarray:
    reached = np.zeros(grid.shape, dtype=bool)
    seen = np.zeros(grid.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    sr, sc = _seed_position(grid)
    seen[sr, sc] = True
    q.append((sr, sc))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                continue
            if int(grid[nr, nc]) != BLACK:
                continue
            seen[nr, nc] = True
            reached[nr, nc] = True
            q.append((nr, nc))
    return reached


def _is_active(rect: dict, reached: np.ndarray) -> bool:
    for r, c in rect["coords"]:
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(reached.shape, nr, nc) and reached[nr, nc]:
                return True
    return False


def _solve_with_variant(
    grid: np.ndarray,
    *,
    activate_all: bool = False,
    classify_by_height: bool = False,
    shift_by_width: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    reached = _flood_reached_black(g)
    rects = _green_rectangles(g)

    planned: list[tuple[dict, int]] = []
    occupied_targets = np.zeros(g.shape, dtype=bool)
    for rect in rects:
        active = True if activate_all else _is_active(rect, reached)
        if not active:
            continue

        size_key = rect["h"] if classify_by_height else rect["area"]
        shift = rect["w"] if shift_by_width else rect["h"]
        dr = -shift if size_key > BIG_THRESHOLD else shift
        r0, c0, r1, c1 = rect["bbox"]
        dst_r0 = r0 + dr
        dst_r1 = r1 + dr
        assert 0 <= dst_r0 <= dst_r1 < g.shape[0]
        assert np.all(g[dst_r0 : dst_r1 + 1, c0 : c1 + 1] == BLACK), "destination must be empty black"
        assert not np.any(occupied_targets[dst_r0 : dst_r1 + 1, c0 : c1 + 1]), "moved rectangles must not overlap"
        occupied_targets[dst_r0 : dst_r1 + 1, c0 : c1 + 1] = True
        planned.append((rect, dst_r0))

    out = g.copy()
    for rect, _ in planned:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for rect, dst_r0 in planned:
        r0, c0, r1, c1 = rect["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        paste(out, patch, dst_r0, c0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only the flood-activated green rectangles, sending large ones upward and small ones downward by their own height.

    Input structure:
    - A black grid contains exactly one blue seed pixel, several solid green rectangles, and grey one-cell-thick walls.
    - The walls split the black background into corridor regions; green rectangles do not touch walls and each green component is a solid axis-aligned rectangle.
    - The blue seed sits on a black-channel cell, and at least one rectangle lies in a wall-separated region the seed cannot reach through black cells.

    Rule:
    - Flood from the blue seed through 4-neighbor black cells only; grey walls and green rectangles block the flood, and the flood is only used for reachability tests.
    - A green rectangle is activated iff any of its pixels is 4-neighbor-adjacent to a flooded black cell.
    - For each activated rectangle, erase it from its source location, then move it vertically by its own height: area greater than 4 moves upward, area at most 4 moves downward.
    - Unactivated rectangles, the blue seed, and all grey walls remain unchanged.

    Key insight:
    - The wall-constrained flood decides which rectangles participate at all, and only those activated rectangles are size-split into two movement directions with a per-rectangle shift equal to the rectangle's own height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, GREEN, GREY}
    assert np.count_nonzero(g == BLUE) == 1

    return _solve_with_variant(g)


def _padded_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _mark_reserved(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = _padded_bbox(bbox, mask.shape, pad=pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _bbox_has_reserved(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = _padded_bbox(bbox, mask.shape, pad=pad)
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _cells_are_black(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK))


def _bbox_inside(bbox: tuple[int, int, int, int], limits: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    lr0, lc0, lr1, lc1 = limits
    return lr0 <= r0 <= r1 <= lr1 and lc0 <= c0 <= c1 <= lc1


def _candidate_positions(
    chamber: tuple[int, int, int, int],
    h: int,
    w: int,
    move_dr: int | None,
) -> list[tuple[int, int]]:
    top, left, bottom, right = chamber
    cells: list[tuple[int, int]] = []
    for r0 in range(top, bottom - h + 2):
        for c0 in range(left, right - w + 2):
            src = (r0, c0, r0 + h - 1, c0 + w - 1)
            if move_dr is not None:
                dst = (r0 + move_dr, c0, r0 + move_dr + h - 1, c0 + w - 1)
                if not _bbox_inside(dst, chamber):
                    continue
            cells.append((r0, c0))
    return cells


def _place_rectangle(
    grid: np.ndarray,
    reserved: np.ndarray,
    chamber: tuple[int, int, int, int],
    h: int,
    w: int,
    move_dr: int | None,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    candidates = _candidate_positions(chamber, h, w, move_dr)
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        src = (r0, c0, r0 + h - 1, c0 + w - 1)
        if _bbox_has_reserved(reserved, src, pad=1):
            continue
        if not _cells_are_black(grid, src):
            continue
        if move_dr is not None:
            dst = (r0 + move_dr, c0, r0 + move_dr + h - 1, c0 + w - 1)
            if _bbox_has_reserved(reserved, dst, pad=1):
                continue
            if not _cells_are_black(grid, dst):
                continue
            _mark_reserved(reserved, dst, pad=1)
        grid[r0 : r0 + h, c0 : c0 + w] = GREEN
        _mark_reserved(reserved, src, pad=1)
        return src
    return None


def _select_shape(options: list[tuple[int, int]], min_width: int, rng: random.Random, avoid_height: int | None = None) -> tuple[int, int]:
    candidates = [shape for shape in options if shape[1] <= min_width]
    if avoid_height is not None:
        preferred = [shape for shape in candidates if shape[0] != avoid_height]
        if preferred:
            candidates = preferred
    if not candidates:
        raise ValueError("no shape fits chamber")
    return rng.choice(candidates)


def _instance_metadata(grid: np.ndarray) -> dict:
    reached = _flood_reached_black(grid)
    rects = _green_rectangles(grid)
    info = []
    for rect in rects:
        active = _is_active(rect, reached)
        move = "stay"
        if active:
            move = "up" if rect["area"] > BIG_THRESHOLD else "down"
        info.append(
            {
                "bbox": rect["bbox"],
                "h": rect["h"],
                "w": rect["w"],
                "area": rect["area"],
                "active": active,
                "move": move,
            }
        )
    return {"rectangles": info}


def _invalid_shortcuts(grid: np.ndarray, expected: np.ndarray) -> bool:
    wrongs = []
    for kwargs in (
        {"activate_all": True},
        {"classify_by_height": True},
        {"shift_by_width": True},
    ):
        try:
            wrongs.append(_solve_with_variant(grid, **kwargs))
        except AssertionError:
            continue
    return any(np.array_equal(expected, wrong) for wrong in wrongs)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 24)
    left_w = rng.randint(7, 9)
    middle_w = rng.randint(7, 9)
    right_w = rng.randint(7, 9)
    width = left_w + middle_w + right_w + 2

    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    wall1 = left_w
    wall2 = left_w + 1 + middle_w
    grid[:, wall1] = GREY
    grid[:, wall2] = GREY
    door_r = rng.randint(2, height - 3)
    grid[door_r, wall1] = BLACK

    chambers = {
        "L": (1, 1, height - 2, wall1 - 1),
        "M": (1, wall1 + 2, height - 2, wall2 - 1),
        "R": (1, wall2 + 2, height - 2, width - 2),
    }

    big_boundary = _select_shape([(1, 5), (5, 1)], left_w - 2, rng)
    big_other = _select_shape([(2, 3), (3, 2), (2, 4), (4, 2), (3, 3)], middle_w - 2, rng, avoid_height=big_boundary[0])
    small_boundary = _select_shape([(1, 4), (2, 2), (4, 1)], middle_w - 2, rng)
    small_other = _select_shape([(1, 2), (2, 1), (1, 3), (3, 1)], left_w - 2, rng, avoid_height=small_boundary[0])
    inactive_shape = _select_shape([(1, 2), (2, 1), (2, 3), (3, 2), (1, 4), (4, 1)], right_w - 2, rng)

    plans = [
        ("L", big_boundary, -big_boundary[0]),
        ("M", big_other, -big_other[0]),
        ("M", small_boundary, small_boundary[0]),
        ("L", small_other, small_other[0]),
        ("R", inactive_shape, None),
    ]
    active_plans = plans[:4]
    rng.shuffle(active_plans)
    plans = active_plans + plans[4:]

    if rng.random() < 0.6:
        extra = _select_shape([(1, 5), (5, 1), (1, 3), (3, 1), (2, 2)], right_w - 2, rng)
        plans.append(("R", extra, None))

    for chamber_name, (h, w), move_dr in plans:
        bbox = _place_rectangle(grid, reserved, chambers[chamber_name], h, w, move_dr, rng)
        if bbox is None:
            return None

    seed_side = rng.choice(["L", "M"])
    seed_box = chambers[seed_side]
    seed_cells = _candidate_positions(seed_box, 1, 1, None)
    rng.shuffle(seed_cells)
    seed = None
    for sr, sc in seed_cells:
        if grid[sr, sc] != BLACK:
            continue
        if reserved[sr, sc]:
            continue
        black_neighbors = 0
        for nr, nc in neighbors4(sr, sc):
            if in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK:
                black_neighbors += 1
        if black_neighbors >= 2:
            seed = (sr, sc)
            break
    if seed is None:
        return None
    grid[seed] = BLUE

    if rng.random() < 0.5:
        grid = np.fliplr(grid).copy()

    expected = main(grid)
    meta = _instance_metadata(grid)
    rects = meta["rectangles"]
    active = [rect for rect in rects if rect["active"]]
    inactive = [rect for rect in rects if not rect["active"]]
    if len(active) < 4 or len(inactive) < 1:
        return None
    if not any(rect["area"] == 4 and rect["move"] == "down" for rect in active):
        return None
    if not any(rect["area"] == 5 and rect["move"] == "up" for rect in active):
        return None
    if not any(rect["move"] == "up" for rect in active):
        return None
    if not any(rect["move"] == "down" for rect in active):
        return None
    if len({rect["h"] for rect in active}) < 3:
        return None
    if _invalid_shortcuts(grid, expected):
        return None

    out_rects = _green_rectangles(expected)
    if len(out_rects) != len(rects):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flood-activation translation input")
