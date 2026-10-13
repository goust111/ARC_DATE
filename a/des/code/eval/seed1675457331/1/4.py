import random

import numpy as np


BG = 0
MAROON = 9
BODY_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_MASKS = {
    "up": np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    "down": np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "left": np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=bool),
    "right": np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
}

GRID_MIN = 17
GRID_MAX = 24
OBJECT_COUNT_RANGE = (2, 4)
SIZE_RANGE = (3, 6)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 500


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _axis_rc(r0: int, c0: int, h: int, w: int) -> tuple[int, int]:
    return r0 + (h - 1) // 2, c0 + (w - 1) // 2


def _patch_origin(r0: int, c0: int, h: int, w: int) -> tuple[int, int]:
    axis_r, axis_c = _axis_rc(r0, c0, h, w)
    return axis_r - 1, axis_c - 1


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, h, w = box
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _first_shrink(size: int) -> int:
    return max(size - 1, 2)


def _later_shrink(size: int) -> int:
    return max(size - 1, 1)


def _fit_box(shape: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    h, w = shape
    r0, c0, bh, bw = box
    return 0 <= r0 and 0 <= c0 and r0 + bh <= h and c0 + bw <= w


def _clear_region(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _draw_rectangle(grid: np.ndarray, rect: dict) -> None:
    r0, c0, h, w = rect["r0"], rect["c0"], rect["h"], rect["w"]
    color = rect["color"]
    _paint_box(grid, (r0, c0, h, w), color)
    pr0, pc0 = _patch_origin(r0, c0, h, w)
    patch = grid[pr0 : pr0 + 3, pc0 : pc0 + 3]
    patch[ARROW_MASKS[rect["dir"]]] = MAROON


def _parse_rectangle(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(window == BG):
        return None

    pr0, pc0 = _patch_origin(r0, c0, h, w)
    patch = grid[pr0 : pr0 + 3, pc0 : pc0 + 3]
    if patch.shape != (3, 3):
        return None

    maroon_mask = patch == MAROON
    direction = None
    for name, arrow_mask in ARROW_MASKS.items():
        if np.array_equal(maroon_mask, arrow_mask):
            direction = name
            break
    if direction is None:
        return None

    if int(np.count_nonzero(window == MAROON)) != int(np.count_nonzero(maroon_mask)):
        return None

    colors = set(map(int, np.unique(window)))
    colors.discard(MAROON)
    if len(colors) != 1:
        return None
    color = colors.pop()

    return {
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "color": color,
        "dir": direction,
    }


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in _components4(grid != BG):
        rect = _parse_rectangle(grid, comp)
        if rect is not None:
            rects.append(rect)
    rects.sort(key=lambda rect: (rect["r0"], rect["c0"]))
    return rects


def _next_box(prev: tuple[int, int, int, int], rect: dict, next_h: int, next_w: int) -> tuple[int, int, int, int]:
    prev_r0, prev_c0, prev_h, prev_w = prev
    axis_r, axis_c = _axis_rc(rect["r0"], rect["c0"], rect["h"], rect["w"])
    if rect["dir"] == "right":
        return prev_r0 + 0, prev_c0 + prev_w + 1, next_h, next_w
    if rect["dir"] == "left":
        return prev_r0 + 0, prev_c0 - next_w - 1, next_h, next_w
    if rect["dir"] == "down":
        return prev_r0 + prev_h + 1, prev_c0 + 0, next_h, next_w
    if rect["dir"] == "up":
        return prev_r0 - next_h - 1, prev_c0 + 0, next_h, next_w
    raise ValueError(rect["dir"])


def _aligned_box(rect: dict, prev: tuple[int, int, int, int], next_h: int, next_w: int) -> tuple[int, int, int, int]:
    raw_r0, raw_c0, _, _ = _next_box(prev, rect, next_h, next_w)
    axis_r, axis_c = _axis_rc(rect["r0"], rect["c0"], rect["h"], rect["w"])
    if rect["dir"] in ("left", "right"):
        return axis_r - (next_h - 1) // 2, raw_c0, next_h, next_w
    return raw_r0, axis_c - (next_w - 1) // 2, next_h, next_w


def _grow_chain(
    rect: dict,
    occupied: np.ndarray,
    original_mask: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, int, int]], str, bool]:
    occ = occupied.copy()
    boxes: list[tuple[int, int, int, int]] = []
    prev = (rect["r0"], rect["c0"], rect["h"], rect["w"])
    cur_h, cur_w = rect["h"], rect["w"]
    first = True
    blocked_by_original = False

    while True:
        if first:
            next_h = _first_shrink(cur_h)
            next_w = _first_shrink(cur_w)
            first = False
        else:
            next_h = _later_shrink(cur_h)
            next_w = _later_shrink(cur_w)

        box = _aligned_box(rect, prev, next_h, next_w)
        if not _fit_box(occ.shape, box):
            return boxes, "boundary", blocked_by_original

        r0, c0, h, w = box
        if bool(occ[r0 : r0 + h, c0 : c0 + w].any()):
            if original_mask is not None and bool(original_mask[r0 : r0 + h, c0 : c0 + w].any()):
                blocked_by_original = True
            return boxes, "collision", blocked_by_original

        boxes.append(box)
        occ[r0 : r0 + h, c0 : c0 + w] = True
        prev = box
        cur_h, cur_w = next_h, next_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow progressively smaller monochrome copies of each marked rectangle
    in the direction indicated by its centered maroon arrow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _find_rectangles(g)
    assert rects, "expected at least one marked rectangle"

    original_occupied = g != BG
    out = g.copy()
    for rect in rects:
        boxes, _reason, _blocked = _grow_chain(rect, original_occupied)
        for box in boxes:
            _paint_box(out, box, rect["color"])
    return out


def _count_1x1(boxes: list[tuple[int, int, int, int]]) -> int:
    return sum(1 for _r0, _c0, h, w in boxes if h == 1 and w == 1)


def _random_direction(rng: random.Random) -> str:
    return rng.choice(tuple(DIR_VECS))


def _sample_boundary_origin(
    rng: random.Random,
    plan: str,
    direction: str,
    h: int,
    w: int,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    if grid_h - h - 1 < 1 or grid_w - w - 1 < 1:
        return None

    if direction in ("left", "right"):
        r0 = rng.randint(1, grid_h - h - 1)
        max_clear = grid_w - w - 1
        if max_clear < 1:
            return None
        if plan == "long":
            low = max(4, max_clear // 2)
            if low > max_clear:
                return None
            clear = rng.randint(low, max_clear)
        else:
            high = min(4, max_clear)
            if high < 1:
                return None
            clear = rng.randint(1, high)
        c0 = clear if direction == "left" else grid_w - w - clear
        return r0, c0

    c0 = rng.randint(1, grid_w - w - 1)
    max_clear = grid_h - h - 1
    if max_clear < 1:
        return None
    if plan == "long":
        low = max(4, max_clear // 2)
        if low > max_clear:
            return None
        clear = rng.randint(low, max_clear)
    else:
        high = min(4, max_clear)
        if high < 1:
            return None
        clear = rng.randint(1, high)
    r0 = clear if direction == "up" else grid_h - h - clear
    return r0, c0


def _place_boundary_object(
    rng: random.Random,
    plan: str,
    color: int,
    grid_h: int,
    grid_w: int,
    reserved_all: np.ndarray,
    reserved_orig: np.ndarray,
) -> tuple[dict, list[tuple[int, int, int, int]]] | None:
    for _ in range(PLACEMENT_RETRIES):
        h = rng.randint(*SIZE_RANGE)
        w = rng.randint(*SIZE_RANGE)
        direction = _random_direction(rng)
        origin = _sample_boundary_origin(rng, plan, direction, h, w, grid_h, grid_w)
        if origin is None:
            continue
        r0, c0 = origin
        if not _clear_region(reserved_all, r0, c0, h, w):
            continue
        if not _clear_region(reserved_orig, r0, c0, h, w, pad=1):
            continue

        rect = {"r0": r0, "c0": c0, "h": h, "w": w, "color": color, "dir": direction}
        boxes, reason, _blocked = _grow_chain(rect, reserved_all, reserved_orig)
        if not boxes or reason != "boundary":
            continue

        singleton_count = _count_1x1(boxes)
        if plan == "long" and singleton_count >= 2:
            return rect, boxes
        if plan == "short_boundary" and singleton_count == 0:
            return rect, boxes
    return None


def _collision_origin(
    rng: random.Random,
    blocker: dict,
    direction: str,
    h: int,
    w: int,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    if direction == "right":
        axis_r = rng.randint(blocker["r0"], blocker["r0"] + blocker["h"] - 1)
        r0 = axis_r - (h - 1) // 2
        max_c0 = blocker["c0"] - w - 3
        if r0 < 1 or r0 + h > grid_h - 1 or max_c0 < 1:
            return None
        c0 = rng.randint(max(1, max_c0 - 6), max_c0)
        return r0, c0
    if direction == "left":
        axis_r = rng.randint(blocker["r0"], blocker["r0"] + blocker["h"] - 1)
        r0 = axis_r - (h - 1) // 2
        min_c0 = blocker["c0"] + blocker["w"] + 2
        max_c0 = grid_w - w - 1
        if r0 < 1 or r0 + h > grid_h - 1 or min_c0 > max_c0:
            return None
        c0 = rng.randint(min_c0, min(max_c0, min_c0 + 6))
        return r0, c0
    if direction == "down":
        axis_c = rng.randint(blocker["c0"], blocker["c0"] + blocker["w"] - 1)
        c0 = axis_c - (w - 1) // 2
        max_r0 = blocker["r0"] - h - 3
        if c0 < 1 or c0 + w > grid_w - 1 or max_r0 < 1:
            return None
        r0 = rng.randint(max(1, max_r0 - 6), max_r0)
        return r0, c0
    axis_c = rng.randint(blocker["c0"], blocker["c0"] + blocker["w"] - 1)
    c0 = axis_c - (w - 1) // 2
    min_r0 = blocker["r0"] + blocker["h"] + 2
    max_r0 = grid_h - h - 1
    if c0 < 1 or c0 + w > grid_w - 1 or min_r0 > max_r0:
        return None
    r0 = rng.randint(min_r0, min(max_r0, min_r0 + 6))
    return r0, c0


def _place_collision_object(
    rng: random.Random,
    color: int,
    grid_h: int,
    grid_w: int,
    reserved_all: np.ndarray,
    reserved_orig: np.ndarray,
    blockers: list[dict],
) -> tuple[dict, list[tuple[int, int, int, int]]] | None:
    for _ in range(PLACEMENT_RETRIES * 2):
        blocker = rng.choice(blockers)
        h = rng.randint(*SIZE_RANGE)
        w = rng.randint(*SIZE_RANGE)
        direction = _random_direction(rng)
        origin = _collision_origin(rng, blocker, direction, h, w, grid_h, grid_w)
        if origin is None:
            continue
        r0, c0 = origin
        if not _clear_region(reserved_all, r0, c0, h, w):
            continue
        if not _clear_region(reserved_orig, r0, c0, h, w, pad=1):
            continue

        rect = {"r0": r0, "c0": c0, "h": h, "w": w, "color": color, "dir": direction}
        boxes, reason, blocked_by_original = _grow_chain(rect, reserved_all, reserved_orig)
        if boxes and reason == "collision" and blocked_by_original:
            return rect, boxes
    return None


def _reserve(mask: np.ndarray, rect: dict, boxes: list[tuple[int, int, int, int]]) -> None:
    _paint_box(mask, (rect["r0"], rect["c0"], rect["h"], rect["w"]), True)
    for box in boxes:
        _paint_box(mask, box, True)


def _generation_plans(rng: random.Random, n_objects: int) -> list[str]:
    plans = ["long", "short_boundary"]
    if n_objects >= 3 or rng.random() < 0.6:
        plans.append("collision")
    while len(plans) < n_objects:
        plans.append(rng.choice(("long", "short_boundary", "collision")))
    plans = plans[:n_objects]
    plans.sort(key=lambda plan: plan == "collision")
    return plans


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    grid_h = rng.randint(max(GRID_MIN, 15 + n_objects), GRID_MAX)
    grid_w = rng.randint(max(GRID_MIN, 15 + n_objects), GRID_MAX)
    colors = rng.sample(BODY_COLORS, n_objects)

    reserved_all = np.zeros((grid_h, grid_w), dtype=bool)
    reserved_orig = np.zeros((grid_h, grid_w), dtype=bool)
    rects: list[dict] = []
    boxes_by_rect: list[list[tuple[int, int, int, int]]] = []

    for plan, color in zip(_generation_plans(rng, n_objects), colors):
        placed = None
        if plan == "collision" and rects:
            placed = _place_collision_object(
                rng, color, grid_h, grid_w, reserved_all, reserved_orig, rects
            )
        if placed is None:
            fallback_plan = "long" if plan == "collision" else plan
            placed = _place_boundary_object(
                rng, fallback_plan, color, grid_h, grid_w, reserved_all, reserved_orig
            )
        if placed is None:
            return None

        rect, boxes = placed
        rects.append(rect)
        boxes_by_rect.append(boxes)
        _reserve(reserved_all, rect, boxes)
        _paint_box(reserved_orig, (rect["r0"], rect["c0"], rect["h"], rect["w"]), True)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for rect in rects:
        _draw_rectangle(grid, rect)

    parsed = _find_rectangles(grid)
    if len(parsed) != len(rects):
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
