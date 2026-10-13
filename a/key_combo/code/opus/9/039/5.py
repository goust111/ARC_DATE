from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
SQUARE_COLOR = 3
MARK_COLOR = 2

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
SQUARE_COUNT_CHOICES = [1, 2, 2, 3]
DISTRACTOR_COUNT_BY_SQUARES = {
    1: [2, 3, 4],
    2: [1, 2, 3],
    3: [1, 2],
}
SQUARE_SIZES = [2, 3, 4]
DISTRACTOR_TEMPLATES = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
]
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, color: int = SQUARE_COLOR) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    objects = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            objects.append(
                {
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return objects


def _has_complete_square_border(grid: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 2:
        return False
    return bool(
        np.all(grid[r0, c0 : c1 + 1] == SQUARE_COLOR)
        and np.all(grid[r1, c0 : c1 + 1] == SQUARE_COLOR)
        and np.all(grid[r0 : r1 + 1, c0] == SQUARE_COLOR)
        and np.all(grid[r0 : r1 + 1, c1] == SQUARE_COLOR)
    )


def _corner_markers(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _square_sprite(obj: dict) -> tuple[int, list[tuple[int, int, int]]]:
    r0, c0, r1, c1 = obj["bbox"]
    side = r1 - r0 + 1
    pixels = [(r - r0, c - c0, SQUARE_COLOR) for r, c in obj["coords"]]
    pixels.extend(
        [
            (-1, 0, MARK_COLOR),
            (0, -1, MARK_COLOR),
            (-1, side - 1, MARK_COLOR),
            (0, side, MARK_COLOR),
            (side, 0, MARK_COLOR),
            (side - 1, -1, MARK_COLOR),
            (side, side - 1, MARK_COLOR),
            (side - 1, side, MARK_COLOR),
        ]
    )
    return side + 2, pixels


def _trajectory_stats(shape: tuple[int, int], obj: dict) -> dict[int, dict[str, int]]:
    H, W = shape
    r0, c0, r1, c1 = obj["bbox"]
    side = r1 - r0 + 1
    step = side + 2
    stats: dict[int, dict[str, int]] = {}
    for sign in (-1, 1):
        full = 0
        partial = 0
        touched = 0
        k = 1
        while True:
            ar = r0 + k * step
            ac = c0 + sign * k * step
            br0, bc0 = ar - 1, ac - 1
            br1, bc1 = ar + side, ac + side
            if br0 >= H or br1 < 0 or bc0 >= W or bc1 < 0:
                break
            touched += 1
            if 0 <= br0 and br1 < H and 0 <= bc0 and bc1 < W:
                full += 1
            else:
                partial += 1
            k += 1
        stats[sign] = {"full": full, "partial": partial, "touched": touched}
    return stats


def _stamp_with_order(grid: np.ndarray, square_objects: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    H, W = out.shape
    for obj in square_objects:
        r0, c0, r1, c1 = obj["bbox"]
        for mr, mc in _corner_markers(r0, c0, r1, c1):
            if _in_bounds(out.shape, mr, mc):
                out[mr, mc] = MARK_COLOR

    for obj in square_objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        step, pixels = _square_sprite(obj)
        for sign in (1, -1):
            k = 1
            while True:
                anchor_r = r0 + k * step
                anchor_c = c0 + sign * k * step
                bbox_r0 = anchor_r - 1
                bbox_c0 = anchor_c - 1
                bbox_r1 = anchor_r + step - 2
                bbox_c1 = anchor_c + step - 2
                if bbox_r0 >= H or bbox_r1 < 0 or bbox_c0 >= W or bbox_c1 < 0:
                    break
                for dr, dc, color in pixels:
                    rr = anchor_r + dr
                    cc = anchor_c + dc
                    if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BACKGROUND:
                        out[rr, cc] = color
                k += 1
    return out


def _wrong_stride_output(grid: np.ndarray) -> np.ndarray:
    square_objects = [obj for obj in _find_components(grid) if _has_complete_square_border(grid, obj)]
    out = np.array(grid, dtype=int, copy=True)
    H, W = out.shape
    for obj in square_objects:
        r0, c0, r1, c1 = obj["bbox"]
        for mr, mc in _corner_markers(r0, c0, r1, c1):
            if _in_bounds(out.shape, mr, mc):
                out[mr, mc] = MARK_COLOR

    for obj in square_objects:
        r0, c0, r1, c1 = obj["bbox"]
        side = r1 - r0 + 1
        pixels = _square_sprite(obj)[1]
        for sign in (1, -1):
            k = 1
            while True:
                anchor_r = r0 + k * side
                anchor_c = c0 + sign * k * side
                bbox_r0 = anchor_r - 1
                bbox_c0 = anchor_c - 1
                bbox_r1 = anchor_r + side
                bbox_c1 = anchor_c + side
                if bbox_r0 >= H or bbox_r1 < 0 or bbox_c0 >= W or bbox_c1 < 0:
                    break
                for dr, dc, color in pixels:
                    rr = anchor_r + dr
                    cc = anchor_c + dc
                    if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BACKGROUND:
                        out[rr, cc] = color
                k += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect complete green squares, add red outer-corner markers, and stamp each marked square diagonally using the expanded marked bbox as the stride.

    Input structure:
    - The grid is a black background with several separated green 4-connected objects.
    - Some green objects are complete squares: their bounding box is square and every cell on that bbox border is green.
    - Other green objects are non-square bars or irregular fragments and remain unchanged.

    Rule:
    - Keep every original object in place.
    - For each complete square, add red pixels on the two orthogonally adjacent outside cells of each corner.
    - Treat the original green square pattern plus those eight red markers as one composite sprite.
    - Repeatedly stamp that composite sprite at offsets (+h', +w'), (+2h', +2w'), ... and (+h', -w'), (+2h', -2w'), ..., where h' = w' = side + 2 is the marked sprite bbox size.
    - Clip out-of-bounds pixels, and skip any stamped pixel whose destination is already non-background.

    Key insight:
    - The corner markers are not decorative: they enlarge the effective sprite bbox, so the diagonal step is side + 2 rather than the square's original side.
    """
    g = np.asarray(grid, dtype=int)
    square_objects = [obj for obj in _find_components(g) if _has_complete_square_border(g, obj)]
    square_objects.sort(key=lambda obj: obj["bbox"])
    return _stamp_with_order(g, square_objects)


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _bbox_far_enough(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int], gap: int = 2) -> bool:
    row_gap = _interval_gap(bbox_a[0], bbox_a[2], bbox_b[0], bbox_b[2])
    col_gap = _interval_gap(bbox_a[1], bbox_a[3], bbox_b[1], bbox_b[3])
    return not (row_gap < gap and col_gap < gap)


def _apply_shape(grid: np.ndarray, top: int, left: int, shape: np.ndarray) -> None:
    h, w = shape.shape
    mask = shape.astype(bool)
    grid[top : top + h, left : left + w][mask] = SQUARE_COLOR


def _can_place_shape(
    H: int,
    W: int,
    placed_bboxes: list[tuple[int, int, int, int]],
    top: int,
    left: int,
    shape: np.ndarray,
) -> bool:
    h, w = shape.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    bbox = (top, left, top + h - 1, left + w - 1)
    return all(_bbox_far_enough(bbox, prev_bbox, gap=2) for prev_bbox in placed_bboxes)


def _square_shape(side: int, filled: bool) -> np.ndarray:
    if filled or side == 2:
        return np.ones((side, side), dtype=int)
    shape = np.zeros((side, side), dtype=int)
    shape[0, :] = 1
    shape[-1, :] = 1
    shape[:, 0] = 1
    shape[:, -1] = 1
    return shape


def _future_stamp_pixels(square_objects: list[dict], shape: tuple[int, int]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    H, W = shape
    for obj in square_objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        step, sprite = _square_sprite(obj)
        for sign in (-1, 1):
            k = 1
            while True:
                anchor_r = r0 + k * step
                anchor_c = c0 + sign * k * step
                bbox_r0 = anchor_r - 1
                bbox_c0 = anchor_c - 1
                bbox_r1 = anchor_r + step - 2
                bbox_c1 = anchor_c + step - 2
                if bbox_r0 >= H or bbox_r1 < 0 or bbox_c0 >= W or bbox_c1 < 0:
                    break
                for dr, dc, _color in sprite:
                    rr = anchor_r + dr
                    cc = anchor_c + dc
                    if _in_bounds(shape, rr, cc):
                        pixels.add((rr, cc))
                k += 1
    return pixels


def _place_square_objects(
    H: int, W: int, square_specs: list[tuple[int, bool]]
) -> list[tuple[tuple[int, int, int, int], np.ndarray]] | None:
    placed: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    placed_bboxes: list[tuple[int, int, int, int]] = []
    need_partial = True
    specs = sorted(square_specs, key=lambda item: item[0], reverse=True)
    for idx, (side, filled) in enumerate(specs):
        shape = _square_shape(side, filled)
        candidates = []
        for top in range(1, H - side):
            for left in range(1, W - side):
                bbox = (top, left, top + side - 1, left + side - 1)
                if not all(_bbox_far_enough(bbox, prev_bbox, gap=2) for prev_bbox in placed_bboxes):
                    continue
                dummy = {"bbox": bbox, "coords": [(top + r, left + c) for r, c in zip(*np.where(shape == 1))]}
                stats = _trajectory_stats((H, W), dummy)
                if min(stats[-1]["touched"], stats[1]["touched"]) < 1:
                    continue
                if max(stats[-1]["full"], stats[1]["full"]) < 2:
                    continue
                if idx == 0 and need_partial and stats[-1]["partial"] + stats[1]["partial"] == 0:
                    continue
                score = (
                    (stats[-1]["partial"] + stats[1]["partial"]) > 0,
                    stats[-1]["touched"] + stats[1]["touched"],
                    -abs((left + side / 2) - W / 2),
                )
                candidates.append((score, bbox, shape.copy()))
        if not candidates:
            return None
        top_candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:20]
        _score, bbox, chosen_shape = random.choice(top_candidates)
        placed.append((bbox, chosen_shape))
        placed_bboxes.append(bbox)
        need_partial = False
    return placed


def _place_distractors(
    H: int,
    W: int,
    square_bboxes: list[tuple[int, int, int, int]],
    square_objects: list[dict],
    distractor_count: int,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]] | None:
    placed_bboxes = list(square_bboxes)
    distractors: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    future_pixels = _future_stamp_pixels(square_objects, (H, W))
    want_blocker = distractor_count > 0 and bool(future_pixels) and random.random() < 0.7

    for idx in range(distractor_count):
        templates = random.sample(DISTRACTOR_TEMPLATES, len(DISTRACTOR_TEMPLATES))
        candidates = []
        for shape in templates:
            h, w = shape.shape
            for top in range(0, H - h + 1):
                for left in range(0, W - w + 1):
                    if not _can_place_shape(H, W, placed_bboxes, top, left, shape):
                        continue
                    bbox = (top, left, top + h - 1, left + w - 1)
                    coords = {
                        (top + r, left + c)
                        for r, c in zip(*np.where(shape == 1))
                    }
                    if want_blocker and idx == 0:
                        if not (coords & future_pixels):
                            continue
                    candidates.append((bbox, shape.copy()))
        if not candidates:
            return None
        bbox, shape = random.choice(candidates[:60] if len(candidates) > 60 else candidates)
        distractors.append((bbox, shape))
        placed_bboxes.append(bbox)
    return distractors


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.all(np.isin(grid, [BACKGROUND, SQUARE_COLOR])):
        return False

    objects = _find_components(grid)
    if not (3 <= len(objects) <= 5):
        return False

    square_objects = [obj for obj in objects if _has_complete_square_border(grid, obj)]
    distractors = [obj for obj in objects if not _has_complete_square_border(grid, obj)]
    if not (1 <= len(square_objects) <= 3):
        return False
    if len(distractors) < 1:
        return False

    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            if not _bbox_far_enough(obj_a["bbox"], obj_b["bbox"], gap=2):
                return False

    clipped_square = False
    for obj in square_objects:
        r0, c0, r1, c1 = obj["bbox"]
        for mr, mc in _corner_markers(r0, c0, r1, c1):
            if not _in_bounds(grid.shape, mr, mc):
                return False
            if int(grid[mr, mc]) != BACKGROUND:
                return False
        stats = _trajectory_stats(grid.shape, obj)
        if min(stats[-1]["touched"], stats[1]["touched"]) < 1:
            return False
        if max(stats[-1]["full"], stats[1]["full"]) < 2:
            return False
        if stats[-1]["partial"] + stats[1]["partial"] > 0:
            clipped_square = True
    if not clipped_square:
        return False

    out1 = main(grid)
    ordered = sorted(square_objects, key=lambda obj: obj["bbox"])
    out2 = _stamp_with_order(grid, ordered[::-1])
    if not np.array_equal(out1, out2):
        return False

    wrong = _wrong_stride_output(grid)
    if np.array_equal(out1, wrong):
        return False
    if np.array_equal(out1, grid):
        return False
    return True


def _build_grid(
    H: int,
    W: int,
    square_specs: list[tuple[int, bool]],
    distractor_count: int,
) -> np.ndarray | None:
    square_placements = _place_square_objects(H, W, square_specs)
    if square_placements is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    square_objects: list[dict] = []
    square_bboxes: list[tuple[int, int, int, int]] = []
    for bbox, shape in square_placements:
        top, left, _, _ = bbox
        _apply_shape(grid, top, left, shape)
        coords = [(top + r, left + c) for r, c in zip(*np.where(shape == 1))]
        square_objects.append({"bbox": bbox, "coords": coords, "area": len(coords)})
        square_bboxes.append(bbox)

    distractor_placements = _place_distractors(H, W, square_bboxes, square_objects, distractor_count)
    if distractor_placements is None:
        return None
    for bbox, shape in distractor_placements:
        top, left, _, _ = bbox
        _apply_shape(grid, top, left, shape)

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        H = random.randint(*GRID_H_RANGE)
        W = random.randint(*GRID_W_RANGE)
        square_count = random.choice(SQUARE_COUNT_CHOICES)
        distractor_count = random.choice(DISTRACTOR_COUNT_BY_SQUARES[square_count])

        sizes = [random.choice(SQUARE_SIZES) for _ in range(square_count)]
        if square_count >= 2 and len(set(sizes)) == 1 and random.random() < 0.8:
            sizes[random.randrange(square_count)] = random.choice([s for s in SQUARE_SIZES if s != sizes[0]])
        square_specs = []
        for side in sizes:
            filled = side == 2 or random.random() < 0.55
            square_specs.append((side, filled))

        grid = _build_grid(H, W, square_specs, distractor_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
