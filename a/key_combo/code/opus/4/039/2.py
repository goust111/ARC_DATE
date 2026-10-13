import numpy as np
from collections import deque


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MARKER_COLORS = (BLUE, RED, GREEN)

GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (10, 17)
GEN_RETRIES = 400
PLACE_RETRIES = 300
MIN_GAP_BETWEEN_MARKERS = 3
IRREGULAR_PROB = 0.35


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _enclosed_black_mask(sprite: np.ndarray) -> np.ndarray:
    h, w = sprite.shape
    black = sprite == BLACK
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))
    return black & ~exterior


def _classify_hole(sprite: np.ndarray) -> tuple[bool, np.ndarray]:
    hole_mask = _enclosed_black_mask(sprite)
    hole_comps = components_mask(hole_mask, connectivity=4)
    if len(hole_comps) != 1:
        return False, hole_mask

    comp_mask = np.zeros_like(hole_mask, dtype=bool)
    coords = hole_comps[0]
    comp_mask[coords[:, 0], coords[:, 1]] = True
    box = bbox(comp_mask)
    if box is None:
        return False, comp_mask
    r0, c0, r1, c1 = box
    crop = comp_mask[r0 : r1 + 1, c0 : c1 + 1]
    intercepts = crop.shape[0] == crop.shape[1] and bool(np.all(crop))
    return intercepts, comp_mask


def _find_objects(grid: np.ndarray) -> list[dict]:
    out: list[dict] = []
    for coords in components_mask(grid == GREY, connectivity=4):
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        box = bbox(mask)
        if box is None:
            continue
        sprite = crop_bbox(grid, box)
        intercepts, hole_mask = _classify_hole(sprite)
        out.append(
            {
                "bbox": box,
                "sprite": sprite,
                "intercepts": intercepts,
                "hole_mask": hole_mask,
            }
        )
    out.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return out


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[str, tuple[int, int, int, int] | None]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _find_objects(g)
    markers = [(int(c), int(g[0, c])) for c in range(g.shape[1]) if int(g[0, c]) in MARKER_COLORS]
    outcomes: dict[int, tuple[str, tuple[int, int, int, int] | None]] = {}

    for c, color in markers:
        out[0, c] = BLACK
        relevant = [obj for obj in objects if obj["bbox"][1] <= c <= obj["bbox"][3]]
        relevant.sort(key=lambda item: item["bbox"][0])
        stopped = False
        for obj in relevant:
            if not obj["intercepts"]:
                continue
            r0, c0, _, _ = obj["bbox"]
            hole_mask = obj["hole_mask"]
            rows, cols = np.where(hole_mask)
            out[r0 + rows, c0 + cols] = color
            outcomes[c] = ("intercept", obj["bbox"])
            stopped = True
            break
        if not stopped:
            out[-1, c] = color
            outcomes[c] = ("bottom", None)
    return out, outcomes


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Drop top-row colored markers through gray hollow rectangles and stop only inside square holes.

    Input structure:
    - The background is black.
    - The top row contains 2 or 3 single-pixel markers in distinct columns, using red, blue, and/or green.
    - The middle and lower area contains several disconnected gray rectangular objects whose interiors include one enclosed black hole.

    Rule:
    - Process each top-row marker independently by scanning downward along its column.
    - Whenever the column overlaps a gray object's column span, inspect that object's enclosed black hole inside the object's bounding box.
    - If the hole is one connected solid square, fill that hole with the marker color and stop that marker there.
    - Otherwise the marker passes through that object and continues downward; if no object intercepts it, place the marker on the bottom row in the same column.
    - Remove the original top-row marker pixels.

    Key insight:
    - The gray objects are filtered by the geometry of their enclosed holes: only a fully filled square hole acts as a catcher, so the final landing point requires combining gravity order with per-object hole classification.
    """
    out, _ = _simulate(np.asarray(input_grid, dtype=int))
    return out


def _marker_columns(rng: np.random.Generator, width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    if len(candidates) < count:
        return None
    for _ in range(200):
        cols = sorted(int(x) for x in rng.choice(candidates, size=count, replace=False))
        if min(cols[i + 1] - cols[i] for i in range(len(cols) - 1)) >= MIN_GAP_BETWEEN_MARKERS:
            return cols
    return None


def _square_hole_object(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h = int(rng.integers(4, 7))
    w = int(rng.integers(4, 7))
    s = int(rng.integers(2, min(h, w) - 1))
    top = int(rng.integers(1, h - s))
    left = int(rng.integers(1, w - s))
    sprite = np.full((h, w), GREY, dtype=int)
    sprite[top : top + s, left : left + s] = BLACK
    hole = np.zeros((h, w), dtype=bool)
    hole[top : top + s, left : left + s] = True
    return sprite, hole


def _rectangular_non_square_object(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        hh = int(rng.integers(1, h - 1))
        hw = int(rng.integers(1, w - 1))
        if hh == hw:
            continue
        top = int(rng.integers(1, h - hh))
        left = int(rng.integers(1, w - hw))
        sprite = np.full((h, w), GREY, dtype=int)
        sprite[top : top + hh, left : left + hw] = BLACK
        hole = np.zeros((h, w), dtype=bool)
        hole[top : top + hh, left : left + hw] = True
        return sprite, hole
    raise RuntimeError("failed to build non-square rectangular hole")


def _irregular_square_bbox_object(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shapes = {
        2: [
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (1, 0), (1, 1)],
        ],
        3: [
            [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
            [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0)],
            [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
        ],
    }
    for _ in range(100):
        h = int(rng.integers(4, 7))
        w = int(rng.integers(4, 7))
        side_choices = [side for side in (2, 3) if side <= h - 2 and side <= w - 2]
        if not side_choices:
            continue
        side = int(rng.choice(side_choices))
        shape = list(shapes[side][int(rng.integers(0, len(shapes[side])))] )
        top = int(rng.integers(1, h - side))
        left = int(rng.integers(1, w - side))
        sprite = np.full((h, w), GREY, dtype=int)
        hole = np.zeros((h, w), dtype=bool)
        for dr, dc in shape:
            sprite[top + dr, left + dc] = BLACK
            hole[top + dr, left + dc] = True
        return sprite, hole
    raise RuntimeError("failed to build irregular square-bbox hole")


def _make_object(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if kind == "square":
        return _square_hole_object(rng)
    if kind == "rect":
        return _rectangular_non_square_object(rng)
    if kind == "irregular":
        return _irregular_square_bbox_object(rng)
    raise ValueError(f"unknown object kind: {kind}")


def _non_intercept_kind(rng: np.random.Generator) -> str:
    return "irregular" if rng.random() < IRREGULAR_PROB else "rect"


def _lane_plans(rng: np.random.Generator, count: int) -> list[list[str]]:
    base = [["rect", "square"], ["rect", _non_intercept_kind(rng)]]
    if count == 2:
        rng.shuffle(base)
        return base

    plans = base[:]
    extra_count = int(rng.choice([1, 2]))
    extra_type = "intercept" if rng.random() < 0.5 else "bottom"
    if extra_type == "intercept":
        plans.append(["square"] if extra_count == 1 else ["rect", "square"])
    else:
        plans.append([_non_intercept_kind(rng)] if extra_count == 1 else [_non_intercept_kind(rng), _non_intercept_kind(rng)])

    while sum(len(plan) for plan in plans) < 4:
        idx = int(rng.integers(0, len(plans)))
        if len(plans[idx]) == 1:
            if "square" in plans[idx]:
                plans[idx] = ["rect", "square"]
            else:
                plans[idx].append(_non_intercept_kind(rng))
    while sum(len(plan) for plan in plans) > 6:
        removable = [i for i, plan in enumerate(plans) if len(plan) == 2 and plan[-1] != "square"]
        if not removable:
            break
        plans[int(rng.choice(removable))] = plans[int(rng.choice(removable))][:1]

    rng.shuffle(plans)
    return plans


def _object_overlaps_other_marker(c0: int, width: int, target_col: int, marker_cols: list[int]) -> bool:
    c1 = c0 + width - 1
    for col in marker_cols:
        if col != target_col and c0 <= col <= c1:
            return True
    return False


def _place_objects(
    grid: np.ndarray,
    rng: np.random.Generator,
    marker_cols: list[int],
    plans: list[list[str]],
) -> tuple[list[dict], dict[int, str]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []
    expected_by_col: dict[int, str] = {}

    for marker_col, plan in zip(marker_cols, plans):
        min_row = 3
        expected_by_col[marker_col] = "bottom"
        for kind in plan:
            placed = False
            for _ in range(PLACE_RETRIES):
                sprite, hole = _make_object(kind, rng)
                oh, ow = sprite.shape
                c0_candidates = [c0 for c0 in range(max(0, marker_col - ow + 1), min(marker_col, w - ow) + 1)]
                rng.shuffle(c0_candidates)
                for c0 in c0_candidates:
                    if _object_overlaps_other_marker(c0, ow, marker_col, marker_cols):
                        continue
                    max_row = h - oh - 2
                    if min_row > max_row:
                        continue
                    row_candidates = list(range(min_row, max_row + 1))
                    rng.shuffle(row_candidates)
                    for r0 in row_candidates[: min(40, len(row_candidates))]:
                        r1 = r0 + oh - 1
                        c1 = c0 + ow - 1
                        rr0 = max(0, r0 - 1)
                        rr1 = min(h, r1 + 2)
                        cc0 = max(0, c0 - 1)
                        cc1 = min(w, c1 + 2)
                        if np.any(blocked[rr0:rr1, cc0:cc1]):
                            continue
                        grid[r0 : r1 + 1, c0 : c1 + 1] = sprite
                        blocked[rr0:rr1, cc0:cc1] = True
                        placements.append(
                            {
                                "bbox": (r0, c0, r1, c1),
                                "hole_coords": [(r0 + int(hr), c0 + int(hc)) for hr, hc in np.argwhere(hole)],
                                "kind": kind,
                                "marker_col": marker_col,
                            }
                        )
                        min_row = r1 + 2
                        if kind == "square":
                            expected_by_col[marker_col] = "intercept"
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                return None
    return placements, expected_by_col


def _expected_output_from_plan(
    grid: np.ndarray,
    marker_info: list[tuple[int, int]],
    placements: list[dict],
) -> tuple[np.ndarray, dict[int, str]]:
    out = grid.copy()
    outcomes: dict[int, str] = {}
    by_col: dict[int, list[dict]] = {}
    for item in placements:
        by_col.setdefault(int(item["marker_col"]), []).append(item)
    for col in by_col:
        by_col[col].sort(key=lambda item: item["bbox"][0])

    for col, color in marker_info:
        out[0, col] = BLACK
        stopped = False
        for item in by_col.get(col, []):
            if item["kind"] != "square":
                continue
            for r, c in item["hole_coords"]:
                out[r, c] = color
            outcomes[col] = "intercept"
            stopped = True
            break
        if not stopped:
            out[-1, col] = color
            outcomes[col] = "bottom"
    return out, outcomes


def _objects_are_separated(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        for rr0, cc0, rr1, cc1 in boxes[i + 1 :]:
            if not (r1 + 1 < rr0 or rr1 + 1 < r0 or c1 + 1 < cc0 or cc1 + 1 < c0):
                return False
    return True


def _validate_case(
    grid: np.ndarray,
    marker_info: list[tuple[int, int]],
    placements: list[dict],
    expected_by_col: dict[int, str],
) -> bool:
    objects = _find_objects(grid)
    if not (4 <= len(objects) <= 6):
        return False
    if len(marker_info) not in (2, 3):
        return False
    if len({color for _, color in marker_info}) != len(marker_info):
        return False
    if not _objects_are_separated([obj["bbox"] for obj in objects]):
        return False

    for col, _ in marker_info:
        overlaps = [obj for obj in objects if obj["bbox"][1] <= col <= obj["bbox"][3]]
        if not (1 <= len(overlaps) <= 2):
            return False

    actual_out, actual_events = _simulate(grid)
    planned_out, planned_events = _expected_output_from_plan(grid, marker_info, placements)
    if not np.array_equal(actual_out, planned_out):
        return False
    if any(actual_events[col][0] != expected_by_col[col] for col, _ in marker_info):
        return False
    if set(planned_events.values()) != {"intercept", "bottom"} and len(marker_info) >= 2:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    marker_count = int(rng.choice([2, 3]))
    h = int(rng.integers(*GRID_H_RANGE))
    w_low = max(GRID_W_RANGE[0], 12 if marker_count == 3 else GRID_W_RANGE[0])
    w = int(rng.integers(w_low, GRID_W_RANGE[1]))
    cols = _marker_columns(rng, w, marker_count)
    if cols is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    colors = list(rng.choice(MARKER_COLORS, size=marker_count, replace=False))
    marker_info = list(zip(cols, colors))
    for col, color in marker_info:
        grid[0, col] = int(color)

    plans = _lane_plans(rng, marker_count)
    placed = _place_objects(grid, rng, cols, plans)
    if placed is None:
        return None
    placements, expected_by_col = placed

    if not _validate_case(grid, marker_info, placements, expected_by_col):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
