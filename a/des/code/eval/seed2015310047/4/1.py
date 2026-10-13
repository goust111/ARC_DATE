import random
from typing import Iterable

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OUTER_H_RANGE = (5, 8)
OUTER_W_RANGE = (5, 8)
BOX_COUNT_RANGE = (2, 5)
GEN_RETRIES = 500
PLACEMENT_TRIES = 200
TARGETED_TRIES = 120
RAY_GAP_RANGE = (2, 6)
BOX_PAD = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*points)
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            coords.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            coords.append((r, c0))
    return coords


def _side_from_gap(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0:
        assert c0 < gc < c1
        return "top"
    if gr == r1:
        assert c0 < gc < c1
        return "bottom"
    if gc == c0:
        assert r0 < gr < r1
        return "left"
    assert gc == c1 and r0 < gr < r1
    return "right"


def _extract_fences(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    fences: list[dict] = []
    for comp in _components4(grid == wall):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "fence must enclose an interior"

        perimeter = _perimeter_coords(r0, c0, r1, c1)
        perim_set = set(perimeter)
        comp_set = set(comp)
        assert comp_set <= perim_set, "wall pixels must lie on the rectangle perimeter"
        assert len(comp_set) == len(perimeter) - 1, "fence must have exactly one missing perimeter cell"

        gap_candidates = [cell for cell in perimeter if cell not in comp_set]
        assert len(gap_candidates) == 1, "expected a single-pixel gap"
        gap = gap_candidates[0]
        side = _side_from_gap(gap, box)
        assert int(grid[gap]) == bg, "gap should be background in the input"

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0, "interior must be non-empty"
        assert not np.any(interior == wall), "fence interior must be hollow"

        seed_positions = np.argwhere((interior != bg) & (interior != wall))
        assert seed_positions.shape[0] == 1, "expected exactly one seed in each fence"
        seed_r = int(seed_positions[0, 0] + r0 + 1)
        seed_c = int(seed_positions[0, 1] + c0 + 1)
        seed_color = int(grid[seed_r, seed_c])

        fences.append(
            {
                "bbox": box,
                "gap": gap,
                "side": side,
                "seed": (seed_r, seed_c),
                "color": seed_color,
            }
        )

    fences.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert fences, "expected at least one fence"
    return fences


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _most_common_color(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected fences and seeds"
    wall = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))

    fences = _extract_fences(g, bg, wall)
    out = g.copy()

    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        seed_r, seed_c = fence["seed"]
        color = int(fence["color"])
        side = fence["side"]

        if side in ("top", "bottom"):
            for r in range(r0 + 1, r1):
                fill = color if abs(r - seed_r) % 2 == 0 else bg
                out[r, c0 + 1 : c1] = fill
        else:
            for c in range(c0 + 1, c1):
                fill = color if abs(c - seed_c) % 2 == 0 else bg
                out[r0 + 1 : r1, c] = fill

        dr, dc = DIRS[side]
        rr, cc = fence["gap"]
        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != wall:
            out[rr, cc] = color
            rr += dr
            cc += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed-colored stripes inside each gapped rectangular fence, then emit a
    same-colored ray outward through the gap.
    """
    return _apply_rule(grid)


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = BOX_PAD) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _choose_gap(rng: random.Random, side: str, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return (r0, rng.randint(c0 + 1, c1 - 1))
    if side == "bottom":
        return (r1, rng.randint(c0 + 1, c1 - 1))
    if side == "left":
        return (rng.randint(r0 + 1, r1 - 1), c0)
    return (rng.randint(r0 + 1, r1 - 1), c1)


def _choose_seed(rng: random.Random, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (rng.randint(r0 + 1, r1 - 1), rng.randint(c0 + 1, c1 - 1))


def _wall_hit_options(target: dict, candidate_side: str) -> list[int]:
    r0, c0, r1, c1 = target["bbox"]
    gap = target["gap"]
    if candidate_side == "top":
        return [c for c in range(c0 + 1, c1) if gap != (r1, c)]
    if candidate_side == "bottom":
        return [c for c in range(c0 + 1, c1) if gap != (r0, c)]
    if candidate_side == "left":
        return [r for r in range(r0 + 1, r1) if gap != (r, c1)]
    return [r for r in range(r0 + 1, r1) if gap != (r, c0)]


def _candidate_in_bounds(h: int, w: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _free_box(rng: random.Random, h: int, w: int, boxes: list[dict]) -> dict | None:
    for _ in range(PLACEMENT_TRIES):
        box_h = rng.randint(*OUTER_H_RANGE)
        box_w = rng.randint(*OUTER_W_RANGE)
        side = rng.choice(SIDES)

        min_r0 = 2 if side == "top" else 1
        max_r0 = h - box_h - (2 if side == "bottom" else 1)
        min_c0 = 2 if side == "left" else 1
        max_c0 = w - box_w - (2 if side == "right" else 1)
        if min_r0 > max_r0 or min_c0 > max_c0:
            continue

        r0 = rng.randint(min_r0, max_r0)
        c0 = rng.randint(min_c0, max_c0)
        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if not all(_bboxes_separated(bbox, other["bbox"]) for other in boxes):
            continue

        return {
            "bbox": bbox,
            "side": side,
            "gap": _choose_gap(rng, side, bbox),
            "seed": _choose_seed(rng, bbox),
        }
    return None


def _targeted_box(rng: random.Random, h: int, w: int, boxes: list[dict]) -> dict | None:
    if not boxes:
        return None

    for _ in range(TARGETED_TRIES):
        target = rng.choice(boxes)
        side = rng.choice(SIDES)
        box_h = rng.randint(*OUTER_H_RANGE)
        box_w = rng.randint(*OUTER_W_RANGE)
        options = _wall_hit_options(target, side)
        if not options:
            continue

        gap_distance = rng.randint(*RAY_GAP_RANGE)
        tr0, tc0, tr1, tc1 = target["bbox"]

        if side in ("top", "bottom"):
            hit_col = rng.choice(options)
            offset = rng.randint(1, box_w - 2)
            c0 = hit_col - offset
            c1 = c0 + box_w - 1
            if side == "top":
                r0 = tr1 + gap_distance
                r1 = r0 + box_h - 1
                gap = (r0, hit_col)
            else:
                r1 = tr0 - gap_distance
                r0 = r1 - box_h + 1
                gap = (r1, hit_col)
            bbox = (r0, c0, r1, c1)
        else:
            hit_row = rng.choice(options)
            offset = rng.randint(1, box_h - 2)
            r0 = hit_row - offset
            r1 = r0 + box_h - 1
            if side == "left":
                c0 = tc1 + gap_distance
                c1 = c0 + box_w - 1
                gap = (hit_row, c0)
            else:
                c1 = tc0 - gap_distance
                c0 = c1 - box_w + 1
                gap = (hit_row, c1)
            bbox = (r0, c0, r1, c1)

        if not _candidate_in_bounds(h, w, bbox):
            continue
        if not all(_bboxes_separated(bbox, other["bbox"]) for other in boxes):
            continue

        return {
            "bbox": bbox,
            "side": side,
            "gap": gap,
            "seed": _choose_seed(rng, bbox),
        }

    return None


def _all_bbox_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _all_wall_cells(boxes: list[dict]) -> set[tuple[int, int]]:
    walls: set[tuple[int, int]] = set()
    for box in boxes:
        gap = box["gap"]
        for cell in _perimeter_coords(*box["bbox"]):
            if cell != gap:
                walls.add(cell)
    return walls


def _ray_cells(box: dict, h: int, w: int, walls: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], str]:
    dr, dc = DIRS[box["side"]]
    rr, cc = box["gap"]
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w and (rr, cc) not in walls:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    stop_reason = "wall" if 0 <= rr < h and 0 <= cc < w else "boundary"
    return cells, stop_reason


def _validate_layout(h: int, w: int, boxes: list[dict], require_wall_stop: bool) -> bool:
    walls = _all_wall_cells(boxes)
    bbox_cells = [_all_bbox_cells(box["bbox"]) for box in boxes]
    seen_rays: set[tuple[int, int]] = set()
    saw_wall_stop = False

    for idx, box in enumerate(boxes):
        ray, stop_reason = _ray_cells(box, h, w, walls)
        if len(ray) < 2:
            return False
        if stop_reason == "wall":
            saw_wall_stop = True

        for j, other_cells in enumerate(bbox_cells):
            if j != idx and any(cell in other_cells for cell in ray):
                return False
        ray_set = set(ray)
        if ray_set & seen_rays:
            return False
        seen_rays |= ray_set

    return saw_wall_stop or not require_wall_stop


def _draw_grid(rng: random.Random, h: int, w: int, boxes: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG_COLOR, dtype=int)
    colors = rng.sample(SEED_COLORS, k=len(boxes))
    for box, color in zip(boxes, colors):
        box["color"] = int(color)
        for cell in _perimeter_coords(*box["bbox"]):
            if cell != box["gap"]:
                grid[cell] = WALL_COLOR
        sr, sc = box["seed"]
        grid[sr, sc] = int(color)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    max_boxes = min(BOX_COUNT_RANGE[1], len(SEED_COLORS), max(BOX_COUNT_RANGE[0], (h * w) // 90))
    count = rng.randint(BOX_COUNT_RANGE[0], max_boxes)
    require_wall_stop = rng.random() < 0.5

    boxes: list[dict] = []
    blocked_box_made = False
    for idx in range(count):
        box = None
        if idx > 0 and require_wall_stop and not blocked_box_made:
            box = _targeted_box(rng, h, w, boxes)
            blocked_box_made = box is not None
        if box is None:
            box = _free_box(rng, h, w, boxes)
        if box is None:
            return None
        boxes.append(box)

    if not _validate_layout(h, w, boxes, require_wall_stop):
        return None

    grid = _draw_grid(rng, h, w, boxes)

    try:
        fences = _extract_fences(grid, BG_COLOR, WALL_COLOR)
        if len(fences) != len(boxes):
            return None
        out = _apply_rule(grid)
        if np.array_equal(grid, out):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-fence ARC input")
