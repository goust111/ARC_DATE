import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SIDES = ("top", "bottom", "left", "right")
BORDER_COLORS = (RED, BLUE, YELLOW, GREEN)
BORDER_COUNTS = (1, 2, 3, 4)
GEN_ATTEMPTS = 300
PLACE_ATTEMPTS = 160
MASK_ATTEMPTS = 100
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            region: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                region.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes.append(region)
    return holes


def _object_records(grid: np.ndarray) -> list[dict]:
    records: list[dict] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True
        holes = _hole_regions(mask)
        records.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "holes": holes,
                "hole_count": len(holes),
            }
        )
    return records


def _edge_info(grid: np.ndarray) -> dict[str, tuple[int, int]]:
    h, w = grid.shape
    edges = {
        "top": grid[0, 1 : w - 1],
        "bottom": grid[h - 1, 1 : w - 1],
        "left": grid[1 : h - 1, 0],
        "right": grid[1 : h - 1, w - 1],
    }
    out: dict[str, tuple[int, int]] = {}
    for side, line in edges.items():
        nz = [int(v) for v in line if int(v) != BLACK]
        colors = set(nz)
        assert len(colors) == 1 and nz
        out[side] = (len(nz), nz[0])
    return out


def _target_delta(
    side: str, bbox: tuple[int, int, int, int], grid_shape: tuple[int, int]
) -> tuple[int, int]:
    h, w = grid_shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return 1 - r0, 0
    if side == "bottom":
        return (h - 2) - r1, 0
    if side == "left":
        return 0, 1 - c0
    return 0, (w - 2) - c1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    edge_info = _edge_info(g)
    count_to_side = {count: (side, color) for side, (count, color) in edge_info.items()}
    out = g.copy()
    out[g == GREY] = BLACK

    for obj in _object_records(g):
        hole_count = obj["hole_count"]
        assert hole_count in count_to_side
        side, fill_color = count_to_side[hole_count]
        dr, dc = _target_delta(side, obj["bbox"], g.shape)
        r0, c0, _, _ = obj["bbox"]
        new_r0 = r0 + dr
        new_c0 = c0 + dc
        mask = obj["mask"]
        for rr, cc in np.argwhere(mask):
            out[new_r0 + int(rr), new_c0 + int(cc)] = GREY
        for hole in obj["holes"]:
            for rr, cc in hole:
                out[new_r0 + int(rr), new_c0 + int(cc)] = fill_color
    return out


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _box_from_start(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + h - 1, c0 + w - 1


def _corridor_box(
    start_box: tuple[int, int, int, int], end_box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return (
        min(start_box[0], end_box[0]),
        min(start_box[1], end_box[1]),
        max(start_box[2], end_box[2]),
        max(start_box[3], end_box[3]),
    )


def _spread_positions(length: int, count: int) -> list[int]:
    raw = np.rint(np.linspace(0, length - 1, count + 2)[1:-1]).astype(int)
    return [int(x) for x in raw]


def _paint_border(
    grid: np.ndarray, side_counts: dict[str, int], side_colors: dict[str, int]
) -> None:
    h, w = grid.shape
    for side in SIDES:
        count = int(side_counts[side])
        color = int(side_colors[side])
        if side == "top":
            for off in _spread_positions(w - 2, count):
                grid[0, 1 + off] = color
        elif side == "bottom":
            for off in _spread_positions(w - 2, count):
                grid[h - 1, 1 + off] = color
        elif side == "left":
            for off in _spread_positions(h - 2, count):
                grid[1 + off, 0] = color
        else:
            for off in _spread_positions(h - 2, count):
                grid[1 + off, w - 1] = color


def _random_object_mask(hole_count: int) -> np.ndarray:
    layouts = [
        (rows, cols)
        for rows in range(1, min(3, hole_count) + 1)
        for cols in range(1, 4)
        if rows * cols >= hole_count and rows * cols <= hole_count + 2
    ]
    for _ in range(MASK_ATTEMPTS):
        rows, cols = random.choice(layouts)
        hole_h, hole_w = random.choice(((1, 1), (1, 2), (2, 1)))
        pad_t = random.choice((1, 1, 2))
        pad_b = random.choice((1, 1, 2))
        pad_l = random.choice((1, 1, 2))
        pad_r = random.choice((1, 1, 2))
        h = pad_t + pad_b + rows * hole_h + rows + 1
        w = pad_l + pad_r + cols * hole_w + cols + 1
        if h > 8 or w > 8:
            continue

        mask = np.ones((h, w), dtype=bool)
        slots = [(rr, cc) for rr in range(rows) for cc in range(cols)]
        for rr, cc in random.sample(slots, hole_count):
            r0 = pad_t + 1 + rr * (hole_h + 1)
            c0 = pad_l + 1 + cc * (hole_w + 1)
            mask[r0 : r0 + hole_h, c0 : c0 + hole_w] = False

        for _ in range(random.randint(0, 2)):
            side = random.choice(SIDES)
            if side in ("top", "bottom") and w >= 5:
                length = random.randint(1, max(1, w // 3))
                start = random.randint(1, w - length - 2)
                row = 0 if side == "top" else h - 1
                mask[row, start : start + length] = False
            if side in ("left", "right") and h >= 5:
                length = random.randint(1, max(1, h // 3))
                start = random.randint(1, h - length - 2)
                col = 0 if side == "left" else w - 1
                mask[start : start + length, col] = False

        if len(_components(mask)) == 1 and len(_hole_regions(mask)) == hole_count:
            return mask
    raise RuntimeError("failed to create object mask")


def _sample_boxes(
    side: str, mask_shape: tuple[int, int], grid_shape: tuple[int, int], touch: bool
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    grid_h, grid_w = grid_shape
    obj_h, obj_w = mask_shape
    if side in ("top", "bottom"):
        c0_min = 1
        c0_max = grid_w - obj_w - 1
        if c0_min > c0_max:
            return None
        c0 = random.randint(c0_min, c0_max)
        final_r0 = 1 if side == "top" else grid_h - obj_h - 1
        if side == "top":
            r0_min, r0_max = (1, 1) if touch else (2, grid_h - obj_h - 1)
        else:
            r0_min, r0_max = (final_r0, final_r0) if touch else (1, final_r0 - 1)
        if r0_min > r0_max:
            return None
        r0 = random.randint(r0_min, r0_max)
        start_box = _box_from_start(r0, c0, obj_h, obj_w)
        end_box = _box_from_start(final_r0, c0, obj_h, obj_w)
        return start_box, end_box

    r0_min = 1
    r0_max = grid_h - obj_h - 1
    if r0_min > r0_max:
        return None
    r0 = random.randint(r0_min, r0_max)
    final_c0 = 1 if side == "left" else grid_w - obj_w - 1
    if side == "left":
        c0_min, c0_max = (1, 1) if touch else (2, grid_w - obj_w - 1)
    else:
        c0_min, c0_max = (final_c0, final_c0) if touch else (1, final_c0 - 1)
    if c0_min > c0_max:
        return None
    c0 = random.randint(c0_min, c0_max)
    start_box = _box_from_start(r0, c0, obj_h, obj_w)
    end_box = _box_from_start(r0, final_c0, obj_h, obj_w)
    return start_box, end_box


def _can_place(
    start_box: tuple[int, int, int, int],
    end_box: tuple[int, int, int, int],
    placed: list[dict],
) -> bool:
    corridor = _corridor_box(start_box, end_box)
    for other in placed:
        if _boxes_overlap(start_box, other["start_box"], pad=1):
            return False
        if _boxes_overlap(end_box, other["end_box"], pad=1):
            return False
        if _boxes_overlap(corridor, other["start_box"]):
            return False
        if _boxes_overlap(other["corridor"], start_box):
            return False
    return True


def _place_mask(
    grid: np.ndarray, mask: np.ndarray, start_box: tuple[int, int, int, int]
) -> None:
    r0, c0, _, _ = start_box
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = GREY


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)

    side_counts = dict(zip(SIDES, random.sample(BORDER_COUNTS, len(SIDES))))
    side_colors = dict(zip(SIDES, random.sample(BORDER_COLORS, len(SIDES))))
    _paint_border(grid, side_counts, side_colors)

    n_objects = random.randint(2, 4)
    chosen_sides = random.sample(list(SIDES), n_objects)
    items = []
    for side in chosen_sides:
        hole_count = int(side_counts[side])
        items.append(
            {
                "side": side,
                "hole_count": hole_count,
                "mask": _random_object_mask(hole_count),
                "touch_preference": random.random() < 0.5,
            }
        )
    items.sort(key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True)

    placed: list[dict] = []
    for item in items:
        success = False
        for attempt in range(PLACE_ATTEMPTS):
            prefer_touch = item["touch_preference"]
            touch = prefer_touch if attempt < PLACE_ATTEMPTS // 2 else (not prefer_touch)
            sampled = _sample_boxes(item["side"], item["mask"].shape, grid.shape, touch)
            if sampled is None:
                continue
            start_box, end_box = sampled
            if not _can_place(start_box, end_box, placed):
                continue
            _place_mask(grid, item["mask"], start_box)
            placed.append(
                {
                    "start_box": start_box,
                    "end_box": end_box,
                    "corridor": _corridor_box(start_box, end_box),
                }
            )
            success = True
            break
        if not success:
            return None

    objects = _object_records(grid)
    if len(objects) != n_objects:
        return None
    if sorted(obj["hole_count"] for obj in objects) != sorted(
        side_counts[side] for side in chosen_sides
    ):
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
