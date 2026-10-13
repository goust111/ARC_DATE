from __future__ import annotations

from collections import Counter

import numpy as np


BG_COLOR = 0
FLOOR_COLOR = 5
WALL_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
RNG = np.random.default_rng()


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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

    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in neighbors:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _shape_mask_from_intervals(intervals: list[tuple[int, int]]) -> np.ndarray:
    width = max(end for _start, end in intervals)
    mask = np.zeros((len(intervals), width), dtype=bool)
    for r, (start, end) in enumerate(intervals):
        mask[r, start:end] = True
    used = np.any(mask, axis=0)
    return mask[:, used]


def _random_solid_mask(
    rng: np.random.Generator,
    height_range: tuple[int, int],
    width_range: tuple[int, int],
) -> np.ndarray:
    for _ in range(80):
        h = int(rng.integers(height_range[0], height_range[1] + 1))
        w = int(rng.integers(width_range[0], width_range[1] + 1))
        left = int(rng.integers(0, max(1, w - 1)))
        right = int(rng.integers(left + 2, w + 1))
        intervals = [(left, right)]
        for _row in range(1, h):
            prev_left, prev_right = intervals[-1]
            candidates = []
            for new_left in range(max(0, prev_left - 1), min(w - 2, prev_left + 1) + 1):
                lo = max(new_left + 2, prev_right - 1)
                hi = min(w, prev_right + 1)
                for new_right in range(lo, hi + 1):
                    if max(new_left, prev_left) < min(new_right, prev_right):
                        candidates.append((new_left, new_right))
            if not candidates:
                break
            intervals.append(candidates[int(rng.integers(len(candidates)))])
        if len(intervals) != h:
            continue

        mask = _shape_mask_from_intervals(intervals)
        area = int(mask.sum())
        if area < 4 or area > 18:
            continue
        if rng.random() < 0.5:
            mask = mask[:, ::-1]
        if rng.random() < 0.5:
            mask = mask[::-1, :]
        return mask
    raise RuntimeError("failed to sample a solid mask")


def _random_rectangle_mask(
    rng: np.random.Generator,
    height_range: tuple[int, int],
    width_range: tuple[int, int],
) -> np.ndarray:
    h = int(rng.integers(height_range[0], height_range[1] + 1))
    w = int(rng.integers(width_range[0], width_range[1] + 1))
    return np.ones((h, w), dtype=bool)


def _object_from_mask(mask: np.ndarray, color: int, r0: int, c0: int) -> dict:
    rel = np.argwhere(mask)
    coords = rel + np.array([[int(r0), int(c0)]], dtype=int)
    return {
        "color": int(color),
        "coords": coords.astype(int),
        "bbox": bbox_coords(coords),
        "shape": mask.copy(),
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 5 and w >= 5

    left_color = int(g[0, 0])
    right_color = int(g[0, w - 1])
    floor_color = int(Counter(map(int, g[h - 1, :])).most_common(1)[0][0])

    assert left_color != right_color
    assert np.all(g[:-1, 0] == left_color)
    assert np.all(g[:-1, w - 1] == right_color)
    assert np.all(g[h - 1, :] == floor_color)

    interior = g[:-1, 1:-1]
    bg = int(Counter(map(int, interior.flatten())).most_common(1)[0][0])

    objects: list[dict] = []
    for color in sorted(set(map(int, np.unique(interior))) - {bg}):
        mask = np.zeros_like(g, dtype=bool)
        mask[:-1, 1:-1] = interior == color
        for coords in components_mask(mask, connectivity=4):
            obj = {
                "color": int(color),
                "coords": coords.astype(int),
                "bbox": bbox_coords(coords),
            }
            objects.append(obj)

    return bg, left_color, right_color, floor_color, objects


def _drop_coords(coords: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    h, _w = occupied.shape
    drop = 0
    while True:
        shifted = coords + np.array([[drop + 1, 0]], dtype=int)
        if np.any(shifted[:, 0] >= h):
            break
        if np.any(occupied[shifted[:, 0], shifted[:, 1]]):
            break
        drop += 1
    return coords + np.array([[drop, 0]], dtype=int)


def _shear_coords(coords: np.ndarray, direction: int) -> tuple[np.ndarray, int]:
    bottom = int(coords[:, 0].max())
    sheared = coords.copy()
    sheared[:, 1] = coords[:, 1] + direction * (bottom - coords[:, 0])
    return sheared, bottom


def _simulate_objects(
    objects: list[dict],
    shape: tuple[int, int],
    left_color: int,
    right_color: int,
) -> list[dict]:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, 0] = True
    occupied[:, w - 1] = True
    occupied[h - 1, :] = True

    settled: list[dict] = []
    ordered = sorted(objects, key=lambda obj: (-obj["bbox"][2], -obj["bbox"][0], obj["bbox"][1]))
    for obj in ordered:
        landed = _drop_coords(obj["coords"], occupied)
        direction = -1 if int(obj["color"]) == int(left_color) else 1
        sheared, anchor_bottom = _shear_coords(landed, direction)
        settled.append(
            {
                "color": int(obj["color"]),
                "initial": obj["coords"].copy(),
                "landed": landed,
                "sheared": sheared,
                "anchor_bottom": int(anchor_bottom),
                "direction": int(direction),
            }
        )
        occupied[landed[:, 0], landed[:, 1]] = True

    return settled


def _render_output(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg, left_color, right_color, _floor_color, objects = _extract_scene(g)
    out = np.full_like(g, bg)
    out[:, 0] = g[:, 0]
    out[:, -1] = g[:, -1]
    out[-1, :] = g[-1, :]

    occupied = out != bg
    settled = _simulate_objects(objects, g.shape, left_color, right_color)
    for obj in settled:
        coords = obj["sheared"]
        assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < g.shape[0] - 1)
        assert np.all(coords[:, 1] > 0) and np.all(coords[:, 1] < g.shape[1] - 1)
        assert not np.any(occupied[coords[:, 0], coords[:, 1]])
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
        occupied[coords[:, 0], coords[:, 1]] = True
    return out, settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop floating wall-colored objects into a stack, then shear each settled object toward its matching wall from its own landing row.

    Input structure:
    - The left boundary above the floor is a full-height wall of one color, and
      the right boundary above the floor is a full-height wall of a different color.
    - The bottom row is a solid gray floor.
    - Between the walls are several disconnected solid monochrome objects; each
      object uses either the left-wall color or the right-wall color.

    Rule:
    - Ignore the floating objects' original support and process them by increasing
      initial bottom row, from lower objects to higher ones.
    - Each object falls straight down until one of its pixels would hit the floor
      or a previously settled object.
    - After all objects have landed, shear each object independently around its
      own landed bottom row: a pixel d rows above that row shifts d columns toward
      the wall that has the same color as the object.
    - Keep both side walls and the gray floor unchanged.

    Key insight:
    - The anchor row is not global: every object first gets a private landing row
      from the gravity-and-stacking phase, and only then tilts row-by-row toward
      the same-colored side wall.
    """
    out, _settled = _render_output(grid)
    return out


def _safe_start_bounds(mask: np.ndarray, direction: int, width: int) -> tuple[int, int]:
    rel = np.argwhere(mask)
    h = mask.shape[0]
    deltas = [int(c + direction * ((h - 1) - r)) for r, c in rel]
    min_delta = min(deltas)
    max_delta = max(deltas)
    lo = max(2, 2 - min_delta)
    hi = min(width - mask.shape[1] - 2, width - 3 - max_delta)
    return int(lo), int(hi)


def _can_place(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if occupied[rr, cc]:
            return False
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _paint_occupied(occupied: np.ndarray, coords: np.ndarray) -> None:
    occupied[coords[:, 0], coords[:, 1]] = True


def _sample_row(
    rng: np.random.Generator,
    mask: np.ndarray,
    c0: int,
    row_lo: int,
    row_hi: int,
    occupied: np.ndarray,
) -> dict | None:
    if row_lo > row_hi:
        return None
    rows = list(range(int(row_lo), int(row_hi) + 1))
    rng.shuffle(rows)
    for r0 in rows:
        obj = _object_from_mask(mask, 0, r0, c0)
        if _can_place(obj["coords"], occupied):
            return obj
    return None


def _touching_or_overlapping(a: np.ndarray, b: np.ndarray) -> bool:
    bset = {tuple(map(int, xy)) for xy in b}
    for r, c in a:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in bset:
                    return True
    return False


def _valid_generated_scene(grid: np.ndarray) -> bool:
    try:
        out, settled = _render_output(grid)
    except AssertionError:
        return False

    h, w = grid.shape
    if np.array_equal(out, grid):
        return False

    left_color = int(grid[0, 0])
    right_color = int(grid[0, w - 1])
    colors = {int(obj["color"]) for obj in settled}
    if colors != {left_color, right_color}:
        return False

    bottoms = [int(obj["anchor_bottom"]) for obj in settled]
    if not any(bottom == h - 2 for bottom in bottoms):
        return False
    if not any(bottom < h - 2 for bottom in bottoms):
        return False

    for obj in settled:
        coords = obj["sheared"]
        if np.min(coords[:, 1]) <= 1 or np.max(coords[:, 1]) >= w - 2:
            return False

    for i in range(len(settled)):
        for j in range(i + 1, len(settled)):
            if _touching_or_overlapping(settled[i]["sheared"], settled[j]["sheared"]):
                return False

    return True


def _valid_settled_layout(
    settled: list[dict],
    shape: tuple[int, int],
    left_color: int,
    right_color: int,
) -> bool:
    h, w = shape
    colors = {int(obj["color"]) for obj in settled}
    if colors != {left_color, right_color}:
        return False

    bottoms = [int(obj["anchor_bottom"]) for obj in settled]
    if not any(bottom == h - 2 for bottom in bottoms):
        return False
    if not any(bottom < h - 2 for bottom in bottoms):
        return False

    for obj in settled:
        coords = obj["sheared"]
        if np.min(coords[:, 1]) <= 1 or np.max(coords[:, 1]) >= w - 2:
            return False

    for i in range(len(settled)):
        for j in range(i + 1, len(settled)):
            if _touching_or_overlapping(settled[i]["sheared"], settled[j]["sheared"]):
                return False
    return True


def _pick_center_pair_columns(
    rng: np.random.Generator,
    mask_a: np.ndarray,
    dir_a: int,
    mask_b: np.ndarray,
    dir_b: int,
    width: int,
) -> tuple[int, int] | None:
    lo_a, hi_a = _safe_start_bounds(mask_a, dir_a, width)
    lo_b, hi_b = _safe_start_bounds(mask_b, dir_b, width)
    if lo_a > hi_a or lo_b > hi_b:
        return None

    overlap_need = 2 if min(mask_a.shape[1], mask_b.shape[1]) >= 3 else 1
    center = (width - 1) / 2
    pairs = []
    for ca in range(lo_a, hi_a + 1):
        for cb in range(lo_b, hi_b + 1):
            overlap = min(ca + mask_a.shape[1] - 1, cb + mask_b.shape[1] - 1) - max(ca, cb) + 1
            if overlap < overlap_need:
                continue
            score = abs((ca + cb) / 2 - center) + abs(ca - cb) * 0.2
            pairs.append((float(score), int(ca), int(cb)))
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0])
    shortlist = pairs[: min(12, len(pairs))]
    _, ca, cb = shortlist[int(rng.integers(len(shortlist)))]
    return ca, cb


def _center_pair_column_options(
    mask_a: np.ndarray,
    dir_a: int,
    mask_b: np.ndarray,
    dir_b: int,
    width: int,
) -> list[tuple[int, int]]:
    lo_a, hi_a = _safe_start_bounds(mask_a, dir_a, width)
    lo_b, hi_b = _safe_start_bounds(mask_b, dir_b, width)
    if lo_a > hi_a or lo_b > hi_b:
        return []

    overlap_need = 2 if min(mask_a.shape[1], mask_b.shape[1]) >= 3 else 1
    center = (width - 1) / 2
    pairs = []
    for ca in range(lo_a, hi_a + 1):
        for cb in range(lo_b, hi_b + 1):
            overlap = min(ca + mask_a.shape[1] - 1, cb + mask_b.shape[1] - 1) - max(ca, cb) + 1
            if overlap < overlap_need:
                continue
            score = abs((ca + cb) / 2 - center) + abs(ca - cb) * 0.2
            pairs.append((float(score), int(ca), int(cb)))
    pairs.sort(key=lambda item: item[0])
    return [(ca, cb) for _score, ca, cb in pairs[:40]]


def _pick_side_column(
    rng: np.random.Generator,
    mask: np.ndarray,
    direction: int,
    width: int,
    prefer_right_half: bool,
) -> int | None:
    lo, hi = _safe_start_bounds(mask, direction, width)
    if lo > hi:
        return None
    center = width // 2
    candidates = []
    for c0 in range(lo, hi + 1):
        mid = c0 + (mask.shape[1] - 1) / 2
        if prefer_right_half and mid <= center:
            continue
        if not prefer_right_half and mid >= center:
            continue
        candidates.append(int(c0))
    if not candidates:
        return None
    rng.shuffle(candidates)
    return candidates[0]


def _side_column_options(
    mask: np.ndarray,
    direction: int,
    width: int,
    prefer_right_half: bool,
) -> list[int]:
    lo, hi = _safe_start_bounds(mask, direction, width)
    if lo > hi:
        return []
    center = width // 2
    candidates = []
    for c0 in range(lo, hi + 1):
        mid = c0 + (mask.shape[1] - 1) / 2
        if prefer_right_half and mid <= center:
            continue
        if not prefer_right_half and mid >= center:
            continue
        candidates.append(int(c0))
    return candidates


def _settle_specs_from_top(
    specs: list[dict],
    shape: tuple[int, int],
    left_color: int,
    right_color: int,
) -> list[dict]:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, 0] = True
    occupied[:, w - 1] = True
    occupied[h - 1, :] = True

    settled = []
    for spec in specs:
        start = _object_from_mask(spec["mask"], spec["color"], 1, spec["c0"])
        landed = _drop_coords(start["coords"], occupied)
        direction = -1 if int(spec["color"]) == int(left_color) else 1
        sheared, anchor_bottom = _shear_coords(landed, direction)
        settled.append(
            {
                "color": int(spec["color"]),
                "c0": int(spec["c0"]),
                "mask": spec["mask"],
                "landed": landed,
                "sheared": sheared,
                "anchor_bottom": int(anchor_bottom),
            }
        )
        occupied[landed[:, 0], landed[:, 1]] = True
    return settled


def _gen_attempt() -> np.ndarray | None:
    rng = RNG
    h = int(rng.integers(16, 23))
    w = int(rng.integers(18, 25))
    left_color, right_color = rng.choice(WALL_PALETTE, size=2, replace=False).tolist()

    center_bottom_mask = _random_rectangle_mask(rng, (3, 4), (4, 5))
    max_top_width = min(3, center_bottom_mask.shape[0] - 1)
    if max_top_width < 2:
        return None
    center_top_mask = _random_rectangle_mask(rng, (2, 4), (2, max_top_width))
    left_side_mask = _random_solid_mask(rng, (2, 3), (2, 3))
    right_side_mask = _random_solid_mask(rng, (2, 3), (2, 3))

    bottom_color = int(left_color if rng.random() < 0.5 else right_color)
    top_color = int(right_color if bottom_color == left_color else left_color)
    bottom_dir = -1 if bottom_color == left_color else 1
    top_dir = -1 if top_color == left_color else 1

    bottom_lo, bottom_hi = _safe_start_bounds(center_bottom_mask, bottom_dir, w)
    top_lo, top_hi = _safe_start_bounds(center_top_mask, top_dir, w)
    left_options = sorted(_side_column_options(left_side_mask, 1, w, prefer_right_half=False))
    right_options = sorted(_side_column_options(right_side_mask, -1, w, prefer_right_half=True))
    if bottom_lo > bottom_hi or top_lo > top_hi or not left_options or not right_options:
        return None

    support_candidates = list(range(bottom_lo, bottom_hi + 1))
    support_candidates.sort(
        key=lambda c0: abs((c0 + (center_bottom_mask.shape[1] - 1) / 2) - (w - 1) / 2)
    )
    support_candidates = support_candidates[:6]
    left_candidates = left_options[:4]
    right_candidates = list(reversed(right_options[-4:]))

    chosen_specs = None
    for bottom_c0 in support_candidates:
        if bottom_color == right_color:
            top_c0 = bottom_c0
        else:
            top_c0 = bottom_c0 + center_bottom_mask.shape[1] - center_top_mask.shape[1]
        if not (top_lo <= top_c0 <= top_hi):
            continue
        for left_floor_c0 in left_candidates:
            for right_floor_c0 in right_candidates:
                spec_templates = [
                    {"mask": center_bottom_mask, "color": bottom_color, "c0": bottom_c0},
                    {"mask": center_top_mask, "color": top_color, "c0": int(top_c0)},
                    {"mask": left_side_mask, "color": right_color, "c0": left_floor_c0},
                    {"mask": right_side_mask, "color": left_color, "c0": right_floor_c0},
                ]
                settled_template = _settle_specs_from_top(spec_templates, (h, w), left_color, right_color)
                if _valid_settled_layout(settled_template, (h, w), left_color, right_color):
                    chosen_specs = spec_templates
                    break
            if chosen_specs is not None:
                break
        if chosen_specs is not None:
            break
    if chosen_specs is None:
        return None
    bottom_c0 = chosen_specs[0]["c0"]
    top_c0 = chosen_specs[1]["c0"]
    left_floor_c0 = chosen_specs[2]["c0"]
    right_floor_c0 = chosen_specs[3]["c0"]

    occupied = np.zeros((h, w), dtype=bool)
    bottom_row_lo = max(3, h // 2 - 1)
    bottom_row_hi = h - center_bottom_mask.shape[0] - 3
    if bottom_row_lo > bottom_row_hi:
        return None
    bottom_obj = _sample_row(rng, center_bottom_mask, bottom_c0, bottom_row_lo, bottom_row_hi, occupied)
    if bottom_obj is None:
        return None
    bottom_obj["color"] = bottom_color
    _paint_occupied(occupied, bottom_obj["coords"])

    top_row_hi = bottom_obj["bbox"][0] - center_top_mask.shape[0] - 2
    top_obj = _sample_row(rng, center_top_mask, top_c0, 1, top_row_hi, occupied)
    if top_obj is None:
        return None
    top_obj["color"] = top_color
    _paint_occupied(occupied, top_obj["coords"])

    left_floor_obj = _sample_row(rng, left_side_mask, left_floor_c0, 1, h - left_side_mask.shape[0] - 3, occupied)
    if left_floor_obj is None:
        return None
    left_floor_obj["color"] = right_color
    _paint_occupied(occupied, left_floor_obj["coords"])

    right_floor_obj = _sample_row(rng, right_side_mask, right_floor_c0, 1, h - right_side_mask.shape[0] - 3, occupied)
    if right_floor_obj is None:
        return None
    right_floor_obj["color"] = left_color

    objects = [bottom_obj, top_obj, left_floor_obj, right_floor_obj]
    for obj in objects:
        obj["bbox"] = bbox_coords(obj["coords"])

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[:-1, 0] = left_color
    grid[:-1, w - 1] = right_color
    grid[h - 1, :] = FLOOR_COLOR
    for obj in objects:
        grid[obj["coords"][:, 0], obj["coords"][:, 1]] = int(obj["color"])

    if not _valid_generated_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacked-shear ARC sample")
