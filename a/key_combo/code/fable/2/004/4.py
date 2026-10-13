from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, paste


SHAPE_LIBRARY = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
]

GEN_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_CANONICAL_HEIGHT = (14, 21)
GEN_CANONICAL_WIDTH = (14, 22)
GEN_LANE_COUNT = (2, 3)
GEN_FREE_COUNT = (1, 2)
GEN_RETRY_LIMIT = 400
GEN_SHAPE_RETRY = 80
GEN_PLACE_RETRY = 160


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_shape(*, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(GEN_SHAPE_RETRY):
        mask = SHAPE_LIBRARY[random.randrange(len(SHAPE_LIBRARY))].copy()
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        mask = _trim_mask(mask)
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w:
            return mask
    raise RuntimeError("failed to sample a fitting shape")


def _fits(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > occ.shape[0] or c0 + w > occ.shape[1]:
        return False
    return not bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any())


def _overlaps(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > occ.shape[0] or c0 + w > occ.shape[1]:
        return True
    return bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any())


def _stamp_occ(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = mask.shape
    occ[r0 : r0 + h, c0 : c0 + w] |= mask


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int, bg: int) -> None:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    paste(grid, patch, r0, c0, transparent=bg)


def _detect_stripe(g: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = g.shape
    candidates: list[tuple[str, int]] = []
    if int(g[0, 0]) != bg and np.all(g[0, :] == g[0, 0]):
        candidates.append(("top", int(g[0, 0])))
    if int(g[h - 1, 0]) != bg and np.all(g[h - 1, :] == g[h - 1, 0]):
        candidates.append(("bottom", int(g[h - 1, 0])))
    if int(g[0, 0]) != bg and np.all(g[:, 0] == g[0, 0]):
        candidates.append(("left", int(g[0, 0])))
    if int(g[0, w - 1]) != bg and np.all(g[:, w - 1] == g[0, w - 1]):
        candidates.append(("right", int(g[0, w - 1])))
    assert len(candidates) == 1, "expected exactly one full-span edge stripe"
    return candidates[0]


def _stripe_component(comp: dict, side: str, shape: tuple[int, int]) -> bool:
    coords = comp["coords"]
    h, w = shape
    if side == "top":
        return int(comp["area"]) == w and np.all(coords[:, 0] == 0)
    if side == "bottom":
        return int(comp["area"]) == w and np.all(coords[:, 0] == h - 1)
    if side == "left":
        return int(comp["area"]) == h and np.all(coords[:, 1] == 0)
    return int(comp["area"]) == h and np.all(coords[:, 1] == w - 1)


def _distance_to_stripe(comp: dict, side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    if side == "top":
        return int(r0 - 1)
    if side == "bottom":
        return int(h - 2 - r1)
    if side == "left":
        return int(c0 - 1)
    return int(w - 2 - c1)


def _sort_key(comp: dict, side: str, shape: tuple[int, int]) -> tuple[int, int, int, int, int]:
    dist = _distance_to_stripe(comp, side, shape)
    r0, c0, r1, c1 = comp["bbox"]
    if side in ("top", "bottom"):
        return (dist, int(c0), int(c1), int(r0), int(r1))
    return (dist, int(r0), int(r1), int(c0), int(c1))


def _movement(side: str) -> tuple[int, int]:
    if side == "top":
        return -1, 0
    if side == "bottom":
        return 1, 0
    if side == "left":
        return 0, -1
    return 0, 1


def _init_output(shape: tuple[int, int], bg: int, side: str, stripe_color: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(shape, bg, dtype=int)
    occ = np.zeros(shape, dtype=bool)
    if side == "top":
        out[0, :] = int(stripe_color)
        occ[0, :] = True
    elif side == "bottom":
        out[-1, :] = int(stripe_color)
        occ[-1, :] = True
    elif side == "left":
        out[:, 0] = int(stripe_color)
        occ[:, 0] = True
    else:
        out[:, -1] = int(stripe_color)
        occ[:, -1] = True
    return out, occ


def _parse_scene(g: np.ndarray) -> tuple[int, str, int, list[dict]]:
    bg = int(get_bg(g))
    side, stripe_color = _detect_stripe(g, bg)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    objects: list[dict] = []
    stripe_count = 0
    for comp in components:
        if _stripe_component(comp, side, g.shape):
            stripe_count += 1
            assert int(comp["color"]) == stripe_color
            continue
        dist = _distance_to_stripe(comp, side, g.shape)
        assert dist >= 1, "moving objects must start off the stripe"
        objects.append(comp)
    assert stripe_count == 1, "expected one stripe component"
    assert objects, "expected at least one moving object"
    return bg, side, stripe_color, objects


def _translate_coords(coords: np.ndarray, dr: int, dc: int, steps: int) -> np.ndarray:
    return coords + np.array([dr * steps, dc * steps], dtype=int)


def _can_shift(coords: np.ndarray, occupied: np.ndarray, dr: int, dc: int, steps: int) -> bool:
    shifted = _translate_coords(coords, dr, dc, steps)
    rr = shifted[:, 0]
    cc = shifted[:, 1]
    if rr.min() < 0 or cc.min() < 0 or rr.max() >= occupied.shape[0] or cc.max() >= occupied.shape[1]:
        return False
    return not bool(occupied[rr, cc].any())


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, side, stripe_color, objects = _parse_scene(g)
    dr, dc = _movement(side)
    out, occupied = _init_output(g.shape, bg, side, stripe_color)

    ordered = sorted(objects, key=lambda comp: _sort_key(comp, side, g.shape))
    for comp in ordered:
        coords = comp["coords"]
        budget = int(comp["area"])
        shift = 0
        while shift < budget and _can_shift(coords, occupied, dr, dc, shift + 1):
            shift += 1
        target = _translate_coords(coords, dr, dc, shift)
        rr = target[:, 0]
        cc = target[:, 1]
        occupied[rr, cc] = True
        out[rr, cc] = int(comp["color"])
    return out


def _bruteforce_oracle(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, side, stripe_color, objects = _parse_scene(g)
    dr, dc = _movement(side)
    out, occupied = _init_output(g.shape, bg, side, stripe_color)

    ordered = sorted(objects, key=lambda comp: _sort_key(comp, side, g.shape))
    for comp in ordered:
        current = {(int(r), int(c)) for r, c in comp["coords"]}
        for _ in range(int(comp["area"])):
            moved = {(r + dr, c + dc) for r, c in current}
            blocked = False
            for nr, nc in moved:
                if nr < 0 or nc < 0 or nr >= occupied.shape[0] or nc >= occupied.shape[1] or occupied[nr, nc]:
                    blocked = True
                    break
            if blocked:
                break
            current = moved
        for r, c in current:
            occupied[r, c] = True
            out[r, c] = int(comp["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each floating object toward the edge stripe by up to its own area, stopping early when a nearer settled object or the stripe blocks the path.

    Input structure:
    - The background is the dominant color.
    - Exactly one border carries a single-color full-span stripe: a full row on the top or bottom, or a full column on the left or right.
    - The remaining non-background cells are one or more 4-connected monochrome objects on the stripe-facing side of that border.

    Rule:
    - Infer the motion direction from the stripe edge, then process objects from nearest to farthest by the gap between their front face and the stripe.
    - Each object is removed from its start position and translated straight toward the stripe with a step budget equal to its pixel area.
    - If the next step would hit the stripe or an already settled nearer object, the object stops immediately before that collision; otherwise it uses its full budget and may remain suspended away from the stripe.

    Key insight:
    - The displacement is not determined by area alone: the same area budget can be cut short by previously settled objects, so order-by-distance and collision truncation are both essential.
    """
    return _solve(grid)


def _choose_status(area: int, allowed: tuple[str, ...]) -> tuple[str, int]:
    if "stripe_exact" in allowed and (area == 1 or random.random() < 0.45):
        return "stripe_exact", area
    if "stripe_blocked" in allowed and area >= 2 and random.random() < 0.7:
        return "stripe_blocked", random.randint(1, area - 1)
    if "hover_budget" in allowed:
        return "hover_budget", area
    if "stripe_exact" in allowed:
        return "stripe_exact", area
    return "stripe_blocked", max(1, area - 1)


def _column_options(ranges: list[tuple[int, int]], width: int) -> list[int]:
    cols: list[int] = []
    for c0, c1 in ranges:
        for c in range(c0, c1 - width + 2):
            cols.append(c)
    random.shuffle(cols)
    return cols


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(*GEN_CANONICAL_HEIGHT)
    w = random.randint(*GEN_CANONICAL_WIDTH)
    bg = random.choice(GEN_BG_COLORS)
    palette = [color for color in range(10) if color != bg]

    lane_count = random.randint(*GEN_LANE_COUNT)
    free_count = random.randint(*GEN_FREE_COUNT)
    total_objects = lane_count + free_count

    stripe_color = random.choice(palette)
    object_colors = random.sample([color for color in palette if color != stripe_color], total_objects)

    lane_width = random.randint(4, 6)
    lane_on_left = random.random() < 0.5
    if lane_on_left:
        lane_left = random.randint(0, 1)
        lane_right = lane_left + lane_width - 1
        free_ranges = [(lane_right + 2, w - 1)]
    else:
        lane_right = w - 1 - random.randint(0, 1)
        lane_left = lane_right - lane_width + 1
        free_ranges = [(0, lane_left - 2)]
    assert free_ranges[0][1] - free_ranges[0][0] + 1 >= 4

    stripe_occ = np.zeros((h, w), dtype=bool)
    stripe_occ[0, :] = True
    final_occ = stripe_occ.copy()
    input_occ = stripe_occ.copy()
    lane_final_occ = np.zeros((h, w), dtype=bool)

    specs: list[dict] = []
    used_front_rows: set[int] = set()
    lane_front_rows: list[int] = []

    color_iter = iter(object_colors)

    anchor_mask = _random_shape(max_h=4, max_w=lane_width)
    anchor_area = int(anchor_mask.sum())
    anchor_status, anchor_shift = _choose_status(anchor_area, ("stripe_exact", "stripe_blocked"))
    anchor_cols = list(range(lane_left, lane_right - anchor_mask.shape[1] + 2))
    random.shuffle(anchor_cols)
    anchor_c = anchor_cols[0]
    anchor_final_r = 1
    anchor_input_r = anchor_final_r + anchor_shift
    assert _fits(final_occ, anchor_mask, anchor_final_r, anchor_c)
    assert _fits(input_occ, anchor_mask, anchor_input_r, anchor_c)

    specs.append(
        {
            "mask": anchor_mask,
            "color": next(color_iter),
            "final_r": anchor_final_r,
            "final_c": anchor_c,
            "input_r": anchor_input_r,
            "status": anchor_status,
        }
    )
    _stamp_occ(final_occ, anchor_mask, anchor_final_r, anchor_c)
    _stamp_occ(input_occ, anchor_mask, anchor_input_r, anchor_c)
    _stamp_occ(lane_final_occ, anchor_mask, anchor_final_r, anchor_c)
    used_front_rows.add(anchor_input_r)
    lane_front_rows.append(anchor_input_r)

    for _ in range(lane_count - 1):
        placed = False
        for _attempt in range(GEN_PLACE_RETRY):
            mask = _random_shape(max_h=4, max_w=lane_width)
            area = int(mask.sum())
            if area < 2:
                continue

            col_options = list(range(lane_left, lane_right - mask.shape[1] + 2))
            random.shuffle(col_options)
            row_options = list(range(2, h - mask.shape[0]))
            random.shuffle(row_options)
            for final_r in row_options:
                for final_c in col_options:
                    if not _fits(final_occ, mask, final_r, final_c):
                        continue
                    if not _overlaps(lane_final_occ, mask, final_r - 1, final_c):
                        continue
                    shifts = list(range(1, area))
                    random.shuffle(shifts)
                    for shift in shifts:
                        input_r = final_r + shift
                        if input_r + mask.shape[0] > h:
                            continue
                        if input_r in used_front_rows or input_r <= lane_front_rows[-1]:
                            continue
                        if not _fits(input_occ, mask, input_r, final_c):
                            continue
                        specs.append(
                            {
                                "mask": mask,
                                "color": next(color_iter),
                                "final_r": final_r,
                                "final_c": final_c,
                                "input_r": input_r,
                                "status": "blocked_by_object",
                            }
                        )
                        _stamp_occ(final_occ, mask, final_r, final_c)
                        _stamp_occ(input_occ, mask, input_r, final_c)
                        _stamp_occ(lane_final_occ, mask, final_r, final_c)
                        used_front_rows.add(input_r)
                        lane_front_rows.append(input_r)
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            raise RuntimeError("failed to place collision follower")

    mandatory_free_statuses = ["hover_budget"]
    if free_count == 2:
        mandatory_free_statuses.append(random.choice(["hover_budget", "stripe_blocked", "stripe_exact"]))

    for idx in range(free_count):
        target_status = mandatory_free_statuses[idx]
        placed = False
        for _attempt in range(GEN_PLACE_RETRY):
            free_width = max(1, free_ranges[0][1] - free_ranges[0][0] + 1)
            mask = _random_shape(max_h=4, max_w=min(4, free_width))
            area = int(mask.sum())
            if target_status == "stripe_blocked" and area < 2:
                continue

            col_options = _column_options(free_ranges, mask.shape[1])
            if not col_options:
                continue

            if target_status == "hover_budget":
                shifts = [area]
                row_options = list(range(2, h - mask.shape[0] - area + 1))
            elif target_status == "stripe_exact":
                shifts = [area]
                row_options = [1]
            else:
                shifts = list(range(1, area))
                random.shuffle(shifts)
                row_options = [1]

            random.shuffle(row_options)
            for final_r in row_options:
                for final_c in col_options:
                    if not _fits(final_occ, mask, final_r, final_c):
                        continue
                    if target_status == "hover_budget" and _overlaps(final_occ, mask, final_r - 1, final_c):
                        continue
                    for shift in shifts:
                        input_r = final_r + shift
                        if input_r + mask.shape[0] > h:
                            continue
                        if input_r in used_front_rows:
                            continue
                        if not _fits(input_occ, mask, input_r, final_c):
                            continue
                        specs.append(
                            {
                                "mask": mask,
                                "color": next(color_iter),
                                "final_r": final_r,
                                "final_c": final_c,
                                "input_r": input_r,
                                "status": target_status,
                            }
                        )
                        _stamp_occ(final_occ, mask, final_r, final_c)
                        _stamp_occ(input_occ, mask, input_r, final_c)
                        used_front_rows.add(input_r)
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            raise RuntimeError("failed to place free object")

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    inp[0, :] = stripe_color
    out[0, :] = stripe_color
    for spec in specs:
        _paint_mask(inp, spec["mask"], spec["input_r"], spec["final_c"], spec["color"], bg)
        _paint_mask(out, spec["mask"], spec["final_r"], spec["final_c"], spec["color"], bg)

    side = random.choice(["top", "bottom", "left", "right"])
    k = {"top": 0, "left": 1, "bottom": 2, "right": 3}[side]
    return np.rot90(inp, k), np.rot90(out, k)


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
        brute = _bruteforce_oracle(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(brute, expected):
        return False

    _bg, side, _stripe_color, objects = _parse_scene(inp)
    distances = [_distance_to_stripe(comp, side, inp.shape) for comp in objects]
    if len(distances) != len(set(distances)):
        return False

    comps = find_components(inp, bg=get_bg(inp), connectivity=4, monochromatic=True, return_format="dict")
    stripe_like = sum(1 for comp in comps if _stripe_component(comp, side, inp.shape))
    return stripe_like == 1


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            inp, expected = _build_instance()
        except RuntimeError:
            continue
        if _validate_generated(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid budget-fall instance")
