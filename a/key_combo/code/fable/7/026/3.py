from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, in_bounds


TOP = "top"
BOTTOM = "bottom"
LEFT = "left"
RIGHT = "right"
SIDES = (TOP, BOTTOM, LEFT, RIGHT)
MAX_GENERATION_ATTEMPTS = 400


def _touches_border(coord: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = coord
    h, w = shape
    return r in (0, h - 1) or c in (0, w - 1)


def _marker_side(coord: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = coord
    h, w = shape
    assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "corner markers are ambiguous"
    if r == 0:
        return TOP
    if r == h - 1:
        return BOTTOM
    if c == 0:
        return LEFT
    if c == w - 1:
        return RIGHT
    raise AssertionError("marker must lie on the border")


def _ray_cells(shape: tuple[int, int], marker: tuple[int, int], side: str, length: int) -> list[tuple[int, int]]:
    h, w = shape
    r, c = marker
    length = int(length)
    assert length >= 1
    if side == TOP:
        assert length <= h - 2
        return [(rr, c) for rr in range(1, length + 1)]
    if side == BOTTOM:
        assert length <= h - 2
        return [(rr, c) for rr in range(h - 2, h - 2 - length, -1)]
    if side == LEFT:
        assert length <= w - 2
        return [(r, cc) for cc in range(1, length + 1)]
    if side == RIGHT:
        assert length <= w - 2
        return [(r, cc) for cc in range(w - 2, w - 2 - length, -1)]
    raise ValueError(f"unknown side: {side}")


def _parse_case(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    specs = []
    colors = sorted(int(c) for c in np.unique(g) if int(c) != bg)
    assert colors, "expected colored markers and objects"

    seen_markers = set()
    for color in colors:
        comps = components_mask(g == int(color), connectivity=4)
        assert len(comps) == 2, f"color {color} must form exactly one marker and one object"

        marker_candidates = []
        object_candidates = []
        for comp in comps:
            coords = [(int(r), int(c)) for r, c in comp]
            border_flags = [_touches_border(coord, (h, w)) for coord in coords]
            if len(coords) == 1 and border_flags[0]:
                marker_candidates.append(coords[0])
            elif not any(border_flags):
                object_candidates.append(coords)
            else:
                raise AssertionError(f"color {color} has an invalid component")

        assert len(marker_candidates) == 1, f"color {color} must have one border marker"
        assert len(object_candidates) == 1, f"color {color} must have one interior object"

        marker = marker_candidates[0]
        assert marker not in seen_markers
        seen_markers.add(marker)

        specs.append(
            {
                "color": int(color),
                "marker": marker,
                "side": _marker_side(marker, (h, w)),
                "area": int(len(object_candidates[0])),
                "object_coords": object_candidates[0],
            }
        )

    areas = [spec["area"] for spec in specs]
    assert len(set(areas)) == len(areas), "object areas must be pairwise distinct"
    return bg, specs


def _render_output(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    best_area = np.full(shape, -1, dtype=int)

    for spec in specs:
        mr, mc = spec["marker"]
        out[mr, mc] = int(spec["color"])

    for spec in sorted(specs, key=lambda item: int(item["area"])):
        for r, c in _ray_cells(shape, spec["marker"], spec["side"], spec["area"]):
            if int(spec["area"]) > int(best_area[r, c]):
                best_area[r, c] = int(spec["area"])
                out[r, c] = int(spec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each interior object's area into the length of its same-colored edge ray, with larger areas winning overlaps.

    Input structure:
    - The background is black.
    - Each non-black color appears as exactly one single-pixel edge marker on the border and one interior same-colored 4-connected solid object.
    - The interior objects float away from the border, and their areas are all distinct.

    Rule:
    - Erase every interior object.
    - Keep each border marker and extend a same-colored ray straight inward, perpendicular to the border containing that marker.
    - The ray length equals the pixel area of the same-colored interior object, excluding the marker itself; when multiple rays claim one cell, the color whose source object has larger area remains.

    Key insight:
    - The interior objects are not copied by shape or position; only their areas matter, and the same area controls both ray length and overlap priority.
    """
    g = np.asarray(grid, dtype=int)
    bg, specs = _parse_case(g)
    return _render_output(g.shape, bg, specs)


def _capacity(shape: tuple[int, int], side: str) -> int:
    h, w = shape
    return h - 2 if side in (TOP, BOTTOM) else w - 2


def _divisor_dims(area: int, side: str) -> tuple[int, int]:
    pairs = []
    for h in range(1, area + 1):
        if area % h == 0:
            w = area // h
            pairs.append((h, w))
    pairs.sort(key=lambda hw: (abs(hw[0] - hw[1]), max(hw), min(hw)))
    h, w = pairs[0]
    if side in (TOP, BOTTOM) and h > w:
        h, w = w, h
    if side in (LEFT, RIGHT) and h < w:
        h, w = w, h
    return int(h), int(w)


def _crossing_count(shape: tuple[int, int], specs: list[dict]) -> int:
    claims: dict[tuple[int, int], list[int]] = {}
    for spec in specs:
        for cell in _ray_cells(shape, spec["marker"], spec["side"], spec["area"]):
            claims.setdefault(cell, []).append(int(spec["area"]))
    return sum(1 for vals in claims.values() if len(vals) > 1)


def _assign_areas(shape: tuple[int, int], specs: list[dict], require_crossing: bool) -> bool:
    max_cap = min(max(_capacity(shape, spec["side"]) for spec in specs), 10)
    area_pool = list(range(2, max_cap + 1))
    if len(area_pool) < len(specs):
        return False

    order = sorted(range(len(specs)), key=lambda idx: _capacity(shape, specs[idx]["side"]))
    chosen = [None] * len(specs)
    used: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            for idx, area in enumerate(chosen):
                specs[idx]["area"] = int(area)
            crosses = _crossing_count(shape, specs)
            return crosses > 0 if require_crossing else crosses == 0

        idx = order[pos]
        cap = _capacity(shape, specs[idx]["side"])
        choices = [area for area in area_pool if area <= cap and area not in used]
        random.shuffle(choices)
        choices.sort(key=lambda area: abs(area - (cap // 2)))
        for area in choices:
            chosen[idx] = int(area)
            used.add(int(area))
            if backtrack(pos + 1):
                return True
            used.remove(int(area))
            chosen[idx] = None
        return False

    return backtrack(0)


def _candidate_positions(
    shape: tuple[int, int],
    side: str,
    marker: tuple[int, int],
    dims: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    obj_h, obj_w = dims
    r_values = range(1, h - 1 - obj_h + 1)
    c_values = range(1, w - 1 - obj_w + 1)
    candidates = [(r0, c0) for r0 in r_values for c0 in c_values]

    if side in (TOP, BOTTOM):
        marker_c = marker[1]
        if side == TOP:
            candidates.sort(key=lambda rc: (-(rc[0]), abs((rc[1] + (obj_w - 1) / 2) - marker_c), rc[1]))
        else:
            candidates.sort(key=lambda rc: (rc[0], abs((rc[1] + (obj_w - 1) / 2) - marker_c), rc[1]))
    else:
        marker_r = marker[0]
        if side == LEFT:
            candidates.sort(key=lambda rc: (-(rc[1]), abs((rc[0] + (obj_h - 1) / 2) - marker_r), rc[0]))
        else:
            candidates.sort(key=lambda rc: (rc[1], abs((rc[0] + (obj_h - 1) / 2) - marker_r), rc[0]))
    return candidates


def _can_place_rect(
    occupied: np.ndarray,
    forbidden: set[tuple[int, int]],
    top_left: tuple[int, int],
    dims: tuple[int, int],
) -> bool:
    r0, c0 = top_left
    h, w = dims
    hh, ww = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if not (1 <= r0 <= r1 < hh - 1 and 1 <= c0 <= c1 < ww - 1):
        return False
    if occupied[max(0, r0 - 1) : min(hh, r1 + 2), max(0, c0 - 1) : min(ww, c1 + 2)].any():
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in forbidden:
                return False
    return True


def _place_objects(shape: tuple[int, int], specs: list[dict]) -> bool:
    occupied = np.zeros(shape, dtype=bool)
    forbidden = {
        cell
        for spec in specs
        for cell in _ray_cells(shape, spec["marker"], spec["side"], spec["area"])
    }

    for spec in sorted(specs, key=lambda item: int(item["area"]), reverse=True):
        dims = _divisor_dims(spec["area"], spec["side"])
        placed = None
        for top_left in _candidate_positions(shape, spec["side"], spec["marker"], dims):
            if _can_place_rect(occupied, forbidden, top_left, dims):
                placed = top_left
                break
        if placed is None:
            return False

        r0, c0 = placed
        obj_h, obj_w = dims
        coords = [(r, c) for r in range(r0, r0 + obj_h) for c in range(c0, c0 + obj_w)]
        spec["object_coords"] = coords
        occupied[r0 : r0 + obj_h, c0 : c0 + obj_w] = True
    return True


def _build_specs(shape: tuple[int, int], force_crossing: bool) -> list[dict] | None:
    h, w = shape
    n_colors = random.choices([3, 4, 5], weights=[4, 4, 2], k=1)[0]
    n_vertical = random.randint(1, n_colors - 1)
    n_horizontal = n_colors - n_vertical

    cols = random.sample(list(range(1, w - 1)), n_vertical)
    rows = random.sample(list(range(1, h - 1)), n_horizontal)
    colors = random.sample(Color.NOT_BLACK, n_colors)

    specs = []
    for idx in range(n_vertical):
        side = random.choice([TOP, BOTTOM])
        marker = (0, cols[idx]) if side == TOP else (h - 1, cols[idx])
        specs.append({"color": int(colors[len(specs)]), "marker": marker, "side": side})
    for idx in range(n_horizontal):
        side = random.choice([LEFT, RIGHT])
        marker = (rows[idx], 0) if side == LEFT else (rows[idx], w - 1)
        specs.append({"color": int(colors[len(specs)]), "marker": marker, "side": side})

    random.shuffle(specs)
    if not _assign_areas(shape, specs, require_crossing=force_crossing):
        return None
    if not _place_objects(shape, specs):
        return None
    return specs


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        mr, mc = spec["marker"]
        grid[mr, mc] = int(spec["color"])
        for r, c in spec["object_coords"]:
            grid[r, c] = int(spec["color"])
    return grid


def _generate_case(return_spec: bool = False) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = (random.randint(11, 18), random.randint(11, 18))
        force_crossing = random.random() < 0.72
        specs = _build_specs(shape, force_crossing=force_crossing)
        if specs is None:
            continue

        grid = _render_input(shape, specs)
        try:
            parsed_bg, parsed_specs = _parse_case(grid)
        except AssertionError:
            continue

        expected = _render_output(shape, parsed_bg, parsed_specs)
        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue

        # Reject overly plain samples; the task should vary in side usage and crossings.
        sides_used = {spec["side"] for spec in specs}
        crosses = _crossing_count(shape, specs)
        if len(sides_used) < 2:
            continue
        if force_crossing and crosses == 0:
            continue
        if (not force_crossing) and crosses != 0:
            continue
        if return_spec:
            return grid, specs
        return grid
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _generate_case(return_spec=False)
