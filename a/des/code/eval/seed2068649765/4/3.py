import random
from collections import deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "up": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
}

FIRST_DIR_BY_SIDE = {
    "left": "right",
    "right": "left",
    "top": "down",
    "bottom": "up",
}

ORTHOGONALS = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
NUM_OBJECTS_RANGE = (2, 4)
ARROW_COUNT_VALUES = (1, 2, 3, 4)
ARROW_COUNT_WEIGHTS = (2, 4, 3, 1)
GEN_ATTEMPTS = 400
SPEC_ATTEMPTS = 120


def _mode_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(vals)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _bboxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _arrow_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - 1, c - 1, r + 1, c + 1


def _arrow_cells(center: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r, c = center
    pattern = ARROW_PATTERNS[direction]
    cells: list[tuple[int, int]] = []
    for pr in range(3):
        for pc in range(3):
            if pattern[pr, pc]:
                cells.append((r - 1 + pr, c - 1 + pc))
    return cells


def _distance_bbox_to_point(bbox: tuple[int, int, int, int], point: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    r, c = point
    if r < r0:
        dr = r0 - r
    elif r > r1:
        dr = r - r1
    else:
        dr = 0
    if c < c0:
        dc = c0 - c
    elif c > c1:
        dc = c - c1
    else:
        dc = 0
    return int(dr + dc)


def _forward_and_perp_delta(
    origin: tuple[int, int], point: tuple[int, int], direction: str
) -> tuple[int, int]:
    orow, ocol = origin
    prow, pcol = point
    if direction == "right":
        return pcol - ocol, abs(prow - orow)
    if direction == "left":
        return ocol - pcol, abs(prow - orow)
    if direction == "down":
        return prow - orow, abs(pcol - ocol)
    return orow - prow, abs(pcol - ocol)


def _is_ahead(origin: tuple[int, int], point: tuple[int, int], direction: str) -> bool:
    forward, _ = _forward_and_perp_delta(origin, point, direction)
    return forward > 0


def _shift_bbox_to_reach(
    bbox: tuple[int, int, int, int], direction: str, target_center: tuple[int, int]
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    tr, tc = target_center
    if direction == "right":
        delta = tc - c1
        assert delta > 0
        return r0, c0 + delta, r1, c1 + delta
    if direction == "left":
        delta = tc - c0
        assert delta < 0
        return r0, c0 + delta, r1, c1 + delta
    if direction == "down":
        delta = tr - r1
        assert delta > 0
        return r0 + delta, c0, r1 + delta, c1
    delta = tr - r0
    assert delta < 0
    return r0 + delta, c0, r1 + delta, c1


def _unique_best(items, key_fn):
    keyed = sorted((key_fn(item), idx, item) for idx, item in enumerate(items))
    assert keyed, "no candidates available"
    if len(keyed) >= 2:
        assert keyed[0][0] != keyed[1][0], "ambiguous minimum candidate"
    return keyed[0][2]


def _component_kind(comp: list[tuple[int, int]]) -> tuple[str, str | None]:
    bbox = _bbox_from_points(comp)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = len(comp)
    if (h, w, area) in {(2, 2, 4), (3, 3, 9)}:
        return "cargo", None
    if (h, w, area) == (3, 3, 5):
        local = np.zeros((3, 3), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        for direction, pattern in ARROW_PATTERNS.items():
            if np.array_equal(local, pattern):
                return "arrow", direction
    raise AssertionError("unrecognized component shape")


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    infos: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        mask = g == color
        comps = _components(mask)
        cargo_bbox = None
        arrows: list[dict] = []
        for comp in comps:
            kind, direction = _component_kind(comp)
            bbox = _bbox_from_points(comp)
            if kind == "cargo":
                assert cargo_bbox is None, "each color must have exactly one cargo"
                cargo_bbox = bbox
            else:
                assert direction is not None
                arrows.append(
                    {
                        "center": (bbox[0] + 1, bbox[1] + 1),
                        "dir": direction,
                        "bbox": bbox,
                    }
                )
        assert cargo_bbox is not None, "missing cargo"
        assert arrows, "missing arrows"
        first_idx = int(np.flatnonzero(mask.ravel())[0])
        infos.append(
            {
                "color": color,
                "first_idx": first_idx,
                "cargo_bbox": cargo_bbox,
                "arrows": arrows,
            }
        )
    infos.sort(key=lambda item: item["first_idx"])
    return bg, infos


def _build_arrow_chain(cargo_bbox: tuple[int, int, int, int], arrows: list[dict]) -> list[dict]:
    remaining = list(arrows)
    first = _unique_best(
        remaining,
        lambda arrow: (
            _distance_bbox_to_point(cargo_bbox, arrow["center"]),
            arrow["center"][0],
            arrow["center"][1],
        ),
    )
    chain = [first]
    remaining.remove(first)
    while remaining:
        current = chain[-1]
        candidates = [arrow for arrow in remaining if _is_ahead(current["center"], arrow["center"], current["dir"])]
        assert candidates, "arrow chain must continue in the current arrow direction"
        nxt = _unique_best(
            candidates,
            lambda arrow: (
                _forward_and_perp_delta(current["center"], arrow["center"], current["dir"]),
                arrow["center"][0],
                arrow["center"][1],
            ),
        )
        chain.append(nxt)
        remaining.remove(nxt)
    return chain


def _final_bbox_from_chain(cargo_bbox: tuple[int, int, int, int], chain: list[dict]) -> tuple[int, int, int, int]:
    rect = cargo_bbox
    for idx, arrow in enumerate(chain):
        move_dir = arrow["dir"] if idx == 0 else chain[idx - 1]["dir"]
        rect = _shift_bbox_to_reach(rect, move_dir, arrow["center"])
        rect = _union_bbox(rect, arrow["bbox"])
    return rect


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each colored cargo block into a single final body by following its same-color arrow chain.
    """
    g = np.asarray(grid, dtype=int)
    bg, infos = _parse_scene(g)
    out = np.full_like(g, bg)
    for info in infos:
        chain = _build_arrow_chain(info["cargo_bbox"], info["arrows"])
        r0, c0, r1, c1 = _final_bbox_from_chain(info["cargo_bbox"], chain)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(info["color"])
    return out


def _spec_first_index(spec: dict, w: int) -> int:
    cells = _bbox_cells(spec["cargo_bbox"])
    for arrow in spec["arrows"]:
        cells.extend(_arrow_cells(arrow["center"], arrow["dir"]))
    return min(r * w + c for r, c in cells)


def _paint_output_from_specs(specs: list[dict], h: int, w: int, bg: int = 0) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    ordered = sorted(specs, key=lambda spec: _spec_first_index(spec, w))
    for spec in ordered:
        chain = []
        for arrow in spec["arrows"]:
            chain.append({"center": arrow["center"], "dir": arrow["dir"], "bbox": _arrow_bbox(arrow["center"])})
        r0, c0, r1, c1 = _final_bbox_from_chain(spec["cargo_bbox"], chain)
        out[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]
    return out


def _render_specs(specs: list[dict], h: int, w: int, bg: int = 0) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        color = spec["color"]
        r0, c0, r1, c1 = spec["cargo_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        for arrow in spec["arrows"]:
            for r, c in _arrow_cells(arrow["center"], arrow["dir"]):
                grid[r, c] = color
    return grid


def _candidate_axis_values(low: int, high: int, rng: random.Random) -> list[int]:
    if low > high:
        return []
    vals = list(range(low, high + 1))
    vals.sort(key=lambda v: (abs(v - low), rng.random()))
    return vals


def _candidate_perp_values(
    base: int, low: int, high: int, rng: random.Random, max_offset: int = 3
) -> list[int]:
    vals: list[int] = []
    for offset in range(max_offset + 1):
        for sign in (-1, 1):
            v = base + sign * offset
            if offset == 0:
                v = base
            if low <= v <= high and v not in vals:
                vals.append(v)
    extras = [v for v in range(low, high + 1) if v not in vals]
    rng.shuffle(extras)
    vals.extend(extras)
    return vals


def _bbox_in_bounds(bbox: tuple[int, int, int, int], h: int, w: int) -> bool:
    return 0 <= bbox[0] <= bbox[2] < h and 0 <= bbox[1] <= bbox[3] < w


def _can_place_bbox(candidate: tuple[int, int, int, int], reserved: list[tuple[int, int, int, int]], h: int, w: int) -> bool:
    if not _bbox_in_bounds(candidate, h, w):
        return False
    padded = _expand_bbox(candidate, 1)
    for other in reserved:
        if _bboxes_intersect(padded, other):
            return False
    return True


def _occupied_bboxes_for_spec(spec: dict) -> list[tuple[int, int, int, int]]:
    bboxes = [spec["cargo_bbox"]]
    for arrow in spec["arrows"]:
        bboxes.append(_arrow_bbox(arrow["center"]))
    return bboxes


def _pick_center_ahead(
    rng: random.Random,
    rect: tuple[int, int, int, int],
    move_dir: str,
    reserved: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    preferred_perp: int | None = None,
    preferred_axis: int | None = None,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = rect
    if move_dir == "right":
        axis_low, axis_high = c1 + 2, w - 2
        perp_low, perp_high = 1, h - 2
        base_perp = (r0 + r1) // 2 if preferred_perp is None else preferred_perp
        axis_vals = [preferred_axis] if preferred_axis is not None else _candidate_axis_values(axis_low, axis_high, rng)
        perp_vals = _candidate_perp_values(base_perp, perp_low, perp_high, rng)
        for axis in axis_vals:
            if not (axis_low <= axis <= axis_high):
                continue
            for perp in perp_vals:
                bbox = _arrow_bbox((perp, axis))
                if _can_place_bbox(bbox, reserved, h, w):
                    return perp, axis
        return None
    if move_dir == "left":
        axis_low, axis_high = 1, c0 - 2
        perp_low, perp_high = 1, h - 2
        base_perp = (r0 + r1) // 2 if preferred_perp is None else preferred_perp
        axis_vals = [preferred_axis] if preferred_axis is not None else _candidate_axis_values(axis_low, axis_high, rng)
        axis_vals.reverse()
        perp_vals = _candidate_perp_values(base_perp, perp_low, perp_high, rng)
        for axis in axis_vals:
            if not (axis_low <= axis <= axis_high):
                continue
            for perp in perp_vals:
                bbox = _arrow_bbox((perp, axis))
                if _can_place_bbox(bbox, reserved, h, w):
                    return perp, axis
        return None
    if move_dir == "down":
        axis_low, axis_high = r1 + 2, h - 2
        perp_low, perp_high = 1, w - 2
        base_perp = (c0 + c1) // 2 if preferred_perp is None else preferred_perp
        axis_vals = [preferred_axis] if preferred_axis is not None else _candidate_axis_values(axis_low, axis_high, rng)
        perp_vals = _candidate_perp_values(base_perp, perp_low, perp_high, rng)
        for axis in axis_vals:
            if not (axis_low <= axis <= axis_high):
                continue
            for perp in perp_vals:
                bbox = _arrow_bbox((axis, perp))
                if _can_place_bbox(bbox, reserved, h, w):
                    return axis, perp
        return None
    axis_low, axis_high = 1, r0 - 2
    perp_low, perp_high = 1, w - 2
    base_perp = (c0 + c1) // 2 if preferred_perp is None else preferred_perp
    axis_vals = [preferred_axis] if preferred_axis is not None else _candidate_axis_values(axis_low, axis_high, rng)
    axis_vals.reverse()
    perp_vals = _candidate_perp_values(base_perp, perp_low, perp_high, rng)
    for axis in axis_vals:
        if not (axis_low <= axis <= axis_high):
            continue
        for perp in perp_vals:
            bbox = _arrow_bbox((axis, perp))
            if _can_place_bbox(bbox, reserved, h, w):
                return axis, perp
    return None


def _pick_cargo_bbox(
    rng: random.Random,
    size: int,
    side: str,
    reserved: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    preferred_band: int | None = None,
) -> tuple[int, int, int, int] | None:
    if side in ("left", "right"):
        row_low, row_high = 1, h - size - 1
        if row_low > row_high:
            return None
        if preferred_band is None:
            row_vals = list(range(row_low, row_high + 1))
            rng.shuffle(row_vals)
        else:
            row_vals = _candidate_perp_values(preferred_band, row_low, row_high, rng, max_offset=4)
        if side == "left":
            col_low, col_high = 1, max(1, w // 5)
            col_vals = list(range(col_low, min(col_high, w - size - 1) + 1))
        else:
            col_low, col_high = max(1, w - size - 1 - max(1, w // 5)), w - size - 1
            col_vals = list(range(col_low, col_high + 1))
        rng.shuffle(col_vals)
        for r0 in row_vals:
            for c0 in col_vals:
                bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
                if _can_place_bbox(bbox, reserved, h, w):
                    return bbox
        return None
    col_low, col_high = 1, w - size - 1
    if col_low > col_high:
        return None
    if preferred_band is None:
        col_vals = list(range(col_low, col_high + 1))
        rng.shuffle(col_vals)
    else:
        col_vals = _candidate_perp_values(preferred_band, col_low, col_high, rng, max_offset=4)
    if side == "top":
        row_low, row_high = 1, max(1, h // 5)
        row_vals = list(range(row_low, min(row_high, h - size - 1) + 1))
    else:
        row_low, row_high = max(1, h - size - 1 - max(1, h // 5)), h - size - 1
        row_vals = list(range(row_low, row_high + 1))
    rng.shuffle(row_vals)
    for c0 in col_vals:
        for r0 in row_vals:
            bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
            if _can_place_bbox(bbox, reserved, h, w):
                return bbox
    return None


def _choose_turn_dir(rng: random.Random, move_dir: str) -> str:
    options = [move_dir, ORTHOGONALS[move_dir][0], ORTHOGONALS[move_dir][1]]
    return rng.choices(options, weights=[2, 1, 1], k=1)[0]


def _build_random_spec(
    rng: random.Random,
    h: int,
    w: int,
    color: int,
    reserved: list[tuple[int, int, int, int]],
    avoid_output: list[tuple[int, int, int, int]],
    require_separate_output: bool,
) -> dict | None:
    for _ in range(SPEC_ATTEMPTS):
        side = rng.choice(list(FIRST_DIR_BY_SIDE))
        size = rng.choice([2, 3])
        arrow_count = rng.choices(ARROW_COUNT_VALUES, weights=ARROW_COUNT_WEIGHTS, k=1)[0]
        cargo_bbox = _pick_cargo_bbox(rng, size, side, reserved, h, w)
        if cargo_bbox is None:
            continue
        local_reserved = list(reserved)
        local_reserved.append(cargo_bbox)
        arrows: list[dict] = []
        rect = cargo_bbox
        first_dir = FIRST_DIR_BY_SIDE[side]
        center = _pick_center_ahead(rng, rect, first_dir, local_reserved, h, w)
        if center is None:
            continue
        arrows.append({"center": center, "dir": first_dir})
        local_reserved.append(_arrow_bbox(center))
        rect = _shift_bbox_to_reach(rect, first_dir, center)
        rect = _union_bbox(rect, _arrow_bbox(center))
        for idx in range(1, arrow_count):
            move_dir = arrows[-1]["dir"]
            pref_perp = (rect[0] + rect[2]) // 2 if move_dir in ("left", "right") else (rect[1] + rect[3]) // 2
            center = _pick_center_ahead(
                rng,
                rect,
                move_dir,
                local_reserved,
                h,
                w,
                preferred_perp=pref_perp,
            )
            if center is None:
                break
            own_dir = rng.choice(list(DIRS)) if idx == arrow_count - 1 else _choose_turn_dir(rng, move_dir)
            arrows.append({"center": center, "dir": own_dir})
            local_reserved.append(_arrow_bbox(center))
            rect = _shift_bbox_to_reach(rect, move_dir, center)
            rect = _union_bbox(rect, _arrow_bbox(center))
        if len(arrows) != arrow_count:
            continue
        final_rect = rect
        if require_separate_output and any(_bboxes_intersect(final_rect, other) for other in avoid_output):
            continue
        return {"color": color, "cargo_bbox": cargo_bbox, "arrows": arrows, "final_bbox": final_rect}
    return None


def _build_crossing_pair(
    rng: random.Random,
    h: int,
    w: int,
    colors: tuple[int, int],
    reserved: list[tuple[int, int, int, int]],
) -> list[dict] | None:
    color_a, color_b = colors
    for _ in range(SPEC_ATTEMPTS):
        center_r = rng.randint(h // 3, (2 * h) // 3)
        center_c = rng.randint(w // 3, (2 * w) // 3)
        reserved_now = list(reserved)

        size_a = rng.choice([2, 3])
        cargo_a = _pick_cargo_bbox(rng, size_a, "left", reserved_now, h, w, preferred_band=center_r - size_a // 2)
        if cargo_a is None:
            continue
        arrows_a: list[dict] = []
        local_a = reserved_now + [cargo_a]
        center1_a = _pick_center_ahead(rng, cargo_a, "right", local_a, h, w, preferred_perp=center_r - 1)
        if center1_a is None:
            continue
        arrows_a.append({"center": center1_a, "dir": "right"})
        local_a.append(_arrow_bbox(center1_a))
        rect_a = _shift_bbox_to_reach(cargo_a, "right", center1_a)
        rect_a = _union_bbox(rect_a, _arrow_bbox(center1_a))
        target_c = max(rect_a[3] + 2, center_c)
        center2_a = _pick_center_ahead(
            rng,
            rect_a,
            "right",
            local_a,
            h,
            w,
            preferred_perp=center_r - 1,
            preferred_axis=target_c,
        )
        if center2_a is None:
            continue
        arrows_a.append({"center": center2_a, "dir": rng.choice(["right", "up", "down"])})
        local_a.append(_arrow_bbox(center2_a))
        rect_a = _shift_bbox_to_reach(rect_a, "right", center2_a)
        rect_a = _union_bbox(rect_a, _arrow_bbox(center2_a))
        spec_a = {"color": color_a, "cargo_bbox": cargo_a, "arrows": arrows_a, "final_bbox": rect_a}

        reserved_b = local_a
        size_b = rng.choice([2, 3])
        cargo_b = _pick_cargo_bbox(rng, size_b, "top", reserved_b, h, w, preferred_band=center_c + 1 - size_b // 2)
        if cargo_b is None:
            continue
        arrows_b: list[dict] = []
        local_b = reserved_b + [cargo_b]
        center1_b = _pick_center_ahead(rng, cargo_b, "down", local_b, h, w, preferred_perp=center_c + 1)
        if center1_b is None:
            continue
        arrows_b.append({"center": center1_b, "dir": "down"})
        local_b.append(_arrow_bbox(center1_b))
        rect_b = _shift_bbox_to_reach(cargo_b, "down", center1_b)
        rect_b = _union_bbox(rect_b, _arrow_bbox(center1_b))
        target_r = max(rect_b[2] + 2, center_r + 1)
        center2_b = _pick_center_ahead(
            rng,
            rect_b,
            "down",
            local_b,
            h,
            w,
            preferred_perp=center_c + 1,
            preferred_axis=target_r,
        )
        if center2_b is None:
            continue
        arrows_b.append({"center": center2_b, "dir": rng.choice(["down", "left", "right"])})
        local_b.append(_arrow_bbox(center2_b))
        rect_b = _shift_bbox_to_reach(rect_b, "down", center2_b)
        rect_b = _union_bbox(rect_b, _arrow_bbox(center2_b))
        spec_b = {"color": color_b, "cargo_bbox": cargo_b, "arrows": arrows_b, "final_bbox": rect_b}

        if not _bboxes_intersect(spec_a["final_bbox"], spec_b["final_bbox"]):
            continue
        return [spec_a, spec_b]
    return None


def _validate_specs(specs: list[dict], h: int, w: int) -> np.ndarray | None:
    grid = _render_specs(specs, h, w)
    expected = _paint_output_from_specs(specs, h, w)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(*GRID_H_RANGE)
        w = rng.randint(*GRID_W_RANGE)
        want_overlap = rng.random() < 0.5
        n_objects = rng.randint(*NUM_OBJECTS_RANGE)
        if want_overlap:
            n_objects = max(2, n_objects)
        picked = rng.sample(colors, n_objects)
        specs: list[dict] = []
        reserved: list[tuple[int, int, int, int]] = []
        final_rects: list[tuple[int, int, int, int]] = []

        if want_overlap:
            pair = _build_crossing_pair(rng, h, w, (picked[0], picked[1]), reserved)
            if pair is None:
                continue
            specs.extend(pair)
            for spec in pair:
                reserved.extend(_occupied_bboxes_for_spec(spec))
                final_rects.append(spec["final_bbox"])
            start_idx = 2
        else:
            start_idx = 0

        ok = True
        for color in picked[start_idx:]:
            spec = _build_random_spec(
                rng,
                h,
                w,
                color,
                reserved,
                [] if want_overlap else final_rects,
                require_separate_output=not want_overlap,
            )
            if spec is None:
                ok = False
                break
            specs.append(spec)
            reserved.extend(_occupied_bboxes_for_spec(spec))
            final_rects.append(spec["final_bbox"])
        if not ok:
            continue

        if want_overlap:
            overlaps = 0
            for i in range(len(final_rects)):
                for j in range(i + 1, len(final_rects)):
                    if _bboxes_intersect(final_rects[i], final_rects[j]):
                        overlaps += 1
            if overlaps == 0:
                continue

        grid = _validate_specs(specs, h, w)
        if grid is None:
            continue
        return grid
    raise RuntimeError("generate_input failed to produce a valid puzzle instance")
