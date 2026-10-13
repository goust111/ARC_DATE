from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
MARKER = 2
BODY_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)

AREA_DIMS = {
    6: ((2, 3), (3, 2)),
    8: ((2, 4), (4, 2)),
    9: ((3, 3),),
    10: ((2, 5), (5, 2)),
    12: ((3, 4), (4, 3)),
    15: ((3, 5), (5, 3)),
    16: ((4, 4),),
    20: ((4, 5), (5, 4)),
}

_MARKER_CACHE: dict[tuple[int, int, int, tuple[int, int]], list[tuple[tuple[int, int], ...]]] = {}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _direction_from_markers(marker_coords: np.ndarray, h: int, w: int) -> tuple[int, int]:
    cr, cc = marker_coords.mean(axis=0)
    mr = (h - 1) / 2.0
    mc = (w - 1) / 2.0
    dr = float(cr - mr)
    dc = float(cc - mc)
    assert dr != 0.0 or dc != 0.0, "marker centroid must not equal rectangle center"
    if abs(dr) >= abs(dc):
        return UP if dr < 0 else DOWN
    return LEFT if dc < 0 else RIGHT


def _is_tie_case(marker_coords: np.ndarray, h: int, w: int) -> bool:
    cr, cc = marker_coords.mean(axis=0)
    mr = (h - 1) / 2.0
    mc = (w - 1) / 2.0
    dr = abs(float(cr - mr))
    dc = abs(float(cc - mc))
    return dr > 0.0 and abs(dr - dc) < 1e-9


def _marker_patterns(
    h: int, w: int, count: int, direction: tuple[int, int]
) -> list[tuple[tuple[int, int], ...]]:
    key = (h, w, count, direction)
    if key in _MARKER_CACHE:
        return _MARKER_CACHE[key]

    cells = [(r, c) for r in range(h) for c in range(w)]
    patterns: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(cells, count):
        coords = np.array(combo, dtype=float)
        cr, cc = coords.mean(axis=0)
        mr = (h - 1) / 2.0
        mc = (w - 1) / 2.0
        if abs(float(cr - mr)) < 1e-9 and abs(float(cc - mc)) < 1e-9:
            continue
        if _direction_from_markers(coords, h, w) != direction:
            continue
        patterns.append(tuple((int(r), int(c)) for r, c in combo))
    _MARKER_CACHE[key] = patterns
    return patterns


def _choose_markers(
    h: int,
    w: int,
    count: int,
    direction: tuple[int, int],
    *,
    prefer_tie: bool,
) -> tuple[tuple[int, int], ...]:
    candidates = _marker_patterns(h, w, count, direction)
    assert candidates, f"no marker pattern for {(h, w, count, direction)}"
    if prefer_tie:
        tied = [
            pattern
            for pattern in candidates
            if _is_tie_case(np.array(pattern, dtype=float), h, w)
        ]
        if tied:
            return random.choice(tied)
    non_tied = [
        pattern
        for pattern in candidates
        if not _is_tie_case(np.array(pattern, dtype=float), h, w)
    ]
    return random.choice(non_tied or candidates)


def _box_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1)


def _shift_box(
    box: tuple[int, int, int, int], direction: tuple[int, int], steps: int
) -> tuple[int, int, int, int]:
    r, c, h, w = box
    dr, dc = direction
    return int(r + dr * steps), int(c + dc * steps), int(h), int(w)


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dr, -dc


def _box_in_bounds(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r, c, h, w = box
    return 0 <= r and 0 <= c and r + h <= shape[0] and c + w <= shape[1]


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (ar + ah <= br or br + bh <= ar or ac + aw <= bc or bc + bw <= ac)


def _box_touch_or_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + pad <= br
        or br + bh + pad <= ar
        or ac + aw + pad <= bc
        or bc + bw + pad <= ac
    )


def _next_step_cause(
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
    obstacles: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> str | None:
    nxt = _shift_box(box, direction, 1)
    if not _box_in_bounds(nxt, shape):
        return "boundary"
    if any(_box_overlap(nxt, obs) for obs in obstacles):
        return "object"
    return None


def _path_clear(
    start_box: tuple[int, int, int, int],
    direction: tuple[int, int],
    steps: int,
    obstacles: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> bool:
    for k in range(steps + 1):
        cur = _shift_box(start_box, direction, k)
        if not _box_in_bounds(cur, shape):
            return False
        if any(_box_overlap(cur, obs) for obs in obstacles):
            return False
    return True


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects: list[dict] = []
    for coords in _find_components(g != BG):
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(g, bbox)
        assert np.all(patch != BG), "each component must be a filled rectangle with interior markers"
        colors = set(map(int, np.unique(patch)))
        assert MARKER in colors, "each rectangle must contain red markers"
        body_colors = sorted(colors - {MARKER})
        assert len(body_colors) == 1, "each rectangle must have exactly one body color"
        body_color = body_colors[0]
        marker_coords = np.argwhere(patch == MARKER)
        assert 1 <= len(marker_coords) <= 3, "marker count must be in [1, 3]"
        h, w = map(int, patch.shape)
        assert int(coords.shape[0]) == h * w, "component bbox must be a solid rectangle"
        direction = _direction_from_markers(marker_coords.astype(float), h, w)
        rects.append(
            {
                "bbox": bbox,
                "box": _box_from_bbox(bbox),
                "patch": patch,
                "body_color": int(body_color),
                "pure_patch": np.full((h, w), int(body_color), dtype=int),
                "distance": int(len(marker_coords)),
                "direction": direction,
                "area": int(h * w),
                "tie_case": _is_tie_case(marker_coords.astype(float), h, w),
            }
        )
    return rects


def _simulate(
    objects: list[dict],
    shape: tuple[int, int],
    *,
    order: str = "asc",
    force_distance: int | None = None,
    reverse_direction: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    if order == "asc":
        seq = sorted(objects, key=lambda obj: (obj["area"], obj["bbox"][0], obj["bbox"][1]))
    elif order == "desc":
        seq = sorted(objects, key=lambda obj: (-obj["area"], obj["bbox"][0], obj["bbox"][1]))
    else:
        raise ValueError(f"unsupported order: {order}")

    out = np.zeros(shape, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    trace: list[dict] = []
    for obj in seq:
        direction = _opposite(obj["direction"]) if reverse_direction else obj["direction"]
        distance = int(force_distance if force_distance is not None else obj["distance"])
        cur = obj["box"]
        actual = 0
        cause = "distance"
        for _ in range(distance):
            nxt = _shift_box(cur, direction, 1)
            if not _box_in_bounds(nxt, shape):
                cause = "boundary"
                break
            if any(_box_overlap(nxt, other) for other in placed_boxes):
                cause = "object"
                break
            cur = nxt
            actual += 1
        r, c, h, w = cur
        out[r : r + h, c : c + w] = obj["pure_patch"]
        placed_boxes.append(cur)
        trace.append(
            {
                "area": obj["area"],
                "body_color": obj["body_color"],
                "direction": obj["direction"],
                "distance": obj["distance"],
                "used_direction": direction,
                "used_distance": distance,
                "actual": actual,
                "cause": cause,
                "final_box": cur,
                "tie_case": obj["tie_case"],
            }
        )
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked rectangle by the direction and distance encoded by its red interior markers.

    Input structure:
    - The grid has a black background and 3 to 5 disjoint filled rectangles.
    - Each rectangle is a single body color with 1 to 3 red marker cells embedded inside it.
    - Rectangle areas are distinct, so the rectangles define a unique small-to-large processing order.

    Rule:
    - For each rectangle, count its red markers to get the slide distance.
    - Compare the red-marker centroid with the rectangle-center; the dominant signed row/col bias gives the slide direction, with row bias winning ties.
    - Process rectangles from smallest area to largest, sliding each whole rectangle on an otherwise empty canvas until it has moved its counted distance or the next step would hit the boundary or an already settled rectangle.
    - After all rectangles settle, recolor their former red marker cells back to the rectangle's body color.

    Key insight:
    - The same red markers encode two independent parameters at once: how far the rectangle moves and which cardinal direction it moves in.
    """
    objects = _parse_rectangles(np.asarray(grid, dtype=int))
    out, _trace = _simulate(objects, tuple(map(int, grid.shape)))
    return out


def _choose_dims(
    area: int, direction: tuple[int, int], distance: int, prefer_tie: bool
) -> tuple[int, int]:
    dims = list(AREA_DIMS[area])
    random.shuffle(dims)
    ranked: list[tuple[int, int]] = []
    tied: list[tuple[int, int]] = []
    for h, w in dims:
        patterns = _marker_patterns(h, w, distance, direction)
        if not patterns:
            continue
        ranked.append((h, w))
        if any(_is_tie_case(np.array(pattern, dtype=float), h, w) for pattern in patterns):
            tied.append((h, w))
    assert ranked, f"no valid dims for area={area}, distance={distance}, direction={direction}"
    if prefer_tie and tied:
        return random.choice(tied)
    return random.choice(ranked)


def _candidate_positions_full(
    h: int, w: int, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    H, W = shape
    out = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(out)
    return out


def _candidate_positions_boundary(
    h: int, w: int, shape: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    H, W = shape
    if direction == UP:
        out = [(0, c) for c in range(W - w + 1)]
    elif direction == DOWN:
        out = [(H - h, c) for c in range(W - w + 1)]
    elif direction == LEFT:
        out = [(r, 0) for r in range(H - h + 1)]
    else:
        out = [(r, W - w) for r in range(H - h + 1)]
    random.shuffle(out)
    return out


def _place_free_mover(
    spec: dict,
    shape: tuple[int, int],
    final_boxes: list[tuple[int, int, int, int]],
    input_boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = spec["dims"]
    if spec["stop"] == "boundary":
        positions = _candidate_positions_boundary(h, w, shape, spec["direction"])
    else:
        positions = _candidate_positions_full(h, w, shape)

    for r, c in positions:
        final_box = (r, c, h, w)
        if any(_box_overlap(final_box, other) for other in final_boxes):
            continue
        if any(_box_touch_or_overlap(final_box, other, pad=1) for other in final_boxes):
            continue
        start_box = _shift_box(final_box, _opposite(spec["direction"]), spec["move"])
        if not _box_in_bounds(start_box, shape):
            continue
        if any(_box_touch_or_overlap(start_box, other, pad=1) for other in input_boxes):
            continue
        if not _path_clear(start_box, spec["direction"], spec["move"], final_boxes, shape):
            continue
        cause = _next_step_cause(final_box, spec["direction"], final_boxes, shape)
        if spec["stop"] == "boundary" and cause != "boundary":
            continue
        return final_box, start_box
    return None


def _overlap_interval_starts(
    fixed0: int, fixed_len: int, moving_len: int, limit: int
) -> list[int]:
    lo = max(0, fixed0 - moving_len + 1)
    hi = min(limit - moving_len, fixed0 + fixed_len - 1)
    if lo > hi:
        return []
    out = list(range(lo, hi + 1))
    random.shuffle(out)
    return out


def _object_collision_candidates(
    obstacle: tuple[int, int, int, int],
    h: int,
    w: int,
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    H, W = shape
    orow, ocol, oh, ow = obstacle
    if direction == UP:
        r = orow + oh
        return [(r, c) for c in _overlap_interval_starts(ocol, ow, w, W) if r + h <= H]
    if direction == DOWN:
        r = orow - h
        return [(r, c) for c in _overlap_interval_starts(ocol, ow, w, W) if r >= 0]
    if direction == LEFT:
        c = ocol + ow
        return [(r, c) for r in _overlap_interval_starts(orow, oh, h, H) if c + w <= W]
    c = ocol - w
    return [(r, c) for r in _overlap_interval_starts(orow, oh, h, H) if c >= 0]


def _place_object_collision(
    spec: dict,
    shape: tuple[int, int],
    final_boxes: list[tuple[int, int, int, int]],
    input_boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = spec["dims"]
    obstacle_indices = list(range(len(final_boxes)))
    random.shuffle(obstacle_indices)
    for idx in obstacle_indices:
        obstacle = final_boxes[idx]
        candidates = _object_collision_candidates(obstacle, h, w, shape, spec["direction"])
        random.shuffle(candidates)
        for r, c in candidates:
            final_box = (r, c, h, w)
            if any(_box_overlap(final_box, other) for other in final_boxes):
                continue
            if any(
                i != idx and _box_touch_or_overlap(final_box, other, pad=1)
                for i, other in enumerate(final_boxes)
            ):
                continue
            start_box = _shift_box(final_box, _opposite(spec["direction"]), spec["move"])
            if not _box_in_bounds(start_box, shape):
                continue
            if any(_box_touch_or_overlap(start_box, other, pad=1) for other in input_boxes):
                continue
            if not _path_clear(start_box, spec["direction"], spec["move"], final_boxes, shape):
                continue
            if _next_step_cause(final_box, spec["direction"], final_boxes, shape) != "object":
                continue
            return final_box, start_box
    return None


def _build_specs(n_rects: int) -> list[dict]:
    areas = sorted(random.sample(list(AREA_DIMS), n_rects))
    colors = random.sample(BODY_COLORS, n_rects)

    dir0 = random.choice(DIRECTIONS)
    verticals = [UP, DOWN]
    if dir0 in verticals:
        dir1 = DOWN if dir0 == UP else UP
    else:
        dir1 = random.choice(verticals)
    d0 = random.choice((2, 3))
    d2 = 2 if d0 == 3 else 3

    raw_specs = [
        {
            "area": areas[0],
            "body_color": colors[0],
            "direction": dir0,
            "distance": d0,
            "move": d0,
            "stop": "distance",
            "prefer_tie": False,
        },
        {
            "area": areas[1],
            "body_color": colors[1],
            "direction": dir1,
            "distance": d0,
            "move": d0,
            "stop": "distance",
            "prefer_tie": True,
        },
        {
            "area": areas[2],
            "body_color": colors[2],
            "direction": dir0,
            "distance": d2,
            "move": random.randint(1, d2 - 1),
            "stop": "object",
            "prefer_tie": False,
        },
    ]

    for i in range(3, n_rects):
        direction = random.choice(DIRECTIONS)
        distance = random.randint(1, 3)
        if distance >= 2 and random.random() < 0.4:
            move = random.randint(1, distance - 1)
            stop = "boundary"
        else:
            move = distance
            stop = "distance"
        raw_specs.append(
            {
                "area": areas[i],
                "body_color": colors[i],
                "direction": direction,
                "distance": distance,
                "move": move,
                "stop": stop,
                "prefer_tie": False,
            }
        )

    for spec in raw_specs:
        spec["dims"] = _choose_dims(
            spec["area"],
            spec["direction"],
            spec["distance"],
            prefer_tie=bool(spec["prefer_tie"]),
        )
        spec["markers"] = _choose_markers(
            spec["dims"][0],
            spec["dims"][1],
            spec["distance"],
            spec["direction"],
            prefer_tie=bool(spec["prefer_tie"]),
        )
    return raw_specs


def _render_input(specs: list[dict], boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec, box in zip(specs, boxes):
        r, c, h, w = box
        grid[r : r + h, c : c + w] = int(spec["body_color"])
        for mr, mc in spec["markers"]:
            grid[r + mr, c + mc] = MARKER
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        objects = _parse_rectangles(grid)
    except AssertionError:
        return False

    if not (3 <= len(objects) <= 5):
        return False
    if len({obj["area"] for obj in objects}) != len(objects):
        return False
    if len({obj["body_color"] for obj in objects}) != len(objects):
        return False
    if len({obj["direction"] for obj in objects}) < 2:
        return False
    if len({obj["distance"] for obj in objects}) < 2:
        return False

    has_same_distance_diff_dir = any(
        a["distance"] == b["distance"] and a["direction"] != b["direction"]
        for a, b in itertools.combinations(objects, 2)
    )
    has_same_dir_diff_distance = any(
        a["direction"] == b["direction"] and a["distance"] != b["distance"]
        for a, b in itertools.combinations(objects, 2)
    )
    if not has_same_distance_diff_dir or not has_same_dir_diff_distance:
        return False

    correct, trace = _simulate(objects, tuple(map(int, grid.shape)))
    if not np.array_equal(main(grid), correct):
        return False
    if not any(t["cause"] == "object" and t["actual"] < t["distance"] for t in trace):
        return False
    if not any(t["actual"] == t["distance"] and t["distance"] > 1 for t in trace):
        return False

    wrong_distance, _ = _simulate(objects, tuple(map(int, grid.shape)), force_distance=1)
    wrong_direction, _ = _simulate(objects, tuple(map(int, grid.shape)), reverse_direction=True)
    wrong_order, _ = _simulate(objects, tuple(map(int, grid.shape)), order="desc")
    if np.array_equal(wrong_distance, correct):
        return False
    if np.array_equal(wrong_direction, correct):
        return False
    if np.array_equal(wrong_order, correct):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(14, 18), random.randint(14, 18))
    n_rects = random.randint(3, 5)
    specs = _build_specs(n_rects)

    final_boxes: list[tuple[int, int, int, int]] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    for spec in specs:
        if spec["stop"] == "object":
            placed = _place_object_collision(spec, shape, final_boxes, input_boxes)
        else:
            placed = _place_free_mover(spec, shape, final_boxes, input_boxes)
        if placed is None:
            return None
        final_box, input_box = placed
        final_boxes.append(final_box)
        input_boxes.append(input_box)

    grid = _render_input(specs, input_boxes, shape)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marked-dual-signal sliding input")
