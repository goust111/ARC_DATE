import random
from collections import defaultdict

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_ALL_SIDES = ("top", "right", "bottom", "left")

# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    owner = np.full((h, w), -1, dtype=int)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            obj_id = len(objects)
            stack = [(r, c)]
            seen[r, c] = True
            owner[r, c] = obj_id
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    owner[nr, nc] = obj_id
                    stack.append((nr, nc))

            anchor = min(coords)
            objects.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "anchor": (int(anchor[0]), int(anchor[1])),
                }
            )

    return objects, owner


def _neighbor_contact_counts(
    objects: list[dict], owner: np.ndarray, obj_id: int
) -> dict[int, int]:
    h, w = owner.shape
    counts: dict[int, int] = defaultdict(int)
    for r, c in objects[obj_id]["coords"]:
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                other_id = int(owner[nr, nc])
                if other_id >= 0 and other_id != obj_id:
                    counts[other_id] += 1
    return dict(counts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object to the color of the touching neighbor object
    with the largest 4-neighbor contact area. Isolated objects stay unchanged.
    Ties are broken by the touching object's row-major anchor position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    objects, owner = _find_objects(g, bg)

    out = g.copy()
    for obj_id, obj in enumerate(objects):
        counts = _neighbor_contact_counts(objects, owner, obj_id)
        if not counts:
            continue

        winner = min(
            counts,
            key=lambda other_id: (
                -int(counts[other_id]),
                int(objects[other_id]["anchor"][0]),
                int(objects[other_id]["anchor"][1]),
                int(other_id),
            ),
        )
        new_color = int(objects[winner]["color"])
        for r, c in obj["coords"]:
            out[r, c] = new_color

    return out


# --------------- generator ---------------

GRID_H_BOUNDS = (18, 24)
GRID_W_BOUNDS = (18, 24)
OBJECT_SIDE_BOUNDS = (2, 5)
OBJECT_SIDE_SMALL_BOUNDS = (2, 4)
ISOLATED_COUNT_BOUNDS = (1, 2)
PROTRUSION_SPAN_MAX = 2
PROTRUSION_DEPTH_MAX = 2
PROTRUSION_PROB = 0.65
ISOLATED_PLACEMENT_TRIES = 150
GENERATION_ATTEMPTS = 400


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _clone_specs(specs: list[dict]) -> list[dict]:
    return [{"cells": set(spec["cells"]), "free_sides": set(spec["free_sides"])} for spec in specs]


def _cells_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_specs(specs: list[dict]) -> tuple[list[dict], int, int]:
    min_r = min(r for spec in specs for r, _ in spec["cells"])
    min_c = min(c for spec in specs for _, c in spec["cells"])
    normalized = []
    for spec in specs:
        shifted = {(r - min_r, c - min_c) for r, c in spec["cells"]}
        normalized.append({"cells": shifted, "free_sides": set(spec["free_sides"])})
    max_r = max(r for spec in normalized for r, _ in spec["cells"])
    max_c = max(c for spec in normalized for _, c in spec["cells"])
    return normalized, max_r + 1, max_c + 1


def _rotate_side(side: str, turns: int) -> str:
    idx = _ALL_SIDES.index(side)
    return _ALL_SIDES[(idx + turns) % 4]


def _rotate_specs(specs: list[dict], turns: int) -> tuple[list[dict], int, int]:
    turns %= 4
    rotated, h, w = _normalize_specs(_clone_specs(specs))
    if turns == 0:
        return rotated, h, w

    current_h, current_w = h, w
    for _ in range(turns):
        next_specs = []
        for spec in rotated:
            next_cells = {(c, current_h - 1 - r) for r, c in spec["cells"]}
            next_sides = {_rotate_side(side, 1) for side in spec["free_sides"]}
            next_specs.append({"cells": next_cells, "free_sides": next_sides})
        rotated, current_h, current_w = _normalize_specs(next_specs)

    return rotated, current_h, current_w


def _translate_specs(specs: list[dict], dr: int, dc: int) -> list[dict]:
    out = []
    for spec in specs:
        out.append(
            {
                "cells": {(r + dr, c + dc) for r, c in spec["cells"]},
                "free_sides": set(spec["free_sides"]),
            }
        )
    return out


def _interval_start(prev_start: int, prev_len: int, next_len: int, overlap: int) -> int:
    assert 1 <= overlap <= min(prev_len, next_len)
    overlap_start = random.randint(prev_start, prev_start + prev_len - overlap)
    if random.random() < 0.5:
        return overlap_start
    return overlap_start + overlap - next_len


def _build_line_specs(tie: bool) -> tuple[list[dict], int, int]:
    h0 = random.randint(*OBJECT_SIDE_BOUNDS)
    w0 = random.randint(*OBJECT_SIDE_BOUNDS)
    h1 = random.randint(*OBJECT_SIDE_BOUNDS)
    w1 = random.randint(*OBJECT_SIDE_BOUNDS)
    h2 = random.randint(*OBJECT_SIDE_BOUNDS)
    w2 = random.randint(*OBJECT_SIDE_BOUNDS)

    lim01 = min(h0, h1)
    lim12 = min(h1, h2)
    if tie:
        common = random.randint(1, min(lim01, lim12))
        overlap01 = common
        overlap12 = common
    else:
        choices = [(a, b) for a in range(1, lim01 + 1) for b in range(1, lim12 + 1) if a != b]
        overlap01, overlap12 = random.choice(choices)

    r0, c0 = 0, 0
    r1, c1 = _interval_start(r0, h0, h1, overlap01), c0 + w0
    r2, c2 = _interval_start(r1, h1, h2, overlap12), c1 + w1

    specs = [
        {"cells": _rect_cells(r0, c0, h0, w0), "free_sides": {"top", "bottom", "left"}},
        {"cells": _rect_cells(r1, c1, h1, w1), "free_sides": {"top", "bottom"}},
        {"cells": _rect_cells(r2, c2, h2, w2), "free_sides": {"top", "bottom", "right"}},
    ]

    return _rotate_specs(specs, random.randrange(4))


def _build_hub_specs(tie: bool) -> tuple[list[dict], int, int]:
    hc = random.randint(3, 5)
    wc = random.randint(3, 5)
    hl = random.randint(*OBJECT_SIDE_SMALL_BOUNDS)
    wl = random.randint(*OBJECT_SIDE_SMALL_BOUNDS)
    hr = random.randint(*OBJECT_SIDE_SMALL_BOUNDS)
    wr = random.randint(*OBJECT_SIDE_SMALL_BOUNDS)
    ht = random.randint(*OBJECT_SIDE_SMALL_BOUNDS)
    wt = random.randint(2, 5)

    left_lim = min(hc, hl)
    right_lim = min(hc, hr)
    top_lim = min(wc, wt)

    if tie:
        tie_span = random.randint(2, min(left_lim, top_lim))
        right_span = random.randint(1, min(right_lim, tie_span - 1))
        left_span = tie_span
        top_span = tie_span
    else:
        winner = random.choice(("left", "right", "top"))
        if winner == "left":
            left_span = random.randint(2, left_lim)
            right_span = random.randint(1, min(right_lim, left_span - 1))
            top_span = random.randint(1, min(top_lim, left_span - 1))
        elif winner == "right":
            right_span = random.randint(2, right_lim)
            left_span = random.randint(1, min(left_lim, right_span - 1))
            top_span = random.randint(1, min(top_lim, right_span - 1))
        else:
            top_span = random.randint(2, top_lim)
            left_span = random.randint(1, min(left_lim, top_span - 1))
            right_span = random.randint(1, min(right_lim, top_span - 1))

    rc, cc = 0, 0
    rl, cl = _interval_start(rc, hc, hl, left_span), -wl
    rr, cr = _interval_start(rc, hc, hr, right_span), wc
    rt, ct = -ht, _interval_start(cc, wc, wt, top_span)

    specs = [
        {"cells": _rect_cells(rc, cc, hc, wc), "free_sides": {"bottom"}},
        {"cells": _rect_cells(rl, cl, hl, wl), "free_sides": {"top", "bottom", "left"}},
        {"cells": _rect_cells(rr, cr, hr, wr), "free_sides": {"top", "bottom", "right"}},
        {"cells": _rect_cells(rt, ct, ht, wt), "free_sides": {"top", "left", "right"}},
    ]

    return _rotate_specs(specs, random.randrange(4))


def _specs_occupied(specs: list[dict]) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        occupied |= spec["cells"]
    return occupied


def _try_add_protrusion(spec: dict, other_cells: set[tuple[int, int]], h: int, w: int) -> None:
    if random.random() >= PROTRUSION_PROB:
        return

    r0, c0, r1, c1 = _cells_bbox(spec["cells"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    sides = list(spec["free_sides"])
    random.shuffle(sides)

    for side in sides:
        side_len = width if side in {"top", "bottom"} else height
        span_hi = min(PROTRUSION_SPAN_MAX, side_len)
        if span_hi < 1:
            continue

        span = random.randint(1, span_hi)
        offset = random.randint(0, side_len - span)
        depth = random.randint(1, PROTRUSION_DEPTH_MAX)
        arm: set[tuple[int, int]] = set()

        if side == "top":
            for rr in range(r0 - depth, r0):
                for cc in range(c0 + offset, c0 + offset + span):
                    arm.add((rr, cc))
        elif side == "bottom":
            for rr in range(r1 + 1, r1 + 1 + depth):
                for cc in range(c0 + offset, c0 + offset + span):
                    arm.add((rr, cc))
        elif side == "left":
            for rr in range(r0 + offset, r0 + offset + span):
                for cc in range(c0 - depth, c0):
                    arm.add((rr, cc))
        else:
            for rr in range(r0 + offset, r0 + offset + span):
                for cc in range(c1 + 1, c1 + 1 + depth):
                    arm.add((rr, cc))

        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in arm):
            continue
        if arm & other_cells or arm & spec["cells"]:
            continue

        bad = False
        for rr, cc in arm:
            for dr, dc in _NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if (nr, nc) in other_cells:
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue

        spec["cells"] |= arm
        return


def _apply_protrusions(specs: list[dict], h: int, w: int) -> None:
    for idx, spec in enumerate(specs):
        other_cells: set[tuple[int, int]] = set()
        for j, other in enumerate(specs):
            if j != idx:
                other_cells |= other["cells"]
        _try_add_protrusion(spec, other_cells, h, w)


def _can_place_isolated(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for dr, dc in _NEIGHBORS4:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _place_isolated_objects(specs: list[dict], h: int, w: int, count: int) -> bool:
    occupied = _specs_occupied(specs)
    for _ in range(count):
        placed = False
        for _attempt in range(ISOLATED_PLACEMENT_TRIES):
            hh = random.randint(*OBJECT_SIDE_SMALL_BOUNDS)
            ww = random.randint(*OBJECT_SIDE_SMALL_BOUNDS)
            r0 = random.randint(0, h - hh)
            c0 = random.randint(0, w - ww)
            candidate = _rect_cells(r0, c0, hh, ww)
            if not _can_place_isolated(candidate, occupied, h, w):
                continue

            spec = {"cells": candidate, "free_sides": set(_ALL_SIDES)}
            _try_add_protrusion(spec, occupied, h, w)
            if not _can_place_isolated(spec["cells"], occupied, h, w):
                continue

            specs.append(spec)
            occupied |= spec["cells"]
            placed = True
            break
        if not placed:
            return False
    return True


def _render_grid(shape: tuple[int, int], bg: int, specs: list[dict], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec, color in zip(specs, colors):
        for r, c in spec["cells"]:
            grid[r, c] = int(color)
    return grid


def _branch_summary(grid: np.ndarray) -> tuple[bool, bool, int]:
    bg = _background_color(grid)
    objects, owner = _find_objects(grid, bg)
    has_tie = False
    has_isolated = False
    max_degree = 0

    for obj_id in range(len(objects)):
        counts = _neighbor_contact_counts(objects, owner, obj_id)
        degree = len(counts)
        max_degree = max(max_degree, degree)
        if degree == 0:
            has_isolated = True
            continue
        best = max(counts.values())
        if sum(value == best for value in counts.values()) > 1:
            has_tie = True

    return has_tie, has_isolated, max_degree


def _valid_generated_grid(
    grid: np.ndarray, *, want_tie: bool, want_isolated: bool, expected_objects: int
) -> bool:
    bg = _background_color(grid)
    objects, _owner = _find_objects(grid, bg)
    if len(objects) != expected_objects:
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False

    bg_count = int(np.count_nonzero(grid == bg))
    fg_counts = [len(obj["coords"]) for obj in objects]
    if not fg_counts or bg_count <= max(fg_counts):
        return False

    has_tie, has_isolated, max_degree = _branch_summary(grid)
    if has_tie != want_tie:
        return False
    if has_isolated != want_isolated:
        return False
    if max_degree > 3:
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def _build_mode_specs(mode: str) -> tuple[list[dict], int, int, bool, bool, int]:
    if mode == "line_unique_iso":
        base_specs, cluster_h, cluster_w = _build_line_specs(tie=False)
        want_tie = False
        want_isolated = True
        isolated_count = random.randint(*ISOLATED_COUNT_BOUNDS)
    elif mode == "hub_unique_iso":
        base_specs, cluster_h, cluster_w = _build_hub_specs(tie=False)
        want_tie = False
        want_isolated = True
        isolated_count = random.randint(*ISOLATED_COUNT_BOUNDS)
    elif mode == "line_tie":
        base_specs, cluster_h, cluster_w = _build_line_specs(tie=True)
        want_tie = True
        want_isolated = False
        isolated_count = 0
    else:
        assert mode == "hub_tie"
        base_specs, cluster_h, cluster_w = _build_hub_specs(tie=True)
        want_tie = True
        want_isolated = False
        isolated_count = 0

    return base_specs, cluster_h, cluster_w, want_tie, want_isolated, isolated_count


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("line_unique_iso", "hub_unique_iso", "line_tie", "hub_tie"))
    h = random.randint(*GRID_H_BOUNDS)
    w = random.randint(*GRID_W_BOUNDS)

    base_specs, cluster_h, cluster_w, want_tie, want_isolated, isolated_count = _build_mode_specs(mode)
    if cluster_h > h or cluster_w > w:
        return None

    shift_r = random.randint(0, h - cluster_h)
    shift_c = random.randint(0, w - cluster_w)
    specs = _translate_specs(base_specs, shift_r, shift_c)
    _apply_protrusions(specs, h, w)

    if isolated_count and not _place_isolated_objects(specs, h, w, isolated_count):
        return None

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    if len(specs) > len(palette):
        return None
    colors = random.sample(palette, len(specs))
    grid = _render_grid((h, w), bg, specs, colors)

    if _valid_generated_grid(
        grid,
        want_tie=want_tie,
        want_isolated=want_isolated,
        expected_objects=len(specs),
    ):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
