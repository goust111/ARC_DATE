from __future__ import annotations

import heapq
from collections import Counter

import numpy as np


DIR4 = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
EDGE_ORDER = ("U", "D", "L", "R")
NON_BG_COLORS = tuple(range(1, 10))

GRID_MIN = 14
GRID_MAX = 22
MAX_CUPS = 3
MAX_CONTENT = 6
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(Counter(map(int, vals)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    return sorted(int(v) for v in np.unique(grid) if int(v) not in exclude)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_coords(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _edge_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "D":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "L":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _analyze_cup(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    border_color = int(grid[coords[0, 0], coords[0, 1]])
    r0, c0, r1, c1 = _bbox_coords(coords)
    bbox = (r0, c0, r1, c1)
    border_set = {(int(r), int(c)) for r, c in coords}
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

    candidates = []
    for side in EDGE_ORDER:
        on_edge = [cell for cell in border_set if cell in set(_edge_cells(bbox, side))]
        if len(on_edge) == 1 and on_edge[0] not in corners:
            candidates.append((side, on_edge[0]))

    passing = []
    for tail_dir, spike in candidates:
        mouth_side = OPPOSITE[tail_dir]
        mouth_region = _flood_from_bbox_side(border_set, bbox, mouth_side)
        mouth_edge = [cell for cell in _edge_cells(bbox, mouth_side) if cell in mouth_region]
        if not mouth_edge:
            continue
        content_colors = {
            int(grid[r, c])
            for (r, c) in mouth_region
            if int(grid[r, c]) not in {bg, border_color}
        }
        if len(content_colors) != 1:
            continue
        content_color = next(iter(content_colors))
        content_cells = sorted((r, c) for (r, c) in mouth_region if int(grid[r, c]) == content_color)
        if not content_cells:
            continue
        passing.append(
            {
                "tail_dir": tail_dir,
                "spike": spike,
                "mouth_side": mouth_side,
                "spacing": len(mouth_edge),
                "content_color": content_color,
                "content_cells": content_cells,
                "border_color": border_color,
                "bbox": bbox,
            }
        )

    if len(passing) != 1:
        return None
    return passing[0]


def _analyze_all_cups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    cups: list[dict] = []
    for color in _colors_present(g, exclude={bg}):
        for coords in _component_coords(g == color, connectivity=8):
            if coords.shape[0] < 5:
                continue
            cup = _analyze_cup(g, coords, bg)
            if cup is not None:
                cups.append(cup)
    return cups


def _settle_emissions(base: np.ndarray, cups: list[dict]) -> tuple[np.ndarray, dict]:
    out = np.array(base, dtype=int, copy=True)
    bg = _get_bg(out)

    occupied: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(out != bg):
        occupied[(int(r), int(c))] = -1

    heap: list[tuple[int, int, int, int]] = []
    for cup_idx, cup in enumerate(cups):
        for r, c in cup["content_cells"]:
            out[r, c] = bg
            occupied.pop((r, c), None)
        count = len(cup["content_cells"])
        for slot in range(1, count + 1):
            heapq.heappush(heap, (cup["spacing"] * slot, slot, cup_idx, slot))

    stats = {
        "placements": 0,
        "delays": 0,
        "cross_cup_delays": 0,
        "self_delays": 0,
        "static_delays": 0,
    }

    while heap:
        dist, order_slot, cup_idx, _token = heapq.heappop(heap)
        cup = cups[cup_idx]
        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        rr = sr + dr * dist
        cc = sc + dc * dist
        if not _in_bounds(out.shape, rr, cc):
            continue
        owner = occupied.get((rr, cc))
        if owner is None:
            occupied[(rr, cc)] = cup_idx
            out[rr, cc] = cup["content_color"]
            stats["placements"] += 1
        else:
            stats["delays"] += 1
            if owner == -1:
                stats["static_delays"] += 1
            elif owner == cup_idx:
                stats["self_delays"] += 1
            else:
                stats["cross_cup_delays"] += 1
            heapq.heappush(heap, (dist + cup["spacing"], order_slot, cup_idx, _token))

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each cup's content along its tail with cadence set by the mouth width, settling conflicts by nearer-first cascading.

    Input structure:
    - The dominant color is background.
    - One or more hollow cup frames appear, each as an 8-connected single-color border with a single tail spike on one bbox edge and an open mouth on the opposite edge.
    - Each cup contains one content color inside its mouth region.

    Rule:
    - For each cup, identify the tail direction from the unique one-pixel edge spike and identify the mouth on the opposite side.
    - Count the cup's content pixels, erase them from the cup, and use the number of open mouth-edge cells as the ejection spacing.
    - Launch one content pixel per erased pixel outward from the spike at distances spacing, 2*spacing, 3*spacing, ...
    - Resolve all landing conflicts globally by nearest-first order: if a target is already claimed by a nearer pixel, the loser moves to the next cadence position on its own ray, repeating until it lands or leaves the grid.

    Key insight:
    - The launch period comes from the cup's geometry, not from the content arrangement, and conflict resolution must be settled as a global queue of tentative distances rather than by simple overwriting.
    """
    g = np.asarray(grid, dtype=int)
    cups = _analyze_all_cups(g)
    out, _stats = _settle_emissions(g, cups)
    return out


def _build_canonical_cup(mouth_width: int, spike_col: int, extra_left: int, extra_right: int) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    width = mouth_width + 2
    left_len = spike_col
    right_len = width - 1 - spike_col
    height = max(left_len + extra_left, right_len + extra_right) + 1
    patch = np.zeros((height, width), dtype=int)
    patch[0, spike_col] = 1
    for i in range(1, left_len + 1):
        patch[i, spike_col - i] = 1
    for i in range(1, right_len + 1):
        patch[i, spike_col + i] = 1
    patch[left_len + 1 :, 0] = 1
    patch[right_len + 1 :, width - 1] = 1
    border_set = {(int(r), int(c)) for r, c in np.argwhere(patch == 1)}
    mouth = sorted(_flood_from_bbox_side(border_set, (0, 0, height - 1, width - 1), "D"))
    return patch, (0, spike_col), mouth


def _rotate_coords(coords: list[tuple[int, int]], shape: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = shape
    if direction == "U":
        return list(coords)
    if direction == "D":
        return [(h - 1 - r, c) for r, c in coords]
    if direction == "L":
        return [(w - 1 - c, r) for r, c in coords]
    return [(c, h - 1 - r) for r, c in coords]


def _rotate_patch_and_points(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    if direction == "U":
        return patch.copy(), spike, list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - spike[0], spike[1]), _rotate_coords(mouth, (h, w), "D")
    if direction == "L":
        return np.rot90(patch, 1).copy(), (w - 1 - spike[1], spike[0]), _rotate_coords(mouth, (h, w), "L")
    return np.rot90(patch, -1).copy(), (spike[1], h - 1 - spike[0]), _rotate_coords(mouth, (h, w), "R")


def _mouth_order(cells: list[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    if direction == "U":
        key = lambda p: (-p[0], p[1])
    elif direction == "D":
        key = lambda p: (p[0], p[1])
    elif direction == "L":
        key = lambda p: (-p[1], p[0])
    else:
        key = lambda p: (p[1], p[0])
    return sorted(cells, key=key)


def _sample_cup_spec(rng: np.random.Generator, direction: str, border_color: int, content_color: int) -> dict:
    while True:
        mouth_width = int(rng.integers(2, 6))
        spike_col = int(rng.integers(1, mouth_width + 1))
        extra_left = int(rng.integers(1, 4))
        extra_right = int(rng.integers(1, 4))
        patch, spike, mouth = _build_canonical_cup(mouth_width, spike_col, extra_left, extra_right)
        patch, spike, mouth = _rotate_patch_and_points(patch, spike, mouth, direction)
        ordered_mouth = _mouth_order(mouth, direction)
        if len(ordered_mouth) >= 3:
            return {
                "direction": direction,
                "border_color": border_color,
                "content_color": content_color,
                "patch": patch,
                "spike_local": spike,
                "mouth_local": ordered_mouth,
                "spacing": mouth_width,
                "capacity": min(MAX_CONTENT, len(ordered_mouth)),
            }


def _tip_candidates(spec: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = spec["patch"].shape
    sr, sc = spec["spike_local"]
    h_grid, w_grid = shape
    rows = range(sr, h_grid - (h - sr) + 1)
    cols = range(sc, w_grid - (w - sc) + 1)
    return [(int(r), int(c)) for r in rows for c in cols]


def _instance_from_tip(spec: dict, tip: tuple[int, int]) -> dict:
    sr, sc = spec["spike_local"]
    patch = spec["patch"]
    ar = tip[0] - sr
    ac = tip[1] - sc
    border_local = np.argwhere(patch == 1)
    border_cells = [(ar + int(r), ac + int(c)) for r, c in border_local]
    content_local = spec["mouth_local"]
    content_cells = [(ar + r, ac + c) for r, c in content_local]
    bbox = (ar, ac, ar + patch.shape[0] - 1, ac + patch.shape[1] - 1)
    return {
        **spec,
        "tip": tip,
        "spike": tip,
        "anchor": (ar, ac),
        "bbox": bbox,
        "border_cells": border_cells,
        "content_global_all": content_cells,
    }


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _ray_path_cells(shape: tuple[int, int], spike: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR4[direction]
    r, c = spike
    cells = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _cell_in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _ray_hits_bbox(shape: tuple[int, int], spike: tuple[int, int], direction: str, bbox: tuple[int, int, int, int]) -> bool:
    return any(_cell_in_bbox(cell, bbox) for cell in _ray_path_cells(shape, spike, direction))


def _target_positions(shape: tuple[int, int], inst: dict) -> dict[tuple[int, int], tuple[int, int]]:
    dr, dc = DIR4[inst["direction"]]
    sr, sc = inst["spike"]
    targets: dict[tuple[int, int], tuple[int, int]] = {}
    for k in range(1, inst["capacity"] + 1):
        dist = inst["spacing"] * k
        rr = sr + dr * dist
        cc = sc + dc * dist
        if not _in_bounds(shape, rr, cc):
            break
        targets[(rr, cc)] = (k, dist)
    return targets


def _shared_targets(shape: tuple[int, int], inst_a: dict, inst_b: dict) -> list[dict]:
    ta = _target_positions(shape, inst_a)
    tb = _target_positions(shape, inst_b)
    shared = []
    for cell in sorted(set(ta) & set(tb)):
        ka, da = ta[cell]
        kb, db = tb[cell]
        if da != db:
            shared.append({"cell": cell, "a_k": ka, "b_k": kb, "a_dist": da, "b_dist": db})
    return shared


def _available_slots(shape: tuple[int, int], spike: tuple[int, int], direction: str, spacing: int) -> int:
    dr, dc = DIR4[direction]
    r, c = spike
    steps = 0
    while True:
        r += dr * spacing
        c += dc * spacing
        if not _in_bounds(shape, r, c):
            return steps
        steps += 1


def _place_pair_with_conflict(
    rng: np.random.Generator,
    shape: tuple[int, int],
    spec_a: dict,
    spec_b: dict,
    scenario: str,
) -> tuple[dict, dict, dict] | None:
    tips_a = list(_tip_candidates(spec_a, shape))
    tips_b = list(_tip_candidates(spec_b, shape))
    rng.shuffle(tips_a)
    rng.shuffle(tips_b)
    for tip_a in tips_a:
        inst_a = _instance_from_tip(spec_a, tip_a)
        for tip_b in tips_b:
            if scenario == "collinear":
                same_line = (
                    (spec_a["direction"] in ("L", "R") and tip_a[0] == tip_b[0])
                    or (spec_a["direction"] in ("U", "D") and tip_a[1] == tip_b[1])
                )
                if not same_line:
                    continue
            else:
                horiz = spec_a["direction"] in ("L", "R")
                other_horiz = spec_b["direction"] in ("L", "R")
                if horiz == other_horiz:
                    continue

            inst_b = _instance_from_tip(spec_b, tip_b)
            if not _bbox_gap_ok(inst_a["bbox"], inst_b["bbox"]):
                continue
            if _ray_hits_bbox(shape, inst_a["spike"], inst_a["direction"], inst_b["bbox"]):
                continue
            if _ray_hits_bbox(shape, inst_b["spike"], inst_b["direction"], inst_a["bbox"]):
                continue
            shared = _shared_targets(shape, inst_a, inst_b)
            if shared:
                choice = shared[int(rng.integers(0, len(shared)))]
                return inst_a, inst_b, choice
    return None


def _find_extra_placement(
    rng: np.random.Generator,
    shape: tuple[int, int],
    spec: dict,
    placed: list[dict],
) -> dict | None:
    tips = list(_tip_candidates(spec, shape))
    rng.shuffle(tips)
    for tip in tips:
        inst = _instance_from_tip(spec, tip)
        if any(not _bbox_gap_ok(inst["bbox"], other["bbox"]) for other in placed):
            continue
        if any(_ray_hits_bbox(shape, inst["spike"], inst["direction"], other["bbox"]) for other in placed):
            continue
        if any(_ray_hits_bbox(shape, other["spike"], other["direction"], inst["bbox"]) for other in placed):
            continue
        if any(_shared_targets(shape, inst, other) for other in placed):
            continue
        return inst
    return None


def _choose_count(
    rng: np.random.Generator,
    inst: dict,
    min_needed: int,
    allow_overflow_bias: bool,
    shape: tuple[int, int],
) -> int:
    lo = max(1, min_needed)
    hi = inst["capacity"]
    if lo > hi:
        raise ValueError("count lower bound exceeds capacity")
    slots = _available_slots(shape, inst["spike"], inst["direction"], inst["spacing"])
    counts = list(range(lo, hi + 1))
    if allow_overflow_bias:
        overflow = [n for n in counts if n > slots]
        if overflow and rng.random() < 0.45:
            return int(overflow[int(rng.integers(0, len(overflow)))])
    return int(counts[int(rng.integers(0, len(counts)))])


def _render_input(shape: tuple[int, int], placed: list[dict], counts: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros(shape, dtype=int)
    cups = []
    for inst, count in zip(placed, counts):
        for r, c in inst["border_cells"]:
            grid[r, c] = inst["border_color"]
        content_cells = inst["content_global_all"][:count]
        for r, c in content_cells:
            grid[r, c] = inst["content_color"]
        cups.append(
            {
                "tail_dir": inst["direction"],
                "spike": inst["spike"],
                "spacing": inst["spacing"],
                "content_color": inst["content_color"],
                "content_cells": content_cells,
            }
        )
    return grid, cups


def _validate_generated(shape: tuple[int, int], grid: np.ndarray, placed: list[dict], counts: list[int], scenario: str) -> bool:
    if max(grid.shape) > 30:
        return False
    if len(placed) != len(counts):
        return False

    if not any(count != inst["spacing"] for inst, count in zip(placed, counts)):
        return False

    expected_input, cups = _render_input(shape, placed, counts)
    if not np.array_equal(grid, expected_input):
        return False

    expected_out, stats = _settle_emissions(grid, cups)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return False
    if np.array_equal(actual_out, grid):
        return False
    if stats["placements"] == 0 or stats["static_delays"] != 0:
        return False

    analyzed = _analyze_all_cups(grid)
    if len(analyzed) != len(placed):
        return False
    analyzed_sorted = sorted(
        [(cup["spike"], cup["tail_dir"], cup["spacing"], len(cup["content_cells"])) for cup in analyzed]
    )
    placed_sorted = sorted(
        [(inst["spike"], inst["direction"], inst["spacing"], count) for inst, count in zip(placed, counts)]
    )
    if analyzed_sorted != placed_sorted:
        return False

    if scenario in {"collinear", "cross"}:
        if stats["cross_cup_delays"] == 0:
            return False
    else:
        if stats["delays"] != 0:
            return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(GRID_MIN, GRID_MAX + 1)),
        int(rng.integers(GRID_MIN, GRID_MAX + 1)),
    )
    scenario = rng.choice(["none", "collinear", "cross"], p=[0.28, 0.42, 0.30])
    if scenario == "none":
        n_cups = int(rng.choice([1, 2, 3], p=[0.15, 0.45, 0.40]))
    else:
        n_cups = int(rng.choice([2, 3], p=[0.55, 0.45]))

    colors = list(map(int, rng.choice(NON_BG_COLORS, size=2 * n_cups, replace=False)))
    directions: list[str] = []
    if scenario == "none":
        directions = [str(rng.choice(["U", "D", "L", "R"])) for _ in range(n_cups)]
    elif scenario == "collinear":
        if bool(rng.integers(0, 2)):
            directions = ["R", "L"]
        else:
            directions = ["D", "U"]
        while len(directions) < n_cups:
            directions.append(str(rng.choice(["U", "D", "L", "R"])))
    else:
        horiz = str(rng.choice(["L", "R"]))
        vert = str(rng.choice(["U", "D"]))
        directions = [horiz, vert]
        while len(directions) < n_cups:
            directions.append(str(rng.choice(["U", "D", "L", "R"])))

    specs = [
        _sample_cup_spec(rng, directions[i], colors[2 * i], colors[2 * i + 1])
        for i in range(n_cups)
    ]

    placed: list[dict] = []
    min_needed = [1] * n_cups

    if scenario == "none":
        for spec in specs:
            inst = _find_extra_placement(rng, shape, spec, placed)
            if inst is None:
                return None
            placed.append(inst)
    else:
        pair = _place_pair_with_conflict(rng, shape, specs[0], specs[1], scenario)
        if pair is None:
            return None
        inst_a, inst_b, conflict = pair
        placed.extend([inst_a, inst_b])
        min_needed[0] = int(conflict["a_k"])
        min_needed[1] = int(conflict["b_k"])
        for idx in range(2, n_cups):
            inst = _find_extra_placement(rng, shape, specs[idx], placed)
            if inst is None:
                return None
            placed.append(inst)

    counts = []
    for idx, inst in enumerate(placed):
        counts.append(_choose_count(rng, inst, min_needed[idx], allow_overflow_bias=True, shape=shape))

    if not any(count != inst["spacing"] for inst, count in zip(placed, counts)):
        for idx, inst in enumerate(placed):
            for candidate in range(min_needed[idx], inst["capacity"] + 1):
                if candidate != inst["spacing"]:
                    counts[idx] = candidate
                    break
        if not any(count != inst["spacing"] for inst, count in zip(placed, counts)):
            return None

    grid, _cups = _render_input(shape, placed, counts)
    if not _validate_generated(shape, grid, placed, counts, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid cup-ejection puzzle")
