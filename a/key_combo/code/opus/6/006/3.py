from __future__ import annotations

import random

import numpy as np


WALL = 5
BG = 0
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 23)
GRID_W_RANGE = (18, 23)
MAX_ATTEMPTS = 600


def _side_from_axis(axis2: int, width: int) -> int:
    return -1 if int(axis2) < int(width) else 1


def _distance_to_target_wall(axis2: int, width: int) -> int:
    if _side_from_axis(axis2, width) == -1:
        return int(axis2)
    return int(2 * (width - 1) - axis2)


def _paired_subset(coords: np.ndarray, axis2: int) -> np.ndarray:
    pts = {(int(r), int(c)) for r, c in coords}
    kept: list[tuple[int, int]] = []
    for r, c in pts:
        mc = int(axis2) - c
        if (r, mc) in pts:
            kept.append((r, c))
    kept.sort()
    return np.array(kept, dtype=int) if kept else np.zeros((0, 2), dtype=int)


def _best_axis_info(coords: np.ndarray, width: int) -> dict:
    assert len(coords) > 0
    mean2 = int(round(2.0 * float(coords[:, 1].mean())))
    best = None
    rankings: list[tuple[int, int, int]] = []
    for axis2 in range(0, 2 * width - 1):
        kept = _paired_subset(coords, axis2)
        score = int(len(kept))
        center_bias = -abs(int(axis2) - mean2)
        rankings.append((score, center_bias, int(axis2)))
        key = (score, center_bias, -int(axis2))
        if best is None or key > best["key"]:
            best = {
                "axis2": int(axis2),
                "clean": kept,
                "score": score,
                "key": key,
            }
    rankings.sort(reverse=True)
    top_score = rankings[0][0]
    second_score = rankings[1][0] if len(rankings) > 1 else -1
    best["unique_top_score"] = bool(top_score > second_score)
    del best["key"]
    return best


def _adjacent_to_occupied(coords: np.ndarray, occupied: np.ndarray) -> tuple[bool, bool]:
    h, w = occupied.shape
    hits_wall = False
    hits_other = False
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not bool(occupied[nr, nc]):
                continue
            if nc in (0, w - 1):
                hits_wall = True
            else:
                hits_other = True
    return hits_wall or hits_other, hits_other


def _slide_until_stop(coords: np.ndarray, direction: int, occupied: np.ndarray) -> tuple[np.ndarray, str]:
    h, w = occupied.shape
    cur = coords.copy()
    stop_reason = "none"
    while True:
        cand = cur + np.array([0, int(direction)], dtype=int)
        if int(cand[:, 1].min()) < 0 or int(cand[:, 1].max()) >= w:
            break
        if np.any(occupied[cand[:, 0], cand[:, 1]]):
            break
        cur = cand
        touches, hits_other = _adjacent_to_occupied(cur, occupied)
        if touches:
            stop_reason = "collision" if hits_other else "wall"
            break
    if stop_reason == "none":
        touches, hits_other = _adjacent_to_occupied(cur, occupied)
        if touches:
            stop_reason = "collision" if hits_other else "wall"
    return cur, stop_reason


def _analyze_layers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, WALL)]
    layers: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert len(coords) > 0
        info = _best_axis_info(coords, w)
        clean = info["clean"]
        assert len(clean) > 0
        cols = coords[:, 1]
        raw_bbox_center2 = int(cols.min() + cols.max())
        raw_mean2 = 2.0 * float(cols.mean())
        layers.append(
            {
                "color": int(color),
                "raw": coords,
                "axis2": int(info["axis2"]),
                "clean": clean,
                "score": int(info["score"]),
                "unique_top_score": bool(info["unique_top_score"]),
                "direction": _side_from_axis(int(info["axis2"]), w),
                "distance2": _distance_to_target_wall(int(info["axis2"]), w),
                "raw_bbox_center2": raw_bbox_center2,
                "raw_mean2": raw_mean2,
            }
        )
    return layers


def _transform_with_trace(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and w >= 4
    assert np.all(g[:, 0] == WALL) and np.all(g[:, w - 1] == WALL)

    layers = _analyze_layers(g)
    order = sorted(layers, key=lambda item: (item["distance2"], item["axis2"], item["color"]))

    out = np.full_like(g, BG)
    out[:, 0] = WALL
    out[:, w - 1] = WALL
    occupied = out != BG

    trace: list[dict] = []
    for item in order:
        placed, stop_reason = _slide_until_stop(item["clean"], item["direction"], occupied)
        out[placed[:, 0], placed[:, 1]] = int(item["color"])
        occupied[placed[:, 0], placed[:, 1]] = True
        trace.append(
            {
                "color": int(item["color"]),
                "axis2": int(item["axis2"]),
                "direction": int(item["direction"]),
                "distance2": int(item["distance2"]),
                "clean": item["clean"],
                "placed": placed,
                "stop_reason": stop_reason,
            }
        )
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove asymmetric per-color noise, then slide each cleaned color layer toward the side implied by its vertical symmetry axis.

    Input structure:
    - Black background fills the interior, and grey wall columns occupy the full left and right borders.
    - The interior contains 3 or 4 non-black, non-grey colors, each treated as one color layer.
    - Every color layer is almost vertically symmetric but includes 1 to 3 noisy pixels that break the symmetry.

    Rule:
    - For each color independently, search all integer and half-integer vertical axes and keep only the pixels that have mirrored same-color partners under the best axis.
    - Recompute that cleaned layer's axis position from the kept pixels; layers with axis < width / 2 slide left, otherwise slide right.
    - Process layers from smaller axis-to-target-wall distance to larger distance, rigidly translating each layer horizontally.
    - A moving layer stops as soon as its next legal position is 4-neighbor adjacent to a wall or to an already settled layer; original noisy pixels disappear.

    Key insight:
    - The symmetry cleanup is not cosmetic: it decides which pixels survive and therefore which side of the board each color must slide toward.
    """
    out, _trace = _transform_with_trace(grid)
    return out


def _centered_interval(width: int, length: int) -> tuple[int, int]:
    mid = (width - 1) / 2.0
    start = int(round(mid - (length - 1) / 2.0))
    return start, start + int(length) - 1


def _make_symmetric_mask(axis2: int) -> np.ndarray | None:
    odd_axis = (int(axis2) % 2) == 0
    width_choices = [3, 5, 7] if odd_axis else [2, 4, 6]
    width = random.choice(width_choices)
    height = random.randint(3, 5)
    allowed = [k for k in range(1 if odd_axis else 2, width + 1, 2)]
    for _ in range(80):
        lengths = [random.choice(allowed) for _ in range(height)]
        if max(lengths) == min(lengths):
            continue
        if any(abs(a - b) > 2 for a, b in zip(lengths, lengths[1:])):
            continue
        area = int(sum(lengths))
        if not (5 <= area <= 15):
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, length in enumerate(lengths):
            c0, c1 = _centered_interval(width, length)
            mask[r, c0 : c1 + 1] = True
        if np.all(mask[:, width // 2]) and width >= 5 and random.random() < 0.5:
            continue
        return mask
    return None


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([int(r0), int(c0)], dtype=int)


def _chebyshev_clear(candidate: np.ndarray, occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in candidate:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 1 <= cc < w - 1):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in occupied:
                    return False
    return True


def _layer_spec_sequence(count: int) -> list[int]:
    if count == 3:
        return random.choice([[-1, -1, 1], [-1, 1, 1]])
    choices = [
        [-1, -1, 1, 1],
        [-1, -1, -1, 1],
        [-1, 1, 1, 1],
    ]
    return random.choice(choices)


def _axis_candidates(width: int, direction: int, near_mid: bool) -> list[int]:
    mid2 = int(width)
    vals = []
    for axis2 in range(2, 2 * width - 3):
        if direction == -1 and axis2 >= mid2:
            continue
        if direction == 1 and axis2 < mid2:
            continue
        dist_mid = abs(axis2 - mid2)
        if near_mid and dist_mid > 3:
            continue
        if (not near_mid) and dist_mid < 3:
            continue
        vals.append(int(axis2))
    random.shuffle(vals)
    return vals


def _pick_axes(width: int, directions: list[int]) -> list[int] | None:
    chosen: list[int] = []
    used = set()
    used_dists = set()
    flip_idx = random.randrange(len(directions))
    for i, direction in enumerate(directions):
        candidates = _axis_candidates(width, direction, near_mid=(i == flip_idx))
        for axis2 in candidates:
            dist2 = _distance_to_target_wall(axis2, width)
            if axis2 in used or dist2 in used_dists:
                continue
            chosen.append(axis2)
            used.add(axis2)
            used_dists.add(dist2)
            break
        else:
            return None
    if abs(chosen[flip_idx] - width) > 3:
        return None
    return chosen


def _place_clean_layers(h: int, w: int, colors: list[int], directions: list[int], axes: list[int]) -> list[dict] | None:
    occupied: set[tuple[int, int]] = set()
    layers: list[dict] = []
    counts = {d: directions.count(d) for d in (-1, 1)}
    collision_dir = -1 if counts[-1] >= 2 else 1
    overlap_anchor: tuple[int, int] | None = None

    for idx, (color, direction, axis2) in enumerate(zip(colors, directions, axes)):
        mask = _make_symmetric_mask(axis2)
        if mask is None:
            return None
        mh, mw = mask.shape
        c0 = (int(axis2) - (mw - 1)) // 2
        if not (1 <= c0 and c0 + mw - 1 <= w - 2):
            return None

        rows = list(range(1, h - mh - 1))
        random.shuffle(rows)
        placed = None
        for r0 in rows:
            coords = _coords_from_mask(mask, r0, c0)
            if not _chebyshev_clear(coords, occupied, h, w):
                continue
            if direction == collision_dir:
                if overlap_anchor is None:
                    overlap_anchor = (int(r0), int(r0 + mh - 1))
                else:
                    a0, a1 = overlap_anchor
                    b0, b1 = int(r0), int(r0 + mh - 1)
                    if min(a1, b1) < max(a0, b0):
                        continue
            placed = (r0, coords, mask)
            break
        if placed is None:
            return None
        r0, coords, mask = placed
        for rc in coords:
            occupied.add((int(rc[0]), int(rc[1])))
        layers.append(
            {
                "color": int(color),
                "direction": int(direction),
                "axis2": int(axis2),
                "mask": mask,
                "clean": coords,
                "r0": int(r0),
                "c0": int(c0),
            }
        )
    return layers


def _mirror_cell(cell: tuple[int, int], axis2: int) -> tuple[int, int]:
    r, c = cell
    return int(r), int(axis2) - int(c)


def _candidate_noise_cells(
    clean_set: set[tuple[int, int]],
    axis2: int,
    h: int,
    w: int,
    global_used: set[tuple[int, int]],
    prefer_far_side: int | None = None,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cell = (int(r), int(c))
            if cell in clean_set or cell in global_used:
                continue
            if _mirror_cell(cell, axis2) in clean_set:
                continue
            if prefer_far_side is not None:
                if prefer_far_side == -1 and c >= axis2 / 2.0:
                    continue
                if prefer_far_side == 1 and c <= axis2 / 2.0:
                    continue
            out.append(cell)
    random.shuffle(out)
    return out


def _perturb_layer(
    layer: dict,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    require_flip: bool,
) -> np.ndarray | None:
    clean = layer["clean"]
    axis2 = int(layer["axis2"])
    direction = int(layer["direction"])
    clean_set = {(int(r), int(c)) for r, c in clean}
    opposite_side = 1 if direction == -1 else -1
    removable = [cell for cell in clean_set if _mirror_cell(cell, axis2) in clean_set and _mirror_cell(cell, axis2) != cell]
    far_candidates = _candidate_noise_cells(clean_set, axis2, h, w, blocked, prefer_far_side=opposite_side)
    any_candidates = _candidate_noise_cells(clean_set, axis2, h, w, blocked)

    for _ in range(140):
        raw = set(clean_set)
        ops = random.randint(1, 3)
        did_add = False
        did_remove = False
        local_used = set(raw)
        if require_flip and far_candidates:
            cell = far_candidates[0]
            raw.add(cell)
            local_used.add(cell)
            did_add = True
        for _step in range(ops):
            choices = []
            if removable:
                choices.append("remove")
            if any_candidates:
                choices.append("add")
            if not choices:
                break
            move = random.choice(choices)
            if move == "remove":
                cell = random.choice(removable)
                if cell in raw:
                    raw.remove(cell)
                    did_remove = True
            else:
                pool = far_candidates if require_flip and random.random() < 0.7 and far_candidates else any_candidates
                cell = random.choice(pool)
                if cell not in local_used:
                    raw.add(cell)
                    local_used.add(cell)
                    did_add = True
        if len(raw) <= 0:
            continue
        coords = np.array(sorted(raw), dtype=int)
        info = _best_axis_info(coords, w)
        if int(info["axis2"]) != axis2:
            continue
        clean2 = info["clean"]
        if len(clean2) < 5 or len(clean2) > 15:
            continue
        cols = coords[:, 1]
        bbox_center_side = _side_from_axis(int(cols.min() + cols.max()), w)
        if require_flip and bbox_center_side == direction:
            continue
        if not did_add and not did_remove:
            continue
        return coords
    return None


def _render_grid(h: int, w: int, raw_layers: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[:, 0] = WALL
    grid[:, w - 1] = WALL
    for layer in raw_layers:
        coords = layer["raw"]
        grid[coords[:, 0], coords[:, 1]] = int(layer["color"])
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if g.ndim != 2 or h > 30 or w > 30:
        return False
    if not (np.all(g[:, 0] == WALL) and np.all(g[:, w - 1] == WALL)):
        return False

    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, WALL)]
    if len(colors) not in (3, 4):
        return False

    layers = _analyze_layers(g)
    axes = [int(item["axis2"]) for item in layers]
    dists = [int(item["distance2"]) for item in layers]
    if len(set(axes)) != len(axes):
        return False
    if len(set(dists)) != len(dists):
        return False
    if not all(item["unique_top_score"] for item in layers):
        return False
    if set(item["direction"] for item in layers) != {-1, 1}:
        return False
    if not all(np.array_equal(item["clean"], _paired_subset(item["clean"], item["axis2"])) for item in layers):
        return False
    if not any(_side_from_axis(int(item["raw_bbox_center2"]), w) != item["direction"] for item in layers):
        return False

    out, trace = _transform_with_trace(g)
    if out.shape != g.shape:
        return False
    if np.array_equal(out, g):
        return False
    if not any(step["stop_reason"] == "collision" for step in trace):
        return False

    cleaned_only = np.full_like(g, BG)
    cleaned_only[:, 0] = WALL
    cleaned_only[:, w - 1] = WALL
    for item in layers:
        cleaned_only[item["clean"][:, 0], item["clean"][:, 1]] = int(item["color"])
    if np.array_equal(cleaned_only, out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    count = 3 if random.random() < 0.55 else 4
    colors = random.sample(COLOR_POOL, count)
    directions = _layer_spec_sequence(count)
    axes = _pick_axes(w, directions)
    if axes is None:
        return None

    layers = _place_clean_layers(h, w, colors, directions, axes)
    if layers is None:
        return None

    flip_idx = min(range(len(layers)), key=lambda i: abs(int(layers[i]["axis2"]) - w))
    reserved = {
        (int(r), int(c))
        for layer in layers
        for r, c in layer["clean"]
    }
    used: set[tuple[int, int]] = set()
    raw_layers: list[dict] = []
    for idx, layer in enumerate(layers):
        own_clean = {(int(r), int(c)) for r, c in layer["clean"]}
        blocked = used | (reserved - own_clean)
        raw = _perturb_layer(layer, h, w, blocked, require_flip=(idx == flip_idx))
        if raw is None:
            return None
        for r, c in raw:
            if (int(r), int(c)) in used:
                return None
            used.add((int(r), int(c)))
        raw_layers.append({**layer, "raw": raw})

    grid = _render_grid(h, w, raw_layers)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
