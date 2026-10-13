from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
ANCHOR = 8
STRIPE = 4
OBJECT_COLORS = (1, 2, 3, 5, 6, 7)
GRID_SIDE_RANGE = (13, 20)
DISTANCE_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _shape_coords(shape: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(shape > 0)]


def _translate_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _direction_candidates(
    object_coords: list[tuple[int, int]],
    anchor_bbox: tuple[int, int, int, int],
) -> list[dict[str, int]]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    row_to_cols: dict[int, list[int]] = defaultdict(list)
    col_to_rows: dict[int, list[int]] = defaultdict(list)
    for r, c in object_coords:
        row_to_cols[int(r)].append(int(c))
        col_to_rows[int(c)].append(int(r))

    candidates: list[dict[str, int]] = []

    overlap_rows = [r for r in row_to_cols if ar0 <= r <= ar1]
    if overlap_rows:
        right_phase = max(max(row_to_cols[r]) for r in overlap_rows)
        right_d = ac0 - right_phase - 1
        if right_d > 0:
            candidates.append({"direction": "right", "distance": int(right_d), "phase": int(right_phase)})

        left_phase = min(min(row_to_cols[r]) for r in overlap_rows)
        left_d = left_phase - ac1 - 1
        if left_d > 0:
            candidates.append({"direction": "left", "distance": int(left_d), "phase": int(left_phase)})

    overlap_cols = [c for c in col_to_rows if ac0 <= c <= ac1]
    if overlap_cols:
        down_phase = max(max(col_to_rows[c]) for c in overlap_cols)
        down_d = ar0 - down_phase - 1
        if down_d > 0:
            candidates.append({"direction": "down", "distance": int(down_d), "phase": int(down_phase)})

        up_phase = min(min(col_to_rows[c]) for c in overlap_cols)
        up_d = up_phase - ar1 - 1
        if up_d > 0:
            candidates.append({"direction": "up", "distance": int(up_d), "phase": int(up_phase)})

    anchor_cells = {(ar0, ac0), (ar0, ac1), (ar1, ac0), (ar1, ac1)}
    filtered: list[dict[str, int]] = []
    for cand in candidates:
        dr = dc = 0
        if cand["direction"] == "right":
            dc = cand["distance"]
        elif cand["direction"] == "left":
            dc = -cand["distance"]
        elif cand["direction"] == "down":
            dr = cand["distance"]
        else:
            dr = -cand["distance"]

        moved = _translate_coords(object_coords, dr, dc)
        moved_set = set(moved)
        if moved_set & anchor_cells:
            continue

        touching = False
        for r, c in moved:
            if (r - 1, c) in anchor_cells or (r + 1, c) in anchor_cells or (r, c - 1) in anchor_cells or (r, c + 1) in anchor_cells:
                touching = True
                break
        if touching:
            filtered.append(cand)

    return filtered


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    anchor_comps = _find_components(g, ANCHOR)
    assert len(anchor_comps) == 1
    anchor_coords = anchor_comps[0]
    assert len(anchor_coords) == 4
    anchor_bbox = _bbox(anchor_coords)
    ar0, ac0, ar1, ac1 = anchor_bbox
    assert (ar1 - ar0, ac1 - ac0) == (1, 1)
    assert {(r, c) for r, c in anchor_coords} == {(ar0, ac0), (ar0, ac1), (ar1, ac0), (ar1, ac1)}

    colors = sorted({int(v) for v in g.flatten()} - {BG, ANCHOR})
    assert len(colors) == 1
    obj_color = colors[0]
    assert obj_color in OBJECT_COLORS
    object_comps = _find_components(g, obj_color)
    assert len(object_comps) == 1
    object_coords = object_comps[0]
    assert 3 <= len(object_coords) <= 6

    candidates = _direction_candidates(object_coords, anchor_bbox)
    assert candidates
    best_d = min(cand["distance"] for cand in candidates)
    best = [cand for cand in candidates if cand["distance"] == best_d]
    assert len(best) == 1
    chosen = best[0]

    direction = str(chosen["direction"])
    distance = int(chosen["distance"])
    phase = int(chosen["phase"])

    dr = dc = 0
    if direction == "right":
        dc = distance
        stripe_orientation = "vertical"
        stripe_positions = [c for c in range(phase, g.shape[1], distance)]
        stripe_positions.extend([c for c in range(phase - distance, -1, -distance)])
    elif direction == "left":
        dc = -distance
        stripe_orientation = "vertical"
        stripe_positions = [c for c in range(phase, g.shape[1], distance)]
        stripe_positions.extend([c for c in range(phase - distance, -1, -distance)])
    elif direction == "down":
        dr = distance
        stripe_orientation = "horizontal"
        stripe_positions = [r for r in range(phase, g.shape[0], distance)]
        stripe_positions.extend([r for r in range(phase - distance, -1, -distance)])
    else:
        dr = -distance
        stripe_orientation = "horizontal"
        stripe_positions = [r for r in range(phase, g.shape[0], distance)]
        stripe_positions.extend([r for r in range(phase - distance, -1, -distance)])

    stripe_positions = sorted(set(int(v) for v in stripe_positions))

    return {
        "grid": g,
        "anchor_bbox": anchor_bbox,
        "anchor_coords": anchor_coords,
        "object_color": obj_color,
        "object_coords": object_coords,
        "direction": direction,
        "distance": distance,
        "phase": phase,
        "dr": dr,
        "dc": dc,
        "stripe_orientation": stripe_orientation,
        "stripe_positions": stripe_positions,
    }


def _render_output(info: dict[str, object], *, orientation: str | None = None, distance: int | None = None, phase: int | None = None) -> np.ndarray:
    g = np.asarray(info["grid"], dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)

    ar0, ac0, ar1, ac1 = info["anchor_bbox"]  # type: ignore[index]
    out[ar0 : ar1 + 1, ac0 : ac1 + 1] = ANCHOR

    used_orientation = str(info["stripe_orientation"] if orientation is None else orientation)
    used_distance = int(info["distance"] if distance is None else distance)
    used_phase = int(info["phase"] if phase is None else phase)

    if used_orientation == "vertical":
        for c in range(used_phase, w, used_distance):
            if 0 <= c < w:
                out[:, c] = STRIPE
        for c in range(used_phase - used_distance, -1, -used_distance):
            if 0 <= c < w:
                out[:, c] = STRIPE
    else:
        for r in range(used_phase, h, used_distance):
            if 0 <= r < h:
                out[r, :] = STRIPE
        for r in range(used_phase - used_distance, -1, -used_distance):
            if 0 <= r < h:
                out[r, :] = STRIPE

    moved = _translate_coords(info["object_coords"], int(info["dr"]), int(info["dc"]))  # type: ignore[arg-type]
    for r, c in moved:
        out[r, c] = int(info["object_color"])
    out[ar0 : ar1 + 1, ac0 : ac1 + 1] = ANCHOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide the small object into the cyan anchor and turn the slide distance into perpendicular yellow stripes.

    Input structure:
    - A black background contains exactly one cyan 2x2 anchor square and one other monochrome 4-connected object.
    - The movable object has 3-6 pixels, uses a non-black non-cyan non-yellow color, and lies in a row band or column band that permits straight sliding toward the anchor.
    - The empty gap between the object and the anchor along that slide direction is at least one cell, with valid task instances using distances 2-6.

    Rule:
    - Determine the unique cardinal slide direction that brings the object into 4-neighbor contact with the anchor using the shortest straight translation.
    - Let D be that translation distance in cells; erase the original object, redraw the object translated by D, and keep the anchor unchanged.
    - Draw full-grid yellow stripes perpendicular to the slide direction with period D, using the object's original anchor-facing row or column as the stripe phase, while never overwriting anchor pixels or the moved object.

    Key insight:
    - The same spatial gap controls two different effects at once: it is both the object's motion distance and the stripe spacing, while the stripe orientation is orthogonal to the motion.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    return _render_output(info)


_BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
]


def _canonical(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    coords = np.argwhere(mask > 0)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    cropped = mask[r0 : int(coords[:, 0].max()) + 1, c0 : int(coords[:, 1].max()) + 1]
    return tuple(tuple(int(v) for v in row) for row in cropped.tolist())


def _shape_variants(shape: np.ndarray) -> list[np.ndarray]:
    seen = set()
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(shape, k)
        for arr in (rot, np.fliplr(rot)):
            key = _canonical(arr)
            if key in seen:
                continue
            seen.add(key)
            coords = np.argwhere(arr > 0)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            variants.append(arr[r0 : int(coords[:, 0].max()) + 1, c0 : int(coords[:, 1].max()) + 1].copy())
    return variants


_SHAPE_VARIANTS = [variant for shape in _BASE_SHAPES for variant in _shape_variants(shape)]


def _global_facing_phase(coords: list[tuple[int, int]], direction: str) -> int:
    if direction == "right":
        return max(c for _, c in coords)
    if direction == "left":
        return min(c for _, c in coords)
    if direction == "down":
        return max(r for r, _ in coords)
    return min(r for r, _ in coords)


def _parallel_orientation(info: dict[str, object]) -> str:
    return "horizontal" if str(info["stripe_orientation"]) == "vertical" else "vertical"


def _count_stripes(info: dict[str, object]) -> int:
    return len(info["stripe_positions"])  # type: ignore[arg-type]


def _build_grid(
    shape: np.ndarray,
    color: int,
    object_top_left: tuple[int, int],
    anchor_top_left: tuple[int, int],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    h, w = grid_shape
    out = np.zeros((h, w), dtype=int)
    or0, oc0 = object_top_left
    ar0, ac0 = anchor_top_left
    for r, c in _shape_coords(shape):
        out[or0 + r, oc0 + c] = int(color)
    out[ar0 : ar0 + 2, ac0 : ac0 + 2] = ANCHOR
    return out


def _candidate_anchor_positions(
    shape: np.ndarray,
    grid_shape: tuple[int, int],
    object_top_left: tuple[int, int],
    desired_direction: str,
    desired_distance: int,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    obj_coords = _translate_coords(_shape_coords(shape), object_top_left[0], object_top_left[1])
    obj_set = set(obj_coords)

    anchors: list[tuple[int, int]] = []
    for ar0 in range(h - 1):
        for ac0 in range(w - 1):
            anchor_cells = {(ar0, ac0), (ar0, ac0 + 1), (ar0 + 1, ac0), (ar0 + 1, ac0 + 1)}
            if obj_set & anchor_cells:
                continue
            candidates = _direction_candidates(obj_coords, (ar0, ac0, ar0 + 1, ac0 + 1))
            best_d = min((cand["distance"] for cand in candidates), default=None)
            if best_d is None:
                continue
            best = [cand for cand in candidates if cand["distance"] == best_d]
            if len(best) != 1:
                continue
            cand = best[0]
            if cand["direction"] == desired_direction and cand["distance"] == desired_distance:
                anchors.append((ar0, ac0))
    return anchors


def _passes_semantic_checks(grid: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    if not (DISTANCE_RANGE[0] <= int(info["distance"]) <= DISTANCE_RANGE[1]):
        return False
    if _count_stripes(info) < 2:
        return False

    object_coords = info["object_coords"]  # type: ignore[assignment]
    direction = str(info["direction"])
    phase = int(info["phase"])
    global_phase = _global_facing_phase(object_coords, direction)

    expected = _render_output(info)
    if np.array_equal(expected, grid):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    anchor_bbox = info["anchor_bbox"]  # type: ignore[assignment]
    ar0, ac0, ar1, ac1 = anchor_bbox
    anchor_phase = ac0 if str(info["stripe_orientation"]) == "vertical" else ar0
    if np.array_equal(expected, _render_output(info, phase=anchor_phase)):
        return False

    if np.array_equal(expected, _render_output(info, orientation=_parallel_orientation(info))):
        return False

    alt_period = int(info["distance"]) + 1 if int(info["distance"]) < DISTANCE_RANGE[1] else int(info["distance"]) - 1
    if alt_period > 0 and np.array_equal(expected, _render_output(info, distance=alt_period)):
        return False

    if phase == global_phase and len(object_coords) <= 4:
        return False

    if str(info["stripe_orientation"]) == "vertical":
        if not any(c == phase for _r, c in object_coords):
            return False
    else:
        if not any(r == phase for r, _c in object_coords):
            return False

    if np.count_nonzero(expected == STRIPE) == 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    shape = random.choice(_SHAPE_VARIANTS)
    sh, sw = shape.shape
    if sh >= h - 2 or sw >= w - 2:
        return None

    color = random.choice(OBJECT_COLORS)
    direction = random.choice(["right", "left", "down", "up"])
    distance = random.randint(*DISTANCE_RANGE)

    row_margin = max(1, sh // 2)
    col_margin = max(1, sw // 2)
    object_rows = range(1, h - sh - 1)
    object_cols = range(1, w - sw - 1)
    if not object_rows or not object_cols:
        return None

    for _ in range(40):
        or0 = random.choice(list(object_rows))
        oc0 = random.choice(list(object_cols))
        anchors = _candidate_anchor_positions(shape, (h, w), (or0, oc0), direction, distance)
        random.shuffle(anchors)
        for ar0, ac0 in anchors:
            grid = _build_grid(shape, color, (or0, oc0), (ar0, ac0), (h, w))
            if _passes_semantic_checks(grid):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid sliding-stripe puzzle input")
