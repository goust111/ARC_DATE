from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
OBJECT_COLOR = 8
PALETTE = tuple(color for color in range(10) if color not in (BG, OBJECT_COLOR))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 20
GRID_MAX = 28
MAX_ATTEMPTS = 400


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _contiguous_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0, "expected a nonempty side run"
    lo = int(indices[0])
    hi = int(indices[-1])
    assert np.array_equal(indices, np.arange(lo, hi + 1)), "side run must be contiguous"
    return lo, hi


def _centered_span(lo: int, hi: int, length: int) -> tuple[int, int]:
    center2 = int(lo) + int(hi)
    new_lo = (center2 - (int(length) - 1)) // 2
    new_hi = (center2 + (int(length) - 1)) // 2
    return int(new_lo), int(new_hi)


def _layer_count(run_length: int) -> int:
    return max(0, (int(run_length) - 1) // 2)


def _read_cycle(grid: np.ndarray) -> list[int]:
    indicator = np.asarray(grid[-1], dtype=int)
    cols = np.flatnonzero(indicator != BG)
    assert 2 <= cols.size <= 4, "indicator row must contain 2-4 colored pixels"
    lo = int(cols[0])
    hi = int(cols[-1])
    assert np.array_equal(cols, np.arange(lo, hi + 1)), "indicator row colors must be adjacent"
    colors = [int(indicator[c]) for c in cols]
    assert all(color not in (BG, OBJECT_COLOR) for color in colors)
    return colors


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    search = np.asarray(grid[:-1], dtype=int)
    components = _component_cells(search == OBJECT_COLOR)
    objects: list[dict[str, object]] = []
    for cells in components:
        r0, c0, r1, c1 = _bbox(cells)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True

        top_lo, top_hi = _contiguous_span(np.flatnonzero(mask[0]))
        bottom_lo, bottom_hi = _contiguous_span(np.flatnonzero(mask[-1]))
        left_lo, left_hi = _contiguous_span(np.flatnonzero(mask[:, 0]))
        right_lo, right_hi = _contiguous_span(np.flatnonzero(mask[:, -1]))

        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "key": (c0, r0),
                "top": (c0 + top_lo, c0 + top_hi),
                "bottom": (c0 + bottom_lo, c0 + bottom_hi),
                "left": (r0 + left_lo, r0 + left_hi),
                "right": (r0 + right_lo, r0 + right_hi),
            }
        )
    objects.sort(key=lambda obj: obj["key"])
    return objects


def _claim_better(new: tuple[int, int, int, int, int, int, int], old: tuple[int, int, int, int, int, int, int]) -> bool:
    if new[0] != old[0]:
        return new[0] < old[0]
    if new[1] == old[1]:
        if new[4] != old[4]:
            return new[4] < old[4]
        return new[5] < old[5]
    if new[2] != old[2]:
        return new[2] < old[2]
    if new[3] != old[3]:
        return new[3] < old[3]
    return new[5] < old[5]


def _collect_claims(grid: np.ndarray) -> tuple[list[dict[str, object]], list[int], dict[tuple[int, int], list[tuple[int, int, int, int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cycle = _read_cycle(g)
    objects = _extract_objects(g)
    claims: dict[tuple[int, int], list[tuple[int, int, int, int, int, int, int]]] = {}

    def add_claim(r: int, c: int, layer: int, color: int, obj_id: int, obj_key: tuple[int, int], dir_rank: int, dir_order: int) -> None:
        if not (0 <= r < h - 1 and 0 <= c < w):
            return
        if int(g[r, c]) != BG:
            return
        claim = (int(layer), int(obj_id), int(obj_key[0]), int(obj_key[1]), int(dir_rank), int(dir_order), int(color))
        claims.setdefault((int(r), int(c)), []).append(claim)

    for obj_id, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        obj_key = obj["key"]
        top_c0, top_c1 = obj["top"]
        bottom_c0, bottom_c1 = obj["bottom"]
        left_r0, left_r1 = obj["left"]
        right_r0, right_r1 = obj["right"]

        k = 1
        while True:
            length = (top_c1 - top_c0 + 1) - 2 * k
            if length < 1:
                break
            lo, hi = _centered_span(top_c0, top_c1, length)
            for col in range(lo, hi + 1):
                add_claim(r0 - k, col, k, cycle[(k - 1) % len(cycle)], obj_id, obj_key, 0, 0)
            k += 1

        k = 1
        while True:
            length = (bottom_c1 - bottom_c0 + 1) - 2 * k
            if length < 1:
                break
            lo, hi = _centered_span(bottom_c0, bottom_c1, length)
            for col in range(lo, hi + 1):
                add_claim(r1 + k, col, k, cycle[(k - 1) % len(cycle)], obj_id, obj_key, 0, 1)
            k += 1

        k = 1
        while True:
            length = (left_r1 - left_r0 + 1) - 2 * k
            if length < 1:
                break
            lo, hi = _centered_span(left_r0, left_r1, length)
            for row in range(lo, hi + 1):
                add_claim(row, c0 - k, k, cycle[(k - 1) % len(cycle)], obj_id, obj_key, 1, 2)
            k += 1

        k = 1
        while True:
            length = (right_r1 - right_r0 + 1) - 2 * k
            if length < 1:
                break
            lo, hi = _centered_span(right_r0, right_r1, length)
            for row in range(lo, hi + 1):
                add_claim(row, c1 + k, k, cycle[(k - 1) % len(cycle)], obj_id, obj_key, 1, 3)
            k += 1

    return objects, cycle, claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each teal rounded object with shrinking halo bands whose layer colors follow the bottom cycle.

    Input structure:
    - The background is black and the bottom row is a color-indicator row with one contiguous 2-4 pixel sequence of non-black, non-teal colors.
    - Above that row there are 1-2 teal 4-connected rounded objects, each with contiguous top, bottom, left, and right side runs.
    - The objects are separated from each other and from the grid boundary so at least the first two shrinking layers are defined in-bounds.

    Rule:
    - Read the bottom-row non-background pixels from left to right as a color cycle.
    - For every teal object, extend each of its four side runs outward one layer at a time; layer k is offset by k cells and its run length is the original side-run length minus 2k, kept centered on that side.
    - Paint only background cells, color layer k with the k-th color of the cycle modulo its length, resolve overlaps by smaller layer first, then by horizontal-over-vertical inside one object, and by leftmost-then-upper bbox priority across different objects.
    - Keep the teal objects unchanged and clear the indicator row to black in the output.

    Key insight:
    - The geometry comes from the shrinking side-run extrapolation, but each geometric layer must be paired with the independently read bottom color cycle to produce the rainbow halo.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects, _cycle, claims = _collect_claims(g)
    assert 1 <= len(objects) <= 2, "expected 1-2 teal objects"

    out = g.copy()
    out[-1, :] = BG
    for (r, c), cell_claims in claims.items():
        best = cell_claims[0]
        for claim in cell_claims[1:]:
            if _claim_better(claim, best):
                best = claim
        out[r, c] = best[6]
    return out


def _rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for r in range(int(height)):
        edge = min(r, int(height) - 1 - r, int(radius))
        inset = int(radius) - edge
        mask[r, inset : int(width) - inset] = True
    return mask


def _make_shape(radius: int, height: int, width: int) -> dict[str, object]:
    top_run = int(width) - 2 * int(radius)
    side_run = int(height) - 2 * int(radius)
    return {
        "radius": int(radius),
        "height": int(height),
        "width": int(width),
        "mask": _rounded_mask(height, width, radius),
        "top_layers": _layer_count(top_run),
        "side_layers": _layer_count(side_run),
    }


def _shape_candidates(
    *,
    max_height: int,
    max_width: int,
    min_top_layers: int = 2,
    min_side_layers: int = 2,
    min_any_layers: int = 0,
    exact_radius: int | None = None,
    exact_height: int | None = None,
    exact_width: int | None = None,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for radius in range(1, 4):
        if exact_radius is not None and radius != exact_radius:
            continue
        min_height = 2 * radius + 5
        min_width = 2 * radius + 5
        for height in range(min_height, min(max_height, 15) + 1):
            if exact_height is not None and height != exact_height:
                continue
            side_layers = _layer_count(height - 2 * radius)
            if side_layers < min_side_layers:
                continue
            for width in range(min_width, min(max_width, 15) + 1):
                if exact_width is not None and width != exact_width:
                    continue
                top_layers = _layer_count(width - 2 * radius)
                if top_layers < min_top_layers:
                    continue
                if max(top_layers, side_layers) < min_any_layers:
                    continue
                candidates.append(_make_shape(radius, height, width))
    return candidates


def _sample_shape(**kwargs: object) -> dict[str, object]:
    candidates = _shape_candidates(**kwargs)
    if not candidates:
        raise ValueError("no valid rounded shape candidates")
    return random.choice(candidates)


def _paint_shape(grid: np.ndarray, shape: dict[str, object], top: int, left: int) -> None:
    mask = shape["mask"]
    rows, cols = np.nonzero(mask)
    grid[rows + int(top), cols + int(left)] = OBJECT_COLOR


def _build_grid(height: int, width: int, placements: list[tuple[dict[str, object], int, int]], cycle: list[int]) -> np.ndarray:
    grid = np.zeros((int(height), int(width)), dtype=int)
    for shape, top, left in placements:
        _paint_shape(grid, shape, int(top), int(left))
    start = random.randint(0, width - len(cycle))
    grid[-1, start : start + len(cycle)] = np.array(cycle, dtype=int)
    return grid


def _conflict_signature(
    claims: dict[tuple[int, int], list[tuple[int, int, int, int, int, int, int]]]
) -> tuple[int, int]:
    same_layer = 0
    diff_layer = 0
    for cell_claims in claims.values():
        obj_ids = {claim[1] for claim in cell_claims}
        if len(obj_ids) < 2:
            continue
        min_layer = min(claim[0] for claim in cell_claims)
        best_layer_objs = {claim[1] for claim in cell_claims if claim[0] == min_layer}
        if len(best_layer_objs) >= 2:
            same_layer += 1
        else:
            diff_layer += 1
    return same_layer, diff_layer


def _max_visible_layers(objects: list[dict[str, object]]) -> int:
    visible = 0
    for obj in objects:
        top = obj["top"]
        left = obj["left"]
        visible = max(visible, _layer_count(top[1] - top[0] + 1), _layer_count(left[1] - left[0] + 1))
    return visible


def _validate_generated(grid: np.ndarray, mode: str, branch: str | None) -> bool:
    try:
        objects, cycle, claims = _collect_claims(grid)
    except AssertionError:
        return False

    if len(objects) not in (1, 2):
        return False
    if len(set(cycle)) != len(cycle):
        return False
    if _max_visible_layers(objects) < len(cycle):
        return False

    used_colors = {claim[6] for cell_claims in claims.values() for claim in cell_claims}
    if not set(cycle).issubset(used_colors):
        return False

    h, w = grid.shape
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        if r0 < 2 or c0 < 2 or c1 > w - 3 or r1 > h - 4:
            return False
        if (obj["top"][1] - obj["top"][0] + 1) < 5:
            return False
        if (obj["left"][1] - obj["left"][0] + 1) < 5:
            return False

    same_layer, diff_layer = _conflict_signature(claims)
    if mode == "conflict":
        if branch == "same" and same_layer == 0:
            return False
        if branch == "diff" and diff_layer == 0:
            return False
    else:
        if same_layer or diff_layer:
            return False

    return True


def _single_mode(height: int, width: int, cycle_len: int) -> list[tuple[dict[str, object], int, int]]:
    shape = _sample_shape(
        max_height=height - 5,
        max_width=width - 4,
        min_top_layers=2,
        min_side_layers=2,
        min_any_layers=max(cycle_len, 3),
    )
    top = random.randint(2, height - int(shape["height"]) - 3)
    left = random.randint(2, width - int(shape["width"]) - 2)
    return [(shape, top, left)]


def _pick_horizontal_pair(
    height: int,
    width: int,
    gap: int,
    cycle_len: int,
    min_side_layers: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    max_height = min(height - 5, 13)
    max_width = min(width - 4, 12)
    rich = _shape_candidates(
        max_height=max_height,
        max_width=max_width,
        min_top_layers=2,
        min_side_layers=min_side_layers,
        min_any_layers=max(cycle_len, min_side_layers),
    )
    random.shuffle(rich)
    total_width_limit = width - gap - 4
    for shape1 in rich:
        remaining_width = min(max_width, total_width_limit - int(shape1["width"]))
        if remaining_width < 5:
            continue
        pool = _shape_candidates(
            max_height=max_height,
            max_width=remaining_width,
            min_top_layers=2,
            min_side_layers=min_side_layers,
            exact_radius=int(shape1["radius"]),
            exact_height=int(shape1["height"]),
        )
        pool = [shape2 for shape2 in pool if int(shape1["width"]) + int(shape2["width"]) <= total_width_limit]
        if pool:
            return shape1, random.choice(pool)
    return None


def _pick_vertical_pair(
    height: int,
    width: int,
    gap: int,
    cycle_len: int,
    min_top_layers: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    max_height = min(height - 5, 13)
    max_width = min(width - 4, 12)
    rich = _shape_candidates(
        max_height=max_height,
        max_width=max_width,
        min_top_layers=min_top_layers,
        min_side_layers=2,
        min_any_layers=max(cycle_len, min_top_layers),
    )
    random.shuffle(rich)
    total_height_limit = height - gap - 5
    for shape1 in rich:
        remaining_height = min(max_height, total_height_limit - int(shape1["height"]))
        if remaining_height < 5:
            continue
        pool = _shape_candidates(
            max_height=remaining_height,
            max_width=max_width,
            min_top_layers=min_top_layers,
            min_side_layers=2,
            exact_radius=int(shape1["radius"]),
            exact_width=int(shape1["width"]),
        )
        pool = [shape2 for shape2 in pool if int(shape1["height"]) + int(shape2["height"]) <= total_height_limit]
        if pool:
            return shape1, random.choice(pool)
    return None


def _separate_mode(height: int, width: int, cycle_len: int) -> list[tuple[dict[str, object], int, int]] | None:
    orientation = random.choice(("horizontal", "vertical"))
    gap = random.randint(8, 9)
    if orientation == "horizontal":
        pair = _pick_horizontal_pair(height, width, gap, cycle_len, min_side_layers=2)
        if pair is None:
            return None
        shape1, shape2 = pair
        total_width = int(shape1["width"]) + gap + int(shape2["width"])
        max_height = max(int(shape1["height"]), int(shape2["height"]))
        if total_width > width - 4 or max_height > height - 5:
            return None
        top = random.randint(2, height - max_height - 3)
        left = random.randint(2, width - total_width - 2)
        return [(shape1, top, left), (shape2, top, left + int(shape1["width"]) + gap)]

    pair = _pick_vertical_pair(height, width, gap, cycle_len, min_top_layers=2)
    if pair is None:
        return None
    shape1, shape2 = pair
    total_height = int(shape1["height"]) + gap + int(shape2["height"])
    max_width = max(int(shape1["width"]), int(shape2["width"]))
    if total_height > height - 5 or max_width > width - 4:
        return None
    top = random.randint(2, height - total_height - 3)
    left = random.randint(2, width - max_width - 2)
    return [(shape1, top, left), (shape2, top + int(shape1["height"]) + gap, left)]


def _conflict_mode(height: int, width: int, cycle_len: int) -> tuple[list[tuple[dict[str, object], int, int]] | None, str]:
    branch = random.choice(("same", "diff"))
    if branch == "same":
        k1, k2 = 4, 4
    else:
        k1, k2 = random.choice(((3, 4), (4, 3)))
    gap = k1 + k2 - 1
    orientation = random.choice(("horizontal", "vertical"))

    if orientation == "horizontal":
        pair = _pick_horizontal_pair(height, width, gap, cycle_len, min_side_layers=max(k1, k2))
        if pair is None:
            return None, branch
        shape1, shape2 = pair
        total_width = int(shape1["width"]) + gap + int(shape2["width"])
        if total_width > width - 4:
            return None, branch
        top = random.randint(2, height - int(shape1["height"]) - 3)
        left = random.randint(2, width - total_width - 2)
        placements = [(shape1, top, left), (shape2, top, left + int(shape1["width"]) + gap)]
        return placements, branch

    pair = _pick_vertical_pair(height, width, gap, cycle_len, min_top_layers=max(k1, k2))
    if pair is None:
        return None, branch
    shape1, shape2 = pair
    total_height = int(shape1["height"]) + gap + int(shape2["height"])
    if total_height > height - 5:
        return None, branch
    top = random.randint(2, height - total_height - 3)
    left = random.randint(2, width - int(shape1["width"]) - 2)
    placements = [(shape1, top, left), (shape2, top + int(shape1["height"]) + gap, left)]
    return placements, branch


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    cycle_len = random.choice((2, 3, 4))
    cycle = random.sample(PALETTE, cycle_len)

    roll = random.random()
    if roll < 0.30:
        mode = "single"
        branch = None
        placements = _single_mode(height, width, cycle_len)
    elif roll < 0.60:
        mode = "separate"
        branch = None
        placements = _separate_mode(height, width, cycle_len)
    else:
        mode = "conflict"
        placements, branch = _conflict_mode(height, width, cycle_len)

    if placements is None:
        return None

    grid = _build_grid(height, width, placements, cycle)
    if not _validate_generated(grid, mode, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid decreasing-halo cyclic-band input")
