from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components


MARKER_COLOR = Color.BLUE
BACKGROUND_COLOR = Color.BLACK
FRAME_PALETTE = [Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
FRAME_CACHE: dict[tuple[int, int], np.ndarray] = {}
GEN_RETRIES = 400


def _marker_side(grid: np.ndarray) -> str:
    candidates = []
    if np.all(grid[0, :] == MARKER_COLOR):
        candidates.append("top")
    if np.all(grid[-1, :] == MARKER_COLOR):
        candidates.append("bottom")
    if np.all(grid[:, 0] == MARKER_COLOR):
        candidates.append("left")
    if np.all(grid[:, -1] == MARKER_COLOR):
        candidates.append("right")
    assert len(candidates) == 1, "expected exactly one full-edge blue marker line"
    return candidates[0]


def _to_bottom(grid: np.ndarray, side: str) -> np.ndarray:
    turns = {"bottom": 0, "left": 1, "top": 2, "right": 3}[side]
    return np.rot90(grid, turns)


def _from_bottom(grid: np.ndarray, side: str) -> np.ndarray:
    turns = {"bottom": 0, "left": 3, "top": 2, "right": 1}[side]
    return np.rot90(grid, turns)


def _frame_coords(h: int, w: int) -> np.ndarray:
    key = (int(h), int(w))
    if key not in FRAME_CACHE:
        coords = []
        for c in range(w):
            coords.append((0, c))
            coords.append((h - 1, c))
        for r in range(1, h - 1):
            coords.append((r, 0))
            coords.append((r, w - 1))
        FRAME_CACHE[key] = np.array(coords, dtype=int)
    return FRAME_CACHE[key]


def _draw_frame(grid: np.ndarray, r: int, c: int, h: int, w: int, color: int) -> None:
    grid[r, c : c + w] = color
    grid[r + h - 1, c : c + w] = color
    grid[r : r + h, c] = color
    grid[r : r + h, c + w - 1] = color


def _non_marker_background(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid[grid != MARKER_COLOR].ravel()))
    assert counts, "grid must contain non-marker cells"
    bg = int(counts.most_common(1)[0][0])
    assert bg != MARKER_COLOR, "marker cannot be the background"
    return bg


def _component_to_rect(grid: np.ndarray, comp: dict, bg: int) -> dict:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 3 and w >= 3, "every foreground object must be a hollow rectangle at least 3x3"

    patch = crop_bbox(grid, (r0, c0, r1, c1))
    frame_mask = np.zeros((h, w), dtype=bool)
    frame_mask[0, :] = True
    frame_mask[-1, :] = True
    frame_mask[:, 0] = True
    frame_mask[:, -1] = True

    assert np.all(patch[frame_mask] == color), "frame border must be monochromatic"
    assert np.all(patch[~frame_mask] == bg), "rectangle interior must be hollow"

    return {
        "r": r0,
        "c": c0,
        "h": h,
        "w": w,
        "color": color,
        "area": h * w,
        "coords": _frame_coords(h, w),
    }


def _extract_rects(grid: np.ndarray, bg: int) -> list[dict]:
    colors = sorted(set(map(int, grid.ravel())) - {bg, MARKER_COLOR})
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=colors)
    rects = [_component_to_rect(grid, comp, bg) for comp in comps]
    assert len(rects) >= 2, "expected multiple hollow rectangles"
    areas = [rect["area"] for rect in rects]
    assert len(set(areas)) == len(areas), "outer areas must be unique"
    return rects


def _drop_one(rect: dict, occupied: np.ndarray) -> tuple[int, int]:
    max_shift = occupied.shape[0]
    for shift in range(max_shift):
        blockers = set()
        for dr, dc in rect["coords"]:
            nr = rect["r"] + shift + dr + 1
            nc = rect["c"] + dc
            if nr >= occupied.shape[0]:
                raise AssertionError("object should stop on the marker before leaving the canvas")
            if occupied[nr, nc] != 0:
                blockers.add(int(occupied[nr, nc]))
        if blockers:
            assert len(blockers) == 1, "each falling rectangle must have exactly one direct supporter"
            return shift, next(iter(blockers))
    raise AssertionError("rectangle never found support")


def _simulate_specs(shape: tuple[int, int], rects: list[dict], order: list[int]) -> list[dict]:
    occupied = np.zeros(shape, dtype=int)
    occupied[-1, :] = -1
    settled = []

    for rect_index in order:
        src = rects[rect_index]
        shift, support = _drop_one(src, occupied)
        final_rect = dict(src)
        final_rect["r"] = int(src["r"] + shift)
        final_rect["support"] = int(support)
        final_rect["occ_id"] = int(rect_index + 1)

        rr = final_rect["r"] + final_rect["coords"][:, 0]
        cc = final_rect["c"] + final_rect["coords"][:, 1]
        assert np.all(occupied[rr, cc] == 0), "settled frames must not overlap"
        occupied[rr, cc] = final_rect["occ_id"]
        settled.append(final_rect)

    return settled


def _render(shape: tuple[int, int], bg: int, settled: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[-1, :] = MARKER_COLOR

    support_colors = {-1: MARKER_COLOR}
    for rect in settled:
        _draw_frame(out, rect["r"], rect["c"], rect["h"], rect["w"], rect["color"])
        support_colors[rect["occ_id"]] = rect["color"]

    for rect in settled:
        fill_color = support_colors[rect["support"]]
        out[rect["r"] + 1 : rect["r"] + rect["h"] - 1, rect["c"] + 1 : rect["c"] + rect["w"] - 1] = fill_color

    return out


def _area_order(rects: list[dict]) -> list[int]:
    return sorted(range(len(rects)), key=lambda i: rects[i]["area"], reverse=True)


def _distance_order(rects: list[dict]) -> list[int]:
    return sorted(range(len(rects)), key=lambda i: rects[i]["r"] + rects[i]["h"] - 1, reverse=True)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _chain_dims(rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(200):
        first = (rng.randint(4, 5), rng.randint(7, 11))
        dims = [first]
        used_areas = {first[0] * first[1]}
        while len(dims) < 3:
            prev_h, prev_w = dims[-1]
            prev_area = prev_h * prev_w
            candidates = []
            for h in range(3, 6):
                for w in range(3, prev_w):
                    area = h * w
                    if area < prev_area and area not in used_areas:
                        candidates.append((h, w))
            if not candidates:
                break
            choice = rng.choice(candidates)
            dims.append(choice)
            used_areas.add(choice[0] * choice[1])
        if len(dims) == 3 and sum(h for h, _ in dims) <= 13:
            return dims
    raise RuntimeError("failed to sample chain dimensions")


def _independent_dim(rng: random.Random, forbidden_areas: set[int]) -> tuple[int, int]:
    candidates = []
    for h in range(3, 6):
        for w in range(3, 7):
            if h * w not in forbidden_areas:
                candidates.append((h, w))
    assert candidates, "need at least one independent rectangle size"
    return rng.choice(candidates)


def _attempt_canonical(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    chain_dims = _chain_dims(rng)
    use_independent = rng.random() < 0.45

    all_dims = list(chain_dims)
    used_areas = {h * w for h, w in chain_dims}
    if use_independent:
        all_dims.append(_independent_dim(rng, used_areas))

    sum_chain_h = sum(h for h, _ in chain_dims)
    h_min = 2 * sum_chain_h + 4
    if h_min > 30:
        return None
    height = rng.randint(h_min, 30)

    largest_chain_w = chain_dims[0][1]
    widest_other = all_dims[-1][1] if use_independent else 0
    w_min = max(16, largest_chain_w + widest_other + 7 if use_independent else largest_chain_w + 6)
    if w_min > 30:
        return None
    width = rng.randint(w_min, 30)

    chain_starts = []
    start0 = rng.randint(1, width - largest_chain_w - 1)
    chain_starts.append(start0)
    for i in range(1, 3):
        prev_start = chain_starts[-1]
        prev_w = chain_dims[i - 1][1]
        h, w = chain_dims[i]
        choices = []
        for delta in (-1, 0, 1):
            start = prev_start + delta
            if not (1 <= start <= width - w - 1):
                continue
            overlap = _interval_overlap(prev_start, prev_start + prev_w - 1, start, start + w - 1)
            if overlap >= 2:
                choices.append(start)
        if not choices:
            return None
        chain_starts.append(rng.choice(choices))

    chain_left = min(chain_starts[i] for i in range(3))
    chain_right = max(chain_starts[i] + chain_dims[i][1] - 1 for i in range(3))

    independent_start = None
    if use_independent:
        _, ind_w = all_dims[-1]
        left_max = chain_left - ind_w - 2
        right_min = chain_right + 3
        options = []
        if left_max >= 1:
            options.append(("left", rng.randint(1, left_max)))
        if right_min <= width - ind_w - 1:
            options.append(("right", rng.randint(right_min, width - ind_w - 1)))
        if not options:
            return None
        independent_start = rng.choice(options)[1]

    final_rows = []
    cursor = height - 1
    for h, _w in chain_dims:
        final_rows.append(cursor - h)
        cursor -= h

    initial_rows = [None, None, None]
    row_order = [0, 1, 2]
    rng.shuffle(row_order)
    if row_order[-1] == 0:
        return None
    cursor = 0
    for idx in row_order:
        initial_rows[idx] = cursor
        cursor += chain_dims[idx][0] + 1
    if cursor >= final_rows[-1]:
        return None

    rect_specs = []
    colors = rng.sample(FRAME_PALETTE, len(all_dims))

    for i, ((h, w), c) in enumerate(zip(chain_dims, chain_starts)):
        rect_specs.append(
            {
                "r": initial_rows[i],
                "c": c,
                "h": h,
                "w": w,
                "color": colors[i],
                "area": h * w,
                "coords": _frame_coords(h, w),
            }
        )

    if use_independent:
        h, w = all_dims[-1]
        max_init_r = max(0, final_rows[-1] - h - 1)
        rect_specs.append(
            {
                "r": rng.randint(0, max_init_r),
                "c": independent_start,
                "h": h,
                "w": w,
                "color": colors[-1],
                "area": h * w,
                "coords": _frame_coords(h, w),
            }
        )

    rect_specs.sort(key=lambda rect: rect["area"], reverse=True)

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    grid[-1, :] = MARKER_COLOR
    for rect in rect_specs:
        _draw_frame(grid, rect["r"], rect["c"], rect["h"], rect["w"], rect["color"])

    area_order = _area_order(rect_specs)
    settled = _simulate_specs(grid.shape, rect_specs, area_order)
    out = _render(grid.shape, BACKGROUND_COLOR, settled)

    supports = [rect["support"] for rect in settled]
    if not any(support == -1 for support in supports):
        return None
    if not any(support != -1 for support in supports):
        return None

    if not any(
        _interval_overlap(
            rect_specs[i]["c"],
            rect_specs[i]["c"] + rect_specs[i]["w"] - 1,
            rect_specs[j]["c"],
            rect_specs[j]["c"] + rect_specs[j]["w"] - 1,
        )
        > 0
        for i in range(len(rect_specs))
        for j in range(i + 1, len(rect_specs))
    ):
        return None

    wrong_order = _distance_order(rect_specs)
    if wrong_order == area_order:
        return None
    wrong_out = _render(grid.shape, BACKGROUND_COLOR, _simulate_specs(grid.shape, rect_specs, wrong_order))
    if np.array_equal(out, wrong_out):
        return None

    for rect in rect_specs:
        patch = crop_bbox(grid, (rect["r"], rect["c"], rect["r"] + rect["h"] - 1, rect["c"] + rect["w"] - 1))
        if np.any(patch[1:-1, 1:-1] != BACKGROUND_COLOR):
            return None

    return grid, out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop larger hollow rectangles first toward the blue edge, then fill each interior with its direct supporter's color.

    Input structure:
    - One canvas edge is a full blue marker line that indicates the fall direction.
    - The remaining foreground objects are separated hollow axis-aligned rectangles with distinct frame colors.
    - Each rectangle has a unique outer bbox area, and some rectangles overlap in their projection onto the fall axis.

    Rule:
    - Process rectangles in descending outer bbox area order.
    - Each rectangle translates straight toward the blue edge until one more step would overlap the marker line or a previously settled rectangle, then it stops there.
    - After every rectangle settles, fill each rectangle's interior with blue if the marker line directly supports it, otherwise fill it with the frame color of the unique rectangle directly supporting it; keep all frame pixels and the marker line unchanged.

    Key insight:
    - Fall order is determined by rectangle area rather than initial distance to the marked edge, and the interior fill records the resulting support chain explicitly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    side = _marker_side(g)
    work = _to_bottom(g, side)
    assert np.all(work[-1, :] == MARKER_COLOR), "canonicalized grid must have the blue marker on the bottom edge"
    bg = _non_marker_background(work)

    rects = _extract_rects(work, bg)
    settled = _simulate_specs(work.shape, rects, _area_order(rects))
    out = _render(work.shape, bg, settled)
    return _from_bottom(out, side)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        attempt = _attempt_canonical(rng)
        if attempt is None:
            continue

        canonical_inp, canonical_out = attempt
        side = rng.choice(["bottom", "top", "left", "right"])
        rotated_inp = _from_bottom(canonical_inp, side)
        rotated_out = _from_bottom(canonical_out, side)

        if not np.array_equal(main(rotated_inp), rotated_out):
            continue
        return rotated_inp

    raise RuntimeError("failed to generate a valid large-first-drop input")
