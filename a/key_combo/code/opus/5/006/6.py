from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SQUARE_COLORS = (7, 2, 9, 1)
ALLOWED_SIDES = (1, 3, 5, 7, 9)
GEN_INPUT_SIDES = (3, 5, 7, 9)  # avoid singleton-square ambiguity in generated inputs
SIDE_CYCLE = (1, 9, 7, 5, 3)
DISPLACEMENTS = {
    7: (-2, 0),  # orange -> up
    2: (0, -2),  # red -> left
    9: (2, 0),   # maroon -> down
    1: (0, 1),   # blue -> right
}
GEN_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_solid_square(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    side_r = int(r1 - r0 + 1)
    side_c = int(c1 - c0 + 1)
    if side_r != side_c or side_r not in ALLOWED_SIDES or side_r % 2 == 0:
        return False
    if int(comp["area"]) != side_r * side_r:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _advance_side(side: int, steps: int) -> int:
    idx = SIDE_CYCLE.index(int(side))
    return int(SIDE_CYCLE[(idx + int(steps)) % len(SIDE_CYCLE)])


def _center_distance_key(center: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = center
    h, w = shape
    return abs(2 * int(r) - (h - 1)) + abs(2 * int(c) - (w - 1))


def _draw_square_clipped(grid: np.ndarray, center: tuple[int, int], side: int, color: int) -> None:
    h, w = grid.shape
    half = side // 2
    r0 = int(center[0] - half)
    c0 = int(center[1] - half)
    r1 = int(center[0] + half)
    c1 = int(center[1] + half)
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    if rr0 <= rr1 and cc0 <= cc1:
        grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)


def _identify_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    bg = _get_bg(grid)
    comps = _components4(grid, bg)
    squares: list[dict] = []
    markers: list[dict] = []
    for comp in comps:
        if _is_solid_square(grid, comp) and int(comp["area"]) > 1:
            squares.append(comp)
        elif int(comp["area"]) == 1:
            markers.append(comp)
        else:
            raise AssertionError("input must contain only solid odd squares and isolated singleton markers")
    assert squares, "expected at least one square"
    return squares, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count color-coded singleton markers, resize each colored square by that many cycle steps, then translate the resized squares by a color-specific offset.

    Input structure:
    - The background is black.
    - Foreground contains several separated monochrome solid odd-sided squares.
    - Additional foreground cells are isolated singleton markers whose colors are drawn from the square colors.

    Rule:
    - Detect every solid odd square and count, for each square color, how many isolated singleton markers use that color.
    - Advance the square side length by that many steps in the cycle 1 -> 9 -> 7 -> 5 -> 3 -> 1 while keeping the square center fixed.
    - Clear the canvas, sort squares by Manhattan distance from their original centers to the grid center from farthest to nearest, then draw the resized square after the color-coded translation: orange up 2, red left 2, maroon down 2, blue right 1.
    - Clip translated squares at the grid boundary and omit all singleton markers from the output.

    Key insight:
    - The singleton marker counts do not select colors; they encode how far each same-color square moves around the odd-side-length cycle before the translated redraw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    squares, markers = _identify_objects(g)

    counts = Counter(int(obj["color"]) for obj in markers)
    out = np.full_like(g, bg)
    parsed: list[dict] = []
    for sq in squares:
        color = int(sq["color"])
        assert color in DISPLACEMENTS
        r0, c0, r1, c1 = sq["bbox"]
        side = int(r1 - r0 + 1)
        steps = int(counts[color])
        assert 1 <= steps <= 4, "each square color must have 1 to 4 singleton markers"
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        parsed.append(
            {
                "color": color,
                "center": center,
                "side": side,
                "new_side": _advance_side(side, steps),
                "dist": _center_distance_key(center, g.shape),
            }
        )

    parsed.sort(key=lambda obj: (-int(obj["dist"]), int(obj["center"][0]), int(obj["center"][1])))
    for obj in parsed:
        dr, dc = DISPLACEMENTS[int(obj["color"])]
        new_center = (int(obj["center"][0] + dr), int(obj["center"][1] + dc))
        _draw_square_clipped(out, new_center, int(obj["new_side"]), int(obj["color"]))
    return out


def _bbox_from_center(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    half = side // 2
    r, c = center
    return r - half, c - half, r + half, c + half


def _clip_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = box
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h - 1, int(r1))
    cc1 = min(w - 1, int(c1))
    if rr0 > rr1 or cc0 > cc1:
        return None
    return rr0, cc0, rr1, cc1


def _boxes_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return not (
        int(box_a[2]) < int(box_b[0])
        or int(box_b[2]) < int(box_a[0])
        or int(box_a[3]) < int(box_b[1])
        or int(box_b[3]) < int(box_a[1])
    )


def _overlap_area(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    if not _boxes_overlap(box_a, box_b):
        return 0
    r0 = max(int(box_a[0]), int(box_b[0]))
    c0 = max(int(box_a[1]), int(box_b[1]))
    r1 = min(int(box_a[2]), int(box_b[2]))
    c1 = min(int(box_a[3]), int(box_b[3]))
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _separated_with_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        int(box_a[2]) + gap < int(box_b[0])
        or int(box_b[2]) + gap < int(box_a[0])
        or int(box_a[3]) + gap < int(box_b[1])
        or int(box_b[3]) + gap < int(box_a[1])
    )


def _visible_output_box(spec: dict, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    dr, dc = DISPLACEMENTS[int(spec["color"])]
    full = _bbox_from_center((int(spec["center"][0] + dr), int(spec["center"][1] + dc)), int(spec["new_side"]))
    return _clip_box(full, shape)


def _output_crops(spec: dict, shape: tuple[int, int]) -> bool:
    dr, dc = DISPLACEMENTS[int(spec["color"])]
    full = _bbox_from_center((int(spec["center"][0] + dr), int(spec["center"][1] + dc)), int(spec["new_side"]))
    clipped = _clip_box(full, shape)
    if clipped is None:
        return True
    return clipped != full


def _render_from_specs(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    order = sorted(specs, key=lambda obj: (-int(obj["dist"]), int(obj["center"][0]), int(obj["center"][1])))
    for spec in order:
        dr, dc = DISPLACEMENTS[int(spec["color"])]
        center = (int(spec["center"][0] + dr), int(spec["center"][1] + dc))
        _draw_square_clipped(out, center, int(spec["new_side"]), int(spec["color"]))
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _place_markers(inp: np.ndarray, specs: list[dict]) -> bool:
    h, w = inp.shape
    occupied = inp != BG
    blocked = _dilate8(occupied)
    todo = [int(spec["color"]) for spec in specs for _ in range(int(spec["count"]))]
    random.shuffle(todo)
    for color in todo:
        candidates = np.argwhere(~occupied & ~blocked)
        if candidates.size == 0:
            return False
        idx = random.randrange(len(candidates))
        r, c = map(int, candidates[idx])
        inp[r, c] = color
        single = np.zeros((h, w), dtype=bool)
        single[r, c] = True
        occupied[r, c] = True
        blocked |= _dilate8(single)
        blocked[r, c] = True
    return True


def _sample_pair_for_same_new_side() -> tuple[tuple[int, int], tuple[int, int]]:
    options: dict[int, list[tuple[int, int]]] = {side: [] for side in SIDE_CYCLE}
    for side in GEN_INPUT_SIDES:
        for count in range(1, 5):
            options[_advance_side(side, count)].append((side, count))
    target = random.choice(list(options))
    combos = random.sample(options[target], 2)
    return combos[0], combos[1]


def _choose_specs() -> tuple[list[dict], tuple[int, int] | None, int | None]:
    count = 4 if random.random() < 0.65 else 3
    colors = random.sample(list(SQUARE_COLORS), count)
    want_overlap = count >= 3 and random.random() < 0.45
    overlap_pair = None
    specs: list[dict] = []

    if want_overlap:
        pair_colors = random.sample(colors, 2)
        combo_a, combo_b = _sample_pair_for_same_new_side()
        assigned = {pair_colors[0]: combo_a, pair_colors[1]: combo_b}
    else:
        assigned = {}

    for color in colors:
        if color in assigned:
            side, marker_count = assigned[color]
        else:
            side = random.choice(GEN_INPUT_SIDES)
            marker_count = random.randint(1, 4)
        specs.append({"color": int(color), "side": int(side), "count": int(marker_count)})

    if len({int(spec["count"]) for spec in specs}) == 1:
        tweak = random.randrange(len(specs))
        current = int(specs[tweak]["count"])
        specs[tweak]["count"] = 1 + (current % 4)

    if max(int(spec["count"]) for spec in specs) == 1:
        tweak = random.randrange(len(specs))
        specs[tweak]["count"] = random.randint(2, 4)

    if len({int(spec["side"]) for spec in specs}) == 1 and len(specs) >= 2 and random.random() < 0.7:
        tweak = random.randrange(len(specs))
        alternatives = [side for side in GEN_INPUT_SIDES if side != int(specs[tweak]["side"])]
        specs[tweak]["side"] = random.choice(alternatives)

    for spec in specs:
        spec["new_side"] = _advance_side(int(spec["side"]), int(spec["count"]))

    if want_overlap:
        pair_indices = [idx for idx, spec in enumerate(specs) if int(spec["color"]) in pair_colors]
        overlap_pair = (pair_indices[0], pair_indices[1])

    want_crop = random.random() < 0.35
    crop_index = random.randrange(len(specs)) if want_crop else None
    return specs, overlap_pair, crop_index


def _candidate_centers(spec: dict, shape: tuple[int, int], require_crop: bool) -> list[tuple[int, int]]:
    h, w = shape
    half = int(spec["side"]) // 2
    centers = [(r, c) for r in range(half, h - half) for c in range(half, w - half)]
    random.shuffle(centers)
    if not require_crop:
        return centers
    return [center for center in centers if _output_crops({**spec, "center": center}, shape)]


def _place_specs(
    specs: list[dict],
    shape: tuple[int, int],
    overlap_pair: tuple[int, int] | None,
    crop_index: int | None,
) -> bool:
    placed_input: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    placed_output: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    used_dists: set[int] = set()

    def can_place(idx: int, center: tuple[int, int]) -> bool:
        trial = dict(specs[idx])
        trial["center"] = center
        dist = _center_distance_key(center, shape)
        if dist in used_dists:
            return False
        input_box = _bbox_from_center(center, int(trial["side"]))
        output_box = _visible_output_box(trial, shape)
        for j, other_box in enumerate(placed_input):
            if other_box is None:
                continue
            if not _separated_with_gap(input_box, other_box, gap=1):
                return False
            visible_other = placed_output[j]
            if output_box is None or visible_other is None:
                continue
            if _boxes_overlap(output_box, visible_other):
                allowed = overlap_pair is not None and {idx, j} == set(overlap_pair)
                if not allowed:
                    return False
                if _overlap_area(output_box, visible_other) > max(9, min(int(trial["new_side"]) ** 2, int(specs[j]["new_side"]) ** 2) // 2):
                    return False
        return True

    def commit(idx: int, center: tuple[int, int]) -> None:
        specs[idx]["center"] = center
        specs[idx]["dist"] = _center_distance_key(center, shape)
        placed_input[idx] = _bbox_from_center(center, int(specs[idx]["side"]))
        placed_output[idx] = _visible_output_box(specs[idx], shape)
        used_dists.add(int(specs[idx]["dist"]))

    special: list[int] = []
    if overlap_pair is not None:
        special.extend(list(overlap_pair))
    if crop_index is not None and crop_index not in special:
        special.append(crop_index)

    if overlap_pair is not None:
        i, j = overlap_pair
        centers_i = _candidate_centers(specs[i], shape, require_crop=(crop_index == i))
        centers_j = _candidate_centers(specs[j], shape, require_crop=(crop_index == j))
        trial_budget = min(200, len(centers_i) * max(1, min(20, len(centers_j))))
        for _ in range(trial_budget):
            if not centers_i or not centers_j:
                break
            ci = random.choice(centers_i)
            cj = random.choice(centers_j)
            if ci == cj:
                continue
            if not can_place(i, ci):
                continue
            temp_i = dict(specs[i], center=ci)
            input_i = _bbox_from_center(ci, int(temp_i["side"]))
            out_i = _visible_output_box(temp_i, shape)
            dist_i = _center_distance_key(ci, shape)

            if _center_distance_key(cj, shape) == dist_i:
                continue
            input_j = _bbox_from_center(cj, int(specs[j]["side"]))
            if not _separated_with_gap(input_i, input_j, gap=1):
                continue
            temp_j = dict(specs[j], center=cj)
            out_j = _visible_output_box(temp_j, shape)
            if out_i is None or out_j is None or not _boxes_overlap(out_i, out_j):
                continue
            area = _overlap_area(out_i, out_j)
            if not (1 <= area <= max(9, int(specs[i]["new_side"]))):
                continue
            commit(i, ci)
            if can_place(j, cj):
                commit(j, cj)
                break
            used_dists.remove(int(specs[i]["dist"]))
            placed_input[i] = None
            placed_output[i] = None
            specs[i].pop("center", None)
            specs[i].pop("dist", None)
        else:
            return False

    order = sorted(
        [idx for idx in range(len(specs)) if placed_input[idx] is None],
        key=lambda idx: max(int(specs[idx]["side"]), int(specs[idx]["new_side"])),
        reverse=True,
    )
    for idx in order:
        centers = _candidate_centers(specs[idx], shape, require_crop=(crop_index == idx))
        for center in centers:
            if can_place(idx, center):
                commit(idx, center)
                break
        else:
            return False
    return True


def _validate_scene(
    inp: np.ndarray,
    specs: list[dict],
    overlap_pair: tuple[int, int] | None,
    crop_index: int | None,
) -> bool:
    shape = inp.shape
    if len({int(spec["dist"]) for spec in specs}) != len(specs):
        return False
    if len({int(spec["count"]) for spec in specs}) < 2:
        return False
    if max(int(spec["count"]) for spec in specs) < 2:
        return False

    visible_overlaps: list[tuple[int, int]] = []
    crops = 0
    for i, spec_i in enumerate(specs):
        crops += int(_output_crops(spec_i, shape))
        box_i = _visible_output_box(spec_i, shape)
        for j in range(i + 1, len(specs)):
            box_j = _visible_output_box(specs[j], shape)
            if box_i is not None and box_j is not None and _boxes_overlap(box_i, box_j):
                visible_overlaps.append((i, j))

    if overlap_pair is None and visible_overlaps:
        return False
    if overlap_pair is not None:
        if len(visible_overlaps) != 1:
            return False
        if set(visible_overlaps[0]) != set(overlap_pair):
            return False
    if crop_index is None and crops != 0:
        return False
    if crop_index is not None and crops < 1:
        return False

    comps = _components4(inp, BG)
    for comp in comps:
        if comp["area"] == 1:
            r, c = comp["coords"][0]
            for nr in range(max(0, r - 1), min(inp.shape[0], r + 2)):
                for nc in range(max(0, c - 1), min(inp.shape[1], c + 2)):
                    if (nr, nc) != (r, c) and int(inp[nr, nc]) != BG:
                        return False
    expected = _render_from_specs(specs, shape)
    return bool(np.array_equal(main(inp), expected))


def _gen_attempt() -> np.ndarray | None:
    specs, overlap_pair, crop_index = _choose_specs()
    total_fg = sum(int(spec["side"]) ** 2 for spec in specs) + sum(int(spec["count"]) for spec in specs)
    size_options = [(h, w) for h in range(25, 30) for w in range(25, 30) if h * w > 2 * total_fg + 30]
    if not size_options:
        return None
    shape = random.choice(size_options)

    if not _place_specs(specs, shape, overlap_pair, crop_index):
        return None

    inp = np.full(shape, BG, dtype=int)
    for spec in specs:
        _draw_square_clipped(inp, tuple(spec["center"]), int(spec["side"]), int(spec["color"]))
    if not _place_markers(inp, specs):
        return None
    if _get_bg(inp) != BG:
        return None
    if np.array_equal(inp, _render_from_specs(specs, shape)):
        return None
    if not _validate_scene(inp, specs, overlap_pair, crop_index):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
