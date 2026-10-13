from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

bbox = _ARC_UTILS.bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


BG = 0
FRAME_COLOR = 3
ABSORBER_COLOR = 5
MOVER_COLORS = [1, 2, 4, 6, 7, 8, 9]

ANCHOR_SHAPES = [
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (2, 1)},
]

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    if random.random() < 0.5:
        max_c = max(c for _, c in out)
        out = {(r, max_c - c) for r, c in out}
    if random.random() < 0.5:
        max_r = max(r for r, _ in out)
        out = {(max_r - r, c) for r, c in out}
    return _normalize(out)


def _patch_from_cells(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    cells = _normalize(cells)
    height = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    patch = np.zeros((height, width), dtype=int)
    for r, c in cells:
        patch[r, c] = int(color)
    return patch


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _frame_bbox(anchor_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, anchor_bbox)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    top = int(r0 - height // 2)
    left = int(c0 - width // 2)
    bottom = int(top + 2 * height - 1)
    right = int(left + 2 * width - 1)
    return top, left, bottom, right


def _frame_pixels(frame_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    top, left, bottom, right = map(int, frame_bbox)
    pixels: set[tuple[int, int]] = set()
    for c in range(left, right + 1):
        pixels.add((top, c))
        pixels.add((bottom, c))
    for r in range(top, bottom + 1):
        pixels.add((r, left))
        pixels.add((r, right))
    return pixels


def _touches_or_overlaps(coords: np.ndarray, target: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if (rr, cc) in target:
            return True
        for dr, dc in NEIGHBORS4:
            if (rr + dr, cc + dc) in target:
                return True
    return False


def _direction_to_absorber(
    comp_bbox: tuple[int, int, int, int],
    absorber_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], int]:
    r0, c0, r1, c1 = map(int, comp_bbox)
    ar0, ac0, ar1, ac1 = map(int, absorber_bbox)
    candidates: list[tuple[int, tuple[int, int]]] = []

    if _bbox_overlap(r0, r1, ar0, ar1):
        if c1 < ac0:
            candidates.append((int(ac0 - c1 - 1), (0, 1)))
        if ac1 < c0:
            candidates.append((int(c0 - ac1 - 1), (0, -1)))
    if _bbox_overlap(c0, c1, ac0, ac1):
        if r1 < ar0:
            candidates.append((int(ar0 - r1 - 1), (1, 0)))
        if ar1 < r0:
            candidates.append((int(r0 - ar1 - 1), (-1, 0)))

    candidates = [item for item in candidates if item[0] >= 0]
    assert candidates, "each mover must have at least one straight slide direction to the absorber"
    best_dist = min(dist for dist, _ in candidates)
    best = [item for item in candidates if item[0] == best_dist]
    assert len(best) == 1, "shortest slide direction must be unique"
    return best[0][1], int(best[0][0])


def _solve(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected foreground components"

    absorbers = [comp for comp in comps if int(comp["color"]) == ABSORBER_COLOR]
    movers_and_anchor = [comp for comp in comps if int(comp["color"]) != ABSORBER_COLOR]
    assert len(absorbers) == 1, "expected exactly one gray absorber"
    assert 4 <= len(movers_and_anchor) <= 6, "expected 4-6 non-absorber components"

    absorber = absorbers[0]
    absorber_bbox = tuple(map(int, absorber["bbox"]))
    ar0, ac0, ar1, ac1 = absorber_bbox
    assert int(absorber["area"]) == (ar1 - ar0 + 1) * (ac1 - ac0 + 1), "gray absorber must be a solid rectangle"

    max_area = max(int(comp["area"]) for comp in movers_and_anchor)
    anchors = [comp for comp in movers_and_anchor if int(comp["area"]) == max_area]
    assert len(anchors) == 1, "largest non-absorber component must be unique"
    anchor = anchors[0]
    mover_comps = [comp for comp in movers_and_anchor if comp is not anchor]

    frame_bbox = _frame_bbox(tuple(map(int, anchor["bbox"])))
    top, left, bottom, right = frame_bbox
    assert 0 <= top < g.shape[0] and 0 <= left < g.shape[1]
    assert bottom < g.shape[0] and right < g.shape[1], "green frame must stay on the grid"
    frame = _frame_pixels(frame_bbox)
    absorber_pixels = {tuple(map(int, rc)) for rc in absorber["coords"]}
    assert frame.isdisjoint(absorber_pixels), "green frame must not overlap the absorber"

    out = np.zeros_like(g)
    out[absorber["coords"][:, 0], absorber["coords"][:, 1]] = ABSORBER_COLOR
    for r, c in frame:
        out[int(r), int(c)] = FRAME_COLOR
    out[anchor["coords"][:, 0], anchor["coords"][:, 1]] = int(anchor["color"])

    ordered_movers = []
    for comp in mover_comps:
        direction, distance = _direction_to_absorber(tuple(map(int, comp["bbox"])), absorber_bbox)
        ordered_movers.append(
            (
                int(distance),
                tuple(map(int, comp["bbox"][:2])),
                comp,
                direction,
            )
        )
    ordered_movers.sort(key=lambda item: (item[0], item[1][0], item[1][1]))

    occupied = set(frame) | absorber_pixels | {tuple(map(int, rc)) for rc in anchor["coords"]}
    trace = {
        "anchor_color": int(anchor["color"]),
        "anchor_area": int(anchor["area"]),
        "frame_bbox": frame_bbox,
        "absorber_bbox": absorber_bbox,
        "movers": [],
    }

    for distance, _key, comp, direction in ordered_movers:
        coords = np.array(comp["coords"], dtype=int)
        color = int(comp["color"])
        dr, dc = map(int, direction)
        while True:
            next_coords = coords + np.array([dr, dc], dtype=int)
            assert np.all(next_coords[:, 0] >= 0) and np.all(next_coords[:, 0] < g.shape[0])
            assert np.all(next_coords[:, 1] >= 0) and np.all(next_coords[:, 1] < g.shape[1])

            if _touches_or_overlaps(next_coords, frame):
                coords = next_coords
                final_color = FRAME_COLOR
                status = "frame"
                break
            if _touches_or_overlaps(next_coords, absorber_pixels):
                coords = next_coords
                final_color = ABSORBER_COLOR
                status = "absorber"
                break

            next_set = {tuple(map(int, rc)) for rc in next_coords}
            assert next_set.isdisjoint(occupied), "mover path collided with an earlier obstacle before reaching a stop rule"
            coords = next_coords

        out[coords[:, 0], coords[:, 1]] = int(final_color)
        occupied |= {tuple(map(int, rc)) for rc in coords}
        trace["movers"].append(
            {
                "input_color": color,
                "distance": int(distance),
                "direction": (int(dr), int(dc)),
                "status": status,
                "final_color": int(final_color),
                "final_bbox": _bbox_from_coords(coords),
            }
        )

    if with_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique largest component, draw its doubled green frame, then slide the other components toward the gray absorber.

    Input structure:
    - A black background contains one solid gray rectangle and 4-6 other monochrome 4-connected foreground components.
    - The non-gray components have distinct colors, one has strictly largest area, and there is enough empty space around it for a doubled green frame.
    - Other components are positioned so each has a unique shortest straight slide direction toward the gray absorber.

    Rule:
    - Identify the gray rectangle as the absorber and choose the unique largest non-gray component as the anchor.
    - Draw a hollow green frame whose height and width are twice the anchor bbox size, centered on that bbox, while keeping the anchor unchanged inside it.
    - Process every other component in increasing absorber distance order, sliding it one step at a time along its unique nearest cardinal direction.
    - If the next moved position would touch or overlap the green frame, stop there and recolor the component green; otherwise, if it would touch or overlap the gray absorber, stop there and recolor it gray. Stopped components then become later obstacles.

    Key insight:
    - The largest component matters twice: its area selects the anchor, and the anchor's doubled frame becomes a real physical barrier that changes which sliding components end up green versus gray.
    """
    return _solve(grid, with_trace=False)


def _row_slots(absorber_r0: int, intercept_count: int) -> list[int]:
    if intercept_count == 1:
        return [absorber_r0, absorber_r0 + 1, absorber_r0 + 2]
    if intercept_count == 2:
        return [absorber_r0, absorber_r0 + 1, absorber_r0 + 2]
    raise ValueError("intercept_count must be 1 or 2")


def _split_nonnegative(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _rotate_grid(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, k=int(turns)).copy()


def _build_input_instance() -> np.ndarray:
    height = random.randint(15, 21)
    width = random.randint(22, 28)
    grid = np.zeros((height, width), dtype=int)

    absorber_w = random.randint(4, 5)
    absorber_r0 = random.randint(6, height - 4)
    absorber_c0 = width - absorber_w
    grid[absorber_r0 : absorber_r0 + 3, absorber_c0 : absorber_c0 + absorber_w] = ABSORBER_COLOR

    intercept_count = random.choice([1, 2])
    anchor_cells = _transform_cells(set(random.choice(ANCHOR_SHAPES)))
    anchor_patch = _patch_from_cells(anchor_cells, 9)
    anchor_h, anchor_w = anchor_patch.shape
    assert anchor_h == 3, "generator expects 3-row anchor shapes"
    anchor_r0 = absorber_r0 - (5 if intercept_count == 1 else 4)

    frame_w = 2 * anchor_w
    mover_widths = [random.randint(2, 4) for _ in range(3)]
    min_needed_left = sum(mover_widths) + 2
    frame_left_min = min_needed_left + 1
    frame_left_max = absorber_c0 - frame_w - 3
    if frame_left_min > frame_left_max:
        raise ValueError("not enough horizontal space for movers and frame")
    frame_left = random.randint(frame_left_min, frame_left_max)
    anchor_c0 = frame_left + anchor_w // 2

    palette = random.sample(MOVER_COLORS, 4)
    anchor_color = int(palette[0])
    mover_colors = [int(color) for color in palette[1:]]
    anchor_patch[anchor_patch != 0] = int(anchor_color)
    paste(grid, anchor_patch, anchor_r0, anchor_c0, transparent=0)

    usable_cols = frame_left - 2
    spare = usable_cols - (sum(mover_widths) + 2)
    if spare < 0:
        raise ValueError("not enough mover columns before the frame")
    slack = _split_nonnegative(spare, 4)
    cursor = 1 + slack[0]
    mover_starts = []
    for idx, width_i in enumerate(mover_widths):
        mover_starts.append(cursor)
        cursor += width_i
        if idx < 2:
            cursor += 1 + slack[idx + 1]

    mover_rows = _row_slots(absorber_r0, intercept_count)
    paired = list(zip(mover_rows, mover_widths, mover_starts, mover_colors))
    random.shuffle(paired)
    for row, width_i, col, color in paired:
        patch = np.full((1, width_i), int(color), dtype=int)
        paste(grid, patch, int(row), int(col), transparent=0)

    turns = random.randint(0, 3)
    return _rotate_grid(grid, turns)


def _validate_input(grid: np.ndarray) -> bool:
    try:
        out, trace = _solve(grid, with_trace=True)
    except Exception:
        return False

    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    non_absorbers = [comp for comp in comps if int(comp["color"]) != ABSORBER_COLOR]
    if len(non_absorbers) != 4:
        return False
    if len({int(comp["color"]) for comp in non_absorbers}) != 4:
        return False

    areas = sorted(int(comp["area"]) for comp in non_absorbers)
    if areas[-1] <= areas[-2]:
        return False

    for i, comp_a in enumerate(non_absorbers):
        a0, b0, a1, b1 = map(int, comp_a["bbox"])
        for comp_b in non_absorbers[i + 1 :]:
            c0, d0, c1, d1 = map(int, comp_b["bbox"])
            if _bbox_overlap(a0, a1, c0, c1):
                return False
            if _bbox_overlap(b0, b1, d0, d1):
                return False

    statuses = [item["status"] for item in trace["movers"]]
    if not ("frame" in statuses and "absorber" in statuses):
        return False

    unique_dirs = {tuple(item["direction"]) for item in trace["movers"]}
    if len(unique_dirs) != 1:
        return False

    if np.count_nonzero(out == FRAME_COLOR) <= len(_frame_pixels(trace["frame_bbox"])):
        return False
    if np.count_nonzero(out == ABSORBER_COLOR) <= (trace["absorber_bbox"][2] - trace["absorber_bbox"][0] + 1) * (
        trace["absorber_bbox"][3] - trace["absorber_bbox"][1] + 1
    ):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _build_input_instance()
        except ValueError:
            continue
        if _validate_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid new ARC input")
