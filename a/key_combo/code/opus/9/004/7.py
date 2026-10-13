from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg


RECT_COLORS = (Color.BLUE, Color.GREEN)
NOISE_COLOR = Color.ORANGE
DISTRACTOR_COLOR = Color.RED
WEAK_COLOR = Color.GREY
BG = Color.BLACK
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIGHT_TRANSPARENT = {BG, DISTRACTOR_COLOR}


def _coord_key(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _rect_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_solid_large_rectangle(comp: dict, grid: np.ndarray) -> bool:
    color = int(comp["color"])
    if color not in RECT_COLORS or int(comp["area"]) < 9:
        return False
    if _rect_area(comp["bbox"]) != int(comp["area"]):
        return False
    patch = crop_bbox(grid, comp["bbox"])
    return bool((patch == color).all())


def _is_small_noise(comp: dict) -> bool:
    return int(comp["color"]) == NOISE_COLOR and 1 <= int(comp["area"]) <= 3


def _adjacency_count(noise_coords: np.ndarray, rect_mask: np.ndarray) -> int:
    total = 0
    h, w = rect_mask.shape
    for r, c in noise_coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and rect_mask[nr, nc]:
                total += 1
    return total


def _clear_segment(values: np.ndarray) -> bool:
    return bool(all(int(v) in SIGHT_TRANSPARENT for v in values))


def _has_sightline_to_rect(
    grid: np.ndarray,
    coord: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
) -> bool:
    r, c = map(int, coord)
    r0, c0, r1, c1 = map(int, rect_bbox)

    if r0 <= r <= r1:
        if c <= c0 - 2 and _clear_segment(grid[r, c + 1 : c0]):
            return True
        if c >= c1 + 2 and _clear_segment(grid[r, c1 + 1 : c]):
            return True
    if c0 <= c <= c1:
        if r <= r0 - 2 and _clear_segment(grid[r + 1 : r0, c]):
            return True
        if r >= r1 + 2 and _clear_segment(grid[r1 + 1 : r, c]):
            return True
    return False


def _classify_noise(
    grid: np.ndarray,
    noise_comp: dict,
    rectangles: list[dict],
) -> tuple[str, int | None, dict[int, int], bool]:
    adjacency_by_index: dict[int, int] = {}
    for idx, rect in enumerate(rectangles):
        rect_mask = np.zeros(grid.shape, dtype=bool)
        coords = rect["coords"]
        rect_mask[coords[:, 0], coords[:, 1]] = True
        adjacency_by_index[idx] = _adjacency_count(noise_comp["coords"], rect_mask)

    max_adj = max(adjacency_by_index.values(), default=0)
    adjacency_ok = max_adj >= 2
    sightline_ok = any(
        _has_sightline_to_rect(grid, (int(r), int(c)), rect["bbox"])
        for r, c in noise_comp["coords"]
        for rect in rectangles
    )

    if adjacency_ok and sightline_ok:
        best_idx = sorted(
            rectangles,
            key=lambda rect: (
                -adjacency_by_index[rect["_index"]],
                tuple(map(int, rect["bbox"])),
            ),
        )[0]["_index"]
        return "strong", int(rectangles[best_idx]["color"]), adjacency_by_index, True
    if adjacency_ok:
        return "weak", WEAK_COLOR, adjacency_by_index, False
    return "none", BG, adjacency_by_index, sightline_ok


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reclassify orange noise by combining rectangle adjacency with unobstructed sightlines.

    Input structure:
    - A black background contains 2-3 large solid rectangles in blue and green.
    - Small orange 4-connected objects of area at most three pixels act as candidate noise.
    - One or two red 2x2 blocks may appear as distractors and must be ignored during noise analysis.

    Rule:
    - Detect only the large solid blue/green rectangles as targets; ignore red blocks entirely.
    - For each orange noise object, count how many 4-neighbor contacts it has with each target rectangle, and separately test whether any noise pixel has a same-row or same-column clear sightline to any target rectangle through only background or red distractor cells.
    - If a noise object has at least two rectangle-adjacent sides and also has such a sightline, recolor it to the color of the rectangle with the highest adjacency count.
    - If it has the adjacency condition but no sightline, recolor it grey; otherwise erase it to black.

    Key insight:
    - The orange objects are split into three outcomes only by combining two independent signals: local 4-neighbor attachment and longer-range line-of-sight, while the red blocks are transparent distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    assert bg == BG

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    rectangles = [comp.copy() for comp in components if _is_solid_large_rectangle(comp, g)]
    noises = [comp.copy() for comp in components if _is_small_noise(comp)]

    for idx, rect in enumerate(sorted(rectangles, key=lambda comp: tuple(map(int, comp["bbox"])))):
        rect["_index"] = idx
    rectangles = sorted(rectangles, key=lambda comp: int(comp["_index"]))

    out = g.copy()
    for noise in noises:
        category, fill, _adj_map, _sight = _classify_noise(g, noise, rectangles)
        coords = noise["coords"]
        if category == "none":
            out[coords[:, 0], coords[:, 1]] = bg
        else:
            out[coords[:, 0], coords[:, 1]] = int(fill)
    return out


_SMALL_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
)


def _bbox_from_coords(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touch_or_overlap(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _paint_cells(grid: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _make_rectangle(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _can_place_cells(
    grid: np.ndarray,
    coords: set[tuple[int, int]],
    *,
    require_bg: bool = True,
    orange_gap: set[tuple[int, int]] | None = None,
) -> bool:
    h, w = grid.shape
    orange_gap = orange_gap or set()
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if require_bg and int(grid[r, c]) != BG:
            return False
        if (r, c) in orange_gap:
            return False
    return True


def _orange_buffer(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(coords)
    for r, c in list(coords):
        for dr, dc in DIR4:
            out.add((r + dr, c + dc))
    return out


def _place_optional_rectangle(
    grid: np.ndarray,
    occupied_boxes: list[tuple[int, int, int, int]],
    weak_row: int,
    weak_cols: set[int],
) -> set[tuple[int, int]] | None:
    if random.random() < 0.45:
        return None

    h, w = grid.shape
    for _ in range(80):
        rect_h = random.randint(3, 4)
        rect_w = random.randint(3, 5)
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if weak_row in range(r0, r0 + rect_h):
            continue
        if any(c in weak_cols for c in range(c0, c0 + rect_w)):
            continue
        if any(_touch_or_overlap(bbox, other, pad=1) for other in occupied_boxes):
            continue
        coords = _make_rectangle(r0, c0, rect_h, rect_w)
        if not _can_place_cells(grid, coords):
            continue
        return coords
    return None


def _find_none_far(
    grid: np.ndarray,
    rectangle_boxes: list[tuple[int, int, int, int]],
    orange_gap: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    shapes = list(_SMALL_SHAPES)
    random.shuffle(shapes)
    rect_rows = {r for r0, _c0, r1, _c1 in rectangle_boxes for r in range(r0, r1 + 1)}
    rect_cols = {c for _r0, c0, _r1, c1 in rectangle_boxes for c in range(c0, c1 + 1)}

    for shape in shapes:
        rel_rows = [r for r, _ in shape]
        rel_cols = [c for _, c in shape]
        max_r = h - max(rel_rows) - 1
        max_c = w - max(rel_cols) - 1
        candidates = [(r0, c0) for r0 in range(max_r + 1) for c0 in range(max_c + 1)]
        random.shuffle(candidates)
        for r0, c0 in candidates[:100]:
            coords = {(r0 + dr, c0 + dc) for dr, dc in shape}
            if not _can_place_cells(grid, coords, orange_gap=orange_gap):
                continue
            if any(r in rect_rows or c in rect_cols for r, c in coords):
                continue
            good = True
            for bbox in rectangle_boxes:
                rect_cells = _make_rectangle(*bbox[:2], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
                for r, c in coords:
                    if any((r + dr, c + dc) in rect_cells for dr, dc in DIR4):
                        good = False
                        break
                if not good:
                    break
            if good:
                return coords
    return None


def _place_random_red(
    grid: np.ndarray,
    occupied_boxes: list[tuple[int, int, int, int]],
    orange_gap: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    for _ in range(120):
        r0 = random.randint(0, h - 2)
        c0 = random.randint(0, w - 2)
        bbox = (r0, c0, r0 + 1, c0 + 1)
        if any(_touch_or_overlap(bbox, other, pad=1) for other in occupied_boxes):
            continue
        coords = _make_rectangle(r0, c0, 2, 2)
        if not _can_place_cells(grid, coords, orange_gap=orange_gap):
            continue
        return coords
    return None


def _validate_semantics(
    inp: np.ndarray,
    planned: dict[str, frozenset[tuple[int, int]]],
    expected_colors: dict[str, int],
) -> bool:
    components = find_components(inp, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    rectangles = [comp.copy() for comp in components if _is_solid_large_rectangle(comp, inp)]
    rectangles = sorted(rectangles, key=lambda comp: tuple(map(int, comp["bbox"])))
    for idx, rect in enumerate(rectangles):
        rect["_index"] = idx

    noises = [comp.copy() for comp in components if _is_small_noise(comp)]
    comp_map = {_coord_key(comp["coords"]): comp for comp in noises}

    if not all(key in comp_map for key in planned.values()):
        return False

    seen = {"strong": 0, "weak": 0, "none": 0}
    none_sight_ok = False
    out = inp.copy()

    for key, comp in comp_map.items():
        label = next((name for name, planned_key in planned.items() if planned_key == key), None)
        category, fill, adj_map, sightline = _classify_noise(inp, comp, rectangles)
        if label is None:
            if category != "none" or max(adj_map.values(), default=0) >= 2:
                return False
        else:
            if category != label:
                return False
            if fill != expected_colors[label]:
                return False
            if label == "strong":
                if max(adj_map.values(), default=0) < 2 or not sightline:
                    return False
            elif label == "weak":
                if max(adj_map.values(), default=0) < 2 or sightline:
                    return False
            else:
                if max(adj_map.values(), default=0) >= 2:
                    return False
                if sightline:
                    none_sight_ok = True

        seen[category] += 1
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(fill)

    if min(seen.values()) < 1 or not none_sight_ok:
        return False
    return np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    overlap_len = 3
    strong_len = 2
    weak_len = random.choice([2, 3])
    h_a = random.randint(max(5, overlap_len + 2), 6)
    w_a = random.randint(max(3, weak_len), 5)
    h_b = overlap_len + random.randint(1, 2)
    w_b = random.randint(3, 5)
    gap_ab = random.randint(5, 6)
    weak_side = random.choice(["top", "bottom"])

    extra_b = h_b - overlap_len
    min_h = max(14, h_a + extra_b + 4)
    h = random.randint(min_h, 18)

    if weak_side == "top":
        a_r0_max = h - (h_a + extra_b) - 2
        if a_r0_max < 2:
            return None
        a_r0 = random.randint(2, a_r0_max)
        b_r0 = a_r0 + h_a - overlap_len
    else:
        a_r0_min = extra_b + 2
        a_r0_max = h - h_a - 2
        if a_r0_max < a_r0_min:
            return None
        a_r0 = random.randint(a_r0_min, a_r0_max)
        b_r0 = a_r0 - extra_b

    a_r1 = a_r0 + h_a - 1
    b_r1 = b_r0 + h_b - 1

    a_c0 = random.randint(1, 3)
    min_w = max(16, a_c0 + w_a + gap_ab + w_b + 3)
    w = random.randint(min_w, 22)
    b_c0 = a_c0 + w_a + gap_ab

    a_color, b_color = random.sample(list(RECT_COLORS), 2)
    grid = np.zeros((h, w), dtype=int)

    rect_a = _make_rectangle(a_r0, a_c0, h_a, w_a)
    rect_b = _make_rectangle(b_r0, b_c0, h_b, w_b)
    _paint_cells(grid, rect_a, a_color)
    _paint_cells(grid, rect_b, b_color)

    occupied_boxes = [
        (a_r0, a_c0, a_r1, a_c0 + w_a - 1),
        (b_r0, b_c0, b_r1, b_c0 + w_b - 1),
    ]

    overlap_start = max(a_r0, b_r0)
    overlap_end = min(a_r1, b_r1)
    if overlap_end - overlap_start + 1 < strong_len:
        return None

    strong_r0 = random.randint(overlap_start, overlap_end - strong_len + 1)
    strong_c = a_c0 + w_a
    strong = {(strong_r0 + i, strong_c) for i in range(strong_len)}

    weak_row = a_r0 - 1 if weak_side == "top" else a_r1 + 1
    weak_c0 = random.randint(a_c0, a_c0 + w_a - weak_len)
    weak = {(weak_row, weak_c0 + i) for i in range(weak_len)}

    strong_rows = {strong_r0 + i for i in range(strong_len)}
    spare_rows = [r for r in range(overlap_start, overlap_end + 1) if r not in strong_rows]
    if not spare_rows:
        return None
    none_row = random.choice(spare_rows)
    none_col = strong_c + 2
    if none_col >= b_c0 - 2:
        return None
    none_sight = {(none_row, none_col)}

    orange_gap = set()
    for coords in [strong, weak, none_sight]:
        if not _can_place_cells(grid, coords, orange_gap=orange_gap):
            return None
        orange_gap |= _orange_buffer(coords)
        _paint_cells(grid, coords, NOISE_COLOR)

    weak_cols = {c for _, c in weak}
    rect_c = _place_optional_rectangle(grid, occupied_boxes, weak_row, weak_cols)
    if rect_c is not None:
        c_bbox = _bbox_from_coords(rect_c)
        occupied_boxes.append(c_bbox)
        _paint_cells(grid, rect_c, random.choice(RECT_COLORS))

    red_row = random.randint(overlap_start, overlap_end - 1)
    red_col = strong_c + 3
    corridor_red = _make_rectangle(red_row, red_col, 2, 2)
    if not _can_place_cells(grid, corridor_red, orange_gap=orange_gap):
        return None
    _paint_cells(grid, corridor_red, DISTRACTOR_COLOR)
    occupied_boxes.append((red_row, red_col, red_row + 1, red_col + 1))

    none_far = _find_none_far(grid, occupied_boxes[: 2 + int(rect_c is not None)], orange_gap)
    if none_far is None:
        return None
    if not _can_place_cells(grid, none_far, orange_gap=orange_gap):
        return None
    orange_gap |= _orange_buffer(none_far)
    _paint_cells(grid, none_far, NOISE_COLOR)

    if random.random() < 0.55:
        extra_red = _place_random_red(grid, occupied_boxes, orange_gap)
        if extra_red is not None:
            _paint_cells(grid, extra_red, DISTRACTOR_COLOR)
            occupied_boxes.append(_bbox_from_coords(extra_red))

    planned = {
        "strong": frozenset(strong),
        "weak": frozenset(weak),
        "none": frozenset(none_sight),
    }
    expected_colors = {
        "strong": a_color,
        "weak": WEAK_COLOR,
        "none": BG,
    }

    if not _validate_semantics(grid, planned, expected_colors):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
