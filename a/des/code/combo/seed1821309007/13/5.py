import numpy as np
from collections import Counter, defaultdict, deque


ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

# Generator configuration.
GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
MAJORITY_COUNT_CHOICES = (3, 4, 5)
MINORITY_TOTAL_CHOICES = (2, 3, 4)
AREA_TO_DIMS = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
    12: [(3, 4), (4, 3)],
    15: [(3, 5), (5, 3)],
    16: [(4, 4)],
}
AREA_VALUES = sorted(AREA_TO_DIMS)
GEN_RETRIES = 400


def _component(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    q = deque([start])
    seen[start] = True
    coords = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for dr, dc in ORTHO_DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return np.array(coords, dtype=int)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            same = g == color
            coords = _component(same, (r, c), seen)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                    "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _extract_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _paste_mask(out: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = out.shape
    mh, mw = mask.shape
    for rr in range(mh):
        for cc in range(mw):
            if not mask[rr, cc]:
                continue
            r = top + rr
            c = left + cc
            if 0 <= r < h and 0 <= c < w and out[r, c] == 0:
                out[r, c] = color


def _majority_color(color_counts: Counter) -> int:
    return max(sorted(color_counts), key=lambda color: (color_counts[color], -color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep majority-color objects fixed. Move each minority-color object to the
    180-degree-rotated bbox position of the closest-size majority object.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _find_components(g)
    assert comps, "expected at least one foreground object"

    color_counts = Counter(comp["color"] for comp in comps)
    major_color = _majority_color(color_counts)
    major = [comp for comp in comps if comp["color"] == major_color]
    minor = [comp for comp in comps if comp["color"] != major_color]
    assert major and minor

    out = np.zeros_like(g)
    for comp in major:
        mask = _extract_mask(comp)
        r0, c0, _r1, _c1 = comp["bbox"]
        _paste_mask(out, mask, comp["color"], r0, c0)

    groups = defaultdict(list)
    for idx, comp in enumerate(major):
        groups[idx] = []

    for comp in minor:
        distances = [abs(comp["bbox_area"] - mj["bbox_area"]) for mj in major]
        best = min(range(len(major)), key=lambda i: (distances[i], major[i]["bbox"][0], major[i]["bbox"][1]))
        groups[best].append(comp)

    for idx, bucket in groups.items():
        bucket.sort(key=lambda comp: (-comp["area"], comp["bbox"][0], comp["bbox"][1]))
        mr0, mc0, mr1, mc1 = major[idx]["bbox"]
        mh = mr1 - mr0 + 1
        mw = mc1 - mc0 + 1
        target_r0 = h - mh - mr0
        target_c0 = w - mw - mc0
        for comp in bucket:
            mask = _extract_mask(comp)
            _paste_mask(out, mask, comp["color"], target_r0, target_c0)

    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(r, c)]
    step_r = 1 if tr > r else -1
    while r != tr:
        r += step_r
        cells.append((r, c))
    step_c = 1 if tc > c else -1
    while c != tc:
        c += step_c
        cells.append((r, c))
    return cells


def _random_connected_mask(height: int, width: int, cells: int) -> np.ndarray:
    cells = max(cells, height + width - 1)
    top = (0, int(RNG.integers(width)))
    bottom = (height - 1, int(RNG.integers(width)))
    left = (int(RNG.integers(height)), 0)
    right = (int(RNG.integers(height)), width - 1)
    seeds = [top, left, bottom, right]

    chosen = set()
    cur = seeds[0]
    chosen.add(cur)
    for nxt in seeds[1:]:
        for cell in _manhattan_path(cur, nxt):
            chosen.add(cell)
        cur = nxt

    frontier = set()
    for r, c in chosen:
        for dr, dc in ORTHO_DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in chosen:
                frontier.add((rr, cc))

    while len(chosen) < min(cells, height * width) and frontier:
        frontier_list = list(frontier)
        pick = frontier_list[int(RNG.integers(len(frontier_list)))]
        frontier.remove(pick)
        chosen.add(pick)
        r, c = pick
        for dr, dc in ORTHO_DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in chosen:
                frontier.add((rr, cc))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _bbox_clear(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_bbox(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _closest_major_index(area: int, major_areas: list[int]) -> int | None:
    diffs = [abs(area - value) for value in major_areas]
    best = min(diffs)
    winners = [i for i, diff in enumerate(diffs) if diff == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _choose_minority_bbox_area(target_idx: int, major_areas: list[int]) -> int:
    options = [area for area in AREA_VALUES if _closest_major_index(area, major_areas) == target_idx]
    assert options
    exact = [area for area in options if area == major_areas[target_idx]]
    pool = exact if exact and RNG.random() < 0.65 else options
    return int(pool[int(RNG.integers(len(pool)))])


def _sample_target_indices(major_count: int, minority_total: int) -> list[int]:
    duplicate_branch = minority_total >= 2 and RNG.random() < 0.5
    if duplicate_branch:
        repeated = int(RNG.integers(major_count))
        indices = [repeated, repeated]
        while len(indices) < minority_total:
            indices.append(int(RNG.integers(major_count)))
        RNG.shuffle(indices)
        return indices
    if minority_total <= major_count:
        picked = RNG.choice(major_count, size=minority_total, replace=False)
        return [int(x) for x in picked]
    picked = list(range(major_count))
    while len(picked) < minority_total:
        picked.append(int(RNG.integers(major_count)))
    RNG.shuffle(picked)
    return picked


def _assign_minority_colors(total: int) -> list[int]:
    palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    major_color = int(RNG.choice(palette))
    remaining = [color for color in palette if color != major_color]
    use_three_colors = RNG.random() < 0.5
    if not use_three_colors:
        minor_colors = [int(RNG.choice(remaining))]
        return [major_color] + minor_colors
    minor_pair = RNG.choice(remaining, size=2, replace=False)
    return [major_color, int(minor_pair[0]), int(minor_pair[1])]


def _layout_majorities(height: int, width: int, specs: list[tuple[int, int]]) -> list[dict] | None:
    reserved = np.zeros((height, width), dtype=bool)
    placed = []
    order = list(range(len(specs)))
    RNG.shuffle(order)
    for idx in order:
        obj_h, obj_w = specs[idx]
        success = False
        for _ in range(200):
            top = int(RNG.integers(1, height - obj_h))
            left = int(RNG.integers(1, width - obj_w))
            sym_top = height - obj_h - top
            sym_left = width - obj_w - left
            if not _bbox_clear(reserved, top, left, obj_h, obj_w):
                continue
            if not _bbox_clear(reserved, sym_top, sym_left, obj_h, obj_w):
                continue
            if max(abs(top - sym_top), abs(left - sym_left)) <= 1:
                continue
            _mark_bbox(reserved, top, left, obj_h, obj_w)
            _mark_bbox(reserved, sym_top, sym_left, obj_h, obj_w)
            placed.append(
                {
                    "index": idx,
                    "top": top,
                    "left": left,
                    "height": obj_h,
                    "width": obj_w,
                    "target_top": sym_top,
                    "target_left": sym_left,
                }
            )
            success = True
            break
        if not success:
            return None
    placed.sort(key=lambda item: item["index"])
    return placed


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(*GRID_H_RANGE))
    width = int(RNG.integers(*GRID_W_RANGE))
    major_count = int(RNG.choice(MAJORITY_COUNT_CHOICES))
    minor_total = int(RNG.choice(MINORITY_TOTAL_CHOICES))

    color_setup = _assign_minority_colors(minor_total)
    major_color = color_setup[0]
    minor_palette = color_setup[1:]

    available_areas = AREA_VALUES.copy()
    RNG.shuffle(available_areas)
    major_areas = sorted(available_areas[:major_count])
    major_specs = [AREA_TO_DIMS[area][int(RNG.integers(len(AREA_TO_DIMS[area])))] for area in major_areas]
    major_layout = _layout_majorities(height, width, major_specs)
    if major_layout is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    actual_reserved = np.zeros((height, width), dtype=bool)
    target_reserved = np.zeros((height, width), dtype=bool)

    major_objects = []
    for area, spec, loc in zip(major_areas, major_specs, major_layout):
        obj_h, obj_w = spec
        cell_min = max(obj_h + obj_w - 1, area // 2)
        cell_max = max(cell_min, area - 1)
        cells = int(RNG.integers(cell_min, cell_max + 1))
        mask = _random_connected_mask(obj_h, obj_w, cells)
        _paste_mask(grid, mask, major_color, loc["top"], loc["left"])
        _mark_bbox(actual_reserved, loc["top"], loc["left"], obj_h, obj_w)
        _mark_bbox(target_reserved, loc["target_top"], loc["target_left"], obj_h, obj_w)
        major_objects.append(
            {
                "bbox_area": area,
                "height": obj_h,
                "width": obj_w,
                "target_top": loc["target_top"],
                "target_left": loc["target_left"],
            }
        )

    target_indices = _sample_target_indices(major_count, minor_total)
    target_counts = Counter(target_indices)

    minority_colors = []
    if len(minor_palette) == 1:
        minority_colors = [minor_palette[0]] * minor_total
    else:
        split = int(RNG.integers(1, minor_total))
        minority_colors = [minor_palette[0]] * split + [minor_palette[1]] * (minor_total - split)
        RNG.shuffle(minority_colors)

    minority_specs = []
    used_area_by_target = defaultdict(set)
    used_cells_by_target = defaultdict(set)
    for idx, color in zip(target_indices, minority_colors):
        bbox_area = _choose_minority_bbox_area(idx, major_areas)
        dims = AREA_TO_DIMS[bbox_area][int(RNG.integers(len(AREA_TO_DIMS[bbox_area])))]
        obj_h, obj_w = dims
        min_cells = max(obj_h + obj_w - 1, bbox_area // 2)
        max_cells = bbox_area
        if target_counts[idx] >= 2:
            choices = [value for value in range(min_cells, max_cells + 1) if value not in used_cells_by_target[idx]]
            if not choices:
                choices = list(range(min_cells, max_cells + 1))
            cells = int(choices[int(RNG.integers(len(choices)))])
            used_cells_by_target[idx].add(cells)
        else:
            cells = int(RNG.integers(min_cells, max_cells + 1))
        minority_specs.append(
            {
                "target_idx": idx,
                "color": color,
                "bbox_area": bbox_area,
                "height": obj_h,
                "width": obj_w,
                "cells": cells,
            }
        )

    for spec in minority_specs:
        placed = False
        for _ in range(200):
            top = int(RNG.integers(1, height - spec["height"]))
            left = int(RNG.integers(1, width - spec["width"]))
            if not _bbox_clear(actual_reserved, top, left, spec["height"], spec["width"]):
                continue
            if not _bbox_clear(target_reserved, top, left, spec["height"], spec["width"]):
                continue
            mask = _random_connected_mask(spec["height"], spec["width"], spec["cells"])
            _paste_mask(grid, mask, spec["color"], top, left)
            _mark_bbox(actual_reserved, top, left, spec["height"], spec["width"])
            spec["input_top"] = top
            spec["input_left"] = left
            placed = True
            break
        if not placed:
            return None

    comps = _find_components(grid)
    color_counts = Counter(comp["color"] for comp in comps)
    if color_counts[major_color] != major_count:
        return None
    if max(v for k, v in color_counts.items() if k != major_color) >= major_count:
        return None

    detected_major = _majority_color(color_counts)
    if detected_major != major_color:
        return None

    major_comps = [comp for comp in comps if comp["color"] == major_color]
    minor_comps = [comp for comp in comps if comp["color"] != major_color]
    if len(major_comps) != major_count or len(minor_comps) != minor_total:
        return None

    for comp in minor_comps:
        winner = _closest_major_index(comp["bbox_area"], [mj["bbox_area"] for mj in major_objects])
        if winner is None:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any(out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
