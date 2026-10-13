import random
from collections import Counter

import numpy as np


BLACK = 0
BAR_COLOR = 8
GROUP_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (15, 19)
GRID_W_RANGE = (15, 19)
BAR_LENGTH_RANGE = (5, 9)
GROUP_COUNT_RANGE = (4, 6)
MAX_ATTEMPTS = 400

SHAPES_BY_AREA = {
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ],
    5: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ],
    6: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
    ],
}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, exclude=None) -> list[int]:
    exclude = set() if exclude is None else set(map(int, exclude))
    return sorted(int(v) for v in np.unique(grid) if int(v) not in exclude)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    comps = []
    for color in colors_present(grid, exclude={bg}):
        for coords in _component_mask(grid == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _diamond_reach(col: int, c0: int, c1: int) -> int:
    if col < c0 or col > c1:
        return -1
    return min(col - c0, c1 - col)


def _in_diamond(row: int, col: int, axis_row: int, c0: int, c1: int) -> bool:
    reach = _diamond_reach(int(col), int(c0), int(c1))
    return reach >= 0 and abs(int(row) - int(axis_row)) <= reach


def _reflect_coords(coords: np.ndarray, axis_row: int) -> np.ndarray:
    reflected = coords.copy()
    reflected[:, 0] = 2 * int(axis_row) - reflected[:, 0]
    return reflected


def _is_solid_horizontal_bar(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return r0 == r1 and int(comp["area"]) == (c1 - c0 + 1)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    comps = find_components(g, bg=bg)
    assert comps, "expected foreground components"

    bar_candidates = [comp for comp in comps if comp["color"] == BAR_COLOR and _is_solid_horizontal_bar(comp)]
    assert len(bar_candidates) == 1, "expected exactly one cyan horizontal bar"
    bar = bar_candidates[0]
    axis_row = int(bar["bbox"][0])
    c0 = int(bar["bbox"][1])
    c1 = int(bar["bbox"][3])
    bar_length = int(bar["area"])
    assert BAR_COLOR != bg, "bar color must be distinct from background"
    assert 5 <= bar_length <= 9, "bar length must stay within puzzle bounds"

    groups = [comp for comp in comps if comp is not bar]
    assert 4 <= len(groups) <= 6, "expected 4-6 colored groups"
    assert len({int(comp["color"]) for comp in groups}) == len(groups), "group colors must be unique"
    assert all(int(comp["color"]) != BAR_COLOR for comp in groups), "cyan is reserved for the bar"

    inside = []
    outside = []
    for comp in groups:
        center = _bbox_center(comp["bbox"])
        if _in_diamond(center[0], center[1], axis_row, c0, c1):
            inside.append(comp)
        else:
            outside.append(comp)

    assert 2 <= len(inside) <= 3, "diamond region must contain exactly 2-3 candidate groups"
    min_area = min(int(comp["area"]) for comp in inside)
    target_candidates = [comp for comp in inside if int(comp["area"]) == min_area]
    assert len(target_candidates) == 1, "inside groups must have a unique smallest area"
    target = target_candidates[0]

    reflected = _reflect_coords(target["coords"], axis_row)
    h, w = g.shape
    for rr, cc in reflected:
        assert 0 <= int(rr) < h and 0 <= int(cc) < w, "reflection must stay within the grid"

    return {
        "bg": int(bg),
        "bar": bar,
        "groups": groups,
        "inside": inside,
        "outside": outside,
        "target": target,
        "axis_row": axis_row,
        "bar_c0": c0,
        "bar_c1": c1,
        "reflected": reflected,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the unique smallest group whose bbox center falls inside the bar-defined diamond.

    Input structure:
    - The background is black.
    - One cyan one-cell-thick horizontal solid bar appears near the vertical middle of the grid.
    - Four to six other monochromatic 4-connected groups of distinct colors are scattered away from the bar.

    Rule:
    - Use the cyan bar as both a horizontal mirror axis and the middle line of an invisible diamond envelope.
    - A group's eligibility is determined only by whether the floor-rounded center of its bounding box lies inside or on that diamond.
    - Among the eligible groups, select the unique smallest one by pixel count and copy it by vertical reflection across the bar row.
    - Keep the original grid unchanged, then paint the reflected copy on top, overwriting any pixels at the landing cells.

    Key insight:
    - The same cyan bar simultaneously defines the candidate region, chooses which groups are even considered, and provides the reflection axis for the final duplication.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_scene(g)
    out = g.copy()
    color = int(info["target"]["color"])
    for rr, cc in info["reflected"]:
        out[int(rr), int(cc)] = color
    return out


def _coords_from_template(center: tuple[int, int], template: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    arr = np.array(template, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    top = int(center[0]) - (h - 1) // 2
    left = int(center[1]) - (w - 1) // 2
    return [(top + int(r), left + int(c)) for r, c in template]


def _mark_group_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        blocked[r, c] = True
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _can_place(coords: list[tuple[int, int]], occupied: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(occupied[r, c]) or bool(blocked[r, c]):
            return False
    return True


def _try_place_at_center(
    rng: random.Random,
    occupied: np.ndarray,
    blocked: np.ndarray,
    center: tuple[int, int],
    area: int,
    color: int,
) -> dict | None:
    templates = list(SHAPES_BY_AREA[int(area)])
    rng.shuffle(templates)
    for template in templates:
        coords = _coords_from_template(center, template)
        if _can_place(coords, occupied, blocked):
            arr = np.array(coords, dtype=int)
            occupied[arr[:, 0], arr[:, 1]] = True
            _mark_group_blocked(blocked, coords)
            return {
                "color": int(color),
                "coords": arr,
                "bbox": bbox_coords(arr),
                "area": int(arr.shape[0]),
                "center": tuple(map(int, center)),
            }
    return None


def _search_group(
    rng: random.Random,
    occupied: np.ndarray,
    blocked: np.ndarray,
    candidate_centers: list[tuple[int, int]],
    areas: list[int],
    color: int,
    predicate,
) -> dict | None:
    centers = list(candidate_centers)
    rng.shuffle(centers)
    areas = list(areas)
    rng.shuffle(areas)
    for center in centers:
        for area in areas:
            templates = list(SHAPES_BY_AREA[int(area)])
            rng.shuffle(templates)
            for template in templates:
                coords = _coords_from_template(center, template)
                if not _can_place(coords, occupied, blocked):
                    continue
                arr = np.array(coords, dtype=int)
                comp = {
                    "color": int(color),
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                    "center": tuple(map(int, center)),
                }
                if not bool(predicate(comp)):
                    continue
                occupied[arr[:, 0], arr[:, 1]] = True
                _mark_group_blocked(blocked, coords)
                return comp
    return None


def _center_positions(h: int, w: int, axis_row: int, c0: int, c1: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    inside = []
    boundary = []
    outside = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            inside_flag = _in_diamond(r, c, axis_row, c0, c1)
            if inside_flag:
                inside.append((r, c))
                reach = _diamond_reach(c, c0, c1)
                if abs(r - axis_row) == reach and reach > 0:
                    outside_r = axis_row + (reach + 1 if r > axis_row else -(reach + 1))
                    if 1 <= outside_r < h - 1:
                        boundary.append((r, c))
            else:
                outside.append((r, c))
    return inside, boundary, outside


def _center_gap_ok(existing: list[tuple[int, int]], candidate: tuple[int, int], minimum: int = 2) -> bool:
    return all(abs(candidate[0] - r) + abs(candidate[1] - c) >= minimum for r, c in existing)


def _build_grid(h: int, w: int, bar_row: int, c0: int, length: int, groups: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[bar_row, c0 : c0 + length] = BAR_COLOR
    for comp in groups:
        rr = comp["coords"][:, 0]
        cc = comp["coords"][:, 1]
        grid[rr, cc] = int(comp["color"])
    return grid


def _choose_bar(rng: random.Random, h: int, w: int) -> tuple[int, int, int]:
    length = rng.randint(BAR_LENGTH_RANGE[0], BAR_LENGTH_RANGE[1])
    row_low = max(3, h // 2 - 2)
    row_high = min(h - 4, h // 2 + 2)
    if row_low > row_high:
        row_low, row_high = 3, h - 4
    row = rng.randint(row_low, row_high)
    c0 = rng.randint(2, w - length - 2)
    return row, c0, length


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    axis_row, bar_c0, bar_len = _choose_bar(rng, h, w)
    bar_c1 = bar_c0 + bar_len - 1

    inside_centers, boundary_centers, outside_centers = _center_positions(h, w, axis_row, bar_c0, bar_c1)
    if len(inside_centers) < 3 or len(outside_centers) < 3:
        return None

    inside_count = rng.choice([2, 3])
    boundary_mode = bool(boundary_centers) and rng.random() < 0.7
    overlap_mode = rng.random() < 0.45
    special_outside_needed = int(boundary_mode) + int(overlap_mode)
    outside_count_min = max(2, special_outside_needed)
    outside_count_max = GROUP_COUNT_RANGE[1] - inside_count
    if outside_count_min > outside_count_max:
        return None
    outside_count = rng.randint(outside_count_min, outside_count_max)
    total_groups = inside_count + outside_count

    outside_smaller_mode = rng.random() < 0.65
    if not (boundary_mode or overlap_mode or outside_smaller_mode):
        outside_smaller_mode = True

    target_area = rng.randint(3 if outside_smaller_mode else 2, 4)
    larger_area_pool = [area for area in range(target_area + 1, 7)]
    if len(larger_area_pool) < inside_count - 1:
        return None
    rng.shuffle(larger_area_pool)
    inside_areas = [target_area] + larger_area_pool[: inside_count - 1]
    rng.shuffle(inside_areas[1:])

    color_pool = list(GROUP_COLORS)
    rng.shuffle(color_pool)
    colors = color_pool[:total_groups]

    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    occupied[axis_row, bar_c0 : bar_c1 + 1] = True

    groups = []
    chosen_centers = []
    meta = {
        "boundary_mode": boundary_mode,
        "overlap_mode": overlap_mode,
        "outside_smaller_mode": outside_smaller_mode,
    }

    if boundary_mode:
        boundary_options = [pos for pos in boundary_centers if _center_gap_ok(chosen_centers, pos)]
        if not boundary_options:
            return None
        target_center = rng.choice(boundary_options)
    else:
        target_options = [pos for pos in inside_centers if _center_gap_ok(chosen_centers, pos)]
        if not target_options:
            return None
        target_center = rng.choice(target_options)

    target_color = colors.pop()
    target = _try_place_at_center(rng, occupied, blocked, target_center, target_area, target_color)
    if target is None:
        return None
    groups.append(target)
    chosen_centers.append(target_center)
    meta["target_color"] = target_color
    meta["target_center"] = target_center
    meta["target_area"] = target_area

    for area in inside_areas[1:]:
        candidates = [pos for pos in inside_centers if pos != target_center and _center_gap_ok(chosen_centers, pos)]
        if not candidates:
            return None
        placed = _search_group(
            rng,
            occupied,
            blocked,
            candidates,
            [area],
            colors.pop(),
            lambda comp: True,
        )
        if placed is None:
            return None
        groups.append(placed)
        chosen_centers.append(placed["center"])

    reflected_target = {tuple(map(int, rc)) for rc in _reflect_coords(target["coords"], axis_row)}

    special_outside_centers = set()
    if boundary_mode:
        dr = 1 if target_center[0] > axis_row else -1
        reach = _diamond_reach(target_center[1], bar_c0, bar_c1)
        just_outside = (axis_row + dr * (reach + 1), target_center[1])
        special_outside_centers.add(just_outside)
        boundary_area = rng.randint(2, max(2, target_area - 1)) if outside_smaller_mode else rng.randint(2, 6)
        placed = _try_place_at_center(rng, occupied, blocked, just_outside, boundary_area, colors.pop())
        if placed is None:
            return None
        groups.append(placed)
        chosen_centers.append(placed["center"])
        outside_count -= 1
        meta["boundary_outside_center"] = just_outside

    if overlap_mode:
        candidate_centers = [pos for pos in outside_centers if pos not in special_outside_centers and _center_gap_ok(chosen_centers, pos, minimum=1)]
        overlap_area_choices = [max(4, target_area + 1), 5, 6]
        overlap_area_choices = [area for area in overlap_area_choices if 2 <= area <= 6]
        placed = _search_group(
            rng,
            occupied,
            blocked,
            candidate_centers,
            overlap_area_choices,
            colors.pop(),
            lambda comp: any(tuple(map(int, rc)) in reflected_target for rc in comp["coords"])
            and not _in_diamond(_bbox_center(comp["bbox"])[0], _bbox_center(comp["bbox"])[1], axis_row, bar_c0, bar_c1),
        )
        if placed is None:
            return None
        groups.append(placed)
        chosen_centers.append(placed["center"])
        outside_count -= 1

    remaining_outside_areas = []
    if outside_smaller_mode and not any(int(comp["area"]) < target_area and not _in_diamond(_bbox_center(comp["bbox"])[0], _bbox_center(comp["bbox"])[1], axis_row, bar_c0, bar_c1) for comp in groups):
        if outside_count <= 0:
            return None
        remaining_outside_areas.append(rng.randint(2, target_area - 1))
        outside_count -= 1
    while outside_count > 0:
        remaining_outside_areas.append(rng.randint(2, 6))
        outside_count -= 1

    for area in remaining_outside_areas:
        candidates = [pos for pos in outside_centers if _center_gap_ok(chosen_centers, pos, minimum=1)]
        if not candidates:
            return None
        placed = _search_group(
            rng,
            occupied,
            blocked,
            candidates,
            [area],
            colors.pop(),
            lambda comp: not _in_diamond(_bbox_center(comp["bbox"])[0], _bbox_center(comp["bbox"])[1], axis_row, bar_c0, bar_c1),
        )
        if placed is None:
            return None
        groups.append(placed)
        chosen_centers.append(placed["center"])

    if len(groups) != total_groups:
        return None

    grid = _build_grid(h, w, axis_row, bar_c0, bar_len, groups)
    try:
        _validate_generation(grid, meta)
    except AssertionError:
        return None
    return grid


def _validate_generation(grid: np.ndarray, meta: dict) -> None:
    info = _analyze_scene(grid)
    target = info["target"]
    target_center = _bbox_center(target["bbox"])
    assert target_center == tuple(meta["target_center"])
    assert int(target["area"]) == int(meta["target_area"])
    assert int(target["color"]) == int(meta["target_color"])

    out = main(grid)
    diff = np.argwhere(out != grid)
    assert diff.size > 0, "reflection must add or overwrite at least one cell"
    reflected = {tuple(map(int, rc)) for rc in info["reflected"]}
    changed = {tuple(map(int, rc)) for rc in diff}
    assert changed <= reflected, "only reflected target cells may change"

    if meta["outside_smaller_mode"]:
        outside_areas = [int(comp["area"]) for comp in info["outside"]]
        assert outside_areas and min(outside_areas) < int(target["area"]), "must include a smaller outside distractor"
        global_smallest = min(info["groups"], key=lambda comp: int(comp["area"]))
        assert global_smallest is not target, "diamond filtering must matter"

    if meta["boundary_mode"]:
        assert any(
            abs(_bbox_center(comp["bbox"])[0] - info["axis_row"]) == _diamond_reach(_bbox_center(comp["bbox"])[1], info["bar_c0"], info["bar_c1"])
            for comp in info["inside"]
        ), "need an inside center on the diamond boundary"
        r, c = map(int, meta["boundary_outside_center"])
        assert not _in_diamond(r, c, info["axis_row"], info["bar_c0"], info["bar_c1"])

    if meta["overlap_mode"]:
        before = grid.copy()
        overwritten = []
        for rr, cc in info["reflected"]:
            rr = int(rr)
            cc = int(cc)
            if int(before[rr, cc]) not in {info["bg"], int(target["color"])}:
                overwritten.append((rr, cc))
        assert overwritten, "need a reflection-overwrite case"
        for rr, cc in overwritten:
            assert int(out[rr, cc]) == int(target["color"])


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 4")
