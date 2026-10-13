from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
CANDIDATE_COLORS = (3, 4, 6, 7, 8)

MAX_ATTEMPTS = 400
GRID_H_RANGE = (16, 23)
GRID_W_RANGE = (18, 25)
BAND_HEIGHTS = (4, 5)
POCKET_COUNT_RANGE = (2, 4)
STRUCTURE_SIDE_MARGIN = (1, 2)
POCKET_MIN_GAP = 1
MIN_CANDIDATE_GAP_ROWS = 3


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        deltas = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in deltas:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, colors: list[int]) -> list[dict]:
    comps: list[dict] = []
    for color in colors:
        for coords in _connected_components(grid == color, connectivity=4):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return comps


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = _normalize_mask(np.rot90(mask, k))
        key = (tuple(map(int, rotated.shape)), rotated.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(rotated)
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for rotated in _unique_rotations(mask):
        keys.append((tuple(map(int, rotated.shape)), rotated.tobytes()))
    return min(keys)


def _pocket_columns(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 1].min()), int(coords[:, 1].max())


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(Counter(map(int, g.ravel())).most_common(1)[0][0]) == BLACK

    blue_components = _connected_components(g == BLUE, connectivity=4)
    assert len(blue_components) == 1
    blue_coords = blue_components[0]
    blue_bbox = _bbox_from_coords(blue_coords)
    br0, bc0, br1, bc1 = blue_bbox

    band_mask = g[br0 : br1 + 1, bc0 : bc1 + 1] == BLUE
    pockets = []
    for rel_coords in _connected_components(~band_mask, connectivity=4):
        rs = rel_coords[:, 0]
        cs = rel_coords[:, 1]
        touches_top = bool(np.any(rs == 0))
        touches_bottom = bool(np.any(rs == band_mask.shape[0] - 1))
        touches_left = bool(np.any(cs == 0))
        touches_right = bool(np.any(cs == band_mask.shape[1] - 1))
        if touches_bottom and not (touches_top or touches_left or touches_right):
            coords = rel_coords + np.array([br0, bc0])
            mask = _mask_from_coords(coords)
            pockets.append(
                {
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "mask": mask,
                    "cols": _pocket_columns(coords),
                    "key": _canonical_key(mask),
                }
            )
    assert pockets
    pockets.sort(key=lambda p: (p["cols"][0], p["cols"][1]))

    candidate_colors = sorted(
        int(c) for c in np.unique(g) if int(c) not in (BLACK, BLUE)
    )
    candidates = []
    for comp in _monochrome_components(g, candidate_colors):
        coords = comp["coords"]
        mask = _mask_from_coords(coords)
        candidates.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": comp["bbox"],
                "mask": mask,
                "cols": _pocket_columns(coords),
                "key": _canonical_key(mask),
            }
        )
    assert candidates
    return {
        "blue_bbox": blue_bbox,
        "pockets": pockets,
        "candidates": candidates,
    }


def _assigned_pocket(candidate: dict, pockets: list[dict]) -> dict | None:
    c0, c1 = candidate["cols"]
    matches = [p for p in pockets if p["cols"][0] <= c0 and c1 <= p["cols"][1]]
    if len(matches) != 1:
        return None
    return matches[0]


def _shape_matches(candidate_mask: np.ndarray, pocket_mask: np.ndarray) -> bool:
    return any(np.array_equal(rot, pocket_mask) for rot in _unique_rotations(candidate_mask))


def _matched_candidate_colors(grid: np.ndarray) -> set[int]:
    scene = _extract_scene(grid)
    matched: set[int] = set()
    for candidate in scene["candidates"]:
        pocket = _assigned_pocket(candidate, scene["pockets"])
        if pocket is not None and _shape_matches(candidate["mask"], pocket["mask"]):
            matched.add(int(candidate["color"]))
    return matched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor lower candidate shapes red when their upward-landing column range points to a pocket they can match by rotation.

    Input structure:
    - The grid has a black background, one upper blue 4-connected structure, and several lower 4-connected single-color candidate shapes.
    - Inside the blue structure's bbox are downward-opening black pockets: 4-connected cavities that touch only the bbox's bottom edge.

    Rule:
    - Treat each lower candidate as sliding straight upward until its top row would sit one row below the blue structure's bottom row, without changing its columns.
    - Use that landed column range to choose the unique pocket whose column range fully contains the candidate's column range; if none does, the candidate has no pocket.
    - If a chosen pocket exists and some 0/90/180/270 rotation of the candidate's tight mask exactly equals the pocket's tight mask, recolor the candidate at its original location to red.
    - Leave unmatched candidates, the blue structure, the pockets, and the background unchanged.

    Key insight:
    - The upward slide is only a routing step: it fixes which pocket a candidate is allowed to test, and only that routed pocket is then checked for rotation-exact shape equivalence.
    """
    g = np.asarray(grid, dtype=int)
    scene = _extract_scene(g)

    out = g.copy()
    for candidate in scene["candidates"]:
        pocket = _assigned_pocket(candidate, scene["pockets"])
        if pocket is not None and _shape_matches(candidate["mask"], pocket["mask"]):
            out[candidate["coords"][:, 0], candidate["coords"][:, 1]] = RED
    return out


_TEMPLATE_MASKS = [
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
]


def _unique_template_pool(max_height: int) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for mask in _TEMPLATE_MASKS:
        if mask.shape[0] >= max_height:
            continue
        if int(mask.sum()) < 3 or int(mask.sum()) > 6:
            continue
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            pool.append(mask.copy())
    return pool


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _choose_pockets(
    band_h: int, grid_w: int, pocket_count: int
) -> tuple[list[dict], tuple[int, int]] | None:
    pool = _unique_template_pool(band_h)
    if len(pool) < pocket_count:
        return None

    left_margin = random.randint(*STRUCTURE_SIDE_MARGIN)
    right_margin = random.randint(*STRUCTURE_SIDE_MARGIN)
    structure_left = left_margin
    structure_right = grid_w - 1 - right_margin
    structure_w = structure_right - structure_left + 1
    interior_w = structure_w - 2
    if interior_w < 6:
        return None

    selected = random.sample(pool, pocket_count)
    widths = [int(mask.shape[1]) for mask in selected]
    base = sum(widths) + POCKET_MIN_GAP * (pocket_count - 1)
    if base > interior_w:
        return None
    extras = _random_partition(interior_w - base, pocket_count + 1)

    pockets = []
    cursor = structure_left + 1 + extras[0]
    for idx, mask in enumerate(selected):
        if idx > 0:
            cursor += POCKET_MIN_GAP + extras[idx]
        width = int(mask.shape[1])
        height = int(mask.shape[0])
        col0 = cursor
        row0 = band_h - height
        local_coords = np.argwhere(mask) + np.array([row0, col0 - structure_left])
        pockets.append(
            {
                "mask": mask.copy(),
                "local_coords": local_coords,
                "cols": (col0, col0 + width - 1),
                "key": _canonical_key(mask),
            }
        )
        cursor += width
    return pockets, (structure_left, structure_right)


def _gap_slots(grid_w: int, pocket_slots: list[tuple[int, int]]) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cur = 0
    for start, end in sorted(pocket_slots):
        if cur <= start - 1:
            gaps.append((cur, start - 1))
        cur = end + 1
    if cur <= grid_w - 1:
        gaps.append((cur, grid_w - 1))
    return [gap for gap in gaps if gap[0] <= gap[1]]


def _option_space(
    base_masks: list[np.ndarray],
    pocket_slots: list[tuple[int, int]],
    gap_slots: list[tuple[int, int]],
) -> list[list[dict]]:
    all_options: list[list[dict]] = []
    for index, base_mask in enumerate(base_masks):
        options: list[dict] = []
        for variant in _unique_rotations(base_mask):
            height, width = map(int, variant.shape)
            for slot_index, slot in enumerate(pocket_slots):
                slot_w = slot[1] - slot[0] + 1
                if width <= slot_w:
                    branch = "match" if slot_index == index else "wrong"
                    options.append(
                        {
                            "mask": variant.copy(),
                            "slot_kind": "pocket",
                            "slot_index": slot_index,
                            "branch": branch,
                            "height": height,
                            "width": width,
                        }
                    )
            for slot_index, slot in enumerate(gap_slots):
                slot_w = slot[1] - slot[0] + 1
                if width <= slot_w:
                    options.append(
                        {
                            "mask": variant.copy(),
                            "slot_kind": "gap",
                            "slot_index": slot_index,
                            "branch": "gap",
                            "height": height,
                            "width": width,
                        }
                    )
        dedup: dict[tuple, dict] = {}
        for option in options:
            key = (
                option["slot_kind"],
                option["slot_index"],
                option["branch"],
                tuple(map(int, option["mask"].shape)),
                option["mask"].tobytes(),
            )
            dedup[key] = option
        all_options.append(list(dedup.values()))
    return all_options


def _choose_assignment(
    base_masks: list[np.ndarray],
    pocket_slots: list[tuple[int, int]],
    gap_slots: list[tuple[int, int]],
) -> list[dict] | None:
    options = _option_space(base_masks, pocket_slots, gap_slots)
    order = sorted(range(len(options)), key=lambda i: len(options[i]))
    solutions: list[list[dict]] = []

    def backtrack(pos: int, used_slots: set[tuple[str, int]], chosen: dict[int, dict]) -> None:
        if len(solutions) >= 200:
            return
        if pos == len(order):
            assignment = [chosen[i] for i in range(len(base_masks))]
            branches = [item["branch"] for item in assignment]
            match_count = branches.count("match")
            if 0 < match_count < len(base_masks):
                solutions.append(assignment)
            return
        idx = order[pos]
        shuffled = options[idx][:]
        random.shuffle(shuffled)
        for option in shuffled:
            slot_key = (option["slot_kind"], int(option["slot_index"]))
            if slot_key in used_slots:
                continue
            chosen[idx] = option
            used_slots.add(slot_key)
            backtrack(pos + 1, used_slots, chosen)
            used_slots.remove(slot_key)
            chosen.pop(idx)

    backtrack(0, set(), {})
    if not solutions:
        return None

    def score(assignment: list[dict]) -> tuple[int, int, int]:
        branches = [item["branch"] for item in assignment]
        mismatch_types = len({b for b in branches if b != "match"})
        matches = branches.count("match")
        return mismatch_types, int("wrong" in branches), matches

    best = max(score(sol) for sol in solutions)
    finalists = [sol for sol in solutions if score(sol) == best]
    return random.choice(finalists)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[coords[:, 0] + r0, coords[:, 1] + c0] = int(color)


def _candidates_touch(grid: np.ndarray, colors: list[int]) -> bool:
    h, w = grid.shape
    color_set = set(map(int, colors))
    for color in colors:
        for r, c in np.argwhere(grid == int(color)):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w:
                    neighbor = int(grid[nr, nc])
                    if neighbor in color_set and neighbor != int(color):
                        return True
    return False


def _build_grid() -> tuple[np.ndarray, set[int]] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    band_h = random.choice(BAND_HEIGHTS)
    pocket_count = random.randint(*POCKET_COUNT_RANGE)

    chosen = _choose_pockets(band_h, w, pocket_count)
    if chosen is None:
        return None
    pockets, (structure_left, structure_right) = chosen

    top = random.randint(1, 3)
    bottom = top + band_h - 1
    if bottom >= h - 7:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[top : bottom + 1, structure_left : structure_right + 1] = BLUE
    for pocket in pockets:
        coords = pocket["local_coords"] + np.array([top, structure_left])
        grid[coords[:, 0], coords[:, 1]] = BLACK

    pocket_slots = [p["cols"] for p in pockets]
    gap_slots = _gap_slots(w, pocket_slots)

    base_masks = [p["mask"] for p in pockets]
    assignment = _choose_assignment(base_masks, pocket_slots, gap_slots)
    if assignment is None:
        return None

    colors = random.sample(CANDIDATE_COLORS, pocket_count)
    candidate_floor = bottom + MIN_CANDIDATE_GAP_ROWS + 1
    if candidate_floor >= h - 1:
        return None

    planned_matches: set[int] = set()
    for idx, option in enumerate(assignment):
        slot = pocket_slots[option["slot_index"]] if option["slot_kind"] == "pocket" else gap_slots[option["slot_index"]]
        c0_min = slot[0]
        c0_max = slot[1] - option["width"] + 1
        if c0_min > c0_max:
            return None
        c0 = random.randint(c0_min, c0_max)
        r0_max = h - option["height"] - 1
        if candidate_floor > r0_max:
            return None
        r0 = random.randint(candidate_floor, r0_max)
        color = int(colors[idx])
        _paint_mask(grid, option["mask"], r0, c0, color)
        if option["branch"] == "match":
            planned_matches.add(color)

    if _candidates_touch(grid, colors):
        return None
    return grid, planned_matches


def _validate_generated(grid: np.ndarray, planned_matches: set[int]) -> bool:
    scene = _extract_scene(grid)
    pockets = scene["pockets"]
    candidates = scene["candidates"]
    if len(pockets) != len(candidates):
        return False
    if len(pockets) < 2:
        return False

    pocket_keys = [p["key"] for p in pockets]
    candidate_keys = [c["key"] for c in candidates]
    if len(set(pocket_keys)) != len(pocket_keys):
        return False
    if len(set(candidate_keys)) != len(candidate_keys):
        return False
    if Counter(pocket_keys) != Counter(candidate_keys):
        return False

    actual_matches = _matched_candidate_colors(grid)
    if actual_matches != planned_matches:
        return False
    if not actual_matches or len(actual_matches) == len(candidates):
        return False

    expected = grid.copy()
    for candidate in candidates:
        if int(candidate["color"]) in planned_matches:
            expected[candidate["coords"][:, 0], candidate["coords"][:, 1]] = RED
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_grid()
        if built is None:
            continue
        grid, planned_matches = built
        if _validate_generated(grid, planned_matches):
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
