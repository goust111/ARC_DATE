import random
from collections import Counter

import numpy as np


BG = 0
BRACKET = 5
SOURCE = 8
UNIQUE = 2
REPEATED = 3

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    out[:, 0] -= int(out[:, 0].min())
    out[:, 1] -= int(out[:, 1].min())
    return out


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, ...], ...]:
    norm = _normalize_coords(coords)
    r0, c0, r1, c1 = _bbox(norm)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    mask[norm[:, 0], norm[:, 1]] = 1
    return tuple(tuple(int(x) for x in row) for row in mask)


def _coords_from_mask(mask: np.ndarray, r0: int = 0, c0: int = 0) -> np.ndarray:
    coords = np.argwhere(mask.astype(bool))
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords.astype(int)


def _translate(coords: np.ndarray, dr: int = 0, dc: int = 0) -> np.ndarray:
    moved = np.asarray(coords, dtype=int).copy()
    moved[:, 0] += int(dr)
    moved[:, 1] += int(dc)
    return moved


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"coords": coords, "bbox": _bbox(coords), "area": int(len(cells)), "color": int(color)})
    return comps


def _find_bracket(grid: np.ndarray) -> dict:
    h, w = grid.shape
    candidates = []
    for comp in _components_of_color(grid, BRACKET):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 != h - 1 or r1 - r0 + 1 != 2:
            continue
        width = c1 - c0 + 1
        if not (4 <= width <= 6):
            continue
        patch = grid[r0:r1 + 1, c0:c1 + 1]
        top_ok = np.all(patch[0, 1:-1] == BG) and int(patch[0, 0]) == BRACKET and int(patch[0, -1]) == BRACKET
        bottom_ok = np.all(patch[1] == BRACKET)
        if top_ok and bottom_ok:
            candidates.append(
                {
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "width": width,
                    "interior_left": c0 + 1,
                    "interior_right": c1 - 1,
                }
            )
    assert len(candidates) == 1, "expected exactly one bottom U-bracket"
    return candidates[0]


def _needs_reroute(coords: np.ndarray, bracket: dict) -> bool:
    il = int(bracket["interior_left"])
    ir = int(bracket["interior_right"])
    return bool(np.any((coords[:, 1] >= il) & (coords[:, 1] <= ir)))


def _reroute_repeat(coords: np.ndarray, bracket: dict, grid_width: int) -> tuple[np.ndarray, str, bool]:
    coords = np.asarray(coords, dtype=int)
    _r0, c0, _r1, c1 = _bbox(coords)
    obj_w = int(c1 - c0 + 1)
    left_target = int(bracket["bbox"][1]) - obj_w
    right_target = int(bracket["bbox"][3]) + 1
    left_dist = abs(c0 - left_target) if left_target >= 0 else 10**9
    right_dist = abs(c0 - right_target) if right_target + obj_w <= grid_width else 10**9
    assert min(left_dist, right_dist) < 10**9, "reroute target must fit the grid"
    tie = left_dist == right_dist
    if left_dist <= right_dist:
        return _translate(coords, dc=left_target - c0), "left", tie
    return _translate(coords, dc=right_target - c0), "right", False


def _drop_until_blocked(coords: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    moved = np.asarray(coords, dtype=int).copy()
    h, w = occupied.shape
    while True:
        nxt = _translate(moved, dr=1)
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= h) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= w):
            return moved
        if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
            return moved
        moved = nxt


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor teal objects by shape frequency, then drop only the singleton shape into the bracket channel.

    Input structure:
    - Black background contains one grey U-shaped bracket on the bottom edge, open upward.
    - Above the bracket are 4-6 separate teal 4-connected objects, all using the same source color.
    - Exactly one normalized shape occurs once; at least one other normalized shape occurs multiple times.

    Rule:
    - Extract each teal object as a 4-connected component and normalize its bbox mask.
    - Recolor the singleton-shape object red and all repeated-shape objects green.
    - Process objects from lower input bbox bottom to higher, breaking ties by smaller left column.
    - The red singleton keeps its columns and falls into the bracket interior until blocked by the bracket floor or an already placed object.
    - Each green repeated object falls toward the bottom, but if any of its columns lie in the bracket interior span, it is shifted to the nearest side just outside the bracket before dropping; ties choose the left side.
    - Clear the original teal pixels, keep the grey bracket unchanged, and show only the landed recolored objects.

    Key insight:
    - Shape-frequency classification is the control signal for two coupled consequences at once: color and access to the bracket channel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bracket = _find_bracket(g)
    comps = _components_of_color(g, SOURCE)
    assert 4 <= len(comps) <= 6, "expected 4-6 teal objects"

    shape_keys = [_shape_key(comp["coords"]) for comp in comps]
    counts = Counter(shape_keys)
    unique_indices = [i for i, key in enumerate(shape_keys) if counts[key] == 1]
    assert len(unique_indices) == 1, "expected exactly one unique shape"
    assert any(v > 1 for v in counts.values()), "expected at least one repeated shape"

    il = int(bracket["interior_left"])
    ir = int(bracket["interior_right"])
    objects = []
    for idx, comp in enumerate(comps):
        kind = "unique" if idx in unique_indices else "repeat"
        color = UNIQUE if kind == "unique" else REPEATED
        objects.append(
            {
                "coords": comp["coords"].copy(),
                "bbox": comp["bbox"],
                "kind": kind,
                "color": color,
            }
        )

    objects.sort(key=lambda obj: (-int(obj["bbox"][2]), int(obj["bbox"][1])))

    out = np.zeros_like(g)
    occupied = np.zeros_like(g, dtype=bool)
    _paint(out, bracket["coords"], BRACKET)
    occupied[bracket["coords"][:, 0], bracket["coords"][:, 1]] = True

    for obj in objects:
        coords = obj["coords"]
        if obj["kind"] == "unique":
            _, c0, _, c1 = _bbox(coords)
            assert il <= c0 and c1 <= ir, "unique object must fit inside the channel columns"
            placed = _drop_until_blocked(coords, occupied)
        else:
            placed = coords
            if _needs_reroute(placed, bracket):
                placed, _side, _tie = _reroute_repeat(placed, bracket, g.shape[1])
            placed = _drop_until_blocked(placed, occupied)
            assert not _needs_reroute(placed, bracket), "repeated objects must end outside the channel"
        assert not np.any(occupied[placed[:, 0], placed[:, 1]])
        _paint(out, placed, obj["color"])
        occupied[placed[:, 0], placed[:, 1]] = True

    return out


def _random_polyomino(
    rng: random.Random,
    *,
    min_cells: int = 3,
    max_cells: int = 6,
    max_h: int = 4,
    max_w: int = 3,
    width_cap: int | None = None,
) -> np.ndarray:
    for _ in range(120):
        target = rng.randint(min_cells, max_cells)
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0), (-1, 0), (0, -1)}
        while len(cells) < target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in DIR4:
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt not in cells:
                    frontier.add(nxt)

        coords = np.array(sorted(cells), dtype=int)
        coords = _normalize_coords(coords)
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h > max_h or w > max_w:
            continue
        if width_cap is not None and w > int(width_cap):
            continue
        mask = np.zeros((h, w), dtype=int)
        mask[coords[:, 0], coords[:, 1]] = 1
        if mask.sum() < min_cells or mask.sum() > max_cells:
            continue
        if width_cap is not None and mask.shape[1] > int(width_cap):
            continue
        return mask.astype(bool)
    raise RuntimeError("failed to sample a polyomino")


def _build_bracket(grid: np.ndarray, c0: int, width: int) -> dict:
    h = grid.shape[0]
    r0 = h - 2
    c1 = c0 + width - 1
    grid[r0, c0] = BRACKET
    grid[r0, c1] = BRACKET
    grid[h - 1, c0:c1 + 1] = BRACKET
    coords = _coords_from_mask(grid == BRACKET)
    return {
        "coords": coords,
        "bbox": (r0, c0, h - 1, c1),
        "width": width,
        "interior_left": c0 + 1,
        "interior_right": c1 - 1,
    }


def _occupied_cols(mask: np.ndarray, c0: int) -> set[int]:
    cols = set()
    for rr, cc in np.argwhere(mask):
        cols.add(int(c0 + cc))
    return cols


def _route_metadata(mask: np.ndarray, c0: int, bracket: dict, width: int) -> dict:
    bbox_w = int(mask.shape[1])
    left_target = int(bracket["bbox"][1]) - bbox_w
    right_target = int(bracket["bbox"][3]) + 1
    left_dist = abs(int(c0) - left_target) if left_target >= 0 else 10**9
    right_dist = abs(int(c0) - right_target) if right_target + bbox_w <= width else 10**9
    cols = _occupied_cols(mask, c0)
    overlaps = any(int(bracket["interior_left"]) <= col <= int(bracket["interior_right"]) for col in cols)
    side = None
    if overlaps:
        side = "left" if left_dist <= right_dist else "right"
    return {
        "overlaps": overlaps,
        "left_dist": left_dist,
        "right_dist": right_dist,
        "chosen_side": side,
        "tie": overlaps and left_dist == right_dist,
    }


def _candidate_cols(mode: str, mask: np.ndarray, bracket: dict, width: int) -> list[int]:
    cols = []
    obj_w = int(mask.shape[1])
    b0 = int(bracket["bbox"][1])
    b1 = int(bracket["bbox"][3])
    for c0 in range(0, width - obj_w + 1):
        meta = _route_metadata(mask, c0, bracket, width)
        if mode == "unique":
            if int(bracket["interior_left"]) <= c0 and c0 + obj_w - 1 <= int(bracket["interior_right"]):
                cols.append(c0)
        elif mode == "no_left":
            if c0 + obj_w - 1 < b0:
                cols.append(c0)
        elif mode == "no_right":
            if c0 > b1:
                cols.append(c0)
        elif mode == "reroute_left":
            if meta["overlaps"] and meta["chosen_side"] == "left" and not meta["tie"]:
                cols.append(c0)
        elif mode == "reroute_right":
            if meta["overlaps"] and meta["chosen_side"] == "right":
                cols.append(c0)
        elif mode == "reroute_tie":
            if meta["overlaps"] and meta["chosen_side"] == "left" and meta["tie"]:
                cols.append(c0)
    return cols


def _project_final_span(mode: str, mask: np.ndarray, bracket: dict, c0: int) -> tuple[int, int]:
    obj_w = int(mask.shape[1])
    if mode in {"reroute_left", "reroute_tie"}:
        start = int(bracket["bbox"][1]) - obj_w
        return start, start + obj_w - 1
    if mode == "reroute_right":
        start = int(bracket["bbox"][3]) + 1
        return start, start + obj_w - 1
    return int(c0), int(c0 + obj_w - 1)


def _can_place(mask: np.ndarray, r0: int, c0: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + mh + 1)
    cc1 = min(w, c0 + mw + 1)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False
    return True


def _mark_place(mask: np.ndarray, r0: int, c0: int, occupied: np.ndarray, value: bool) -> None:
    for rr, cc in np.argwhere(mask):
        occupied[r0 + rr, c0 + cc] = bool(value)


def _simulate_specs(shape: tuple[int, int], bracket: dict, specs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    _paint(out, bracket["coords"], BRACKET)
    occupied[bracket["coords"][:, 0], bracket["coords"][:, 1]] = True

    order = []
    for spec in specs:
        coords = _coords_from_mask(spec["mask"], spec["r0"], spec["c0"])
        order.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "kind": spec["kind"],
                "color": UNIQUE if spec["kind"] == "unique" else REPEATED,
                "id": spec["id"],
            }
        )
    order.sort(key=lambda obj: (-int(obj["bbox"][2]), int(obj["bbox"][1])))

    placements = []
    for obj in order:
        coords = obj["coords"]
        routed = False
        route_side = None
        tie = False
        if obj["kind"] == "repeat" and _needs_reroute(coords, bracket):
            coords, route_side, tie = _reroute_repeat(coords, bracket, w)
            routed = True
        placed = _drop_until_blocked(coords, occupied)
        _paint(out, placed, obj["color"])
        occupied[placed[:, 0], placed[:, 1]] = True
        placements.append(
            {
                "id": obj["id"],
                "kind": obj["kind"],
                "coords": placed,
                "bbox": _bbox(placed),
                "routed": routed,
                "route_side": route_side,
                "tie": tie,
            }
        )
    return out, placements


def _count_components(grid: np.ndarray, color: int) -> int:
    return len(_components_of_color(grid, color))


def _validate_instance(grid: np.ndarray, specs: list[dict], bracket: dict, expected: np.ndarray, placements: list[dict], profile: str) -> bool:
    comps = _components_of_color(grid, SOURCE)
    if len(comps) != len(specs):
        return False
    keys = [_shape_key(comp["coords"]) for comp in comps]
    counts = Counter(keys)
    if sum(v == 1 for v in counts.values()) != 1:
        return False
    if not any(v > 1 for v in counts.values()):
        return False

    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    if _count_components(out, UNIQUE) != 1:
        return False
    if _count_components(out, REPEATED) != sum(spec["kind"] == "repeat" for spec in specs):
        return False

    il = int(bracket["interior_left"])
    ir = int(bracket["interior_right"])
    unique_comp = _components_of_color(out, UNIQUE)[0]
    if not (int(unique_comp["bbox"][1]) >= il and int(unique_comp["bbox"][3]) <= ir):
        return False
    if np.any((unique_comp["coords"][:, 1] < il) | (unique_comp["coords"][:, 1] > ir)):
        return False

    repeated_comps = _components_of_color(out, REPEATED)
    if any(np.any((comp["coords"][:, 1] >= il) & (comp["coords"][:, 1] <= ir)) for comp in repeated_comps):
        return False

    repeat_placements = [p for p in placements if p["kind"] == "repeat"]
    routed_count = sum(int(p["routed"]) for p in repeat_placements)
    if profile == "mixed" and not (0 < routed_count < len(repeat_placements)):
        return False
    if profile == "tie_case" and not any(p["tie"] and p["route_side"] == "left" for p in repeat_placements):
        return False
    if profile == "reroute_heavy" and routed_count < 2:
        return False

    return True


def _choose_repeated_counts(rng: random.Random) -> list[int]:
    return rng.choice([[3], [2, 2], [4]])


def _assign_modes(rng: random.Random, repeated_count: int, profile: str) -> list[str]:
    if profile == "mixed":
        base = ["reroute_left", "no_left", "no_right"]
        pool = ["no_left", "no_right", "reroute_right"]
    elif profile == "tie_case":
        base = ["reroute_tie", "no_left", "no_right"]
        pool = ["no_left", "no_right", "reroute_right"]
    else:
        base = ["reroute_left", "reroute_right", "no_left"]
        pool = ["no_left", "no_right"]
    modes = []
    for i in range(repeated_count):
        if i < len(base):
            modes.append(base[i])
        else:
            modes.append(rng.choice(pool))
    rng.shuffle(modes)
    return modes


def _place_specs(
    rng: random.Random,
    h: int,
    w: int,
    bracket: dict,
    specs: list[dict],
) -> bool:
    order = list(range(len(specs)))
    rng.shuffle(order)
    order.sort(key=lambda idx: (0 if specs[idx]["mode"] == "unique" else 1, 0 if specs[idx]["mode"] == "reroute_tie" else 1))

    final_spans = []
    cursor = 0
    for idx in order:
        spec = specs[idx]
        cols = _candidate_cols(spec["mode"], spec["mask"], bracket, w)
        if not cols:
            return False
        rng.shuffle(cols)
        chosen_c0 = None
        for c0 in cols:
            span = _project_final_span(spec["mode"], spec["mask"], bracket, c0)
            if spec["kind"] == "repeat":
                if any(not (span[1] < other[0] - 1 or span[0] > other[1] + 1) for other in final_spans):
                    continue
            chosen_c0 = int(c0)
            if spec["kind"] == "repeat":
                final_spans.append(span)
            break
        if chosen_c0 is None:
            return False
        r0 = cursor + rng.randint(0, 1)
        if r0 + spec["mask"].shape[0] - 1 > h - 3:
            return False
        spec["r0"] = int(r0)
        spec["c0"] = chosen_c0
        cursor = spec["r0"] + spec["mask"].shape[0] + rng.randint(1, 2)

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    w = rng.randint(11, 14)
    h = rng.randint(max(17, w + 1), 24)
    bracket_width = rng.randint(4, 6)
    left_margin_min = 3
    right_margin_min = 3
    possible_c0 = [
        c0
        for c0 in range(left_margin_min, w - bracket_width - right_margin_min + 1)
        if c0 >= 2 and (w - (c0 + bracket_width)) >= 2
    ]
    if not possible_c0:
        return None

    grid = np.zeros((h, w), dtype=int)
    bracket = _build_bracket(grid, rng.choice(possible_c0), bracket_width)
    profile = rng.choice(["mixed", "reroute_heavy", "tie_case"])
    repeated_counts = _choose_repeated_counts(rng)
    repeated_total = sum(repeated_counts)
    total_objects = repeated_total + 1
    if total_objects > 6:
        return None

    key_seen = set()
    specs = []

    try:
        unique_mask = _random_polyomino(
            rng,
            min_cells=3,
            max_cells=6,
            max_h=3,
            max_w=min(4, bracket_width - 2),
            width_cap=bracket_width - 2,
        )
    except RuntimeError:
        return None
    key_seen.add(_shape_key(_coords_from_mask(unique_mask)))
    specs.append({"id": 0, "kind": "unique", "mode": "unique", "mask": unique_mask, "r0": None, "c0": None})

    next_id = 1
    for group_count in repeated_counts:
        for _ in range(80):
            try:
                mask = _random_polyomino(rng, min_cells=3, max_cells=6, max_h=3, max_w=3, width_cap=3)
            except RuntimeError:
                return None
            key = _shape_key(_coords_from_mask(mask))
            if key in key_seen:
                continue
            key_seen.add(key)
            for _copy in range(group_count):
                specs.append({"id": next_id, "kind": "repeat", "mask": mask.copy(), "mode": None, "r0": None, "c0": None})
                next_id += 1
            break
        else:
            return None

    modes = _assign_modes(rng, repeated_total, profile)
    rep_i = 0
    for spec in specs:
        if spec["kind"] == "repeat":
            spec["mode"] = modes[rep_i]
            rep_i += 1

    if not _place_specs(rng, h, w, bracket, specs):
        return None

    for spec in specs:
        grid[_coords_from_mask(spec["mask"], spec["r0"], spec["c0"])[:, 0], _coords_from_mask(spec["mask"], spec["r0"], spec["c0"])[:, 1]] = SOURCE

    expected, placements = _simulate_specs((h, w), bracket, specs)
    if not _validate_instance(grid, specs, bracket, expected, placements, profile):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
