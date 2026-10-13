from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
SPRITE_COLORS = (2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _transform(cells: frozenset[tuple[int, int]], op: int) -> frozenset[tuple[int, int]]:
    pts = []
    for r, c in cells:
        if op == 0:
            rr, cc = r, c
        elif op == 1:
            rr, cc = r, -c
        elif op == 2:
            rr, cc = -r, c
        elif op == 3:
            rr, cc = -r, -c
        elif op == 4:
            rr, cc = c, r
        elif op == 5:
            rr, cc = c, -r
        elif op == 6:
            rr, cc = -c, r
        else:
            rr, cc = -c, -r
        pts.append((rr, cc))
    return _normalize(set(pts))


def _all_variants(cells: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    seen = set()
    out = []
    for op in range(8):
        variant = _transform(cells, op)
        if variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _hole_count(cells: set[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    holes = 0
    for sr in range(h):
        for sc in range(w):
            if not bg[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            touches = False
            while q:
                r, c = q.popleft()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches:
                holes += 1
    return holes


def _extract_components(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], dict[int, dict]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comp_map: dict[tuple[int, int], int] = {}
    comps: dict[int, dict] = {}
    cid = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, BLUE) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                comp_map[(rr, cc)] = cid
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            holes = _hole_count(cells)
            comps[cid] = {
                "id": cid,
                "color": color,
                "coords": cells,
                "holes": holes,
                "reflector": holes % 2 == 0,
            }
            cid += 1
    return comp_map, comps


def _find_seed(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == BLUE)]
    assert 2 <= len(coords) <= 3, "expected a 2-3 pixel blue seed"
    coord_set = set(coords)
    if len({r - c for r, c in coords}) == 1:
        ordered = sorted(coords)
        endpoints = [
            (ordered[0], (1, 1)),
            (ordered[-1], (-1, -1)),
        ]
    elif len({r + c for r, c in coords}) == 1:
        ordered = sorted(coords)
        endpoints = [
            (ordered[0], (1, -1)),
            (ordered[-1], (-1, 1)),
        ]
    else:
        raise AssertionError("blue seed must be a contiguous diagonal")

    assert len(coord_set) == len(ordered)
    for a, b in zip(ordered, ordered[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1, "seed must be 8-connected"

    candidates = []
    for pt, direction in endpoints:
        rr = pt[0] - direction[0]
        cc = pt[1] - direction[1]
        if not _in_bounds(grid.shape, rr, cc):
            candidates.append((pt, direction))

    if len(candidates) == 1:
        start, direction = candidates[0]
        return ordered, start, direction

    def border_score(pt: tuple[int, int]) -> tuple[int, int]:
        r, c = pt
        h, w = grid.shape
        return (
            min(r, c, h - 1 - r, w - 1 - c),
            r + c + abs((h - 1 - r) - (w - 1 - c)),
        )

    start, direction = min(endpoints, key=lambda item: border_score(item[0]))
    return ordered, start, direction


def _trace_ray(
    grid: np.ndarray,
    comp_map: dict[tuple[int, int], int],
    comps: dict[int, dict],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    out = np.array(grid, dtype=int, copy=True)
    r, c = start
    dr, dc = direction
    current_color = BLUE
    seen_states = set()
    interactions = []
    tested_cells = []
    painted_cells = []
    occupied_ray_cells = [start]

    limit = 12 * grid.shape[0] * grid.shape[1]
    steps = 0
    while steps < limit:
        steps += 1
        state = (r, c, dr, dc, current_color)
        assert state not in seen_states, "ray path must not loop"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            break
        tested_cells.append((nr, nc))

        if int(grid[nr, nc]) == BLUE:
            r, c = nr, nc
            occupied_ray_cells.append((r, c))
            continue

        cid = comp_map.get((nr, nc))
        if cid is None:
            out[nr, nc] = current_color
            r, c = nr, nc
            painted_cells.append((r, c))
            occupied_ray_cells.append((r, c))
            continue

        comp = comps[cid]
        if comp["reflector"]:
            hit_h = (r + dr, c) in comp["coords"]
            hit_v = (r, c + dc) in comp["coords"]
            assert hit_h or hit_v, "reflector contact must expose a horizontal or vertical face"
            if hit_h:
                dr = -dr
            if hit_v:
                dc = -dc
            current_color = comp["color"]
            interactions.append(
                {
                    "kind": "R",
                    "cid": cid,
                    "coord": (nr, nc),
                    "color": comp["color"],
                    "hit_h": hit_h,
                    "hit_v": hit_v,
                }
            )
            continue

        out[nr, nc] = current_color
        r, c = nr, nc
        painted_cells.append((r, c))
        occupied_ray_cells.append((r, c))
        interactions.append(
            {
                "kind": "T",
                "cid": cid,
                "coord": (nr, nc),
                "color": comp["color"],
            }
        )

    return out, {
        "interactions": interactions,
        "tested_cells": tested_cells,
        "painted_cells": painted_cells,
        "ray_cells": occupied_ray_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a diagonal ray so that odd-hole sprites transmit and even-hole sprites reflect while passing along reflector color.

    Input structure:
    - Black background with one short blue diagonal seed near a corner.
    - Several non-blue single-color 4-connected sprites are scattered away from one another.
    - Each sprite may contain enclosed black holes inside its own bounding box.

    Rule:
    - Count the enclosed holes of every non-blue sprite; odd-hole sprites are transmissive and even-hole sprites are reflective.
    - Start from the seed's outer endpoint and follow the seed's diagonal direction through the grid.
    - Background cells are painted with the current ray color; transmissive sprite cells are overwritten without changing direction.
    - When the next cell belongs to a reflective sprite, do not paint that cell; flip the row sign on a horizontal-face hit and the column sign on a vertical-face hit, then update the current ray color to the reflector's color.
    - Stop when the ray leaves the grid.

    Key insight:
    - The ray's behavior is controlled by the hidden topology of each sprite, so counting holes is necessary before the geometric ray tracing can be resolved.
    """
    g = np.asarray(grid, dtype=int)
    _seed_coords, start, direction = _find_seed(g)
    comp_map, comps = _extract_components(g)
    out, _meta = _trace_ray(g, comp_map, comps, start, direction)
    return out


def _template_cells(rows: list[str]) -> frozenset[tuple[int, int]]:
    cells = set()
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "X":
                cells.add((r, c))
    return _normalize(cells)


_RAW_TEMPLATES = [
    ("solid3", _template_cells(["XXX", "XXX", "XXX"])),
    ("solid3_notch", _template_cells(["XXX", "XXX", "XX."])),
    ("ring3", _template_cells(["XXX", "X.X", "XXX"])),
    ("ring3_tail", _template_cells(["XXX.", "X.XX", "XXX."])),
    ("frame35", _template_cells(["XXXXX", "X...X", "XXXXX"])),
    ("double35", _template_cells(["XXXXX", "X.X.X", "XXXXX"])),
]

TEMPLATES: list[dict] = []
for name, cells in _RAW_TEMPLATES:
    hole_count = _hole_count(set(cells))
    TEMPLATES.append(
        {
            "name": name,
            "holes": hole_count,
            "reflector": hole_count % 2 == 0,
            "variants": _all_variants(cells),
        }
    )


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _place_absolute(
    variant: frozenset[tuple[int, int]],
    anchor_local: tuple[int, int],
    anchor_abs: tuple[int, int],
) -> set[tuple[int, int]]:
    dr = anchor_abs[0] - anchor_local[0]
    dc = anchor_abs[1] - anchor_local[1]
    return {(r + dr, c + dc) for r, c in variant}


def _path_clear_from(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    r, c = start
    dr, dc = direction
    while _in_bounds(shape, r, c):
        if (r, c) in cells:
            return False
        r += dr
        c += dc
    return True


def _choose_shape_for_event(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    event_kind: str,
    axis: str | None,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    r, c = current
    dr, dc = direction
    anchor = (r + dr, c + dc)
    if not _in_bounds(shape, anchor[0], anchor[1]):
        return None

    candidates = []
    want_reflector = event_kind == "R"
    for template in TEMPLATES:
        if template["reflector"] != want_reflector:
            continue
        for variant in template["variants"]:
            for local_anchor in variant:
                cells = _place_absolute(variant, local_anchor, anchor)
                if any(not _in_bounds(shape, rr, cc) for rr, cc in cells):
                    continue
                if current in cells or cells & forbidden:
                    continue
                if event_kind == "T":
                    if not _path_clear_from(
                        cells,
                        (anchor[0] + direction[0], anchor[1] + direction[1]),
                        direction,
                        shape,
                    ):
                        continue
                else:
                    if axis == "H":
                        if (r + dr, c) not in cells or (r, c + dc) in cells:
                            continue
                        new_dir = (-dr, dc)
                    else:
                        if (r, c + dc) not in cells or (r + dr, c) in cells:
                            continue
                        new_dir = (dr, -dc)
                    if not _path_clear_from(cells, (r + new_dir[0], c + new_dir[1]), new_dir, shape):
                        continue
                candidates.append(cells)

    if not candidates:
        return None
    return random.choice(candidates)


def _ray_distance_to_exit(current: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = current
    dr, dc = direction
    dist = 0
    while _in_bounds(shape, r + dr, c + dc):
        dist += 1
        r += dr
        c += dc
    return dist


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _random_corner_and_dir(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        return (0, 0), (1, 1)
    if corner == "tr":
        return (0, w - 1), (1, -1)
    if corner == "bl":
        return (h - 1, 0), (-1, 1)
    return (h - 1, w - 1), (-1, -1)


def _build_event_grid() -> tuple[np.ndarray, dict] | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)

    outer, direction = _random_corner_and_dir(h, w)
    seed_len = random.randint(2, 3)
    seed_cells = {(outer[0] + i * direction[0], outer[1] + i * direction[1]) for i in range(seed_len)}
    if any(not _in_bounds((h, w), r, c) for r, c in seed_cells):
        return None
    _paint_cells(grid, seed_cells, BLUE)

    current = (outer[0] + (seed_len - 1) * direction[0], outer[1] + (seed_len - 1) * direction[1])
    event_patterns = [
        ["T", "R", "T"],
        ["R", "T", "T"],
        ["T", "R", "T", "R"],
        ["R", "T", "R", "T"],
    ]
    events = random.choice(event_patterns)
    sprite_count_target = random.randint(4, 6)
    colors = random.sample(SPRITE_COLORS, sprite_count_target)
    used_color_idx = 0
    occupied = set(seed_cells)
    forbidden = _expanded(occupied, (h, w), radius=1)
    sprite_cells = []
    planned = []

    for idx, event in enumerate(events):
        advance = random.randint(2, 5)
        for _ in range(advance):
            nr, nc = current[0] + direction[0], current[1] + direction[1]
            if not _in_bounds((h, w), nr, nc):
                return None
            current = (nr, nc)

        if event == "T":
            cells = _choose_shape_for_event((h, w), current, direction, "T", None, occupied, forbidden)
            if cells is None:
                return None
            current = (current[0] + direction[0], current[1] + direction[1])
            planned.append(("T", None))
        else:
            axes = ["H", "V"]
            random.shuffle(axes)
            cells = None
            chosen_axis = None
            for axis in axes:
                trial = _choose_shape_for_event((h, w), current, direction, "R", axis, occupied, forbidden)
                if trial is not None:
                    cells = trial
                    chosen_axis = axis
                    break
            if cells is None:
                return None
            if chosen_axis == "H":
                direction = (-direction[0], direction[1])
            else:
                direction = (direction[0], -direction[1])
            planned.append(("R", chosen_axis))

        color = colors[used_color_idx]
        used_color_idx += 1
        _paint_cells(grid, cells, color)
        sprite_cells.append(cells)
        occupied |= cells
        forbidden |= _expanded(cells, (h, w), radius=1)

    if _ray_distance_to_exit(current, direction, (h, w)) < 3:
        return None

    comp_map, comps = _extract_components(grid)
    _seed, start, traced_dir = _find_seed(grid)
    _, meta = _trace_ray(grid, comp_map, comps, start, traced_dir)
    interactions = meta["interactions"]
    if len(interactions) < 3:
        return None
    if not any(x["kind"] == "R" for x in interactions):
        return None
    if not any(x["kind"] == "T" for x in interactions):
        return None
    first_reflect = next(i for i, x in enumerate(interactions) if x["kind"] == "R")
    if not any(x["kind"] == "T" for x in interactions[first_reflect + 1 :]):
        return None

    avoid = _expanded(set(meta["tested_cells"]) | set(meta["ray_cells"]), (h, w), radius=1)
    while used_color_idx < sprite_count_target:
        template = random.choice(TEMPLATES)
        variant = random.choice(template["variants"])
        tries = 50
        placed = None
        while tries > 0:
            tries -= 1
            r0 = random.randint(0, h - 1)
            c0 = random.randint(0, w - 1)
            cells = {(r + r0, c + c0) for r, c in variant}
            if any(not _in_bounds((h, w), rr, cc) for rr, cc in cells):
                continue
            if cells & forbidden or cells & avoid:
                continue
            placed = cells
            break
        if placed is None:
            return None
        _paint_cells(grid, placed, colors[used_color_idx])
        used_color_idx += 1
        occupied |= placed
        forbidden |= _expanded(placed, (h, w), radius=1)

    return grid, {"planned": planned}


def _component_gap_ok(comps: dict[int, dict]) -> bool:
    comp_list = list(comps.values())
    for i in range(len(comp_list)):
        a = comp_list[i]["coords"]
        for j in range(i + 1, len(comp_list)):
            b = comp_list[j]["coords"]
            best = min(max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in a for r2, c2 in b)
            if best < 2:
                return False
    return True


def _validate_example(grid: np.ndarray) -> bool:
    try:
        seed, start, direction = _find_seed(grid)
        if len(seed) not in (2, 3):
            return False
        comp_map, comps = _extract_components(grid)
        if not (4 <= len(comps) <= 6):
            return False
        if not _component_gap_ok(comps):
            return False

        out, meta = _trace_ray(grid, comp_map, comps, start, direction)
        interactions = meta["interactions"]
        if len({item["cid"] for item in interactions}) < 2:
            return False
        if not any(item["kind"] == "R" for item in interactions):
            return False
        if not any(item["kind"] == "T" for item in interactions):
            return False
        first_reflect = next(i for i, item in enumerate(interactions) if item["kind"] == "R")
        if not any(item["kind"] == "T" for item in interactions[first_reflect + 1 :]):
            return False
        for item in interactions:
            if item["kind"] == "R" and item["hit_h"] == item["hit_v"]:
                return False
        if len(meta["painted_cells"]) < 8:
            return False
        if np.array_equal(out, grid):
            return False

        newly_painted_bg = sum(1 for r, c in meta["painted_cells"] if int(grid[r, c]) == BG)
        overwritten_sprite = sum(1 for r, c in meta["painted_cells"] if int(grid[r, c]) not in (BG, BLUE))
        if newly_painted_bg < 5 or overwritten_sprite < 1:
            return False

        ray_colors = {int(out[r, c]) for r, c in meta["painted_cells"]}
        if len(ray_colors) < 2:
            return False

        if not np.array_equal(main(grid), out):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_event_grid()
        if built is None:
            continue
        grid, _meta = built
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid topological-ray example")
