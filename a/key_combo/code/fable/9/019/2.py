from __future__ import annotations

from collections import Counter, deque
from itertools import product

import numpy as np


MARKER_COLOR = 5
CORNERS = ("TL", "TR", "BR", "BL")
CORNER_INDEX = {corner: idx for idx, corner in enumerate(CORNERS)}
PALETTE = tuple(range(10))
RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    exclude_colors: set[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    excluded = set() if exclude_colors is None else set(map(int, exclude_colors))
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if visited[r, c] or color == bg or color in excluded:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds((h, w), nr, nc) or visited[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": bbox_from_coords(coord_arr),
                    "area": int(coord_arr.shape[0]),
                }
            )
    return comps


def count_holes(patch: np.ndarray, color: int) -> int:
    h, w = patch.shape
    obj = patch == int(color)
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if obj[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if obj[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def corner_of_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = map(int, bbox)
    matches = []
    if r0 == 0 and c0 == 0:
        matches.append("TL")
    if r0 == 0 and c1 == w - 1:
        matches.append("TR")
    if r1 == h - 1 and c1 == w - 1:
        matches.append("BR")
    if r1 == h - 1 and c0 == 0:
        matches.append("BL")
    if len(matches) != 1:
        return None
    return matches[0]


def corner_anchor(corner: str, h: int, w: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - obj_w
    if corner == "BR":
        return h - obj_h, w - obj_w
    if corner == "BL":
        return h - obj_h, 0
    raise ValueError(f"unknown corner {corner}")


def advance_corner(corner: str, steps: int) -> str:
    return CORNERS[(CORNER_INDEX[corner] + int(steps)) % 4]


def hole_steps(hole_count: int) -> int:
    steps = int(hole_count) % 4
    if steps == 0:
        raise AssertionError("hole count modulo 4 must be non-zero for movable corner objects")
    return steps


def parse_scene(grid: np.ndarray) -> tuple[int, tuple[int, int], dict[str, dict], list[dict], str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    marker_coords = np.argwhere(g == MARKER_COLOR)
    assert marker_coords.shape == (1, 2), "expected exactly one grey marker"
    marker = tuple(map(int, marker_coords[0]))

    corner_objects: dict[str, dict] = {}
    distractors: list[dict] = []
    for comp in find_components(g, bg=bg, exclude_colors={MARKER_COLOR}):
        bbox = comp["bbox"]
        patch = crop_bbox(g, bbox)
        holes = count_holes(patch, int(comp["color"]))
        r0, c0, r1, c1 = bbox
        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        coords_abs = [tuple(map(int, rc)) for rc in comp["coords"]]
        record = {
            "id": None,
            "color": int(comp["color"]),
            "bbox": bbox,
            "patch": patch,
            "holes": holes,
            "step": None,
            "corner": None,
            "rel_coords": rel_coords,
            "coords_abs": coords_abs,
            "coord_set": set(coords_abs),
            "h": int(r1 - r0 + 1),
            "w": int(c1 - c0 + 1),
        }
        corner = corner_of_bbox(bbox, h, w)
        if corner is None:
            distractors.append(record)
        else:
            assert corner not in corner_objects, "expected at most one corner object per corner"
            record["id"] = corner
            record["corner"] = corner
            record["step"] = hole_steps(holes)
            corner_objects[corner] = record

    assert len(corner_objects) >= 1, "expected at least one corner object"

    adjacent = []
    for corner, obj in corner_objects.items():
        if any((nr, nc) in obj["coord_set"] for nr, nc in neighbors4(*marker)):
            adjacent.append(corner)
    assert len(adjacent) == 1, "marker must select exactly one corner object"
    starter = adjacent[0]

    return bg, marker, corner_objects, distractors, starter


def simulate_chain_plan(
    occupied_corners: list[str],
    starter: str,
    steps_by_object: dict[str, int],
) -> dict | None:
    occupancy = {corner: corner for corner in occupied_corners}
    current_corner = starter
    moving = starter
    order: list[str] = []
    for _ in range(len(occupied_corners) + 2):
        order.append(moving)
        del occupancy[current_corner]
        target = advance_corner(current_corner, steps_by_object[moving])
        evicted = occupancy.get(target)
        occupancy[target] = moving
        if evicted is None:
            return {"order": order, "final": occupancy, "end": target}
        current_corner = target
        moving = evicted
    return None


def apply_chain(grid: np.ndarray, marker: tuple[int, int], corner_objects: dict[str, dict], starter: str) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    bg = get_bg(out)
    out[marker] = bg

    active = {}
    for corner, obj in corner_objects.items():
        active[corner] = {
            **obj,
            "coords_abs": list(obj["coords_abs"]),
            "coord_set": set(obj["coord_set"]),
        }

    current_corner = starter
    moving = active[current_corner]
    for _ in range(len(active) + 2):
        for r, c in moving["coords_abs"]:
            out[r, c] = bg
        del active[current_corner]

        target = advance_corner(current_corner, moving["step"])
        evicted = active.get(target)
        if evicted is not None:
            for r, c in evicted["coords_abs"]:
                out[r, c] = bg
            del active[target]

        dst_r0, dst_c0 = corner_anchor(target, h, w, moving["h"], moving["w"])
        mask = moving["patch"] != bg
        region = out[dst_r0 : dst_r0 + moving["h"], dst_c0 : dst_c0 + moving["w"]]
        region[mask] = moving["patch"][mask]
        new_coords = [(dst_r0 + rr, dst_c0 + cc) for rr, cc in moving["rel_coords"]]
        moving = {
            **moving,
            "corner": target,
            "bbox": (dst_r0, dst_c0, dst_r0 + moving["h"] - 1, dst_c0 + moving["w"] - 1),
            "coords_abs": new_coords,
            "coord_set": set(new_coords),
        }
        active[target] = moving

        if evicted is None:
            return out
        current_corner = target
        moving = evicted

    raise AssertionError("eviction chain did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the grey-marked corner object clockwise by its hole-count program, evicting occupied corners in a chain.

    Input structure:
    - The dominant color is the background; all other colored cells form monochromatic 4-connected objects.
    - Relevant objects are exactly the ones whose bounding boxes touch two adjacent grid borders at a corner.
    - Each corner object may contain fully enclosed background holes, and exactly one grey marker pixel is 4-neighbor adjacent to one corner object.

    Rule:
    - Count the marked corner object's enclosed background holes, take that count modulo 4, and move the object that many clockwise corner steps while preserving its orientation and corner-tight bbox alignment.
    - If the destination corner is occupied by another corner object, remove that occupant and continue the same process from that corner using the evicted object's own hole count; continue until an object lands in an empty corner.
    - Clear every moved source corner to background, remove the grey marker, and leave distractors plus untouched corner objects unchanged.

    Key insight:
    - The grey marker picks the starter, but the final layout is determined by per-object hole counts and the exact eviction order, so the chain must be simulated sequentially rather than as one global shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, marker, corner_objects, _distractors, starter = parse_scene(g)
    assert bg == get_bg(g)
    return apply_chain(g, marker, corner_objects, starter)


def checker_capacity(h: int, w: int) -> int:
    interior_h = h - 2
    interior_w = w - 2
    if interior_h <= 0 or interior_w <= 0:
        return 0
    cells = interior_h * interior_w
    return (cells + 1) // 2


def build_holey_rect_patch(
    h: int,
    w: int,
    color: int,
    hole_count: int,
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    assert checker_capacity(h, w) >= hole_count
    patch = np.full((h, w), int(color), dtype=int)
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    parity_groups = [
        [cell for cell in interior if (cell[0] + cell[1]) % 2 == parity]
        for parity in (0, 1)
    ]
    candidates = max(parity_groups, key=len)
    holes = rng.choice(len(candidates), size=hole_count, replace=False)
    for idx in holes:
        rr, cc = candidates[int(idx)]
        patch[rr, cc] = int(bg)
    return patch


def build_blob_patch(h: int, w: int, color: int, area: int, bg: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.full((h, w), int(bg), dtype=int)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    frontier = [start]
    while len(cells) < area:
        base = frontier[int(rng.integers(0, len(frontier)))]
        options = []
        for nr, nc in neighbors4(*base):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                options.append((nr, nc))
        if options:
            cell = options[int(rng.integers(0, len(options)))]
            cells.add(cell)
            frontier.append(cell)
        else:
            frontier.pop(int(rng.integers(0, len(frontier))))
            if not frontier:
                frontier = list(cells)
    for rr, cc in cells:
        patch[rr, cc] = int(color)
    return patch


def pick_corner_layout(rng: np.random.Generator) -> dict:
    chain_weights = {1: 0.22, 2: 0.36, 3: 0.26, 4: 0.16}
    while True:
        occupied_count = int(rng.choice([2, 3, 4], p=[0.24, 0.46, 0.30]))
        occupied = list(rng.choice(CORNERS, size=occupied_count, replace=False))
        starter = occupied[int(rng.integers(0, occupied_count))]

        candidates = []
        for steps in product((1, 2, 3), repeat=occupied_count):
            step_map = dict(zip(occupied, steps))
            plan = simulate_chain_plan(occupied, starter, step_map)
            if plan is None:
                continue
            order = plan["order"]
            if len(order) == 1 and step_map[starter] == 1:
                continue
            if len(order) > 1 and not any(step_map[obj] > 1 for obj in order):
                continue
            candidates.append((step_map, plan))

        if not candidates:
            continue

        lengths = sorted({len(plan["order"]) for _, plan in candidates})
        weights = np.array([chain_weights[length] for length in lengths], dtype=float)
        weights /= weights.sum()
        desired = int(rng.choice(lengths, p=weights))
        same_length = [(steps, plan) for steps, plan in candidates if len(plan["order"]) == desired]
        step_map, plan = same_length[int(rng.integers(0, len(same_length)))]
        return {
            "occupied": occupied,
            "starter": starter,
            "steps": step_map,
            "chain_order": plan["order"],
            "chain_length": len(plan["order"]),
        }


def choose_hole_count(step: int, rng: np.random.Generator) -> int:
    if rng.random() < 0.45:
        return step + 4
    return step


def choose_corner_objects(
    layout: dict,
    bg: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    corner_objects: dict[str, dict] = {}
    colors = [c for c in PALETTE if c not in {bg, MARKER_COLOR}]
    rng.shuffle(colors)
    for idx, corner in enumerate(layout["occupied"]):
        step = int(layout["steps"][corner])
        hole_count = choose_hole_count(step, rng)
        dims = [
            (h, w)
            for h in range(4, 9)
            for w in range(4, 9)
            if checker_capacity(h, w) >= hole_count
        ]
        h, w = dims[int(rng.integers(0, len(dims)))]
        color = int(colors[idx])
        patch = build_holey_rect_patch(h, w, color, hole_count, bg, rng)
        corner_objects[corner] = {
            "color": color,
            "patch": patch,
            "holes": hole_count,
            "step": step,
            "h": h,
            "w": w,
        }
    return corner_objects


def place_corner_objects(
    grid: np.ndarray,
    corner_objects: dict[str, dict],
) -> dict[str, dict] | None:
    g = grid
    h, w = g.shape
    if "TL" in corner_objects and "TR" in corner_objects:
        if corner_objects["TL"]["w"] + corner_objects["TR"]["w"] > w - 1:
            return None
    if "BL" in corner_objects and "BR" in corner_objects:
        if corner_objects["BL"]["w"] + corner_objects["BR"]["w"] > w - 1:
            return None
    if "TL" in corner_objects and "BL" in corner_objects:
        if corner_objects["TL"]["h"] + corner_objects["BL"]["h"] > h - 1:
            return None
    if "TR" in corner_objects and "BR" in corner_objects:
        if corner_objects["TR"]["h"] + corner_objects["BR"]["h"] > h - 1:
            return None

    placed: dict[str, dict] = {}
    bg = get_bg(g)
    for corner, obj in corner_objects.items():
        r0, c0 = corner_anchor(corner, h, w, obj["h"], obj["w"])
        patch = obj["patch"]
        mask = patch != bg
        region = g[r0 : r0 + obj["h"], c0 : c0 + obj["w"]]
        if np.any(region[mask] != bg):
            return None
        region[mask] = patch[mask]
        coords_abs = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(mask))]
        placed[corner] = {
            **obj,
            "corner": corner,
            "bbox": (r0, c0, r0 + obj["h"] - 1, c0 + obj["w"] - 1),
            "rel_coords": [(int(rr), int(cc)) for rr, cc in zip(*np.where(mask))],
            "coords_abs": coords_abs,
            "coord_set": set(coords_abs),
        }
    return placed


def marker_candidates(
    grid: np.ndarray,
    obj: dict,
    bg: int,
) -> list[tuple[int, int]]:
    g = grid
    candidates = []
    for r, c in obj["coords_abs"]:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(g.shape, nr, nc):
                continue
            if int(g[nr, nc]) != bg:
                continue
            candidates.append((nr, nc))
    uniq = []
    seen = set()
    for cell in candidates:
        if cell not in seen:
            seen.add(cell)
            uniq.append(cell)
    return uniq


def can_place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    bg: int,
    *,
    gap: int = 1,
    top_margin: int = 1,
    bottom_margin: int = 1,
    left_margin: int = 1,
    right_margin: int = 1,
) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    mask = patch != bg
    if top < top_margin or left < left_margin:
        return False
    if top + h > gh - bottom_margin or left + w > gw - right_margin:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(gh, top + h + gap)
    c1 = min(gw, left + w + gap)
    if np.any(grid[r0:r1, c0:c1] != bg):
        return False
    return bool(mask.any())


def stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    region = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    region[mask] = patch[mask]


def add_distractors(
    grid: np.ndarray,
    bg: int,
    rng: np.random.Generator,
    *,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> bool:
    distractor_count = int(rng.integers(2, 5))
    holey_slots = 1 if rng.random() < 0.45 else 0
    colors = [c for c in PALETTE if c not in {bg, MARKER_COLOR}]
    rng.shuffle(colors)

    for idx in range(distractor_count):
        color = int(colors[idx % len(colors)])
        if idx < holey_slots:
            holes = int(rng.choice([1, 2, 3]))
            dims = [
                (h, w)
                for h in range(4, 7)
                for w in range(4, 7)
                if checker_capacity(h, w) >= holes
            ]
            ph, pw = dims[int(rng.integers(0, len(dims)))]
            patch = build_holey_rect_patch(ph, pw, color, holes, bg, rng)
        else:
            ph = int(rng.integers(2, 6))
            pw = int(rng.integers(2, 6))
            area = int(rng.integers(2, min(ph * pw, 7)))
            patch = build_blob_patch(ph, pw, color, area, bg, rng)

        placed = False
        for _ in range(80):
            top_lo = top_margin
            top_hi = grid.shape[0] - bottom_margin - patch.shape[0] + 1
            left_lo = left_margin
            left_hi = grid.shape[1] - right_margin - patch.shape[1] + 1
            if top_lo >= top_hi or left_lo >= left_hi:
                return False
            top = int(rng.integers(top_lo, top_hi))
            left = int(rng.integers(left_lo, left_hi))
            if not can_place_patch(
                grid,
                patch,
                top,
                left,
                bg,
                gap=1,
                top_margin=top_margin,
                bottom_margin=bottom_margin,
                left_margin=left_margin,
                right_margin=right_margin,
            ):
                continue
            stamp_patch(grid, patch, top, left, bg)
            placed = True
            break
        if not placed:
            return False
    return True


def validate_generated(
    grid: np.ndarray,
    *,
    intended_layout: dict,
) -> bool:
    try:
        bg, marker, corner_objects, distractors, starter = parse_scene(grid)
    except AssertionError:
        return False

    if bg == MARKER_COLOR:
        return False
    if len(corner_objects) != len(intended_layout["occupied"]):
        return False
    if len(distractors) < 2:
        return False
    if starter != intended_layout["starter"]:
        return False
    if any(obj["step"] not in (1, 2, 3) for obj in corner_objects.values()):
        return False

    steps = {corner: obj["step"] for corner, obj in corner_objects.items()}
    if steps != intended_layout["steps"]:
        return False

    plan = simulate_chain_plan(list(corner_objects.keys()), starter, steps)
    if plan is None or plan["order"] != intended_layout["chain_order"]:
        return False
    if len(plan["order"]) != intended_layout["chain_length"]:
        return False
    if len(plan["order"]) == 1 and steps[starter] == 1:
        return False

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return False
    if np.sum(out != grid) <= 1:
        return False
    if out[marker] != bg:
        return False

    moved = set(plan["order"])
    for dist in distractors:
        for r, c in dist["coords_abs"]:
            if int(out[r, c]) != int(grid[r, c]):
                return False
    for corner, obj in corner_objects.items():
        if corner not in moved:
            for r, c in obj["coords_abs"]:
                if int(out[r, c]) != obj["color"]:
                    return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = int(RNG.integers(18, 27))
        w = int(RNG.integers(18, 27))
        bg = int(RNG.choice([c for c in PALETTE if c != MARKER_COLOR]))
        grid = np.full((h, w), bg, dtype=int)

        layout = pick_corner_layout(RNG)
        corner_specs = choose_corner_objects(layout, bg, RNG)
        placed = place_corner_objects(grid, corner_specs)
        if placed is None:
            continue

        starter_obj = placed[layout["starter"]]
        candidates = marker_candidates(grid, starter_obj, bg)
        if not candidates:
            continue
        marker = candidates[int(RNG.integers(0, len(candidates)))]
        grid[marker] = MARKER_COLOR

        max_h = max(obj["h"] for obj in placed.values())
        max_w = max(obj["w"] for obj in placed.values())
        if not add_distractors(
            grid,
            bg,
            RNG,
            top_margin=max_h + 1,
            bottom_margin=max_h + 1,
            left_margin=max_w + 1,
            right_margin=max_w + 1,
        ):
            continue

        if validate_generated(grid, intended_layout=layout):
            return grid

    raise RuntimeError("failed to generate a valid chained-corner sample")
