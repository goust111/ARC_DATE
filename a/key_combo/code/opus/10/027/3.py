from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components


BG = 0
MOVABLE_COLORS = (1, 2, 3)
CORNER_MARKER = 4
OBSTACLE = 5
CORNER_NAMES = ("TL", "TR", "BL", "BR")
GEN_ATTEMPTS = 500


_BASE_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _shape_variants(shape: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rotated = np.rot90(shape, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = tuple(tuple(int(v) for v in row) for row in candidate.astype(int))
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
    return variants


_SHAPES_BY_AREA: dict[int, list[np.ndarray]] = {2: [], 3: [], 4: []}
for _base in _BASE_SHAPES:
    for _variant in _shape_variants(_base):
        area = int(_variant.sum())
        if area in _SHAPES_BY_AREA:
            key = tuple(tuple(int(v) for v in row) for row in _variant.astype(int))
            existing = {
                tuple(tuple(int(v) for v in row) for row in arr.astype(int))
                for arr in _SHAPES_BY_AREA[area]
            }
            if key not in existing:
                _SHAPES_BY_AREA[area].append(_variant)


def _corner_position(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - 1
    if corner == "BL":
        return h - 1, 0
    return h - 1, w - 1


def _corner_distances(shape: tuple[int, int], corner: str, r: int, c: int) -> tuple[int, int]:
    h, w = shape
    if corner == "TL":
        return int(r), int(c)
    if corner == "TR":
        return int(r), int(w - 1 - c)
    if corner == "BL":
        return int(h - 1 - r), int(c)
    return int(h - 1 - r), int(w - 1 - c)


def _paint_triangle(out: np.ndarray, corner: str, side: int, color: int) -> None:
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            dr, dc = _corner_distances((h, w), corner, r, c)
            if dr + dc < side:
                out[r, c] = int(color)


def _object_sort_key(obj: dict) -> tuple[int, int, int, int]:
    coords = obj["coords"]
    bottom = int(coords[:, 0].max())
    left = int(coords[:, 1].min())
    top = int(coords[:, 0].min())
    color = int(obj["color"])
    return (-bottom, left, top, color)


def _below_blocked(coords: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    shifted = coords + np.array([1, 0], dtype=int)
    return blocked[shifted[:, 0], shifted[:, 1]]


def _drop_object(coords: np.ndarray, blocked: np.ndarray) -> np.ndarray | None:
    h, _ = blocked.shape
    offset = 0
    while True:
        shifted = coords + np.array([offset + 1, 0], dtype=int)
        if int(shifted[:, 0].max()) >= h:
            return None
        if np.any(blocked[shifted[:, 0], shifted[:, 1]]):
            return coords + np.array([offset, 0], dtype=int)
        offset += 1


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=MOVABLE_COLORS,
        return_format="dict",
    )
    objects = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        area = int(coords.shape[0])
        if 2 <= area <= 4:
            objects.append(
                {
                    "color": int(comp["color"]),
                    "coords": coords,
                    "area": area,
                }
            )
    return sorted(objects, key=_object_sort_key)


def _settled_signature(objects: list[dict]) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    sig = []
    for obj in objects:
        coords = tuple(sorted((int(r), int(c)) for r, c in np.asarray(obj["coords"], dtype=int)))
        sig.append((int(obj["color"]), coords))
    return tuple(sorted(sig))


def _supports_from_previous(final_coords: np.ndarray, landed: list[dict], obstacle_mask: np.ndarray) -> set[int]:
    support_ids: set[int] = set()
    below = final_coords + np.array([1, 0], dtype=int)
    obstacle_hits = obstacle_mask[below[:, 0], below[:, 1]]
    if np.any(obstacle_hits):
        support_ids.add(-1)
    for obj in landed:
        coords_set = obj["coord_set"]
        for r, c in below:
            if (int(r), int(c)) in coords_set:
                support_ids.add(int(obj["id"]))
                break
    return support_ids


def _simulate_landing(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    obstacle_mask = g == OBSTACLE
    objects = _extract_objects(g)
    landed_mask = np.zeros_like(obstacle_mask, dtype=bool)
    landed: list[dict] = []
    for idx, obj in enumerate(objects):
        final_coords = _drop_object(obj["coords"], obstacle_mask | landed_mask)
        if final_coords is None:
            continue
        support_ids = _supports_from_previous(final_coords, landed, obstacle_mask)
        landed_mask[final_coords[:, 0], final_coords[:, 1]] = True
        landed.append(
            {
                "id": idx,
                "color": int(obj["color"]),
                "coords": final_coords,
                "coord_set": {(int(r), int(c)) for r, c in final_coords},
                "supports": support_ids,
            }
        )

    assert landed, "expected at least one landed object"
    color_counts = Counter(int(obj["color"]) for obj in landed)
    most_common = color_counts.most_common()
    assert len(most_common) >= 1
    if len(most_common) > 1:
        assert int(most_common[0][1]) > int(most_common[1][1]), "majority color must be unique"
    majority_color = int(most_common[0][0])
    return {
        "objects": objects,
        "landed": landed,
        "count": int(len(landed)),
        "majority_color": majority_color,
        "color_counts": color_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the movable objects, count the landed pieces, and replace the whole scene with a corner triangle.

    Input structure:
    - The background is black, with exactly one yellow corner marker and grey obstacle pixels confined to the bottom 1-2 rows.
    - The movable pieces are 4-connected monochrome objects in blue, red, or green; each object has 2-4 pixels.
    - At least two movable colors are present, and one color occurs on strictly more objects than any other color.

    Rule:
    - Process movable objects in descending order of their current bottom row. Each object falls straight down until moving one more row would place any of its pixels directly above a grey obstacle pixel or an already-landed object pixel.
    - Count the landed objects; that count N is the side length of the output right triangle.
    - Take the landed-object majority color by object count, erase every movable pixel and every grey obstacle, and paint the corner triangle where the row-distance plus column-distance from the marked corner is less than N.

    Key insight:
    - Gravity is not cosmetic: the landed-object set supplies both output parameters at once, with object count determining the triangle size and landed-color majority determining the fill color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    corner_hits = [(corner, _corner_position((h, w), corner)) for corner in CORNER_NAMES if int(g[_corner_position((h, w), corner)]) == CORNER_MARKER]
    assert len(corner_hits) == 1, "expected exactly one yellow corner marker"
    corner = corner_hits[0][0]

    analysis = _simulate_landing(g)
    side = int(analysis["count"])
    fill_color = int(analysis["majority_color"])

    out = np.full_like(g, BG)
    _paint_triangle(out, corner, side, fill_color)
    return out


def _sample_counts(n_objects: int) -> tuple[dict[int, int], int]:
    majority = random.choice(MOVABLE_COLORS)
    counts = {color: 0 for color in MOVABLE_COLORS}
    if n_objects <= 3 or random.random() < 0.5:
        other = random.choice([color for color in MOVABLE_COLORS if color != majority])
        maj_count = random.randint(n_objects // 2 + 1, n_objects - 1)
        counts[majority] = maj_count
        counts[other] = n_objects - maj_count
    else:
        others = [color for color in MOVABLE_COLORS if color != majority]
        options = []
        for maj_count in range(2, n_objects - 1):
            rem = n_objects - maj_count
            if rem < 2:
                continue
            for split in range(1, rem):
                counts = {majority: maj_count, others[0]: split, others[1]: rem - split}
                if counts[majority] > max(counts[others[0]], counts[others[1]]):
                    options.append(counts)
        assert options, "expected at least one valid 3-color count split"
        counts = random.choice(options)
        return counts, majority
    assert counts[majority] > max(counts[color] for color in MOVABLE_COLORS if color != majority)
    return counts, majority


def _sample_areas(colors: list[int], majority_color: int, require_pixel_counterexample: bool) -> list[int] | None:
    for _ in range(200):
        areas = []
        for color in colors:
            if require_pixel_counterexample and color == majority_color:
                areas.append(random.choice([2, 2, 3]))
            elif require_pixel_counterexample:
                areas.append(random.choice([3, 4, 4]))
            else:
                areas.append(random.randint(2, 4))
        pixel_counts = Counter()
        for color, area in zip(colors, areas):
            pixel_counts[int(color)] += int(area)
        if require_pixel_counterexample:
            winners = [color for color, total in pixel_counts.items() if total == max(pixel_counts.values())]
            if not (len(winners) == 1 and int(winners[0]) == int(majority_color)):
                return areas
        else:
            return areas
    return None


def _generate_obstacles(h: int, w: int, corner: str) -> np.ndarray | None:
    for _ in range(200):
        bottom_prob = random.uniform(0.45, 0.72)
        bottom = np.random.random(w) < bottom_prob
        if corner == "BL":
            bottom[0] = False
        if corner == "BR":
            bottom[-1] = False
        if int(bottom.sum()) < max(4, w // 3) or int(bottom.sum()) > w - 2:
            continue
        if np.all(bottom):
            continue
        if not np.any(~bottom):
            continue
        upper = bottom & (np.random.random(w) < random.uniform(0.2, 0.55))
        if corner == "BL":
            upper[0] = False
        if corner == "BR":
            upper[-1] = False
        if int(upper.sum()) == 0:
            continue
        if np.array_equal(upper, bottom):
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[h - 1, bottom] = True
        mask[h - 2, upper] = True
        return mask
    return None


def _shape_coords(shape: np.ndarray, left: int) -> np.ndarray:
    coords = np.argwhere(shape)
    return coords + np.array([0, int(left)], dtype=int)


def _has_same_color_contact(coords: np.ndarray, color: int, placed: list[dict]) -> bool:
    coord_set = {(int(r), int(c)) for r, c in coords}
    for prev in placed:
        if int(prev["color"]) != int(color):
            continue
        for r, c in coord_set:
            if (r - 1, c) in prev["coord_set"] or (r + 1, c) in prev["coord_set"] or (r, c - 1) in prev["coord_set"] or (r, c + 1) in prev["coord_set"]:
                return True
    return False


def _enumerate_candidates(
    h: int,
    w: int,
    obstacle_mask: np.ndarray,
    occupied_mask: np.ndarray,
    placed: list[dict],
    color: int,
    area: int,
    require_object_support: bool,
    forbid_object_support: bool,
) -> list[dict]:
    candidates: list[dict] = []
    landed_only = np.zeros((h, w), dtype=bool)
    for prev in placed:
        coords = np.asarray(prev["coords"], dtype=int)
        landed_only[coords[:, 0], coords[:, 1]] = True

    for shape in _SHAPES_BY_AREA[int(area)]:
        sh, sw = shape.shape
        for left in range(w - sw + 1):
            start_coords = _shape_coords(shape, left)
            final_coords = _drop_object(start_coords, occupied_mask)
            if final_coords is None:
                continue
            top = int(final_coords[:, 0].min())
            if top < 2:
                continue
            if _has_same_color_contact(final_coords, color, placed):
                continue
            below = final_coords + np.array([1, 0], dtype=int)
            support_obstacle = bool(np.any(obstacle_mask[below[:, 0], below[:, 1]]))
            support_object = bool(np.any(landed_only[below[:, 0], below[:, 1]]))
            if require_object_support and not support_object:
                continue
            if forbid_object_support and support_object:
                continue
            if not support_obstacle and not support_object:
                continue
            support_ids = set()
            if support_obstacle:
                support_ids.add(-1)
            for prev in placed:
                if any((int(r), int(c)) in prev["coord_set"] for r, c in below):
                    support_ids.add(int(prev["id"]))
            candidates.append(
                {
                    "coords": final_coords,
                    "coord_set": {(int(r), int(c)) for r, c in final_coords},
                    "color": int(color),
                    "shape": shape,
                    "supports": support_ids,
                    "support_object": support_object,
                    "support_obstacle": support_obstacle,
                }
            )
    random.shuffle(candidates)
    return candidates


def _build_settled_scene(
    h: int,
    w: int,
    obstacle_mask: np.ndarray,
    colors: list[int],
    areas: list[int],
    require_stack: bool,
) -> list[dict] | None:
    occupied = obstacle_mask.copy()
    placed: list[dict] = []
    stack_done = False
    stack_deadline = random.randint(1, len(colors) - 1) if require_stack else len(colors)
    for idx, (color, area) in enumerate(zip(colors, areas)):
        must_stack = require_stack and (not stack_done) and idx >= stack_deadline
        forbid_stack = not require_stack
        candidates = _enumerate_candidates(
            h=h,
            w=w,
            obstacle_mask=obstacle_mask,
            occupied_mask=occupied,
            placed=placed,
            color=int(color),
            area=int(area),
            require_object_support=must_stack,
            forbid_object_support=forbid_stack,
        )
        if not candidates and must_stack:
            return None
        if not candidates:
            return None
        choice = random.choice(candidates[: min(12, len(candidates))])
        obj = {
            "id": idx,
            "color": int(color),
            "coords": np.asarray(choice["coords"], dtype=int),
            "coord_set": set(choice["coord_set"]),
            "supports": set(choice["supports"]),
        }
        placed.append(obj)
        coords = obj["coords"]
        occupied[coords[:, 0], coords[:, 1]] = True
        if any(support_id >= 0 for support_id in obj["supports"]):
            stack_done = True
    if require_stack and not stack_done:
        return None
    return placed


def _can_place_input_object(coords: np.ndarray, occupied: np.ndarray, existing: list[dict], color: int) -> bool:
    if np.any(occupied[coords[:, 0], coords[:, 1]]):
        return False
    coord_set = {(int(r), int(c)) for r, c in coords}
    for prev in existing:
        for r, c in coord_set:
            if (r - 1, c) in prev["coord_set"] or (r + 1, c) in prev["coord_set"] or (r, c - 1) in prev["coord_set"] or (r, c + 1) in prev["coord_set"]:
                return False
            if int(prev["color"]) == int(color):
                if (r - 1, c) in prev["coord_set"] or (r + 1, c) in prev["coord_set"] or (r, c - 1) in prev["coord_set"] or (r, c + 1) in prev["coord_set"]:
                    return False
    return True


def _choose_input_positions(
    h: int,
    w: int,
    obstacle_mask: np.ndarray,
    corner: str,
    settled: list[dict],
) -> list[dict] | None:
    corner_cell = _corner_position((h, w), corner)
    occupied = obstacle_mask.copy()
    placed_input: list[dict] = []
    order = sorted(range(len(settled)), key=lambda i: int(settled[i]["coords"][:, 0].max()), reverse=True)
    bottoms: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return any(int(settled[idx]["coords"][:, 0].min() - placed_input_map[idx]["coords"][:, 0].min()) >= 2 for idx in placed_input_map)

        idx = order[pos]
        final_coords = np.asarray(settled[idx]["coords"], dtype=int)
        final_top = int(final_coords[:, 0].min())
        final_bottom = int(final_coords[:, 0].max())
        preferred = []
        relaxed = []
        for lift in range(1, final_top + 1):
            start_coords = final_coords - np.array([lift, 0], dtype=int)
            start_bottom = int(start_coords[:, 0].max())
            if corner_cell in {(int(r), int(c)) for r, c in start_coords}:
                continue
            if not _can_place_input_object(start_coords, occupied, placed_input, int(settled[idx]["color"])):
                continue
            deps_ok = True
            for support_id in settled[idx]["supports"]:
                if support_id < 0 or support_id not in bottoms:
                    continue
                if not (int(bottoms[support_id]) > int(start_bottom)):
                    deps_ok = False
                    break
            if not deps_ok:
                continue
            if pos > 0:
                prev_idx = order[pos - 1]
                if not (int(bottoms[prev_idx]) >= int(start_bottom) + 1):
                    continue
            target_upper = h // 2 + 1
            payload = {"lift": lift, "coords": start_coords, "bottom": start_bottom}
            if start_bottom <= target_upper:
                preferred.append(payload)
            else:
                relaxed.append(payload)
        candidates = preferred if preferred else relaxed
        random.shuffle(candidates)
        for candidate in candidates:
            start_coords = np.asarray(candidate["coords"], dtype=int)
            record = {
                "id": idx,
                "color": int(settled[idx]["color"]),
                "coords": start_coords,
                "coord_set": {(int(r), int(c)) for r, c in start_coords},
            }
            placed_input.append(record)
            placed_input_map[idx] = record
            bottoms[idx] = int(candidate["bottom"])
            occupied[start_coords[:, 0], start_coords[:, 1]] = True
            if backtrack(pos + 1):
                return True
            occupied[start_coords[:, 0], start_coords[:, 1]] = False
            del bottoms[idx]
            del placed_input_map[idx]
            placed_input.pop()
        return False

    placed_input_map: dict[int, dict] = {}
    if not backtrack(0):
        return None
    return [placed_input_map[idx] for idx in range(len(settled))]


def _render_input(h: int, w: int, obstacle_mask: np.ndarray, corner: str, input_objects: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[obstacle_mask] = OBSTACLE
    for obj in input_objects:
        coords = np.asarray(obj["coords"], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = int(obj["color"])
    grid[_corner_position((h, w), corner)] = CORNER_MARKER
    return grid


def _expected_output(shape: tuple[int, int], corner: str, side: int, color: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    _paint_triangle(out, corner, side, color)
    return out


def _validate_semantics(
    inp: np.ndarray,
    expected_settled: list[dict],
    expected_count: int,
    expected_majority: int,
    require_stack: bool,
    require_pixel_counterexample: bool,
) -> bool:
    analysis = _simulate_landing(inp)
    if int(analysis["count"]) != int(expected_count):
        return False
    if int(analysis["majority_color"]) != int(expected_majority):
        return False
    if _settled_signature(analysis["landed"]) != _settled_signature(expected_settled):
        return False
    if require_stack and not any(any(support_id >= 0 for support_id in obj["supports"]) for obj in analysis["landed"]):
        return False
    if require_pixel_counterexample:
        pixel_counts = Counter()
        for obj in analysis["landed"]:
            pixel_counts[int(obj["color"])] += int(len(obj["coords"]))
        winners = [color for color, total in pixel_counts.items() if total == max(pixel_counts.values())]
        if len(winners) == 1 and int(winners[0]) == int(expected_majority):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    corner = random.choice(CORNER_NAMES)
    obstacle_mask = _generate_obstacles(h, w, corner)
    if obstacle_mask is None:
        return None

    n_objects = random.choice([3, 4, 4, 5, 5, 6])
    require_stack = random.random() < 0.5
    require_pixel_counterexample = n_objects >= 4

    counts, majority_color = _sample_counts(n_objects)
    colors = [color for color, count in counts.items() for _ in range(count)]
    random.shuffle(colors)
    areas = _sample_areas(colors, majority_color, require_pixel_counterexample)
    if areas is None:
        return None

    settled = _build_settled_scene(
        h=h,
        w=w,
        obstacle_mask=obstacle_mask,
        colors=colors,
        areas=areas,
        require_stack=require_stack,
    )
    if settled is None:
        return None

    input_objects = _choose_input_positions(h, w, obstacle_mask, corner, settled)
    if input_objects is None:
        return None

    inp = _render_input(h, w, obstacle_mask, corner, input_objects)
    expected = _expected_output((h, w), corner, len(settled), majority_color)
    if not np.array_equal(main(inp), expected):
        return None
    if not _validate_semantics(
        inp=inp,
        expected_settled=settled,
        expected_count=len(settled),
        expected_majority=majority_color,
        require_stack=require_stack,
        require_pixel_counterexample=require_pixel_counterexample,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
