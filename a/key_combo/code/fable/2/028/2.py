from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 22
GRID_MAX = 30
RECT_COUNT_CHOICES = (2, 3)
DISTRACTOR_COUNT_CHOICES = (1, 2)
RECT_SIZE_MIN = 4
RECT_SIZE_MAX = 8
PLACEMENT_MARGIN = 1
FOREGROUND_COLORS = tuple(range(1, 10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    assert mask.any(), "shape signature requires a non-empty mask"
    signatures: list[tuple[tuple[int, int], ...]] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        coords = np.argwhere(rotated)
        coords = coords - coords.min(axis=0)
        signatures.append(tuple((int(r), int(c)) for r, c in coords))
    return min(signatures)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = _find_components(g, bg)
    masks = [_component_mask(component) for component in components]
    shape_sigs = [_shape_signature(mask) for mask in masks]

    sig_to_indices: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    for idx, sig in enumerate(shape_sigs):
        sig_to_indices[sig].append(idx)

    matches: list[dict] = []
    donor_uses: Counter[int] = Counter()
    for rect_idx, component in enumerate(components):
        gap_mask = ~masks[rect_idx]
        if not gap_mask.any() or not _is_connected(gap_mask):
            continue

        gap_sig = _shape_signature(gap_mask)
        donor_candidates = [idx for idx in sig_to_indices.get(gap_sig, []) if idx != rect_idx]
        if len(donor_candidates) != 1:
            continue

        donor_idx = donor_candidates[0]
        donor_uses[donor_idx] += 1
        matches.append(
            {
                "rect_idx": rect_idx,
                "donor_idx": donor_idx,
                "gap_mask": gap_mask,
                "gap_sig": gap_sig,
            }
        )

    matches = [match for match in matches if donor_uses[match["donor_idx"]] == 1]
    return {
        "bg": bg,
        "components": components,
        "component_masks": masks,
        "shape_sigs": shape_sigs,
        "matches": matches,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each incomplete rectangle's missing patch, using the gap shape to retrieve the donor sprite color.

    Input structure:
    - The grid has a black background, several monochrome rectangles with one connected missing patch,
      and several separate monochrome donor sprites.
    - For each rectangle, the bbox of its surviving pixels is the full rectangle extent, and the black cells
      inside that bbox are exactly the gap.
    - Each gap shape matches exactly one donor sprite up to 90-degree rotation, while extra donor sprites
      do not match any gap.

    Rule:
    - For every incomplete rectangle, infer the full rectangle from its current bbox and read the black gap cells inside it.
    - Find the unique donor sprite whose shape matches that gap under rotation, recolor every gap cell with the donor's color,
      and erase the used donor sprite.
    - Leave all unmatched donor sprites unchanged.

    Key insight:
    - The rectangle's own color is not the completion color; the missing shape is a lookup key that selects which external sprite supplies the fill color.
    """
    analysis = _analyze_scene(np.asarray(grid, dtype=int))
    bg = int(analysis["bg"])
    components = analysis["components"]
    matches = analysis["matches"]

    rect_indices = {match["rect_idx"] for match in matches}
    donor_indices = {match["donor_idx"] for match in matches}
    assert matches, "expected at least one rectangle-donor match"
    assert len(rect_indices) == len(matches), "rectangle matches must be one-to-one"
    assert len(donor_indices) == len(matches), "each donor must be used at most once"
    assert rect_indices.isdisjoint(donor_indices), "donor sprites must not also be rectangles"

    out = np.asarray(grid, dtype=int).copy()
    for match in matches:
        rectangle = components[match["rect_idx"]]
        donor = components[match["donor_idx"]]
        donor_color = int(donor["color"])
        r0, c0, r1, c1 = rectangle["bbox"]
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[match["gap_mask"]] = donor_color

    for donor_idx in donor_indices:
        coords = components[donor_idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    return out


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _shape_entry(name: str, rows: list[str]) -> dict:
    mask = _rows_to_mask(rows)
    return {
        "name": name,
        "mask": mask,
        "sig": _shape_signature(mask),
        "area": int(mask.sum()),
    }


SHAPES = [
    _shape_entry("domino", ["11"]),
    _shape_entry("line3", ["111"]),
    _shape_entry("l3", ["10", "11"]),
    _shape_entry("square2", ["11", "11"]),
    _shape_entry("line4", ["1111"]),
    _shape_entry("t4", ["010", "111"]),
    _shape_entry("z4", ["110", "011"]),
    _shape_entry("u5", ["111", "101"]),
    _shape_entry("p5", ["110", "111"]),
]
SHAPE_BY_NAME = {shape["name"]: shape for shape in SHAPES}
AREA4_NAMES = ("square2", "line4", "t4", "z4")
IRREGULAR_NAMES = {"l3", "t4", "z4", "u5", "p5"}


def _make_rectangle_patch(color: int, gap_mask: np.ndarray, gap_kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    gap_h, gap_w = gap_mask.shape
    if gap_kind == "interior":
        h_low = max(RECT_SIZE_MIN, gap_h + 2)
        w_low = max(RECT_SIZE_MIN, gap_w + 2)
    else:
        h_low = max(RECT_SIZE_MIN, gap_h + 1)
        w_low = max(RECT_SIZE_MIN, gap_w + 1)

    h_high = max(h_low, min(RECT_SIZE_MAX, gap_h + 5))
    w_high = max(w_low, min(RECT_SIZE_MAX, gap_w + 5))

    for _ in range(40):
        h = random.randint(h_low, h_high)
        w = random.randint(w_low, w_high)

        if gap_kind == "interior":
            top = random.randint(1, h - gap_h - 1)
            left = random.randint(1, w - gap_w - 1)
        else:
            corner = random.choice(("tl", "tr", "bl", "br"))
            top = 0 if corner[0] == "t" else h - gap_h
            left = 0 if corner[1] == "l" else w - gap_w

        patch = np.full((h, w), int(color), dtype=int)
        gap_local = np.zeros((h, w), dtype=bool)
        gap_local[top : top + gap_h, left : left + gap_w] = gap_mask
        patch[gap_local] = BACKGROUND

        if _is_connected(patch != BACKGROUND):
            return patch, gap_local
    return None


def _make_sprite_patch(color: int, shape_mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(shape_mask.shape, dtype=int)
    patch[shape_mask] = int(color)
    return patch


def _rotation_changes(mask: np.ndarray, k: int) -> bool:
    return not np.array_equal(mask, np.rot90(mask, k))


def _choose_gap_shapes(rect_count: int) -> list[dict]:
    first = SHAPE_BY_NAME[random.choice(("t4", "z4"))]
    second_pool = [SHAPE_BY_NAME[name] for name in AREA4_NAMES if name != first["name"]]
    second = random.choice(second_pool)
    chosen = [first, second]

    remaining = [shape for shape in SHAPES if shape["name"] not in {first["name"], second["name"]}]
    while len(chosen) < rect_count:
        shape = random.choice(remaining)
        chosen.append(shape)
        remaining = [item for item in remaining if item["name"] != shape["name"]]

    random.shuffle(chosen)
    return chosen


def _choose_distractor_shapes(gap_shapes: list[dict], distractor_count: int) -> list[dict]:
    used = {shape["name"] for shape in gap_shapes}
    distractors: list[dict] = []

    same_area = [SHAPE_BY_NAME[name] for name in AREA4_NAMES if name not in used]
    assert same_area, "expected an unused area-4 distractor"
    first = random.choice(same_area)
    distractors.append(first)
    used.add(first["name"])

    pool = [shape for shape in SHAPES if shape["name"] not in used]
    while len(distractors) < distractor_count:
        shape = random.choice(pool)
        distractors.append(shape)
        used.add(shape["name"])
        pool = [item for item in pool if item["name"] != shape["name"]]
    return distractors


def _choose_rotations(oriented_gaps: list[np.ndarray]) -> list[int]:
    rotations = [random.randint(0, 3) for _ in oriented_gaps]
    if any(_rotation_changes(mask, rotations[idx]) for idx, mask in enumerate(oriented_gaps)):
        return rotations

    options = [
        idx
        for idx, mask in enumerate(oriented_gaps)
        if any(_rotation_changes(mask, k) for k in (1, 2, 3))
    ]
    if options:
        idx = random.choice(options)
        valid = [k for k in (1, 2, 3) if _rotation_changes(oriented_gaps[idx], k)]
        rotations[idx] = random.choice(valid)
    return rotations


def _free_positions(blocked: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = blocked.shape
    patch_h, patch_w = map(int, shape)
    positions: list[tuple[int, int]] = []
    for r in range(h - patch_h + 1):
        for c in range(w - patch_w + 1):
            r0 = max(0, r - PLACEMENT_MARGIN)
            c0 = max(0, c - PLACEMENT_MARGIN)
            r1 = min(h, r + patch_h + PLACEMENT_MARGIN)
            c1 = min(w, c + patch_w + PLACEMENT_MARGIN)
            if not blocked[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _place_objects(objects: list[dict]) -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx]["patch"].shape[0] * objects[idx]["patch"].shape[1],
        reverse=True,
    )

    for idx in order:
        patch = objects[idx]["patch"]
        positions = _free_positions(blocked, patch.shape)
        if not positions:
            return None
        top, left = random.choice(positions)
        mask = patch != BACKGROUND
        grid[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = patch[mask]

        r0 = max(0, top - PLACEMENT_MARGIN)
        c0 = max(0, left - PLACEMENT_MARGIN)
        r1 = min(h, top + patch.shape[0] + PLACEMENT_MARGIN)
        c1 = min(w, left + patch.shape[1] + PLACEMENT_MARGIN)
        blocked[r0:r1, c0:c1] = True

        objects[idx]["top"] = top
        objects[idx]["left"] = left
        coords = np.argwhere(mask)
        objects[idx]["coords"] = coords + np.array([top, left], dtype=int)
    return grid


def _build_expected(input_grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    matched_donors = {obj["pair_id"]: obj for obj in objects if obj["kind"] == "matched_donor"}

    for obj in objects:
        if obj["kind"] != "rectangle":
            continue
        donor = matched_donors[obj["pair_id"]]
        rr, cc = np.where(obj["gap_local"])
        out[obj["top"] + rr, obj["left"] + cc] = donor["color"]

    for donor in matched_donors.values():
        coords = donor["coords"]
        out[coords[:, 0], coords[:, 1]] = BACKGROUND
    return out


def _wrong_keep_donors(input_grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    matched_donors = {obj["pair_id"]: obj for obj in objects if obj["kind"] == "matched_donor"}
    for obj in objects:
        if obj["kind"] != "rectangle":
            continue
        donor = matched_donors[obj["pair_id"]]
        rr, cc = np.where(obj["gap_local"])
        out[obj["top"] + rr, obj["left"] + cc] = donor["color"]
    return out


def _wrong_fill_with_rectangle_color(input_grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    for obj in objects:
        if obj["kind"] != "rectangle":
            continue
        rr, cc = np.where(obj["gap_local"])
        out[obj["top"] + rr, obj["left"] + cc] = obj["color"]
    for obj in objects:
        if obj["kind"] == "matched_donor":
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = BACKGROUND
    return out


def _validate_instance(input_grid: np.ndarray, expected: np.ndarray, objects: list[dict]) -> bool:
    rectangles = [obj for obj in objects if obj["kind"] == "rectangle"]
    donors = [obj for obj in objects if obj["kind"] == "matched_donor"]
    distractors = [obj for obj in objects if obj["kind"] == "distractor"]

    if len(rectangles) < 2 or len(donors) != len(rectangles) or not distractors:
        return False
    if {rect["gap_kind"] for rect in rectangles} != {"corner", "interior"}:
        return False
    if not any(rect["shape_area"] == other["shape_area"] and rect["shape_sig"] != other["shape_sig"]
               for idx, rect in enumerate(rectangles)
               for other in rectangles[idx + 1 :]):
        return False
    if not any(distractor["shape_area"] in {rect["shape_area"] for rect in rectangles} for distractor in distractors):
        return False
    if not any(rect["donor_rotated"] for rect in rectangles):
        return False

    solved = main(input_grid)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(_wrong_keep_donors(input_grid, objects), expected):
        return False
    if np.array_equal(_wrong_fill_with_rectangle_color(input_grid, objects), expected):
        return False

    analysis = _analyze_scene(input_grid)
    matches = analysis["matches"]
    components = analysis["components"]
    if len(matches) != len(rectangles):
        return False

    rect_by_color = {}
    for match in matches:
        rect_component = components[match["rect_idx"]]
        donor_component = components[match["donor_idx"]]
        rect_by_color[int(rect_component["color"])] = {
            "donor_color": int(donor_component["color"]),
            "bbox": tuple(map(int, rect_component["bbox"])),
            "gap_mask": match["gap_mask"],
        }

    for rect in rectangles:
        info = rect_by_color.get(int(rect["color"]))
        if info is None:
            return False
        if info["donor_color"] != int(rect["donor_color"]):
            return False
        bbox = (
            int(rect["top"]),
            int(rect["left"]),
            int(rect["top"] + rect["patch"].shape[0] - 1),
            int(rect["left"] + rect["patch"].shape[1] - 1),
        )
        if info["bbox"] != bbox or not np.array_equal(info["gap_mask"], rect["gap_local"]):
            return False

        rr, cc = np.where(rect["gap_local"])
        if not np.all(input_grid[rect["top"] + rr, rect["left"] + cc] == BACKGROUND):
            return False
        if not np.all(expected[rect["top"] + rr, rect["left"] + cc] == rect["donor_color"]):
            return False

    for donor in donors:
        coords = donor["coords"]
        if not np.all(input_grid[coords[:, 0], coords[:, 1]] == donor["color"]):
            return False
        if not np.all(expected[coords[:, 0], coords[:, 1]] == BACKGROUND):
            return False

    for distractor in distractors:
        coords = distractor["coords"]
        if not np.array_equal(input_grid[coords[:, 0], coords[:, 1]], expected[coords[:, 0], coords[:, 1]]):
            return False
    return True


def _generation_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    distractor_count = random.choice(DISTRACTOR_COUNT_CHOICES)
    gap_shapes = _choose_gap_shapes(rect_count)
    distractor_shapes = _choose_distractor_shapes(gap_shapes, distractor_count)

    object_count = rect_count * 2 + distractor_count
    colors = random.sample(list(FOREGROUND_COLORS), object_count)
    rectangle_colors = colors[:rect_count]
    donor_colors = colors[rect_count : 2 * rect_count]
    distractor_colors = colors[2 * rect_count :]

    gap_kinds = ["interior", "corner"]
    while len(gap_kinds) < rect_count:
        gap_kinds.append(random.choice(("interior", "corner")))
    random.shuffle(gap_kinds)

    oriented_gaps: list[np.ndarray] = []
    for shape in gap_shapes:
        oriented_gaps.append(np.rot90(shape["mask"], random.randint(0, 3)))
    donor_rotations = _choose_rotations(oriented_gaps)

    objects: list[dict] = []
    for pair_id, shape in enumerate(gap_shapes):
        gap_mask = oriented_gaps[pair_id]
        rectangle_data = _make_rectangle_patch(
            rectangle_colors[pair_id], gap_mask, gap_kinds[pair_id]
        )
        if rectangle_data is None:
            return None
        rect_patch, gap_local = rectangle_data
        donor_mask = np.rot90(gap_mask, donor_rotations[pair_id])
        donor_patch = _make_sprite_patch(donor_colors[pair_id], donor_mask)
        objects.append(
            {
                "kind": "rectangle",
                "pair_id": pair_id,
                "patch": rect_patch,
                "color": int(rectangle_colors[pair_id]),
                "donor_color": int(donor_colors[pair_id]),
                "gap_local": gap_local,
                "gap_kind": gap_kinds[pair_id],
                "shape_sig": shape["sig"],
                "shape_area": int(shape["area"]),
                "donor_rotated": bool(_rotation_changes(gap_mask, donor_rotations[pair_id])),
            }
        )
        objects.append(
            {
                "kind": "matched_donor",
                "pair_id": pair_id,
                "patch": donor_patch,
                "color": int(donor_colors[pair_id]),
                "shape_sig": shape["sig"],
                "shape_area": int(shape["area"]),
            }
        )

    for idx, shape in enumerate(distractor_shapes):
        distractor_mask = np.rot90(shape["mask"], random.randint(0, 3))
        objects.append(
            {
                "kind": "distractor",
                "pair_id": rect_count + idx,
                "patch": _make_sprite_patch(distractor_colors[idx], distractor_mask),
                "color": int(distractor_colors[idx]),
                "shape_sig": shape["sig"],
                "shape_area": int(shape["area"]),
            }
        )

    input_grid = _place_objects(objects)
    if input_grid is None:
        return None

    expected = _build_expected(input_grid, objects)
    if np.array_equal(input_grid, expected):
        return None
    if max(input_grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _validate_instance(input_grid, expected, objects):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance for puzzle 2")
