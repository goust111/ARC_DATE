import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_slide_swallow",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BG = 4
INPUT_FG = 1
LEFT_COLOR = 2
RIGHT_COLOR = 3
MAX_ATTEMPTS = 400
GRID_SIDE_RANGE = (15, 19)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _rows_to_mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _line_has_gap(line: np.ndarray) -> bool:
    pos = np.flatnonzero(line)
    return bool(pos.size >= 2 and int(pos[-1] - pos[0] + 1) > int(pos.size))


def _has_boundary_notch(mask: np.ndarray) -> bool:
    return any(
        _line_has_gap(line)
        for line in (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])
    )


def _transform_mask(mask: np.ndarray, rotation: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, rotation)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _template_variants(base_rows: tuple[tuple[str, ...], ...], want_notch: bool) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rows in base_rows:
        base = _rows_to_mask(rows)
        for rotation in range(4):
            for flip_lr in (False, True):
                mask = _transform_mask(base, rotation, flip_lr)
                key = (tuple(mask.shape), mask.astype(np.uint8).tobytes())
                if key in seen:
                    continue
                seen.add(key)
                if _has_boundary_notch(mask) == want_notch:
                    area = int(mask.sum())
                    if 4 <= area <= 12:
                        variants.append(mask)
    return variants


NO_NOTCH_BASES = (
    ("#..", "###"),
    ("###", ".#."),
    ("##.", ".##"),
    ("#..", "##.", ".##"),
    (".#.", "###", ".#."),
    ("###", "#..", "#.."),
    ("####", "#...", "###."),
    ("#...", "####", "..#."),
)

NOTCH_BASES = (
    ("#.#", "###"),
    ("##", ".#", "##"),
    ("#..#", "####"),
    ("#.#", "###", "#.#"),
    ("###", ".#.", "###"),
    ("#...#", "#####"),
    ("#####", "#...#"),
    ("#.#.", "####"),
)

LEFT_TEMPLATES = _template_variants(NO_NOTCH_BASES, want_notch=False)
RIGHT_TEMPLATES = _template_variants(NOTCH_BASES, want_notch=True)


def _template_specs(direction: str) -> list[dict]:
    masks = LEFT_TEMPLATES if direction == "left" else RIGHT_TEMPLATES
    return [
        {
            "direction": direction,
            "mask": mask.copy(),
            "h": int(mask.shape[0]),
            "w": int(mask.shape[1]),
            "area": int(mask.sum()),
        }
        for mask in masks
    ]


LEFT_SPECS = _template_specs("left")
RIGHT_SPECS = _template_specs("right")


def _copy_spec(spec: dict) -> dict:
    return {
        "direction": spec["direction"],
        "mask": spec["mask"].copy(),
        "h": int(spec["h"]),
        "w": int(spec["w"]),
        "area": int(spec["area"]),
    }


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert len(fg_colors) == 1, "expected a single foreground color"
    objects = find_components(
        g,
        bg=bg,
        colors=fg_colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    out = []
    for comp in objects:
        coords = comp["coords"].astype(int)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        direction = "right" if _has_boundary_notch(mask) else "left"
        shift = (g.shape[1] - 1 - c1) if direction == "right" else -c0
        moved = coords.copy()
        moved[:, 1] += int(shift)
        out.append(
            {
                "direction": direction,
                "area": int(comp["area"]),
                "src_bbox": (r0, c0, r1, c1),
                "dst_bbox": (r0, c0 + int(shift), r1, c1 + int(shift)),
                "src_coords": coords,
                "dst_coords": moved,
            }
        )
    return out


def _deleted_indices(objects: list[dict]) -> set[int]:
    deleted = set()
    for i, obj in enumerate(objects):
        for j, other in enumerate(objects):
            if i == j:
                continue
            if int(other["area"]) > int(obj["area"]) and _bbox_contains(other["dst_bbox"], obj["dst_bbox"]):
                deleted.add(i)
                break
    return deleted


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.argwhere(mask).T
    grid[r0 + rr, c0 + cc] = int(color)


def _pair_candidates(direction: str) -> list[tuple[dict, dict]]:
    specs = LEFT_SPECS if direction == "left" else RIGHT_SPECS
    pairs: list[tuple[dict, dict]] = []
    for outer in specs:
        for inner in specs:
            if outer["area"] <= inner["area"]:
                continue
            if outer["h"] < inner["h"] or outer["w"] < inner["w"]:
                continue
            if outer["h"] == inner["h"] and outer["w"] == inner["w"]:
                continue
            pairs.append((_copy_spec(outer), _copy_spec(inner)))
    return pairs


LEFT_PAIR_CANDIDATES = _pair_candidates("left")
RIGHT_PAIR_CANDIDATES = _pair_candidates("right")


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each blue object to the edge chosen by its bbox notch, erase objects swallowed by post-slide bbox containment, and recolor the survivors.

    Input structure:
    - The grid has a yellow background and 4-connected monochromatic foreground objects of one shared color.
    - Some objects have a bbox-edge notch: on at least one bbox boundary row or column, background cells split that boundary line into two foreground runs.
    - The objects start separated, and any bbox containment relation is meant to emerge only after the horizontal slide.

    Rule:
    - Detect each object's notch status from its own bbox boundary.
    - Slide notched objects horizontally until their bbox is flush with the right edge, and slide non-notched objects until their bbox is flush with the left edge.
    - After all slides, erase any object whose new bbox is contained in the new bbox of a larger object.
    - Recolor the surviving right-sliding objects green, recolor the surviving left-sliding objects red, and keep the yellow background unchanged.

    Key insight:
    - The decisive containment test happens after the independent edge slides, so the deletion pattern is created by the motion rather than visible in the initial placement.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    bg = int(get_bg(g))
    deleted = _deleted_indices(objects)

    out = np.full_like(g, bg)
    for index, obj in enumerate(objects):
        if index in deleted:
            continue
        color = RIGHT_COLOR if obj["direction"] == "right" else LEFT_COLOR
        for r, c in obj["dst_coords"]:
            rr = int(r)
            cc = int(c)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "translated pixel left the grid"
            assert int(out[rr, cc]) in (bg, color), "valid inputs should not create conflicting surviving overlaps"
            out[rr, cc] = int(color)
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _choose_pair(direction: str, max_width: int, max_height: int) -> tuple[dict, dict] | None:
    candidates = LEFT_PAIR_CANDIDATES if direction == "left" else RIGHT_PAIR_CANDIDATES
    valid = [
        (_copy_spec(outer), _copy_spec(inner))
        for outer, inner in candidates
        if outer["w"] < max_width
        and inner["w"] < max_width
        and outer["h"] <= max_height
        and inner["h"] <= max_height
        and outer["w"] + inner["w"] + 3 <= max_width
    ]
    if not valid:
        return None
    return random.choice(valid)


def _choose_single(direction: str, max_width: int, max_height: int) -> dict | None:
    specs = LEFT_SPECS if direction == "left" else RIGHT_SPECS
    valid = [
        _copy_spec(spec)
        for spec in specs
        if spec["w"] < max_width and spec["h"] <= max_height
    ]
    if not valid:
        return None
    return random.choice(valid)


def _assign_group_rows(group_heights: list[int], height: int) -> list[int] | None:
    min_total = sum(group_heights) + max(0, len(group_heights) - 1)
    if min_total > height:
        return None
    extra = height - min_total
    gaps = [0] + [1] * (len(group_heights) - 1) + [0]
    add = _random_composition(extra, len(gaps))
    gaps = [g + a for g, a in zip(gaps, add)]
    rows = []
    cur = gaps[0]
    for index, group_height in enumerate(group_heights):
        rows.append(cur)
        cur += int(group_height) + gaps[index + 1]
    return rows


def _initial_col(spec: dict, width: int, shift: int) -> int:
    if spec["direction"] == "left":
        return int(shift)
    return int(width - spec["w"] - shift)


def _pick_pair_shifts(width: int, outer: dict, inner: dict) -> tuple[int, int] | None:
    max_inner = width - inner["w"] - 1
    max_outer = width - outer["w"] - 1
    if max_inner < 1 or max_outer < 1:
        return None
    pairs = []
    for inner_shift in range(1, max_inner + 1):
        min_outer_shift = inner_shift + inner["w"] + 1
        if min_outer_shift > max_outer:
            continue
        for outer_shift in range(min_outer_shift, max_outer + 1):
            pairs.append((inner_shift, outer_shift))
    if not pairs:
        return None
    return random.choice(pairs)


def _pick_single_shift(width: int, spec: dict) -> int | None:
    max_shift = width - spec["w"] - 1
    if max_shift < 1:
        return None
    return random.randint(1, max_shift)


def _build_groups(height: int, width: int) -> list[dict] | None:
    pair_count = random.choice((1, 2))
    if pair_count == 1:
        pair_sides = [random.choice(("left", "right"))]
    else:
        patterns = (("left", "right"), ("left", "left"), ("right", "right"))
        pair_sides = list(random.choices(patterns, weights=(6, 2, 2), k=1)[0])

    min_total = max(4, 2 * pair_count)
    total_objects = random.randint(min_total, 6)
    extras = total_objects - 2 * pair_count

    pair_directions = [side for side in pair_sides for _ in range(2)]
    single_directions: list[str] = []
    if extras > 0 and "left" not in pair_directions:
        single_directions.append("left")
    if len(single_directions) < extras and "right" not in pair_directions:
        single_directions.append("right")
    while len(single_directions) < extras:
        single_directions.append(random.choice(("left", "right")))
    random.shuffle(single_directions)

    groups: list[dict] = []
    for side in pair_sides:
        pair = _choose_pair(side, max_width=width, max_height=5)
        if pair is None:
            return None
        outer, inner = pair
        groups.append({"kind": "pair", "direction": side, "outer": outer, "inner": inner})

    for side in single_directions:
        spec = _choose_single(side, max_width=width, max_height=5)
        if spec is None:
            return None
        groups.append({"kind": "single", "direction": side, "obj": spec})

    random.shuffle(groups)
    return groups


def _place_groups(height: int, width: int) -> tuple[np.ndarray, int] | None:
    groups = _build_groups(height, width)
    if groups is None:
        return None

    group_heights = [
        int(group["outer"]["h"] if group["kind"] == "pair" else group["obj"]["h"])
        for group in groups
    ]
    group_rows = _assign_group_rows(group_heights, height)
    if group_rows is None:
        return None

    objects: list[dict] = []
    for row0, group in zip(group_rows, groups):
        if group["kind"] == "pair":
            outer = _copy_spec(group["outer"])
            inner = _copy_spec(group["inner"])
            offset = random.randint(0, outer["h"] - inner["h"])
            inner_row = int(row0 + offset)
            shifts = _pick_pair_shifts(width, outer, inner)
            if shifts is None:
                return None
            inner_shift, outer_shift = shifts
            outer["r0"] = int(row0)
            inner["r0"] = inner_row
            outer["c0"] = _initial_col(outer, width, outer_shift)
            inner["c0"] = _initial_col(inner, width, inner_shift)
            objects.extend([outer, inner])
        else:
            spec = _copy_spec(group["obj"])
            shift = _pick_single_shift(width, spec)
            if shift is None:
                return None
            spec["r0"] = int(row0)
            spec["c0"] = _initial_col(spec, width, shift)
            objects.append(spec)

    grid = np.full((height, width), BG, dtype=int)
    for spec in objects:
        _paint_mask(grid, spec["mask"], spec["r0"], spec["c0"], INPUT_FG)

    objects_found = _extract_objects(grid)
    if len(objects_found) != len(objects):
        return None
    comps8 = find_components(
        grid,
        bg=BG,
        colors=[INPUT_FG],
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    if len(comps8) != len(objects):
        return None
    if not any(obj["direction"] == "left" for obj in objects_found):
        return None
    if not any(obj["direction"] == "right" for obj in objects_found):
        return None

    initial_boxes = [obj["src_bbox"] for obj in objects_found]
    for i, bbox_i in enumerate(initial_boxes):
        for j, bbox_j in enumerate(initial_boxes):
            if i == j:
                continue
            if _bbox_contains(bbox_i, bbox_j) or _bbox_contains(bbox_j, bbox_i):
                return None

    deleted = _deleted_indices(objects_found)
    if len(deleted) not in (1, 2):
        return None
    survivors = [obj for idx, obj in enumerate(objects_found) if idx not in deleted]
    if not any(obj["direction"] == "left" for obj in survivors):
        return None
    if not any(obj["direction"] == "right" for obj in survivors):
        return None
    return grid, len(deleted)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        height = random.randint(*GRID_SIDE_RANGE)
        width = random.randint(*GRID_SIDE_RANGE)
        placed = _place_groups(height, width)
        if placed is None:
            continue
        grid, deleted_count = placed
        if deleted_count not in (1, 2):
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid slide-swallow ARC input")
