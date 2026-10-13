from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
GREY = 5
SEPARATOR_ROW = 12
STAMP_ROW = 14
INPUT_HEIGHT = 18
WIDTH_RANGE = (18, 22)
OBJECT_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
PARTITIONS = [
    (3, 3),
    (2, 2, 2),
    (2, 2, 3),
    (2, 3, 3),
]
PARTITION_WEIGHTS = [1, 2, 3, 3]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


SHAPE_LIBRARY = {
    "tri_l": np.array(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=bool,
    ),
    "tet_l": np.array(
        [
            [1, 0],
            [1, 0],
            [1, 1],
        ],
        dtype=bool,
    ),
    "pent_p": np.array(
        [
            [1, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=bool,
    ),
}


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _all_dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_mask(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()
    for k in range(4):
        rotated = np.rot90(base, k)
        for variant in (rotated, np.fliplr(rotated)):
            tight = _tight_mask(variant)
            key = _mask_key(tight)
            if key not in seen:
                seen.add(key)
                variants.append(tight)
    return variants


def _shape_signature(mask: np.ndarray, *, include_flips: bool = True) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants = []
    rotations = [_tight_mask(np.rot90(base, k)) for k in range(4)]
    pool = rotations
    if include_flips:
        pool = pool + [_tight_mask(np.fliplr(rot)) for rot in rotations]
    for variant in pool:
        variants.append(_mask_key(variant))
    return min(variants)


def _variant_catalog(mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    base = _tight_mask(mask)
    rotation_keys = {_mask_key(_tight_mask(np.rot90(base, k))) for k in range(4)}
    catalog: list[tuple[np.ndarray, bool]] = []
    seen: set[tuple[str, ...]] = set()
    for k in range(4):
        rotated = _tight_mask(np.rot90(base, k))
        key = _mask_key(rotated)
        if key not in seen:
            seen.add(key)
            catalog.append((rotated, False))

        flipped = _tight_mask(np.fliplr(np.rot90(base, k)))
        key = _mask_key(flipped)
        if key not in seen:
            seen.add(key)
            catalog.append((flipped, key not in rotation_keys))
    return catalog


def _extract_components(grid: np.ndarray) -> list[dict]:
    top = np.asarray(grid[:SEPARATOR_ROW, :], dtype=int)
    h, w = top.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(top[r, c]) == BLACK:
                continue
            color = int(top[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(top[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            for rr, cc in coords:
                mask[rr - int(r0), cc - int(c0)] = True
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return components


def _group_components(components: list[dict], *, include_flips: bool = True) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for comp in components:
        grouped[_shape_signature(comp["mask"], include_flips=include_flips)].append(comp)

    groups = []
    for members in grouped.values():
        representative = min(members, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
        groups.append(
            {
                "members": members,
                "count": len(members),
                "representative": representative,
                "area": int(representative["area"]),
            }
        )
    return groups


def _paint_patch(out: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > out.shape[0] or left + w > out.shape[1]:
        raise AssertionError("stamp placement exceeds the output canvas")
    target = out[top : top + h, left : left + w]
    if np.any((target != BLACK) & mask):
        raise AssertionError("stamp placement still overlaps")
    target[mask] = int(color)


def _render_groups(groups: list[dict], width: int) -> tuple[np.ndarray, list[dict]]:
    out = np.full((INPUT_HEIGHT, width), BLACK, dtype=int)
    out[SEPARATOR_ROW, :] = GREY

    placement_info: list[dict] = []
    order = sorted(
        groups,
        key=lambda group: (
            -int(group["count"]),
            -int(group["area"]),
            int(group["representative"]["bbox"][0]),
            int(group["representative"]["bbox"][1]),
        ),
    )

    for group in order:
        representative = group["representative"]
        mask = representative["mask"]
        color = int(representative["color"])
        base_col = (int(group["count"]) - 1) * 5
        col = base_col
        h, w = mask.shape
        while True:
            if col + w > out.shape[1]:
                raise AssertionError("not enough output width for non-overlapping placement")
            target = out[STAMP_ROW : STAMP_ROW + h, col : col + w]
            if not np.any((target != BLACK) & mask):
                break
            col += 1
        _paint_patch(out, mask, color, STAMP_ROW, col)
        placement_info.append(
            {
                "group": group,
                "base_col": base_col,
                "final_col": col,
            }
        )

    return out, placement_info


def _solve(grid: np.ndarray, *, include_flips: bool = True) -> tuple[np.ndarray, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    if g.shape[0] <= STAMP_ROW:
        raise AssertionError("grid is too short for the output stamping row")

    components = _extract_components(g)
    groups = _group_components(components, include_flips=include_flips)
    if not groups:
        out = np.full_like(g, BLACK)
        out[SEPARATOR_ROW, :] = GREY
        return out, groups, []

    out = np.full_like(g, BLACK)
    out[SEPARATOR_ROW, :] = GREY
    rendered, placement_info = _render_groups(groups, g.shape[1])
    out[:, : rendered.shape[1]] = rendered
    return out, groups, placement_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group the upper sprites by rotation/reflection equivalence, then stamp one representative per group in a count-chosen output column.

    Input structure:
    - The grid has a black background, a full grey separator line on row 12, and a blank lower output band.
    - Above the separator sit 6 to 8 small 4-connected monochrome sprites, each in its own color.
    - Each sprite fits in a tight box no larger than 3x3, while positions and orientations are arbitrary.

    Rule:
    - Extract every upper sprite and normalize its binary mask inside its tight bounding box.
    - Group sprites whose masks match under any rotation or horizontal/vertical reflection.
    - For each group, count its members, choose the representative whose bounding-box top-left corner is lexicographically smallest, and keep that representative's original color and orientation.
    - Erase the upper sprites, keep the grey separator, and stamp each representative into the lower band with top row fixed at 14 and base column `(count - 1) * 5`.
    - Place larger groups first; if two groups have the same size, place the one with larger representative area first. Any later stamp that would overlap is shifted right until the overlap disappears.

    Key insight:
    - The lower layout is determined by three linked steps: dihedral shape grouping decides which sprites belong together, group size picks the target column, and the largest-first non-overlap rule resolves collisions between groups that share the same count column.
    """
    out, _groups, _placement_info = _solve(grid, include_flips=True)
    return out


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(mask)
    }


def _place_object(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    mask: np.ndarray,
    color: int,
    row_range: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = mask.shape
    candidates = []
    min_row, max_row = row_range
    for top in range(min_row, max_row + 1):
        if top + h > SEPARATOR_ROW:
            continue
        for left in range(0, grid.shape[1] - w + 1):
            cells = _mask_cells(mask, top, left)
            if cells & forbidden:
                continue
            candidates.append((top, left, cells))
    if not candidates:
        return None

    random.shuffle(candidates)
    for top, left, cells in candidates:
        if cells & occupied:
            continue
        for r, c in cells:
            grid[r, c] = int(color)
        occupied |= cells
        forbidden |= _dilate_cells(cells, (SEPARATOR_ROW, grid.shape[1]))
        return top, left
    return None


def _choose_partition() -> tuple[int, ...]:
    return random.choices(PARTITIONS, weights=PARTITION_WEIGHTS, k=1)[0]


def _group_shape_names(partition: tuple[int, ...]) -> list[str]:
    if len(partition) == 2:
        return random.sample(list(SHAPE_LIBRARY), 2)
    return random.sample(list(SHAPE_LIBRARY), 3)


def _build_group_members(
    shape_name: str,
    size: int,
    force_reflection: bool,
) -> list[np.ndarray]:
    canonical = SHAPE_LIBRARY[shape_name]
    catalog = _variant_catalog(canonical)
    rotation_variants = [variant for variant, reflected in catalog if not reflected]
    reflection_variants = [variant for variant, reflected in catalog if reflected]

    members = [canonical.copy()]
    nonrep_choices = catalog[:]
    random.shuffle(nonrep_choices)
    if force_reflection and reflection_variants:
        members.append(random.choice(reflection_variants).copy())

    while len(members) < size:
        variant, _reflected = random.choice(nonrep_choices)
        members.append(variant.copy())
    if size >= 2 and all(np.array_equal(member, canonical) for member in members[1:]) and len(rotation_variants) > 1:
        members[1] = random.choice(rotation_variants[1:]).copy()
    return members


def _expected_from_plan(width: int, planned_groups: list[dict]) -> tuple[np.ndarray, list[dict]]:
    groups = []
    for planned in planned_groups:
        members = []
        for member in planned["placed_members"]:
            members.append(
                {
                    "color": member["color"],
                    "bbox": member["bbox"],
                    "mask": member["mask"],
                    "area": int(member["mask"].sum()),
                }
            )
        representative = min(members, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
        groups.append(
            {
                "members": members,
                "count": len(members),
                "representative": representative,
                "area": int(representative["area"]),
            }
        )

    out = np.full((INPUT_HEIGHT, width), BLACK, dtype=int)
    out[SEPARATOR_ROW, :] = GREY
    rendered, placement_info = _render_groups(groups, width)
    out[:, : rendered.shape[1]] = rendered
    return out, placement_info


def _alternative_count_mapping(groups: list[dict], width: int) -> np.ndarray:
    out = np.full((INPUT_HEIGHT, width), BLACK, dtype=int)
    out[SEPARATOR_ROW, :] = GREY
    mapping = {1: 0, 2: 10, 3: 5}
    order = sorted(groups, key=lambda group: (-int(group["count"]), -int(group["area"])))
    for group in order:
        representative = group["representative"]
        mask = representative["mask"]
        color = int(representative["color"])
        col = mapping[int(group["count"])]
        while np.any((out[STAMP_ROW : STAMP_ROW + mask.shape[0], col : col + mask.shape[1]] != BLACK) & mask):
            col += 1
        _paint_patch(out, mask, color, STAMP_ROW, col)
    return out


def _generate_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    grid = np.full((INPUT_HEIGHT, width), BLACK, dtype=int)
    grid[SEPARATOR_ROW, :] = GREY

    partition = _choose_partition()
    shape_names = _group_shape_names(partition)
    num_objects = sum(partition)
    colors = random.sample(OBJECT_COLOR_POOL, num_objects)

    groups_plan = []
    reflection_group_index = None
    chiral_indices = [i for i, name in enumerate(shape_names) if name in {"tet_l", "pent_p"}]
    if chiral_indices:
        reflection_group_index = random.choice(chiral_indices)

    color_cursor = 0
    for idx, (size, shape_name) in enumerate(zip(partition, shape_names)):
        groups_plan.append(
            {
                "size": size,
                "shape_name": shape_name,
                "members": _build_group_members(shape_name, size, idx == reflection_group_index),
                "placed_members": [],
            }
        )

    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    group_order = list(range(len(groups_plan)))
    random.shuffle(group_order)
    for group_index in group_order:
        group = groups_plan[group_index]
        rep_color = colors[color_cursor]
        color_cursor += 1
        rep_mask = group["members"][0]
        rep_loc = _place_object(grid, occupied, forbidden, rep_mask, rep_color, (0, 3))
        if rep_loc is None:
            return None
        rep_top, rep_left = rep_loc
        group["placed_members"].append(
            {
                "color": rep_color,
                "mask": rep_mask.copy(),
                "bbox": (rep_top, rep_left, rep_top + rep_mask.shape[0] - 1, rep_left + rep_mask.shape[1] - 1),
            }
        )

        for member_mask in group["members"][1:]:
            color = colors[color_cursor]
            color_cursor += 1
            loc = _place_object(grid, occupied, forbidden, member_mask, color, (5, 9))
            if loc is None:
                return None
            top, left = loc
            group["placed_members"].append(
                {
                    "color": color,
                    "mask": member_mask.copy(),
                    "bbox": (top, left, top + member_mask.shape[0] - 1, left + member_mask.shape[1] - 1),
                }
            )

    expected, planned_placements = _expected_from_plan(width, groups_plan)
    actual, groups, placement_info = _solve(grid, include_flips=True)
    if not np.array_equal(actual, expected):
        return None

    if len(groups) != len(partition):
        return None
    counts = sorted(group["count"] for group in groups)
    if counts != sorted(partition):
        return None
    if any(not (2 <= group["count"] <= 3) for group in groups):
        return None
    if len({group["representative"]["color"] for group in groups}) != len(groups):
        return None

    same_count_areas: dict[int, list[int]] = defaultdict(list)
    for group in groups:
        same_count_areas[int(group["count"])].append(int(group["area"]))
    if any(len(areas) != len(set(areas)) for areas in same_count_areas.values() if len(areas) > 1):
        return None

    if not any(info["final_col"] > info["base_col"] for info in placement_info):
        return None

    if reflection_group_index is None:
        return None
    rotation_only, rotation_only_groups, _rotation_only_placements = _solve(grid, include_flips=False)
    if np.array_equal(rotation_only, expected):
        return None
    if len(rotation_only_groups) == len(groups):
        return None

    has_mixed_counts = len(set(partition)) > 1
    if has_mixed_counts:
        alt = _alternative_count_mapping(groups, width)
        if np.array_equal(alt, expected):
            return None

    if any(np.array_equal(member["mask"], group["placed_members"][0]["mask"]) for group in groups_plan for member in group["placed_members"][1:]) and len(partition) == 2:
        # Keep 2-group cases from collapsing into identical-orientation exemplars too often.
        return None

    # Cross-check that the independently recorded placement data agrees with the recomputed one.
    if [p["final_col"] for p in planned_placements] != [p["final_col"] for p in placement_info]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
