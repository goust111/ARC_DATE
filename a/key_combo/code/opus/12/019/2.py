from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7)
MIN_SIZE = 18
MAX_SIZE = 24
MIN_OBJECT_AREA = 3
MAX_OBJECT_AREA = 6
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_mask(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()
    for k in range(4):
        rotated = np.rot90(base, k)
        for arr in (rotated, np.fliplr(rotated)):
            trimmed = _tight_mask(arr)
            key = _mask_key(trimmed)
            if key in seen:
                continue
            seen.add(key)
            variants.append(trimmed.astype(bool))
    return variants


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(_mask_key(arr) for arr in _dihedral_variants(mask))


def _has_unique_vertical_extremes(mask: np.ndarray) -> bool:
    return bool(mask[0].sum() == 1 and mask[-1].sum() == 1)


def _normalize_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _cell_frontier(cells: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    occupied = set(cells)
    frontier: set[tuple[int, int]] = set()
    for r, c in occupied:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in occupied:
                frontier.add((nr, nc))
    return frontier


def _mask_from_cells(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    h = max(rows) + 1
    w = max(cols) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _build_shape_library() -> tuple[dict[str, object], ...]:
    by_area: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for area in range(1, MAX_OBJECT_AREA):
        next_shapes: set[tuple[tuple[int, int], ...]] = set()
        for shape in by_area[area]:
            for cell in _cell_frontier(shape):
                next_shapes.add(_normalize_cells(shape + (cell,)))
        by_area[area + 1] = next_shapes

    library: dict[tuple[str, ...], list[np.ndarray]] = {}
    for area in range(MIN_OBJECT_AREA, MAX_OBJECT_AREA + 1):
        for cells in by_area[area]:
            mask = _mask_from_cells(cells)
            variants = [arr for arr in _dihedral_variants(mask) if _has_unique_vertical_extremes(arr)]
            if not variants:
                continue
            signature = _shape_signature(mask)
            slot = library.setdefault(signature, [])
            known = {_mask_key(arr) for arr in slot}
            for arr in variants:
                key = _mask_key(arr)
                if key not in known:
                    slot.append(arr)
                    known.add(key)

    items: list[dict[str, object]] = []
    for signature, variants in library.items():
        items.append(
            {
                "signature": signature,
                "variants": tuple(variants),
                "area": int(variants[0].sum()),
            }
        )
    items.sort(key=lambda item: (int(item["area"]), item["signature"]))
    if len(items) < 12:
        raise RuntimeError("shape library is too small for diversified generation")
    return tuple(items)


SHAPE_LIBRARY = _build_shape_library()


def _find_divider_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    if len(rows) != 1:
        raise ValueError("expected exactly one full-width gray divider row")
    return int(rows[0])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    divider_row = _find_divider_row(grid)
    objects: list[dict[str, object]] = []
    next_index = 0
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BACKGROUND, DIVIDER)):
        for coords in _component_coords(grid == color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0 = int(rows.min())
            r1 = int(rows.max())
            c0 = int(cols.min())
            c1 = int(cols.max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rows - r0, cols - c0] = True
            side = "top" if r1 < divider_row else "bottom"
            objects.append(
                {
                    "index": next_index,
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "side": side,
                }
            )
            next_index += 1
    return objects


def _attach_signatures(objects: list[dict[str, object]]) -> dict[tuple[str, ...], list[dict[str, object]]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for obj in objects:
        signature = _shape_signature(np.asarray(obj["mask"], dtype=bool))
        obj["signature"] = signature
        groups[signature].append(obj)
    return groups


def _distance_to_divider(obj: dict[str, object], divider_row: int) -> int:
    r0, _c0, r1, _c1 = obj["bbox"]
    if obj["side"] == "top":
        return int(divider_row - r1)
    return int(r0 - divider_row)


def _pair_roles(
    members: list[dict[str, object]], divider_row: int
) -> tuple[dict[str, object], dict[str, object]]:
    if len(members) != 2:
        raise ValueError("pair roles require exactly two members")
    a, b = members
    da = _distance_to_divider(a, divider_row)
    db = _distance_to_divider(b, divider_row)
    if da < db:
        return a, b
    if db < da:
        return b, a
    ordered = sorted(members, key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["index"]))
    return ordered[0], ordered[1]


def _start_pixel(obj: dict[str, object], toward_divider: bool) -> tuple[int, int]:
    coords = np.asarray(obj["coords"], dtype=int)
    rows = coords[:, 0]
    if obj["side"] == "top":
        target_row = int(rows.max()) if toward_divider else int(rows.min())
    else:
        target_row = int(rows.min()) if toward_divider else int(rows.max())
    options = sorted((int(r), int(c)) for r, c in coords if int(r) == target_row)
    return options[0]


def _paint_vertical_ray(
    out: np.ndarray,
    obj: dict[str, object],
    divider_row: int,
    toward_divider: bool,
) -> None:
    start_r, start_c = _start_pixel(obj, toward_divider)
    color = int(obj["color"])
    if obj["side"] == "top":
        rows = range(start_r, divider_row) if toward_divider else range(start_r, -1, -1)
    else:
        rows = range(start_r, divider_row, -1) if toward_divider else range(start_r, out.shape[0])
    for r in rows:
        out[int(r), start_c] = color


def _render_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    divider_row = _find_divider_row(g)
    objects = _extract_objects(g)
    groups = _attach_signatures(objects)
    out = g.copy()
    ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["index"]))
    for obj in ordered:
        members = groups[obj["signature"]]
        if len(members) != 2:
            continue
        near_obj, far_obj = _pair_roles(members, divider_row)
        _paint_vertical_ray(out, obj, divider_row, toward_divider=(obj["index"] == near_obj["index"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use shape-equivalence classes across a gray divider to decide which sprites fire toward it and which fire away.

    Input structure:
    - A black background contains one full-width gray horizontal divider row.
    - Above and below the divider sit several small monochrome 4-connected sprites, separated from each other and from the divider.
    - Some top/bottom sprite pairs share the same tight mask up to rotation or reflection, while other sprites have no partner.

    Rule:
    - Split all non-black, non-gray pixels into monochrome 4-connected sprites and normalize each sprite to its tight mask.
    - Group sprites by dihedral shape equivalence (four rotations and optional mirror); only groups of exactly two matter.
    - Inside each paired group, the sprite with the smaller distance to the divider draws a vertical ray from its nearest pixel toward the divider and stops just before the gray row, while the farther sprite draws from its farthest pixel away from the divider until the canvas edge.
    - Leave unpaired sprites unchanged, and resolve any ray overlaps by processing sprites in row-major order of their bbox top-left corners.

    Key insight:
    - Color does not encode direction: you must first detect which sprites are shape-equivalent across the divider, then compare their distances to decide attraction versus repulsion.
    """
    return _render_expected(np.asarray(grid, dtype=int))


def _distance_options(side: str, divider_row: int, total_rows: int, mask: np.ndarray) -> list[int]:
    h = int(mask.shape[0])
    if side == "top":
        max_distance = divider_row - (h - 1)
    else:
        max_distance = total_rows - 1 - divider_row - (h - 1)
    return list(range(2, max_distance + 1))


def _pair_distance_choices(
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    height: int,
    divider_row: int,
    near_side: str,
) -> tuple[int, int] | None:
    top_choices = _distance_options("top", divider_row, height, top_mask)
    bottom_choices = _distance_options("bottom", divider_row, height, bottom_mask)
    feasible: list[tuple[int, int]] = []
    for top_d in top_choices:
        for bottom_d in bottom_choices:
            if near_side == "top" and top_d < bottom_d:
                feasible.append((top_d, bottom_d))
            if near_side == "bottom" and bottom_d < top_d:
                feasible.append((top_d, bottom_d))
    if not feasible:
        return None
    random.shuffle(feasible)
    feasible.sort(key=lambda pair: (abs(pair[0] - pair[1]), pair[0] + pair[1]), reverse=True)
    return feasible[0]


def _row_from_distance(side: str, divider_row: int, distance: int, mask: np.ndarray) -> int:
    if side == "top":
        return int(divider_row - distance - (mask.shape[0] - 1))
    return int(divider_row + distance)


def _candidate_cells(mask: np.ndarray, row0: int, col0: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [(int(row0 + r), int(col0 + c)) for r, c in coords]


def _can_place(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _place_side_objects(
    specs: list[dict[str, object]],
    side: str,
    height: int,
    width: int,
    divider_row: int,
) -> bool:
    occupied: set[tuple[int, int]] = set()
    specs.sort(key=lambda spec: (-int(np.asarray(spec["mask"], dtype=bool).sum()), -spec["mask"].shape[1]))
    for spec in specs:
        mask = np.asarray(spec["mask"], dtype=bool)
        if spec["distance_options"] is None:
            options = [int(spec["distance"])]
        else:
            options = list(spec["distance_options"])
            random.shuffle(options)
        placed = False
        for distance in options:
            row0 = _row_from_distance(side, divider_row, int(distance), mask)
            cols = list(range(0, width - mask.shape[1] + 1))
            random.shuffle(cols)
            for col0 in cols:
                cells = _candidate_cells(mask, row0, col0)
                if not _can_place(cells, occupied, height, width):
                    continue
                spec["distance"] = int(distance)
                spec["row0"] = int(row0)
                spec["col0"] = int(col0)
                spec["cells"] = cells
                occupied.update(cells)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _materialize_grid(
    top_specs: list[dict[str, object]],
    bottom_specs: list[dict[str, object]],
    height: int,
    width: int,
    divider_row: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[divider_row, :] = DIVIDER
    for spec in top_specs + bottom_specs:
        for r, c in spec["cells"]:
            grid[int(r), int(c)] = int(spec["color"])
    return grid


def _sample_pair_colors(pair_count: int) -> tuple[list[int], list[int]]:
    top_colors = random.sample(PALETTE, pair_count)
    for _ in range(50):
        bottom_colors = random.sample(PALETTE, pair_count)
        if all(tc != bc for tc, bc in zip(top_colors, bottom_colors)):
            return top_colors, bottom_colors
    raise RuntimeError("failed to sample distinct pair colors")


def _choose_variant(entry: dict[str, object], avoid_key: tuple[str, ...] | None = None) -> np.ndarray:
    variants = list(entry["variants"])
    random.shuffle(variants)
    if avoid_key is None:
        return np.asarray(variants[0], dtype=bool)
    for variant in variants:
        if _mask_key(np.asarray(variant, dtype=bool)) != avoid_key:
            return np.asarray(variant, dtype=bool)
    return np.asarray(variants[0], dtype=bool)


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)
    divider_row = random.randint(7, height - 8)
    pair_count = random.choice((2, 3))
    count_options = [
        (top_count, bottom_count)
        for top_count in range(3, 6)
        for bottom_count in range(3, 6)
        if top_count >= pair_count and bottom_count >= pair_count and top_count + bottom_count > 2 * pair_count
    ]
    top_count, bottom_count = random.choice(count_options)
    unique_top = top_count - pair_count
    unique_bottom = bottom_count - pair_count
    needed_signatures = pair_count + unique_top + unique_bottom
    entries = random.sample(list(SHAPE_LIBRARY), needed_signatures)
    pair_entries = entries[:pair_count]
    unique_entries = entries[pair_count:]

    near_sides = [random.choice(("top", "bottom")) for _ in range(pair_count)]
    if pair_count >= 2 and len(set(near_sides)) == 1:
        near_sides[0] = "top"
        near_sides[1] = "bottom"

    pair_top_colors, pair_bottom_colors = _sample_pair_colors(pair_count)
    top_unique_colors = random.sample([color for color in PALETTE if color not in pair_top_colors], unique_top)
    bottom_unique_colors = random.sample([color for color in PALETTE if color not in pair_bottom_colors], unique_bottom)

    top_specs: list[dict[str, object]] = []
    bottom_specs: list[dict[str, object]] = []
    for idx, entry in enumerate(pair_entries):
        top_mask = _choose_variant(entry)
        bottom_mask = _choose_variant(entry, avoid_key=_mask_key(top_mask))
        distances = _pair_distance_choices(top_mask, bottom_mask, height, divider_row, near_sides[idx])
        if distances is None:
            return None
        top_distance, bottom_distance = distances
        top_specs.append(
            {
                "mask": top_mask,
                "signature": entry["signature"],
                "color": pair_top_colors[idx],
                "distance": top_distance,
                "distance_options": None,
            }
        )
        bottom_specs.append(
            {
                "mask": bottom_mask,
                "signature": entry["signature"],
                "color": pair_bottom_colors[idx],
                "distance": bottom_distance,
                "distance_options": None,
            }
        )

    for entry, color in zip(unique_entries[:unique_top], top_unique_colors):
        mask = _choose_variant(entry)
        options = _distance_options("top", divider_row, height, mask)
        if not options:
            return None
        top_specs.append(
            {
                "mask": mask,
                "signature": entry["signature"],
                "color": color,
                "distance": None,
                "distance_options": options,
            }
        )

    for entry, color in zip(unique_entries[unique_top:], bottom_unique_colors):
        mask = _choose_variant(entry)
        options = _distance_options("bottom", divider_row, height, mask)
        if not options:
            return None
        bottom_specs.append(
            {
                "mask": mask,
                "signature": entry["signature"],
                "color": color,
                "distance": None,
                "distance_options": options,
            }
        )

    if not _place_side_objects(top_specs, "top", height, width, divider_row):
        return None
    if not _place_side_objects(bottom_specs, "bottom", height, width, divider_row):
        return None

    grid = _materialize_grid(top_specs, bottom_specs, height, width, divider_row)
    if not _validate_generated_grid(grid):
        return None
    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        divider_row = _find_divider_row(g)
        objects = _extract_objects(g)
        if not (6 <= len(objects) <= 10):
            return False
        top_objects = [obj for obj in objects if obj["side"] == "top"]
        bottom_objects = [obj for obj in objects if obj["side"] == "bottom"]
        if not (3 <= len(top_objects) <= 5 and 3 <= len(bottom_objects) <= 5):
            return False
        if len({obj["color"] for obj in top_objects}) != len(top_objects):
            return False
        if len({obj["color"] for obj in bottom_objects}) != len(bottom_objects):
            return False
        for obj in objects:
            area = int(len(obj["coords"]))
            if not (MIN_OBJECT_AREA <= area <= MAX_OBJECT_AREA):
                return False
            mask = np.asarray(obj["mask"], dtype=bool)
            if not _has_unique_vertical_extremes(mask):
                return False
            if _distance_to_divider(obj, divider_row) < 2:
                return False

        groups = _attach_signatures(objects)
        sizes = sorted(len(members) for members in groups.values())
        if any(size not in (1, 2) for size in sizes):
            return False
        pair_groups = [members for members in groups.values() if len(members) == 2]
        if not (2 <= len(pair_groups) <= 3):
            return False
        if len(pair_groups) == len(groups):
            return False

        near_sides: list[str] = []
        for members in pair_groups:
            if {obj["side"] for obj in members} != {"top", "bottom"}:
                return False
            near_obj, far_obj = _pair_roles(members, divider_row)
            if _distance_to_divider(near_obj, divider_row) == _distance_to_divider(far_obj, divider_row):
                return False
            near_sides.append(str(near_obj["side"]))
        if len(pair_groups) >= 2 and len(set(near_sides)) < 2:
            return False

        expected = _render_expected(g)
        actual = main(g)
        if np.array_equal(actual, g):
            return False
        if not np.array_equal(actual, expected):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
