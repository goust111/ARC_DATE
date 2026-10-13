from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg  # noqa: E402


BG = int(Color.BLACK)
GREY = int(Color.GREY)
MARKER_COLORS = (int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW))

GRID_MIN = 21
GRID_MAX = 25
GRID_STEP = 4
SLOT_INNER = 3
GROUP_COUNT_CHOICES = (3, 4)
GROUP_SIZE_MIN = 1
GROUP_SIZE_MAX = 4
TOTAL_OBJECT_MIN = 8
TOTAL_OBJECT_MAX = 15
SHAPE_AREA_MIN = 3
SHAPE_AREA_MAX = 6
SHAPE_DIM_MAX = 3
GENERATE_RETRIES = 400


def _normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in cells}
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = 1 + max(r for r, _ in cells)
    w = 1 + max(c for _, c in cells)
    return int(h), int(w)


def _random_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        area = random.randint(SHAPE_AREA_MIN, SHAPE_AREA_MAX)
        cells = {(0, 0)}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
            cells.add((r + dr, c + dc))
        shape = _normalize_cells(cells)
        h, w = _shape_dims(shape)
        if h <= SHAPE_DIM_MAX and w <= SHAPE_DIM_MAX:
            return shape
    raise RuntimeError("failed to generate a compact 4-connected shape")


def _sample_group_counts(group_count: int) -> list[int]:
    for _ in range(200):
        counts = [random.randint(GROUP_SIZE_MIN, GROUP_SIZE_MAX) for _ in range(group_count)]
        total = sum(counts)
        if TOTAL_OBJECT_MIN <= total <= TOTAL_OBJECT_MAX:
            return counts
    raise RuntimeError("failed to sample group counts")


def _sample_group_specs() -> list[dict]:
    group_count = random.choice(GROUP_COUNT_CHOICES)
    counts = _sample_group_counts(group_count)
    colors = random.sample(MARKER_COLORS, group_count)

    shared_shape = _random_shape()
    marker_pair = random.sample(shared_shape, 2)
    signatures = {(shared_shape, marker_pair[0]), (shared_shape, marker_pair[1])}
    specs = [
        {"shape": shared_shape, "marker": marker_pair[0], "count": counts[0], "color": colors[0]},
        {"shape": shared_shape, "marker": marker_pair[1], "count": counts[1], "color": colors[1]},
    ]

    while len(specs) < group_count:
        shape = _random_shape()
        marker = random.choice(shape)
        key = (shape, marker)
        if key in signatures:
            continue
        if shape == shared_shape:
            continue
        signatures.add(key)
        idx = len(specs)
        specs.append({"shape": shape, "marker": marker, "count": counts[idx], "color": colors[idx]})
    return specs


def _slot_positions(height: int, width: int) -> list[tuple[int, int]]:
    rows = list(range(1, height - SLOT_INNER, GRID_STEP))
    cols = list(range(1, width - SLOT_INNER, GRID_STEP))
    return [(r, c) for r in rows for c in cols]


def _paint_object(
    grid: np.ndarray,
    top: int,
    left: int,
    shape: tuple[tuple[int, int], ...],
    marker: tuple[int, int],
    color: int,
) -> tuple[int, int, int, int]:
    for dr, dc in shape:
        grid[top + dr, left + dc] = GREY
    mr, mc = marker
    grid[top + mr, left + mc] = int(color)
    h, w = _shape_dims(shape)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _build_input(specs: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    slots = _slot_positions(height, width)
    leader_cut = len(slots) // 2
    leader_slots = sorted(random.sample(slots[:leader_cut], len(specs)))
    extra_slots = slots[leader_cut:]
    random.shuffle(extra_slots)

    order = list(range(len(specs)))
    random.shuffle(order)
    leaders_by_group = {}
    for group_idx, slot in zip(order, leader_slots):
        leaders_by_group[group_idx] = slot

    extra_needed = sum(spec["count"] - 1 for spec in specs)
    if extra_needed > len(extra_slots):
        raise RuntimeError("not enough slots")

    extra_index = 0
    for group_idx, spec in enumerate(specs):
        placements = [leaders_by_group[group_idx]]
        for _ in range(spec["count"] - 1):
            placements.append(extra_slots[extra_index])
            extra_index += 1
        for slot_r, slot_c in placements:
            h, w = _shape_dims(spec["shape"])
            top = slot_r + random.randint(0, SLOT_INNER - h)
            left = slot_c + random.randint(0, SLOT_INNER - w)
            _paint_object(grid, top, left, spec["shape"], spec["marker"], spec["color"])
    return grid


def _object_signature(patch: np.ndarray, bg: int) -> tuple[tuple[int, ...], ...]:
    encoded = np.zeros_like(patch, dtype=int)
    fg = patch != int(bg)
    encoded[fg & (patch == GREY)] = 1
    encoded[fg & (patch != GREY)] = 2
    return tuple(tuple(int(v) for v in row) for row in encoded)


def _shape_only_signature(patch: np.ndarray, bg: int) -> tuple[tuple[int, ...], ...]:
    encoded = (patch != int(bg)).astype(int)
    return tuple(tuple(int(v) for v in row) for row in encoded)


def _analyze_grid(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects: list[dict] = []

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert components, "expected at least one foreground component"

    for comp in components:
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, bbox)
        fg = patch != bg
        marker_mask = fg & (patch != GREY)
        marker_coords = np.argwhere(marker_mask)
        marker_colors = sorted(set(map(int, patch[marker_mask].tolist())))

        assert int(comp["area"]) == int(fg.sum())
        assert SHAPE_AREA_MIN <= int(comp["area"]) <= SHAPE_AREA_MAX
        assert len(marker_coords) == 1
        assert len(marker_colors) == 1
        assert int((patch == GREY).sum()) == int(comp["area"]) - 1

        r0, c0, r1, c1 = bbox
        assert 0 < r0 and 0 < c0 and r1 < g.shape[0] - 1 and c1 < g.shape[1] - 1

        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h <= SHAPE_DIM_MAX and w <= SHAPE_DIM_MAX

        mr, mc = map(int, marker_coords[0])
        objects.append(
            {
                "bbox": bbox,
                "area": int(comp["area"]),
                "marker_color": int(marker_colors[0]),
                "signature": _object_signature(patch, bg),
                "shape_only_signature": _shape_only_signature(patch, bg),
                "marker_rel": (mr, mc),
            }
        )

    grouped: dict[tuple[tuple[int, ...], ...], list[dict]] = {}
    for obj in objects:
        grouped.setdefault(obj["signature"], []).append(obj)

    groups: list[dict] = []
    for signature, members in grouped.items():
        colors = {m["marker_color"] for m in members}
        areas = {m["area"] for m in members}
        shapes = {m["shape_only_signature"] for m in members}
        assert len(colors) == 1
        assert len(areas) == 1
        assert len(shapes) == 1

        groups.append(
            {
                "signature": signature,
                "shape_only_signature": next(iter(shapes)),
                "color": next(iter(colors)),
                "area": next(iter(areas)),
                "count": len(members),
                "sort_key": min((m["bbox"][0], m["bbox"][1]) for m in members),
            }
        )

    groups.sort(key=lambda group: group["sort_key"])
    return bg, objects, groups


def _build_output(groups: list[dict], thicknesses: list[int]) -> np.ndarray:
    assert groups and len(groups) == len(thicknesses)
    side = int(thicknesses[0])
    out = np.full((side, side), int(groups[0]["color"]), dtype=int)
    for group, thickness in zip(groups[1:], thicknesses[1:]):
        t = int(thickness)
        new_side = out.shape[0] + 2 * t
        wrapped = np.full((new_side, new_side), int(group["color"]), dtype=int)
        wrapped[t : t + out.shape[0], t : t + out.shape[1]] = out
        out = wrapped
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered marked shapes by normalized form, count each group, then build nested square layers from those counts.

    Input structure:
    - The background is black.
    - Foreground objects are small 4-connected shapes of grey cells with exactly one non-grey marker cell inside each shape.
    - Objects that share the same translated shape together with the same relative marker position belong to one group, and each group uses one marker color consistently.

    Rule:
    - Normalize every foreground object by translating its bounding box to the origin, keeping track of which normalized cell is the marker.
    - Group objects with identical normalized marked shapes, and count how many instances each group has.
    - Sort groups by the smallest bounding-box top-left among their members, using row first and column second.
    - Build a concentric square structure in that order: the first group becomes a solid center square of side equal to its instance count, and every later group wraps the current square with a layer whose thickness equals that group's instance count, using the group's marker color.

    Key insight:
    - Shape identity chooses the groups, spatial position chooses the layer order, and group multiplicity chooses each layer thickness, so all three cues are necessary.
    """
    _bg, _objects, groups = _analyze_grid(np.asarray(grid, dtype=int))
    thicknesses = [int(group["count"]) for group in groups]
    return _build_output(groups, thicknesses)


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        _bg, objects, groups = _analyze_grid(grid)
        true_out = main(grid)
    except Exception:
        return False

    if not (TOTAL_OBJECT_MIN <= len(objects) <= TOTAL_OBJECT_MAX):
        return False
    if len(groups) not in GROUP_COUNT_CHOICES:
        return False
    if len({group["color"] for group in groups}) != len(groups):
        return False
    if max(true_out.shape) > 30:
        return False

    shape_only_buckets: dict[tuple[tuple[int, ...], ...], int] = {}
    for group in groups:
        shape_only_buckets[group["shape_only_signature"]] = shape_only_buckets.get(group["shape_only_signature"], 0) + 1
    if not any(count >= 2 for count in shape_only_buckets.values()):
        return False

    actual_keys = [group["signature"] for group in groups]
    color_order = [group["signature"] for group in sorted(groups, key=lambda group: group["color"])]
    count_asc_order = [group["signature"] for group in sorted(groups, key=lambda group: (group["count"], group["color"]))]
    count_desc_order = [group["signature"] for group in sorted(groups, key=lambda group: (-group["count"], group["color"]))]
    area_asc_order = [group["signature"] for group in sorted(groups, key=lambda group: (group["area"], group["color"]))]
    area_desc_order = [group["signature"] for group in sorted(groups, key=lambda group: (-group["area"], group["color"]))]

    if actual_keys in (color_order, count_asc_order, count_desc_order, area_asc_order, area_desc_order):
        return False

    counts = [group["count"] for group in groups]
    areas = [group["area"] for group in groups]
    if counts == areas:
        return False

    if np.array_equal(true_out, _build_output(groups, areas)):
        return False

    if np.array_equal(true_out, _build_output(sorted(groups, key=lambda group: group["color"]), counts)):
        return False

    if np.array_equal(true_out, _build_output(sorted(groups, key=lambda group: (group["count"], group["color"])), counts)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)
        specs = _sample_group_specs()
        grid = _build_input(specs, height, width)
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
