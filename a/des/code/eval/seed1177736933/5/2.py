from __future__ import annotations

import numpy as np


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.asarray(coords, dtype=int))
    return components


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _most_common_color(grid: np.ndarray) -> int:
    values = np.bincount(grid.ravel().astype(int), minlength=10)
    return int(np.argmax(values))


def _find_slots(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, int]]]:
    candidates: list[tuple[int, list[dict[str, int]]]] = []

    for color in np.unique(grid):
        slot_color = int(color)
        if slot_color == bg:
            continue

        components = _connected_components(grid == slot_color)
        if len(components) < 2:
            continue

        slots: list[dict[str, int]] = []
        ok = True
        for comp in components:
            r0, c0, r1, c1 = _bbox(comp)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height < 3 or width < 3:
                ok = False
                break

            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = _frame_mask(height, width)
            if not np.all(patch[border] == slot_color):
                ok = False
                break
            if np.any(patch[~border] == slot_color):
                ok = False
                break
            if not np.all(patch[1:-1, 1:-1] == bg):
                ok = False
                break

            slots.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "height": height,
                    "width": width,
                    "inner_h": height - 2,
                    "inner_w": width - 2,
                }
            )

        if not ok:
            continue

        heights = {slot["height"] for slot in slots}
        widths = {slot["width"] for slot in slots}
        bottoms = {slot["r1"] for slot in slots}
        if len(heights) != 1 or len(widths) != 1 or len(bottoms) != 1:
            continue

        slots.sort(key=lambda slot: slot["c0"])
        gaps = [slots[i + 1]["c0"] - slots[i]["c1"] - 1 for i in range(len(slots) - 1)]
        if gaps and len(set(gaps)) != 1:
            continue
        if any(gap < 1 for gap in gaps):
            continue

        candidates.append((slot_color, slots))

    assert len(candidates) == 1, "expected one slot-color family"
    return candidates[0]


def _assemble_rectangle(coords: np.ndarray) -> tuple[int, int]:
    used_rows = sorted(int(v) for v in np.unique(coords[:, 0]))
    used_cols = sorted(int(v) for v in np.unique(coords[:, 1]))
    row_index = {row: i for i, row in enumerate(used_rows)}
    col_index = {col: i for i, col in enumerate(used_cols)}

    height = len(used_rows)
    width = len(used_cols)
    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[row_index[int(r)], col_index[int(c)]] = True

    assert bool(mask.all()), "same-color fragments must compact into a full rectangle"
    return height, width


def _paint_frame(grid: np.ndarray, color: int, r0: int, c0: int, height: int, width: int) -> None:
    grid[r0, c0 : c0 + width] = color
    grid[r0 + height - 1, c0 : c0 + width] = color
    grid[r0 : r0 + height, c0] = color
    grid[r0 : r0 + height, c0 + width - 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered fragments into a solid rectangle, sort by area,
    and place the rectangles into the bottom slot frames from left to right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    slot_color, slots = _find_slots(g, bg)

    rectangles: list[dict[str, int]] = []
    for color in np.unique(g):
        obj_color = int(color)
        if obj_color in (bg, slot_color):
            continue
        coords = np.argwhere(g == obj_color)
        assert coords.size > 0
        r0, c0, _r1, _c1 = _bbox(coords)
        height, width = _assemble_rectangle(coords)
        rectangles.append(
            {
                "color": obj_color,
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
                "r0": int(r0),
                "c0": int(c0),
            }
        )

    assert rectangles
    assert len(rectangles) == len(slots)
    areas = [rect["area"] for rect in rectangles]
    assert len(set(areas)) == len(areas), "areas must define a unique order"
    rectangles.sort(key=lambda rect: (rect["area"], rect["r0"], rect["c0"]))

    out = np.full_like(g, bg)
    for slot in slots:
        _paint_frame(out, slot_color, slot["r0"], slot["c0"], slot["height"], slot["width"])

    for slot, rect in zip(slots, rectangles):
        assert rect["height"] <= slot["inner_h"]
        assert rect["width"] <= slot["inner_w"]
        r0 = int(slot["r0"] + 1)
        c0 = int(slot["c0"] + 1)
        out[r0 : r0 + rect["height"], c0 : c0 + rect["width"]] = rect["color"]

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

_BG = 0
_SLOT_COLOR = 5
_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
_STRUCTURES = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
)
_GROUP_COUNT_RANGE = (2, 5)
_RECT_MAX_SIDE = 7
_GAP_RANGE = (1, 4)
_SLOT_EXTRA_RANGE = (0, 3)
_SLOT_GAP_RANGE = (1, 4)
_MARGIN_RANGE = (1, 4)
_BOTTOM_MARGIN_RANGE = (1, 3)
_VERTICAL_GAP_RANGE = (2, 5)
_PLACEMENT_PADDING = 1
_GROUP_PLACEMENT_TRIES = 250
_GENERATION_RETRIES = 400


def _distinct_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, need: int, used: set[int], prefix: list[int]) -> None:
        if need == 0:
            if remaining == 0:
                out.append(tuple(prefix))
            return
        for value in range(1, remaining + 1):
            if value in used:
                continue
            if remaining - value < 0:
                continue
            rec(remaining - value, need - 1, used | {value}, prefix + [value])

    rec(total, parts, set(), [])
    return out


def _sample_group(rng: np.random.Generator, used_areas: set[int], color: int) -> dict | None:
    for _ in range(200):
        row_parts, col_parts = _STRUCTURES[int(rng.integers(0, len(_STRUCTURES)))]
        min_h = row_parts * (row_parts + 1) // 2
        min_w = col_parts * (col_parts + 1) // 2
        if min_h > _RECT_MAX_SIDE or min_w > _RECT_MAX_SIDE:
            continue

        rect_h = int(rng.integers(min_h, _RECT_MAX_SIDE + 1))
        rect_w = int(rng.integers(min_w, _RECT_MAX_SIDE + 1))
        area = rect_h * rect_w
        if area in used_areas:
            continue

        row_choices = _distinct_compositions(rect_h, row_parts)
        col_choices = _distinct_compositions(rect_w, col_parts)
        if not row_choices or not col_choices:
            continue

        row_sizes = row_choices[int(rng.integers(0, len(row_choices)))]
        col_sizes = col_choices[int(rng.integers(0, len(col_choices)))]
        row_gaps = [int(v) for v in rng.integers(*_GAP_RANGE, size=max(0, row_parts - 1))]
        col_gaps = [int(v) for v in rng.integers(*_GAP_RANGE, size=max(0, col_parts - 1))]

        patch_h = rect_h + sum(row_gaps)
        patch_w = rect_w + sum(col_gaps)
        patch = np.full((patch_h, patch_w), _BG, dtype=int)

        r = 0
        for i, row_size in enumerate(row_sizes):
            c = 0
            for j, col_size in enumerate(col_sizes):
                patch[r : r + row_size, c : c + col_size] = color
                if j < len(col_gaps):
                    c += col_size + col_gaps[j]
                else:
                    c += col_size
            if i < len(row_gaps):
                r += row_size + row_gaps[i]
            else:
                r += row_size

        used_areas.add(area)
        return {
            "color": int(color),
            "rect_h": int(rect_h),
            "rect_w": int(rect_w),
            "area": int(area),
            "patch": patch,
        }
    return None


def _try_place_groups(
    rng: np.random.Generator, patches: list[np.ndarray], region_h: int, region_w: int
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((region_h, region_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = [int(v) for v in rng.permutation(len(patches))]

    for idx in order:
        patch = patches[idx]
        patch_h, patch_w = patch.shape
        if patch_h > region_h or patch_w > region_w:
            return None

        placed = False
        for _ in range(_GROUP_PLACEMENT_TRIES):
            r0 = int(rng.integers(0, region_h - patch_h + 1))
            c0 = int(rng.integers(0, region_w - patch_w + 1))
            rr0 = max(0, r0 - _PLACEMENT_PADDING)
            cc0 = max(0, c0 - _PLACEMENT_PADDING)
            rr1 = min(region_h, r0 + patch_h + _PLACEMENT_PADDING)
            cc1 = min(region_w, c0 + patch_w + _PLACEMENT_PADDING)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue

            occupied[r0 : r0 + patch_h, c0 : c0 + patch_w] = True
            placements[idx] = (r0, c0)
            placed = True
            break

        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _draw_slots(
    grid: np.ndarray,
    slot_color: int,
    slot_top: int,
    left_margin: int,
    slot_h: int,
    slot_w: int,
    slot_gap: int,
    count: int,
) -> list[dict[str, int]]:
    slots: list[dict[str, int]] = []
    for i in range(count):
        c0 = left_margin + i * (slot_w + slot_gap)
        _paint_frame(grid, slot_color, slot_top, c0, slot_h, slot_w)
        slots.append(
            {
                "r0": int(slot_top),
                "c0": int(c0),
                "height": int(slot_h),
                "width": int(slot_w),
            }
        )
    return slots


def _expected_output(
    shape: tuple[int, int],
    slot_color: int,
    slot_top: int,
    left_margin: int,
    slot_h: int,
    slot_w: int,
    slot_gap: int,
    groups: list[dict],
) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    slots = _draw_slots(out, slot_color, slot_top, left_margin, slot_h, slot_w, slot_gap, len(groups))

    ordered = sorted(groups, key=lambda group: group["area"])
    for slot, group in zip(slots, ordered):
        r0 = slot["r0"] + 1
        c0 = slot["c0"] + 1
        out[r0 : r0 + group["rect_h"], c0 : c0 + group["rect_w"]] = int(group["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    group_count = int(rng.integers(*_GROUP_COUNT_RANGE))
    colors = [int(v) for v in rng.choice(_COLOR_POOL, size=group_count, replace=False)]

    groups: list[dict] = []
    used_areas: set[int] = set()
    for color in colors:
        group = _sample_group(rng, used_areas, color)
        if group is None:
            return None
        groups.append(group)

    max_rect_h = max(group["rect_h"] for group in groups)
    max_rect_w = max(group["rect_w"] for group in groups)
    slot_inner_h = max_rect_h + int(rng.integers(*_SLOT_EXTRA_RANGE))
    slot_inner_w = max_rect_w + int(rng.integers(*_SLOT_EXTRA_RANGE))
    slot_h = slot_inner_h + 2
    slot_w = slot_inner_w + 2
    slot_gap = int(rng.integers(*_SLOT_GAP_RANGE))
    left_margin = int(rng.integers(*_MARGIN_RANGE))
    right_margin = int(rng.integers(*_MARGIN_RANGE))
    bottom_margin = int(rng.integers(*_BOTTOM_MARGIN_RANGE))
    vertical_gap = int(rng.integers(*_VERTICAL_GAP_RANGE))

    slot_strip_w = left_margin + right_margin + group_count * slot_w + (group_count - 1) * slot_gap
    patch_widths = sorted((group["patch"].shape[1] for group in groups), reverse=True)
    patch_heights = sorted((group["patch"].shape[0] for group in groups), reverse=True)
    upper_w = max(slot_strip_w, sum(patch_widths[:2]) + 6, patch_widths[0] + 4) + int(rng.integers(0, 5))
    upper_h = max(sum(patch_heights[:2]) + 6, patch_heights[0] + 4) + int(rng.integers(0, 5))
    grid_h = upper_h + vertical_gap + slot_h + bottom_margin
    grid_w = upper_w
    slot_top = upper_h + vertical_gap

    patches = [group["patch"] for group in groups]
    placements = _try_place_groups(rng, patches, upper_h, upper_w)
    if placements is None or len(placements) != len(groups):
        return None

    inp = np.full((grid_h, grid_w), _BG, dtype=int)
    _draw_slots(inp, _SLOT_COLOR, slot_top, left_margin, slot_h, slot_w, slot_gap, group_count)
    for group, (r0, c0) in zip(groups, placements):
        patch = group["patch"]
        mask = patch != _BG
        inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]

    expected = _expected_output(
        inp.shape,
        _SLOT_COLOR,
        slot_top,
        left_margin,
        slot_h,
        slot_w,
        slot_gap,
        groups,
    )
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
