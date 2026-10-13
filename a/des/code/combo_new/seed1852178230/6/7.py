from __future__ import annotations

import random

import numpy as np


BG = 0
LINE = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_GRID_MIN = 18
GEN_GRID_MAX = 28
GEN_PAIR_COUNT_OPTIONS = (1, 2, 3)
GEN_HORIZONTAL_HEIGHTS = (3, 5, 7)
GEN_VERTICAL_WIDTHS = (3, 5, 7)
GEN_HALF_WIDTH_MIN = 2
GEN_HALF_WIDTH_MAX = 5
GEN_HALF_HEIGHT_MIN = 2
GEN_HALF_HEIGHT_MAX = 5
GEN_LINE_LENGTH_MIN = 2
GEN_LINE_LENGTH_MAX = 8
GEN_PAIR_TRIES = 180
GEN_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _make_component(color: int, coords: np.ndarray) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return {
        "color": int(color),
        "coords": coords.astype(int, copy=False),
        "bbox": (
            int(rows.min()),
            int(cols.min()),
            int(rows.max()),
            int(cols.max()),
        ),
    }


def _find_object_components(grid: np.ndarray, bg: int) -> list[dict]:
    components: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, LINE}:
            continue
        for coords in _mask_components(grid == color):
            components.append(_make_component(color, coords))
    return components


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _local_coord_set(comp: dict) -> set[tuple[int, int]]:
    r0, c0, _, _ = comp["bbox"]
    return {(int(r - r0), int(c - c0)) for r, c in comp["coords"]}


def _side_is_full(comp: dict, side: str) -> bool:
    h, w = _bbox_dims(comp["bbox"])
    cells = _local_coord_set(comp)
    if side == "left":
        return all((r, 0) in cells for r in range(h))
    if side == "right":
        return all((r, w - 1) in cells for r in range(h))
    if side == "top":
        return all((0, c) in cells for c in range(w))
    if side == "bottom":
        return all((h - 1, c) in cells for c in range(w))
    raise ValueError(f"unknown side {side}")


def _line_kind(coords: np.ndarray, coord_to_object: dict[tuple[int, int], int]) -> tuple[str, int, int, int]:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if len(rows) == 1 and (len(coords) == int(coords[:, 1].max() - coords[:, 1].min() + 1)):
        return ("H", int(rows[0]), int(coords[:, 1].min()), int(coords[:, 1].max()))
    if len(cols) == 1 and (len(coords) == int(coords[:, 0].max() - coords[:, 0].min() + 1)):
        return ("V", int(cols[0]), int(coords[:, 0].min()), int(coords[:, 0].max()))
    if len(coords) == 1:
        r, c = map(int, coords[0])
        horizontal = (r, c - 1) in coord_to_object and (r, c + 1) in coord_to_object
        vertical = (r - 1, c) in coord_to_object and (r + 1, c) in coord_to_object
        assert horizontal ^ vertical, "single-cell line must have a unique axis"
        if horizontal:
            return ("H", r, c, c)
        return ("V", c, r, r)
    raise AssertionError("grey components must be straight line segments")


def _component_center_matches(comp: dict, axis: str, index: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if axis == "row":
        return r0 + r1 == 2 * index
    if axis == "col":
        return c0 + c1 == 2 * index
    raise ValueError(axis)


def _merge_horizontal(left: dict, right: dict) -> tuple[np.ndarray, int, int]:
    h_left, w_left = _bbox_dims(left["bbox"])
    h_right, w_right = _bbox_dims(right["bbox"])
    assert h_left % 2 == 1 and h_right % 2 == 1, "horizontal pairs need odd heights"
    merged_h = max(h_left, h_right)
    row_off_left = (merged_h - h_left) // 2
    row_off_right = (merged_h - h_right) // 2

    coords_left = left["coords"] - np.array(left["bbox"][:2], dtype=int)
    coords_right = right["coords"] - np.array(right["bbox"][:2], dtype=int)
    merged = []
    for r, c in coords_left:
        merged.append((int(r) + row_off_left, int(c)))
    for r, c in coords_right:
        merged.append((int(r) + row_off_right, int(c) + w_left))
    return np.asarray(merged, dtype=int), merged_h, w_left + w_right


def _merge_vertical(top: dict, bottom: dict) -> tuple[np.ndarray, int, int]:
    h_top, w_top = _bbox_dims(top["bbox"])
    h_bottom, w_bottom = _bbox_dims(bottom["bbox"])
    assert w_top % 2 == 1 and w_bottom % 2 == 1, "vertical pairs need odd widths"
    merged_w = max(w_top, w_bottom)
    col_off_top = (merged_w - w_top) // 2
    col_off_bottom = (merged_w - w_bottom) // 2

    coords_top = top["coords"] - np.array(top["bbox"][:2], dtype=int)
    coords_bottom = bottom["coords"] - np.array(bottom["bbox"][:2], dtype=int)
    merged = []
    for r, c in coords_top:
        merged.append((int(r), int(c) + col_off_top))
    for r, c in coords_bottom:
        merged.append((int(r) + h_top, int(c) + col_off_bottom))
    return np.asarray(merged, dtype=int), h_top + h_bottom, merged_w


def _place_merged(
    out: np.ndarray,
    color: int,
    local_coords: np.ndarray,
    merged_h: int,
    merged_w: int,
    mid_r2: int,
    mid_c2: int,
    bg: int,
) -> None:
    start_r_num = mid_r2 - (merged_h - 1)
    start_c_num = mid_c2 - (merged_w - 1)
    assert start_r_num % 2 == 0 and start_c_num % 2 == 0, "merged shape center must land on the line midpoint"
    start_r = start_r_num // 2
    start_c = start_c_num // 2
    h, w = out.shape
    for lr, lc in local_coords:
        gr = start_r + int(lr)
        gc = start_c + int(lc)
        assert 0 <= gr < h and 0 <= gc < w, "merged shape must stay inside the grid"
        assert int(out[gr, gc]) == int(bg), "merged output shapes must not overlap"
        out[gr, gc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair connected half-patterns through grey guide lines, join each pair, and redraw the complete shape at the line midpoint.

    Input structure:
    - The background is black and the only guide marks are grey one-cell-thick horizontal or vertical line segments.
    - Every non-grey foreground object is a monochrome 4-connected half-pattern, and each half-pattern has one straight bbox side that is fully filled.
    - Each grey segment touches exactly one half-pattern at each endpoint; the two touched half-patterns share a color and form one pair.

    Rule:
    - For every grey segment, identify the two half-patterns at its endpoints.
    - If the segment is horizontal, place the left half immediately to the left of the right half; if it is vertical, place the top half immediately above the bottom half.
    - Align the paired halves by the centers of their straight facing edges, and redraw the merged shape so that its bbox center coincides with the grey segment midpoint.
    - Remove all original half-patterns and all grey guide segments.

    Key insight:
    - The grey segment simultaneously encodes pairing, join direction, and destination: it tells which halves belong together, whether they join left-right or top-bottom, and where the completed figure must be centered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)

    objects = _find_object_components(g, bg)
    assert objects and len(objects) % 2 == 0, "expected an even number of half-patterns"
    coord_to_object: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(objects):
        for r, c in comp["coords"]:
            key = (int(r), int(c))
            assert key not in coord_to_object
            coord_to_object[key] = idx

    line_components = [_make_component(LINE, coords) for coords in _mask_components(g == LINE)]
    assert line_components, "expected grey pairing lines"

    out = np.full_like(g, int(bg))
    used_objects: set[int] = set()

    for line_comp in line_components:
        orient, fixed, start, stop = _line_kind(line_comp["coords"], coord_to_object)
        if orient == "H":
            row = fixed
            left_idx = coord_to_object.get((row, start - 1))
            right_idx = coord_to_object.get((row, stop + 1))
            assert left_idx is not None and right_idx is not None and left_idx != right_idx
            left = objects[left_idx]
            right = objects[right_idx]
            assert left["color"] == right["color"], "paired half-patterns must share a color"
            assert _side_is_full(left, "right") and _side_is_full(right, "left")
            assert _component_center_matches(left, "row", row)
            assert _component_center_matches(right, "row", row)
            local_coords, merged_h, merged_w = _merge_horizontal(left, right)
            _place_merged(out, left["color"], local_coords, merged_h, merged_w, 2 * row, start + stop, bg)
            used_objects.update((left_idx, right_idx))
        else:
            col = fixed
            top_idx = coord_to_object.get((start - 1, col))
            bottom_idx = coord_to_object.get((stop + 1, col))
            assert top_idx is not None and bottom_idx is not None and top_idx != bottom_idx
            top = objects[top_idx]
            bottom = objects[bottom_idx]
            assert top["color"] == bottom["color"], "paired half-patterns must share a color"
            assert _side_is_full(top, "bottom") and _side_is_full(bottom, "top")
            assert _component_center_matches(top, "col", col)
            assert _component_center_matches(bottom, "col", col)
            local_coords, merged_h, merged_w = _merge_vertical(top, bottom)
            _place_merged(out, top["color"], local_coords, merged_h, merged_w, start + stop, 2 * col, bg)
            used_objects.update((top_idx, bottom_idx))

    assert len(used_objects) == len(objects), "every half-pattern should belong to exactly one guide line"
    return out


def _gen_side_mask(height: int, width: int, side: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if side in {"left", "right"}:
        lengths = [random.randint(1, width) for _ in range(height)]
        if width > 2 and len(set(lengths)) == 1 and random.random() < 0.8:
            lengths[random.randrange(height)] = random.randint(1, width - 1)
        for r, length in enumerate(lengths):
            if side == "left":
                mask[r, :length] = True
            else:
                mask[r, width - length :] = True
    else:
        lengths = [random.randint(1, height) for _ in range(width)]
        if height > 2 and len(set(lengths)) == 1 and random.random() < 0.8:
            lengths[random.randrange(width)] = random.randint(1, height - 1)
        for c, length in enumerate(lengths):
            if side == "top":
                mask[:length, c] = True
            else:
                mask[height - length :, c] = True
    return mask


def _merge_masks(a_mask: np.ndarray, b_mask: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "H":
        merged_h = max(a_mask.shape[0], b_mask.shape[0])
        merged = np.zeros((merged_h, a_mask.shape[1] + b_mask.shape[1]), dtype=bool)
        a_off = (merged_h - a_mask.shape[0]) // 2
        b_off = (merged_h - b_mask.shape[0]) // 2
        a_coords = np.argwhere(a_mask)
        b_coords = np.argwhere(b_mask)
        merged[a_coords[:, 0] + a_off, a_coords[:, 1]] = True
        merged[b_coords[:, 0] + b_off, b_coords[:, 1] + a_mask.shape[1]] = True
        return merged

    merged_w = max(a_mask.shape[1], b_mask.shape[1])
    merged = np.zeros((a_mask.shape[0] + b_mask.shape[0], merged_w), dtype=bool)
    a_off = (merged_w - a_mask.shape[1]) // 2
    b_off = (merged_w - b_mask.shape[1]) // 2
    a_coords = np.argwhere(a_mask)
    b_coords = np.argwhere(b_mask)
    merged[a_coords[:, 0], a_coords[:, 1] + a_off] = True
    merged[b_coords[:, 0] + a_mask.shape[0], b_coords[:, 1] + b_off] = True
    return merged


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask).astype(int)
    if coords.size == 0:
        return coords.reshape(0, 2)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _line_coords(orientation: str, fixed: int, start: int, stop: int) -> np.ndarray:
    if orientation == "H":
        cols = np.arange(start, stop + 1, dtype=int)
        return np.column_stack([np.full(cols.shape, int(fixed), dtype=int), cols])
    rows = np.arange(start, stop + 1, dtype=int)
    return np.column_stack([rows, np.full(rows.shape, int(fixed), dtype=int)])


def _coords_in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    if coords.size == 0:
        return True
    h, w = shape
    return bool(
        np.all((0 <= coords[:, 0]) & (coords[:, 0] < h) & (0 <= coords[:, 1]) & (coords[:, 1] < w))
    )


def _coords_clear(coords: np.ndarray, reserve: np.ndarray) -> bool:
    if coords.size == 0:
        return True
    return bool(np.all(~reserve[coords[:, 0], coords[:, 1]]))


def _mark_reserved(reserve: np.ndarray, coords: np.ndarray) -> None:
    if coords.size == 0:
        return
    h, w = reserve.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = coords[:, 0] + dr
            cc = coords[:, 1] + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            reserve[rr[ok], cc[ok]] = True


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    if coords.size:
        grid[coords[:, 0], coords[:, 1]] = int(color)


def _sample_line_length(parity: int) -> int:
    choices = [length for length in range(GEN_LINE_LENGTH_MIN, GEN_LINE_LENGTH_MAX + 1) if length % 2 == parity]
    assert choices, "need a line length matching merged parity"
    return random.choice(choices)


def _make_pair_spec(color: int, orientation: str) -> dict:
    if orientation == "H":
        left_h = random.choice(GEN_HORIZONTAL_HEIGHTS)
        right_h = random.choice(GEN_HORIZONTAL_HEIGHTS)
        left_w = random.randint(GEN_HALF_WIDTH_MIN, GEN_HALF_WIDTH_MAX)
        right_w = random.randint(GEN_HALF_WIDTH_MIN, GEN_HALF_WIDTH_MAX)
        left_mask = _gen_side_mask(left_h, left_w, "right")
        right_mask = _gen_side_mask(right_h, right_w, "left")
        merged_mask = _merge_masks(left_mask, right_mask, "H")
        gap = _sample_line_length(merged_mask.shape[1] % 2)
        footprint_h = merged_mask.shape[0]
        footprint_w = left_w + gap + right_w
        return {
            "color": int(color),
            "orientation": "H",
            "a_mask": left_mask,
            "b_mask": right_mask,
            "merged_mask": merged_mask,
            "gap": int(gap),
            "sort_key": int(footprint_h * footprint_w),
        }

    top_w = random.choice(GEN_VERTICAL_WIDTHS)
    bottom_w = random.choice(GEN_VERTICAL_WIDTHS)
    top_h = random.randint(GEN_HALF_HEIGHT_MIN, GEN_HALF_HEIGHT_MAX)
    bottom_h = random.randint(GEN_HALF_HEIGHT_MIN, GEN_HALF_HEIGHT_MAX)
    top_mask = _gen_side_mask(top_h, top_w, "bottom")
    bottom_mask = _gen_side_mask(bottom_h, bottom_w, "top")
    merged_mask = _merge_masks(top_mask, bottom_mask, "V")
    gap = _sample_line_length(merged_mask.shape[0] % 2)
    footprint_h = top_h + gap + bottom_h
    footprint_w = merged_mask.shape[1]
    return {
        "color": int(color),
        "orientation": "V",
        "a_mask": top_mask,
        "b_mask": bottom_mask,
        "merged_mask": merged_mask,
        "gap": int(gap),
        "sort_key": int(footprint_h * footprint_w),
    }


def _layout_from_output_anchor(spec: dict, out_r0: int, out_c0: int, shape: tuple[int, int]) -> dict | None:
    merged_h, merged_w = spec["merged_mask"].shape
    a_h, a_w = spec["a_mask"].shape
    b_h, b_w = spec["b_mask"].shape
    gap = spec["gap"]

    if spec["orientation"] == "H":
        mid_r = out_r0 + merged_h // 2
        mid_c2 = 2 * out_c0 + merged_w - 1
        start_num = mid_c2 - gap + 1
        if start_num % 2 != 0:
            return None
        line_c0 = start_num // 2
        line_c1 = line_c0 + gap - 1
        a_top = mid_r - a_h // 2
        a_left = line_c0 - a_w
        b_top = mid_r - b_h // 2
        b_left = line_c1 + 1

        a_coords = _coords_from_mask(spec["a_mask"], a_top, a_left)
        b_coords = _coords_from_mask(spec["b_mask"], b_top, b_left)
        line_coords = _line_coords("H", mid_r, line_c0, line_c1)
    else:
        mid_c = out_c0 + merged_w // 2
        mid_r2 = 2 * out_r0 + merged_h - 1
        start_num = mid_r2 - gap + 1
        if start_num % 2 != 0:
            return None
        line_r0 = start_num // 2
        line_r1 = line_r0 + gap - 1
        a_top = line_r0 - a_h
        a_left = mid_c - a_w // 2
        b_top = line_r1 + 1
        b_left = mid_c - b_w // 2

        a_coords = _coords_from_mask(spec["a_mask"], a_top, a_left)
        b_coords = _coords_from_mask(spec["b_mask"], b_top, b_left)
        line_coords = _line_coords("V", mid_c, line_r0, line_r1)

    out_coords = _coords_from_mask(spec["merged_mask"], out_r0, out_c0)
    all_input = np.vstack([a_coords, b_coords, line_coords])
    if not _coords_in_bounds(all_input, shape) or not _coords_in_bounds(out_coords, shape):
        return None

    return {
        "a_coords": a_coords,
        "b_coords": b_coords,
        "line_coords": line_coords,
        "out_coords": out_coords,
    }


def _try_place_pair(
    inp: np.ndarray,
    out: np.ndarray,
    inp_reserve: np.ndarray,
    out_reserve: np.ndarray,
    spec: dict,
) -> bool:
    h, w = inp.shape
    merged_h, merged_w = spec["merged_mask"].shape
    if merged_h > h or merged_w > w:
        return False

    for _ in range(GEN_PAIR_TRIES):
        out_r0 = random.randint(0, h - merged_h)
        out_c0 = random.randint(0, w - merged_w)
        layout = _layout_from_output_anchor(spec, out_r0, out_c0, inp.shape)
        if layout is None:
            continue
        input_coords = np.vstack([layout["a_coords"], layout["b_coords"], layout["line_coords"]])
        if not _coords_clear(input_coords, inp_reserve):
            continue
        if not _coords_clear(layout["out_coords"], out_reserve):
            continue

        _paint(inp, layout["a_coords"], spec["color"])
        _paint(inp, layout["b_coords"], spec["color"])
        _paint(inp, layout["line_coords"], LINE)
        _paint(out, layout["out_coords"], spec["color"])
        _mark_reserved(inp_reserve, input_coords)
        _mark_reserved(out_reserve, layout["out_coords"])
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    pair_count = random.choice(GEN_PAIR_COUNT_OPTIONS)
    if pair_count == 3 and min(h, w) < 22:
        pair_count = 2

    colors = random.sample(OBJECT_COLORS, pair_count)
    specs = [_make_pair_spec(color, random.choice(("H", "V"))) for color in colors]
    specs.sort(key=lambda spec: spec["sort_key"], reverse=True)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp_reserve = np.zeros((h, w), dtype=bool)
    out_reserve = np.zeros((h, w), dtype=bool)

    for spec in specs:
        if not _try_place_pair(inp, out, inp_reserve, out_reserve, spec):
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
