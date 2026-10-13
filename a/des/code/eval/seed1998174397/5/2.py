from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

GROUP_COUNT_RANGE = (3, 4)
FRAGMENT_COUNT_VALUES = (2, 3, 4, 5, 6)
TARGET_HEIGHT_RANGE = (2, 6)
TARGET_WIDTH_RANGE = (2, 7)
MAX_ROWS_PER_TARGET = 3
LANE_GAP_RANGE = (1, 3)
SHELF_GAP_RANGE = (2, 4)
CANVAS_LIMIT = 30
GEN_ATTEMPTS = 400
GROUP_ATTEMPTS = 200


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi + 1))


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    return sorted(int(v) for v in np.unique(grid) if int(v) not in exclude)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "area": len(coords),
                }
            )
    return comps


def _ordered_shape_from_dims(dims: list[tuple[int, int]]) -> tuple[int, int] | None:
    total = sum(h * w for h, w in dims)
    min_width = max(w for _, w in dims)
    candidates: list[tuple[int, int]] = []
    for width in range(min_width, total + 1):
        if total % width != 0:
            continue
        height = total // width
        x = 0
        y = 0
        row_h = None
        ok = True
        for frag_h, frag_w in dims:
            if x == 0:
                row_h = frag_h
            elif frag_h != row_h:
                ok = False
                break
            if x + frag_w > width:
                ok = False
                break
            x += frag_w
            if x == width:
                y += int(row_h)
                x = 0
                row_h = None
        if ok and x == 0 and y == height:
            candidates.append((height, width))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _infer_group_shape(components: list[dict[str, object]]) -> tuple[int, int]:
    for comp in components:
        if int(comp["area"]) != int(comp["height"]) * int(comp["width"]):
            raise AssertionError("every fragment must be a solid rectangle")

    col_major = sorted(components, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    row_major = sorted(components, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    col_shape = _ordered_shape_from_dims([(int(comp["height"]), int(comp["width"])) for comp in col_major])
    if col_shape is not None:
        return col_shape
    row_shape = _ordered_shape_from_dims([(int(comp["height"]), int(comp["width"])) for comp in row_major])
    if row_shape is not None:
        return row_shape
    raise AssertionError("could not infer a unique target rectangle")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-colored rectangular fragments into full rectangles and sort groups by fragment count.
    """
    g = np.asarray(grid, dtype=int)
    colors = _colors_present(g, exclude={BACKGROUND})
    assert colors, "expected at least one colored group"

    groups: list[dict[str, int]] = []
    for color in colors:
        comps = _find_components(g, color)
        assert comps, "missing fragments for color"
        height, width = _infer_group_shape(comps)
        area = sum(int(comp["area"]) for comp in comps)
        anchor_c = min(int(comp["bbox"][1]) for comp in comps)
        anchor_r = min(int(comp["bbox"][0]) for comp in comps)
        groups.append(
            {
                "color": color,
                "count": len(comps),
                "height": height,
                "width": width,
                "area": area,
                "anchor_c": anchor_c,
                "anchor_r": anchor_r,
            }
        )

    groups.sort(key=lambda item: (item["count"], item["area"], item["anchor_c"], item["anchor_r"]))
    out_h = max(item["height"] for item in groups)
    out_w = sum(item["width"] for item in groups) + len(groups) - 1
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)

    cursor = 0
    for item in groups:
        out[: item["height"], cursor : cursor + item["width"]] = item["color"]
        cursor += item["width"] + 1
    return out


def _random_composition(total: int, parts: int) -> list[int] | None:
    if parts < 1 or total < parts:
        return None
    if parts == 1:
        return [total]
    cuts = sorted(int(v) for v in RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _make_group_spec(fragment_count: int) -> dict[str, object] | None:
    for _ in range(GROUP_ATTEMPTS):
        max_rows = min(MAX_ROWS_PER_TARGET, fragment_count)
        row_options = [rows for rows in range(1, max_rows + 1)]
        weights = np.array([0.15 if rows == 1 else 0.45 if rows == 2 else 0.40 for rows in row_options], dtype=float)
        weights /= weights.sum()
        row_count = int(RNG.choice(row_options, p=weights))

        row_piece_counts = _random_composition(fragment_count, row_count)
        if row_piece_counts is None:
            continue
        width_lo = max(TARGET_WIDTH_RANGE[0], max(row_piece_counts))
        width_hi = TARGET_WIDTH_RANGE[1]
        if width_lo > width_hi:
            continue
        target_w = _randint(width_lo, width_hi)

        height_lo = max(TARGET_HEIGHT_RANGE[0], row_count)
        height_hi = TARGET_HEIGHT_RANGE[1]
        if height_lo > height_hi:
            continue
        target_h = _randint(height_lo, height_hi)

        row_heights = _random_composition(target_h, row_count)
        if row_heights is None:
            continue

        dims: list[tuple[int, int]] = []
        for band_h, pieces_in_row in zip(row_heights, row_piece_counts):
            widths = _random_composition(target_w, pieces_in_row)
            if widths is None:
                dims = []
                break
            dims.extend((band_h, frag_w) for frag_w in widths)
        if not dims or not any(h > 1 or w > 1 for h, w in dims):
            continue

        inferred = _ordered_shape_from_dims(dims)
        if inferred != (target_h, target_w):
            continue

        return {
            "count": fragment_count,
            "height": target_h,
            "width": target_w,
            "area": target_h * target_w,
            "dims": dims,
        }
    return None


def _choose_fragment_counts(n_groups: int) -> list[int]:
    assert 2 <= n_groups <= len(FRAGMENT_COUNT_VALUES)
    if n_groups >= 3 and RNG.random() < 0.5:
        repeated = int(RNG.choice(FRAGMENT_COUNT_VALUES))
        others = [v for v in FRAGMENT_COUNT_VALUES if v != repeated]
        chosen = list(int(v) for v in RNG.choice(others, size=n_groups - 2, replace=False))
        counts = chosen + [repeated, repeated]
    else:
        counts = list(int(v) for v in RNG.choice(FRAGMENT_COUNT_VALUES, size=n_groups, replace=False))
    RNG.shuffle(counts)
    return counts


def _expected_output(specs: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: (int(item["count"]), int(item["area"])))
    out_h = max(int(item["height"]) for item in ordered)
    out_w = sum(int(item["width"]) for item in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)
    c = 0
    for spec in ordered:
        h = int(spec["height"])
        w = int(spec["width"])
        out[:h, c : c + w] = int(spec["color"])
        c += w + 1
    return out


def _lane_layout(dims: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    max_h = max(h for h, _ in dims)
    lane_h = max_h + _randint(0, 2)
    local_boxes: list[tuple[int, int, int, int]] = []
    x = 0
    for idx, (frag_h, frag_w) in enumerate(dims):
        y = _randint(0, lane_h - frag_h)
        local_boxes.append((y, x, frag_h, frag_w))
        x += frag_w
        if idx < len(dims) - 1:
            x += _randint(*LANE_GAP_RANGE)
    return lane_h, x, local_boxes


def _pack_lanes(lanes: list[tuple[int, int, list[tuple[int, int, int, int]]]]) -> tuple[list[tuple[int, int]], int, int] | None:
    order = list(range(len(lanes)))
    RNG.shuffle(order)
    anchors = [(-1, -1)] * len(lanes)
    x = _randint(0, 1)
    y = _randint(0, 1)
    shelf_h = 0
    max_w = 0
    for idx in order:
        lane_h, lane_w, _ = lanes[idx]
        if lane_w > CANVAS_LIMIT:
            return None
        if x > 1 and x + lane_w > CANVAS_LIMIT:
            y += shelf_h + _randint(*SHELF_GAP_RANGE)
            x = _randint(0, 1)
            shelf_h = 0
        anchors[idx] = (y, x)
        x += lane_w + _randint(*SHELF_GAP_RANGE)
        shelf_h = max(shelf_h, lane_h)
        max_w = max(max_w, x)
    total_h = y + shelf_h
    total_w = max_w
    if total_h > CANVAS_LIMIT or total_w > CANVAS_LIMIT:
        return None
    return anchors, total_h, total_w


def _paint_group(grid: np.ndarray, color: int, anchor: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> None:
    base_r, base_c = anchor
    for local_r, local_c, frag_h, frag_w in boxes:
        r0 = base_r + local_r
        c0 = base_c + local_c
        if np.any(grid[r0 : r0 + frag_h, c0 : c0 + frag_w] != BACKGROUND):
            raise AssertionError("fragments must not overlap")
        grid[r0 : r0 + frag_h, c0 : c0 + frag_w] = color


def _validate_specs_against_grid(grid: np.ndarray, specs: list[dict[str, object]]) -> bool:
    for spec in specs:
        color = int(spec["color"])
        comps = _find_components(grid, color)
        if len(comps) != int(spec["count"]):
            return False
        col_order = sorted(comps, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
        col_shape = _ordered_shape_from_dims([(int(comp["height"]), int(comp["width"])) for comp in col_order])
        if col_shape != (int(spec["height"]), int(spec["width"])):
            return False
        row_order = sorted(comps, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
        row_shape = _ordered_shape_from_dims([(int(comp["height"]), int(comp["width"])) for comp in row_order])
        if row_shape is not None and row_shape != (int(spec["height"]), int(spec["width"])):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_groups = _randint(*GROUP_COUNT_RANGE)
    counts = _choose_fragment_counts(n_groups)
    specs: list[dict[str, object]] = []
    used_pairs: set[tuple[int, int]] = set()
    for count in counts:
        spec = _make_group_spec(count)
        if spec is None:
            return None
        pair = (int(spec["count"]), int(spec["area"]))
        if pair in used_pairs:
            return None
        used_pairs.add(pair)
        specs.append(spec)

    colors = list(int(v) for v in RNG.choice(PALETTE, size=n_groups, replace=False))
    lanes: list[tuple[int, int, list[tuple[int, int, int, int]]]] = []
    for spec, color in zip(specs, colors):
        spec["color"] = color
        lanes.append(_lane_layout(list(spec["dims"])))

    packed = _pack_lanes(lanes)
    if packed is None:
        return None
    anchors, core_h, core_w = packed

    pad_top = _randint(0, max(0, CANVAS_LIMIT - core_h))
    remaining_h = CANVAS_LIMIT - core_h - pad_top
    pad_bottom = _randint(0, remaining_h)
    pad_left = _randint(0, max(0, CANVAS_LIMIT - core_w))
    remaining_w = CANVAS_LIMIT - core_w - pad_left
    pad_right = _randint(0, remaining_w)
    grid = np.full((core_h + pad_top + pad_bottom, core_w + pad_left + pad_right), BACKGROUND, dtype=int)

    for spec, lane, anchor in zip(specs, lanes, anchors):
        global_anchor = (pad_top + int(anchor[0]), pad_left + int(anchor[1]))
        _paint_group(grid, int(spec["color"]), global_anchor, lane[2])

    if not _validate_specs_against_grid(grid, specs):
        return None

    expected = _expected_output(specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragmented-rectangle input")
