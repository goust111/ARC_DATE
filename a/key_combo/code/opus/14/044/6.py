from __future__ import annotations

from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
MAROON = 9
RECT_PALETTE = (3, 4, 6, 7)
STRUCTURAL = {BG, BLUE, RED, GREY, MAROON}
MAX_GEN_ATTEMPTS = 400
WIDTH_SETS = {
    3: [2, 3, 4],
    4: [1, 2, 3, 4],
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            box = _bbox(coords)
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": box,
                    "area": int(coords.shape[0]),
                    "patch": _crop(grid, box),
                }
            )
    return out


def _parse_blue_vertical(grid: np.ndarray) -> bool:
    assert int(grid[0, 0]) == MAROON
    top_vals = set(map(int, grid[0, 1:]))
    left_vals = set(map(int, grid[1:, 0]))
    assert len(top_vals) == 1 and len(left_vals) == 1
    top_color = next(iter(top_vals))
    left_color = next(iter(left_vals))
    assert {top_color, left_color} == {BLUE, RED}
    return left_color == BLUE


def _find_frame_and_slots(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    grey_components = _components_of_color(grid, GREY)
    assert grey_components
    frame = max(grey_components, key=lambda comp: comp["area"])
    r0, c0, r1, c1 = frame["bbox"]
    assert r1 - r0 >= 2 and c1 - c0 >= 4

    interior_rows = slice(r0 + 1, r1)
    separators = [
        c
        for c in range(c0 + 1, c1)
        if bool(np.all(grid[interior_rows, c] == GREY))
    ]
    walls = [c0, *separators, c1]
    slots: list[dict] = []
    for left, right in zip(walls, walls[1:]):
        if right - left <= 1:
            continue
        slots.append(
            {
                "c0": int(left + 1),
                "c1": int(right - 1),
                "width": int(right - left - 1),
            }
        )
    assert 3 <= len(slots) <= 4
    return (int(r0), int(c0), int(r1), int(c1)), slots


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - STRUCTURAL):
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = comp["bbox"]
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if comp["area"] != h * w:
                continue
            if not bool(np.all(comp["patch"] == int(color))):
                continue
            rects.append(comp)
    assert 3 <= len(rects) <= 4
    rects.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return rects


def _transform_rectangles(rects: list[dict], blue_vertical: bool) -> list[dict]:
    out: list[dict] = []
    for rect in rects:
        patch = np.rot90(rect["patch"], 1) if blue_vertical else rect["patch"].copy()
        h, w = patch.shape
        out.append(
            {
                "source_bbox": rect["bbox"],
                "color": rect["color"],
                "patch": patch,
                "height": int(h),
                "width": int(w),
            }
        )
    return out


def _pair_width_matched(transformed: list[dict], slots: list[dict]) -> list[tuple[dict, dict]] | None:
    rect_groups: dict[int, list[dict]] = defaultdict(list)
    slot_groups: dict[int, list[dict]] = defaultdict(list)
    for rect in transformed:
        rect_groups[int(rect["width"])].append(rect)
    for slot in slots:
        slot_groups[int(slot["width"])].append(slot)
    if set(rect_groups) != set(slot_groups):
        return None

    pairs: list[tuple[dict, dict]] = []
    for width in sorted(slot_groups):
        slot_list = sorted(slot_groups[width], key=lambda slot: slot["c0"])
        rect_list = sorted(
            rect_groups[width],
            key=lambda rect: (rect["height"], rect["color"], rect["source_bbox"]),
        )
        if len(slot_list) != len(rect_list):
            return None
        pairs.extend(zip(slot_list, rect_list))
    pairs.sort(key=lambda pair: pair[0]["c0"])
    return pairs


def _pair_source_order(transformed: list[dict], slots: list[dict]) -> list[tuple[dict, dict]] | None:
    rects = sorted(transformed, key=lambda rect: (rect["source_bbox"][0], rect["source_bbox"][1]))
    ordered_slots = sorted(slots, key=lambda slot: slot["c0"])
    if len(rects) != len(ordered_slots):
        return None
    pairs = list(zip(ordered_slots, rects))
    for slot, rect in pairs:
        if rect["width"] > slot["width"]:
            return None
    return pairs


def _render_from_pairs(
    grid: np.ndarray,
    rects: list[dict],
    frame_box: tuple[int, int, int, int],
    pairs: list[tuple[dict, dict]] | None,
) -> np.ndarray | None:
    if pairs is None:
        return None
    r0, _c0, r1, _c1 = frame_box
    interior_bottom = int(r1 - 1)
    out = np.array(grid, dtype=int, copy=True)
    for rect in rects:
        rr, cc = rect["coords"][:, 0], rect["coords"][:, 1]
        out[rr, cc] = BG
    for slot, rect in pairs:
        patch = rect["patch"]
        ph, pw = patch.shape
        if pw > int(slot["width"]):
            return None
        dest_r = int(interior_bottom - ph + 1)
        dest_c = int(slot["c0"] + (slot["width"] - pw) // 2)
        if dest_r < r0 + 1:
            return None
        _paste(out, patch, dest_r, dest_c)
    return out


def _solve(grid: np.ndarray, force_blue_vertical: bool | None = None, pair_mode: str = "width") -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    blue_vertical = _parse_blue_vertical(g) if force_blue_vertical is None else bool(force_blue_vertical)
    frame_box, slots = _find_frame_and_slots(g)
    rects = _find_rectangles(g)
    transformed = _transform_rectangles(rects, blue_vertical)
    if pair_mode == "width":
        pairs = _pair_width_matched(transformed, slots)
    elif pair_mode == "source":
        pairs = _pair_source_order(transformed, slots)
    else:
        raise ValueError("unknown pair_mode")
    return _render_from_pairs(g, rects, frame_box, pairs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the free rectangles according to which edge axis is blue, then drop them into the matching frame slots.

    Input structure:
    - Black background with a maroon corner marker at the top-left, one full colored line on the top edge, and one full colored line on the left edge.
    - Exactly one of those edge lines is blue and the other is red; together they encode the axis orientation.
    - A grey rectangular frame in the lower half is split by grey vertical separators into 3 or 4 vertical slots.
    - Elsewhere are 3 or 4 solid colored rectangles, distinct from the structural colors.

    Rule:
    - If the blue line is the vertical left edge, rotate every colored rectangle 90 degrees counterclockwise; if the blue line is the horizontal top edge, keep them unrotated.
    - For each transformed rectangle, find the unique frame slot whose width matches the rectangle's width.
    - Erase the rectangles from their original positions and place each one into its matching slot, horizontally centered within the slot and bottom-aligned to the slot interior.
    - Keep the axes, corner marker, grey frame, and grey separators unchanged.

    Key insight:
    - The colored edge axes do not move anything directly; they encode whether rectangle height or rectangle width should be treated as the slot-matching width.
    """
    result = _solve(grid)
    assert result is not None
    return result


def _signature_value(values: list[int]) -> int:
    total = 0
    for i, value in enumerate(values):
        total += (i + 3) * int(value)
    return total


def _sample_final_heights(rng: np.random.Generator, slot_widths: list[int]) -> list[int] | None:
    for _ in range(80):
        heights = []
        for width in slot_widths:
            choices = [h for h in range(1, 5) if h != int(width)]
            heights.append(int(rng.choice(choices)))
        if sorted(heights) != sorted(slot_widths):
            return heights
    return None


def _deterministic_order(colors: list[int], slot_widths: list[int], final_heights: list[int], blue_vertical: bool) -> list[int]:
    keys = []
    axis_term = 5 if blue_vertical else 11
    for i, (color, width, height) in enumerate(zip(colors, slot_widths, final_heights)):
        key = (color * 13 + width * 7 + height * axis_term + i * 17) % 97
        keys.append((int(key), i))
    return [idx for _key, idx in sorted(keys)]


def _row_start(width_limit: int, row_width: int, salt: int, min_col: int) -> int:
    max_start = int(width_limit - row_width - 1)
    if max_start <= min_col:
        return int(min_col)
    return int(min_col + salt % (max_start - min_col + 1))


def _build_layout(
    rng: np.random.Generator,
    colors: list[int],
    slot_widths: list[int],
    final_heights: list[int],
    blue_vertical: bool,
) -> dict | None:
    count = len(slot_widths)
    rect_specs: list[dict] = []
    for color, slot_width, final_height in zip(colors, slot_widths, final_heights):
        if blue_vertical:
            orig_h, orig_w = int(slot_width), int(final_height)
        else:
            orig_h, orig_w = int(final_height), int(slot_width)
        rect_specs.append(
            {
                "color": int(color),
                "slot_width": int(slot_width),
                "final_height": int(final_height),
                "orig_h": int(orig_h),
                "orig_w": int(orig_w),
                "orig_patch": np.full((orig_h, orig_w), int(color), dtype=int),
                "final_patch": np.full((final_height, slot_width), int(color), dtype=int),
            }
        )

    inner_h = max(spec["final_height"] for spec in rect_specs)
    if inner_h < 4 and rng.random() < 0.45:
        inner_h += 1
    frame_h = int(inner_h + 2)
    frame_w = int(sum(slot_widths) + (count - 1) + 2)

    order = _deterministic_order(colors, slot_widths, final_heights, blue_vertical)
    split = (count + 1) // 2
    row1 = order[:split]
    row2 = order[split:]
    gap = 2

    row1_w = sum(rect_specs[i]["orig_w"] for i in row1) + gap * max(0, len(row1) - 1)
    row2_w = sum(rect_specs[i]["orig_w"] for i in row2) + gap * max(0, len(row2) - 1)
    min_w = max(18, frame_w + 4, row1_w + 4, row2_w + 5)
    if min_w > 30:
        return None
    max_w = min(30, min_w + 6)
    width = int(rng.integers(min_w, max_w + 1))

    salt_values = colors + slot_widths + final_heights + [width, int(blue_vertical)]
    salt = _signature_value(salt_values)
    row1_start = _row_start(width, row1_w, salt, 2)
    row2_start = _row_start(width, row2_w, salt * 3 + 7, 3)

    y1 = 2
    row1_h = max(rect_specs[i]["orig_h"] for i in row1)
    y2 = y1 + row1_h + 2 if row2 else y1
    row2_h = max((rect_specs[i]["orig_h"] for i in row2), default=0)
    layout_bottom = max(y1 + row1_h - 1, y2 + row2_h - 1)

    min_h = max(18, layout_bottom + 2 + frame_h + 1)
    if min_h > 30:
        return None
    max_h = min(30, min_h + 6)
    height = int(rng.integers(min_h, max_h + 1))

    frame_r0_min = max(layout_bottom + 2, height // 2)
    frame_r0_max = height - frame_h - 1
    if frame_r0_min > frame_r0_max:
        return None
    frame_r0 = int(rng.integers(frame_r0_min, frame_r0_max + 1))

    frame_c0_min = 2
    frame_c0_max = width - frame_w - 1
    if frame_c0_min > frame_c0_max:
        return None
    frame_c0 = int(rng.integers(frame_c0_min, frame_c0_max + 1))

    rect_positions: dict[int, tuple[int, int]] = {}
    cursor = row1_start
    for idx in row1:
        rect_positions[idx] = (int(y1), int(cursor))
        cursor += rect_specs[idx]["orig_w"] + gap
    cursor = row2_start
    for idx in row2:
        rect_positions[idx] = (int(y2), int(cursor))
        cursor += rect_specs[idx]["orig_w"] + gap

    slot_specs: list[dict] = []
    cursor = frame_c0 + 1
    for i, slot_width in enumerate(slot_widths):
        slot_specs.append(
            {
                "index": i,
                "c0": int(cursor),
                "c1": int(cursor + slot_width - 1),
                "width": int(slot_width),
            }
        )
        cursor += int(slot_width)
        if i != count - 1:
            cursor += 1

    return {
        "height": height,
        "width": width,
        "blue_vertical": bool(blue_vertical),
        "frame_r0": frame_r0,
        "frame_c0": frame_c0,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "inner_h": inner_h,
        "rect_specs": rect_specs,
        "rect_positions": rect_positions,
        "slot_specs": slot_specs,
    }


def _draw_frame(grid: np.ndarray, frame_r0: int, frame_c0: int, frame_h: int, frame_w: int, slot_specs: list[dict]) -> None:
    r1 = int(frame_r0 + frame_h - 1)
    c1 = int(frame_c0 + frame_w - 1)
    grid[frame_r0, frame_c0 : c1 + 1] = GREY
    grid[r1, frame_c0 : c1 + 1] = GREY
    grid[frame_r0 : r1 + 1, frame_c0] = GREY
    grid[frame_r0 : r1 + 1, c1] = GREY
    for slot in slot_specs[:-1]:
        sep_c = int(slot["c1"] + 1)
        grid[frame_r0 + 1 : r1, sep_c] = GREY


def _render_input(meta: dict) -> np.ndarray:
    h = int(meta["height"])
    w = int(meta["width"])
    grid = np.zeros((h, w), dtype=int)
    if meta["blue_vertical"]:
        grid[:, 0] = BLUE
        grid[0, :] = RED
    else:
        grid[:, 0] = RED
        grid[0, :] = BLUE
    grid[0, 0] = MAROON

    _draw_frame(
        grid,
        int(meta["frame_r0"]),
        int(meta["frame_c0"]),
        int(meta["frame_h"]),
        int(meta["frame_w"]),
        meta["slot_specs"],
    )

    for idx, spec in enumerate(meta["rect_specs"]):
        r0, c0 = meta["rect_positions"][idx]
        _paste(grid, spec["orig_patch"], r0, c0)
    return grid


def _render_expected(meta: dict, inp: np.ndarray) -> np.ndarray:
    out = np.array(inp, copy=True)
    frame_bottom = int(meta["frame_r0"] + meta["frame_h"] - 2)
    for idx, spec in enumerate(meta["rect_specs"]):
        r0, c0 = meta["rect_positions"][idx]
        h, w = spec["orig_patch"].shape
        out[r0 : r0 + h, c0 : c0 + w] = BG
    for spec, slot in zip(meta["rect_specs"], meta["slot_specs"]):
        patch = spec["final_patch"]
        ph, _pw = patch.shape
        dest_r = int(frame_bottom - ph + 1)
        dest_c = int(slot["c0"])
        _paste(out, patch, dest_r, dest_c)
    return out


def _wrong_axis_output(inp: np.ndarray) -> np.ndarray | None:
    return _solve(inp, force_blue_vertical=not _parse_blue_vertical(inp))


def _source_shortcut_output(inp: np.ndarray) -> np.ndarray | None:
    return _solve(inp, pair_mode="source")


def _validate_generated(meta: dict, inp: np.ndarray, expected: np.ndarray) -> bool:
    rects = _find_rectangles(inp)
    frame_box, slots = _find_frame_and_slots(inp)
    if len(rects) != len(meta["rect_specs"]):
        return False
    if [slot["width"] for slot in slots] != [slot["width"] for slot in meta["slot_specs"]]:
        return False
    if frame_box[0] != meta["frame_r0"] or frame_box[1] != meta["frame_c0"]:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    wrong = _wrong_axis_output(inp)
    if wrong is not None and np.array_equal(wrong, expected):
        return False
    shortcut = _source_shortcut_output(inp)
    if shortcut is not None and np.array_equal(shortcut, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(3, 5))
    slot_widths = WIDTH_SETS[count].copy()
    rng.shuffle(slot_widths)
    colors = [int(c) for c in rng.choice(np.asarray(RECT_PALETTE, dtype=int), size=count, replace=False)]
    blue_vertical = bool(rng.integers(0, 2))
    final_heights = _sample_final_heights(rng, slot_widths)
    if final_heights is None:
        return None

    meta = _build_layout(rng, colors, slot_widths, final_heights, blue_vertical)
    if meta is None:
        return None
    inp = _render_input(meta)
    expected = _render_expected(meta, inp)
    if not _validate_generated(meta, inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
