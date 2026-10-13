import random
import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _frame_info(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None
    for r, c in coords:
        if int(r) not in (r0, r1) and int(c) not in (c0, c1):
            return None
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return None
    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "h": h, "w": w}


def _find_slots(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    groups: dict[tuple[int, int, int, int, int], list[dict]] = {}
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components4(g == color):
            info = _frame_info(g, coords, bg)
            if info is None:
                continue
            key = (color, info["h"], info["w"], info["c0"], info["c1"])
            groups.setdefault(key, []).append(info)

    best_key = None
    best_slots: list[dict] | None = None
    for key, slots in groups.items():
        slots = sorted(slots, key=lambda item: item["r0"])
        score = (len(slots), slots[0]["h"] * slots[0]["w"])
        if best_slots is None or score > (len(best_slots), best_slots[0]["h"] * best_slots[0]["w"]):
            best_key = key
            best_slots = slots

    assert best_key is not None and best_slots is not None, "failed to find slot frames"
    return int(best_key[0]), best_slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each stacked rectangle, sort by visible area, and place them into the right-side slots.

    Input structure:
    - Background fills the grid.
    - The left side contains several unique-color solid rectangles that partially overlap.
    - The right side contains the same number of empty grey rectangular slots.

    Rule:
    - Count visible pixels of each colored rectangle in the stacked area.
    - Reconstruct each full rectangle from its visible color footprint.
    - Sort rectangles by visible pixel count from largest to smallest.
    - Clear the stacked area and place the full rectangles into the slots top-to-bottom.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    grey, slots = _find_slots(g, bg)

    items = []
    for color in map(int, np.unique(g)):
        if color in (bg, grey):
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        items.append(
            {
                "color": color,
                "visible": int(coords.shape[0]),
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "bbox": (r0, c0, r1, c1),
            }
        )

    items.sort(key=lambda item: (-item["visible"], item["bbox"], item["color"]))
    slots = sorted(slots, key=lambda item: item["r0"])
    assert len(items) == len(slots), "slot count must match rectangle count"

    out = g.copy()
    mask = (out != bg) & (out != grey)
    out[mask] = bg

    for item, slot in zip(items, slots):
        inner_r0 = slot["r0"] + 1
        inner_c0 = slot["c0"] + 1
        assert item["h"] <= slot["h"] - 2 and item["w"] <= slot["w"] - 2
        out[inner_r0 : inner_r0 + item["h"], inner_c0 : inner_c0 + item["w"]] = item["color"]

    return out


# --------------- generator ---------------

COLORS = tuple(range(10))
RECT_COUNT_RANGE = (3, 5)
HEIGHT_CAPS = {3: 6, 4: 5, 5: 4}
WIDTH_CAPS = {3: 8, 4: 7, 5: 6}
SLOT_GAP_RANGE = (1, 2)
STACK_SLOT_GAP_RANGE = (2, 4)
TOP_BOTTOM_MARGIN_RANGE = (1, 2)
LEFT_RIGHT_MARGIN_RANGE = (1, 2)
INNER_EXTRA_RANGE = (0, 1)
GEN_ATTEMPTS = 500


def _rect_area(rect: dict) -> int:
    return int((rect["r1"] - rect["r0"] + 1) * (rect["c1"] - rect["c0"] + 1))


def _render_input(
    h: int,
    w: int,
    bg: int,
    grey: int,
    slots: list[dict],
    rects: list[dict],
    order: list[int],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for slot in slots:
        r0, c0, r1, c1 = slot["r0"], slot["c0"], slot["r1"], slot["c1"]
        grid[r0, c0 : c1 + 1] = grey
        grid[r1, c0 : c1 + 1] = grey
        grid[r0 : r1 + 1, c0] = grey
        grid[r0 : r1 + 1, c1] = grey
    for idx in order:
        rect = rects[idx]
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return grid


def _expected_output(
    h: int,
    w: int,
    bg: int,
    grey: int,
    slots: list[dict],
    rects: list[dict],
    visible_counts: dict[int, int],
) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for slot in slots:
        r0, c0, r1, c1 = slot["r0"], slot["c0"], slot["r1"], slot["c1"]
        out[r0, c0 : c1 + 1] = grey
        out[r1, c0 : c1 + 1] = grey
        out[r0 : r1 + 1, c0] = grey
        out[r0 : r1 + 1, c1] = grey

    ranked = sorted(
        rects,
        key=lambda rect: (-visible_counts[rect["color"]], rect["r0"], rect["c0"], rect["color"]),
    )
    for slot, rect in zip(sorted(slots, key=lambda item: item["r0"]), ranked):
        rr = slot["r0"] + 1
        cc = slot["c0"] + 1
        rh = rect["r1"] - rect["r0"] + 1
        rw = rect["c1"] - rect["c0"] + 1
        out[rr : rr + rh, cc : cc + rw] = rect["color"]
    return out


def _sample_slots(n_rects: int, inner_h: int, inner_w: int, stack_w: int) -> tuple[int, int, list[dict]]:
    slot_h = inner_h + 2
    slot_w = inner_w + 2
    slot_gap = random.randint(*SLOT_GAP_RANGE)
    top_margin = random.randint(*TOP_BOTTOM_MARGIN_RANGE)
    bottom_margin = random.randint(*TOP_BOTTOM_MARGIN_RANGE)
    left_margin = random.randint(*LEFT_RIGHT_MARGIN_RANGE)
    right_margin = random.randint(*LEFT_RIGHT_MARGIN_RANGE)
    middle_gap = random.randint(*STACK_SLOT_GAP_RANGE)

    h = top_margin + n_rects * slot_h + (n_rects - 1) * slot_gap + bottom_margin
    w = left_margin + stack_w + middle_gap + slot_w + right_margin
    slot_c0 = left_margin + stack_w + middle_gap
    slots = []
    row = top_margin
    for _ in range(n_rects):
        slots.append({"r0": row, "c0": slot_c0, "r1": row + slot_h - 1, "c1": slot_c0 + slot_w - 1})
        row += slot_h + slot_gap
    return h, w, slots


def _sample_nested_dims(n_rects: int) -> list[tuple[int, int]] | None:
    height_cap = HEIGHT_CAPS[n_rects]
    width_cap = WIDTH_CAPS[n_rects]
    for _ in range(200):
        top_h = random.randint(2, min(3, height_cap))
        top_w = random.randint(2, min(4, width_cap))
        dims = [(top_h, top_w)]
        cur_h = top_h
        cur_w = top_w
        ok = True

        for _ in range(n_rects - 1):
            expanded = False
            for _ in range(80):
                add_top = random.randint(0, 1 if cur_h < height_cap else 0)
                add_bottom = random.randint(0, 1 if cur_h < height_cap else 0)
                add_left = random.randint(0, 2 if cur_w < width_cap else 0)
                add_right = random.randint(0, 2 if cur_w < width_cap else 0)
                if add_top + add_bottom + add_left + add_right == 0:
                    continue

                next_h = cur_h + add_top + add_bottom
                next_w = cur_w + add_left + add_right
                if next_h > height_cap or next_w > width_cap:
                    continue

                # If the inner rectangle keeps the same width, it must leave room
                # above and below; same idea for equal-height inner rectangles.
                if next_w == cur_w and next_h - cur_h < 2:
                    continue
                if next_h == cur_h and next_w - cur_w < 2:
                    continue

                cur_h = next_h
                cur_w = next_w
                dims.append((cur_h, cur_w))
                expanded = True
                break

            if not expanded:
                ok = False
                break

        if ok and len({h * w for h, w in dims}) == n_rects:
            return list(reversed(dims))
    return None


def _place_nested_rects(
    colors: list[int],
    dims: list[tuple[int, int]],
    h: int,
    stack_c_max: int,
) -> list[dict] | None:
    bottom_h, bottom_w = dims[0]
    if bottom_h > h or bottom_w > stack_c_max:
        return None

    base_r0 = random.randint(0, h - bottom_h)
    base_c0 = random.randint(0, stack_c_max - bottom_w)
    rects = [
        {
            "color": colors[0],
            "r0": base_r0,
            "c0": base_c0,
            "r1": base_r0 + bottom_h - 1,
            "c1": base_c0 + bottom_w - 1,
        }
    ]

    for color, (rh, rw) in zip(colors[1:], dims[1:]):
        outer = rects[-1]
        outer_h = outer["r1"] - outer["r0"] + 1
        outer_w = outer["c1"] - outer["c0"] + 1

        row_min = outer["r0"] + (1 if rw == outer_w else 0)
        row_max = outer["r1"] - rh + 1 - (1 if rw == outer_w else 0)
        col_min = outer["c0"] + (1 if rh == outer_h else 0)
        col_max = outer["c1"] - rw + 1 - (1 if rh == outer_h else 0)
        if row_min > row_max or col_min > col_max:
            return None

        r0 = random.randint(row_min, row_max)
        c0 = random.randint(col_min, col_max)
        rects.append({"color": color, "r0": r0, "c0": c0, "r1": r0 + rh - 1, "c1": c0 + rw - 1})

    return rects


def _visibility_ok(grid: np.ndarray, rects: list[dict], order: list[int]) -> tuple[bool, dict[int, int]]:
    visible_counts: dict[int, int] = {}
    touched = 0
    for rect in rects:
        color = rect["color"]
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return False, {}
        rows = coords[:, 0]
        cols = coords[:, 1]
        bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
        full_bbox = (rect["r0"], rect["c0"], rect["r1"], rect["c1"])
        if bbox != full_bbox:
            return False, {}
        visible_counts[color] = int(coords.shape[0])
        if visible_counts[color] < _rect_area(rect):
            touched += 1

    if len(set(visible_counts.values())) != len(rects):
        return False, {}

    top_order = tuple(rects[idx]["color"] for idx in reversed(order))
    visible_order = tuple(
        rect["color"]
        for rect in sorted(rects, key=lambda item: (-visible_counts[item["color"]], item["r0"], item["c0"], item["color"]))
    )
    if visible_order == top_order:
        return False, {}
    if touched < max(1, len(rects) - 2):
        return False, {}
    return True, visible_counts


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    palette = list(COLORS)
    bg, grey = random.sample(palette, 2)
    rect_colors = random.sample([c for c in palette if c not in (bg, grey)], n_rects)

    dims = _sample_nested_dims(n_rects)
    if dims is None:
        return None
    max_rect_h, max_rect_w = dims[0]
    inner_h = max_rect_h + random.randint(*INNER_EXTRA_RANGE)
    inner_w = max_rect_w + random.randint(*INNER_EXTRA_RANGE)
    stack_w = max(inner_w + 4, random.randint(inner_w + 4, inner_w + 10))
    h, w, slots = _sample_slots(n_rects, inner_h, inner_w, stack_w)
    stack_c_max = slots[0]["c0"] - 2
    if stack_c_max <= max_rect_w:
        return None

    rects = _place_nested_rects(rect_colors, dims, h, stack_c_max)
    if rects is None:
        return None
    order = list(range(n_rects))

    inp = _render_input(h, w, bg, grey, slots, rects, order)
    if _most_common_color(inp) != bg:
        return None

    ok, visible_counts = _visibility_ok(inp, rects, order)
    if not ok:
        return None

    expected = _expected_output(h, w, bg, grey, slots, rects, visible_counts)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
