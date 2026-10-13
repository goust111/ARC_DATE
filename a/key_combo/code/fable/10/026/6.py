import random

import numpy as np


BLACK = 0
GREY = 5
AVAILABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_separator(grid: np.ndarray) -> tuple[str, int]:
    grey_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    grey_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    if len(grey_rows) == 1 and len(grey_cols) == 0:
        assert int(np.count_nonzero(grid == GREY)) == grid.shape[1]
        return "h", int(grey_rows[0])
    if len(grey_cols) == 1 and len(grey_rows) == 0:
        assert int(np.count_nonzero(grid == GREY)) == grid.shape[0]
        return "v", int(grey_cols[0])
    raise AssertionError("expected exactly one full grey separator line")


def _slot_color(block: np.ndarray, bg: int) -> int:
    non_bg = np.argwhere(block != bg)
    if non_bg.size == 0:
        return int(bg)

    colors = np.unique(block[block != bg])
    assert len(colors) == 1, "slot bar must be monochromatic"
    rows = np.unique(non_bg[:, 0])
    cols = np.unique(non_bg[:, 1])
    mid = block.shape[0] // 2
    assert block.shape[0] % 2 == 1, "slot height must be odd"
    assert rows.size == 1 and int(rows[0]) == mid, "bar must sit on the slot middle row"
    assert np.array_equal(cols, np.arange(block.shape[1])), "bar must span the full slot width"
    return int(colors[0])


def _build_canvas(slot_colors: np.ndarray, slot_h: int, slot_w: int, bg: int) -> np.ndarray:
    rows, cols = slot_colors.shape
    canvas = np.full((rows * slot_h, cols * slot_w), bg, dtype=int)
    mid = slot_h // 2
    for r in range(rows):
        for c in range(cols):
            color = int(slot_colors[r, c])
            if color == bg:
                continue
            r0 = r * slot_h
            c0 = c * slot_w
            canvas[r0 + mid, c0 : c0 + slot_w] = color
    return canvas


def _extract_slot_colors(canvas: np.ndarray, stamp_shape: tuple[int, int], bg: int) -> np.ndarray:
    stamp_h, stamp_w = stamp_shape
    assert canvas.shape[0] % stamp_h == 0 and canvas.shape[1] % stamp_w == 0
    slot_h = canvas.shape[0] // stamp_h
    slot_w = canvas.shape[1] // stamp_w
    assert slot_h > 1 and slot_w > 1
    slot_colors = np.full(stamp_shape, bg, dtype=int)
    for r in range(stamp_h):
        for c in range(stamp_w):
            r0 = r * slot_h
            c0 = c * slot_w
            slot_colors[r, c] = _slot_color(canvas[r0 : r0 + slot_h, c0 : c0 + slot_w], bg)
    return slot_colors


def _parse_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    assert bg == BLACK, "background is expected to be black"

    axis, sep = _find_separator(g)
    H, W = g.shape
    if axis == "h":
        top = g[:sep, :]
        bottom = g[sep + 1 :, :]
        assert top.size > 0 and bottom.size > 0
        strip_is_top = top.shape[0] < bottom.shape[0]
        strip = top if strip_is_top else bottom
        canvas = bottom if strip_is_top else top
        strip_bbox = _bbox(strip != bg)
        assert strip_bbox is not None
        r0, c0, r1, c1 = strip_bbox
        assert r0 == 0 and r1 == strip.shape[0] - 1
        assert c0 == 0 or c1 == strip.shape[1] - 1
        stamp = strip[:, c0 : c1 + 1].copy()
        canvas_r0 = sep + 1 if strip_is_top else 0
        canvas_c0 = 0
    else:
        left = g[:, :sep]
        right = g[:, sep + 1 :]
        assert left.size > 0 and right.size > 0
        strip_is_left = left.shape[1] < right.shape[1]
        strip = left if strip_is_left else right
        canvas = right if strip_is_left else left
        strip_bbox = _bbox(strip != bg)
        assert strip_bbox is not None
        r0, c0, r1, c1 = strip_bbox
        assert c0 == 0 and c1 == strip.shape[1] - 1
        assert r0 == 0 or r1 == strip.shape[0] - 1
        stamp = strip[r0 : r1 + 1, :].copy()
        canvas_r0 = 0
        canvas_c0 = sep + 1 if strip_is_left else 0

    stamp_h, stamp_w = stamp.shape
    assert stamp_h >= 2 and stamp_w >= 2
    assert np.all(np.any(stamp != bg, axis=1)), "each stamp row must reveal itself"
    assert np.all(np.any(stamp != bg, axis=0)), "each stamp column must reveal itself"

    canvas_h, canvas_w = canvas.shape
    assert canvas_h % stamp_h == 0 and canvas_w % stamp_w == 0
    slot_h = canvas_h // stamp_h
    slot_w = canvas_w // stamp_w
    assert slot_h > 1 and slot_w > 1 and slot_h % 2 == 1

    slot_colors = _extract_slot_colors(canvas, (stamp_h, stamp_w), bg)
    return {
        "bg": bg,
        "axis": axis,
        "sep": sep,
        "stamp": stamp,
        "slot_colors": slot_colors,
        "slot_h": slot_h,
        "slot_w": slot_w,
        "canvas_r0": canvas_r0,
        "canvas_c0": canvas_c0,
        "canvas_h": canvas_h,
        "canvas_w": canvas_w,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the corner stamp into the slot canvas by fixing horizontal bars to the mirrored blueprint.

    Input structure
    - A single grey straight line separates a large canvas from a narrow strip that
      contains one small corner stamp rectangle.
    - The stamp cells are black or foreground-colored, and the canvas can be split
      into a same-shaped array of equal slots.
    - Some canvas slots already contain full-width horizontal bars on their middle
      rows, but those bars may be missing, extra, or the wrong color.

    Rule
    - Read the corner stamp as a coarse blueprint.
    - For each stamp cell `(r, c)` with color `x`, the mirrored canvas slot
      `(r, width-1-c)` must contain a middle-row full-width bar of color `x`.
    - If the stamp cell is black, that mirrored slot must be empty.
    - Keep the stamp and grey separator unchanged, and rebuild the canvas so every
      slot matches that mirrored rule exactly.

    Key insight
    - The small block is not a count legend and not a same-direction map: it is a
      miniature slot blueprint whose imprint appears only after a left-right mirror.
    """
    info = _parse_layout(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    target_slots = np.fliplr(info["stamp"])
    canvas = _build_canvas(target_slots, info["slot_h"], info["slot_w"], info["bg"])
    r0 = info["canvas_r0"]
    c0 = info["canvas_c0"]
    out[r0 : r0 + info["canvas_h"], c0 : c0 + info["canvas_w"]] = canvas
    return out


def _random_stamp(rows: int, cols: int, palette: tuple[int, ...]) -> np.ndarray | None:
    total = rows * cols
    min_colored = max(rows + cols - 1, 4)
    max_colored = total - 1
    if max_colored < min_colored:
        return None

    for _ in range(200):
        stamp = np.zeros((rows, cols), dtype=int)
        colored = set()
        for r in range(rows):
            colored.add((r, random.randrange(cols)))
        for c in range(cols):
            colored.add((random.randrange(rows), c))

        want = random.randint(max(min_colored, len(colored)), max_colored)
        while len(colored) < want:
            colored.add((random.randrange(rows), random.randrange(cols)))

        for r, c in colored:
            stamp[r, c] = random.choice(palette)

        if np.count_nonzero(stamp) < 4 or np.count_nonzero(stamp == 0) < 1:
            continue
        if len(np.unique(stamp[stamp != 0])) < 2:
            continue
        if np.array_equal(stamp, np.fliplr(stamp)):
            continue
        decoy_candidates = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if stamp[r, c] != 0 and stamp[r, c] != stamp[r, cols - 1 - c]
        ]
        if not decoy_candidates:
            continue
        return stamp
    return None


def _choose_wrong_color(target_color: int, palette: tuple[int, ...]) -> int:
    return random.choice([color for color in palette if color != target_color])


def _perturb_slots(stamp: np.ndarray, palette: tuple[int, ...]) -> tuple[np.ndarray, set[str], tuple[int, int]]:
    target = np.fliplr(stamp)
    rows, cols = target.shape
    actual = target.copy()

    colored_slots = [(r, c) for r in range(rows) for c in range(cols) if target[r, c] != 0]
    empty_slots = [(r, c) for r in range(rows) for c in range(cols) if target[r, c] == 0]
    decoy_candidates = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if stamp[r, c] != 0 and stamp[r, c] != target[r, c]
    ]
    assert len(colored_slots) >= 3 and empty_slots and decoy_candidates

    categories: set[str] = set()
    changed: set[tuple[int, int]] = set()
    decoy = random.choice(decoy_candidates)
    actual[decoy] = stamp[decoy]
    changed.add(decoy)
    categories.add("wrong" if target[decoy] != 0 else "extra")

    missing_slot = random.choice([pos for pos in colored_slots if pos not in changed])
    actual[missing_slot] = 0
    changed.add(missing_slot)
    categories.add("missing")

    if "wrong" not in categories:
        wrong_slot = random.choice([pos for pos in colored_slots if pos not in changed])
        actual[wrong_slot] = _choose_wrong_color(int(target[wrong_slot]), palette)
        changed.add(wrong_slot)
        categories.add("wrong")

    if "extra" not in categories:
        extra_slot = random.choice([pos for pos in empty_slots if pos not in changed])
        actual[extra_slot] = stamp[extra_slot] if stamp[extra_slot] != 0 else random.choice(palette)
        changed.add(extra_slot)
        categories.add("extra")

    untouched_colored = [pos for pos in colored_slots if pos not in changed]
    correct_slot = random.choice(untouched_colored)
    actual[correct_slot] = target[correct_slot]
    categories.add("correct")

    for r, c in colored_slots:
        if (r, c) in changed or (r, c) == correct_slot:
            continue
        roll = random.random()
        if roll < 0.50:
            actual[r, c] = target[r, c]
        elif roll < 0.75:
            actual[r, c] = 0
        else:
            actual[r, c] = _choose_wrong_color(int(target[r, c]), palette)

    for r, c in empty_slots:
        if (r, c) in changed:
            continue
        if random.random() < 0.25:
            actual[r, c] = stamp[r, c] if stamp[r, c] != 0 and random.random() < 0.7 else random.choice(palette)
        else:
            actual[r, c] = 0

    return actual, categories, decoy


def _render_layout(
    stamp: np.ndarray,
    actual_slots: np.ndarray,
    slot_h: int,
    slot_w: int,
    axis: str,
    strip_side: str,
    corner_side: str,
) -> np.ndarray:
    stamp_h, stamp_w = stamp.shape
    canvas = _build_canvas(actual_slots, slot_h, slot_w, BLACK)
    canvas_h, canvas_w = canvas.shape

    if axis == "h":
        H = stamp_h + 1 + canvas_h
        W = canvas_w
        grid = np.zeros((H, W), dtype=int)
        if strip_side == "top":
            sep = stamp_h
            strip_r0 = 0
            canvas_r0 = sep + 1
        else:
            sep = canvas_h
            canvas_r0 = 0
            strip_r0 = sep + 1
        grid[sep, :] = GREY
        grid[canvas_r0 : canvas_r0 + canvas_h, :] = canvas
        stamp_c0 = 0 if corner_side == "left" else W - stamp_w
        grid[strip_r0 : strip_r0 + stamp_h, stamp_c0 : stamp_c0 + stamp_w] = stamp
        return grid

    H = canvas_h
    W = stamp_w + 1 + canvas_w
    grid = np.zeros((H, W), dtype=int)
    if strip_side == "left":
        sep = stamp_w
        strip_c0 = 0
        canvas_c0 = sep + 1
    else:
        sep = canvas_w
        canvas_c0 = 0
        strip_c0 = sep + 1
    grid[:, sep] = GREY
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas
    stamp_r0 = 0 if corner_side == "top" else H - stamp_h
    grid[stamp_r0 : stamp_r0 + stamp_h, strip_c0 : strip_c0 + stamp_w] = stamp
    return grid


def _validate_generated(inp: np.ndarray, stamp: np.ndarray, actual_slots: np.ndarray, categories: set[str], decoy: tuple[int, int]) -> bool:
    if categories != {"correct", "missing", "wrong", "extra"}:
        return False

    expected = main(inp)
    info = _parse_layout(inp)
    target = np.fliplr(stamp)
    if not np.array_equal(info["stamp"], stamp):
        return False
    if not np.array_equal(info["slot_colors"], actual_slots):
        return False

    expected_canvas = _build_canvas(target, info["slot_h"], info["slot_w"], BLACK)
    r0 = info["canvas_r0"]
    c0 = info["canvas_c0"]
    if not np.array_equal(expected[r0 : r0 + info["canvas_h"], c0 : c0 + info["canvas_w"]], expected_canvas):
        return False

    same_direction = _build_canvas(stamp, info["slot_h"], info["slot_w"], BLACK)
    if np.array_equal(expected_canvas, same_direction):
        return False

    dr, dc = decoy
    if actual_slots[dr, dc] != stamp[dr, dc]:
        return False
    if stamp[dr, dc] == 0 or actual_slots[dr, dc] == target[dr, dc]:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    stamp_rows, stamp_cols = random.choice(
        ((2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3))
    )
    slot_h = random.choice((3, 5))
    slot_w = random.choice((3, 4, 5, 6))
    palette = tuple(random.sample(AVAILABLE_COLORS, random.randint(2, 4)))

    stamp = _random_stamp(stamp_rows, stamp_cols, palette)
    if stamp is None:
        return None

    actual_slots, categories, decoy = _perturb_slots(stamp, palette)
    axis = random.choice(("h", "v"))
    if axis == "h":
        strip_side = random.choice(("top", "bottom"))
        corner_side = random.choice(("left", "right"))
    else:
        strip_side = random.choice(("left", "right"))
        corner_side = random.choice(("top", "bottom"))

    inp = _render_layout(stamp, actual_slots, slot_h, slot_w, axis, strip_side, corner_side)
    if max(inp.shape) > 30:
        return None
    if not _validate_generated(inp, stamp, actual_slots, categories, decoy):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-imprint puzzle")
