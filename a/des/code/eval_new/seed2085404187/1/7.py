from __future__ import annotations

import random

import numpy as np


GREEN = 3
YELLOW = 4
RESERVED = {GREEN, YELLOW}
BG_CHOICES = [0, 1, 2, 5, 6, 7, 8, 9]
FG_CHOICES = [0, 1, 2, 5, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
PATTERN_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 80
ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _rotation_pairs(before: np.ndarray, after: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot_before = np.rot90(before, k)
        key = (tuple(map(int, rot_before.shape)), rot_before.tobytes())
        if key in seen:
            continue
        seen.add(key)
        pairs.append((rot_before, np.rot90(after, k)))
    return pairs


def _split_regions(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice], tuple[slice, slice]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_dividers = [r for r in range(h) if np.all(g[r, :] == GREEN)]
    col_dividers = [c for c in range(w) if np.all(g[:, c] == GREEN)]
    assert bool(row_dividers) != bool(col_dividers), "expected exactly one green separator orientation"

    if row_dividers:
        assert len(row_dividers) == 1, "expected a single horizontal separator row"
        r = row_dividers[0]
        top_slice = (slice(0, r), slice(0, w))
        bottom_slice = (slice(r + 1, h), slice(0, w))
        top = g[top_slice]
        bottom = g[bottom_slice]
        if top.size < bottom.size:
            return top, bottom, top_slice, bottom_slice
        assert bottom.size < top.size, "sample window must be strictly smaller than main region"
        return bottom, top, bottom_slice, top_slice

    assert len(col_dividers) == 1, "expected a single vertical separator column"
    c = col_dividers[0]
    left_slice = (slice(0, h), slice(0, c))
    right_slice = (slice(0, h), slice(c + 1, w))
    left = g[left_slice]
    right = g[right_slice]
    if left.size < right.size:
        return left, right, left_slice, right_slice
    assert right.size < left.size, "sample window must be strictly smaller than main region"
    return right, left, right_slice, left_slice


def _extract_demo(sample_region: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    sample = np.asarray(sample_region, dtype=int)
    yellow_mask = sample == YELLOW
    assert yellow_mask.any(), "expected a yellow sample separator"

    r0, c0, r1, c1 = _bbox_from_mask(yellow_mask)
    assert c0 == c1, "yellow separator must be one column wide"
    assert np.all(sample[r0 : r1 + 1, c0] == YELLOW), "yellow separator must be contiguous"

    band = sample[r0 : r1 + 1, :]
    left_non_bg = np.argwhere(band[:, :c0] != bg)
    right_non_bg = np.argwhere(band[:, c0 + 1 :] != bg)
    assert left_non_bg.size > 0 and right_non_bg.size > 0, "expected patterns on both sides of the separator"

    left_rows = left_non_bg[:, 0]
    right_rows = right_non_bg[:, 0]
    assert int(left_rows.min()) == 0 and int(left_rows.max()) == r1 - r0, "before pattern must span separator height"
    assert int(right_rows.min()) == 0 and int(right_rows.max()) == r1 - r0, "after pattern must span separator height"

    left_c0 = int(left_non_bg[:, 1].min())
    left_c1 = int(left_non_bg[:, 1].max())
    right_c0 = int(right_non_bg[:, 1].min())
    right_c1 = int(right_non_bg[:, 1].max())

    before = band[:, left_c0 : left_c1 + 1].copy()
    after = band[:, c0 + 1 + right_c0 : c0 + 1 + right_c1 + 1].copy()
    assert before.shape == after.shape, "before/after sample patches must have the same size"
    return before, after


def main(grid: np.ndarray) -> np.ndarray:
    """Learn a local recoloring patch from the sample window and apply it to every rotated copy in the main region.

    Input structure:
    - The background color is the most common color in the grid.
    - A full green separator row or column splits the grid into a smaller sample window and a larger main region.
    - Inside the sample window, a before-patch and an after-patch of the same size are arranged horizontally with a one-column yellow separator between them.

    Rule:
    - Extract the before/after pair from the sample window.
    - Search the main region for every exact occurrence of the before-patch under the four 90-degree rotations.
    - For each match, replace that local patch with the correspondingly rotated after-patch.
    - Leave the sample window, the separator line, and every non-matching area unchanged.

    Key insight:
    - The sample window encodes a local replacement rule once, while the main region may contain rotated instances that need the same rule rotated before application.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    sample_region, main_region, _sample_slice, main_slice = _split_regions(g)
    before, after = _extract_demo(sample_region, bg)

    main_view = g[main_slice]
    out = g.copy()
    out_main = out[main_slice]
    occupied = np.zeros(main_view.shape, dtype=bool)

    for rot_before, rot_after in _rotation_pairs(before, after):
        ph, pw = rot_before.shape
        for r in range(main_view.shape[0] - ph + 1):
            for c in range(main_view.shape[1] - pw + 1):
                patch = main_view[r : r + ph, c : c + pw]
                if not np.array_equal(patch, rot_before):
                    continue
                assert not occupied[r : r + ph, c : c + pw].any(), "matched instances must not overlap"
                out_main[r : r + ph, c : c + pw] = rot_after
                occupied[r : r + ph, c : c + pw] = True

    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    low = max(3, (area + 2) // 3)
    high = area
    target = random.randint(low, high)

    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ORTHO_DIRS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                frontier.add((rr, cc))

    add_frontier(*seed)
    while len(cells) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if _bbox_from_mask(mask) != (0, 0, h - 1, w - 1):
        return None
    return mask


def _assign_colors(mask: np.ndarray, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    fg_coords = np.argwhere(mask)
    palette = [color for color in FG_CHOICES if color != bg]
    color_count = random.randint(1, min(3, len(fg_coords)))
    chosen = random.sample(palette, color_count)
    values = chosen + [random.choice(chosen) for _ in range(len(fg_coords) - color_count)]
    random.shuffle(values)
    for (r, c), color in zip(fg_coords, values):
        patch[int(r), int(c)] = int(color)
    return patch


def _make_demo_pair(bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATTERN_ATTEMPTS):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        mask = _random_connected_mask(h, w)
        if mask is None:
            continue

        before = _assign_colors(mask, bg)
        rot_keys = {
            (tuple(map(int, np.rot90(before, k).shape)), np.rot90(before, k).tobytes()) for k in range(4)
        }
        if len(rot_keys) != 4:
            continue

        fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        if len(fg_coords) < 2:
            continue
        changed = random.sample(fg_coords, random.randint(1, min(4, len(fg_coords) - 1)))
        after = before.copy()
        for r, c in changed:
            choices = [color for color in FG_CHOICES if color not in {bg, int(after[r, c])}]
            after[r, c] = random.choice(choices)
        if np.array_equal(before, after):
            continue
        return before, after
    return None


def _make_sample_window(before: np.ndarray, after: np.ndarray, bg: int) -> np.ndarray:
    h, w = before.shape
    pad_top = random.randint(0, 2)
    pad_bottom = random.randint(0, 2)
    pad_left = random.randint(0, 1)
    pad_right = random.randint(0, 1)
    rows = h + pad_top + pad_bottom
    cols = 2 * w + 1 + pad_left + pad_right
    sample = np.full((rows, cols), bg, dtype=int)
    sample[pad_top : pad_top + h, pad_left : pad_left + w] = before
    sample[pad_top : pad_top + h, pad_left + w] = YELLOW
    sample[pad_top : pad_top + h, pad_left + w + 1 : pad_left + 2 * w + 1] = after
    return sample


def _separated(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        box1[2] + gap < box2[0]
        or box2[2] + gap < box1[0]
        or box1[3] + gap < box2[1]
        or box2[3] + gap < box1[1]
    )


def _place_arrays(
    canvas_shape: tuple[int, int], arrays: list[np.ndarray]
) -> list[tuple[np.ndarray, int, int]] | None:
    h, w = canvas_shape
    placed: list[tuple[np.ndarray, int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []

    for arr in arrays:
        ah, aw = arr.shape
        options = []
        for r in range(h - ah + 1):
            for c in range(w - aw + 1):
                box = (r, c, r + ah - 1, c + aw - 1)
                if all(_separated(box, other) for other in boxes):
                    options.append((r, c, box))
        if not options:
            return None
        r, c, box = random.choice(options)
        placed.append((arr, r, c))
        boxes.append(box)
    return placed


def _make_decoy(before: np.ndarray, bg: int) -> np.ndarray:
    rot = np.rot90(before, random.randrange(4)).copy()
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(rot != bg)]
    r, c = random.choice(fg_coords)
    choices = [color for color in FG_CHOICES if color not in {bg, int(rot[r, c])}]
    rot[r, c] = random.choice(choices)
    return rot


def _render_main_side(
    side_shape: tuple[int, int], before: np.ndarray, after: np.ndarray, bg: int
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = side_shape
    instance_count = random.randint(2, 5)
    rotations = [random.randrange(4) for _ in range(instance_count)]
    before_arrays = [np.rot90(before, k) for k in rotations]
    after_arrays = [np.rot90(after, k) for k in rotations]

    decoy_count = random.randint(0, 2)
    decoys = [_make_decoy(before, bg) for _ in range(decoy_count)]
    arrays = before_arrays + decoys
    placed = _place_arrays((h, w), arrays)
    if placed is None:
        return None

    main_in = np.full((h, w), bg, dtype=int)
    main_out = np.full((h, w), bg, dtype=int)
    for idx, (arr, r, c) in enumerate(placed):
        ah, aw = arr.shape
        main_in[r : r + ah, c : c + aw] = arr
        main_out[r : r + ah, c : c + aw] = arr
        if idx < instance_count:
            main_out[r : r + ah, c : c + aw] = after_arrays[idx]

    return main_in, main_out


def _embed_window(canvas: np.ndarray, window: np.ndarray) -> None:
    h, w = canvas.shape
    wh, ww = window.shape
    r0 = random.randint(0, h - wh)
    c0 = random.randint(0, w - ww)
    canvas[r0 : r0 + wh, c0 : c0 + ww] = window


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(BG_CHOICES)
    demo = _make_demo_pair(bg)
    if demo is None:
        return None
    before, after = demo
    sample_window = _make_sample_window(before, after, bg)
    sample_h, sample_w = sample_window.shape

    if random.random() < 0.5:
        full_h = random.randint(max(sample_h, 7), 15)
        sample_side_w = sample_w + random.randint(0, 1)
        main_side_w_min = max(sample_side_w + 1, 8)
        main_side_w_max = min(15, 29 - sample_side_w)
        if main_side_w_min > main_side_w_max:
            return None
        main_side_w = random.randint(main_side_w_min, main_side_w_max)

        sample_side = np.full((full_h, sample_side_w), bg, dtype=int)
        if sample_w > sample_side_w:
            return None
        _embed_window(sample_side, sample_window)

        rendered = _render_main_side((full_h, main_side_w), before, after, bg)
        if rendered is None:
            return None
        main_side_in, main_side_out = rendered

        divider = np.full((full_h, 1), GREEN, dtype=int)
        if random.random() < 0.5:
            inp = np.hstack([sample_side, divider, main_side_in])
            out = np.hstack([sample_side, divider, main_side_out])
        else:
            inp = np.hstack([main_side_in, divider, sample_side])
            out = np.hstack([main_side_out, divider, sample_side])
    else:
        full_w = random.randint(max(sample_w, 8), 15)
        sample_side_h = sample_h + random.randint(0, 1)
        main_side_h_min = max(sample_side_h + 1, 8)
        main_side_h_max = min(15, 29 - sample_side_h)
        if main_side_h_min > main_side_h_max:
            return None
        main_side_h = random.randint(main_side_h_min, main_side_h_max)

        sample_side = np.full((sample_side_h, full_w), bg, dtype=int)
        if sample_h > sample_side_h:
            return None
        _embed_window(sample_side, sample_window)

        rendered = _render_main_side((main_side_h, full_w), before, after, bg)
        if rendered is None:
            return None
        main_side_in, main_side_out = rendered

        divider = np.full((1, full_w), GREEN, dtype=int)
        if random.random() < 0.5:
            inp = np.vstack([sample_side, divider, main_side_in])
            out = np.vstack([sample_side, divider, main_side_out])
        else:
            inp = np.vstack([main_side_in, divider, sample_side])
            out = np.vstack([main_side_out, divider, sample_side])

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for task 7")
