from __future__ import annotations

import numpy as np


def _most_common(a: np.ndarray) -> int:
    values, counts = np.unique(a, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _palette_from_row(row: np.ndarray, bg: int) -> list[int]:
    colors = [int(v) for v in row if int(v) != int(bg)]
    assert 2 <= len(colors) <= 4
    assert len(set(colors)) == len(colors)
    return colors


def _frame_bbox(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(g == int(color))
    assert len(pts) > 0, f"missing frame color {color}"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3

    mask = g == int(color)
    border = np.zeros_like(mask, dtype=bool)
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    assert bool(np.all(mask == border)), "frame color must form exactly one hollow rectangle"
    return r0, c0, r1, c1


def _initial_layer_order(body: np.ndarray, palette: list[int]) -> list[int]:
    boxes = []
    for color in palette:
        r0, c0, r1, c1 = _frame_bbox(body, color)
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        boxes.append((area, color, (r0, c0, r1, c1)))
    boxes.sort(reverse=True)

    for (_area_a, _color_a, outer), (_area_b, _color_b, inner) in zip(boxes, boxes[1:]):
        or0, oc0, or1, oc1 = outer
        ir0, ic0, ir1, ic1 = inner
        assert or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1
    return [int(color) for _area, color, _box in boxes]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate nested frame interiors according to the top-row palette order.

    Input structure:
    - The first row is a palette of 2 to 4 distinct frame colors, read from left
      to right among the non-background cells.
    - Below the palette are nested one-cell-thick hollow rectangular frames with
      distinct colors; the innermost frame contains a small multicolor motif.
    - Frame colors appear only in their own rectangular borders, so each frame
      can be located by its color.

    Rule:
    - Remove the palette row from the output.
    - The palette position of a frame color, starting at 1, gives the number of
      clockwise quarter turns for that frame.
    - Process frames from outermost to innermost. At each frame, rotate only the
      current contents strictly inside its border; the frame border itself stays
      fixed during that step.

    Key insight:
    - The visible palette order is a rotation legend, and rotations accumulate
      because every inner frame is part of the content rotated by all outer
      frames before its own interior is rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    body = g[1:, :].copy()
    bg = _most_common(body)
    palette = _palette_from_row(g[0], bg)
    turns_by_color = {color: i + 1 for i, color in enumerate(palette)}
    layer_order = _initial_layer_order(body, palette)

    out = body.copy()
    for color in layer_order:
        r0, c0, r1, c1 = _frame_bbox(out, color)
        interior = out[r0 + 1 : r1, c0 + 1 : c1].copy()
        turns = int(turns_by_color[color] % 4)
        rotated = np.rot90(interior, k=-turns)
        assert rotated.shape == interior.shape
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated
    return out


# --------------- generator ---------------

COLOR_COUNT = 10
LAYER_COUNT_CHOICES = (2, 3, 4)
MAX_FRAME_DIM = 27
MAX_BODY_H = 29
MAX_BODY_W = 30
MIN_FRAME_DIM = 5
MAX_GENERATION_ATTEMPTS = 300


def _odd_values(lo: int, hi: int) -> list[int]:
    start = int(lo) if int(lo) % 2 else int(lo) + 1
    return list(range(start, int(hi) + 1, 2))


def _choose_odd(rng: np.random.Generator, lo: int, hi: int) -> int:
    values = _odd_values(lo, hi)
    assert values
    return int(values[int(rng.integers(len(values)))])


def _choose_rect_dims(
    rng: np.random.Generator, min_h: int, min_w: int, max_h: int, max_w: int
) -> tuple[int, int] | None:
    hs = _odd_values(min_h, max_h)
    ws = _odd_values(min_w, max_w)
    if not hs or not ws:
        return None
    h = int(hs[int(rng.integers(len(hs)))])
    w = int(ws[int(rng.integers(len(ws)))])
    if h == w and len(ws) > 1 and rng.random() < 0.8:
        choices = [x for x in ws if x != h]
        w = int(choices[int(rng.integers(len(choices)))])
    elif h == w and len(hs) > 1 and rng.random() < 0.8:
        choices = [x for x in hs if x != w]
        h = int(choices[int(rng.integers(len(choices)))])
    return h, w


def _choose_frame_dims(rng: np.random.Generator, turns: list[int]) -> list[tuple[int, int]] | None:
    count = len(turns)
    max_inner = MAX_FRAME_DIM - 4 * (count - 1)
    if max_inner < MIN_FRAME_DIM:
        return None

    if turns[-1] % 2:
        side = _choose_odd(rng, MIN_FRAME_DIM, max_inner)
        dims = [(side, side)]
    else:
        chosen = _choose_rect_dims(
            rng, MIN_FRAME_DIM, MIN_FRAME_DIM, max_inner, max_inner
        )
        if chosen is None:
            return None
        dims = [chosen]

    for idx in range(count - 2, -1, -1):
        child_h, child_w = dims[0]
        max_for_layer = MAX_FRAME_DIM - 4 * idx
        if turns[idx] % 2:
            side_min = max(child_h, child_w) + 4
            choices = _odd_values(side_min, max_for_layer)
            if not choices:
                return None
            side = int(choices[int(rng.integers(len(choices)))])
            dims.insert(0, (side, side))
        else:
            chosen = _choose_rect_dims(
                rng,
                child_h + 4,
                child_w + 4,
                max_for_layer,
                max_for_layer,
            )
            if chosen is None:
                return None
            dims.insert(0, chosen)
    return dims


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _make_core(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    colors: list[int],
) -> np.ndarray:
    core = np.full((h, w), int(bg), dtype=int)
    ph = int(rng.integers(2, min(6, h) + 1))
    pw = int(rng.integers(2, min(6, w) + 1))
    r0 = int(rng.integers(0, h - ph + 1))
    c0 = int(rng.integers(0, w - pw + 1))

    required = list(dict.fromkeys([(0, 0), (0, pw - 1), (ph - 1, max(0, pw // 2))]))
    all_cells = [(r, c) for r in range(ph) for c in range(pw) if (r, c) not in required]
    max_count = min(ph * pw, max(len(required), int(np.ceil(ph * pw * 0.7))))
    min_count = min(ph * pw, max(3, len(required)))
    count = int(rng.integers(min_count, max_count + 1))
    need = count - len(required)
    if need > 0:
        picked = rng.choice(len(all_cells), size=need, replace=False)
        required.extend(all_cells[int(i)] for i in np.atleast_1d(picked))
    rng.shuffle(required)

    color_count = int(rng.integers(2, min(4, len(colors)) + 1))
    motif_colors = [int(c) for c in rng.choice(colors, size=color_count, replace=False)]
    for i, (r, c) in enumerate(required):
        core[r0 + int(r), c0 + int(c)] = motif_colors[i % color_count]
    return core


def _render_input(
    rng: np.random.Generator,
    bg: int,
    layer_colors: list[int],
    palette: list[int],
    dims: list[tuple[int, int]],
    core_colors: list[int],
) -> np.ndarray | None:
    outer_h, outer_w = dims[0]
    top_space = int(rng.integers(0, min(3, MAX_BODY_H - outer_h) + 1))
    bottom_max = MAX_BODY_H - outer_h - top_space
    left_space = int(rng.integers(0, min(3, MAX_BODY_W - outer_w) + 1))
    right_max = MAX_BODY_W - outer_w - left_space
    if bottom_max < 0 or right_max < 0:
        return None
    bottom_space = int(rng.integers(0, min(3, bottom_max) + 1))
    right_space = int(rng.integers(0, min(3, right_max) + 1))

    body_h = int(top_space + outer_h + bottom_space)
    body_w = int(left_space + outer_w + right_space)
    if body_h > MAX_BODY_H or body_w > MAX_BODY_W:
        return None

    body = np.full((body_h, body_w), int(bg), dtype=int)
    center_r = int(top_space + outer_h // 2)
    center_c = int(left_space + outer_w // 2)

    for color, (h, w) in zip(layer_colors, dims):
        r0 = int(center_r - h // 2)
        c0 = int(center_c - w // 2)
        _draw_frame(body, r0, c0, h, w, color)

    inner_h, inner_w = dims[-1]
    inner_r0 = int(center_r - inner_h // 2 + 1)
    inner_c0 = int(center_c - inner_w // 2 + 1)
    core = _make_core(rng, inner_h - 2, inner_w - 2, bg, core_colors)
    body[inner_r0 : inner_r0 + core.shape[0], inner_c0 : inner_c0 + core.shape[1]] = core

    top = np.full((1, body_w), int(bg), dtype=int)
    start = int(rng.integers(0, body_w - len(palette) + 1))
    for i, color in enumerate(palette):
        top[0, start + i] = int(color)
    return np.vstack([top, body])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layer_count = int(rng.choice(LAYER_COUNT_CHOICES))
    bg = int(rng.integers(COLOR_COUNT))
    available = [c for c in range(COLOR_COUNT) if c != bg]
    picked = [int(c) for c in rng.choice(available, size=layer_count + 3, replace=False)]
    layer_colors = picked[:layer_count]
    core_colors = picked[layer_count:]
    palette = list(layer_colors)
    rng.shuffle(palette)
    pos = {color: i + 1 for i, color in enumerate(palette)}
    turns = [pos[color] for color in layer_colors]

    dims = _choose_frame_dims(rng, turns)
    if dims is None:
        return None

    inp = _render_input(rng, bg, layer_colors, palette, dims, core_colors)
    if inp is None:
        return None

    if inp.ndim != 2 or max(inp.shape) > 30:
        return None
    try:
        out = main(inp)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if np.array_equal(out, inp[1:, :]):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested palette-frame input")
