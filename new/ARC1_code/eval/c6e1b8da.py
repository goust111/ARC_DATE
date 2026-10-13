from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste
import random

BBox = tuple[int, int, int, int]
Direction = tuple[int, int]
Slider = tuple[BBox, Direction, int]


def _is_full(mask: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _detect_slider(mask: np.ndarray) -> Slider | None:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty color mask"
    r0, c0, r1, c1 = bbox_coords(coords)

    candidates: list[Slider] = []

    def add_if_slider(body: BBox, direction: Direction, length: int, outside: np.ndarray) -> None:
        if length < 2:
            return
        br0, bc0, br1, bc1 = body
        if br1 - br0 + 1 < 2 or bc1 - bc0 + 1 < 2:
            return
        if outside.shape[0] != length or not _is_full(mask, body):
            return

        rows = set(map(int, outside[:, 0]))
        cols = set(map(int, outside[:, 1]))
        dr, dc = direction
        if dr == 0:
            expected_cols = set(range(bc1 + 1, c1 + 1)) if dc > 0 else set(range(c0, bc0))
            if len(rows) == 1 and cols == expected_cols:
                candidates.append((body, direction, length))
        else:
            expected_rows = set(range(br1 + 1, r1 + 1)) if dr > 0 else set(range(r0, br0))
            if len(cols) == 1 and rows == expected_rows:
                candidates.append((body, direction, length))

    for body_c1 in range(c0, c1):
        body = (r0, c0, r1, body_c1)
        add_if_slider(body, (0, 1), c1 - body_c1, coords[coords[:, 1] > body_c1])

    for body_c0 in range(c0 + 1, c1 + 1):
        body = (r0, body_c0, r1, c1)
        add_if_slider(body, (0, -1), body_c0 - c0, coords[coords[:, 1] < body_c0])

    for body_r1 in range(r0, r1):
        body = (r0, c0, body_r1, c1)
        add_if_slider(body, (1, 0), r1 - body_r1, coords[coords[:, 0] > body_r1])

    for body_r0 in range(r0 + 1, r1 + 1):
        body = (body_r0, c0, r1, c1)
        add_if_slider(body, (-1, 0), body_r0 - r0, coords[coords[:, 0] < body_r0])

    assert len(candidates) <= 1, "slider body and tail decomposition must be unique"
    return candidates[0] if candidates else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each solid rectangular body by the length and direction of its one-cell-thick tail, then redraw the scene with sliders on top.

    Input structure
    - The background is black.
    - Each foreground color represents one object: either a filled rectangle, possibly partly hidden by another color, or a filled rectangle with a straight one-cell-thick tail of length at least two.
    - Slider tails extend in one cardinal direction from one side of their rectangle; shifted slider rectangles remain in bounds and do not overlap one another.

    Rule
    - Restore every non-slider object as the filled rectangle of its input bounding box.
    - For every slider, discard the tail and translate only the rectangular body by the tail length in the tail direction.
    - Draw the translated slider rectangles over the restored static rectangles.

    Key insight
    - A thin tail is a displacement vector for its rectangle, while wider missing areas inside another color's bounding box are occlusion that should be restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = int(get_bg(g))
    assert bg == 0, "expected black background"

    statics: list[tuple[int, BBox]] = []
    sliders: list[tuple[int, Slider]] = []
    for color in colors_present(g, exclude={bg}):
        mask = g == int(color)
        slider = _detect_slider(mask)
        if slider is None:
            box = bbox_coords(np.argwhere(mask))
            r0, c0, r1, c1 = box
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] != bg), "static rectangle bbox must contain no background holes"
            statics.append((int(color), box))
        else:
            sliders.append((int(color), slider))

    assert sliders, "expected at least one slider object"

    out = np.full_like(g, bg)
    for color, (r0, c0, r1, c1) in statics:
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        conflict = (region != bg) & (region != color)
        assert not np.any(conflict), "static rectangles must not conflict"
        region[:, :] = color

    moved = np.zeros_like(g, dtype=bool)
    for color, (body, (dr, dc), length) in sliders:
        r0, c0, r1, c1 = body
        nr0 = r0 + dr * length
        nr1 = r1 + dr * length
        nc0 = c0 + dc * length
        nc1 = c1 + dc * length
        assert 0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w, "translated rectangle must stay in bounds"

        dest = np.zeros_like(g, dtype=bool)
        dest[nr0 : nr1 + 1, nc0 : nc1 + 1] = True
        assert not np.any(moved & dest), "translated slider rectangles must not overlap"
        moved |= dest
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
GEN_GRID_H, GEN_GRID_W = 20, 20
GEN_GRID_SHAPE = (GEN_GRID_H, GEN_GRID_W)
GEN_BACKGROUND, GEN_COLOR_BINS = 0, 10
GEN_COLOR_POOL = range(1, 10)

# Object count ranges.
GEN_TOTAL_COLOR_RANGE = (3, 5)
GEN_SLIDER_COUNT_CHOICES = (2, 3)
GEN_STATIC_COUNT_RANGE = (1, 3)

# Slider geometry.
GEN_SLIDER_H_RANGE = (3, 7)
GEN_SLIDER_W_RANGE = (3, 9)
GEN_TAIL_LENGTH_RANGE = (2, 6)
GEN_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))

# Static rectangle geometry.
GEN_STATIC_H_RANGE = (3, 8)
GEN_STATIC_W_RANGE = (5, 9)

# Rejection retry limit.
GEN_ATTEMPTS = 300


def _gen_box(r: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(r), int(c), int(r + h - 1), int(c + w - 1)


def _gen_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _gen_slider_boxes(
    dest_box: tuple[int, int, int, int],
    direction: tuple[int, int],
    length: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    dr, dc = direction
    r0, c0, r1, c1 = dest_box
    pre = (r0 - dr * length, c0 - dc * length, r1 - dr * length, c1 - dc * length)
    pr0, pc0, pr1, pc1 = pre
    if dc > 0:
        full = (pr0, pc0, pr1, pc1 + length)
    elif dc < 0:
        full = (pr0, pc0 - length, pr1, pc1)
    elif dr > 0:
        full = (pr0, pc0, pr1 + length, pc1)
    else:
        full = (pr0 - length, pc0, pr1, pc1)
    return pre, full


def _gen_in_bounds(box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < GEN_GRID_H and 0 <= c0 <= c1 < GEN_GRID_W


def _gen_candidate_boxes(size: tuple[int, int]):
    h, w = size
    for r in range(GEN_GRID_H - h + 1):
        for c in range(GEN_GRID_W - w + 1):
            yield _gen_box(r, c, h, w)


def _gen_slider_cells(
    body: tuple[int, int, int, int],
    direction: tuple[int, int],
    length: int,
    attach: int,
) -> np.ndarray:
    r0, c0, r1, c1 = body
    body_rr, body_cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
    body_cells = np.column_stack((body_rr.ravel(), body_cc.ravel()))
    dr, dc = direction
    if dc > 0:
        tail_cells = np.column_stack((np.full(length, attach), np.arange(c1 + 1, c1 + length + 1)))
    elif dc < 0:
        tail_cells = np.column_stack((np.full(length, attach), np.arange(c0 - length, c0)))
    elif dr > 0:
        tail_cells = np.column_stack((np.arange(r1 + 1, r1 + length + 1), np.full(length, attach)))
    else:
        tail_cells = np.column_stack((np.arange(r0 - length, r0), np.full(length, attach)))
    return np.vstack((body_cells, tail_cells)).astype(int, copy=False)


def _gen_edge_visible(box: tuple[int, int, int, int], covered: np.ndarray) -> bool:
    r0, c0, r1, c1 = box
    region = ~covered[r0 : r1 + 1, c0 : c1 + 1]
    return bool(region[0].any() and region[-1].any() and region[:, 0].any() and region[:, -1].any())


def _gen_place_static(
    size: tuple[int, int],
    static_boxes: list[tuple[int, int, int, int]],
    covered: np.ndarray,
) -> tuple[int, int, int, int] | None:
    choices: list[tuple[int, int, int, int]] = []
    for box in _gen_candidate_boxes(size):
        if any(_gen_overlap(box, other) for other in static_boxes):
            continue
        if not _gen_edge_visible(box, covered):
            continue
        choices.append(box)
    if not choices:
        return None
    return random.choice(choices)


def _gen_place_slider(
    size: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    dest_boxes: list[tuple[int, int, int, int]],
    input_boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    choices: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for dest in _gen_candidate_boxes(size):
        pre, full = _gen_slider_boxes(dest, direction, length)
        if not (_gen_in_bounds(pre) and _gen_in_bounds(full)):
            continue
        if any(_gen_overlap(dest, other) for other in dest_boxes):
            continue
        if any(_gen_overlap(full, other) for other in input_boxes):
            continue
        choices.append((dest, pre, full))
    if not choices:
        return None
    return random.choice(choices)


def _gen_paint_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(color), dtype=int)
    paste(grid, patch, r0, c0)


def _gen_choose_colors() -> tuple[list[int], list[int]]:
    total_colors = random.randint(*GEN_TOTAL_COLOR_RANGE)
    possible_sliders = [n for n in GEN_SLIDER_COUNT_CHOICES if GEN_STATIC_COUNT_RANGE[0] <= total_colors - n <= GEN_STATIC_COUNT_RANGE[1]]
    n_sliders = random.choice(possible_sliders)
    colors = random.sample(GEN_COLOR_POOL, total_colors)
    return colors[:n_sliders], colors[n_sliders:]


def _gen_slider_specs(slider_colors: list[int]) -> list[dict]:
    slider_specs = []
    for color in slider_colors:
        h = random.randint(*GEN_SLIDER_H_RANGE)
        w = random.randint(*GEN_SLIDER_W_RANGE)
        direction = random.choice(GEN_DIRECTIONS)
        length = random.randint(*GEN_TAIL_LENGTH_RANGE)
        slider_specs.append({"color": color, "size": (h, w), "direction": direction, "length": length})
    random.shuffle(slider_specs)
    slider_specs.sort(key=lambda s: s["size"][0] * s["size"][1] + s["length"], reverse=True)
    return slider_specs


def _gen_place_sliders(slider_specs: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    placed_sliders = []
    dest_boxes: list[tuple[int, int, int, int]] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    covered = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    for spec in slider_specs:
        placed = _gen_place_slider(spec["size"], spec["direction"], spec["length"], dest_boxes, input_boxes)
        if placed is None:
            return None
        dest, pre, full = placed
        dr, dc = spec["direction"]
        if dc:
            attach = random.randint(pre[0] + 1, pre[2] - 1)
        else:
            attach = random.randint(pre[1] + 1, pre[3] - 1)
        cells = _gen_slider_cells(pre, (dr, dc), spec["length"], attach)
        covered[cells[:, 0], cells[:, 1]] = True
        dest_boxes.append(dest)
        input_boxes.append(full)
        placed_sliders.append({**spec, "dest": dest, "pre": pre, "full": full, "attach": attach})
    return placed_sliders, covered


def _gen_static_specs(static_colors: list[int]) -> list[dict]:
    static_specs = []
    for color in static_colors:
        static_specs.append({"color": color, "size": (random.randint(*GEN_STATIC_H_RANGE), random.randint(*GEN_STATIC_W_RANGE))})
    random.shuffle(static_specs)
    static_specs.sort(key=lambda s: s["size"][0] * s["size"][1], reverse=True)
    return static_specs


def _gen_place_statics(static_specs: list[dict], covered: np.ndarray) -> list[dict] | None:
    placed_statics = []
    static_boxes: list[tuple[int, int, int, int]] = []
    for spec in static_specs:
        box = _gen_place_static(spec["size"], static_boxes, covered)
        if box is None:
            return None
        static_boxes.append(box)
        placed_statics.append({**spec, "box": box})
    return placed_statics


def _gen_paint_slider(grid: np.ndarray, slider: dict) -> None:
    cells = _gen_slider_cells(slider["pre"], slider["direction"], slider["length"], slider["attach"])
    grid[cells[:, 0], cells[:, 1]] = slider["color"]


def _gen_render_scene(placed_statics: list[dict], placed_sliders: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(GEN_GRID_SHAPE, dtype=int)
    out = np.zeros(GEN_GRID_SHAPE, dtype=int)
    for static in placed_statics:
        _gen_paint_rect(inp, static["box"], static["color"])
        _gen_paint_rect(out, static["box"], static["color"])
    for slider in placed_sliders:
        _gen_paint_slider(inp, slider)
    for slider in placed_sliders:
        _gen_paint_rect(out, slider["dest"], slider["color"])
    return inp, out


def _gen_matches_solver(inp: np.ndarray, out: np.ndarray) -> bool:
    counts = np.bincount(inp.ravel(), minlength=GEN_COLOR_BINS)
    if counts[GEN_BACKGROUND] <= max(counts[1:]):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    slider_colors, static_colors = _gen_choose_colors()
    slider_specs = _gen_slider_specs(slider_colors)
    placed = _gen_place_sliders(slider_specs)
    if placed is None:
        return None
    placed_sliders, covered = placed

    static_specs = _gen_static_specs(static_colors)
    placed_statics = _gen_place_statics(static_specs, covered)
    if placed_statics is None:
        return None

    inp, out = _gen_render_scene(placed_statics, placed_sliders)
    if not _gen_matches_solver(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c6e1b8da input")
