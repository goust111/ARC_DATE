# Combo variant of 7ed72f31 (index 7)
# combo_spec: D2/0+D9/1+D12/1
# Name: 隐藏标记色识别与消耗

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis
# ---------------
# a) Concrete behavior and trace
#    Source task: a fixed red marker defines a row/col mirror axis or a point
#    reflection center, and the red marker survives in the output.
#    Combo behavior:
#    1. First infer the marker color as the unique least-frequent non-background
#       color. It is intentionally non-red, while red may appear as ordinary
#       payload color.
#    2. Not every pixel of that color is a true marker. Only components that form
#       a valid instruction are consumed:
#       - a same-color line attached to a monochrome payload component,
#       - or a singleton same-color point diagonally adjacent to exactly one
#         monochrome 8-connected payload component.
#       Decorative same-color pixels/mini-lines are left untouched.
#    3. After reflecting, true markers are erased, but same-color decoys remain.
#    Small trace:
#       input counts -> bg=8, non-bg counts {6:8, 2:10, 4:9, 7:11}; so marker=6.
#       A 2-cell color-6 row attached to a green sprite mirrors that sprite and
#       then disappears. A lone color-6 decorative dot elsewhere stays. The output
#       therefore contains some 6s, but only the axis-like 6 component was removed.
# b) Complexity calibration
#    The generator always includes one row group, one column group, one isolated
#    point group, plus two same-color decoys. That keeps the fusion visually
#    obvious without introducing too many cases. Payload colors stay unique across
#    active groups so solver matching never depends on ambiguous global color reuse.
# c) Implicit assumption audit for main()
#    - Fixed `MARK = 2` is invalid; replace it with least-frequency inference.
#    - Erasing `out[g == MARK]` is invalid because some MARK-colored pixels are
#      decoys; only validated instruction pixels may be erased.
#    - The source solver's global color+bbox scan is too loose once same-color
#      decoys exist. This variant mirrors only pixels inside the validated local
#      component instead.
#    - Isolated point markers can no longer be identified by color alone; they must
#      have exactly one adjacent 8-connected payload component of one color.
# d) Placement legality
#    The generator is output-first. Each active group is built as a local template
#    containing original payload, mirrored payload, and true marker cells. Groups
#    are placed using the union of input/output footprints, with an 8-neighborhood
#    moat, so no overlap or accidental merging occurs. Decoys use the same moat.
# e) Visual observability
#    Across examples, the marker color changes and is not red, but is always the
#    sparsest non-background color. Some pixels of that color disappear while other
#    pixels of the same color remain, making both "identify the marker color" and
#    "separate real markers from decoys" necessary and visible.
# f) Visual cleanliness
#    Only three active groups are used, each payload is monochrome, and every
#    placed object gets a one-cell moat. Decoys are tiny and detached, so they act
#    as readable distractors instead of clutter.
# g) Change inferability
#    There is no arbitrary recoloring. The only color rule is structural:
#    "choose the least frequent non-background color, then consume only the
#    instruction-shaped components of that color." The generator enforces payload
#    sizes strictly above the total marker-color count so this least-frequency cue
#    is stable from a few examples.
# New solver-side assumptions introduced here
#    - The least-frequent non-background color is unique.
#    - Every true row/col marker component has a monochrome payload on exactly one
#      side of the marker line.
#    - Every true isolated point marker is a singleton and is 8-adjacent to
#      exactly one monochrome payload component.
#    - Same-color decoys are detached from payloads strongly enough that they do
#      not accidentally satisfy the instruction tests.
#    `generate_input()` explicitly guarantees these assumptions.

PALETTE = list(range(10))


def _infer_marker_color(grid: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in grid.flatten() if int(v) != int(bg)]
    if not vals:
        raise ValueError("grid has no non-background colors")
    counts = Counter(vals)
    return min(counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0]


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    cr, cc = map(int, center)
    return int(2 * cr - r), int(2 * cc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _contiguous(values: list[int]) -> bool:
    if not values:
        return False
    lo = min(values)
    hi = max(values)
    return sorted(values) == list(range(int(lo), int(hi) + 1))


def _collect_reflections(
    grid: np.ndarray, bg: int, mark: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int, int]]]:
    g = np.array(grid, dtype=int)
    h, w = map(int, g.shape)
    consumed: set[tuple[int, int]] = set()
    paints: list[tuple[int, int, int]] = []

    lab4, n4 = label_mask(g != int(bg), connectivity=4)
    for lbl in range(1, int(n4) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == int(lbl))]
        if not coords:
            continue
        mark_cells = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == int(mark)]
        if not mark_cells or len(mark_cells) == len(coords):
            continue

        payload = [(int(r), int(c), int(g[r, c])) for r, c in coords if int(g[r, c]) != int(mark)]
        payload_colors = {int(v) for _r, _c, v in payload}
        if len(payload_colors) != 1:
            continue

        axis_kind: str | None = None
        axis_value: int | tuple[int, int]
        if len(mark_cells) == 1:
            axis_kind = "point"
            axis_value = (int(mark_cells[0][0]), int(mark_cells[0][1]))
        else:
            rows = [int(r) for r, _c in mark_cells]
            cols = [int(c) for _r, c in mark_cells]
            row_set = set(rows)
            col_set = set(cols)
            if len(row_set) == 1 and _contiguous(cols):
                axis_row = int(next(iter(row_set)))
                if all(int(r) < int(axis_row) for r, _c, _v in payload) or all(
                    int(r) > int(axis_row) for r, _c, _v in payload
                ):
                    axis_kind = "row"
                    axis_value = int(axis_row)
            elif len(col_set) == 1 and _contiguous(rows):
                axis_col = int(next(iter(col_set)))
                if all(int(c) < int(axis_col) for _r, c, _v in payload) or all(
                    int(c) > int(axis_col) for _r, c, _v in payload
                ):
                    axis_kind = "col"
                    axis_value = int(axis_col)
        if axis_kind is None:
            continue

        for r, c in mark_cells:
            consumed.add((int(r), int(c)))
        for r, c, v in payload:
            if axis_kind == "row":
                rr, cc = _refl_row(int(axis_value), int(r), int(c))
            elif axis_kind == "col":
                rr, cc = _refl_col(int(axis_value), int(r), int(c))
            else:
                rr, cc = _refl_point(axis_value, int(r), int(c))
            if _inb(h, w, int(rr), int(cc)):
                paints.append((int(rr), int(cc), int(v)))

    color_label_cache: dict[int, np.ndarray] = {}
    mark_lab, mark_n = label_mask(g == int(mark), connectivity=4)
    for lbl in range(1, int(mark_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(mark_lab == int(lbl))]
        if len(coords) != 1:
            continue
        pr, pc = coords[0]
        if (int(pr), int(pc)) in consumed:
            continue

        tags: set[tuple[int, int]] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = int(pr + dr)
                cc = int(pc + dc)
                if not _inb(h, w, rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(mark):
                    continue
                if int(v) not in color_label_cache:
                    color_label_cache[int(v)], _ = label_mask(g == int(v), connectivity=8)
                labv = color_label_cache[int(v)]
                tag = int(labv[rr, cc])
                if tag != 0:
                    tags.add((int(v), int(tag)))
        if len(tags) != 1:
            continue

        consumed.add((int(pr), int(pc)))
        color, tag = next(iter(tags))
        labv = color_label_cache[int(color)]
        for r, c in np.argwhere(labv == int(tag)):
            rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
            if _inb(h, w, int(rr), int(cc)):
                paints.append((int(rr), int(cc), int(color)))

    return consumed, paints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror sprites using the unique least-frequent non-background color as the
    marker color, erase only true marker pixels, and keep same-color decoys.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    mark = int(_infer_marker_color(g, bg))

    consumed, paints = _collect_reflections(g, bg, mark)
    out = g.copy()
    for r, c in consumed:
        out[int(r), int(c)] = int(bg)
    for r, c, v in paints:
        out[int(r), int(c)] = int(v)
    return out


def _neighbors(cell: tuple[int, int], h: int, w: int, conn: int) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if int(conn) == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < int(h) and 0 <= cc < int(w):
                out.append((int(rr), int(cc)))
    return out


def _grow_connected(
    size: int, h: int, w: int, conn: int, seed: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]] | None:
    cells = {tuple(map(int, seed))}
    while len(cells) < int(size):
        frontier = set()
        for cell in cells:
            frontier.update(_neighbors(cell, h, w, conn))
        frontier -= cells
        if not frontier:
            return None
        cells.add(rng.choice(sorted(frontier)))
    return sorted((int(r), int(c)) for r, c in cells)


def _transform_pixels(
    pixels: list[tuple[int, int, int]],
    h: int,
    w: int,
    *,
    flip_v: bool = False,
    flip_h: bool = False,
) -> list[tuple[int, int, int]]:
    out = []
    for r, c, v in pixels:
        rr = int(h - 1 - r) if flip_v else int(r)
        cc = int(w - 1 - c) if flip_h else int(c)
        out.append((int(rr), int(cc), int(v)))
    return out


def _build_row_group(
    mark: int, payload: int, rng: random.Random
) -> tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    for _ in range(200):
        line_len = 2
        size = rng.randint(9, 11)
        ph = rng.randint(3, 4)
        width = rng.randint(4, 6)
        if int(ph) * int(width) < int(size):
            continue
        line_c0 = rng.randint(0, int(width - line_len))
        seed = (int(ph - 1), rng.randint(int(line_c0), int(line_c0 + line_len - 1)))
        shape = _grow_connected(size, ph, width, 4, seed, rng)
        if shape is None:
            continue
        axis_r = int(ph)
        inp = [(int(r), int(c), int(payload)) for r, c in shape]
        inp += [(int(axis_r), int(line_c0 + dc), int(mark)) for dc in range(line_len)]
        out = [(int(r), int(c), int(payload)) for r, c in shape]
        out += [(int(2 * axis_r - r), int(c), int(payload)) for r, c in shape]
        full_h = int(2 * ph + 1)
        full_w = int(width)
        flip_v = rng.random() < 0.5
        flip_h = rng.random() < 0.5
        return (
            full_h,
            full_w,
            _transform_pixels(inp, full_h, full_w, flip_v=flip_v, flip_h=flip_h),
            _transform_pixels(out, full_h, full_w, flip_v=flip_v, flip_h=flip_h),
        )
    raise RuntimeError("failed to build row group")


def _build_col_group(
    mark: int, payload: int, rng: random.Random
) -> tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    for _ in range(200):
        line_len = 2
        size = rng.randint(9, 11)
        height = rng.randint(4, 6)
        pw = rng.randint(3, 4)
        if int(height) * int(pw) < int(size):
            continue
        line_r0 = rng.randint(0, int(height - line_len))
        seed = (rng.randint(int(line_r0), int(line_r0 + line_len - 1)), int(pw - 1))
        shape = _grow_connected(size, height, pw, 4, seed, rng)
        if shape is None:
            continue
        axis_c = int(pw)
        inp = [(int(r), int(c), int(payload)) for r, c in shape]
        inp += [(int(line_r0 + dr), int(axis_c), int(mark)) for dr in range(line_len)]
        out = [(int(r), int(c), int(payload)) for r, c in shape]
        out += [(int(r), int(2 * axis_c - c), int(payload)) for r, c in shape]
        full_h = int(height)
        full_w = int(2 * pw + 1)
        flip_v = rng.random() < 0.5
        flip_h = rng.random() < 0.5
        return (
            full_h,
            full_w,
            _transform_pixels(inp, full_h, full_w, flip_v=flip_v, flip_h=flip_h),
            _transform_pixels(out, full_h, full_w, flip_v=flip_v, flip_h=flip_h),
        )
    raise RuntimeError("failed to build col group")


def _build_isolated_group(
    mark: int, payload: int, rng: random.Random
) -> tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    for _ in range(200):
        size = rng.randint(9, 10)
        qh, qw = rng.choice([(3, 3), (3, 4), (4, 3), (4, 4)])
        if int(qh) * int(qw) < int(size):
            continue
        seed = (int(qh - 1), int(qw - 1))
        shape = _grow_connected(size, qh, qw, 8, seed, rng)
        if shape is None:
            continue
        center = (int(qh), int(qw))
        inp = [(int(r), int(c), int(payload)) for r, c in shape]
        inp += [(int(center[0]), int(center[1]), int(mark))]
        out = [(int(r), int(c), int(payload)) for r, c in shape]
        out += [(int(2 * center[0] - r), int(2 * center[1] - c), int(payload)) for r, c in shape]
        full_h = int(2 * qh + 1)
        full_w = int(2 * qw + 1)
        flip_v = rng.random() < 0.5
        flip_h = rng.random() < 0.5
        return (
            full_h,
            full_w,
            _transform_pixels(inp, full_h, full_w, flip_v=flip_v, flip_h=flip_h),
            _transform_pixels(out, full_h, full_w, flip_v=flip_v, flip_h=flip_h),
        )
    raise RuntimeError("failed to build isolated group")


def _build_decoy_single(
    mark: int,
) -> tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    pixels = [(0, 0, int(mark))]
    return 1, 1, list(pixels), list(pixels)


def _build_decoy_line(
    mark: int, rng: random.Random
) -> tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    if rng.random() < 0.5:
        pixels = [(0, 0, int(mark)), (0, 1, int(mark))]
        return 1, 2, list(pixels), list(pixels)
    pixels = [(0, 0, int(mark)), (1, 0, int(mark))]
    return 2, 1, list(pixels), list(pixels)


def _place_group(
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    forbidden: set[tuple[int, int]],
    group: tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]],
    rng: random.Random,
) -> bool:
    h, w = map(int, canvas_out.shape)
    gh, gw, inp_pixels, out_pixels = group
    all_local = {(int(r), int(c)) for r, c, _v in inp_pixels} | {
        (int(r), int(c)) for r, c, _v in out_pixels
    }
    for _ in range(400):
        top = rng.randint(0, int(h - gh))
        left = rng.randint(0, int(w - gw))
        shifted_all = {(int(top + r), int(left + c)) for r, c in all_local}
        if any((int(r), int(c)) in forbidden for r, c in shifted_all):
            continue
        shifted_in = [(int(top + r), int(left + c), int(v)) for r, c, v in inp_pixels]
        shifted_out = [(int(top + r), int(left + c), int(v)) for r, c, v in out_pixels]
        for r, c, v in shifted_in:
            canvas_in[int(r), int(c)] = int(v)
        for r, c, v in shifted_out:
            canvas_out[int(r), int(c)] = int(v)
        for r, c in shifted_all:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    forbidden.add((int(r + dr), int(c + dc)))
        return True
    return False


def generate_input(seed: int | None = None) -> np.ndarray:
    """
    Output-first generator for the combo variant.

    Construction guarantees the fused behavior:
    - marker color is non-red and uniquely least frequent in the input,
    - two same-color decoys survive to the output,
    - true marker cells disappear after reflection.
    """
    rng = random.Random(seed)
    for _attempt in range(500):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        bg = rng.choice(PALETTE)
        mark_choices = [c for c in PALETTE if c not in {int(bg), 2}]
        mark = rng.choice(mark_choices)

        payload_choices = [c for c in PALETTE if c not in {int(bg), int(mark)}]
        payload_colors: list[int] = []
        if 2 in payload_choices and rng.random() < 0.7:
            payload_colors.append(2)
            payload_choices.remove(2)
        payload_colors += rng.sample(payload_choices, 3 - len(payload_colors))
        rng.shuffle(payload_colors)

        groups = [
            _build_row_group(mark, payload_colors[0], rng),
            _build_col_group(mark, payload_colors[1], rng),
            _build_isolated_group(mark, payload_colors[2], rng),
            _build_decoy_single(mark),
            _build_decoy_line(mark, rng),
        ]
        rng.shuffle(groups)

        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full((h, w), int(bg), dtype=int)
        forbidden: set[tuple[int, int]] = set()
        ok = True
        for group in groups:
            if not _place_group(inp, out, forbidden, group, rng):
                ok = False
                break
        if not ok:
            continue

        counts = Counter(int(v) for v in inp.flatten() if int(v) != int(bg))
        if not counts:
            continue
        least_color, least_count = min(counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
        if int(least_color) != int(mark):
            continue
        if sum(1 for v in counts.values() if int(v) == int(least_count)) != 1:
            continue

        pred = main(inp)
        if not np.array_equal(pred, out):
            continue

        true_markers, _ = _collect_reflections(inp, int(bg), int(mark))
        if len(true_markers) != 5:
            continue
        input_mark = int(np.count_nonzero(inp == int(mark)))
        output_mark = int(np.count_nonzero(out == int(mark)))
        if input_mark - output_mark != 5:
            continue
        if output_mark < 3:
            continue
        return inp

    raise RuntimeError("generate_input failed after many attempts")
