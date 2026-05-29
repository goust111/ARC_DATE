# Variant of 7ed72f31
# Dimension: D2 — 反射轴的定义方式 (src)
# Change: 红色(2)像素构成反射轴：多个红像素同行→水平轴，同列→竖直轴，单个红像素→点反射中心... → 非红色的标记颜色

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D2 variant
# a) Concrete behavior and trace
#    The geometry is unchanged from the source task: a marker line in one row means
#    row reflection, a marker line in one column means column reflection, and a
#    single marker pixel means point reflection. The only change is how the marker
#    color is identified: it is no longer fixed red(2), but the unique least
#    frequent non-background color in the input grid.
#    Example trace: suppose the input has bg=8, color counts {3:5, 2:7, 4:9, 6:8}.
#    In the original task, only color 2 would be treated as a marker, so a 3-line
#    would be ignored. In this variant, color 3 is inferred as the marker because
#    it is the least frequent non-bg color; the same row/col/single-pixel geometry
#    is then processed exactly as before.
# b) Complexity calibration
#    If the marker color were allowed to tie with payload colors, the task would
#    become ambiguous. The generator therefore enforces one unique least-frequency
#    marker color. The grid still contains all three source behaviors (row, col,
#    isolated point reflection), so the change is not visually trivial.
# c) Implicit assumptions in the original main()
#    - `MARK = 2` assumed a globally fixed special color. This is invalid now and
#      must be replaced by a counting step.
#    - All later checks `v == MARK` are still valid once MARK is inferred.
#    - The original bbox-based eligibility can accidentally catch same-colored
#      bystanders. To keep the D2 change isolated, the generator uses distinct
#      payload colors per active group and does not add extra same-colored decoys.
# d) Placement legality
#    Every group is built as a small local template first, then placed with its full
#    output footprint on an empty canvas. Placement checks use the output footprint,
#    not just the input footprint, so mirrored pixels never overlap existing objects.
#    A one-cell 8-neighborhood moat is also enforced between groups to avoid
#    accidental component merging.
# e) Visual observability
#    The marker color varies across samples and is not red. In many samples red is
#    reused as an ordinary payload color, which makes the D2 change directly
#    visible: red can appear, but the sparsest non-bg color is the one that drives
#    reflection.
# f) Visual cleanliness
#    Only three active groups are generated, one per source behavior. They are
#    separated spatially, and each group uses one payload color, so the grid stays
#    easy to read.
# g) Color-change inferability
#    There is no arbitrary color remapping. The only color inference is "find the
#    unique least frequent non-background color, then treat it as the axis/center".
#    The generator guarantees enough support for this rule by giving every payload
#    color more pixels than the marker color.
# New solver-side assumptions introduced by the variant
#    - There must be at least one non-background color.
#    - The least frequent non-background color is unique and is the intended marker.
#    - Marker pixels belong only to axis/center geometry, never to payload sprites.
#    `generate_input()` enforces all three.

PALETTE = list(range(10))


def _infer_marker_color(grid: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in grid.flatten() if int(v) != int(bg)]
    if not vals:
        raise ValueError("grid has no non-background colors")
    counts = Counter(vals)
    return min(counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using the unique least-frequent non-background color
    as the reflection marker.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    MARK = int(_infer_marker_color(g, bg))

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append({"coords": coords, "vals": vals, "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs)))})

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        marks = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        rs = {int(r) for r, _c in marks}
        cs = {int(c) for _r, c in marks}
        if len(marks) == 1:
            axis = ("point", (int(marks[0][0]), int(marks[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {int(labv[int(rr), int(cc)]) for rr, cc, vv in neigh_pixels if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0}
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

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


def _grow_connected(size: int, h: int, w: int, conn: int, seed: tuple[int, int], rng: random.Random) -> list[tuple[int, int]] | None:
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


def _build_row_group(mark: int, payload: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    for _ in range(200):
        line_len = 2
        size = rng.randint(7, 10)
        ph = rng.randint(3, 4)
        w = rng.randint(4, 6)
        if int(ph) * int(w) < int(size):
            continue
        line_c0 = rng.randint(0, int(w - line_len))
        seed = (int(ph - 1), rng.randint(int(line_c0), int(line_c0 + line_len - 1)))
        shape = _grow_connected(size, ph, w, 4, seed, rng)
        if shape is None:
            continue
        axis_r = int(ph)
        inp = [(int(r), int(c), int(payload)) for r, c in shape]
        inp += [(int(axis_r), int(line_c0 + dc), int(mark)) for dc in range(line_len)]
        out = list(inp)
        out += [(int(2 * axis_r - r), int(c), int(payload)) for r, c in shape]
        full_h = int(2 * ph + 1)
        full_w = int(w)
        flip_v = rng.random() < 0.5
        flip_h = rng.random() < 0.5
        return full_h, full_w, _transform_pixels(inp, full_h, full_w, flip_v=flip_v, flip_h=flip_h), _transform_pixels(out, full_h, full_w, flip_v=flip_v, flip_h=flip_h)
    raise RuntimeError("failed to build row group")


def _build_col_group(mark: int, payload: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    for _ in range(200):
        line_len = 2
        size = rng.randint(7, 10)
        h = rng.randint(4, 6)
        pw = rng.randint(3, 4)
        if int(h) * int(pw) < int(size):
            continue
        line_r0 = rng.randint(0, int(h - line_len))
        seed = (rng.randint(int(line_r0), int(line_r0 + line_len - 1)), int(pw - 1))
        shape = _grow_connected(size, h, pw, 4, seed, rng)
        if shape is None:
            continue
        axis_c = int(pw)
        inp = [(int(r), int(c), int(payload)) for r, c in shape]
        inp += [(int(line_r0 + dr), int(axis_c), int(mark)) for dr in range(line_len)]
        out = list(inp)
        out += [(int(r), int(2 * axis_c - c), int(payload)) for r, c in shape]
        full_h = int(h)
        full_w = int(2 * pw + 1)
        flip_v = rng.random() < 0.5
        flip_h = rng.random() < 0.5
        return full_h, full_w, _transform_pixels(inp, full_h, full_w, flip_v=flip_v, flip_h=flip_h), _transform_pixels(out, full_h, full_w, flip_v=flip_v, flip_h=flip_h)
    raise RuntimeError("failed to build col group")


def _build_isolated_group(mark: int, payload: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    for _ in range(200):
        size = rng.randint(7, 8)
        qh, qw = rng.choice([(3, 3), (3, 4), (4, 3)])
        if int(qh) * int(qw) < int(size):
            continue
        seed = (int(qh - 1), int(qw - 1))
        shape = _grow_connected(size, qh, qw, 8, seed, rng)
        if shape is None:
            continue
        center = (int(qh), int(qw))
        inp = [(int(r), int(c), int(payload)) for r, c in shape]
        inp += [(int(center[0]), int(center[1]), int(mark))]
        out = list(inp)
        out += [(int(2 * center[0] - r), int(2 * center[1] - c), int(payload)) for r, c in shape]
        full_h = int(2 * qh + 1)
        full_w = int(2 * qw + 1)
        flip_v = rng.random() < 0.5
        flip_h = rng.random() < 0.5
        return full_h, full_w, _transform_pixels(inp, full_h, full_w, flip_v=flip_v, flip_h=flip_h), _transform_pixels(out, full_h, full_w, flip_v=flip_v, flip_h=flip_h)
    raise RuntimeError("failed to build isolated group")


def _place_group(
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    forbidden: set[tuple[int, int]],
    group: tuple[int, int, list[tuple[int, int, int]], list[tuple[int, int, int]]],
    rng: random.Random,
) -> bool:
    H, W = map(int, canvas_out.shape)
    gh, gw, inp_pixels, out_pixels = group
    for _ in range(300):
        top = rng.randint(0, int(H - gh))
        left = rng.randint(0, int(W - gw))
        shifted_out = [(int(top + r), int(left + c), int(v)) for r, c, v in out_pixels]
        if any((int(r), int(c)) in forbidden for r, c, _v in shifted_out):
            continue
        shifted_in = [(int(top + r), int(left + c), int(v)) for r, c, v in inp_pixels]
        for r, c, v in shifted_out:
            canvas_out[int(r), int(c)] = int(v)
        for r, c, v in shifted_in:
            canvas_in[int(r), int(c)] = int(v)
        for r, c, _v in shifted_out:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    forbidden.add((int(r + dr), int(c + dc)))
        return True
    return False


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D2 variant.

    It keeps the source task's three reflection mechanisms but forces the marker
    color to be the unique least-frequent non-background color in the input.
    """
    rng = random.Random()
    for _attempt in range(400):
        H = rng.randint(18, 28)
        W = rng.randint(18, 28)
        bg = rng.choice(PALETTE)
        mark_choices = [c for c in PALETTE if c != int(bg) and c != 2]
        mark = rng.choice(mark_choices)

        payload_choices = [c for c in PALETTE if c != int(bg) and c != int(mark)]
        payload_colors: list[int] = []
        if 2 in payload_choices and rng.random() < 0.6:
            payload_colors.append(2)
            payload_choices.remove(2)
        payload_colors += rng.sample(payload_choices, 3 - len(payload_colors))
        rng.shuffle(payload_colors)

        groups = [
            _build_row_group(mark, payload_colors[0], rng),
            _build_col_group(mark, payload_colors[1], rng),
            _build_isolated_group(mark, payload_colors[2], rng),
        ]
        rng.shuffle(groups)

        inp = np.full((H, W), int(bg), dtype=int)
        out = np.full((H, W), int(bg), dtype=int)
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

        solved = main(inp)
        if np.array_equal(solved, out):
            return inp

    raise RuntimeError("generate_input failed after many attempts")
