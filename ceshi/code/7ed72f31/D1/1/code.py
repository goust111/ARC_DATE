# Variant of 7ed72f31
# Dimension: D1 — 反射变换类型 (transform)
# Change: 根据红色标记几何形状选择镜像方式：红线→轴反射，单红像素→点反射... → 平移替代反射

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import get_bg, label_mask

# Design analysis for the D1 variant
# a) Concrete behavior and trace
#    The input still contains several mixed components made from a red marker line
#    plus one monochrome payload sprite attached on exactly one side. The original
#    task mirrored the payload across that line. In this variant the red line is no
#    longer a reflection axis: its orientation picks the translation axis and its
#    length picks the translation distance.
#    - Horizontal red line of length L: translate the payload vertically by L.
#    - Vertical red line of length L: translate the payload horizontally by L.
#    - The sign is determined by the payload side so that the copy lands on the
#      opposite side of the line. Example: if a 4-long horizontal red line has a
#      blue payload below it, a blue pixel at (line+1, c) moves to (line-3, c).
#      The original task would have mirrored it only to (line-1, c). This is the
#      key visible difference: the offset is size-encoded by the marker length.
# b) Complexity calibration
#    Single red pixels are removed from generation instead of inventing a second
#    translation rule for them. The puzzle still has repeated structure because each
#    grid contains both horizontal and vertical groups and at least two different
#    line lengths. Lengths are limited to 3-6 so the displacement is visible and
#    never collapses to the original reflection with an adjacent depth-1 payload.
# c) Implicit assumptions in the old solver
#    - The old code branched on "single red pixel" versus "red line" and then called
#      reflection helpers. Those assumptions no longer hold, so the new main ignores
#      single-pixel markers and only accepts contiguous row/column lines.
#    - Reflection was safe for payloads at arbitrary depth from the line. Translation
#      is not: if a payload extends too far away, shifting by L could hit the red
#      line or the original payload. The generator therefore guarantees that payload
#      depth away from the line is strictly smaller than the line length.
#    - The old solver used bbox-based eligibility that could sweep in stray same-color
#      pixels. That behavior was tied to the reflection implementation rather than
#      the visual rule, so the variant solver uses only the current 4-connected
#      component. The generator avoids free stray sprites to keep the translation
#      mapping unambiguous to a human.
# d) Placement legality
#    Each group is built output-first in its own local patch. The patch dimensions are
#    chosen so that both the original payload and its translated copy fit without
#    overlap:
#    - horizontal group patch height = payload_depth + line_length
#    - vertical group patch width  = payload_depth + line_length
#    Rectangular group patches are then placed on the global canvas with a 1-cell
#    buffer, so different groups cannot merge under 4-connectivity and solver-side
#    grouping remains correct.
# e) Visual observability
#    The payload masks are irregular 2D connected shapes rather than bars or full
#    rectangles, so preserving orientation versus mirroring is easy to notice. Having
#    multiple line lengths in the same grid makes the "length encodes distance" rule
#    inferable from only a few examples.
# f) Visual cleanliness
#    No group overlaps another, the red line spans one full side of its local patch,
#    and the translated copy is placed into blank background. This keeps the scene
#    legible: marker, original payload, and translated payload are separable by eye.
# g) Color inference
#    Colors do not change. Each group uses a unique non-red, non-background color, so
#    the only thing the solver must infer is the geometric translation rule.

MARK = 2
LINE_LENGTHS = (3, 4, 5, 6)


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _is_consecutive(vals: list[int]) -> bool:
    return all(int(b) == int(a) + 1 for a, b in zip(vals, vals[1:]))


def _detect_line_marker(reds: list[tuple[int, int]]) -> tuple[str, int, int] | None:
    if len(reds) < 2:
        return None
    rows = sorted({int(r) for r, _ in reds})
    cols = sorted({int(c) for _, c in reds})
    if len(rows) == 1:
        col_list = sorted(int(c) for _, c in reds)
        if _is_consecutive(col_list):
            return "row", int(rows[0]), len(col_list)
    if len(cols) == 1:
        row_list = sorted(int(r) for r, _ in reds)
        if _is_consecutive(row_list):
            return "col", int(cols[0]), len(row_list)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = map(int, g.shape)
    out = g.copy()

    lab, n = label_mask(g != int(bg), connectivity=4)
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue

        reds = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == int(MARK)]
        marker = _detect_line_marker(reds)
        if marker is None:
            continue

        payload = [(int(r), int(c), int(g[r, c])) for r, c in coords if int(g[r, c]) != int(MARK)]
        if not payload:
            continue

        kind, anchor, length = marker
        if kind == "row":
            if any(int(r) == int(anchor) for r, _c, _v in payload):
                continue
            above = [(r, c, v) for r, c, v in payload if int(r) < int(anchor)]
            below = [(r, c, v) for r, c, v in payload if int(r) > int(anchor)]
            if bool(above) == bool(below):
                continue
            dr, dc = (int(length), 0) if above else (-int(length), 0)
        else:
            if any(int(c) == int(anchor) for _r, c, _v in payload):
                continue
            left = [(r, c, v) for r, c, v in payload if int(c) < int(anchor)]
            right = [(r, c, v) for r, c, v in payload if int(c) > int(anchor)]
            if bool(left) == bool(right):
                continue
            dr, dc = (0, int(length)) if left else (0, -int(length))

        for r, c, v in payload:
            rr, cc = int(r + dr), int(c + dc)
            if _in_bounds(h, w, rr, cc):
                out[rr, cc] = int(v)

    return out


def _grow_connected_shape(
    rng: random.Random,
    height: int,
    width: int,
    size: int,
    touch: tuple[str, int],
) -> np.ndarray:
    for _ in range(200):
        if touch[0] == "row":
            start = (int(touch[1]), rng.randrange(int(width)))
        else:
            start = (rng.randrange(int(height)), int(touch[1]))

        cells = {start}
        while len(cells) < int(size):
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= int(nr) < int(height) and 0 <= int(nc) < int(width) and (int(nr), int(nc)) not in cells:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(rng.choice(sorted(frontier)))

        if len(cells) != int(size):
            continue

        rows = {int(r) for r, _ in cells}
        cols = {int(c) for _, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue

        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(cells) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
            continue

        mask = np.zeros((int(height), int(width)), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        return mask

    raise RuntimeError("failed to build an irregular connected payload")


def _make_horizontal_group(rng: random.Random, bg: int, color: int, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    depth = rng.randint(2, min(4, int(length) - 1))
    span = rng.randint(2, int(length))
    area = int(depth) * int(span)
    size = rng.randint(3, min(8, area - 1))
    side = rng.choice(["above", "below"])
    touch = ("row", int(depth) - 1) if side == "above" else ("row", 0)
    mask = _grow_connected_shape(rng, int(depth), int(span), int(size), touch)

    patch_h = int(length) + int(depth)
    patch_w = int(length)
    line_row = int(depth) if side == "above" else int(length) - 1
    col0 = rng.randint(0, int(length) - int(span))

    inp = np.full((int(patch_h), int(patch_w)), int(bg), dtype=int)
    out = inp.copy()
    out[line_row, :] = int(MARK)
    inp[line_row, :] = int(MARK)

    dr = int(length) if side == "above" else -int(length)
    row0 = 0 if side == "above" else int(length)
    for r, c in np.argwhere(mask):
        rr = int(row0 + r)
        cc = int(col0 + c)
        inp[rr, cc] = int(color)
        out[rr, cc] = int(color)
        out[int(rr + dr), int(cc)] = int(color)

    meta = {"orientation": "row", "length": int(length), "side": side}
    return inp, out, meta


def _make_vertical_group(rng: random.Random, bg: int, color: int, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    depth = rng.randint(2, min(4, int(length) - 1))
    span = rng.randint(2, int(length))
    area = int(depth) * int(span)
    size = rng.randint(3, min(8, area - 1))
    side = rng.choice(["left", "right"])
    touch = ("col", int(depth) - 1) if side == "left" else ("col", 0)
    mask = _grow_connected_shape(rng, int(span), int(depth), int(size), touch)

    patch_h = int(length)
    patch_w = int(length) + int(depth)
    line_col = int(depth) if side == "left" else int(length) - 1
    row0 = rng.randint(0, int(length) - int(span))

    inp = np.full((int(patch_h), int(patch_w)), int(bg), dtype=int)
    out = inp.copy()
    out[:, line_col] = int(MARK)
    inp[:, line_col] = int(MARK)

    dc = int(length) if side == "left" else -int(length)
    col0 = 0 if side == "left" else int(length)
    for r, c in np.argwhere(mask):
        rr = int(row0 + r)
        cc = int(col0 + c)
        inp[rr, cc] = int(color)
        out[rr, cc] = int(color)
        out[int(rr), int(cc + dc)] = int(color)

    meta = {"orientation": "col", "length": int(length), "side": side}
    return inp, out, meta


def _make_group(rng: random.Random, bg: int, color: int, orientation: str, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if orientation == "row":
        return _make_horizontal_group(rng, bg, color, length)
    return _make_vertical_group(rng, bg, color, length)


def _can_place(used: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = used.shape
    r0 = max(0, int(top) - 1)
    c0 = max(0, int(left) - 1)
    r1 = min(int(h), int(top) + int(height) + 1)
    c1 = min(int(w), int(left) + int(width) + 1)
    return not bool(used[r0:r1, c0:c1].any())


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    height, width = patch.shape
    target = canvas[int(top) : int(top) + int(height), int(left) : int(left) + int(width)]
    mask = patch != int(bg)
    target[mask] = patch[mask]


def _generate_example(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = rng or random.Random()
    palette = list(range(10))

    for _ in range(200):
        height = rng.randint(22, 28)
        width = rng.randint(22, 28)
        bg = rng.choice([c for c in palette if c != int(MARK)])

        group_count = rng.randint(4, 6)
        orientations = ["row", "col"] + [rng.choice(["row", "col"]) for _ in range(group_count - 2)]
        rng.shuffle(orientations)

        lengths = [rng.choice(LINE_LENGTHS) for _ in range(group_count)]
        if len(set(lengths)) == 1:
            lengths[0] = rng.choice([x for x in LINE_LENGTHS if x != lengths[0]])

        colors = rng.sample([c for c in palette if c not in (int(bg), int(MARK))], group_count)
        inp = np.full((int(height), int(width)), int(bg), dtype=int)
        out = inp.copy()
        used = np.zeros((int(height), int(width)), dtype=bool)
        meta: list[dict] = []

        ok = True
        for orientation, length, color in zip(orientations, lengths, colors):
            patch_in, patch_out, info = _make_group(rng, int(bg), int(color), orientation, int(length))
            ph, pw = patch_in.shape
            placed = False
            for _ in range(300):
                top = rng.randint(0, int(height) - int(ph))
                left = rng.randint(0, int(width) - int(pw))
                if not _can_place(used, int(top), int(left), int(ph), int(pw)):
                    continue
                _paste(inp, patch_in, int(top), int(left), int(bg))
                _paste(out, patch_out, int(top), int(left), int(bg))
                used[int(top) : int(top) + int(ph), int(left) : int(left) + int(pw)] = True
                info = dict(info)
                info["top"] = int(top)
                info["left"] = int(left)
                meta.append(info)
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok and np.array_equal(main(inp), out):
            return inp, out, meta

    raise RuntimeError("failed to generate a valid translated example")


def generate_input() -> np.ndarray:
    inp, out, _meta = _generate_example()
    assert np.array_equal(main(inp), out)
    return inp
