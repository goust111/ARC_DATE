from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
YELLOW = 4
RAY_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _rotate_pair(inp: np.ndarray, out: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.rot90(inp, k).copy(), np.rot90(out, k).copy()


def _flip_pair(inp: np.ndarray, out: np.ndarray, horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
    if horizontal:
        return np.fliplr(inp).copy(), np.fliplr(out).copy()
    return np.flipud(inp).copy(), np.flipud(out).copy()


def _parse_rays(grid: np.ndarray) -> dict[str, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    full_h: list[dict[str, int]] = []
    full_v: list[dict[str, int]] = []
    full_h_rows: set[int] = set()
    full_v_cols: set[int] = set()

    for r in range(h):
        row = g[r]
        if row[0] != bg and np.all(row == row[0]):
            full_h.append({"row": int(r), "color": int(row[0])})
            full_h_rows.add(int(r))

    for c in range(w):
        col = g[:, c]
        if col[0] != bg and np.all(col == col[0]):
            full_v.append({"col": int(c), "color": int(col[0])})
            full_v_cols.add(int(c))

    left: list[dict[str, int]] = []
    right: list[dict[str, int]] = []
    top: list[dict[str, int]] = []
    bottom: list[dict[str, int]] = []

    for r in range(h):
        if r in full_h_rows:
            continue
        if g[r, 0] != bg:
            color = int(g[r, 0])
            c = 0
            while c + 1 < w and int(g[r, c + 1]) == color:
                c += 1
            if c < w - 1 and int(g[r, c + 1]) == bg:
                left.append({"row": int(r), "end": int(c), "color": color})
        if g[r, w - 1] != bg:
            color = int(g[r, w - 1])
            c = w - 1
            while c - 1 >= 0 and int(g[r, c - 1]) == color:
                c -= 1
            if c > 0 and int(g[r, c - 1]) == bg:
                right.append({"row": int(r), "start": int(c), "color": color})

    for c in range(w):
        if c in full_v_cols:
            continue
        if g[0, c] != bg:
            color = int(g[0, c])
            r = 0
            while r + 1 < h and int(g[r + 1, c]) == color:
                r += 1
            if r < h - 1 and int(g[r + 1, c]) == bg:
                top.append({"col": int(c), "end": int(r), "color": color})
        if g[h - 1, c] != bg:
            color = int(g[h - 1, c])
            r = h - 1
            while r - 1 >= 0 and int(g[r - 1, c]) == color:
                r -= 1
            if r > 0 and int(g[r - 1, c]) == bg:
                bottom.append({"col": int(c), "start": int(r), "color": color})

    return {
        "full_h": full_h,
        "full_v": full_v,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


def _unique_or_none(values: list[int]) -> int | None:
    uniq = sorted(set(map(int, values)))
    assert len(uniq) <= 1
    return uniq[0] if uniq else None


def _candidate_intervals(
    length: int,
    inside: set[int],
    outside: set[int],
    low_fixed: int | None,
    high_fixed: int | None,
) -> list[tuple[int, int]]:
    starts = [int(low_fixed)] if low_fixed is not None else list(range(length))
    ends = [int(high_fixed)] if high_fixed is not None else list(range(length))
    inside = set(map(int, inside))
    outside = set(map(int, outside))

    out: list[tuple[int, int]] = []
    for a in starts:
        if a < 0 or a >= length:
            continue
        for b in ends:
            if b < 0 or b >= length or a > b:
                continue
            if any(not (a <= x <= b) for x in inside):
                continue
            if any(a <= x <= b for x in outside):
                continue
            out.append((int(a), int(b)))
    return out


def _candidate_rectangles_from_rays(grid: np.ndarray, rays: dict[str, list[dict[str, int]]] | None = None) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rays = _parse_rays(g) if rays is None else rays

    row_inside = {ray["row"] for ray in rays["left"]} | {ray["row"] for ray in rays["right"]}
    row_outside = {ray["row"] for ray in rays["full_h"]}
    col_inside = {ray["col"] for ray in rays["top"]} | {ray["col"] for ray in rays["bottom"]}
    col_outside = {ray["col"] for ray in rays["full_v"]}

    r0_fixed = _unique_or_none([ray["end"] + 1 for ray in rays["top"]])
    r1_fixed = _unique_or_none([ray["start"] - 1 for ray in rays["bottom"]])
    c0_fixed = _unique_or_none([ray["end"] + 1 for ray in rays["left"]])
    c1_fixed = _unique_or_none([ray["start"] - 1 for ray in rays["right"]])

    row_spans = _candidate_intervals(h, row_inside, row_outside, r0_fixed, r1_fixed)
    col_spans = _candidate_intervals(w, col_inside, col_outside, c0_fixed, c1_fixed)
    return [(r0, c0, r1, c1) for (r0, r1) in row_spans for (c0, c1) in col_spans]


def _solve_rectangle(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[str, list[dict[str, int]]]]:
    rays = _parse_rays(grid)
    rects = _candidate_rectangles_from_rays(grid, rays)
    assert len(rects) == 1, f"expected unique hidden rectangle, found {len(rects)}"
    return rects[0], rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique invisible rectangle from interrupted edge probes, fill it yellow, and let every stopped probe continue through it.

    Input structure:
    - A black background contains several one-cell-thick colored probe rays entering perpendicularly from the grid edges.
    - Some probes run all the way across a row or column; others stop exactly one cell before an unseen solid axis-aligned rectangle.
    - The rectangle itself is never drawn in the input, so only the probe endpoints and uninterrupted rows or columns reveal it.

    Rule:
    - Infer the single rectangle consistent with every stopped probe touching one of its sides and every full probe proving that its row or column stays outside the rectangle.
    - Paint that recovered rectangle yellow.
    - For each interrupted probe, keep its original segment and continue the same-color ray from the opposite side of the rectangle to the far border in the same direction.

    Key insight:
    - The rectangle is defined jointly by positive evidence from where probes stop and negative evidence from full rays, especially tangent full rows or columns that rule out larger candidate rectangles.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    (r0, c0, r1, c1), rays = _solve_rectangle(g)

    out[r0 : r1 + 1, c0 : c1 + 1] = YELLOW

    for ray in rays["left"]:
        out[ray["row"], c1 + 1 :] = ray["color"]
    for ray in rays["right"]:
        out[ray["row"], :c0] = ray["color"]
    for ray in rays["top"]:
        out[r1 + 1 :, ray["col"]] = ray["color"]
    for ray in rays["bottom"]:
        out[:r0, ray["col"]] = ray["color"]
    return out


def _candidate_count(grid: np.ndarray, rays: dict[str, list[dict[str, int]]] | None = None) -> int:
    return len(_candidate_rectangles_from_rays(grid, rays))


def _remove_rows(rays: dict[str, list[dict[str, int]]], kind: str, rows_to_remove: set[int]) -> dict[str, list[dict[str, int]]]:
    clone: dict[str, list[dict[str, int]]] = {}
    for key, items in rays.items():
        if key == kind:
            clone[key] = [item.copy() for item in items if int(item["row"]) not in rows_to_remove]
        else:
            clone[key] = [item.copy() for item in items]
    return clone


def _remove_cols(rays: dict[str, list[dict[str, int]]], kind: str, cols_to_remove: set[int]) -> dict[str, list[dict[str, int]]]:
    clone: dict[str, list[dict[str, int]]] = {}
    for key, items in rays.items():
        if key == kind:
            clone[key] = [item.copy() for item in items if int(item["col"]) not in cols_to_remove]
        else:
            clone[key] = [item.copy() for item in items]
    return clone


def _render_from_rays(shape: tuple[int, int], rays: dict[str, list[dict[str, int]]], rect: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, rect)
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    for ray in rays["full_h"]:
        inp[ray["row"], :] = ray["color"]
        out[ray["row"], :] = ray["color"]
    for ray in rays["full_v"]:
        inp[:, ray["col"]] = ray["color"]
        out[:, ray["col"]] = ray["color"]

    for ray in rays["left"]:
        inp[ray["row"], : ray["end"] + 1] = ray["color"]
        out[ray["row"], : ray["end"] + 1] = ray["color"]
        out[ray["row"], c1 + 1 :] = ray["color"]
    for ray in rays["right"]:
        inp[ray["row"], ray["start"] :] = ray["color"]
        out[ray["row"], ray["start"] :] = ray["color"]
        out[ray["row"], :c0] = ray["color"]
    for ray in rays["top"]:
        inp[: ray["end"] + 1, ray["col"]] = ray["color"]
        out[: ray["end"] + 1, ray["col"]] = ray["color"]
        out[r1 + 1 :, ray["col"]] = ray["color"]
    for ray in rays["bottom"]:
        inp[ray["start"] :, ray["col"]] = ray["color"]
        out[ray["start"] :, ray["col"]] = ray["color"]
        out[:r0, ray["col"]] = ray["color"]

    out[r0 : r1 + 1, c0 : c1 + 1] = YELLOW
    return inp, out


def _build_horizontal_case() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(10, 18)
    w = random.randint(12, 24)
    rect_h = random.randint(3, 5)
    rect_w = random.randint(3, min(8, w - 5))

    r0 = random.randint(2, h - rect_h - 3)
    c0 = random.randint(2, w - rect_w - 3)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    rows = list(range(r0, r1 + 1))
    left_rows = {r0}
    right_rows = {r1}
    inner_rows = [r for r in rows if r not in left_rows | right_rows]
    random.shuffle(inner_rows)
    for r in inner_rows:
        choice = random.choice(("left", "right", "none"))
        if choice == "left":
            left_rows.add(r)
        elif choice == "right":
            right_rows.add(r)

    if not left_rows or not right_rows:
        raise AssertionError("both interrupted sides must be non-empty")

    full_rows = {r0 - 1, r1 + 1}
    above = [r for r in range(0, r0 - 1)]
    below = [r for r in range(r1 + 2, h)]
    if above and random.random() < 0.6:
        full_rows.add(random.choice(above))
    if below and random.random() < 0.6:
        full_rows.add(random.choice(below))

    extra_full = sorted(full_rows - {r0 - 1, r1 + 1})
    while len(left_rows) + len(right_rows) + len(full_rows) > len(RAY_COLORS) and extra_full:
        full_rows.remove(extra_full.pop())

    removable_rows = sorted((left_rows | right_rows) - {r0, r1})
    random.shuffle(removable_rows)
    while len(left_rows) + len(right_rows) + len(full_rows) > len(RAY_COLORS) and removable_rows:
        row = removable_rows.pop()
        left_rows.discard(row)
        right_rows.discard(row)

    ray_count = len(left_rows) + len(right_rows) + len(full_rows)
    assert ray_count <= len(RAY_COLORS)
    colors = list(RAY_COLORS)
    random.shuffle(colors)

    rays = {"full_h": [], "full_v": [], "left": [], "right": [], "top": [], "bottom": []}
    idx = 0
    for row in sorted(full_rows):
        rays["full_h"].append({"row": int(row), "color": int(colors[idx])})
        idx += 1
    for row in sorted(left_rows):
        rays["left"].append({"row": int(row), "end": int(c0 - 1), "color": int(colors[idx])})
        idx += 1
    for row in sorted(right_rows):
        rays["right"].append({"row": int(row), "start": int(c1 + 1), "color": int(colors[idx])})
        idx += 1

    return _render_from_rays((h, w), rays, (r0, c0, r1, c1))


def _semantic_checks(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    rays = _parse_rays(inp)
    rects = _candidate_rectangles_from_rays(inp, rays)
    if len(rects) != 1:
        return False
    rect = rects[0]
    if not np.array_equal(main(inp), expected_out):
        return False

    has_horizontal = bool(rays["left"] or rays["right"])
    has_vertical = bool(rays["top"] or rays["bottom"])
    if has_horizontal == has_vertical:
        return False

    r0, c0, r1, c1 = rect
    if has_horizontal:
        if not rays["left"] or not rays["right"] or rays["top"] or rays["bottom"] or rays["full_v"]:
            return False
        full_rows = {ray["row"] for ray in rays["full_h"]}
        if {r0 - 1, r1 + 1} - full_rows:
            return False
        if _candidate_count(inp, _remove_rows(rays, "full_h", full_rows)) <= 1:
            return False
        if _candidate_count(inp, {"full_h": [x.copy() for x in rays["full_h"]], "full_v": [], "left": [], "right": [x.copy() for x in rays["right"]], "top": [], "bottom": []}) <= 1:
            return False
        if _candidate_count(inp, {"full_h": [x.copy() for x in rays["full_h"]], "full_v": [], "left": [x.copy() for x in rays["left"]], "right": [], "top": [], "bottom": []}) <= 1:
            return False
    else:
        if not rays["top"] or not rays["bottom"] or rays["left"] or rays["right"] or rays["full_h"]:
            return False
        full_cols = {ray["col"] for ray in rays["full_v"]}
        if {c0 - 1, c1 + 1} - full_cols:
            return False
        if _candidate_count(inp, _remove_cols(rays, "full_v", full_cols)) <= 1:
            return False
        if _candidate_count(inp, {"full_h": [], "full_v": [x.copy() for x in rays["full_v"]], "left": [], "right": [], "top": [], "bottom": [x.copy() for x in rays["bottom"]]}) <= 1:
            return False
        if _candidate_count(inp, {"full_h": [], "full_v": [x.copy() for x in rays["full_v"]], "left": [], "right": [], "top": [x.copy() for x in rays["top"]], "bottom": []}) <= 1:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, out = _build_horizontal_case()
        if random.random() < 0.5:
            inp, out = _flip_pair(inp, out, horizontal=random.random() < 0.5)
        inp, out = _rotate_pair(inp, out, random.randrange(4))
        if _semantic_checks(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid invisible-prism probe scene")
