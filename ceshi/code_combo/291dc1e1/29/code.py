# Combo variant of 291dc1e1 (index 29)
# combo_spec: D4/2+D9/1
# Name: 不规则分布+十字交叉线
#
# Design analysis
# - Concrete behavior:
#   Original inputs put the blue/red axes on a corner, so distances could be read
#   from two fixed edge directions and the rectangles only lived in two parallel
#   bands at db in {1, 4}. Here the two full-length axes cross in the middle of
#   the board, and rectangles are scattered in all four quadrants. A small example
#   with blue horizontal and red vertical:
#     * original: a 2x4 rectangle at rows 2..3 under a top blue edge has db=1
#       just because it is in the first band;
#     * fused: a 2x4 rectangle in the upper-left quadrant with bbox rows 4..5 and
#       cols 2..5 around a center cross at row 8 / col 10 has db=2 and dr=4, so
#       the solver must compute the true gap from its bbox to each middle axis.
# - Complexity calibration:
#   The visual signature must clearly differ from the seed task, so the generator
#   enforces all four quadrants to be occupied and requires at least one distance
#   value outside the old {1, 4} bands. At the same time the rectangles keep the
#   original size family (thin side 2, thick side in {2,4,6,8}) so the task does
#   not drift into a different object vocabulary.
# - Hidden assumptions broken in the original solver:
#   1. Corner-marker search no longer works: the marker is now at the center.
#   2. "Mask out corner row + corner col" is invalid: the relevant row/col are no
#      longer on the border and depend on which color is horizontal/vertical.
#   3. Tie-breaking "from the corner" is meaningless now, so the solver uses
#      axis-aligned top/left order only as a deterministic fallback; the generator
#      avoids equal (db, dr) pairs so this fallback is not the main signal.
#   4. Band-based placement assumptions are gone: rectangles can appear at many
#      distinct distances, so distance must be measured per bbox.
# - Placement legality:
#   The generator places one or more rectangles in each quadrant, keeps a 1-cell
#   moat around every rectangle, and keeps a 1-cell gap from both cross lines.
#   This guarantees no overlap, no out-of-bounds paste, and clean connected
#   components for the solver.
# - Human observability / cleanliness:
#   The full blue and red lines remain visually simple, crossing once at a black
#   marker. Rectangles use the original palette and remain solid rectangles, so the
#   only new reasoning burden is "measure distances to the center cross" instead of
#   "read one of two bands". The layout stays readable because objects do not touch
#   each other or the lines.
# - New solver assumptions introduced here:
#   1. Blue is still color 1, red is still color 2, marker is still 0.
#   2. Each line is a full row or column with exactly one marker hole at the cross.
#   3. Rectangles do not contain color 0 internally, so the marker stays unique.
#   The generator explicitly enforces all three assumptions.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
THICK_OPTIONS = (2, 4, 6, 8)
QUADRANTS = ("tl", "tr", "bl", "br")


def _find_line(g: np.ndarray, line_color: int, marker: int) -> tuple[str, int, tuple[int, int]]:
    H, W = g.shape
    hits: list[tuple[str, int, tuple[int, int]]] = []

    for r in range(H):
        row = g[r, :]
        if (
            int(np.count_nonzero(row == line_color)) == W - 1
            and int(np.count_nonzero(row == marker)) == 1
            and bool(np.all((row == line_color) | (row == marker)))
        ):
            c = int(np.flatnonzero(row == marker)[0])
            hits.append(("h", int(r), (int(r), c)))

    for c in range(W):
        col = g[:, c]
        if (
            int(np.count_nonzero(col == line_color)) == H - 1
            and int(np.count_nonzero(col == marker)) == 1
            and bool(np.all((col == line_color) | (col == marker)))
        ):
            r = int(np.flatnonzero(col == marker)[0])
            hits.append(("v", int(c), (r, int(c))))

    if len(hits) != 1:
        raise AssertionError(f"expected exactly one line for color {line_color}, got {len(hits)}")
    return hits[0]


def _dist_to_line(box: tuple[int, int, int, int], orient: str, line_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        if c0 > line_idx:
            return int(c0 - line_idx - 1)
        if c1 < line_idx:
            return int(line_idx - c1 - 1)
        return 0
    if r0 > line_idx:
        return int(r0 - line_idx - 1)
    if r1 < line_idx:
        return int(line_idx - r1 - 1)
    return 0


def _along_axis(box: tuple[int, int, int, int], orient: str) -> int:
    r0, c0, _r1, _c1 = map(int, box)
    return int(c0 if orient == "h" else r0)


def _extract_rects(g: np.ndarray, bg: int, line_mask: np.ndarray) -> list[dict]:
    mask = (g != int(bg)) & (~line_mask)
    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": arr.copy()})
    if not rects:
        raise AssertionError("no rectangles found")
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blue_orient, blue_idx, blue_cross = _find_line(g, BLUE, MARKER)
    red_orient, red_idx, red_cross = _find_line(g, RED, MARKER)
    if blue_orient == red_orient:
        raise AssertionError("blue and red lines must be perpendicular")
    if blue_cross != red_cross:
        raise AssertionError("blue/red lines must cross at the same marker")

    H, W = g.shape
    line_mask = np.zeros((H, W), dtype=bool)
    if blue_orient == "h":
        line_mask[blue_idx, :] = True
    else:
        line_mask[:, blue_idx] = True
    if red_orient == "h":
        line_mask[red_idx, :] = True
    else:
        line_mask[:, red_idx] = True

    rects = _extract_rects(g, bg, line_mask)
    for rect in rects:
        rect["db"] = int(_dist_to_line(rect["box"], blue_orient, blue_idx))
        rect["dr"] = int(_dist_to_line(rect["box"], red_orient, red_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            _along_axis(rect["box"], blue_orient),
            _along_axis(rect["box"], red_orient),
        )
    )

    seq: list[np.ndarray] = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        c = int((out_w - w) // 2)
        paste(out, arr, int(r), int(c))
        r += int(h)
    return out


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or float(np.random.rand()) < 0.6:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    choices = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    colors = np.random.choice(choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for i, cell in enumerate(cells):
        flat[int(cell)] = int(colors[i % n_colors])
    return patch


def _make_patch(blue_orient: str, max_thick: int) -> np.ndarray:
    feasible = [int(t) for t in THICK_OPTIONS if int(t) <= int(max_thick)]
    if not feasible:
        raise ValueError("no feasible rectangle thickness")
    thick = int(np.random.choice(feasible))
    shape = (2, thick) if blue_orient == "h" else (thick, 2)
    dominant = int(np.random.choice(DOMINANT_COLORS))
    patch = np.full(shape, dominant, dtype=int)
    return _sprinkle_minorities(patch, dominant)


def _is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(H, int(r0) + int(h) + 1)
    cc1 = min(W, int(c0) + int(w) + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _quadrant_ranges(
    H: int, W: int, hline: int, vline: int, quadrant: str, h: int, w: int
) -> tuple[range, range]:
    if quadrant[0] == "t":
        rows = range(1, int(hline) - int(h))
    else:
        rows = range(int(hline) + 2, int(H) - int(h))

    if quadrant[1] == "l":
        cols = range(1, int(vline) - int(w))
    else:
        cols = range(int(vline) + 2, int(W) - int(w))
    return rows, cols


def _place_rectangles(grid: np.ndarray, blue_orient: str, hline: int, vline: int) -> list[dict] | None:
    H, W = grid.shape
    occupied = grid != BG
    count = int(np.random.randint(5, 8))
    quadrants = list(QUADRANTS)
    quadrants.extend(str(np.random.choice(QUADRANTS)) for _ in range(count - 4))
    np.random.shuffle(quadrants)

    rects: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()

    max_horizontal_span = min(int(vline) - 2, int(W - vline - 3))
    max_vertical_span = min(int(hline) - 2, int(H - hline - 3))
    max_thick = max_horizontal_span if blue_orient == "h" else max_vertical_span
    if max_thick < 2:
        return None

    for quadrant in quadrants:
        placed = False
        for _ in range(50):
            patch = _make_patch(blue_orient, max_thick)
            h, w = map(int, patch.shape)
            rows, cols = _quadrant_ranges(H, W, hline, vline, quadrant, h, w)
            candidates: list[tuple[int, int, int, int]] = []
            for r0 in rows:
                for c0 in cols:
                    if not _is_free(occupied, r0, c0, h, w):
                        continue
                    box = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
                    db = int(
                        _dist_to_line(
                            box,
                            blue_orient,
                            int(hline if blue_orient == "h" else vline),
                        )
                    )
                    dr = int(
                        _dist_to_line(
                            box,
                            "v" if blue_orient == "h" else "h",
                            int(vline if blue_orient == "h" else hline),
                        )
                    )
                    if db <= 0 or dr <= 0 or (db, dr) in used_pairs:
                        continue
                    candidates.append((int(r0), int(c0), db, dr))
            if not candidates:
                continue
            r0, c0, db, dr = candidates[int(np.random.randint(len(candidates)))]
            paste(grid, patch, r0, c0)
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            used_pairs.add((int(db), int(dr)))
            rects.append(
                {
                    "box": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                    "db": int(db),
                    "dr": int(dr),
                    "quadrant": quadrant,
                }
            )
            placed = True
            break
        if not placed:
            return None

    db_values = {int(rect["db"]) for rect in rects}
    dr_values = {int(rect["dr"]) for rect in rects}
    if len(db_values) < 3 or len(dr_values) < 3:
        return None
    if all(int(v) in {1, 4} for v in db_values):
        return None
    if all(int(v) in {1, 4} for v in dr_values):
        return None
    if {str(rect["quadrant"]) for rect in rects} != set(QUADRANTS):
        return None
    return rects


def _build_input() -> np.ndarray | None:
    blue_orient = str(np.random.choice(["h", "v"]))
    if blue_orient == "h":
        H = int(np.random.choice([17, 19, 21]))
        W = int(np.random.choice([21, 23, 25, 27]))
    else:
        H = int(np.random.choice([21, 23, 25, 27]))
        W = int(np.random.choice([17, 19, 21]))

    hline = int(H // 2)
    vline = int(W // 2)
    grid = np.full((H, W), BG, dtype=int)

    if blue_orient == "h":
        grid[hline, :] = BLUE
        grid[:, vline] = RED
    else:
        grid[:, vline] = BLUE
        grid[hline, :] = RED
    grid[hline, vline] = MARKER

    rects = _place_rectangles(grid, blue_orient, hline, vline)
    if rects is None:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo 29")
