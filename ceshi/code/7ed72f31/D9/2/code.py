# Variant of 7ed72f31
# Dimension: D9 — 原始像素的保留与删除 (transform)
# Change: 反射后原始payload像素和红色标记都保留在输出中... → 原始像素重新着色

from __future__ import annotations

"""
Design analysis for the D9 variant
==================================

a) Concrete behavior and trace
spec:
  unchanged_geometry:
    - Horizontal red line -> reflect payload across that row.
    - Vertical red line -> reflect payload across that column.
    - Isolated red pixel -> point-reflect the adjacent 8-connected payload component.
  changed_transform:
    - Input keeps the original payload in its original color.
    - Output keeps the red marker unchanged.
    - Output adds the reflected copy in the original payload color.
    - Output recolors the original payload to a fixed grey (5).

Small trace:
  Original rule:
    input: blue payload above a red row.
    output: blue payload still above the row, plus a blue mirrored copy below.
  Variant rule:
    input: same blue payload above the red row.
    output: the upper original payload becomes grey, the lower mirrored copy stays blue,
            and the red row stays red.
  Difference:
    the geometry is unchanged; only the "which side is original" cue changes.

b) Complexity calibration
  The recolor target is a single fixed color (grey 5), not a large mapping table.
  This keeps the rule inferable while still forcing the solver to distinguish
  original pixels from generated mirror pixels. Each generated grid contains
  multiple groups with different payload colors, so "all originals turn grey"
  is visibly repeated evidence rather than a one-off coincidence.

c) Implicit assumptions in the original main()
  - out = g.copy(): previously harmless because originals stayed unchanged.
    In the variant this is no longer enough; we must explicitly recolor source coords.
  - Reflection loops write mirrored pixels with the source color.
    This still holds; the mirrored copy should remain original-colored.
  - The solver scans the whole grid for eligible payload pixels of the component's color.
    That was harmless when nearby same-color noise was optional. In the variant,
    accidental extra source pixels would also be recolored grey, so the generator must
    avoid same-color distractors and keep groups separated.
  - Isolated-red handling mirrors all adjacent same-color components touched in 8-neighborhood.
    This still works, but now those original component pixels must also be added to the
    recolor set after mirroring.

d) Placement legality
  The generator builds full output groups first, then derives the input by removing only
  the mirrored copy and restoring grey originals back to their payload color. Groups are
  placed on non-overlapping rectangles with a 1-cell moat between rectangles, so:
    - no object overlap,
    - no accidental 4-connected merging,
    - no isolated marker seeing a foreign diagonal neighbor,
    - all reflected pixels land on background only.

e) Visual observability
  The recolor is easy to see because grey appears only in outputs, never in inputs.
  Each grid has at least one row-axis group, one col-axis group, and one isolated-center
  group, so the unchanged geometric rule remains visible while D9 is also obvious.

f) Visual cleanliness
  No free same-color noise is generated. That keeps the scene readable and prevents the
  bbox-based eligibility in the solver from pulling in unrelated pixels.

g) Color-change inferability
  The color rule is fixed and repeated: original payload -> grey (5), mirror -> keep
  original color, marker -> stay red (2). Background is never grey, and payload colors
  are never grey or red, so the appearance of grey in the output is a strong structural cue.

h) New main() assumptions after the change
  - Grey 5 must not be the background, otherwise recoloring would erase originals.
  - Payload colors must not already be grey, otherwise source-vs-mirror would collapse.
  - Different groups should use distinct payload colors, otherwise the original
    bbox-based "same color" search could merge logical groups. The generator enforces all
    three conditions.
"""

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


MARK = 2
RECOLOR = 5
PALETTE = tuple(range(10))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red (2) markers as reflection axes / centers.

    Variant rule:
    - The reflected copy keeps the original payload color.
    - The original payload pixels are recolored to grey (5).
    - Red markers remain red.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()
    source_pixels: set[tuple[int, int]] = set()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    # 1) Components that already contain the red marker.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in vals:
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        rs = {int(r) for r, _ in reds}
        cs = {int(c) for _, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
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

                source_pixels.add((int(r), int(c)))
                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    # 2) Isolated red pixels reflect their adjacent 8-connected component(s).
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
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                pts = np.argwhere(labv == int(lbl))
                for r, c in pts:
                    r = int(r)
                    c = int(c)
                    source_pixels.add((r, c))
                    rr, cc = refl_point((int(pr), int(pc)), r, c)
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    for r, c in source_pixels:
        if int(g[int(r), int(c)]) not in (int(bg), int(MARK)):
            out[int(r), int(c)] = int(RECOLOR)

    return out


def _neighbors(height: int, width: int, cell: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < height and 0 <= cc < width:
                out.append((rr, cc))
    return out


def _grow_connected_shape(
    height: int,
    width: int,
    size: int,
    anchor: tuple[int, int],
    connectivity: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    cells = {anchor}
    frontier = set(_neighbors(height, width, anchor, connectivity)) - cells
    while len(cells) < int(size):
        if not frontier:
            raise ValueError("frontier exhausted while growing shape")
        pick = rng.choice(tuple(frontier))
        frontier.remove(pick)
        cells.add(pick)
        for nxt in _neighbors(height, width, pick, connectivity):
            if nxt not in cells:
                frontier.add(nxt)
    return cells


def _transform_rows(cells: set[tuple[int, int]], axis_r: int) -> set[tuple[int, int]]:
    return {(int(2 * axis_r - r), int(c)) for r, c in cells}


def _transform_cols(cells: set[tuple[int, int]], axis_c: int) -> set[tuple[int, int]]:
    return {(int(r), int(2 * axis_c - c)) for r, c in cells}


def _make_row_group(color: int, rng: random.Random) -> dict:
    line_len = rng.randint(2, 5)
    depth = rng.randint(2, 4)
    size = rng.randint(3, min(9, line_len * depth))
    anchor = (depth - 1, rng.randrange(line_len))
    source = _grow_connected_shape(depth, line_len, size, anchor, 4, rng)
    axis_r = depth
    marker = {(axis_r, c) for c in range(line_len)}
    mirror = {(int(2 * axis_r - r), int(c)) for r, c in source}
    if rng.random() < 0.5:
        source = _transform_rows(source, axis_r)
        mirror = _transform_rows(mirror, axis_r)
    h, w = 2 * depth + 1, line_len
    return {
        "type": "row",
        "color": int(color),
        "h": int(h),
        "w": int(w),
        "source": source,
        "marker": marker,
        "mirror": mirror,
    }


def _make_col_group(color: int, rng: random.Random) -> dict:
    line_len = rng.randint(2, 5)
    depth = rng.randint(2, 4)
    size = rng.randint(3, min(9, line_len * depth))
    anchor = (rng.randrange(line_len), depth - 1)
    source = _grow_connected_shape(line_len, depth, size, anchor, 4, rng)
    axis_c = depth
    marker = {(r, axis_c) for r in range(line_len)}
    mirror = {(int(r), int(2 * axis_c - c)) for r, c in source}
    if rng.random() < 0.5:
        source = _transform_cols(source, axis_c)
        mirror = _transform_cols(mirror, axis_c)
    h, w = line_len, 2 * depth + 1
    return {
        "type": "col",
        "color": int(color),
        "h": int(h),
        "w": int(w),
        "source": source,
        "marker": marker,
        "mirror": mirror,
    }


def _make_isolated_group(color: int, rng: random.Random) -> dict:
    extent = rng.randint(2, 3)
    size = rng.randint(1, min(6, extent * extent))
    anchor = (extent - 1, extent - 1)
    source = _grow_connected_shape(extent, extent, size, anchor, 8, rng)
    marker_pt = (extent, extent)
    mirror = {(int(2 * extent - r), int(2 * extent - c)) for r, c in source}
    flip_r = rng.random() < 0.5
    flip_c = rng.random() < 0.5
    if flip_r:
        source = _transform_rows(source, extent)
        mirror = _transform_rows(mirror, extent)
    if flip_c:
        source = _transform_cols(source, extent)
        mirror = _transform_cols(mirror, extent)
    h = w = 2 * extent + 1
    return {
        "type": "isolated",
        "color": int(color),
        "h": int(h),
        "w": int(w),
        "source": source,
        "marker": {marker_pt},
        "mirror": mirror,
    }


def _place_group(
    inp: np.ndarray,
    out: np.ndarray,
    blocked: np.ndarray,
    group: dict,
    rng: random.Random,
) -> dict | None:
    H, W = inp.shape
    h = int(group["h"])
    w = int(group["w"])
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    rng.shuffle(positions)
    for top, left in positions:
        if np.any(blocked[int(top) : int(top + h), int(left) : int(left + w)]):
            continue

        for r, c in group["source"]:
            inp[int(top + r), int(left + c)] = int(group["color"])
            out[int(top + r), int(left + c)] = int(RECOLOR)
        for r, c in group["marker"]:
            inp[int(top + r), int(left + c)] = int(MARK)
            out[int(top + r), int(left + c)] = int(MARK)
        for r, c in group["mirror"]:
            out[int(top + r), int(left + c)] = int(group["color"])

        br0 = max(0, int(top - 1))
        bc0 = max(0, int(left - 1))
        br1 = min(H, int(top + h + 1))
        bc1 = min(W, int(left + w + 1))
        blocked[br0:br1, bc0:bc1] = True

        return {
            "type": group["type"],
            "color": int(group["color"]),
            "source_size": int(len(group["source"])),
            "top": int(top),
            "left": int(left),
            "h": int(h),
            "w": int(w),
        }
    return None


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    H = rng.randint(18, 28)
    W = rng.randint(18, 28)
    bg = rng.choice([c for c in PALETTE if c not in (MARK, RECOLOR)])
    n_groups = rng.randint(3, 5)

    colors = [c for c in PALETTE if c not in (bg, MARK, RECOLOR)]
    rng.shuffle(colors)
    payload_colors = colors[:n_groups]

    types = ["row", "col", "isolated"]
    while len(types) < n_groups:
        types.append(rng.choice(["row", "col", "isolated"]))
    rng.shuffle(types)

    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    metas = []
    for kind, color in zip(types, payload_colors):
        if kind == "row":
            group = _make_row_group(int(color), rng)
        elif kind == "col":
            group = _make_col_group(int(color), rng)
        else:
            group = _make_isolated_group(int(color), rng)
        placed = _place_group(inp, out, blocked, group, rng)
        if placed is None:
            raise ValueError(f"could not place {kind} group")
        metas.append(placed)

    solved = main(inp)
    if not np.array_equal(solved, out):
        raise AssertionError("generator built a case that solver does not reproduce")

    recolored = int(np.sum((inp != int(bg)) & (inp != int(MARK)) & (out == int(RECOLOR))))
    mirrored = int(np.sum((inp == int(bg)) & (out != int(bg)) & (out != int(MARK))))
    if recolored <= 0 or mirrored <= 0:
        raise AssertionError("variant behavior missing")

    return inp, out, {
        "bg": int(bg),
        "recolored_pixels": recolored,
        "mirrored_pixels": mirrored,
        "types": [m["type"] for m in metas],
        "source_sizes": [m["source_size"] for m in metas],
        "groups": metas,
    }


def generate_input() -> np.ndarray:
    for _ in range(200):
        try:
            inp, _out, _meta = _build_case(random.Random())
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid D9 variant instance after many attempts")

