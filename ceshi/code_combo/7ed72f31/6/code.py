# Combo variant of 7ed72f31 (index 6)
# combo_spec: D2/0+D10/0
# Name: 统计识别标记+图例解读

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg, label_mask


"""
Design analysis for the fused variant
====================================

a) Concrete behavior and trace
------------------------------
- Base task: a special marker color defines a row axis, a column axis, or a point
  center; payload pixels are reflected with the same color.
- Fused task:
  1. Read the top-edge legend first. Each 2x2 stacked chip pair means A->B.
  2. Infer the marker color by counting all non-background cells in the whole
     input; the unique least-frequent non-background color is the marker.
  3. Reflect payload geometry exactly as in the source task, but paint the new
     reflected pixels using the legend target color.
- Concrete trace:
  legend rows say 4->6 and 2->8. The least frequent non-bg color in the full
  grid is 3, so 3 is the marker. A 4-colored sprite attached to a 3-row axis is
  mirrored across that row, but the reflected copy is colored 6 rather than 4.
  A 2-colored sprite next to an isolated 3-pixel is point-reflected, but the new
  copy is colored 8.
- The visible difference from the source task is twofold: the marker is not fixed
  to red, and mirrored copies can change color according to the edge legend.

b) Complexity calibration
-------------------------
- Keep only 2 or 3 legend mappings, arranged as a cycle/permutation among the
  scene colors. This avoids "legend-only" colors that would otherwise become the
  true least-frequent color and break D2.
- Keep exactly three scene groups per sample: one row reflection, one column
  reflection, one isolated point reflection. This preserves the original task's
  variety while keeping marker counts low enough to stay uniquely least-frequent.
- Every legend color appears both as a source and as a destination, and every
  source color is exercised by at least one real reflection group.

c) Implicit-assumption review of the original main()
----------------------------------------------------
- `MARK = 2` is invalid. Replace it with a counting step over all non-bg cells.
- The old solver scanned the full grid using payload color + bbox heuristics.
  That breaks once legend colors repeat across groups. Fix: reflect only the
  concrete payload pixels inside the current component.
- The old solver treated all non-background pixels as scene content. That no
  longer holds because the legend occupies the top edge. Fix: blank rows 0-3
  before component detection.
- New fused assumptions:
  1. The legend always uses 2x2 stacked chips in rows 0-1 with blank separator
     rows 2-3.
  2. The marker color is the unique least-frequent non-background color.
  3. Marker pixels are used only for axis/center geometry, not inside the legend.
  4. Scene groups are separated by at least a one-cell moat so 4/8-connected
     component logic stays unambiguous.
  `generate_input()` explicitly enforces all four.

d) Placement legality
---------------------
- The generator is output-first: it places each full output footprint, including
  mirrored copies, onto an empty scene area below the legend band.
- Placement checks include a one-cell moat around the full output footprint, so
  no group overlaps, no merged components appear, and isolated markers remain the
  unique adjacent center for their payload component.

e) Visual observability
-----------------------
- Humans can see the legend because it is isolated in a dedicated top strip.
- Humans can see that the marker is not fixed red because the marker color varies
  per sample and is selected by rarity, while red can appear as an ordinary scene
  color in some samples.
- Humans can see the recoloring because mirrored copies visibly change from source
  colors to the legend-indicated target colors.

f) Visual cleanliness
---------------------
- The legend is regular and small.
- The scene begins only below two blank separator rows.
- Only three groups are placed, with one group per geometry type, so the scene
  remains sparse and readable.

g) Color-change inferability
----------------------------
- The mapping is a small 2-cycle or 3-cycle among 2 or 3 scene colors.
- Every mapped color appears multiple times: in the legend and in at least one
  source sprite. No arbitrary per-object recoloring is used.
- The non-degeneracy check used after implementation compares the fused output to
  an identity-color baseline with the same inferred marker and geometry; generated
  samples must differ, proving the legend mapping is behaviorally active.
"""


LEGEND_ROWS = 2
SCENE_TOP = 4
PALETTE = list(range(10))


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _infer_marker_color(grid: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in grid.flatten() if int(v) != int(bg)]
    if not vals:
        raise ValueError("grid has no non-background colors")
    counts = Counter(vals)
    least_color, least_count = min(counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
    if sum(1 for v in counts.values() if int(v) == int(least_count)) != 1:
        raise ValueError("marker color is not uniquely least frequent")
    return int(least_color)


def _parse_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if grid.shape[0] < LEGEND_ROWS:
        return mapping
    c = 0
    while c + 1 < grid.shape[1]:
        top0 = int(grid[0, c])
        top1 = int(grid[0, c + 1])
        bot0 = int(grid[1, c])
        bot1 = int(grid[1, c + 1])
        if top0 != int(bg) and top0 == top1 and bot0 != int(bg) and bot0 == bot1:
            mapping[int(top0)] = int(bot0)
            c += 2
            while c < grid.shape[1] and int(grid[0, c]) == int(bg) and int(grid[1, c]) == int(bg):
                c += 1
            continue
        c += 1
    return mapping


def _solve_core(grid: np.ndarray, mapping: dict[int, int], mark: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    out = g.copy()

    scene = g.copy()
    scene[: min(SCENE_TOP, h), :] = int(bg)

    lab, n = label_mask(scene != int(bg), connectivity=4)
    isolated_markers: list[tuple[int, int]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        vals = [int(scene[r, c]) for r, c in coords]
        marks = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(mark)]
        if not marks:
            continue
        if len(coords) == 1 and len(marks) == 1:
            isolated_markers.append(marks[0])
            continue

        rs = {int(r) for r, _c in marks}
        cs = {int(c) for _r, c in marks}
        if len(marks) == 1:
            axis = ("point", marks[0])
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        payload_pixels = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(mark)]
        if not payload_pixels:
            continue

        for r, c, v in payload_pixels:
            if axis[0] == "row":
                rr, cc = _refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = _refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = _refl_point(axis[1], int(r), int(c))
            if _inb((h, w), rr, cc):
                out[int(rr), int(cc)] = int(mapping.get(int(v), int(v)))

    payload_mask = (scene != int(bg)) & (scene != int(mark))
    payload_lab, _payload_n = label_mask(payload_mask, connectivity=8)
    for pr, pc in isolated_markers:
        neigh_labels: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb((h, w), rr, cc):
                    continue
                lbl = int(payload_lab[rr, cc])
                if lbl != 0:
                    neigh_labels.add(lbl)
        for lbl in neigh_labels:
            for r, c in np.argwhere(payload_lab == int(lbl)):
                v = int(scene[int(r), int(c)])
                rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                if _inb((h, w), rr, cc):
                    out[int(rr), int(cc)] = int(mapping.get(int(v), int(v)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    mapping = _parse_legend(g, bg)
    mark = int(_infer_marker_color(g, bg))
    return _solve_core(g, mapping, mark)


def _normalize_from_origin(cells: list[tuple[int, int, int]], min_r: int, min_c: int) -> list[tuple[int, int, int]]:
    return [(int(r - min_r), int(c - min_c), int(v)) for r, c, v in cells]


def _normalize_group(
    output_cells: list[tuple[int, int, int]],
    input_cells: list[tuple[int, int, int]],
    mirror_cells: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]], int, int]:
    min_r = min(int(r) for r, _c, _v in output_cells)
    min_c = min(int(c) for _r, c, _v in output_cells)
    norm_out = _normalize_from_origin(output_cells, min_r, min_c)
    norm_in = _normalize_from_origin(input_cells, min_r, min_c)
    norm_mirror = _normalize_from_origin(mirror_cells, min_r, min_c)
    max_r = max(int(r) for r, _c, _v in norm_out)
    max_c = max(int(c) for _r, c, _v in norm_out)
    return norm_out, norm_in, norm_mirror, int(max_r + 1), int(max_c + 1)


def _grow_shape(
    rng: random.Random,
    n: int,
    box_h: int,
    box_w: int,
    neighbors: list[tuple[int, int]],
    start: tuple[int, int],
) -> set[tuple[int, int]]:
    cells = {tuple(map(int, start))}
    while len(cells) < int(n):
        frontier: list[tuple[int, int]] = []
        for r, c in list(cells):
            for dr, dc in neighbors:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < int(box_h) and 0 <= cc < int(box_w) and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return cells


def _make_row_template(rng: random.Random, mark: int, src: int, dst: int) -> dict:
    for _ in range(200):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(3, 5)
        n = rng.randint(5, min(8, box_h * box_w))
        shape = _grow_shape(rng, n, box_h, box_w, [(-1, 0), (1, 0), (0, -1), (0, 1)], (0, rng.randrange(box_w)))
        if len(shape) != n or not any(r == 0 for r, _c in shape):
            continue
        min_c = min(c for _r, c in shape)
        max_c = max(c for _r, c in shape)
        width = int(max_c - min_c + 1)
        extra = rng.randint(0, 1)
        line_len = int(max(2, width) + extra)
        offset = rng.randint(0, line_len - width)
        below = rng.choice([True, False])
        output_cells: list[tuple[int, int, int]] = []
        input_cells: list[tuple[int, int, int]] = []
        mirror_cells: list[tuple[int, int, int]] = []
        for c in range(line_len):
            output_cells.append((0, int(c), int(mark)))
            input_cells.append((0, int(c), int(mark)))
        for r, c in shape:
            cc = int(c - min_c + offset)
            if below:
                sr = int(1 + r)
                mr = int(-1 - r)
            else:
                sr = int(-1 - r)
                mr = int(1 + r)
            output_cells.append((sr, cc, int(src)))
            output_cells.append((mr, cc, int(dst)))
            input_cells.append((sr, cc, int(src)))
            mirror_cells.append((mr, cc, int(dst)))
        norm_out, norm_in, norm_mirror, hh, ww = _normalize_group(output_cells, input_cells, mirror_cells)
        return {"output": norm_out, "input": norm_in, "mirror": norm_mirror, "h": hh, "w": ww, "kind": "row", "src": int(src)}
    raise RuntimeError("failed to build row template")


def _make_col_template(rng: random.Random, mark: int, src: int, dst: int) -> dict:
    for _ in range(200):
        box_h = rng.randint(3, 5)
        box_w = rng.randint(2, 4)
        n = rng.randint(5, min(8, box_h * box_w))
        shape = _grow_shape(rng, n, box_h, box_w, [(-1, 0), (1, 0), (0, -1), (0, 1)], (rng.randrange(box_h), 0))
        if len(shape) != n or not any(c == 0 for _r, c in shape):
            continue
        min_r = min(r for r, _c in shape)
        max_r = max(r for r, _c in shape)
        height = int(max_r - min_r + 1)
        extra = rng.randint(0, 1)
        line_len = int(max(2, height) + extra)
        offset = rng.randint(0, line_len - height)
        right = rng.choice([True, False])
        output_cells: list[tuple[int, int, int]] = []
        input_cells: list[tuple[int, int, int]] = []
        mirror_cells: list[tuple[int, int, int]] = []
        for r in range(line_len):
            output_cells.append((int(r), 0, int(mark)))
            input_cells.append((int(r), 0, int(mark)))
        for r, c in shape:
            rr = int(r - min_r + offset)
            if right:
                sc = int(1 + c)
                mc = int(-1 - c)
            else:
                sc = int(-1 - c)
                mc = int(1 + c)
            output_cells.append((rr, sc, int(src)))
            output_cells.append((rr, mc, int(dst)))
            input_cells.append((rr, sc, int(src)))
            mirror_cells.append((rr, mc, int(dst)))
        norm_out, norm_in, norm_mirror, hh, ww = _normalize_group(output_cells, input_cells, mirror_cells)
        return {"output": norm_out, "input": norm_in, "mirror": norm_mirror, "h": hh, "w": ww, "kind": "col", "src": int(src)}
    raise RuntimeError("failed to build col template")


def _make_iso_template(rng: random.Random, mark: int, src: int, dst: int) -> dict:
    diag8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(200):
        box_h = rng.randint(1, 3)
        box_w = rng.randint(1, 3)
        if int(box_h) * int(box_w) < 2:
            continue
        n = rng.randint(2, min(5, box_h * box_w))
        shape = _grow_shape(rng, n, box_h, box_w, diag8, (0, 0))
        if len(shape) != n:
            continue
        sr, sc = rng.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        output_cells = [(0, 0, int(mark))]
        input_cells = [(0, 0, int(mark))]
        mirror_cells: list[tuple[int, int, int]] = []
        for r, c in shape:
            rr = int(sr * (r + 1))
            cc = int(sc * (c + 1))
            output_cells.append((rr, cc, int(src)))
            input_cells.append((rr, cc, int(src)))
            output_cells.append((-rr, -cc, int(dst)))
            mirror_cells.append((-rr, -cc, int(dst)))
        norm_out, norm_in, norm_mirror, hh, ww = _normalize_group(output_cells, input_cells, mirror_cells)
        return {"output": norm_out, "input": norm_in, "mirror": norm_mirror, "h": hh, "w": ww, "kind": "iso", "src": int(src)}
    raise RuntimeError("failed to build iso template")


def _make_template(rng: random.Random, kind: str, mark: int, src: int, dst: int) -> dict:
    if kind == "row":
        return _make_row_template(rng, mark, src, dst)
    if kind == "col":
        return _make_col_template(rng, mark, src, dst)
    return _make_iso_template(rng, mark, src, dst)


def _build_mapping(rng: random.Random, bg: int, mark: int) -> dict[int, int]:
    usable = [c for c in PALETTE if c not in (int(bg), int(mark))]
    pair_count = rng.choice([2, 3])
    if 2 in usable and len(usable) > pair_count and rng.random() < 0.7:
        others = [c for c in usable if c != 2]
        colors = [2] + rng.sample(others, pair_count - 1)
    else:
        colors = rng.sample(usable, pair_count)
    if pair_count == 2:
        a, b = map(int, colors)
        return {int(a): int(b), int(b): int(a)}
    a, b, c = map(int, colors)
    return {int(a): int(b), int(b): int(c), int(c): int(a)}


def _draw_legend(grid: np.ndarray, mapping: dict[int, int], bg: int) -> None:
    grid[:SCENE_TOP, :] = int(bg)
    c = 1
    for src, dst in mapping.items():
        grid[0, c : c + 2] = int(src)
        grid[1, c : c + 2] = int(dst)
        c += 3


def _can_place(occupied: np.ndarray, top: int, left: int, cells: list[tuple[int, int, int]]) -> bool:
    h, w = occupied.shape
    for r, c, _v in cells:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _paint(grid: np.ndarray, top: int, left: int, cells: list[tuple[int, int, int]], occupied: np.ndarray | None = None) -> None:
    for r, c, v in cells:
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(v)
        if occupied is not None:
            occupied[rr, cc] = True


def _build_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(400):
        h = rng.randint(20, 28)
        w = rng.randint(20, 28)
        bg = rng.choice(PALETTE)
        mark_choices = [c for c in PALETTE if c != int(bg) and c != 2]
        mark = int(rng.choice(mark_choices))
        mapping = _build_mapping(rng, bg, mark)

        out = np.full((h, w), int(bg), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        _draw_legend(out, mapping, bg)
        occupied[:SCENE_TOP, :] = True

        kinds = ["row", "col", "iso"]
        rng.shuffle(kinds)
        sources = list(mapping.keys())
        while len(sources) < len(kinds):
            sources.append(rng.choice(list(mapping.keys())))
        rng.shuffle(sources)

        groups: list[dict] = []
        ok = True
        for kind, src in zip(kinds, sources):
            dst = int(mapping[int(src)])
            placed = False
            for _attempt in range(300):
                tmpl = _make_template(rng, kind, mark, int(src), dst)
                if tmpl["h"] > h - SCENE_TOP or tmpl["w"] > w:
                    continue
                top = rng.randint(SCENE_TOP, h - int(tmpl["h"]))
                left = rng.randint(0, w - int(tmpl["w"]))
                if not _can_place(occupied, top, left, tmpl["output"]):
                    continue
                _paint(out, top, left, tmpl["output"], occupied)
                groups.append(
                    {
                        "top": int(top),
                        "left": int(left),
                        "input": tmpl["input"],
                        "mirror": tmpl["mirror"],
                        "kind": kind,
                        "src": int(src),
                    }
                )
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        inp = np.full((h, w), int(bg), dtype=int)
        _draw_legend(inp, mapping, bg)
        for grp in groups:
            _paint(inp, int(grp["top"]), int(grp["left"]), grp["input"])

        counts = Counter(int(v) for v in inp.flatten() if int(v) != int(bg))
        if not counts:
            continue
        least_color, least_count = min(counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
        if int(least_color) != int(mark):
            continue
        if sum(1 for v in counts.values() if int(v) == int(least_count)) != 1:
            continue

        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        meta = {
            "mapping": dict(mapping),
            "mark": int(mark),
            "counts": {int(k): int(v) for k, v in counts.items()},
            "kinds": [grp["kind"] for grp in groups],
            "sources": [int(grp["src"]) for grp in groups],
        }
        return inp, out, meta
    raise RuntimeError("failed to generate fused example")


def generate_input() -> np.ndarray:
    rng = random.Random()
    inp, _out, _meta = _build_example(rng)
    return inp
