# Variant of 7ed72f31
# Dimension: D10 — 符号编码与图例机制 (symbol_instructions)
# Change: 无符号编码，红色固定为标记色，变换规则固定... → 图例区域定义颜色-变换映射

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, label_mask


"""
Design analysis for the D10 variant
===================================

Reference note
--------------
- The prompt requires reading the source task `9385bd28` first. In this workspace the
  evaluation JSON exists, but a standalone solver file for `9385bd28` is not present.
  The design below therefore uses the visible I/O behavior of that task as the D10
  baseline: a small edge legend encodes a color rule that must be applied elsewhere.

Concrete variant
----------------
- Keep the original 7ed72f31 geometry rule: red (2) still marks a row-axis, column-axis,
  or point-reflection center.
- Add a legend strip on the top edge:
  - rows 0-1 contain 2x2 stacked color chips
  - top chip color = source color A
  - bottom chip color = reflected color B
  - each pair is separated by at least one background column
  - rows 2-3 are blank background separators
- Scene objects live only from row 4 downward.
- The input contains only the original sprites plus red markers, exactly like the base
  task, but now the output paints the reflected copy in color B rather than color A.

Behavior trace on a concrete mini-example
-----------------------------------------
- Suppose the legend shows:
    row 0: 4 4 . 8 8
    row 1: 6 6 . 3 3
  meaning 4->6 and 8->3.
- If a yellow (4) payload sits below a horizontal red line, the base solver would mirror
  yellow onto the opposite side. The D10 solver must still mirror the same geometry, but
  every reflected yellow pixel becomes magenta (6).
- If a teal (8) sprite sits diagonally next to an isolated red pixel, the base solver
  would point-reflect teal; the D10 solver point-reflects the same sprite but paints the
  reflected copy green (3).

Complexity calibration
----------------------
- Use only 2 or 3 legend pairs per grid so the color rule is inferable from few examples.
- Every source color is reused by at least one scene group; otherwise the legend would be
  decorative instead of explanatory.
- All mappings are non-identity (A != B) so the color-rule change is visually observable.
- Use 4-6 reflection groups total, with a mix of row / column / isolated-center cases.

Implicit-assumption review of the original solver
-------------------------------------------------
- Original assumption: payload colors are effectively unique per group.
  - Old code scans the whole grid and mirrors any pixel whose color matches the component's
    payload color and whose coordinate passes a bbox test.
  - That is harmless when same-color reuse is rare, but it breaks here because the legend
    needs repeated source colors across several groups.
  - Fix: reflect only the concrete payload pixels belonging to the current component.
- Original assumption: the entire non-background area belongs to the scene.
  - No longer true because the legend occupies a reserved edge band.
  - Fix: parse the legend first and exclude rows 0-3 from component detection.
- New assumption introduced here: the legend format is always top-edge 2x2 chips with
  background separators, and the scene begins below blank separator rows.
  - The generator guarantees this exact format.

Placement legality
------------------
- Generator places groups output-first on an empty canvas with a one-cell moat around each
  placed footprint, preventing overlap, accidental component merging, and edge clipping.
- The moat also protects isolated-red groups: their adjacent sprite stays the unique
  8-connected component visible to the solver.
- The legend area is reserved before any group placement, so no scene object touches it.

Visual observability and cleanliness
------------------------------------
- The legend is small, regular, and visually separated from the scene by two blank rows.
- The same source color appears in multiple groups, and every reflected copy consistently
  changes into the paired legend color. That makes the D10 rule recoverable from 2-4
  examples without cluttering the grid.
- No extra distractor sprites are added: the visual burden stays on "read legend, then
  apply the usual reflection geometry with recoloring".

Color-change inferability
-------------------------
- At most three mappings appear in one grid.
- Target colors are distinct from source colors and from red/background.
- Each mapping is exercised by at least one actual reflection group, so every legend pair
  has direct evidence in the scene.
"""


MARK = 2
LEGEND_ROWS = 2
SCENE_TOP = 4
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]


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


def _parse_legend(g: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    if g.shape[0] < 2:
        return mapping, legend_cells
    c = 0
    while c + 1 < g.shape[1]:
        top0 = int(g[0, c])
        top1 = int(g[0, c + 1])
        bot0 = int(g[1, c])
        bot1 = int(g[1, c + 1])
        if top0 != int(bg) and top0 == top1 and bot0 != int(bg) and bot0 == bot1:
            mapping[int(top0)] = int(bot0)
            legend_cells.update({(0, c), (0, c + 1), (1, c), (1, c + 1)})
            c += 2
            while c < g.shape[1] and int(g[0, c]) == int(bg) and int(g[1, c]) == int(bg):
                c += 1
            continue
        c += 1
    return mapping, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    mapping, _legend_cells = _parse_legend(g, bg)
    out = g.copy()

    scene = g.copy()
    scene[: min(SCENE_TOP, h), :] = int(bg)

    lab, n = label_mask(scene != int(bg), connectivity=4)
    marker_pts: list[tuple[int, int]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        vals = [int(scene[r, c]) for r, c in coords]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue
        if len(coords) == 1 and len(reds) == 1:
            marker_pts.append(reds[0])
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", reds[0])
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        for (r, c), v in zip(coords, vals):
            if int(v) == int(MARK):
                continue
            if axis[0] == "row":
                rr, cc = _refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = _refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = _refl_point(axis[1], int(r), int(c))
            if _inb((h, w), rr, cc):
                out[int(rr), int(cc)] = int(mapping.get(int(v), int(v)))

    payload_mask = (scene != int(bg)) & (scene != int(MARK))
    payload_lab, _payload_n = label_mask(payload_mask, connectivity=8)
    for pr, pc in marker_pts:
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


def _normalize_from_origin(
    cells: list[tuple[int, int, int]],
    min_r: int,
    min_c: int,
) -> list[tuple[int, int, int]]:
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


def _make_row_template(rng: random.Random, src: int, dst: int) -> dict:
    for _ in range(200):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(2, 4)
        n = rng.randint(3, min(8, box_h * box_w))
        shape = _grow_shape(rng, n, box_h, box_w, [(-1, 0), (1, 0), (0, -1), (0, 1)], (0, rng.randrange(box_w)))
        if len(shape) != n or not any(r == 0 for r, _c in shape):
            continue
        max_c = max(c for _r, c in shape)
        min_c = min(c for _r, c in shape)
        width = int(max_c - min_c + 1)
        extra = rng.randint(0, 2)
        line_len = int(width + extra)
        offset = rng.randint(0, extra)
        below = rng.choice([True, False])
        output_cells: list[tuple[int, int, int]] = []
        input_cells: list[tuple[int, int, int]] = []
        mirror_cells: list[tuple[int, int, int]] = []
        for c in range(line_len):
            output_cells.append((0, int(c), int(MARK)))
            input_cells.append((0, int(c), int(MARK)))
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
        return {"output": norm_out, "input": norm_in, "mirror": norm_mirror, "h": hh, "w": ww, "kind": "row", "src": src, "dst": dst}
    raise RuntimeError("failed to build row template")


def _make_col_template(rng: random.Random, src: int, dst: int) -> dict:
    for _ in range(200):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(2, 4)
        n = rng.randint(3, min(8, box_h * box_w))
        shape = _grow_shape(rng, n, box_h, box_w, [(-1, 0), (1, 0), (0, -1), (0, 1)], (rng.randrange(box_h), 0))
        if len(shape) != n or not any(c == 0 for _r, c in shape):
            continue
        max_r = max(r for r, _c in shape)
        min_r = min(r for r, _c in shape)
        height = int(max_r - min_r + 1)
        extra = rng.randint(0, 2)
        line_len = int(height + extra)
        offset = rng.randint(0, extra)
        right = rng.choice([True, False])
        output_cells: list[tuple[int, int, int]] = []
        input_cells: list[tuple[int, int, int]] = []
        mirror_cells: list[tuple[int, int, int]] = []
        for r in range(line_len):
            output_cells.append((int(r), 0, int(MARK)))
            input_cells.append((int(r), 0, int(MARK)))
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
        return {"output": norm_out, "input": norm_in, "mirror": norm_mirror, "h": hh, "w": ww, "kind": "col", "src": src, "dst": dst}
    raise RuntimeError("failed to build col template")


def _make_iso_template(rng: random.Random, src: int, dst: int) -> dict:
    diag8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(200):
        box_h = rng.randint(1, 3)
        box_w = rng.randint(1, 3)
        n = rng.randint(1, min(5, box_h * box_w))
        shape = _grow_shape(rng, n, box_h, box_w, diag8, (0, 0))
        if len(shape) != n:
            continue
        sr, sc = rng.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        output_cells = [(0, 0, int(MARK))]
        input_cells = [(0, 0, int(MARK))]
        mirror_cells: list[tuple[int, int, int]] = []
        for r, c in shape:
            rr = int(sr * (r + 1))
            cc = int(sc * (c + 1))
            output_cells.append((rr, cc, int(src)))
            input_cells.append((rr, cc, int(src)))
            output_cells.append((-rr, -cc, int(dst)))
            mirror_cells.append((-rr, -cc, int(dst)))
        norm_out, norm_in, norm_mirror, hh, ww = _normalize_group(output_cells, input_cells, mirror_cells)
        return {"output": norm_out, "input": norm_in, "mirror": norm_mirror, "h": hh, "w": ww, "kind": "iso", "src": src, "dst": dst}
    raise RuntimeError("failed to build isolated template")


def _make_template(rng: random.Random, kind: str, src: int, dst: int) -> dict:
    if kind == "row":
        return _make_row_template(rng, src, dst)
    if kind == "col":
        return _make_col_template(rng, src, dst)
    return _make_iso_template(rng, src, dst)


def _build_mapping(rng: random.Random, bg: int) -> dict[int, int]:
    usable = [c for c in PALETTE if c not in (int(bg), int(MARK))]
    pair_count = rng.randint(2, 3)
    sources = rng.sample(usable, pair_count)
    targets_pool = [c for c in usable if c not in sources]
    targets = rng.sample(targets_pool, pair_count)
    return {int(a): int(b) for a, b in zip(sources, targets)}


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


def _paint(grid: np.ndarray, top: int, left: int, cells: list[tuple[int, int, int]], occupied: np.ndarray) -> None:
    for r, c, v in cells:
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(v)
        occupied[rr, cc] = True


def _build_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(200):
        h = rng.randint(20, 28)
        w = rng.randint(20, 28)
        bg = rng.choice([c for c in range(10) if c != int(MARK)])
        mapping = _build_mapping(rng, bg)
        out = np.full((h, w), int(bg), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        _draw_legend(out, mapping, bg)
        occupied[:SCENE_TOP, :] = True

        sources = list(mapping.keys())
        group_count = rng.randint(max(4, len(sources) + 1), min(6, len(sources) + 3))
        source_seq = sources[:]
        while len(source_seq) < group_count:
            source_seq.append(rng.choice(sources))
        rng.shuffle(source_seq)

        kind_seq = ["row", "col", "iso"]
        while len(kind_seq) < group_count:
            kind_seq.append(rng.choice(["row", "col", "iso"]))
        rng.shuffle(kind_seq)

        groups: list[dict] = []
        ok = True
        for src, kind in zip(source_seq, kind_seq):
            dst = int(mapping[int(src)])
            placed = False
            for _attempt in range(300):
                tmpl = _make_template(rng, kind, int(src), int(dst))
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
                        "dst": int(dst),
                    }
                )
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        inp = out.copy()
        for grp in groups:
            for r, c, _v in grp["mirror"]:
                inp[int(grp["top"] + r), int(grp["left"] + c)] = int(bg)

        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        meta = {
            "mapping": dict(mapping),
            "group_count": len(groups),
            "kinds": [grp["kind"] for grp in groups],
            "sources_used": [grp["src"] for grp in groups],
            "changed_pixels": int(np.sum((solved != inp) & (inp != MARK))),
        }
        return inp, out, meta
    raise RuntimeError("failed to generate a valid D10 instance")


def generate_input() -> np.ndarray:
    rng = random.Random()
    inp, _out, _meta = _build_example(rng)
    return inp
