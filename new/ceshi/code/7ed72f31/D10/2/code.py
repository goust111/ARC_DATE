# Variant of 7ed72f31
# Dimension: D10 — 符号编码与图例机制 (symbol_instructions)
# Change: 无符号编码，红色固定为标记色，变换规则固定... → 背景色区域编码反射区域

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg, label_mask, paste


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior
#    The original task lets every red-marked sprite in the whole grid participate.
#    This variant adds a gating symbol layer: a filled non-main-background rectangle
#    means "this is an active reflection zone". Only sprites whose red marker and
#    payload live inside such a zone are reflected. Red-marked sprites on the main
#    background are distractors and remain unchanged.
#
#    Small trace:
#    - Original: a red vertical line with a green payload on open background mirrors
#      to the opposite side immediately.
#    - Variant: the same red-line sprite mirrors only if it sits inside a filled
#      colored panel. If the same motif appears outside all panels, nothing happens.
#    - Difference: the solver must first detect which rectangular regions are active,
#      then run the old reflection logic only inside those regions.
#
# b) Complexity calibration
#    The control variable is binary and visually stable: "inside panel" vs
#    "outside panel". To make this visible, each sample contains 3-4 active panels
#    and at least one outside distractor that also contains red. Panel colors vary,
#    so the rule is tied to the rectangle structure, not to one special color.
#
# c) Old implicit assumptions that break
#    - `label_mask(g != bg)` over the whole grid used to find mixed red+payload
#      components. With filled panels, that now merges the entire panel area into one
#      component, so the old component logic cannot be applied globally anymore.
#    - The old solver scanned the whole grid for eligible payload-colored pixels.
#      That relied on the absence of a region-gating mechanism. In this variant, the
#      scan must be restricted to one active panel at a time.
#    - The old isolated-red logic assumed global background = task background. Here
#      the local background inside a panel is the panel fill color, not the main bg.
#
# d) Placement legality
#    Panels are placed with at least one-cell gaps, never overlap, and are large
#    enough to contain the original sprite, its reflected copy, and margin. Outside
#    distractors are placed away from panels and from each other. The generator uses
#    output-first construction and verifies `main(input) == output`.
#
# e) Visual observability
#    A human sees that some red-marked motifs duplicate while similar motifs outside
#    colored rectangles do not. Because distractors also contain red markers, the
#    active-rectangle rule is necessary to explain the examples.
#
# f) Visual cleanliness
#    One active sprite per panel keeps each region readable. Panel fill colors occupy
#    most of the rectangle, so the region itself is easy to spot. Distractors stay on
#    the plain background, which separates the two symbol roles cleanly.
#
# g) Color inference
#    Panel colors are not fixed; multiple non-red colors can serve as active-region
#    fills. This prevents accidental overfitting to one color while keeping the rule
#    simple: rectangle structure, not color identity, is the signal.
#
# h) Generator guarantees for new solver assumptions
#    - Every active region is a solid axis-aligned rectangle in the non-main-bg mask.
#    - Each panel's majority color is its fill color and differs from red.
#    - Each panel contains exactly one reflective group.
#    - At least one outside red-marked distractor exists in every sample.
#
# i) Quantitative non-degeneracy target
#    The key metric is the count of red-marked distractor groups outside panels.
#    It is enforced to be >= 1. On such inputs, the original solver and the variant
#    solver should produce different outputs, because the original would act on those
#    outside markers while the variant ignores them.


MARK = 2
COLORS = list(range(10))


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _bbox_of_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _find_active_regions(grid: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(grid != int(bg), connectivity=4)
    panels: list[dict] = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(coords.shape[0])
        if h < 6 or w < 6 or area < 24:
            continue
        if area != h * w:
            continue
        sub = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        vals, counts = np.unique(sub, return_counts=True)
        fill = int(vals[int(np.argmax(counts))])
        if fill == int(bg) or fill == int(MARK):
            continue
        if int(counts.max()) < (h * w) // 2:
            continue
        if not np.any(sub == int(MARK)):
            continue
        if not np.any((sub != int(fill)) & (sub != int(MARK))):
            continue
        panels.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "fill": int(fill),
            }
        )
    panels.sort(key=lambda p: p["bbox"])
    return panels


def _solve_panel(sub: np.ndarray, fill: int) -> np.ndarray:
    g = np.array(sub, dtype=int)
    out = g.copy()
    h, w = g.shape

    lab, n = label_mask(g != int(fill), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals, "bbox": _bbox_of_coords(coords)})

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(vals):
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
            rr1 = min(int(h - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(w - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(h)):
            for c in range(int(w)):
                v = int(g[int(r), int(c)])
                if v == int(fill) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = _refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = _refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = _refl_point(axis[1], int(r), int(c))
                if _inb(h, w, rr, cc):
                    out[int(rr), int(cc)] = int(v)

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(h, w, rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(fill) or v == int(MARK):
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
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    if _inb(h, w, rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    for panel in _find_active_regions(g, bg):
        r0, c0, r1, c1 = map(int, panel["bbox"])
        sub = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        solved = _solve_panel(sub, int(panel["fill"]))
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = solved
    return out


def _neighbors(cell: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            out.append((int(r + dr), int(c + dc)))
    return out


def _grow_connected(
    rng: random.Random,
    h: int,
    w: int,
    target: int,
    connectivity: int,
    seed: tuple[int, int],
) -> list[tuple[int, int]]:
    cells = {tuple(map(int, seed))}
    while len(cells) < int(target):
        frontier = set()
        for cell in list(cells):
            for nr, nc in _neighbors(cell, connectivity):
                if 0 <= int(nr) < int(h) and 0 <= int(nc) < int(w) and (int(nr), int(nc)) not in cells:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            break
        cells.add(rng.choice(sorted(frontier)))
    return sorted(cells)


def _paint_cells(patch: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        patch[int(r), int(c)] = int(color)


def _build_row_panel(rng: random.Random, h: int, w: int, fill: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        max_ph = min(4, (int(h) - 3) // 2)
        if max_ph < 2 or int(w) < 6:
            break
        ph = rng.randint(2, max_ph)
        line_len = rng.randint(3, min(6, int(w) - 2))
        pw = rng.randint(2, min(4, line_len))
        axis_r = rng.randint(int(ph) + 1, int(h) - int(ph) - 2)
        line_c0 = rng.randint(1, int(w) - int(line_len) - 1)
        box_c0 = rng.randint(int(line_c0), int(line_c0 + line_len - pw))
        side = rng.choice(["top", "bottom"])
        box_r0 = int(axis_r - ph) if side == "top" else int(axis_r + 1)
        seed = (ph - 1, rng.randint(0, pw - 1)) if side == "top" else (0, rng.randint(0, pw - 1))
        target = rng.randint(3, min(7, ph * pw))
        shape = _grow_connected(rng, ph, pw, target, 4, seed)
        payload = [(int(box_r0 + r), int(box_c0 + c)) for r, c in shape]
        mirrored = [_refl_row(axis_r, r, c) for r, c in payload]
        if not all(_inb(h, w, r, c) for r, c in mirrored):
            continue
        line = [(int(axis_r), int(c)) for c in range(int(line_c0), int(line_c0 + line_len))]
        patch_in = np.full((int(h), int(w)), int(fill), dtype=int)
        patch_out = patch_in.copy()
        _paint_cells(patch_in, line, MARK)
        _paint_cells(patch_out, line, MARK)
        _paint_cells(patch_in, payload, color)
        _paint_cells(patch_out, payload, color)
        _paint_cells(patch_out, mirrored, color)
        return patch_in, patch_out
    raise RuntimeError("failed to build row panel")


def _build_col_panel(rng: random.Random, h: int, w: int, fill: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        max_pw = min(4, (int(w) - 3) // 2)
        if max_pw < 2 or int(h) < 6:
            break
        pw = rng.randint(2, max_pw)
        line_len = rng.randint(3, min(6, int(h) - 2))
        ph = rng.randint(2, min(4, line_len))
        axis_c = rng.randint(int(pw) + 1, int(w) - int(pw) - 2)
        line_r0 = rng.randint(1, int(h) - int(line_len) - 1)
        box_r0 = rng.randint(int(line_r0), int(line_r0 + line_len - ph))
        side = rng.choice(["left", "right"])
        box_c0 = int(axis_c - pw) if side == "left" else int(axis_c + 1)
        seed = (rng.randint(0, ph - 1), pw - 1) if side == "left" else (rng.randint(0, ph - 1), 0)
        target = rng.randint(3, min(7, ph * pw))
        shape = _grow_connected(rng, ph, pw, target, 4, seed)
        payload = [(int(box_r0 + r), int(box_c0 + c)) for r, c in shape]
        mirrored = [_refl_col(axis_c, r, c) for r, c in payload]
        if not all(_inb(h, w, r, c) for r, c in mirrored):
            continue
        line = [(int(r), int(axis_c)) for r in range(int(line_r0), int(line_r0 + line_len))]
        patch_in = np.full((int(h), int(w)), int(fill), dtype=int)
        patch_out = patch_in.copy()
        _paint_cells(patch_in, line, MARK)
        _paint_cells(patch_out, line, MARK)
        _paint_cells(patch_in, payload, color)
        _paint_cells(patch_out, payload, color)
        _paint_cells(patch_out, mirrored, color)
        return patch_in, patch_out
    raise RuntimeError("failed to build col panel")


def _build_point_panel(rng: random.Random, h: int, w: int, fill: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        max_sh = min(3, (int(h) - 3) // 2)
        max_sw = min(3, (int(w) - 3) // 2)
        if max_sh < 1 or max_sw < 1:
            break
        sh = rng.randint(1, max_sh)
        sw = rng.randint(1, max_sw)
        pr = rng.randint(int(sh) + 1, int(h) - int(sh) - 2)
        pc = rng.randint(int(sw) + 1, int(w) - int(sw) - 2)
        quadrant = rng.choice(["tl", "tr", "bl", "br"])
        if quadrant == "tl":
            box_r0, box_c0 = int(pr - sh), int(pc - sw)
            seed = (sh - 1, sw - 1)
        elif quadrant == "tr":
            box_r0, box_c0 = int(pr - sh), int(pc + 1)
            seed = (sh - 1, 0)
        elif quadrant == "bl":
            box_r0, box_c0 = int(pr + 1), int(pc - sw)
            seed = (0, sw - 1)
        else:
            box_r0, box_c0 = int(pr + 1), int(pc + 1)
            seed = (0, 0)
        target = rng.randint(1, min(6, sh * sw))
        shape = _grow_connected(rng, sh, sw, target, 8, seed)
        payload = [(int(box_r0 + r), int(box_c0 + c)) for r, c in shape]
        mirrored = [_refl_point((pr, pc), r, c) for r, c in payload]
        if not all(_inb(h, w, r, c) for r, c in mirrored):
            continue
        if set(payload) & set(mirrored):
            continue
        patch_in = np.full((int(h), int(w)), int(fill), dtype=int)
        patch_out = patch_in.copy()
        patch_in[int(pr), int(pc)] = int(MARK)
        patch_out[int(pr), int(pc)] = int(MARK)
        _paint_cells(patch_in, payload, color)
        _paint_cells(patch_out, payload, color)
        _paint_cells(patch_out, mirrored, color)
        return patch_in, patch_out
    raise RuntimeError("failed to build point panel")


def _build_group(
    rng: random.Random,
    kind: str,
    h: int,
    w: int,
    fill: int,
    color: int,
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "row":
        return _build_row_panel(rng, h, w, fill, color)
    if kind == "col":
        return _build_col_panel(rng, h, w, fill, color)
    if kind == "point":
        return _build_point_panel(rng, h, w, fill, color)
    raise ValueError(f"unknown kind: {kind}")


def _nonbg_bbox(patch: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(patch != int(bg))
    if coords.size == 0:
        raise ValueError("empty patch")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    hh, ww = reserved.shape
    rr0 = max(0, int(r0) - int(margin))
    cc0 = max(0, int(c0) - int(margin))
    rr1 = min(int(hh), int(r0 + h + margin))
    cc1 = min(int(ww), int(c0 + w + margin))
    return not reserved[int(rr0) : int(rr1), int(cc0) : int(cc1)].any()


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> None:
    hh, ww = reserved.shape
    rr0 = max(0, int(r0) - int(margin))
    cc0 = max(0, int(c0) - int(margin))
    rr1 = min(int(hh), int(r0 + h + margin))
    cc1 = min(int(ww), int(c0 + w + margin))
    reserved[int(rr0) : int(rr1), int(cc0) : int(cc1)] = True


def _place_panel(
    rng: random.Random,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    reserved: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
) -> tuple[int, int]:
    h, w = map(int, patch_in.shape)
    hh, ww = canvas_in.shape
    for _ in range(300):
        r0 = rng.randint(0, int(hh - h))
        c0 = rng.randint(0, int(ww - w))
        if not _can_place(reserved, r0, c0, h, w, margin=1):
            continue
        paste(canvas_in, patch_in, r0, c0)
        paste(canvas_out, patch_out, r0, c0)
        _mark_reserved(reserved, r0, c0, h, w, margin=1)
        return int(r0), int(c0)
    raise RuntimeError("failed to place panel")


def _place_distractor(
    rng: random.Random,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> tuple[int, int]:
    ph, pw = map(int, patch.shape)
    hh, ww = canvas_in.shape
    b0, b1, b2, b3 = _nonbg_bbox(patch, bg)
    bh = int(b2 - b0 + 1)
    bw = int(b3 - b1 + 1)
    for _ in range(300):
        r0 = rng.randint(0, int(hh - ph))
        c0 = rng.randint(0, int(ww - pw))
        br0 = int(r0 + b0)
        bc0 = int(c0 + b1)
        if not _can_place(reserved, br0, bc0, bh, bw, margin=1):
            continue
        paste(canvas_in, patch, r0, c0, transparent=bg)
        paste(canvas_out, patch, r0, c0, transparent=bg)
        _mark_reserved(reserved, br0, bc0, bh, bw, margin=1)
        return int(r0), int(c0)
    raise RuntimeError("failed to place distractor")


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = random.Random() if rng is None else rng
    for _ in range(400):
        h = rng.randint(24, 28)
        w = rng.randint(24, 28)
        bg = rng.choice([c for c in COLORS if c != int(MARK)])
        grid_in = np.full((int(h), int(w)), int(bg), dtype=int)
        grid_out = grid_in.copy()
        reserved = np.zeros((int(h), int(w)), dtype=bool)

        usable = [c for c in COLORS if c not in (int(bg), int(MARK))]
        rng.shuffle(usable)

        panel_count = rng.randint(3, 4)
        payload_colors = usable[:panel_count]
        remaining = usable[panel_count:]
        if len(remaining) < 2:
            continue
        panel_palette = remaining[: max(2, min(3, len(remaining)))]
        kinds = ["row", "col", "point"]
        active_kinds = kinds[:]
        while len(active_kinds) < panel_count:
            active_kinds.append(rng.choice(kinds))
        rng.shuffle(active_kinds)

        try:
            panel_records = []
            total_panel_area = 0
            for idx in range(panel_count):
                kind = active_kinds[idx]
                fill = rng.choice(panel_palette)
                if kind == "point":
                    ph = rng.randint(6, 8)
                    pw = rng.randint(6, 8)
                else:
                    ph = rng.randint(7, 9)
                    pw = rng.randint(7, 9)
                patch_in, patch_out = _build_group(rng, kind, ph, pw, fill, payload_colors[idx])
                total_panel_area += int(ph * pw)
                r0, c0 = _place_panel(rng, grid_in, grid_out, reserved, patch_in, patch_out)
                panel_records.append({"kind": kind, "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1), "fill": fill})
            if total_panel_area >= (h * w) // 2:
                continue

            distractor_count = rng.randint(1, 3)
            distractor_records = []
            for idx in range(distractor_count):
                kind = rng.choice(kinds)
                color = rng.choice(payload_colors)
                if kind == "point":
                    ph = rng.randint(5, 7)
                    pw = rng.randint(5, 7)
                else:
                    ph = rng.randint(6, 8)
                    pw = rng.randint(6, 8)
                patch_in, _ = _build_group(rng, kind, ph, pw, bg, color)
                r0, c0 = _place_distractor(rng, grid_in, grid_out, reserved, patch_in, bg)
                distractor_records.append({"kind": kind, "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1), "color": color})
        except RuntimeError:
            continue

        if int(get_bg(grid_in)) != int(bg):
            continue
        if len(_find_active_regions(grid_in, bg)) != panel_count:
            continue
        outside_red = int(np.sum((grid_in == int(MARK)) & (grid_out == int(MARK))))
        if outside_red < panel_count + 1:
            continue
        solved = main(grid_in)
        if not np.array_equal(solved, grid_out):
            continue

        meta = {
            "bg": int(bg),
            "panel_count": int(panel_count),
            "distractor_count": int(distractor_count),
            "panels": panel_records,
            "distractors": distractor_records,
            "active_regions_found": len(_find_active_regions(grid_in, bg)),
            "red_count": int(np.sum(grid_in == int(MARK))),
        }
        return grid_in, grid_out, meta
    raise RuntimeError("failed to generate a valid D10 variant case")


def generate_input() -> np.ndarray:
    return generate_case()[0]


def variant_metrics(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    panels = _find_active_regions(g, bg)
    panel_mask = np.zeros_like(g, dtype=bool)
    for p in panels:
        r0, c0, r1, c1 = map(int, p["bbox"])
        panel_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    return {
        "bg": int(bg),
        "active_region_count": int(len(panels)),
        "outside_red_pixels": int(np.sum((g == int(MARK)) & (~panel_mask))),
        "panel_fill_histogram": dict(Counter(int(p["fill"]) for p in panels)),
    }
