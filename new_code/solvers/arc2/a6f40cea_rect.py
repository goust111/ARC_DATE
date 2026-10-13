from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from scipy.ndimage import label
from arc_utils import STRUCT4, get_bg

def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0,c0,r1,c1, bc, fill)
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            # "Complete rectangle border" = exactly the perimeter, and all bbox-border cells are bc.
            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill == int(bc):
                continue
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete occluded rectangle borders and clip to a main frame's interior.

    Input structure:
    - bg = background. A 1px rectangular frame with uniform fill inside.
    - Colored fragments near frame edges — parts of other 1px rectangle borders.

    Rule:
    - Output = frame interior, filled with fill color.
    - Reconstruct each fragment's full rectangle border from its bbox.
    - 2-color borders alternate colors; phase matches visible pixels.
    - Paint only cells falling inside the frame.

    Key insight: Fragments on opposite frame edges belong to one rectangle; the full perimeter must be inferred from partial views.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    if found is None:
        return g
    r0, c0, r1, c1, bc, fill = found

    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    # Collect all components.
    # Note: occlusion by the main frame can split one rectangle into multiple disconnected components;
    # we will merge by shared bbox spans.
    comps = []
    for cid in range(1, int(cn) + 1):
        pts = np.argwhere(clab == int(cid))
        if pts.size == 0:
            continue
        rr0, cc0 = pts.min(axis=0)
        rr1, cc1 = pts.max(axis=0)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        if h < 2 or w < 2:
            continue
        colors = tuple(sorted(set(map(int, g[clab == int(cid)].flatten()))))
        cells = {tuple(map(int, rc)) for rc in pts}
        comps.append(
            {
                "cid": int(cid),
                "rr0": rr0,
                "cc0": cc0,
                "rr1": rr1,
                "cc1": cc1,
                "colors": colors,
                "cells": cells,
            }
        )

    # Build rectangle candidates: each comp, plus merges that share horizontal/vertical span.
    cand: dict[tuple, dict] = {}

    def add_cand(rr0_, cc0_, rr1_, cc1_, colors_, cells_):
        key = (int(rr0_), int(cc0_), int(rr1_), int(cc1_), tuple(colors_))
        if key in cand:
            cand[key]["cells"] |= set(cells_)
        else:
            cand[key] = {
                "rr0": int(rr0_),
                "cc0": int(cc0_),
                "rr1": int(rr1_),
                "cc1": int(cc1_),
                "colors": tuple(colors_),
                "cells": set(cells_),
            }

    for comp in comps:
        add_cand(comp["rr0"], comp["cc0"], comp["rr1"], comp["cc1"], comp["colors"], comp["cells"])

    groups_v: dict[tuple, list[dict]] = defaultdict(list)
    groups_h: dict[tuple, list[dict]] = defaultdict(list)
    for comp in comps:
        groups_v[(comp["colors"], comp["cc0"], comp["cc1"])].append(comp)
        groups_h[(comp["colors"], comp["rr0"], comp["rr1"])].append(comp)

    for (colors, cc0_, cc1_), lst in groups_v.items():
        if len(lst) <= 1:
            continue
        rr0_ = min(x["rr0"] for x in lst)
        rr1_ = max(x["rr1"] for x in lst)
        cells_ = set().union(*[x["cells"] for x in lst])
        add_cand(rr0_, cc0_, rr1_, cc1_, colors, cells_)

    for (colors, rr0_, rr1_), lst in groups_h.items():
        if len(lst) <= 1:
            continue
        cc0_ = min(x["cc0"] for x in lst)
        cc1_ = max(x["cc1"] for x in lst)
        cells_ = set().union(*[x["cells"] for x in lst])
        add_cand(rr0_, cc0_, rr1_, cc1_, colors, cells_)

    cand_rects = list(cand.values())
    cand_rects.sort(key=lambda d: -((d["rr1"] - d["rr0"] + 1) + (d["cc1"] - d["cc0"] + 1)))

    for rect in cand_rects:
        rr0, cc0, rr1, cc1 = rect["rr0"], rect["cc0"], rect["rr1"], rect["cc1"]
        colors = list(rect["colors"])
        cells = rect["cells"]

        # Build rectangle perimeter in a fixed clockwise order starting at top-left.
        perim: list[tuple[int, int]] = []
        for cc in range(cc0, cc1 + 1):
            perim.append((int(rr0), int(cc)))
        for rr in range(rr0 + 1, rr1):
            perim.append((int(rr), int(cc1)))
        if rr1 != rr0:
            for cc in range(cc1, cc0 - 1, -1):
                perim.append((int(rr1), int(cc)))
        if cc1 != cc0:
            for rr in range(rr1 - 1, rr0, -1):
                perim.append((int(rr), int(cc0)))

        # Must intersect the interior; otherwise it can't contribute to output.
        if not any((r_int0 <= r <= (r1 - 1) and c_int0 <= c <= (c1 - 1)) for r, c in perim):
            continue

        idx_of = {rc: i for i, rc in enumerate(perim)}
        on = sum(1 for rc in cells if (int(rc[0]), int(rc[1])) in idx_of)
        if on == 0 or (on / max(1, len(cells))) < 0.95:
            # Not plausibly a (possibly occluded) 1px rectangle border.
            continue

        # Observed constraints on perimeter cells.
        obs: list[tuple[int, int]] = []
        for r, c in cells:
            key = (int(r), int(c))
            if key in idx_of:
                obs.append((int(idx_of[key]), int(g[int(r), int(c)])))

        def color_at(i: int) -> int:
            if len(colors) == 1:
                return int(colors[0])
            if len(colors) == 2:
                a, b = map(int, colors)
                if not obs:
                    start = int(a)
                else:
                    score_a = 0
                    score_b = 0
                    for j, colv in obs:
                        exp_a = a if (j % 2 == 0) else b
                        exp_b = b if (j % 2 == 0) else a
                        score_a += int(exp_a == int(colv))
                        score_b += int(exp_b == int(colv))
                    start = int(a) if score_a >= score_b else int(b)
                return int(start) if (i % 2 == 0) else int(b if start == a else a)
            if obs:
                return int(Counter([c for _i, c in obs]).most_common(1)[0][0])
            return int(colors[0])

        # Paint the completed perimeter, clipped to the main-frame interior.
        for i, (r, c) in enumerate(perim):
            if r_int0 <= int(r) <= int(r1 - 1) and c_int0 <= int(c) <= int(c1 - 1):
                out[int(r - r_int0), int(c - c_int0)] = int(color_at(int(i)))

    return out



