# Combo variant of 271d71e2 (index 23)
# combo_spec: D3/1+D5/3+D10/0
# Name: 条纹内部螺旋渐变着色

"""
Design analysis for the fused variant
=====================================

a) Concrete behavior and trace
------------------------------
- The base geometry is unchanged: magenta background, black rectangular containers,
  one maroon head line on a border side, and one aligned maroon destination line farther
  along the same axis.
- The interior rule is now a true fusion of three ideas:
  1. D10 structure: the interior is split into 1-cell stripes orthogonal to motion.
     From the head side inward, stripes alternate GREY / ORANGE / GREY / ORANGE / ...
     The head-side stripe is always GREY, so the set of "chargeable" cells is visible.
  2. D5 order: chargeable cells are not processed by head-facing row/column order.
     Instead, we take the full clockwise spiral order of the whole interior rectangle and
     keep only the cells that belong to GREY stripes. That filtered spiral is the true
     processing order.
  3. D3 color cycle: processed GREY-stripe cells do not all become orange. The k-th
     processed chargeable cell is colored by a fixed cycle 2 -> 4 -> 7 -> 2 -> ...
- Step count stays anchored to the base task:
    steps = min(head_to_dest_distance, remaining_chargeable_grey_cells)
  This keeps D5's unit of progress as individual pixels rather than entire stripes.

  Small trace with head on top and a 4x4 interior:
  - Stripe scaffold by row: rows 0 and 2 are GREY stripes, rows 1 and 3 are ORANGE stripes.
  - Full spiral order begins:
      (0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(3,2),(3,1),(3,0),(2,0),(1,0),...
  - Filter to GREY-stripe cells only:
      (0,0),(0,1),(0,2),(0,3),(2,3),(2,0),(2,1),(2,2)
  - If the input has 3 processed cells, the GREY stripes look:
      row 0 = [2,4,7,5]
      row 2 = [5,5,5,5]
    while rows 1 and 3 stay solid ORANGE.
  - If steps = 2, output moves the box by 2 and the next two filtered-spiral cells
    become 2 and 4 respectively, giving:
      row 0 = [2,4,7,2]
      row 2 = [5,5,5,4]   (positions shown in local spiral order, not row-major order)
  - Compared with the original task:
    * the interior is striped rather than split into one orange block plus one grey block
    * newly charged cells follow spiral order rather than head-side linear order
    * their colors cycle rather than all turning orange

b) Complexity calibration
-------------------------
- A 3-color cycle is enough to be inferable from 2-4 ARC examples and still visibly distinct.
- Stripe orientation has only four possibilities because it is tied to the head direction.
- The spiral itself is global and fixed, so the freedom remains small: every example reuses the
  same "clockwise from top-left" traversal.
- To avoid weak or ambiguous samples, the generator enforces:
  * movers with inner dimensions at least 3x3
  * at least one mover with inner dimensions at least 4x4
  * at least one mover with >= 3 movement steps and >= 3 final processed cells
  * at least one mover where filtered-spiral processing differs from a head-side linear
    processing baseline on the same striped input

c) Implicit-assumption audit of the original solver
---------------------------------------------------
- Original container detection assumed interior colors were only {5,7}. That breaks because
  processed GREY-stripe cells may also be 2 or 4. The new detector accepts {5,2,4,7}.
- Original head-color detection excluded only {bg,border,grey,orange}. That would misread red
  or yellow processed cells as head markers, so the exclusion set must include the whole interior
  palette.
- Original step cap used the number of GREY cells. That assumption still holds, but now "GREY"
  only lives on visible GREY stripes. The generator guarantees that every unprocessed chargeable
  cell is literally color 5.
- Original recoloring logic mutated selected grey cells in place. That is no longer sufficient:
  the final color of each processed cell depends on its prefix index in the filtered spiral, and
  structural ORANGE stripes must remain untouched. The solver therefore rebuilds the entire inner
  rectangle from `(head_side, charged_count)` rather than editing a few cells blindly.
- New assumption introduced by the fusion:
  processed chargeable cells form a prefix of the filtered spiral order. The generator enforces
  this directly, so counting non-grey cells on GREY stripes recovers the exact processed prefix.

d) Placement legality
---------------------
- The base assembly layout is preserved: destination + gap + head + container along the motion
  axis, with a 1-cell background moat between assemblies.
- Since `steps <= gap + 1`, the moved head lands on or before the destination line, so every
  output pose stays within the original assembly strip.
- The generator retries placement until no assembly overlaps another even after accounting for
  the extra 1-cell moat. This preserves clean separation and prevents border/head ambiguity.

e) Visual observability
-----------------------
- Humans can see the three fused cues directly:
  * alternating grey/orange stripe scaffold
  * processed cells appearing only on the grey stripes
  * those processed cells tracing a clockwise perimeter-then-inward progression
  * colors on that progression repeating 2,4,7
- The generator rejects samples where every mover is too small or too weak to reveal these cues.
  In particular, it insists on at least one mover whose combo result differs from a simpler
  head-side striped fill, so the spiral component is not visually lost.

f) Visual cleanliness
---------------------
- No extra markers are introduced beyond the base task's border/head/destination scheme.
- Stripe colors remain the familiar GREY/ORANGE pair, and only processed GREY-stripe cells add
  RED/YELLOW accents. Because these accents live inside the black border, they do not interfere
  with object parsing.
- Objects are kept separated by background moats, so multiple colorful interiors do not merge.

g) Change inferability
----------------------
- The color changes are highly structured:
  * location is constrained to GREY stripes
  * eligible cells are ordered by a visible spiral geometry
  * color is determined by prefix index in a short 3-cycle
- This is much more inferable than an arbitrary recolor map because the same sequence repeats
  across all examples.

Post-implementation solver assumptions checked against the generator
====================================================================
- `main()` assumes stripe orientation is derivable from head direction alone.
  `generate_input()` constructs stripes exactly that way.
- `main()` assumes every processed cell lies on a GREY stripe and every ORANGE stripe cell stays
  ORANGE forever. `generate_input()` never writes cycle colors onto ORANGE stripes.
- `main()` assumes the number of processed cells is recoverable as the count of non-GREY cells on
  GREY stripes. `generate_input()` always builds a prefix of the filtered spiral, so this count
  is exact.
- `main()` assumes there is a unique aligned destination candidate of matching shape in front of
  the head. The assembly layout preserves that original-task guarantee.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

CHARGE_CYCLE = (RED, YELLOW, ORANGE)
INTERIOR_COLORS = {GREY, ORANGE, *CHARGE_CYCLE}

# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _spiral_order(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    """Clockwise spiral of all cells, starting at the top-left corner."""
    order: list[tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = inner_h - 1, inner_w - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            order.append((top, c))
        for r in range(top + 1, bottom + 1):
            order.append((r, right))
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                order.append((bottom, c))
        if left < right:
            for r in range(bottom - 1, top, -1):
                order.append((r, left))
        top += 1
        bottom -= 1
        left += 1
        right -= 1
    return order


def _stripe_layers(
    inner_h: int,
    inner_w: int,
    head_side: int | None = None,
    direction: tuple[int, int] | None = None,
) -> list[list[tuple[int, int]]]:
    """Return full-width/full-height stripes from head side inward, using local coords."""
    if direction is None:
        assert head_side is not None
        direction = SIDE_TO_DIR[int(head_side)]
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return [[(r, c) for c in range(inner_w)] for r in range(inner_h)]
    if (dr, dc) == (1, 0):
        return [[(r, c) for c in range(inner_w)] for r in range(inner_h - 1, -1, -1)]
    if (dr, dc) == (0, -1):
        return [[(r, c) for r in range(inner_h)] for c in range(inner_w)]
    if (dr, dc) == (0, 1):
        return [[(r, c) for r in range(inner_h)] for c in range(inner_w - 1, -1, -1)]
    raise ValueError(f"Unsupported direction: {direction}")


def _grey_stripe_cells(
    inner_h: int,
    inner_w: int,
    head_side: int | None = None,
    direction: tuple[int, int] | None = None,
) -> set[tuple[int, int]]:
    """Cells that belong to the visible GREY stripes."""
    layers = _stripe_layers(inner_h, inner_w, head_side=head_side, direction=direction)
    cells: set[tuple[int, int]] = set()
    for layer in layers[::2]:
        cells.update(layer)
    return cells


def _combo_order(
    inner_h: int,
    inner_w: int,
    head_side: int | None = None,
    direction: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Clockwise spiral order, filtered to cells belonging to GREY stripes only."""
    grey_cells = _grey_stripe_cells(inner_h, inner_w, head_side=head_side, direction=direction)
    return [cell for cell in _spiral_order(inner_h, inner_w) if cell in grey_cells]


def _linear_grey_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """
    Baseline used only for generator anti-degeneration checks:
    original head-side linear order filtered to GREY-stripe cells.
    """
    cells = list(_grey_stripe_cells(inner_h, inner_w, head_side=head_side))
    if head_side == 3:
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _build_combo_interior(inner_h: int, inner_w: int, head_side: int, charged_count: int) -> np.ndarray:
    """Build striped interior with the first charged_count GREY-stripe cells painted by the cycle."""
    interior = np.full((inner_h, inner_w), ORANGE, dtype=int)
    grey_cells = _grey_stripe_cells(inner_h, inner_w, head_side=head_side)
    for r, c in grey_cells:
        interior[r, c] = GREY
    order = _combo_order(inner_h, inner_w, head_side=head_side)
    for idx, (r, c) in enumerate(order[:charged_count]):
        interior[r, c] = CHARGE_CYCLE[idx % len(CHARGE_CYCLE)]
    return interior


def _processed_count(inner: np.ndarray, direction: tuple[int, int]) -> int:
    """Count processed cells on GREY stripes; generator guarantees they form a prefix of combo order."""
    grey_cells = _grey_stripe_cells(inner.shape[0], inner.shape[1], direction=direction)
    return sum(1 for r, c in grey_cells if int(inner[r, c]) != GREY)


def _choose_inner_dims(head_side: int, ensure_large: bool = False) -> tuple[int, int]:
    """
    Pick dimensions that keep stripes and spiral readable.
    The base spec allowed up to 12x5; we stay within that envelope but bias toward medium sizes.
    """
    if head_side in (0, 1):
        h_lo = 4 if ensure_large else 3
        return random.randint(h_lo, 8), random.randint(3, 5)
    w_lo = 4 if ensure_large else 3
    return random.randint(3, 8), random.randint(w_lo, 5)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Fused variant rule:
    - Interior cells form alternating GREY/ORANGE stripes from the head side inward.
    - Chargeable cells are exactly the cells on GREY stripes.
    - Chargeable cells are processed in clockwise spiral order over the whole interior rectangle,
      filtered to GREY-stripe positions.
    - The k-th processed chargeable cell uses color cycle 2 -> 4 -> 7 -> ...
    - Steps = min(head-to-destination distance, remaining GREY chargeable cells).
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

    containers = []
    mask = np.isin(g, list(INTERIOR_COLORS))
    for pts in components_mask(mask, connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, list(INTERIOR_COLORS)).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, *INTERIOR_COLORS})]

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_head_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in {int(bg), int(BORDER), *INTERIOR_COLORS}:
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == int(hc)}
        comp0 = None
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

        border_pts = list(rect_border_coords(r0, c0, r1, c1))
        best_d = None
        dist = {}
        for pr, pc in comp0["pts"]:
            d = min(abs(int(pr - br)) + abs(int(pc - bc)) for br, bc in border_pts)
            dist[(int(pr), int(pc))] = int(d)
            if best_d is None or d < best_d:
                best_d = int(d)
        if best_d is None:
            return None
        head_pts = {p for p, d in dist.items() if int(d) == int(best_d)}

        coords = np.array(list(head_pts), dtype=int)
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        if hc1 < c0:
            direction = (0, -1)
        elif hc0 > c1:
            direction = (0, 1)
        elif hr1 < r0:
            direction = (-1, 0)
        elif hr0 > r1:
            direction = (1, 0)
        else:
            direction = (0, 0)

        r00, c00 = int(hr0), int(hc0)
        head_shape = {(int(r - r00), int(c - c00)) for r, c in head_pts}
        return {
            "color": hc,
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": (r00, c00),
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
        if head_info is None:
            continue
        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape:
                continue
            if d["pts"] == comp_pts:
                continue
            dests.append(d)

        rem = comp_pts - head_pts
        if rem:
            mask_rem = np.zeros((H, W), dtype=bool)
            for rr, cc in rem:
                mask_rem[int(rr), int(cc)] = True
            labr, nr = label(mask_rem, structure=STRUCT4)
            for k in range(1, int(nr) + 1):
                pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
                if not pts:
                    continue
                adj = False
                for r, c in pts:
                    for dr, dc in DIR4:
                        rr, cc = int(r + dr), int(c + dc)
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
                            adj = True
                            break
                    if adj:
                        break
                if adj:
                    continue
                rs = [r for r, _c in pts]
                cs = [c for _r, c in pts]
                r00, c00 = int(min(rs)), int(min(cs))
                shp = {(int(r - r00), int(c - c00)) for r, c in pts}
                if shp != head_shape:
                    continue
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00)})

        if not dests:
            continue

        best = None
        best_dest = None
        for d in dests:
            tr0 = int(d["tl"][0] - head_tl[0])
            tc0 = int(d["tl"][1] - head_tl[1])
            if direction[0] != 0 and tc0 != 0:
                continue
            if direction[1] != 0 and tr0 != 0:
                continue
            if direction[0] == 1 and tr0 <= 0:
                continue
            if direction[0] == -1 and tr0 >= 0:
                continue
            if direction[1] == 1 and tc0 <= 0:
                continue
            if direction[1] == -1 and tc0 >= 0:
                continue
            dist = abs(tr0) + abs(tc0)
            if best is None or dist < best:
                best = dist
                best_dest = d
        if best_dest is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        processed = _processed_count(inner, direction=direction)
        total_chargeable = len(_combo_order(inner.shape[0], inner.shape[1], direction=direction))
        remaining_grey = int(total_chargeable - processed)

        tr0 = int(best_dest["tl"][0] - head_tl[0])
        tc0 = int(best_dest["tl"][1] - head_tl[1])
        steps = min(abs(tr0) + abs(tc0), remaining_grey)

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BORDER, dtype=int)
        patch[1:-1, 1:-1] = _build_combo_interior(inner.shape[0], inner.shape[1], _dir_to_head_side(direction), processed + steps)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _dir_to_head_side(direction: tuple[int, int]) -> int:
    for head_side, d in SIDE_TO_DIR.items():
        if d == direction:
            return head_side
    raise ValueError(f"Unsupported direction: {direction}")


def _combo_signal_strength(specs: list[dict]) -> bool:
    """Reject weak samples where the fusion collapses visually."""
    saw_large = False
    saw_cycle = False
    saw_spiral_difference = False
    for s in specs:
        head_side = s["head_side"]
        inner_h = s["inner_h"]
        inner_w = s["inner_w"]
        initial_charged = s["initial_charged"]
        steps = s["steps"]

        combo_order = _combo_order(inner_h, inner_w, head_side=head_side)
        linear_order = _linear_grey_order(inner_h, inner_w, head_side=head_side)
        final_charged = initial_charged + steps

        if inner_h >= 4 and inner_w >= 4:
            saw_large = True
        if final_charged >= 3 and steps >= 2:
            saw_cycle = True

        combo_final = set(combo_order[:final_charged])
        linear_final = set(linear_order[:final_charged])
        if combo_final != linear_final and steps >= 2:
            saw_spiral_difference = True

    return saw_large and saw_cycle and saw_spiral_difference


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)

    specs = []
    for i in range(n_containers):
        head_side = random.randint(0, 3)
        ensure_large = i == 0
        inner_h, inner_w = _choose_inner_dims(head_side, ensure_large=ensure_large)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        chargeable = len(_combo_order(inner_h, inner_w, head_side=head_side))
        if chargeable < 2:
            return None

        max_initial = chargeable - 1
        if ensure_large:
            initial_charged = random.randint(0, min(max_initial, max(0, chargeable - 3)))
        else:
            initial_charged = random.randint(0, max_initial)
        remaining = chargeable - initial_charged
        if remaining <= 0:
            return None

        max_dist_steps = min(remaining, 7)
        if max_dist_steps < 1:
            return None

        mode = random.choice(["distance", "distance", "grey"])
        if ensure_large and max_dist_steps >= 3:
            steps = random.randint(3, max_dist_steps)
            mode = random.choice(["distance", "grey"])
        else:
            lo_steps = 1
            hi_steps = max_dist_steps
            steps = random.randint(lo_steps, hi_steps)

        if mode == "distance":
            gap = steps - 1
        else:
            gap = random.randint(steps, min(10, steps + 4))

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "gap": gap,
                "steps": steps,
                "initial_charged": initial_charged,
                "chargeable": chargeable,
            }
        )

    if not _combo_signal_strength(specs):
        return None

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]

        if hs == 0:
            total_h = 1 + gap + 1 + oh
            total_w = ow
            cont_r = gap + 2
            cont_c = 0
        elif hs == 1:
            total_h = oh + 1 + gap + 1
            total_w = ow
            cont_r = 0
            cont_c = 0
        elif hs == 2:
            total_h = oh
            total_w = 1 + gap + 1 + ow
            cont_r = 0
            cont_c = gap + 2
        else:
            total_h = oh
            total_w = ow + 1 + gap + 1
            cont_r = 0
            cont_c = 0

        assemblies.append({**s, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c})

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []
    for a in assemblies:
        th, tw = a["total_h"], a["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(300):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            cr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            cr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if not occupied[cr0:cr1, cc0:cc1].any():
                occupied[r0 : r0 + th, c0 : c0 + tw] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = _build_combo_interior(
            inner_h, inner_w, hs, a["initial_charged"]
        )

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 2 - a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 2 - a["gap"]] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + 1 + a["gap"]] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        dr, dc = SIDE_TO_DIR[hs]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        expected[cr : cr + oh, cc : cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

        nr = cr + dr * steps
        nc = cc + dc * steps
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = _build_combo_interior(inner_h, inner_w, hs, a["initial_charged"] + steps)
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr : nr + oh, nc + ow] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid fused combo variant after 500 attempts")
