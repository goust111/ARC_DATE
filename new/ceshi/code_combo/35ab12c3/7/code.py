# Combo variant of 35ab12c3 (index 7)
# combo_spec: D7/2+D8/0
# Name: 边缘像素源+射线延伸到网格边缘

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import colors_present, get_bg, neighbors8


"""
Design analysis for the fused variant
====================================

a) Concretization and behavior trace
------------------------------------
The two requested changes interact in a non-trivial way:

1. D7/2 says the ACTIVE source pixel(s) of a shape live on the border.
2. D8/0 says the stroke should not stop at its same-color partner; it should keep
   going until the grid edge.

If we naively keep the old "border pixel connected to another border pixel" pairing,
the D8 tail disappears: once an aligned partner is also on the border, the partner is
already the last cell on that ray. So the fusion would degenerate into ordinary
edge-to-edge segments and the "ray to edge" behavior would not actually change the
output.

To make both dimensions visible at once, the fused task uses one border same-color
pixel as the EMITTER and one interior same-color pixel as a GUIDE:

- The emitter is on the outer frame and is the only same-color pixel that may start a
  stroke.
- The guide is aligned with the emitter (H/V/45-degree diagonal), but it is not a
  source. It only sets the direction.
- Output draws the full ray from the emitter, through the guide, to the far border.

Concrete mini-example on a 9x9 grid:

- Input has color 3 at (0,2) and (3,2).
- Original 35ab12c3 would treat both as ordinary vertices and draw only rows 0..3 in
  that column.
- This fused variant treats (0,2) as the border source and (3,2) as the direction
  guide, so output fills the entire column 2 down to row 8.
- If a color 7 pixel sits at (1,3), adjacent to the source, then the same offset
  (+1,+1) is copied along the ray as a parallel shadow, provided that offset is not
  collinear with the main ray.

This preserves the original "same-color relation + adjacent different-color shadow"
spirit, but changes which same-color pixel is active and where the path stops.

b) Complexity calibration
-------------------------
The generator keeps a moderate but stable grammar:

- 2-4 actionable color groups per grid.
- Each actionable group is one long ray, not a dense polygon. This is deliberate:
  once every stroke reaches a border, multi-edge shapes create too many crossings.
- Directions still vary over the 8 aligned directions, so examples repeat the rule
  across horizontal, vertical, and diagonal beams.
- Some groups get a shadow, some do not.
- A low-probability distractor group remains possible.

This is simpler than the original polygon family, but not trivial: the solver still
has to infer border-source selection, interior guide usage, and full-border ray
extension from sparse pixels.

c) Implicit-assumption audit against the original main()
--------------------------------------------------------
Original assumptions that no longer hold:

- "All same-color pixels are candidate vertices." No longer true. Only border pixels
  can emit; interior same-color pixels are guides.
- "The other same-color pixel is the stopping endpoint." No longer true. The guide
  determines the direction, but the ray continues past it to the border.
- "For a 2-pixel group, the relationship is symmetric." No longer true. Border status
  breaks the symmetry between source and guide.

New assumptions introduced here:

- Each actionable group has at least one border same-color pixel and at least one
  aligned interior same-color guide.
- For each active source, the intended guide is the nearest aligned interior same-
  color pixel along a clear H/V/diag corridor.
- If a shadow exists, its anchor is adjacent to exactly one source pixel and does not
  lie on the main ray itself.

The generator explicitly guarantees those conditions.

d) Placement legality
---------------------
Because each object is a global ray, local patch pasting is unsafe. The generator
therefore renders each candidate on the full canvas, solves it in isolation, and then
rejects any candidate whose output would overlap or 8-touch existing output cells.
This keeps groups disjoint, prevents shadow collisions, and avoids order-dependent
painting artifacts.

e) Visual observability
-----------------------
The change is visible from a few examples because each actionable color repeats the
same motif:

- one same-color pixel on the border,
- one same-color guide pixel inside,
- output beam starts specifically from the border pixel and continues beyond the guide.

Humans can see that the interior pixel influences direction but is not itself a
launch point, which makes the border-source rule inferable without any color-id trick.

f) Visual cleanliness
---------------------
The generator enforces at least one background cell of separation in the final output.
It also keeps guide pixels away from the frame corners and requires positive pre-guide
and post-guide lengths, so beams are long enough to be legible but not packed into a
single dense hub.

g) Color inferability
---------------------
No color remapping is introduced. A shape keeps its own color, and an optional shadow
keeps the adjacent anchor's color, exactly as in the original family. Colors carry no
hidden ordering semantics.

Why the guide-based fusion is necessary
---------------------------------------
The crucial fusion constraint is: "border source" + "ray continues to edge". If both
same-color points were border vertices, D8 would collapse into the same output as a
finite border-to-border segment. Using an interior same-color guide avoids that
degeneracy while keeping D7/2 and D8/0 both active and visually testable.
"""


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
RAY_DIRS = DIR8[:]


def _is_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    dr, dc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + dr * k, c0 + dc * k


def _ray_cells(r0: int, c0: int, dr: int, dc: int, h: int, w: int):
    rr, cc = r0, c0
    while 0 <= rr < h and 0 <= cc < w:
        yield rr, cc
        nr = rr + dr
        nc = cc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        rr, cc = nr, nc


def _steps_to_border(r0: int, c0: int, dr: int, dc: int, h: int, w: int) -> int:
    steps = 0
    rr, cc = r0, c0
    while 0 <= rr + dr < h and 0 <= cc + dc < w:
        rr += dr
        cc += dc
        steps += 1
    return steps


def _clear_segment_in_original(
    g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]
) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _clear_ray_in_original(
    g: np.ndarray, r0: int, c0: int, dr: int, dc: int, bg: int, allow: set[int]
) -> bool:
    h, w = g.shape
    first = True
    for rr, cc in _ray_cells(r0, c0, dr, dc, h, w):
        if first:
            first = False
            continue
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _draw_ray_bg_only(out: np.ndarray, r0: int, c0: int, dr: int, dc: int, col: int, bg: int) -> None:
    h, w = out.shape
    for rr, cc in _ray_cells(r0, c0, dr, dc, h, w):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _find_guided_rays(
    g: np.ndarray, col: int, bg: int
) -> tuple[list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]], list[tuple[int, int, int]]]:
    h, w = g.shape
    color_pts = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    border_pts = sorted((r, c) for r, c in color_pts if _is_border(r, c, h, w))
    inner_pts = sorted((r, c) for r, c in color_pts if not _is_border(r, c, h, w))

    rays: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for sr, sc in border_pts:
        candidates: list[tuple[int, int, int]] = []
        for gr, gc in inner_pts:
            if not _aligned_8(sr, sc, gr, gc):
                continue
            if not _clear_segment_in_original(g, sr, sc, gr, gc, bg, allow={col}):
                continue
            dr, dc = _step_dir(sr, sc, gr, gc)
            if _steps_to_border(gr, gc, dr, dc, h, w) <= 0:
                continue
            dist = max(abs(gr - sr), abs(gc - sc))
            candidates.append((dist, gr, gc))
        if not candidates:
            continue
        _dist, gr, gc = min(candidates)
        dr, dc = _step_dir(sr, sc, gr, gc)
        if not _clear_ray_in_original(g, sr, sc, dr, dc, bg, allow={col}):
            continue
        rays.append(((sr, sc), (gr, gc), (dr, dc)))

    seed_sources = []
    for i, ((sr, sc), _guide, _step) in enumerate(rays):
        for rr, cc in neighbors8(sr, sc):
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_sources.append(i)
                    break

    if seed_sources:
        seed_idx = min(seed_sources, key=lambda i: (rays[i][0][0], rays[i][0][1]))
    elif rays:
        seed_idx = min(range(len(rays)), key=lambda i: (rays[i][0][0], rays[i][0][1]))
    else:
        return [], []

    seed = rays[seed_idx][0]
    rays = [rays[seed_idx]] + [r for i, r in enumerate(rays) if i != seed_idx]

    carried: list[tuple[int, int, int]] = []
    for drn, dcn in DIR8:
        rr = seed[0] + drn
        cc = seed[1] + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != col:
                carried.append((drn, dcn, v))
    return rays, carried


"""
Overview:
Each active color uses a border pixel as a ray emitter and an aligned interior
same-color pixel as a direction guide.

Input structure:
- Background is 0.
- An actionable color has at least one border pixel and at least one interior pixel
  of the same color, aligned H/V/45 degrees.
- Optional shadow anchors are different-colored pixels adjacent to the chosen border
  source.

Rule:
For each actionable color, ignore interior same-color pixels as sources. From each
border same-color pixel, fire a ray in the direction of its aligned interior guide and
extend that ray all the way to the far border. If the selected source has adjacent
different-colored neighbor pixels, copy those colors along parallel shifted rays,
skipping offsets collinear with the main ray.

Key insight:
The interior same-color pixel is not the endpoint of the drawing. It only tells the
border pixel which direction to emit, and the emission keeps going until the grid edge.
"""
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        rays, carried = _find_guided_rays(g, col, bg)
        if not rays:
            continue

        for (sr, sc), _guide, (dr, dc) in rays:
            _draw_ray_bg_only(out, sr, sc, dr, dc, col, bg)

        for (sr, sc), _guide, (drs, dcs) in rays:
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = sr + drn
                cc0 = sc + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w):
                    continue
                if not _clear_ray_in_original(g, rr0, cc0, drs, dcs, bg, allow=set()):
                    continue
                _draw_ray_bg_only(out, rr0, cc0, drs, dcs, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (16, 23)
CANVAS_W_JITTER = (-2, 3)
CANVAS_W_CLIP = (16, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 2
DISTRACTOR_PROB = 0.25

GEOMETRY_RETRIES = 120
PLACEMENT_RETRIES = 120
DISTRACTOR_RETRIES = 80
GENERATE_RETRIES = 400

GUIDE_MARGIN = 2
MIN_PRE_GUIDE = 2
MAX_PRE_GUIDE = 8
MIN_TAIL = 2


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count : shape_count + shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _candidate_blocked(scene: np.ndarray) -> set[tuple[int, int]]:
    h, w = scene.shape
    blocked: set[tuple[int, int]] = set()
    for r, c in np.argwhere(scene != BG):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _can_place(scene_out: np.ndarray, cand_out: np.ndarray) -> bool:
    blocked = _candidate_blocked(scene_out)
    for r, c in np.argwhere(cand_out != BG):
        if (int(r), int(c)) in blocked:
            return False
    return True


def _commit(scene_in: np.ndarray, scene_out: np.ndarray, cand_in: np.ndarray, cand_out: np.ndarray) -> None:
    in_mask = cand_in != BG
    out_mask = cand_out != BG
    scene_in[in_mask] = cand_in[in_mask]
    scene_out[out_mask] = cand_out[out_mask]


def _walk_back_to_border(guide: tuple[int, int], step: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = guide
    dr, dc = step
    while 0 <= r - dr < h and 0 <= c - dc < w:
        r -= dr
        c -= dc
    return r, c


def _finite_only_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    h, w = g.shape
    for col in colors:
        rays, carried = _find_guided_rays(g, col, bg)
        if not rays:
            continue
        for (sr, sc), (gr, gc), _step in rays:
            _draw_line_bg_only(out, sr, sc, gr, gc, col, bg)
        for (sr, sc), (gr, gc), (drs, dcs) in rays:
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = sr + drn
                cc0 = sc + dcn
                rr1 = gr + drn
                cc1 = gc + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _clear_segment_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)
    return out


def _sample_source_and_guide(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    for _ in range(GEOMETRY_RETRIES):
        step = RAY_DIRS[int(np.random.randint(len(RAY_DIRS)))]
        guide = (
            int(np.random.randint(GUIDE_MARGIN, h - GUIDE_MARGIN)),
            int(np.random.randint(GUIDE_MARGIN, w - GUIDE_MARGIN)),
        )
        source = _walk_back_to_border(guide, step, h, w)
        pre = max(abs(guide[0] - source[0]), abs(guide[1] - source[1]))
        tail = _steps_to_border(guide[0], guide[1], step[0], step[1], h, w)
        if pre < MIN_PRE_GUIDE or pre > MAX_PRE_GUIDE:
            continue
        if tail < MIN_TAIL:
            continue
        return source, guide, step
    return None


def _shadow_candidates(
    h: int, w: int, source: tuple[int, int], guide: tuple[int, int], step: tuple[int, int], color_cells: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if step[0] * dc - step[1] * dr == 0:
            continue
        pos = (source[0] + dr, source[1] + dc)
        if not (0 <= pos[0] < h and 0 <= pos[1] < w):
            continue
        if pos in color_cells:
            continue
        neighbor_hits = 0
        for rr, cc in neighbors8(pos[0], pos[1]):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) in color_cells:
                neighbor_hits += 1
        if neighbor_hits != 1:
            continue
        out.append((dr, dc))
    return out


def _make_shape_candidate(h: int, w: int, shape_color: int, shadow_color: int | None):
    spec = _sample_source_and_guide(h, w)
    if spec is None:
        return None
    source, guide, step = spec
    cand_in = np.full((h, w), BG, dtype=int)
    cand_in[source] = int(shape_color)
    cand_in[guide] = int(shape_color)

    shadow_offset = None
    if shadow_color is not None:
        offsets = _shadow_candidates(h, w, source, guide, step, {source, guide})
        if not offsets:
            return None
        shadow_offset = offsets[int(np.random.randint(len(offsets)))]
        anchor = (source[0] + shadow_offset[0], source[1] + shadow_offset[1])
        cand_in[anchor] = int(shadow_color)

    cand_out = main(cand_in)
    finite_out = _finite_only_output(cand_in)

    tail = _steps_to_border(guide[0], guide[1], step[0], step[1], h, w)
    extra_cells = int(np.count_nonzero((cand_out != BG) & (finite_out == BG)))
    if tail < MIN_TAIL or extra_cells <= 0:
        return None

    rays, carried = _find_guided_rays(cand_in, int(shape_color), BG)
    if len(rays) != 1:
        return None
    if shadow_color is None and carried:
        return None
    if shadow_color is not None:
        if len(carried) != 1 or carried[0][2] != int(shadow_color):
            return None

    meta = {
        "source": source,
        "guide": guide,
        "step": step,
        "tail": int(tail),
        "extra_cells": int(extra_cells),
        "has_shadow": shadow_color is not None,
    }
    return cand_in, cand_out, meta


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (
            int(np.random.randint(1, h - 1)),
            int(np.random.randint(1, w - 1)),
        )
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned_8(coords[i][0], coords[i][1], coords[j][0], coords[j][1]):
                return None
    return coords


def _make_distractor_candidate(h: int, w: int, color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(DISTRACTOR_RETRIES):
        count = int(np.random.randint(1, 4))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        cand = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            cand[r, c] = int(color)
        return cand.copy(), cand.copy()
    return None


def _gen_attempt(return_meta: bool = False):
    h, w = _sample_canvas_size()
    scene_in = np.full((h, w), BG, dtype=int)
    scene_out = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    metrics = {
        "shape_count": int(shape_count),
        "shadow_count": int(len(shadow_colors)),
        "tail_lengths": [],
        "extra_cells": [],
        "guide_count": 0,
    }

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            candidate = _make_shape_candidate(h, w, int(shape_color), shadow_map.get(idx))
            if candidate is None:
                continue
            cand_in, cand_out, meta = candidate
            if not _can_place(scene_out, cand_out):
                continue
            _commit(scene_in, scene_out, cand_in, cand_out)
            metrics["tail_lengths"].append(int(meta["tail"]))
            metrics["extra_cells"].append(int(meta["extra_cells"]))
            metrics["guide_count"] += 1
            placed = True
            break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            candidate = _make_distractor_candidate(h, w, int(distractor_color))
            if candidate is None:
                continue
            cand_in, cand_out = candidate
            if not _can_place(scene_out, cand_out):
                continue
            _commit(scene_in, scene_out, cand_in, cand_out)
            placed = True
            break
        if not placed:
            return None

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    solved = main(scene_in)
    if get_bg(scene_in) != BG or get_bg(scene_out) != BG:
        return None
    if len(colors_present(scene_in, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(solved, scene_out):
        return None
    if not metrics["tail_lengths"] or min(metrics["tail_lengths"]) < MIN_TAIL:
        return None
    if not metrics["extra_cells"] or min(metrics["extra_cells"]) <= 0:
        return None

    if return_meta:
        return scene_in, metrics
    return scene_in


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
