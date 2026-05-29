# Combo variant of a6f40cea (index 40)
# combo_spec: D1/3+D3/0+D4/3+D5/0
# Name: 全网格箭头射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    Original a6f40cea used open rectangles with two ring touches, decoded two
#    endpoints, then drew an interior L-path and finally cropped to the frame
#    interior. This combo changes all four of those stages together:
#    - D3/0: each meaningful source is now an arrow sprite, not an open frame
#    - D1/3: the inferred interior mark is a straight row/column ray, not an L
#    - D4/3: the ray starts at the arrow's touchpoint and continues to the
#      opposite inner border, not to a second source touchpoint
#    - D5/0: the output keeps the full canvas, so arrows remain visible outside
#      while the inferred ray appears inside the frame
#    Example: a top-side arrow has its tip on row r0-1 at column c0+1+k and its
#    head/shaft extend upward, so the arrow visibly points downward into the
#    frame. The solver maps that tip to interior local start (0, k), then paints
#    the full column [(0, k), (1, k), ..., (H-1, k)] inside a copy of the input.
#    If the arrow uses two colors, the tip color is also the first ray color.
#
# b) Complexity calibration
#    The variant is simpler than the original in geometry, so the generator keeps
#    2-3 arrows per sample and commonly uses two-color arrows. To avoid visual
#    ambiguity from crossing rays, each sample uses only one ray family:
#    vertical rays (top/bottom arrows) or horizontal rays (left/right arrows).
#    This still gives repeated evidence inside one grid while keeping the mapping
#    human-legible from a few examples.
#
# c) Hidden assumptions broken from the original solver
#    - "Each useful component contributes >=2 ring touches" is false now.
#      Meaningful arrows have exactly one ring-touching tip.
#    - "A second endpoint must be decoded from geometry" is false now.
#      The destination is forced by D4/3: continue straight to the opposite side.
#    - "Output canvas equals the interior crop" is false now.
#      The solver must preserve the whole input and repaint only the interior.
#    - "Elbow selection matters" is false now because D1/3 removes bends.
#    The combo therefore needs a new primitive: detect one arrow tip, infer its
#    side/direction, then generate a full-span ordered ray from that side.
#
# d) Placement legality
#    The generator uses output-first reasoning. It chooses an interior row/column
#    for each ray, then places the matching arrow just outside the corresponding
#    frame side. Rays are kept parallel within a sample, so output paths do not
#    intersect. Arrow components are validated to be in-bounds, painted only on
#    background cells, and not 4-adjacent to previously placed components.
#
# e) Visual observability
#    A human can infer the rule from 2-4 examples because three cues line up:
#    - arrows visibly replace the old open rectangles
#    - each arrow points into the frame from the side where the ray starts
#    - the output keeps the whole scene, so the outside arrow and inside ray are
#      simultaneously visible and spatially aligned
#    No color-id sorting or invisible numeric ordering is used.
#
# f) Visual cleanliness
#    Parallel-only rays avoid interior crossings. Arrow sprites stay outside the
#    frame and distractors are kept away from the ring. This leaves a clean frame
#    outline, clearly separated arrows, and uncluttered interior stripes.
#
# g) Color inference
#    Monochrome arrows produce solid rays. Two-color arrows alternate along the
#    source sprite and along the interior ray. The visible tip color anchors the
#    first ray cell, so the alternation phase is recoverable from the input.
#
# h) Non-degeneracy target
#    These samples should not collapse back to the original task:
#    - the source geometry has one ring touch instead of two
#    - the inferred path is full-span straight, not endpoint-to-endpoint
#    - the output shape matches the input, not the interior crop
#    Even on the same input, the original solver cannot match this output shape.
#
# i) Solver assumption audit for the new code
#    - Meaningful components must have exactly one ring hit and a valid arrow-tip
#      signature. The generator guarantees both.
#    - Components must use one or two colors. The generator never uses more.
#    - Decoded rays must not need conflict resolution at interior crossings.
#      The generator guarantees a single orientation family per sample and unique
#      row/column choices within that family.
#    - Distractors must not mimic an arrow touching the ring. The generator keeps
#      distractors away from the frame neighborhood, so they fail decoding.
# -----------------------------------------------------------------------------


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such
    rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None
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


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _component_side_of_ring_cell(rr: int, cc: int, r0: int, c0: int, r1: int, c1: int) -> str | None:
    if rr == r0 - 1 and c0 <= cc <= c1:
        return "top"
    if rr == r1 + 1 and c0 <= cc <= c1:
        return "bottom"
    if cc == c0 - 1 and r0 <= rr <= r1:
        return "left"
    if cc == c1 + 1 and r0 <= rr <= r1:
        return "right"
    return None


def _tip_head_cells(tip: tuple[int, int], side: str) -> list[tuple[int, int]]:
    tr, tc = tip
    if side == "top":
        return [(tr - 1, tc - 1), (tr - 1, tc + 1)]
    if side == "bottom":
        return [(tr + 1, tc - 1), (tr + 1, tc + 1)]
    if side == "left":
        return [(tr - 1, tc - 1), (tr + 1, tc - 1)]
    if side == "right":
        return [(tr - 1, tc + 1), (tr + 1, tc + 1)]
    raise ValueError(side)


def _has_arrow_tip(coords: set[tuple[int, int]], tip: tuple[int, int], side: str) -> bool:
    tr, tc = tip
    if side == "top":
        needed = {(tr - 1, tc), (tr - 1, tc - 1), (tr - 1, tc + 1)}
    elif side == "bottom":
        needed = {(tr + 1, tc), (tr + 1, tc - 1), (tr + 1, tc + 1)}
    elif side == "left":
        needed = {(tr, tc - 1), (tr - 1, tc - 1), (tr + 1, tc - 1)}
    elif side == "right":
        needed = {(tr, tc + 1), (tr - 1, tc + 1), (tr + 1, tc + 1)}
    else:
        return False
    return needed.issubset(coords)


def _map_tip_to_local(rr: int, cc: int, side: str, outH: int, outW: int, r0: int, c0: int) -> tuple[int, int]:
    if side == "top":
        return 0, int(cc - (c0 + 1))
    if side == "bottom":
        return outH - 1, int(cc - (c0 + 1))
    if side == "left":
        return int(rr - (r0 + 1)), 0
    if side == "right":
        return int(rr - (r0 + 1)), outW - 1
    raise ValueError(side)


def _build_ray_path(start: tuple[int, int], side: str, outH: int, outW: int) -> list[tuple[int, int]]:
    sr, sc = start
    if side == "top":
        return [(r, sc) for r in range(0, outH)]
    if side == "bottom":
        return [(r, sc) for r in range(outH - 1, -1, -1)]
    if side == "left":
        return [(sr, c) for c in range(0, outW)]
    if side == "right":
        return [(sr, c) for c in range(outW - 1, -1, -1)]
    raise ValueError(side)


def _ring_cells(H: int, W: int, r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    ring = set()
    for cc in range(c0, c1 + 1):
        if 0 <= r0 - 1 < H:
            ring.add((r0 - 1, cc))
        if 0 <= r1 + 1 < H:
            ring.add((r1 + 1, cc))
    for rr in range(r0, r1 + 1):
        if 0 <= c0 - 1 < W:
            ring.add((rr, c0 - 1))
        if 0 <= c1 + 1 < W:
            ring.add((rr, c1 + 1))
    return ring


def _decode_arrow_components(
    g: np.ndarray,
    *,
    bg: int,
    bc: int,
    fill: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[dict[str, object]]:
    H, W = g.shape
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    ring = _ring_cells(H, W, r0, c0, r1, c1)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    decoded: list[dict[str, object]] = []
    for cid in range(1, int(cn) + 1):
        coords_arr = np.argwhere(clab == int(cid))
        if coords_arr.size == 0:
            continue
        coords = {(int(r), int(c)) for r, c in coords_arr}
        ring_hits = [p for p in coords if p in ring]
        if len(ring_hits) != 1:
            continue

        tip = tuple(map(int, ring_hits[0]))
        side = _component_side_of_ring_cell(tip[0], tip[1], r0, c0, r1, c1)
        if side is None or not _has_arrow_tip(coords, tip, side):
            continue

        cols = sorted(set(map(int, g[clab == int(cid)].flatten())))
        if len(cols) not in (1, 2):
            continue

        start = _map_tip_to_local(tip[0], tip[1], side, outH, outW, r0, c0)
        path = _build_ray_path(start, side, outH, outW)
        decoded.append(
            {
                "cid": int(cid),
                "coords": coords,
                "tip": tip,
                "side": side,
                "start": start,
                "path": path,
                "colors": cols,
            }
        )

    return decoded


def _paint_alt_path(
    out: np.ndarray,
    path: list[tuple[int, int]],
    start_color: int,
    other_color: int,
) -> None:
    cur = int(start_color)
    for rr, cc in path:
        out[int(rr), int(cc)] = int(cur)
        cur = int(other_color) if cur == int(start_color) else int(start_color)


def _color_lookup_along_backbone(path: list[tuple[int, int]], colors: tuple[int, ...]) -> dict[tuple[int, int], int]:
    if len(colors) == 1:
        return {p: int(colors[0]) for p in path}
    a, b = map(int, colors)
    return {p: (a if i % 2 == 0 else b) for i, p in enumerate(path)}


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
    else:
        a, b = map(int, colors)
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


def _shift_path(path: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in path]


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and bool(comp_mask[nr, nc]):
                return False
    return True


def _place_arrow_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    backbone: list[tuple[int, int]],
    head_cells: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    coords = list(dict.fromkeys(backbone + head_cells))
    if len(coords) != len(set(coords)):
        return False
    if not _can_place(grid, coords, bg, comp_mask):
        return False

    color_map = _color_lookup_along_backbone(backbone, colors)
    for rr, cc in backbone:
        grid[int(rr), int(cc)] = int(color_map[(rr, cc)])
    shaft = backbone[1] if len(backbone) >= 2 else backbone[0]
    head_color = int(color_map[shaft])
    for rr, cc in head_cells:
        grid[int(rr), int(cc)] = int(head_color)

    rs, cs = zip(*coords)
    comp_mask[rs, cs] = True
    return True


def _arrow_backbone(side: str, tip: tuple[int, int], length: int) -> list[tuple[int, int]]:
    tr, tc = tip
    if side == "top":
        return [(tr - i, tc) for i in range(length)]
    if side == "bottom":
        return [(tr + i, tc) for i in range(length)]
    if side == "left":
        return [(tr, tc - i) for i in range(length)]
    if side == "right":
        return [(tr, tc + i) for i in range(length)]
    raise ValueError(side)


def _tip_for_side(side: str, pos: int, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    if side == "top":
        return (r0 - 1, c0 + 1 + pos)
    if side == "bottom":
        return (r1 + 1, c0 + 1 + pos)
    if side == "left":
        return (r0 + 1 + pos, c0 - 1)
    if side == "right":
        return (r0 + 1 + pos, c1 + 1)
    raise ValueError(side)


def _ray_local_path(side: str, pos: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(r, pos) for r in range(H)]
    if side == "bottom":
        return [(r, pos) for r in range(H - 1, -1, -1)]
    if side == "left":
        return [(pos, c) for c in range(W)]
    if side == "right":
        return [(pos, c) for c in range(W - 1, -1, -1)]
    raise ValueError(side)


def _choose_side_assignments(n: int, axis_len: int, sides: tuple[str, str]) -> list[tuple[str, int]] | None:
    positions = list(range(axis_len))
    for _ in range(60):
        chosen = random.sample(positions, n)
        random.shuffle(chosen)
        labeled = []
        for i, pos in enumerate(chosen):
            if i < len(sides):
                side = sides[i]
            else:
                side = random.choice(sides)
            labeled.append((side, int(pos)))
        random.shuffle(labeled)
        ok = True
        for side in sides:
            same = sorted(pos for s, pos in labeled if s == side)
            if any(b - a <= 3 for a, b in zip(same, same[1:])):
                ok = False
                break
        if ok:
            return labeled
    return None


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _can_place(grid, coords, bg, comp_mask):
            rs, cs = zip(*coords)
            grid[rs, cs] = int(color)
            comp_mask[rs, cs] = True
            return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct full-span interior rays from external arrows while preserving the full canvas.

    Input structure:
    - One 1px closed rectangular frame sits on a uniform background, with a uniform fill inside.
    - Useful external components are arrow sprites whose tips touch the ring just outside the frame.
    - Optional distractor blobs may appear elsewhere, but they do not touch the frame neighborhood.

    Rule:
    - Output keeps the same shape as the input and preserves everything outside the frame interior.
    - Reset the frame interior to its fill color, then decode each valid arrow tip as one boundary startpoint.
    - Extend a straight row or column ray from that startpoint across the interior to the opposite side.
    - Monochrome arrows produce solid rays; two-color arrows alternate starting from the visible tip color.

    Key insight:
    - The outside object now encodes only one touchpoint plus a direction. Because the destination is
      "the opposite side" rather than "another touchpoint", solving reduces to detecting the arrow tip,
      reading which side it touches, and drawing the corresponding full-span ordered ray.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    decoded = _decode_arrow_components(g, bg=bg, bc=bc, fill=fill, r0=r0, c0=c0, r1=r1, c1=c1)
    for item in decoded:
        path = list(item["path"])
        colors = list(item["colors"])
        abs_path = _shift_path(path, r0 + 1, c0 + 1)
        if len(colors) == 1:
            _paint_path(out, abs_path, (int(colors[0]),))
            continue

        tip = tuple(item["tip"])
        start_color = int(g[tip[0], tip[1]])
        other = int(colors[0]) if int(colors[1]) == int(start_color) else int(colors[1])
        _paint_alt_path(out, abs_path, start_color, other)

    return out


# --------------- generator ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_ARROW_LEN = 4
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.65
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    comp_mask = np.zeros_like(grid, dtype=bool)
    used_inner = np.zeros((H, W), dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0

    orientation = random.choice(["vertical", "horizontal"])
    axis_len = W if orientation == "vertical" else H
    sides = ("top", "bottom") if orientation == "vertical" else ("left", "right")
    n_components = random.randint(*_N_COMPONENTS_RANGE)
    assignments = _choose_side_assignments(n_components, axis_len, sides)
    if assignments is None:
        return None

    inner_specs: list[tuple[list[tuple[int, int]], tuple[int, ...]]] = []
    dual_count = 0
    mono_count = 0
    for idx, (side, pos) in enumerate(assignments):
        ray = _ray_local_path(side, pos, H, W)
        irs, ics = zip(*ray)
        if used_inner[irs, ics].any():
            return None

        if color_i >= len(palette):
            return None
        remaining = n_components - idx
        need_dual = dual_count == 0
        need_mono = mono_count == 0
        force_dual = need_dual and remaining == 1
        force_mono = need_mono and remaining == 1
        choose_dual = False
        if force_dual:
            choose_dual = True
        elif force_mono:
            choose_dual = False
        elif need_dual and idx == 0:
            choose_dual = True
        elif need_mono and idx == 1:
            choose_dual = False
        else:
            choose_dual = random.random() < _DUAL_COLOR_PROB

        if choose_dual and color_i + 1 < len(palette):
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
            dual_count += 1
        else:
            colors = (palette[color_i],)
            color_i += 1
            mono_count += 1

        tip = _tip_for_side(side, pos, r0, c0, r1, c1)
        backbone = _arrow_backbone(side, tip, _ARROW_LEN)
        head = _tip_head_cells(tip, side)
        if not _place_arrow_component(grid, comp_mask, backbone, head, colors, bg):
            return None

        inner_specs.append((ray, colors))
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), palette[color_i])
        color_i += 1

    expected = np.array(grid, copy=True)
    expected[r0 + 1 : r1, c0 + 1 : c1] = int(fill)
    for ray, colors in inner_specs:
        abs_ray = _shift_path(ray, r0 + 1, c0 + 1)
        _paint_path(expected, abs_ray, colors)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
