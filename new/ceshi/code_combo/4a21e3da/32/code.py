# Combo variant of 4a21e3da (index 32)
# combo_spec: D1/0+D5/1+D13/0
# Name: 对角切割+镜像翻转+连通归属

from __future__ import annotations

"""
Design analysis for the fused variant
=====================================

a) Concrete behavior and trace
Small canonical example:
  - Input keeps the original visual roles: one centered orange sprite (`fg=7`),
    one red border control pixel (`ctrl=2`), blue background (`bg=1`).
  - Fused variant here:
      the same top control now fires along the unique inward 45-degree diagonal
      that actually hits orange pixels, e.g. `(0, c) -> (1, c+1) -> ...`.
      Orange pixels that lie on that diagonal stay in place.
      Remove those diagonal pixels temporarily and inspect the remaining orange
      4-connected components:
        * a side block with one cell orthogonally adjacent to a diagonal pixel
          is "owned" by the split line, so it survives and slides to its corner
          after being mirrored inside its own bbox;
        * a nearby block that only touches that side block diagonally is a
          separate 4-component and is not adjacent to any diagonal pixel, so it
          disappears.
  - Difference from the original rule:
      keep/drop is no longer determined by geometric side ownership. Two blocks
      on the same side of the same diagonal can behave differently, and kept
      blocks do not preserve their handedness while moving to corners.

b) Complexity calibration
  - The diagonal split is visibly different from the seed: the red ray is no
    longer horizontal/vertical, and the stationary center pixels form a 45-
    degree chain instead of a row/column.
  - The meta-rule is also visibly different: the generator always creates at
    least one "weakly attached" block that is on the same geometric side as a
    surviving block but disappears because it is only diagonally connected.
  - To keep the task learnable, the generator mostly uses one control pixel.
    This removes the unnecessary ambiguity of crossing diagonals and lets the
    topological keep/drop rule be the dominant pattern visible from a few
    examples.

c) Audit of original solver assumptions that break
  - Original assumption: every ctrl determines a row/column split. Broken.
    Here each ctrl determines one diagonal, chosen as the unique inward 45-
    degree path that intersects the sprite.
  - Original assumption: split-line fg are row/column-aligned and therefore
    4-connected. Broken. Diagonal split-line fg are typically only 8-connected.
  - Original assumption: one geometric region maps to one moved piece and only
    needs translation. Broken. We must first cut away the diagonal fg, compute
    4-components on the remaining fg, keep only those components adjacent to
    diagonal fg, and then mirror the kept component before corner placement.
  - Original assumption: global fg is one 4-connected component. Intentionally
    broken by the generator. The fused variant uses an 8-connected but not
    4-connected sprite so that diagonal-only "fragile links" are observable.

d) Placement legality
  - The border still contains only the ctrl pixel(s); all fg stays at least
    four cells away from the border.
  - Generator places every kept component strictly off the diagonal and every
    vanishing component strictly off the diagonal and off direct 4-neighbor
    contact with it.
  - All moved corner pieces are bucketed by destination corner and translated by
    their union bbox, so there is no accidental overlap between separate kept
    components that share a destination.

e) Visual observability
  - The diagonal ray is explicit in the output.
  - The center of the sprite contains stationary diagonal fg pixels.
  - One side always contains a kept block plus a nearby diagonally touching
    vanish block. This makes "same side but different fate" visible with the
    naked eye and forces the solver to reason about 4-connectivity.
  - Kept blocks are intentionally non-symmetric, so their mirrored corner
    silhouettes visibly differ from a pure slide-only interpretation.

f) Visual cleanliness
  - Only three colors are used.
  - The input is a single compact diagonal sprite rather than scattered noise.
  - The weak links are implemented with one diagonal corner-touch between
    components, which is easy to see and not visually messy.

g) New solver assumptions and how the generator guarantees them
  - Assumption: each control pixel has exactly one inward diagonal candidate
    that hits fg. Generator places fg only on one candidate diagonal.
  - Assumption: every off-line component lies on a single side of each active
    diagonal. Generator uses fixed templates that never straddle the line.
  - Assumption: the intended fused behavior is active. Generator enforces:
      * at least one vanished off-line 4-component;
      * whole-fg mask is 8-connected;
      * whole-fg mask is not 4-connected.
"""

import random
from collections import defaultdict

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, is_contiguous


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

TB_LEFT_KEPT = frozenset({(0, -1), (1, -1), (2, -1), (1, -2), (2, -2)})
TB_LEFT_VANISH = frozenset({(3, -3), (3, -4), (4, -3), (4, -4)})
TB_RIGHT_KEPT = frozenset({(0, 1), (1, 1), (2, 1), (0, 2), (1, 2)})
TB_RIGHT_VANISH = frozenset({(-3, 3), (-3, 4), (-4, 3), (-4, 4)})

LEFT_TOP_KEPT = frozenset({(-1, 0), (-2, 0), (-1, 1), (-2, 1), (-1, 2)})
LEFT_TOP_VANISH = frozenset({(-3, 3), (-4, 3), (-3, 4), (-4, 4)})
LEFT_BOTTOM_KEPT = frozenset({(1, 0), (2, 0), (1, 1), (2, 1), (2, 2)})
LEFT_BOTTOM_VANISH = frozenset({(3, 3), (4, 3), (3, 4), (4, 4)})

RIGHT_TOP_KEPT = frozenset({(-1, 0), (-2, 0), (-1, -1), (-2, -1), (-2, -2)})
RIGHT_TOP_VANISH = frozenset({(-3, -3), (-4, -3), (-3, -4), (-4, -4)})
RIGHT_BOTTOM_KEPT = frozenset({(1, 0), (2, 0), (1, -1), (2, -1), (1, -2)})
RIGHT_BOTTOM_VANISH = frozenset({(3, -3), (4, -3), (3, -4), (4, -4)})


def _normalize_points(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> tuple[frozenset[tuple[int, int]], int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin = int(min(rs))
    cmin = int(min(cs))
    shape = frozenset((int(r - rmin), int(c - cmin)) for r, c in pts)
    h = int(max(r for r, _ in shape) + 1)
    w = int(max(c for _, c in shape) + 1)
    return shape, h, w


def _flip_shape(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
) -> frozenset[tuple[int, int]]:
    return frozenset((int(h - 1 - r), int(w - 1 - c)) for r, c in shape)


def _place_shape_in_corner(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    dest: tuple[str, str],
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    dh, dv = dest
    r0 = 0 if dh == "T" else int(H - h)
    c0 = 0 if dv == "L" else int(W - w)
    return frozenset((int(r0 + r), int(c0 + c)) for r, c in shape)


def _edge_of_ctrl(r: int, c: int, H: int, W: int) -> str:
    if r == 0:
        return "top"
    if r == H - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == W - 1:
        return "right"
    raise AssertionError("control pixel must be on the border")


def _candidate_dirs(edge: str) -> list[tuple[int, int]]:
    if edge == "top":
        return [(1, -1), (1, 1)]
    if edge == "bottom":
        return [(-1, -1), (-1, 1)]
    if edge == "left":
        return [(-1, 1), (1, 1)]
    if edge == "right":
        return [(-1, -1), (1, -1)]
    raise ValueError(edge)


def _walk_path(shape: tuple[int, int], start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    H, W = shape
    r, c = start
    path = [(int(r), int(c))]
    while True:
        r += int(dr)
        c += int(dc)
        if not (0 <= r < H and 0 <= c < W):
            break
        path.append((int(r), int(c)))
    return path


def _infer_lines(g: np.ndarray, fg: int, ctrl: int) -> list[dict[str, object]]:
    H, W = g.shape
    ctrl_pts = []
    for r, c in np.argwhere(g == int(ctrl)):
        r = int(r)
        c = int(c)
        if r in (0, H - 1) or c in (0, W - 1):
            ctrl_pts.append((r, c))
    lines: list[dict[str, object]] = []
    for start in ctrl_pts:
        edge = _edge_of_ctrl(start[0], start[1], H, W)
        candidates = []
        for dr, dc in _candidate_dirs(edge):
            path = _walk_path((H, W), start, dr, dc)
            hits = [idx for idx, (r, c) in enumerate(path) if int(g[r, c]) == int(fg)]
            if hits:
                score = (len(hits), hits[-1])
                candidates.append((score, dr, dc, path, hits[-1]))
        assert candidates, "each control must have one diagonal that hits fg"
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0]
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise AssertionError("ambiguous diagonal control direction")
        _score, dr, dc, path, stop_idx = best
        active_path = [(int(r), int(c)) for r, c in path[: stop_idx + 1]]
        line_fg = {(int(r), int(c)) for r, c in active_path if int(g[r, c]) == int(fg)}
        lines.append(
            {
                "edge": edge,
                "start": (int(start[0]), int(start[1])),
                "dir": (int(dr), int(dc)),
                "path": tuple(active_path),
                "line_fg": frozenset(line_fg),
            }
        )
    lines.sort(key=lambda line: (str(line["edge"]), tuple(line["start"])))
    return lines


def _line_side(point: tuple[int, int], line: dict[str, object]) -> str:
    r, c = point
    edge = str(line["edge"])
    r0, c0 = tuple(line["start"])
    dr, dc = tuple(line["dir"])
    if edge in ("top", "bottom"):
        t = int((r - r0) / dr)
        line_c = int(c0 + dc * t)
        return "L" if c < line_c else "R"
    t = int((c - c0) / dc)
    line_r = int(r0 + dr * t)
    return "T" if r < line_r else "B"


def _majority(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(sorted(counts), key=lambda key: counts[key])


def _destination_for_component(comp: list[tuple[int, int]], lines: list[dict[str, object]]) -> tuple[str, str]:
    dh: str | None = None
    dv: str | None = None
    for line in lines:
        edge = str(line["edge"])
        sides = [_line_side(pt, line) for pt in comp]
        side = _majority(sides)
        if edge == "top":
            dh = "T"
            dv = side
        elif edge == "bottom":
            dh = "B"
            dv = side
        elif edge == "left":
            dv = "L"
            dh = side
        elif edge == "right":
            dv = "R"
            dh = side
    if dh is None or dv is None:
        raise AssertionError("component destination could not be inferred")
    return dh, dv


def _paint_rays(out: np.ndarray, g: np.ndarray, lines: list[dict[str, object]], ctrl: int, bg: int) -> None:
    for line in lines:
        for r, c in line["path"]:
            if int(g[r, c]) == int(bg):
                out[r, c] = int(ctrl)
        sr, sc = tuple(line["start"])
        out[sr, sc] = int(ctrl)


def _translate_bucket(
    out: np.ndarray,
    pts: set[tuple[int, int]],
    dest: tuple[str, str],
    fg: int,
    *,
    mirror: bool,
) -> None:
    shape, h, w = _normalize_points(pts)
    if mirror:
        shape = _flip_shape(shape, h, w)
    for r, c in _place_shape_in_corner(shape, h, w, dest, out.shape[0], out.shape[1]):
        out[int(r), int(c)] = int(fg)


def _line_adjacent(comp: list[tuple[int, int]], line_fg: set[tuple[int, int]]) -> bool:
    for r, c in comp:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in line_fg:
                return True
    return False


def _extract_offline_components(g: np.ndarray, fg: int, line_fg: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    mask = (g == int(fg))
    for r, c in line_fg:
        mask[r, c] = False
    comps = []
    for coords in components_mask(mask, connectivity=4):
        comp = [(int(r), int(c)) for r, c in coords]
        comps.append(comp)
    return comps


def _solve(grid: np.ndarray, *, mirror: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))
    lines = _infer_lines(g, fg, ctrl)
    line_fg = set()
    for line in lines:
        line_fg |= set(line["line_fg"])
    lines = _infer_lines(g, fg, ctrl)
    line_fg: set[tuple[int, int]] = set()
    for line in lines:
        line_fg |= set(line["line_fg"])

    out = np.full((H, W), int(bg), dtype=int)
    _paint_rays(out, g, lines, ctrl, bg)

    for r, c in line_fg:
        out[r, c] = int(fg)

    buckets: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for comp in _extract_offline_components(g, fg, line_fg):
        if not _line_adjacent(comp, line_fg):
            continue
        dest = _destination_for_component(comp, lines)
        buckets[dest].update(comp)

    for dest, pts in buckets.items():
        _translate_bucket(out, pts, dest, fg, mirror=mirror)

    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))
    lines = _infer_lines(g, fg, ctrl)
    line_fg = set()
    for line in lines:
        line_fg |= set(line["line_fg"])
    comps = _extract_offline_components(g, fg, line_fg)
    kept = sum(1 for comp in comps if _line_adjacent(comp, line_fg))
    vanished = sum(1 for comp in comps if not _line_adjacent(comp, line_fg))
    fg_mask = (g == int(fg)).astype(int)
    mirrored = _solve(g, mirror=True)
    plain = _solve(g, mirror=False)
    return {
        "line_fg": int(len(line_fg)),
        "offline_components": int(len(comps)),
        "kept_components": int(kept),
        "vanished_components": int(vanished),
        "is_8_connected": int(is_contiguous(fg_mask, connectivity=8)),
        "is_4_connected": int(is_contiguous(fg_mask, connectivity=4)),
        "mirror_delta": int(np.count_nonzero(mirrored != plain)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(np.array(grid, dtype=int), mirror=True)


def _shift(anchor: tuple[int, int], offsets: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + dr), int(ac + dc)) for dr, dc in offsets}


def _build_top_bottom_fg(
    H: int,
    W: int,
    edge: str,
    dc: int,
    ctrl_col: int,
    vanish_side: str,
) -> set[tuple[int, int]] | None:
    t0 = 5
    line_cells = []
    for t in range(t0, t0 + 4):
        r = t if edge == "top" else H - 1 - t
        c = ctrl_col + dc * t
        line_cells.append((int(r), int(c)))
    if any(not (0 <= r < H and 0 <= c < W) for r, c in line_cells):
        return None

    if dc == 1:
        left_anchor = line_cells[1]
        right_anchor = line_cells[-2]
    else:
        left_anchor = line_cells[-2]
        right_anchor = line_cells[1]

    fg = set(line_cells)
    fg |= _shift(left_anchor, TB_LEFT_KEPT)
    fg |= _shift(right_anchor, TB_RIGHT_KEPT)
    if vanish_side == "L":
        fg |= _shift(left_anchor, TB_LEFT_VANISH)
    else:
        fg |= _shift(right_anchor, TB_RIGHT_VANISH)
    return fg


def _build_left_right_fg(
    H: int,
    W: int,
    edge: str,
    dr: int,
    ctrl_row: int,
    vanish_side: str,
) -> set[tuple[int, int]] | None:
    t0 = 5
    line_cells = []
    for t in range(t0, t0 + 4):
        r = ctrl_row + dr * t
        c = t if edge == "left" else W - 1 - t
        line_cells.append((int(r), int(c)))
    if any(not (0 <= r < H and 0 <= c < W) for r, c in line_cells):
        return None

    if dr == 1:
        top_anchor = line_cells[1]
        bottom_anchor = line_cells[-2]
    else:
        top_anchor = line_cells[-2]
        bottom_anchor = line_cells[1]

    if edge == "left":
        top_kept, top_vanish = LEFT_TOP_KEPT, LEFT_TOP_VANISH
        bottom_kept, bottom_vanish = LEFT_BOTTOM_KEPT, LEFT_BOTTOM_VANISH
    else:
        top_kept, top_vanish = RIGHT_TOP_KEPT, RIGHT_TOP_VANISH
        bottom_kept, bottom_vanish = RIGHT_BOTTOM_KEPT, RIGHT_BOTTOM_VANISH

    fg = set(line_cells)
    fg |= _shift(top_anchor, top_kept)
    fg |= _shift(bottom_anchor, bottom_kept)
    if vanish_side == "T":
        fg |= _shift(top_anchor, top_vanish)
    else:
        fg |= _shift(bottom_anchor, bottom_vanish)
    return fg


def _bbox(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl = 1, 7, 2
    edge = random.choice(["top", "bottom", "left", "right"])

    if edge in ("top", "bottom"):
        dc = random.choice([-1, 1])
        vanish_side = random.choice(["L", "R"])
        for _ in range(80):
            ctrl_col = random.randint(0, W - 1)
            fg_pts = _build_top_bottom_fg(H, W, edge, dc, ctrl_col, vanish_side)
            if fg_pts is None:
                continue
            break
        else:
            return None
        ctrl_pt = (0, ctrl_col) if edge == "top" else (H - 1, ctrl_col)
    else:
        dr = random.choice([-1, 1])
        vanish_side = random.choice(["T", "B"])
        for _ in range(80):
            ctrl_row = random.randint(0, H - 1)
            fg_pts = _build_left_right_fg(H, W, edge, dr, ctrl_row, vanish_side)
            if fg_pts is None:
                continue
            break
        else:
            return None
        ctrl_pt = (ctrl_row, 0) if edge == "left" else (ctrl_row, W - 1)

    if ctrl_pt in fg_pts:
        return None
    if any(not (0 <= int(r) < H and 0 <= int(c) < W) for r, c in fg_pts):
        return None

    r0, c0, r1, c1 = _bbox(fg_pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    density = len(fg_pts) / float(h * w)
    if not (4 <= h <= 12 and 6 <= w <= 12):
        return None
    if not (0.22 <= density <= 0.65):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in fg_pts:
        inp[r, c] = int(fg)
    inp[ctrl_pt] = int(ctrl)

    metrics = _behavior_metrics(inp)
    if metrics["vanished_components"] < 1:
        return None
    if metrics["kept_components"] < 2:
        return None
    if metrics["is_8_connected"] != 1:
        return None
    if metrics["is_4_connected"] != 0:
        return None
    if metrics["mirror_delta"] < 4:
        return None

    _ = main(inp)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
