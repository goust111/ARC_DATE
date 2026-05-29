# Combo variant of 8b7bacbf (index 16)
# combo_spec: D1/1+D9/3+D14/0
# Name: 混合连通+开口框架+4连通连接判定

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# combo_variant_spec:
#   visual_core:
#     - Keep the original seed -> line -> frame vocabulary, but the frame geometry
#       must change from the original rounded rectangle to a full rectangular ring.
#       Reason: once cavity recognition switches to 8-connected background, the
#       old "missing-corners" ring is no longer closed; outside background leaks
#       diagonally through the corners even before considering any side opening.
#     - Valid target frames are therefore full 1px rectangular borders, optionally
#       with one monochrome internal zigzag divider of the same frame color.
#     - The outer border may have a short 1-2 cell opening on one side. That
#       opening is not self-closing: it only counts as sealed if traced line-color
#       pixels sit orthogonally outside every missing border cell.
#     - The solver still traces the whole seeded drawing with 8-connectivity, but
#       cavity extraction inside a frame also uses 8-connected background, while
#       frame/line attachment is judged with strict 4-neighbor contact.
#
# behavior_trace:
#   Original task:
#     seed S touches line L; closed background cavities are found globally with
#     4-connected background; any monochrome boundary touching the traced drawing
#     and 8-neighboring L gets filled with S.
#   D9/3 effect in isolation:
#     an open ring would disappear from the 4-connected cavity search, so the
#     solver must infer a near-closed frame and patch the gap conceptually.
#   D1/1 effect in isolation:
#     if a same-color diagonal divider splits the interior, 4-connected background
#     sees two chambers, while 8-connected background merges them through diagonal
#     bg contacts and fills a larger region.
#   D14/0 effect in isolation:
#     a frame that only corner-touches the line is still part of the traced
#     8-component, but it should no longer fill because no orthogonal contact
#     exists between cavity boundary and traced line pixels.
#   This combo:
#     1. infer a rectangular frame component, with optional internal zigzag wall;
#     2. if the frame has a side opening, accept it only when traced line pixels
#        orthogonally cover the missing side cells;
#     3. inside that virtually sealed frame, extract background components using
#        8-connectivity, so a zigzag divider does not fully split the cavity;
#     4. fill only those cavity components whose boundary has traced line-color
#        contact in the 4-neighbor sense.
#   Concrete miniature:
#     a left-attached open frame has one missing side cell and a line strip just
#     outside that opening; inside, a vertical zigzag wall creates two 4-connected
#     lobes but one 8-connected cavity. The combo fills the whole interior. A
#     second closed frame touching the same line only by one corner stays empty.
#
# complexity_calibration:
#   - Each generated sample contains at least one open, fillable frame per seed so
#     the D9 change is visible.
#   - At least one of those fillable frames per sample contains an internal zigzag
#     divider, guaranteeing a measurable D1 delta versus a 4-connected cavity
#     solver.
#   - Every sample also contains a diagonally attached closed distractor frame,
#     guaranteeing a measurable D14 delta versus an 8-neighbor contact solver.
#   - Divider orientation is restricted to {vertical, horizontal}, tied to the
#     side from which the line approaches, so the visual grammar stays learnable.
#
# old_main_assumption_audit:
#   - Original `bg_comps = _components_mask(g == bg, conn8=False)` assumed the
#     intended cavities already exist as globally closed background regions. Open
#     frames break that immediately, so cavity search must move from global bg to
#     per-frame reconstruction.
#   - Original rounded-rectangle ring without corners relied on 4-connected bg to
#     behave as closed. Under 8-connected bg this leaks, so generator and solver
#     both switch to full rectangular borders.
#   - Original `len(bcols) == 1` is still useful, but "boundary" is no longer only
#     the visible colored ring: the missing side cells may become virtual boundary
#     cells if the traced line orthogonally seals them.
#   - Original 8-neighbor line-contact test is wrong for this combo because it
#     would accept corner-only attachments. Replace it with a 4-neighbor test.
#
# placement_legality:
#   - Frames keep a 1-cell pad from the canvas edge.
#   - Ring cells, divider cells, and cavity cells must be empty before placement.
#   - Fillable open frames are attached from one side of the path; the generator
#     reserves a short exterior line strip so at least one visible border cell has
#     orthogonal line contact and every missing border cell has orthogonal support.
#   - Diagonal-only distractors are placed so the frame has at least one 8-neighbor
#     line touch but zero 4-neighbor line touches anywhere on the frame.
#   - Cavity cells are checked against line-color 8-neighbors so the line never
#     directly intrudes into the cavity interior.
#
# visual_observability:
#   - Humans can see open frames that still get filled, so the "opening tolerated
#     if the line effectively seals it" rule is exposed directly.
#   - They can also see one frame whose interior fills across a same-color zigzag
#     barrier, which reveals the 8-connected cavity rule.
#   - Finally, they can compare against a corner-touching frame that remains empty,
#     exposing the strict 4-neighbor connection predicate.
#
# cleanliness_and_inference:
#   - The generator keeps objects separated by halos and uses a small color budget:
#     background, one seed color per network, one line color per network, and a
#     small reusable pool of frame colors.
#   - No color-id ordering is used; all decisions are geometric and visible.
#
# new_main_assumption_audit:
#   - `main()` assumes target frames are full rectangles with optional 1-side
#     opening and optional one-piece zigzag divider of the same color. The
#     generator constructs exactly that family.
#   - `main()` assumes any accepted opening is orthogonally sealed by traced line
#     pixels. `_try_attach_frame(..., mode="ortho_open")` enforces that.
#   - `main()` assumes any diagonal-only distractor has zero 4-neighbor line
#     contact, even though it is in the same traced 8-component. `_try_attach_frame`
#     in `diag_closed` mode enforces that.


GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
BG_COLORS = [0, 5]

SEED_COUNT_CHOICES = [1, 1, 2]
MIN_LINE_LEN = 7
LINE_LEN_RANGE = (10, 34)

FRAME_COLORS_MAX = 4
PARTITION_H_CHOICES = [4, 5, 6]
PARTITION_W_CHOICES = [5, 6, 7]
PLAIN_H_CHOICES = [3, 4, 5]
PLAIN_W_CHOICES = [3, 4, 5, 6]

FRAME_PAD = 1
ATTACH_TRIES = 120
DIAG_TRIES = 120
GEN_RETRIES = 700
EXTRA_FILLABLE_MAX = 1

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("left", "right", "top", "bottom")
DIAG_CORNERS = ("TL", "TR", "BL", "BR")


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) != bg:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _rect_ring_coords(r0: int, c0: int, ch: int, cw: int) -> set[tuple[int, int]]:
    ring = set()
    for rr in range(r0, r0 + ch + 2):
        for cc in range(c0, c0 + cw + 2):
            if rr in (r0, r0 + ch + 1) or cc in (c0, c0 + cw + 1):
                ring.add((rr, cc))
    return ring


def _side_cells(r0: int, c0: int, ch: int, cw: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(r0, c0 + i) for i in range(1, cw + 1)],
        "bottom": [(r0 + ch + 1, c0 + i) for i in range(1, cw + 1)],
        "left": [(r0 + i, c0) for i in range(1, ch + 1)],
        "right": [(r0 + i, c0 + cw + 1) for i in range(1, ch + 1)],
    }


def _side_exterior_cells(r0: int, c0: int, ch: int, cw: int, side: str, idxs: list[int]) -> list[tuple[int, int]]:
    cells = []
    for idx in idxs:
        if side == "left":
            cells.append((r0 + idx, c0 - 1))
        elif side == "right":
            cells.append((r0 + idx, c0 + cw + 2))
        elif side == "top":
            cells.append((r0 - 1, c0 + idx))
        else:
            cells.append((r0 + ch + 2, c0 + idx))
    return cells


def _side_bbox(anchor: tuple[int, int], side: str, ch: int, cw: int, idx: int) -> tuple[int, int]:
    if side == "left":
        return anchor[0] - idx, anchor[1] + 1
    if side == "right":
        return anchor[0] - idx, anchor[1] - (cw + 2)
    if side == "top":
        return anchor[0] + 1, anchor[1] - idx
    if side == "bottom":
        return anchor[0] - (ch + 2), anchor[1] - idx
    raise ValueError(side)


def _diag_bbox(anchor: tuple[int, int], corner: str, ch: int, cw: int) -> tuple[int, int]:
    if corner == "TL":
        return anchor[0] + 1, anchor[1] + 1
    if corner == "TR":
        return anchor[0] + 1, anchor[1] - cw - 2
    if corner == "BL":
        return anchor[0] - ch - 2, anchor[1] + 1
    if corner == "BR":
        return anchor[0] - ch - 2, anchor[1] - cw - 2
    raise ValueError(corner)


def _fits_bbox(h: int, w: int, r0: int, c0: int, ch: int, cw: int) -> bool:
    return r0 >= FRAME_PAD and c0 >= FRAME_PAD and r0 + ch + 2 <= h - FRAME_PAD and c0 + cw + 2 <= w - FRAME_PAD


def _divider_kind_for_side(side: str) -> str:
    return "vertical" if side in ("left", "right") else "horizontal"


def _divider_coords(r0: int, c0: int, ch: int, cw: int, divider_kind: str | None) -> set[tuple[int, int]]:
    if divider_kind is None:
        return set()
    divider = set()
    if divider_kind == "vertical":
        base_c = c0 + 1 + max(1, cw // 2 - 1)
        for i in range(ch):
            divider.add((r0 + 1 + i, base_c + (i % 2)))
        return divider
    if divider_kind == "horizontal":
        base_r = r0 + 1 + max(1, ch // 2 - 1)
        for i in range(cw):
            divider.add((base_r + (i % 2), c0 + 1 + i))
        return divider
    raise ValueError(divider_kind)


def _cavity_cells(r0: int, c0: int, ch: int, cw: int, divider_kind: str | None) -> list[tuple[int, int]]:
    divider = _divider_coords(r0, c0, ch, cw, divider_kind)
    cells = []
    for rr in range(r0 + 1, r0 + ch + 1):
        for cc in range(c0 + 1, c0 + cw + 1):
            if (rr, cc) not in divider:
                cells.append((rr, cc))
    return cells


def _frame_shape(
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    divider_kind: str | None,
    gap_side: str | None = None,
    gap_idxs: list[int] | None = None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]]:
    ring = _rect_ring_coords(r0, c0, ch, cw)
    divider = _divider_coords(r0, c0, ch, cw, divider_kind)
    missing: set[tuple[int, int]] = set()
    if gap_side is not None and gap_idxs:
        side_map = _side_cells(r0, c0, ch, cw)
        for idx in gap_idxs:
            missing.add(side_map[gap_side][idx - 1])
    visible = (ring - missing) | divider
    cavity = _cavity_cells(r0, c0, ch, cw, divider_kind)
    return visible, ring, missing, cavity


def _touch_mode(canvas: np.ndarray, pts: set[tuple[int, int]], line_col: int) -> tuple[bool, bool]:
    h, w = canvas.shape
    has8 = False
    has4 = False
    for r, c in pts:
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == int(line_col):
                has8 = True
                break
        for nr, nc in neighbors4(r, c):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == int(line_col):
                has4 = True
                break
    return has8, has4


def _frame_candidate_from_component(
    g: np.ndarray,
    pts: list[tuple[int, int]],
    bg: int,
) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], str | None, str | None, int, int, int, int] | None:
    pts_set = set(pts)
    if len(pts_set) < 8:
        return None

    rows = [r for r, _ in pts_set]
    cols = [c for _, c in pts_set]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    ch = r1 - r0 - 1
    cw = c1 - c0 - 1
    if ch < 3 or cw < 3:
        return None

    ring = _rect_ring_coords(r0, c0, ch, cw)
    side_map = _side_cells(r0, c0, ch, cw)

    for divider_kind in (None, "vertical", "horizontal"):
        divider = _divider_coords(r0, c0, ch, cw, divider_kind)
        expected = ring | divider
        if pts_set - ring != divider:
            continue
        if not pts_set <= expected:
            continue

        missing = ring - pts_set
        gap_side: str | None = None
        if len(missing) > 2:
            continue
        if missing:
            ok = False
            for side_name, side_pts in side_map.items():
                idxs = [i for i, p in enumerate(side_pts) if p in missing]
                if len(idxs) != len(missing):
                    continue
                if idxs[-1] - idxs[0] + 1 != len(idxs):
                    continue
                ok = True
                gap_side = side_name
                break
            if not ok:
                continue

        for r, c in missing:
            if int(g[r, c]) != int(bg):
                return None
        for r, c in _cavity_cells(r0, c0, ch, cw, divider_kind):
            if int(g[r, c]) != int(bg):
                return None

        frame_col = int(g[next(iter(pts_set))])
        return frame_col, pts_set, ring, missing, divider_kind, gap_side, r0, c0, ch, cw
    return None


def _all_frame_candidates(
    g: np.ndarray,
    bg: int,
) -> list[tuple[int, set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], str | None, str | None, int, int, int, int]]:
    cands = []
    colors = sorted(set(map(int, g.flatten())) - {int(bg)})
    for col in colors:
        for comp in _components_mask(g == int(col), conn8=True):
            found = _frame_candidate_from_component(g, comp, bg)
            if found is not None:
                cands.append(found)
    return cands


def _outer_cell_for_gap(cell: tuple[int, int], gap_side: str) -> tuple[int, int]:
    r, c = cell
    if gap_side == "top":
        return r - 1, c
    if gap_side == "bottom":
        return r + 1, c
    if gap_side == "left":
        return r, c - 1
    if gap_side == "right":
        return r, c + 1
    raise ValueError(gap_side)


def _gap_completed_by_line(
    missing: set[tuple[int, int]],
    gap_side: str | None,
    traced_line: set[tuple[int, int]],
) -> bool:
    if not missing:
        return True
    if gap_side is None:
        return False
    for cell in missing:
        if _outer_cell_for_gap(cell, gap_side) not in traced_line:
            return False
    return True


def _candidate_cavities(
    g: np.ndarray,
    bg: int,
    ring: set[tuple[int, int]],
    missing: set[tuple[int, int]],
    divider_kind: str | None,
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    *,
    bg_conn8: bool,
    close_gap: bool,
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = g.shape
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r0 + ch + 2)
    pc1 = min(w - 1, c0 + cw + 2)

    divider = _divider_coords(r0, c0, ch, cw, divider_kind)
    passable = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
    for rr in range(pr0, pr1 + 1):
        for cc in range(pc0, pc1 + 1):
            if int(g[rr, cc]) == int(bg):
                passable[rr - pr0, cc - pc0] = True

    blocked = set(ring) | divider
    if close_gap:
        blocked |= missing
    for rr, cc in blocked:
        if pr0 <= rr <= pr1 and pc0 <= cc <= pc1:
            passable[rr - pr0, cc - pc0] = False

    boundary_neighbors = neighbors8 if bg_conn8 else neighbors4
    cavities = []
    for comp in _components_mask(passable, conn8=bg_conn8):
        region = {(pr0 + rr, pc0 + cc) for rr, cc in comp}
        if any(rr in (pr0, pr1) or cc in (pc0, pc1) for rr, cc in region):
            continue
        boundary = set()
        for rr, cc in region:
            for nr, nc in boundary_neighbors(rr, cc):
                if not (pr0 <= nr <= pr1 and pc0 <= nc <= pc1):
                    continue
                if (nr, nc) in region:
                    continue
                if int(g[nr, nc]) != int(bg) or (close_gap and (nr, nc) in missing):
                    boundary.add((nr, nc))
        if region:
            cavities.append((region, boundary))
    return cavities


def _boundary_has_line_contact(
    boundary: set[tuple[int, int]],
    traced_line: set[tuple[int, int]],
    *,
    line_contact4: bool,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    adj = neighbors4 if line_contact4 else neighbors8
    for r, c in boundary:
        for nr, nc in adj(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in traced_line:
                return True
    return False


def _solve_with_params(
    grid: np.ndarray,
    *,
    bg_conn8: bool,
    line_contact4: bool,
    allow_open_frames: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()
    frame_candidates = _all_frame_candidates(g, bg)

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
        line_col = Counter(neigh_cols).most_common(1)[0][0]
        traced = _nonbg_component8_from(g, (sr, sc), bg)
        traced_line = {p for p in traced if int(g[p]) == int(line_col)}

        for _frame_col, visible_pts, ring, missing, divider_kind, gap_side, r0, c0, ch, cw in frame_candidates:
            if not visible_pts <= traced:
                continue
            if missing and not allow_open_frames:
                continue
            close_gap = _gap_completed_by_line(missing, gap_side, traced_line)
            if missing and not close_gap:
                continue

            for cavity, boundary in _candidate_cavities(
                g,
                bg,
                ring,
                missing,
                divider_kind,
                r0,
                c0,
                ch,
                cw,
                bg_conn8=bg_conn8,
                close_gap=close_gap,
            ):
                if not _boundary_has_line_contact(boundary, traced_line, line_contact4=line_contact4, shape=g.shape):
                    continue
                for r, c in cavity:
                    if int(out[r, c]) == int(bg):
                        out[r, c] = seed_col
    return out


def _delta_counts(grid: np.ndarray) -> dict[str, int]:
    combo = _solve_with_params(grid, bg_conn8=True, line_contact4=True, allow_open_frames=True)
    bg4 = _solve_with_params(grid, bg_conn8=False, line_contact4=True, allow_open_frames=True)
    closed = _solve_with_params(grid, bg_conn8=True, line_contact4=True, allow_open_frames=False)
    contact8 = _solve_with_params(grid, bg_conn8=True, line_contact4=False, allow_open_frames=True)
    return {
        "delta_bg4": int(np.count_nonzero(combo != bg4)),
        "delta_closed": int(np.count_nonzero(combo != closed)),
        "delta_contact8": int(np.count_nonzero(combo != contact8)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill seed-owned frame cavities under mixed connectivity, near-closed frames, and strict orthogonal line attachment.

    Input structure:
    - The background is the most frequent color.
    - One or two unique single-pixel seed colors lie on the border, each touching a non-background line.
    - Each seeded drawing is traced as one 8-connected non-background component.
    - Target frames are full rectangular borders of one frame color, optionally with one same-color zigzag divider inside.
    - A frame may miss a 1-2 cell segment on one side; some other frames may only touch the line diagonally as distractors.

    Rule:
    - From each seed, trace the whole reachable non-background drawing with 8-connectivity and recover the line color from the seed's orthogonal neighbors.
    - Detect rectangular frame components. If a frame has a short side opening, treat it as closed only when traced line-color pixels sit orthogonally outside every missing border cell.
    - Inside each accepted frame, compute enclosed background regions using 8-connected background, not 4-connected background.
    - Flood-fill a cavity with the seed color only if that cavity's boundary has at least one traced line-color contact in the 4-neighbor sense.

    Key insight:
    - The important object is not a pre-existing global background cavity. It is a frame-shaped structure whose closure may depend on nearby path pixels.
    - After patching a valid opening, 8-connected background merges diagonally adjacent interior pockets that a 4-connected solver would split.
    - Corner-touching frames are still in the traced drawing, but they stay empty because the final ownership test is strictly orthogonal.
    """
    return _solve_with_params(grid, bg_conn8=True, line_contact4=True, allow_open_frames=True)


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    path_set = {start}
    cur = start
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            weight = DIAG_WEIGHT_ALIGNED if (ddr == dr_bias or ddc == dc_bias) else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        rng.shuffle(dirs)

        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or int(grid[nr, nc]) != -1:
                continue
            if any((ar, ac) != cur and (ar, ac) in path_set for ar, ac in neighbors8(nr, nc)):
                continue
            if (nr in (0, h - 1) or nc in (0, w - 1)) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            grid[nr, nc] = int(line_col)
            placed = True
            break
        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break
    return path


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        return (0, 0), rng.choice([(0, 1), (1, 0)])
    if corner_kind == "TR":
        return (0, w - 1), rng.choice([(0, w - 2), (1, w - 1)])
    if corner_kind == "BL":
        return (h - 1, 0), rng.choice([(h - 1, 1), (h - 2, 0)])
    if corner_kind == "BR":
        return (h - 1, w - 1), rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    raise ValueError(corner_kind)


def _near_foreign_color(
    canvas: np.ndarray,
    pts: set[tuple[int, int]],
    allowed: set[int],
) -> bool:
    h, w = canvas.shape
    for r, c in pts:
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in pts:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v not in allowed:
                return True
    return False


def _cavity_touches_line(
    canvas: np.ndarray,
    cavity: list[tuple[int, int]],
    line_col: int,
    extra_line: set[tuple[int, int]],
) -> bool:
    h, w = canvas.shape
    for r, c in cavity:
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in extra_line or int(canvas[nr, nc]) == int(line_col):
                return True
    return False


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    *,
    partitioned: bool,
    mode: str,
    max_tries: int,
) -> bool:
    h, w = canvas.shape
    h_choices = PARTITION_H_CHOICES if partitioned else PLAIN_H_CHOICES
    w_choices = PARTITION_W_CHOICES if partitioned else PLAIN_W_CHOICES

    for _ in range(max_tries):
        ch = rng.choice(h_choices)
        cw = rng.choice(w_choices)
        anchor = rng.choice(line_path)

        if mode == "diag_closed":
            corner = rng.choice(DIAG_CORNERS)
            r0, c0 = _diag_bbox(anchor, corner, ch, cw)
            if not _fits_bbox(h, w, r0, c0, ch, cw):
                continue
            visible, _ring, _missing, cavity = _frame_shape(r0, c0, ch, cw, None, None, None)
            if any(int(canvas[r, c]) != -1 for r, c in visible):
                continue
            if any(int(canvas[r, c]) != -1 for r, c in cavity):
                continue
            if _near_foreign_color(canvas, visible, {int(line_col)}):
                continue
            has8, has4 = _touch_mode(canvas, visible, line_col)
            if not has8 or has4:
                continue
            if _cavity_touches_line(canvas, cavity, line_col, set()):
                continue
            for r, c in visible:
                canvas[r, c] = int(frame_col)
            return True

        side = rng.choice(SIDES)
        divider_kind = _divider_kind_for_side(side) if partitioned else None
        gap_len = rng.choice([1, 2])
        side_len = ch if side in ("left", "right") else cw
        if side_len < gap_len + 3:
            continue
        first_idx = rng.randint(2, side_len - gap_len - 1)
        support_idxs = list(range(first_idx, first_idx + gap_len + 1))
        gap_idxs = support_idxs[1:]

        r0, c0 = _side_bbox(anchor, side, ch, cw, support_idxs[0])
        if not _fits_bbox(h, w, r0, c0, ch, cw):
            continue

        visible, _ring, _missing, cavity = _frame_shape(r0, c0, ch, cw, divider_kind, side, gap_idxs)
        support_cells = _side_exterior_cells(r0, c0, ch, cw, side, support_idxs)
        if support_cells[0] != anchor:
            continue

        extra_line = set(support_cells[1:])
        all_frame_pts = set(visible)
        if any(int(canvas[r, c]) != -1 for r, c in all_frame_pts):
            continue
        if any(int(canvas[r, c]) != -1 for r, c in cavity):
            continue
        if any((not _in_bounds(h, w, r, c)) or (int(canvas[r, c]) not in (-1, int(line_col))) for r, c in extra_line):
            continue
        if _near_foreign_color(canvas, all_frame_pts | extra_line, {int(line_col)}):
            continue
        if _cavity_touches_line(canvas, cavity, line_col, extra_line):
            continue

        future_line = {p for p in line_path if int(canvas[p]) == int(line_col)} | {anchor} | extra_line
        for side_cell in _side_exterior_cells(r0, c0, ch, cw, side, gap_idxs):
            if side_cell not in future_line:
                break
        else:
            for r, c in visible:
                canvas[r, c] = int(frame_col)
            for r, c in extra_line:
                canvas[r, c] = int(line_col)
                if (r, c) not in line_path:
                    line_path.append((r, c))
            return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != int(bg)]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_CHOICES)
    if num_seeds * 2 + 1 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    frame_color_count = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors = [palette.pop() for _ in range(frame_color_count)]

    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    corners = corners[:num_seeds]

    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]] = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners[i], h, w, rng)
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
            return None
        canvas[seed_rc] = int(seed_colors[i])
        canvas[first_rc] = int(line_colors[i])
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w + 6))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    for _seed_col, _seed_rc, line_col, path in lines_info:
        fc = rng.choice(frame_colors)
        if not _try_attach_frame(
            canvas,
            path,
            line_col,
            fc,
            rng,
            partitioned=True,
            mode="ortho_open",
            max_tries=ATTACH_TRIES,
        ):
            return None
        for _ in range(rng.randint(0, EXTRA_FILLABLE_MAX)):
            fc = rng.choice(frame_colors)
            _try_attach_frame(
                canvas,
                path,
                line_col,
                fc,
                rng,
                partitioned=False,
                mode="ortho_open",
                max_tries=ATTACH_TRIES,
            )

    diag_host = rng.randrange(num_seeds)
    if not _try_attach_frame(
        canvas,
        lines_info[diag_host][3],
        lines_info[diag_host][2],
        rng.choice(frame_colors),
        rng,
        partitioned=False,
        mode="diag_closed",
        max_tries=DIAG_TRIES,
    ):
        return None

    grid = np.where(canvas == -1, bg, canvas).astype(int)
    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != int(bg):
        return None
    for seed_col, _seed_rc, _line_col, _path in lines_info:
        if int((grid == int(seed_col)).sum()) != 1:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    deltas = _delta_counts(grid)
    if deltas["delta_bg4"] <= 0 or deltas["delta_closed"] <= 0 or deltas["delta_contact8"] <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
