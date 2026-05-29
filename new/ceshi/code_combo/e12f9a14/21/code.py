# Combo variant of e12f9a14 (index 21)
# combo_spec: D4/0+D8/0+D9/0
# Name: 单像素源+始终单像素宽+无间隙

# Design analysis for combo D4/0 + D8/0 + D9/0
# ------------------------------------------------
# a) Concrete behavior tracking
#    - Original task geometry:
#      each source was a 2x2 monochrome core inside a 4x4 frame; a border gap fired one ray.
#      45-degree collisions could produce a visibly thicker 2-track continuation, and 90-degree
#      collisions left an empty meeting cell.
#    - Fused geometry:
#      each source is now a single colored center pixel inside a 3x3 frame. The gap is still on
#      the frame border, so side gaps emit cardinal rays and corner gaps emit diagonal rays.
#      The important change is not the gap direction but the visual minimalism:
#      every visible stroke must stay 1 pixel wide and every collision cell must stay filled.
#    - Small example:
#      take one east-going beam and one south-going beam aimed at the same cell P.
#      Original output would leave P blank, then continue as a two-line SE redirection.
#      In this combo, P is painted and only one SE continuation survives, so the result is a thin
#      Y-junction instead of a gap plus a double track.
#      For a 45-degree case, an east beam meeting an NE beam at P used to become a two-track zigzag.
#      Here both incoming thin rays meet at P and continue as a single thin zigzag.
#
# b) Complexity calibration
#    - If we kept random multi-gap frames and allowed all original crossing modes, the fused rules
#      would become visually ambiguous because width and gap cues are intentionally removed.
#    - To keep the variant inferable, the generator uses one gap per frame and constructs sparse
#      pairwise collisions only. This preserves the core angle-dependent rule while avoiding clutter.
#
# c) Implicit-assumption audit against the original solver
#    - `_find_blocks_2x2` and the 4x4 frame inference no longer apply; the solver must detect a
#      1-cell center with an 8-cell 3x3 ring containing only frame color or bg.
#    - The original collision code assumed width could branch into lockstep partners.
#      D8/0 breaks that assumption, so every collision handler now merges to a single continuing beam.
#    - The original `blocked` gap mechanism for 90-degree collisions assumed the intersection cell
#      must become background. D9/0 breaks that assumption; collision cells stay painted.
#    - The original generator assumed each frame color was paired with a unique inner color.
#      This combo intentionally allows two same-colored centers to collide, so the merged thin path
#      does not introduce a hidden “which color wins at the junction?” rule.
#
# d) Placement legality
#    - Frames are 3x3, fully in-bounds, and separated by at least two cells between bounding boxes.
#    - Planned beam traces are reserved before stamping objects, so beams never pass through another
#      frame and independent collision gadgets never overlap.
#
# e) Visual observability
#    - A human sees only three object types: bg, one shared frame color, and isolated 1-pixel centers.
#      Output changes are entirely spatial: thin straight rays, thin zigzags, and filled junctions.
#      No color-order or numeric-color trick is required.
#
# f) Visual cleanliness
#    - The generator avoids stacking multiple collision gadgets into the same area.
#      Each gadget is a sparse two-ray motif, so the resulting network remains readable.
#
# g) Change inferability
#    - There is no opaque color remapping. A pair keeps one beam color from input to output.
#      The observable variable is only the approach angle at the collision: 90 degrees -> thin bisector,
#      45 degrees -> thin zigzag, both with a painted meeting point.

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste


FRAME_SIZE = 3
FRAME_BBOX_GAP = 2
GEN_RETRIES = 400
MIN_PRE_STEPS = 2
MIN_POST_STEPS = 3
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}

_RING_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, 1), (1, 1), (1, 0),
    (1, -1), (0, -1),
]
_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}
_GAP_OFFSET = {
    (-1, -1): (0, 0),
    (-1, 0): (0, 1),
    (-1, 1): (0, 2),
    (0, 1): (1, 2),
    (1, 1): (2, 2),
    (1, 0): (2, 1),
    (1, -1): (2, 0),
    (0, -1): (1, 0),
}
_PAIR_DIRS_90 = [
    ((0, 1), (1, 0)),
    ((0, 1), (-1, 0)),
    ((0, -1), (1, 0)),
    ((0, -1), (-1, 0)),
]
_PAIR_DIRS_45 = [
    ((0, 1), (-1, 1)),
    ((0, 1), (1, 1)),
    ((0, -1), (-1, -1)),
    ((0, -1), (1, -1)),
    ((1, 0), (1, 1)),
    ((1, 0), (1, -1)),
    ((-1, 0), (-1, 1)),
    ((-1, 0), (-1, -1)),
]


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0 and dc == 0:
        return (0, 0)
    if dr != 0 and dc != 0:
        return (int(np.sign(dr)), int(np.sign(dc)))
    if dr != 0:
        return (int(np.sign(dr)), 0)
    return (0, int(np.sign(dc)))


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = _DIR_TO_I[a]
    ib = _DIR_TO_I[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


def _frame_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(FRAME_SIZE) for dc in range(FRAME_SIZE)}


def _bbox_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + FRAME_SIZE - 1, ac0 + FRAME_SIZE - 1
    br1, bc1 = br0 + FRAME_SIZE - 1, bc0 + FRAME_SIZE - 1
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _trace_straight(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    cells = [start]
    r, c = start
    dr, dc = direction
    for _ in range(steps):
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _trace_pattern_to_edge(
    start: tuple[int, int],
    pattern: tuple[tuple[int, int], ...],
    n: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    phase = 0
    r, c = start
    while True:
        dr, dc = pattern[phase]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n and 0 <= nc < n):
            break
        cells.append((nr, nc))
        r, c = nr, nc
        phase = (phase + 1) % len(pattern)
    return cells


def _path_set(paths: list[list[tuple[int, int]]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for path in paths:
        out.update(path)
    return out


def _find_single_pixel_frames(g: np.ndarray, bg: int) -> list[tuple[int, int, int, list[tuple[int, int]]]]:
    h, w = g.shape
    found: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(g[r, c])
            if center == bg:
                continue
            ring_vals = [int(g[r + dr, c + dc]) for dr, dc in _RING_OFFSETS]
            frame_vals = sorted({v for v in ring_vals if v != bg})
            if len(frame_vals) != 1:
                continue
            frame_color = frame_vals[0]
            if frame_color == center:
                continue
            frame_count = sum(v == frame_color for v in ring_vals)
            gap_count = sum(v == bg for v in ring_vals)
            if frame_count < 5 or not (1 <= gap_count <= 3):
                continue
            if frame_count + gap_count != 8:
                continue
            gaps = [(r + dr, c + dc) for dr, dc in _RING_OFFSETS if int(g[r + dr, c + dc]) == bg]
            found.append((center, r, c, gaps))
    return found


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def reset_pattern(self, pattern: tuple[tuple[int, int], ...]) -> None:
        self.pattern = pattern
        self.phase = 0
        self.active = True

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _run_simulation(grid: np.ndarray, collect_metrics: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w, "expected a square grid"
    bg = get_bg(g)

    out = g.copy()
    sources = _find_single_pixel_frames(g, bg)
    assert sources, "expected at least one framed single-pixel source"

    beams: dict[int, _Beam] = {}
    next_bid = 0
    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}
    metrics = None
    if collect_metrics:
        metrics = {
            "source_count": len(sources),
            "collisions_45": 0,
            "collisions_90": 0,
            "collision_cells": [],
        }

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        r, c = cell
        out[r, c] = beams[bid].color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    def unpaint_cell(cell: tuple[int, int]) -> None:
        if cell not in cell_owner:
            return
        out[cell[0], cell[1]] = bg
        cell_owner.pop(cell, None)
        cell_enter.pop(cell, None)
        cell_index.pop(cell, None)

    def truncate_beam(bid: int, keep_len: int) -> None:
        b = beams[bid]
        if keep_len >= len(b.path):
            return
        assert keep_len > 0, "expected at least one kept cell"
        for cell in b.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]

    def ensure_head_at_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        owner = cell_owner.get(cell)
        if owner == bid:
            idx = cell_index[cell]
            truncate_beam(bid, idx + 1)
            beams[bid].pos = cell
            return
        if owner is not None:
            idx = cell_index[cell]
            truncate_beam(owner, idx + 1)
            unpaint_cell(cell)
        b = beams[bid]
        b.path.append(cell)
        b.enter_dirs.append(enter_dir)
        b.pos = cell
        paint_cell(bid, cell, enter_dir)

    def spawn_beam(color: int, pos: tuple[int, int], direction: tuple[int, int]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected a frame-gap seed"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, (direction,))
        paint_cell(bid, pos, direction)

    for color, cr, cc, gaps in sources:
        for gr, gc in gaps:
            d = _normalize_dir(gr - cr, gc - cc)
            assert d != (0, 0), "gap must face outward from the center pixel"
            spawn_beam(color, (gr, gc), d)

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_bg_input(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg

    def register_collision(cell: tuple[int, int], diff: int) -> None:
        if metrics is None:
            return
        if diff == 1:
            metrics["collisions_45"] += 1
        else:
            metrics["collisions_90"] += 1
        metrics["collision_cells"].append((cell, diff))

    def resolve_cell_collision(
        cell: tuple[int, int],
        a: int,
        b: int,
        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]],
    ) -> None:
        da = prop[a][1]
        db = prop[b][1] if b in prop and prop[b][0] == cell else cell_enter[cell]
        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected a 45- or 90-degree collision"
        register_collision(cell, diff)

        if diff == 2:
            winner, loser = (a, b) if a < b else (b, a)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            assert nd != (0, 0), "expected a non-zero bisector direction"
            ensure_head_at_cell(winner, cell, prop[winner][1])
            beams[winner].reset_pattern((nd,))
            beams[loser].active = False
            return

        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1
        assert is_card_a != is_card_b, "45-degree collision must be cardinal+diagonal"
        if is_card_a:
            winner, loser = a, b
            card_dir, diag_dir = da, db
        else:
            winner, loser = b, a
            card_dir, diag_dir = db, da
        ensure_head_at_cell(winner, cell, card_dir)
        beams[winner].reset_pattern((diag_dir, card_dir))
        beams[loser].active = False

    max_iters = h * w * 40
    for _ in range(max_iters):
        active = [bid for bid, beam in beams.items() if beam.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            beam = beams[bid]
            d = beam.next_dir()
            nxt = (beam.pos[0] + d[0], beam.pos[1] + d[1])
            if not in_bounds(nxt) or not is_bg_input(nxt):
                beam.active = False
                continue
            prop[bid] = (nxt, d)

        if not prop:
            break

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)

        events: list[tuple[int, int, int, int]] = []
        for cell, bidders in cell_to_bids.items():
            owner = cell_owner.get(cell)
            bidders = sorted(bidders)
            a = bidders[0]
            if owner is not None and owner != a:
                events.append((cell[0], cell[1], a, owner))
                continue
            if len(bidders) >= 2:
                events.append((cell[0], cell[1], bidders[0], bidders[1]))

        if events:
            events.sort()
            r, c, a, b = events[0]
            resolve_cell_collision((r, c), a, b, prop)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected a collision-free target"
            beam = beams[bid]
            beam.path.append(nxt)
            beam.enter_dirs.append(d)
            beam.pos = nxt
            paint_cell(bid, nxt, d)
            beam.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    if collect_metrics:
        return out, metrics
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit thin rays from single-pixel sources framed by 3x3 walls and merge them angle-dependently.

    Input structure:
    - The background is the most common color.
    - Each source is a single colored center pixel inside a 3x3 frame of one shared wall color.
    - Every frame has 1-3 missing border cells; each missing cell is a ray seed.

    Rule:
    - A side gap emits a cardinal ray and a corner gap emits a diagonal ray, always in the outward direction from the center.
    - Rays paint continuously until they leave the grid, hit a non-background input cell, or collide.
    - 90-degree collisions keep the meeting cell painted and redirect to the bisector as one thin straight ray.
    - 45-degree collisions keep the meeting cell painted and continue as one thin zigzag that alternates diagonal then cardinal steps.

    Key insight:
    - The original width/gap cues are removed, so the only evidence is geometric: all outputs are 1-pixel lines, and collision type is inferred from whether the merged continuation is straight-bisector or zigzag.
    """
    return _run_simulation(grid, collect_metrics=False)


def _pick_colors(rng: np.random.Generator, pair_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    colors = list(rng.choice(palette, size=pair_count, replace=False))
    return bg, frame_color, colors


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    center_color: int,
    gap_dir: tuple[int, int],
) -> None:
    patch = np.full((FRAME_SIZE, FRAME_SIZE), frame_color, dtype=int)
    patch[1, 1] = center_color
    gr, gc = _GAP_OFFSET[gap_dir]
    patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _plan_pair(
    n: int,
    kind: str,
    existing_frames: list[tuple[int, int]],
    occupied_frame_cells: set[tuple[int, int]],
    reserved_beam_cells: set[tuple[int, int]],
    rng: np.random.Generator,
):
    dir_choices = _PAIR_DIRS_90 if kind == "90" else _PAIR_DIRS_45
    dirs = dir_choices[int(rng.integers(len(dir_choices)))]

    for _ in range(300):
        t = int(rng.integers(MIN_PRE_STEPS, 6))
        collision = (int(rng.integers(2, n - 2)), int(rng.integers(2, n - 2)))
        seeds = []
        frame_pos = []
        for d in dirs:
            sr = collision[0] - t * d[0]
            sc = collision[1] - t * d[1]
            gr, gc = _GAP_OFFSET[d]
            r0, c0 = sr - gr, sc - gc
            if not (0 <= r0 <= n - FRAME_SIZE and 0 <= c0 <= n - FRAME_SIZE):
                break
            seeds.append((sr, sc))
            frame_pos.append((r0, c0))
        else:
            if len(set(frame_pos)) != 2:
                continue
            if any(not _bbox_separated(frame_pos[i], frame_pos[j]) for i in range(2) for j in range(i + 1, 2)):
                continue
            if any(not _bbox_separated(pos, other) for pos in frame_pos for other in existing_frames):
                continue

            own_frame_cells = _path_set([list(_frame_cells(r0, c0)) for r0, c0 in frame_pos])
            if own_frame_cells & occupied_frame_cells:
                continue
            if own_frame_cells & reserved_beam_cells:
                continue

            pre_paths = [_trace_straight(seed, d, t) for seed, d in zip(seeds, dirs)]
            pre_sets = [set(path) for path in pre_paths]
            if pre_sets[0] & pre_sets[1] != {collision}:
                continue

            if kind == "90":
                pattern = (_normalize_dir(dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]),)
            else:
                card_dir = next(d for d in dirs if abs(d[0]) + abs(d[1]) == 1)
                diag_dir = next(d for d in dirs if abs(d[0]) + abs(d[1]) == 2)
                pattern = (diag_dir, card_dir)
            if pattern[0] == (0, 0):
                continue
            post_path = _trace_pattern_to_edge(collision, pattern, n)
            if len(post_path) < MIN_POST_STEPS:
                continue

            own_beam_cells = set(post_path)
            for path in pre_paths:
                own_beam_cells.update(path)

            if any(cell in occupied_frame_cells for cell in own_beam_cells if cell not in {pre_paths[0][0], pre_paths[1][0]}):
                continue
            if any(cell in reserved_beam_cells for cell in own_beam_cells):
                continue
            if any(cell in own_frame_cells for cell in own_beam_cells if cell not in {pre_paths[0][0], pre_paths[1][0]}):
                continue
            if set(post_path) & (pre_sets[0] | pre_sets[1]) not in (set(), {collision}):
                continue

            return {
                "kind": kind,
                "dirs": dirs,
                "frames": frame_pos,
                "seeds": seeds,
                "collision": collision,
                "post_path": post_path,
                "beam_cells": own_beam_cells,
                "frame_cells": own_frame_cells,
            }
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    mode = str(rng.choice(["90", "45", "both"], p=[0.3, 0.3, 0.4]))
    kinds = [mode] if mode != "both" else list(rng.permutation(["90", "45"]))
    pair_count = len(kinds)

    n_lo = 14 if pair_count == 1 else 20
    n = int(rng.integers(n_lo, 31))
    bg, frame_color, pair_colors = _pick_colors(rng, pair_count)

    plans = []
    existing_frames: list[tuple[int, int]] = []
    occupied_frame_cells: set[tuple[int, int]] = set()
    reserved_beam_cells: set[tuple[int, int]] = set()

    for kind in kinds:
        plan = _plan_pair(n, kind, existing_frames, occupied_frame_cells, reserved_beam_cells, rng)
        if plan is None:
            return None
        plans.append(plan)
        existing_frames.extend(plan["frames"])
        occupied_frame_cells.update(plan["frame_cells"])
        reserved_beam_cells.update(plan["beam_cells"])

    grid = np.full((n, n), bg, dtype=int)
    for plan, color in zip(plans, pair_colors):
        for (r0, c0), d in zip(plan["frames"], plan["dirs"]):
            _draw_frame(grid, r0, c0, bg, frame_color, int(color), d)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, metrics = _run_simulation(grid, collect_metrics=True)
    except Exception:
        return None

    if metrics["collisions_45"] != kinds.count("45"):
        return None
    if metrics["collisions_90"] != kinds.count("90"):
        return None
    if any(int(out[r, c]) == bg for (r, c), _diff in metrics["collision_cells"]):
        return None
    if np.count_nonzero(out != grid) < pair_count * 6:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
