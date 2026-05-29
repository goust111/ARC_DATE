# Combo variant of e12f9a14 (index 36)
# combo_spec: D11/2+D13/2
# Name: 碰撞改框架色+链式传递

# Design analysis for the fused chain-recolor variant
# ----------------------------------------------------
# concretization_and_behavior_trace:
# - Keep the original visual vocabulary: square grids, one shared frame color, and one 2x2 monochrome core per 4x4 frame.
# - Fuse the two changes by turning the layout into a visible relay chain. There is exactly one 1-gap source frame, then
#   three 2-gap relay frames. The source ray reaches frame B, which triggers B; B reaches C, which triggers C; C reaches D.
# - The last relay then emits one more ray that collides with an already-drawn earlier segment. That collision does two
#   things at once: it creates the local collision mark (a 1-cell hole at the meeting point) and it sends the opponent
#   color back to both source frames, recoloring their surviving border pixels.
# - Small concrete trace: frame A emits east into frame B; frame B emits north into frame C; frame C emits west into
#   frame D; frame D emits south into A's old horizontal connector. In the original task family, frames would either all
#   start immediately or would never receive remote color feedback. Here, the old segment from A becomes the "stored
#   message" that D later runs into, so chain order and recolor feedback are coupled.
#
# complexity_calibration:
# - A fully general chain plus all original collision geometries would add too many moving parts for 2-4 ARC examples.
#   The generator therefore fixes the family to a single, legible collision type: a right-angle meeting between the
#   final live segment and one previously drawn segment.
# - The visual novelty is still substantial: middle frames do not fire until reached, and the eventual collision changes
#   two distant frame borders. That makes the fused rule easy to distinguish from both the original puzzle and either
#   single-dimension variant.
# - The chain length is fixed at 4 frames. This is the smallest size that cleanly shows source, relays, and a remote
#   recolor event without clutter.
#
# implicit_assumptions_review:
# - The original solver assumes every gap is an immediate emitter; that is invalid here. main() instead finds the unique
#   1-gap source and follows one activation path.
# - The original collision engine assumes multiple simultaneously active beams. That is also invalid here. Only one
#   segment is live at a time, so a collision can only happen between the current segment and an already painted older
#   segment. The solver stores trail ownership and entry direction for exactly that reason.
# - The recolor step means frame borders can no longer be trusted as immutable output structure, so occupancy checks must
#   read from the input grid, not from the output grid. The solver does this consistently.
# - The fused solver assumes: one source frame, three 2-gap relays, every relay hit is unique, and the only collision is
#   one right-angle meeting between the final segment and an earlier trail cell. The generator constructs and validates
#   precisely that family.
#
# placement_legality:
# - The generator uses a rectangle-like four-frame scaffold with >=2 cells of bbox clearance everywhere, keeping every
#   4x4 frame in bounds with a margin.
# - Link lengths are sampled from short bounded ranges, then converted into exact frame positions so that each intended
#   relay target lies on the correct ray and no unintended frame or connector is crossed.
# - The final south-going segment is positioned to hit an interior cell of the first horizontal connector, not a frame
#   cell or a gap cell, which avoids ambiguous overwrites.
#
# visual_observability:
# - The source frame is visibly unique because it has one gap; the others all have two. The output then shows a clear
#   temporal story: predecessor color enters a frame, that frame's own color exits elsewhere, and a later collision
#   eventually rewrites two remote borders.
# - The recolored borders are not arbitrary. They always belong to the two frames whose segments met, and each border
#   becomes the opponent core color. With only four distinct block colors, that mapping is recoverable by inspection.
#
# visual_tidiness:
# - The generator uses only one collision and avoids overlapping non-collision connectors, so the grid stays sparse.
# - The collision is terminal: after the 1-cell hole is created, no further branching occurs. That keeps the decisive
#   fused effect centered on "ordered activation + information sent back to the sources" instead of on path spaghetti.
#
# color_change_inferability_and_contract:
# - Colors are never ordered by numeric id. Geometry determines source, relay order, and the collision pair; color only
#   labels which frame owns each segment.
# - main() expects the constrained 4-frame family above. generate_input() enforces it and validates every sample by
#   running the same solver and checking activation count, collision count, and border recolor statistics.

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_A_GAP = (1, 3)
_B_GAPS = ((1, 0), (0, 2))
_C_GAPS = ((3, 2), (2, 0))
_D_GAPS = ((2, 3), (3, 1))

BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}

GRID_SIZE_MIN = 22
GRID_SIZE_MAX = 31
PLACEMENT_MARGIN = 2
FRAME_BBOX_GAP = 2
H1_RANGE = (6, 11)
H2_MIN = 4
V_RANGE = (4, 9)
GEN_RETRIES = 400


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected one shared frame color"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))
    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0, "expected frame-border gap"
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    return out


def _frame_wall_cells(g: np.ndarray, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    cells: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == wall:
                cells.append((r, c))
    return cells


def _gap_dir(local_gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = local_gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    d = (dr, dc)
    assert d != (0, 0), "each gap must point outward"
    return d


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = _DIR_TO_I[a]
    ib = _DIR_TO_I[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


def _frames_separated(a: tuple[int, int], b: tuple[int, int], gap: int = FRAME_BBOX_GAP) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _solve(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    blocks = _find_blocks_2x2(g, bg)
    assert len(blocks) >= 2, "expected framed 2x2 blocks"

    frames: list[dict[str, object]] = []
    gap_to_frame: dict[tuple[int, int], tuple[int, int]] = {}
    frame_wall_cells: list[tuple[int, int]] = []

    for color, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, color)
        gaps = sorted(_frame_gaps_for_block(g, bg, frame_col, br, bc))
        assert 1 <= len(gaps) <= 2, "expected one source gap or two relay gaps"
        r0, c0 = br - 1, bc - 1
        dirs = [_gap_dir((gr - r0, gc - c0)) for gr, gc in gaps]
        wall_cells = _frame_wall_cells(g, frame_col, br, bc)
        frames.append(
            {
                "color": int(color),
                "gaps": gaps,
                "dirs": dirs,
                "wall_cells": wall_cells,
            }
        )
        frame_wall_cells.extend(wall_cells)
        for gi, gap in enumerate(gaps):
            assert gap not in gap_to_frame, "gap cells must be unique"
            gap_to_frame[gap] = (len(frames) - 1, gi)

    source_ids = [fi for fi, frame in enumerate(frames) if len(frame["gaps"]) == 1]
    assert len(source_ids) == 1, "expected exactly one source frame"

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    blocked: set[tuple[int, int]] = set()
    activated: set[int] = set()

    stats = {
        "frame_count": len(frames),
        "activated_count": 0,
        "handoff_count": 0,
        "collision_count": 0,
        "recolored_frames": set(),
        "last_event": "",
    }

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def paint_trail(fid: int, cell: tuple[int, int], d: tuple[int, int]) -> None:
        out[cell[0], cell[1]] = int(frames[fid]["color"])
        cell_owner[cell] = fid
        cell_enter[cell] = d

    def recolor_frame(fid: int, color: int) -> None:
        for r, c in frames[fid]["wall_cells"]:
            out[r, c] = int(color)
        stats["recolored_frames"].add(fid)

    def trace_segment(frame_id: int, gap_id: int) -> tuple[str, tuple[int, int] | None]:
        start = tuple(frames[frame_id]["gaps"][gap_id])
        d = tuple(frames[frame_id]["dirs"][gap_id])
        if out[start[0], start[1]] == bg:
            paint_trail(frame_id, start, d)
        cur = start
        while True:
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if not in_bounds(nxt):
                return "edge", None
            owner = gap_to_frame.get(nxt)
            if owner is not None and owner[0] != frame_id:
                next_frame, next_gap = owner
                assert next_frame not in activated, "chain must move to a fresh relay"
                assert int(g[nxt[0], nxt[1]]) == bg and int(out[nxt[0], nxt[1]]) == bg
                out[nxt[0], nxt[1]] = int(frames[frame_id]["color"])
                stats["handoff_count"] += 1
                return "hit", owner
            if nxt in blocked:
                raise AssertionError("segment must not revisit a blocked collision cell")
            if int(g[nxt[0], nxt[1]]) != bg:
                raise AssertionError("relay connectors must pass only through background")
            if nxt in cell_owner:
                other_frame = cell_owner[nxt]
                assert other_frame != frame_id, "self-collision is not supported in this family"
                assert _dir_diff(d, cell_enter[nxt]) == 2, "fused family uses one right-angle collision"
                recolor_frame(frame_id, int(frames[other_frame]["color"]))
                recolor_frame(other_frame, int(frames[frame_id]["color"]))
                stats["collision_count"] += 1
                stats["last_event"] = "collision"
                out[nxt[0], nxt[1]] = bg
                cell_owner.pop(nxt, None)
                cell_enter.pop(nxt, None)
                blocked.add(nxt)
                return "collision", None
            paint_trail(frame_id, nxt, d)
            cur = nxt

    current_frame = source_ids[0]
    current_gap = 0

    while True:
        assert current_frame not in activated, "chain must be acyclic"
        activated.add(current_frame)
        event, payload = trace_segment(current_frame, current_gap)
        if event != "hit":
            stats["last_event"] = event
            break
        next_frame, entry_gap = payload
        assert len(frames[next_frame]["gaps"]) == 2, "every non-source frame must have two gaps"
        current_frame = next_frame
        current_gap = 1 - entry_gap

    stats["activated_count"] = len(activated)
    assert len(activated) == len(frames), "every frame should participate in the relay chain"
    if return_stats:
        stats["recolored_frames"] = len(stats["recolored_frames"])
        stats["frame_border_changes"] = int(sum(out[cell] != g[cell] for cell in frame_wall_cells))
        stats["trail_cells"] = int(np.count_nonzero(out != g) - stats["frame_border_changes"])
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay rays from one framed block to the next, then let the final collision send color back to the sources.

    Input structure:
    - The background is the most common color. Each object is a monochrome 2x2 block surrounded by a shared-color 4x4 frame.
    - Exactly one frame has a single gap and acts as the auto-start source. Every other frame has two gaps: one reachable
      entry and one dormant exit that becomes active only after the frame is reached.

    Rule:
    - Start from the unique 1-gap frame and paint a straight ray of that block's color from its gap.
    - When the ray lands on another frame's gap, stop there, color that entry gap with the predecessor color, and trigger
      the reached frame to emit from its other gap in its own color.
    - Continue this handoff chain. When the final emitted segment hits an earlier painted segment at a right angle, erase
      the meeting cell to background and recolor both source frame borders to the opponent segment's color.

    Key insight:
    - Earlier rays are not just static output; they become delayed information carriers. A later frame can collide with
      an old segment, and that single event simultaneously reveals chain order, marks the collision locally, and rewrites
      two distant frame borders as remote feedback.
    """
    return _solve(grid, return_stats=False)


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=4, replace=False))
    return bg, frame_color, block_colors


def _make_specs(r0: int, c0: int, h1: int, h2: int, v: int) -> list[dict[str, object]]:
    a = (r0, c0)
    b = (r0, c0 + 3 + h1)
    c = (r0 - v - 3, c0 + 3 + h1)
    d = (r0 - v - 3, c0 + h1 - h2)
    return [
        {"pos": a, "gaps": (_A_GAP,)},
        {"pos": b, "gaps": _B_GAPS},
        {"pos": c, "gaps": _C_GAPS},
        {"pos": d, "gaps": _D_GAPS},
    ]


def _layout_ok(specs: list[dict[str, object]], n: int) -> bool:
    positions = [tuple(spec["pos"]) for spec in specs]
    for r0, c0 in positions:
        if not (PLACEMENT_MARGIN <= r0 <= n - 4 - PLACEMENT_MARGIN):
            return False
        if not (PLACEMENT_MARGIN <= c0 <= n - 4 - PLACEMENT_MARGIN):
            return False
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if not _frames_separated(positions[i], positions[j]):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng)

    h1 = int(rng.integers(*H1_RANGE))
    h2_hi = h1 - 1
    if h2_hi < H2_MIN:
        return None
    h2 = int(rng.integers(H2_MIN, h2_hi + 1))
    v = int(rng.integers(*V_RANGE))

    r_lo = v + 5
    r_hi = n - 6
    c_lo = PLACEMENT_MARGIN
    c_hi = n - (h1 + 10)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    r0 = int(rng.integers(r_lo, r_hi + 1))
    c0 = int(rng.integers(c_lo, c_hi + 1))
    specs = _make_specs(r0, c0, h1, h2, v)
    if not _layout_ok(specs, n):
        return None

    grid = np.full((n, n), bg, dtype=int)
    for spec, color in zip(specs, block_colors):
        pr, pc = spec["pos"]
        _draw_framed_block(grid, pr, pc, bg, frame_color, int(color), tuple(spec["gaps"]))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _solve(grid, return_stats=True)
    except Exception:
        return None

    if stats["frame_count"] != 4:
        return None
    if stats["activated_count"] != 4:
        return None
    if stats["handoff_count"] != 3:
        return None
    if stats["collision_count"] != 1:
        return None
    if stats["recolored_frames"] != 2:
        return None
    if stats["frame_border_changes"] < 20:
        return None
    if stats["last_event"] != "collision":
        return None
    if np.count_nonzero(out != grid) < 18:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused e12f9a14 combo input")
