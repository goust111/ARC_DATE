# Variant of 4c416de3
# Dimension: D2 — 源对象：角点图案结构 (src)
# Change: 框架角点处的彩色图案，包含外向像素（dr<0或dc<0）和对角内向像素（k,k），整体关于dr==dc轴对称，4-6个像... → 非对角对称的任意小型精灵

import random

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


# Design analysis for the D2 variant
# ----------------------------------
# concrete_behavior:
#   preserved_backbone:
#     - The grid is still a solid bg color with 2-3 hollow 0-frames.
#     - Each detectable frame corner is either a source corner or a target corner.
#     - Sources still show the full sprite in one color; targets still keep only one
#       colored marker at the deepest inward-diagonal cell.
#   changed_object:
#     - The canonical TL-oriented sprite is no longer required to satisfy
#       (dr, dc) <-> (dc, dr) diagonal symmetry.
#     - A source may therefore contain a one-sided "hook" such as (1,2) without the
#       mirrored (2,1). When transferred to TR / BL / BR corners, that hook visibly
#       flips with the corner orientation.
#   behavior_trace:
#     - Original: source canon {(-1,0),(0,-1),(1,1),(2,2),(1,2),(2,1)} could be
#       mirrored across dr==dc with no information loss, so the solver could assume
#       symmetry and only verify that every cell had its transpose partner.
#     - Variant: source canon {(-1,0),(0,-1),(0,0),(1,1),(2,2),(1,2)} has no
#       transpose partner for (1,2). The solver must read the exact source offsets,
#       keep them as a full sprite, then map those offsets through the corner kind.
# complexity_calibration:
#   - Patterns stay small (5-6 cells, |dr|,|dc| <= 2) so the task remains legible.
#   - Every pattern still has >=2 outward cells and >=1 inward-diagonal anchor, so
#     source detection and target marking remain stable.
#   - The generator only uses truly non-diagonal-symmetric templates; otherwise the
#     variant would visually collapse back to the seed task.
# implicit_assumption_audit:
#   old_dependencies_removed:
#     - The original main() asserted diagonal symmetry for every source offset.
#     - It also relied on symmetry to describe why one extracted source was enough.
#   still_valid_assumptions:
#     - `_rot` remains correct because corner normalization is independent of sprite
#       symmetry; it just converts between actual offsets and TL-canonical offsets.
#     - The deepest diagonal cell is still safe as the unique target marker because
#       the variant changes the sprite shape, not the marker protocol.
#   small_solver_adjustment:
#     - Source extraction now keys off the outward color found during source
#       detection instead of a "most common color" heuristic. This avoids accidental
#       dependence on symmetry or sprite size when counting colored cells.
# placement_legality:
#   - Templates only use outward offsets down to -1 and inward offsets up to +2.
#   - The seed task's spacing is kept, but the minimum frame size is tightened from
#     5 to 6 so opposite corners of the same frame cannot overlap at the center when
#     asymmetric sprites use depth-2 cells.
# visual_observability:
#   - Each generated instance has multiple corners of different kinds, so the same
#     asymmetric source sprite appears flipped in several orientations.
#   - Because targets still use their own marker colors, the user sees both the
#     orientation change and the per-corner recoloring in 2-4 examples.
# cleanliness_and_color:
#   - Colors are copied directly from target markers; there is no hidden ordering or
#     color-id arithmetic.
#   - The grid remains uncluttered because only one sprite touches each corner and
#     the frame gap stays at least 4 cells.


CORNER_KINDS = ("TL", "TR", "BL", "BR")

# Inward-along-frame-edge directions for each corner kind.
EDGE_DIRS = {
    "TL": ((+1, 0), (0, +1)),
    "TR": ((+1, 0), (0, -1)),
    "BL": ((-1, 0), (0, +1)),
    "BR": ((-1, 0), (0, -1)),
}
# Outward-from-corner directions; cells here must not be 0 (guards mid-edge false positives).
OUT_DIRS = {
    "TL": ((-1, 0), (0, -1)),
    "TR": ((-1, 0), (0, +1)),
    "BL": ((+1, 0), (0, -1)),
    "BR": ((+1, 0), (0, +1)),
}

# Radius for scanning a corner's neighborhood. Must exceed max canonical |dr|,|dc| a pattern can have.
_NEIGH_R = 3


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical (TL-orientation) offset; self-inverse."""
    if kind == "TL":
        return (dr, dc)
    if kind == "TR":
        return (dr, -dc)
    if kind == "BL":
        return (-dr, dc)
    if kind == "BR":
        return (-dr, -dc)
    raise ValueError(f"unknown corner kind: {kind}")


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=10):
    """Walk from (r,c) along (dr,dc) until bg/boundary. Return (non_bg_count, saw_zero)."""
    cnt = 0
    saw_zero = False
    for k in range(1, max_step + 1):
        rr, cc = r + k * dr, c + k * dc
        if not (0 <= rr < H and 0 <= cc < W):
            break
        v = int(g[rr, cc])
        if v == bg:
            break
        cnt += 1
        if v == 0:
            saw_zero = True
    return cnt, saw_zero


def _find_corners(g, bg):
    H, W = g.shape
    corners = []
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) == bg:
                continue
            for kind in CORNER_KINDS:
                (d1r, d1c), (d2r, d2c) = EDGE_DIRS[kind]
                r1, c1 = r + d1r, c + d1c
                r2, c2 = r + d2r, c + d2c
                if not (0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W):
                    continue
                if int(g[r1, c1]) == bg or int(g[r2, c2]) == bg:
                    continue
                cnt1, sz1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, sz2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 3 and sz1 and cnt2 >= 3 and sz2):
                    continue
                ok = True
                for odr, odc in OUT_DIRS[kind]:
                    rr, cc = r + odr, c + odc
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == 0:
                        ok = False
                        break
                if not ok:
                    continue
                corners.append((r, c, kind))
                break
    return corners


def _colored_around(g, r, c, kind, bg):
    H, W = g.shape
    items = []
    for dr in range(-_NEIGH_R, _NEIGH_R + 1):
        for dc in range(-_NEIGH_R, _NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


def _is_outward(off):
    dr, dc = off
    return dr < 0 or dc < 0


def _is_diagonal_symmetric(pattern):
    cells = set(pattern)
    return all((dc, dr) in cells for dr, dc in cells)


def _extract_sources_and_canon(per_corner):
    sources = []
    source_colors = {}
    for c_key, items in per_corner.items():
        outward = [(off, v) for off, v in items if _is_outward(off)]
        if len(outward) < 2:
            continue
        colors = {v for _, v in outward}
        assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
        sources.append(c_key)
        source_colors[c_key] = next(iter(colors))
    assert sources, "at least one source corner (pre-drawn outward sprite) required"

    canon = None
    for c_key in sources:
        scol = source_colors[c_key]
        sprite = frozenset(off for off, v in per_corner[c_key] if v == scol)
        assert sprite, f"source {c_key} has no colored sprite cells"
        if canon is None:
            canon = sprite
        else:
            assert sprite == canon, (
                f"source sprites disagree: {sorted(canon)} vs {sorted(sprite)}"
            )

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "sprite has no inward-diagonal cell to anchor markers"
    marker_canon = (max(diag_ks), max(diag_ks))
    assert marker_canon in canon, "deepest diagonal marker missing from source sprite"
    return sources, source_colors, canon, marker_canon


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a fully shown corner sprite to every other marker-bearing corner.

    Input structure:
    - The background is the most frequent color. Zero-valued hollow frames define
      detectable corners, and each corner cell may stay 0 or be recolored.
    - One or more source corners already show the complete sprite in a single
      non-bg color. The sprite includes outward cells plus at least one inward
      diagonal cell used as the marker anchor.
    - Other corners keep only one colored marker at the deepest inward-diagonal
      sprite cell; that marker color is the color the completed sprite should use.

    Rule:
    - Normalize every colored source cell into TL-oriented canonical offsets and
      recover the exact full sprite from any source corner.
    - For each non-source corner whose deepest diagonal marker is colored, repaint
      that entire canonical sprite after mapping the offsets into that corner's
      orientation, using the marker's own color.

    Key insight:
    - The corner orientation is handled by coordinate normalization, not by a
      diagonal-symmetry shortcut. Once the source sprite is expressed in canonical
      offsets, the whole asymmetric shape can be flipped into any corner kind, and
      the lone retained marker supplies only the target color and target identity.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}
    sources, _source_colors, canon, marker_canon = _extract_sources_and_canon(per_corner)

    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        mdr, mdc = _rot(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        v = int(g[mr, mc])
        if v == bg or v == 0:
            continue
        tcol = v
        for off, color in per_corner[c_key]:
            if off in canon:
                assert color == tcol, (
                    f"target ({r},{c},{kind}) has conflicting color {color} at "
                    f"canonical offset {off}; expected {tcol}"
                )
        targets.append((r, c, kind, tcol))

    assert targets, "no target corners detected (every corner appears to be a source)"

    out = g.copy()
    for r, c, kind, tcol in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}) "
                f"color {tcol}: existing color {v}"
            )
            out[rr, cc] = tcol

    return out


# --------------- generator (output-first pattern replication (Codex)) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (6, 8)
SOURCE_MIN_SIZE = 6

# --- structural counts ---
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

# --- generation retries ---
MAX_ATTEMPTS = 300

# --- canonical (TL-oriented) sprite templates ---
# All templates are intentionally non-diagonal-symmetric and keep the original
# task's marker protocol: the deepest inward-diagonal cell is preserved in inputs.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
    [(-1, 0), (0, -1), (1, 1), (2, 2), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (2, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (0, 1)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (1, 2)],
)

# Offsets from a frame's top-left origin (r, c) to each corner cell of a size-N frame:
# applied as (r + dr*(size-1), c + dc*(size-1)).
_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


def _split_total(total: int, parts: int) -> list[int]:
    """Partition `total` non-negative units into `parts` bins uniformly at random."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    """Return (sprite cells, deepest inward-diagonal k used for target markers)."""
    pattern = list(random.choice(PATTERN_TEMPLATES))
    assert not _is_diagonal_symmetric(pattern), "variant pattern unexpectedly symmetric"
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    """Draw a size x size hollow frame of 0 with TL at (r, c)."""
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


def _sample_frame_pair(span: int) -> tuple[int, int, int, int] | None:
    """Pick two frame sizes and their origins along one axis of length `span`."""
    lo, hi = FRAME_SIZE_RANGE
    pair_limit = span - (2 * BORDER + FRAME_GAP)
    max_first = min(hi, pair_limit - lo)
    if max_first < lo:
        return None
    s1 = random.randint(lo, max_first)
    s2 = random.randint(lo, min(hi, pair_limit - s1))
    slack = span - (2 * BORDER + s1 + FRAME_GAP + s2)
    before, between, _after = _split_total(slack, 3)
    o1 = BORDER + before
    o2 = o1 + s1 + FRAME_GAP + between
    return s1, s2, o1, o2


def _sample_two_frames(H: int, W: int) -> list[dict] | None:
    """Place two frames either side-by-side (shared row) or stacked (shared col)."""
    horizontal = random.random() < TWO_FRAME_HORIZONTAL_PROB
    axis = _sample_frame_pair(W if horizontal else H)
    if axis is None:
        return None
    s1, s2, o1, o2 = axis
    if horizontal:
        row = random.randint(BORDER, H - max(s1, s2) - BORDER)
        return [{"r": row, "c": o1, "size": s1}, {"r": row, "c": o2, "size": s2}]
    col = random.randint(BORDER, W - max(s1, s2) - BORDER)
    return [{"r": o1, "c": col, "size": s1}, {"r": o2, "c": col, "size": s2}]


def _sample_three_frames(H: int, W: int) -> list[dict] | None:
    """Place three frames in three of the four corner slots (TL/TR/BL/BR), one missing."""
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(k for k in CORNER_KINDS if k != missing)

    def _bands(sizes: dict[str, int]) -> tuple[int, int, int, int]:
        top = max((sizes[s] for s in present if s[0] == "T"), default=0)
        bot = max((sizes[s] for s in present if s[0] == "B"), default=0)
        left = max((sizes[s] for s in present if s[1] == "L"), default=0)
        right = max((sizes[s] for s in present if s[1] == "R"), default=0)
        return top, bot, left, right

    feasible = []
    n_sizes = hi - lo + 1
    for combo in np.ndindex(*(n_sizes for _ in present)):
        sizes = {slot: lo + i for slot, i in zip(present, combo)}
        top, bot, left, right = _bands(sizes)
        if top + FRAME_GAP + bot > H - 2 * BORDER:
            continue
        if left + FRAME_GAP + right > W - 2 * BORDER:
            continue
        feasible.append(sizes)
    if not feasible:
        return None

    sizes = random.choice(feasible)
    top, bot, left, right = _bands(sizes)
    top_extra, mid_y, _ = _split_total(H - (2 * BORDER + top + FRAME_GAP + bot), 3)
    left_extra, mid_x, _ = _split_total(W - (2 * BORDER + left + FRAME_GAP + right), 3)
    top_row = BORDER + top_extra
    bot_row = top_row + top + FRAME_GAP + mid_y
    left_col = BORDER + left_extra
    right_col = left_col + left + FRAME_GAP + mid_x
    coords = {
        "TL": (top_row, left_col),
        "TR": (top_row, right_col),
        "BL": (bot_row, left_col),
        "BR": (bot_row, right_col),
    }
    return [{"r": coords[s][0], "c": coords[s][1], "size": sizes[s]} for s in present]


def _paint_pattern(grid: np.ndarray, r: int, c: int, kind: str,
                   pattern: list[tuple[int, int]], color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    """Give every corner of every frame an independently sampled non-bg color."""
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    """Choose 1..N_SOURCES_MAX source corners from frames larger than the minimum size."""
    eligible = [
        (fi, kind)
        for fi, frame in enumerate(frames)
        if frame["size"] >= SOURCE_MIN_SIZE
        for kind in CORNER_KINDS
    ]
    if not eligible:
        return None
    n = random.randint(1, min(N_SOURCES_MAX, len(eligible)))
    return set(random.sample(eligible, n))


def _render_output(base: np.ndarray, frames: list[dict],
                   pattern: list[tuple[int, int]]) -> np.ndarray:
    """Paint the canonical sprite at every corner using that corner's color."""
    out = base.copy()
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            _paint_pattern(out, r, c, kind, pattern, frame["corner_colors"][kind])
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], marker_k: int,
                  sources: set[tuple[int, str]]) -> np.ndarray:
    """Erase every non-source corner's sprite back to base, keeping only the marker."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            mdr, mdc = _rot(marker_k, marker_k, kind)
            marker = (r + mdr, c + mdc)
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                rr, cc = r + rr, c + cc
                if (rr, cc) != marker:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, marker_k = _choose_pattern()

    frames = (
        _sample_two_frames(H, W)
        if random.random() < N_FRAMES_2_PROB
        else _sample_three_frames(H, W)
    )
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None

    output = _render_output(base, frames, pattern)
    inp = _derive_input(output, base, frames, pattern, marker_k, sources)

    try:
        corners = _find_corners(inp, bg)
        per_corner = {c: _colored_around(inp, *c, bg) for c in corners}
        _sources, _source_colors, canon, _marker = _extract_sources_and_canon(per_corner)
        if _is_diagonal_symmetric(canon):
            return None
    except Exception:
        return None

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
