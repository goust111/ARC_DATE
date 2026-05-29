# Combo variant of 4c416de3 (index 19)
# combo_spec: D1/2+D4/0+D8/2
# Name: 镜像映射+统一色+L形图案

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The shared scaffold stays close to the original task: 2-3 hollow 0-frames sit on a
#   solid non-zero background, and useful cells still live around detectable TL/TR/BL/BR
#   frame corners.
# - The original task copied a diagonally symmetric glyph and recolored each target with
#   that target's own marker color. Here both parts change together:
#   1. the canonical glyph is now an oriented L-shaped polyomino, not transpose-symmetric;
#   2. corner transport still uses only row/col sign flips, so TR is a left-right mirror,
#      BL is an up-down mirror, and BR is both;
#   3. all completed targets are painted with the source color, while their input marker
#      colors are merely locators and are deliberately different from the source color.
# - Example TL-canonical pattern:
#     {(-1,0), (0,-1), (0,0), (1,0), (2,0), (2,1)}
#   The deepest strictly inward cell is `(2,1)`. At a TR corner this becomes
#     {(-1,0), (0,1), (0,0), (1,0), (2,0), (2,-1)}.
#   The target keeps only the rotated marker, e.g. `(2,-1)`, in some decoy color such as
#   red, but the output draws the whole TR-oriented L in the source's color such as blue.
#
# b) Complexity calibration
# - Pattern size stays at 6 cells, close to the original visual density.
# - Only a small template family is used. Every sample still has repeated support because
#   the same canonical glyph appears at multiple corners, but the family is rich enough
#   that the task does not collapse into a single memorized stencil.
# - Marker colors are drawn from a tiny 2-3 color pool per grid and are excluded from the
#   source color. This keeps color clutter low while making the "uniform source color"
#   rule easy to notice from a few examples.
#
# c) Implicit-assumption audit for main()
# - `_find_corners`, `_edge_metric`, `_colored_around` only depend on the 0-frame geometry,
#   so they remain valid.
# - Source detection by ">=2 outward colored cells of one color" still works because every
#   source L-template preserves at least two outward cells and all source corners share one
#   color.
# - The old diagonal-symmetry assertion is invalid and must be replaced with two checks:
#   the canonical offsets must break transpose symmetry, and the non-negative offsets must
#   form a one-bend L shape.
# - The old marker rule `(k,k)` no longer holds. The preserved target cell is instead the
#   unique deepest strictly inward offset, chosen by maximal `dr+dc` and guaranteed to be
#   off-diagonal by the templates.
# - Old target-color consistency checks are invalid because the marker color is no longer
#   the paint color. The new target assumption is stronger and simpler: among canonical
#   offsets, a target may keep only the marker cell pre-painted in the input.
#
# d) Placement legality
# - All templates stay within |dr|,|dc| <= 2, so the existing scan radius 3 remains safe.
# - The deepest marker is always `(2,1)` or `(1,2)`, so even the minimum frame size 5
#   keeps it inside the frame interior.
# - The generator still uses output-first validation (`main(inp) == output`) to catch any
#   overlap, out-of-bounds, or source/target ambiguity regressions.
#
# e) Visual observability
# - `canon != transpose(canon)` ensures TL/TR/BL/BR completions look visibly different.
# - Because every target marker color differs from the source color, the output cannot be
#   mistaken for the original per-target recoloring rule.
# - The L-templates are directional and therefore easier for a human to compare across
#   examples than a nearly symmetric blob.
#
# f) Visual cleanliness
# - No extra object classes are introduced; the scene still contains only background,
#   0-frames, and corner glyphs.
# - Marker colors come from a tiny pool and frames remain well separated, so the grid stays
#   readable even when several targets are present.
#
# g) Change inferability
# - Color inference is intentionally simplified. Humans only need to notice that the source
#   color is copied everywhere, while target marker colors repeat but never control the
#   output.
# - Shape inference remains non-trivial but local: identify one full L source, find the
#   unique deepest inward marker at each target, then mirror the L according to corner
#   orientation.

import itertools
import random

import numpy as np

from arc_utils import draw_line, get_bg


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


def _marker_from_canon(canon):
    """Pick the unique deepest strictly inward cell used as the target marker."""
    inward = [(dr, dc) for (dr, dc) in canon if dr > 0 and dc > 0]
    assert inward, "pattern has no strictly inward anchor cell"
    best_score = max(dr + dc for dr, dc in inward)
    best = [(dr, dc) for dr, dc in inward if dr + dc == best_score]
    assert len(best) == 1, f"pattern has ambiguous deepest inward marker: {sorted(best)}"
    marker = best[0]
    assert marker[0] != marker[1], "marker must be off diagonal in this combo"
    return marker


def _is_l_shape(canon):
    """Check that the non-negative part forms a one-bend L with contiguous arms."""
    cells = {(dr, dc) for (dr, dc) in canon if dr >= 0 and dc >= 0}
    if len(cells) < 4:
        return False
    for pr, pc in cells:
        row_cells = {(r, c) for (r, c) in cells if r == pr}
        col_cells = {(r, c) for (r, c) in cells if c == pc}
        arm = row_cells | col_cells
        if arm != cells:
            continue
        row_cs = sorted(c for (_, c) in row_cells)
        col_rs = sorted(r for (r, _) in col_cells)
        if row_cs != list(range(min(row_cs), max(row_cs) + 1)):
            continue
        if col_rs != list(range(min(col_rs), max(col_rs) + 1)):
            continue
        if len(row_cells) >= 2 and len(col_cells) >= 2:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror an oriented corner L-glyph from the source corners onto every marker-bearing target.

    Input structure:
    - The background is the most frequent non-zero color, with 2-3 hollow 0-frames.
    - One or more source corners already contain the full L-shaped glyph in a shared source
      color. The glyph has outward pixels plus a non-diagonal inward bend.
    - Every target corner keeps only one colored marker at the glyph's deepest strictly
      inward cell. That marker color may differ from the source and is not the output color.

    Rule:
    - Read the canonical TL-oriented glyph from any source corner.
    - Mirror it into each other corner orientation via row/col sign flips, then paint the
      whole target glyph with the shared source color, overwriting the target marker.

    Key insight:
    - The transport rule is still corner-orientation mirroring, but the canonical glyph is
      now intentionally asymmetric and L-shaped, so horizontal and vertical mirrors are
      visually distinct. Color reasoning simplifies in the opposite direction: markers only
      identify where to paint, while the actual paint color always comes from the source.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    source_infos = []
    for c_key, items in per_corner.items():
        outward = [(off, v) for (off, v) in items if _is_outward(off)]
        if len(outward) < 2:
            continue
        colors = {v for (_, v) in outward}
        assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
        src_color = next(iter(colors))
        same_color_offsets = frozenset(off for (off, v) in items if v == src_color)
        if not any(dr > 0 and dc > 0 for (dr, dc) in same_color_offsets):
            continue
        source_infos.append((c_key, src_color, same_color_offsets))
    assert source_infos, "at least one source corner (pre-drawn outward pattern) required"

    source_color = source_infos[0][1]
    canon = source_infos[0][2]
    for c_key, other_color, other_canon in source_infos[1:]:
        assert other_color == source_color, (
            f"source {c_key} color {other_color} disagrees with source color {source_color}"
        )
        assert other_canon == canon, (
            f"source patterns disagree: {sorted(canon)} vs {sorted(other_canon)}"
        )

    transposed = frozenset((dc, dr) for (dr, dc) in canon)
    assert transposed != canon, "pattern unexpectedly remains diagonally symmetric"
    assert _is_l_shape(canon), f"pattern is not an L-shape: {sorted(canon)}"
    marker_canon = _marker_from_canon(canon)

    src_set = {c_key for (c_key, _, _) in source_infos}
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        mdr, mdc = _rot(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        marker_color = int(g[mr, mc])
        if marker_color == bg or marker_color == 0:
            continue
        for off, color in per_corner[c_key]:
            if off in canon and off != marker_canon:
                raise AssertionError(
                    f"target ({r},{c},{kind}) keeps unexpected painted cell at {off}"
                )
            if off == marker_canon:
                assert color == marker_color, (
                    f"target ({r},{c},{kind}) has inconsistent marker color {color}"
                )
        targets.append((r, c, kind, marker_color))

    assert targets, "no target corners detected (every corner appears to be a source)"

    out = g.copy()
    for r, c, kind, _marker_color in targets:
        mdr, mdc = _rot(*marker_canon, kind)
        marker = (r + mdr, c + mdc)
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            if (rr, cc) != marker:
                assert v == bg or v == 0 or v == source_color, (
                    f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}): "
                    f"existing color {v}, source color {source_color}"
                )
            out[rr, cc] = source_color

    return out


# --------------- generator (output-first pattern replication (Codex)) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (5, 8)
SOURCE_MIN_SIZE = 6

# --- structural counts ---
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

# --- generation retries ---
MAX_ATTEMPTS = 300

# --- canonical (TL-oriented) L templates ---
# Every template:
# - stays within |dr|,|dc| <= 2,
# - contains >=2 outward cells,
# - contains a unique deepest strictly-inward marker,
# - is intentionally not transpose-symmetric,
# - has a non-negative part that is a one-bend L.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 0), (2, 0), (2, 1)],
    [(-1, -1), (-1, 0), (0, 0), (1, 0), (2, 0), (2, 1)],
    [(-1, 0), (0, -1), (0, 0), (0, 1), (0, 2), (1, 2)],
    [(-1, -1), (-1, 0), (0, 0), (0, 1), (0, 2), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 0), (1, 1), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 1), (2, 1)],
)

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


def _choose_pattern() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Return (pattern cells, canonical marker offset)."""
    pattern = list(random.choice(PATTERN_TEMPLATES))
    canon = frozenset(pattern)
    assert frozenset((dc, dr) for (dr, dc) in canon) != canon
    assert _is_l_shape(canon)
    marker = _marker_from_canon(canon)
    return pattern, marker


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
    target_kind_counts = {kind: len(frames) for kind in CORNER_KINDS}
    choices = []
    for n in range(1, min(N_SOURCES_MAX, len(eligible)) + 1):
        for picked in map(set, itertools.combinations(eligible, n)):
            remaining = target_kind_counts.copy()
            for _fi, kind in picked:
                remaining[kind] -= 1
            if all(count >= 1 for count in remaining.values()):
                choices.append(picked)
    if not choices:
        return None
    return random.choice(choices)


def _assign_corner_colors(frames: list[dict], bg: int,
                          sources: set[tuple[int, str]]) -> int:
    """Assign one shared source color and a small non-source marker palette."""
    palette = [c for c in range(1, 10) if c != bg]
    source_color = random.choice(palette)
    marker_palette = [c for c in palette if c != source_color]
    pool_size = random.randint(2, min(3, len(marker_palette)))
    marker_pool = random.sample(marker_palette, pool_size)

    targets = [(fi, kind) for fi, frame in enumerate(frames) for kind in CORNER_KINDS
               if (fi, kind) not in sources]
    target_colors = [random.choice(marker_pool) for _ in targets]
    if len(targets) >= 2 and len(set(target_colors)) == 1 and len(marker_pool) >= 2:
        target_colors[1] = marker_pool[1]

    for frame in frames:
        frame["corner_colors"] = {}
    for fi, kind in sources:
        frames[fi]["corner_colors"][kind] = source_color
    for (fi, kind), color in zip(targets, target_colors):
        frames[fi]["corner_colors"][kind] = color

    assert all(
        frames[fi]["corner_colors"][kind] != source_color for (fi, kind) in targets
    )
    return source_color


def _render_output(base: np.ndarray, frames: list[dict],
                   pattern: list[tuple[int, int]], source_color: int) -> np.ndarray:
    """Paint the canonical pattern at every participating corner using the source color."""
    out = base.copy()
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            _paint_pattern(out, r, c, kind, pattern, source_color)
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], marker_canon: tuple[int, int],
                  sources: set[tuple[int, str]]) -> np.ndarray:
    """Erase every non-source corner back to base, then keep only the recolored marker."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                inp[r + rr, c + cc] = base[r + rr, c + cc]
            mdr, mdc = _rot(*marker_canon, kind)
            marker_color = frame["corner_colors"][kind]
            inp[r + mdr, c + mdc] = marker_color
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, marker_canon = _choose_pattern()

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

    sources = _pick_sources(frames)
    if sources is None:
        return None
    source_color = _assign_corner_colors(frames, bg, sources)

    output = _render_output(base, frames, pattern, source_color)
    inp = _derive_input(output, base, frames, pattern, marker_canon, sources)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
