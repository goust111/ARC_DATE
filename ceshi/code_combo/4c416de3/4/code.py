# Combo variant of 4c416de3 (index 4)
# combo_spec: D3/2+D4/2
# Name: 边框颜色编码目标+图例颜色映射

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


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

# Border-encoded target cue: one colored cell on each inward edge adjacent to the corner.
HINT_OFFSETS = ((1, 0), (0, 1))


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
    raise ValueError(kind)


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


def _colored_around(g, r, c, kind, bg, skip_cells=None):
    H, W = g.shape
    items = []
    for dr in range(-_NEIGH_R, _NEIGH_R + 1):
        for dc in range(-_NEIGH_R, _NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if skip_cells is not None and (rr, cc) in skip_cells:
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


def _read_hint_color(g, r, c, kind, bg):
    """Read the border-encoded key color from the target corner."""
    H, W = g.shape
    vals = []
    saw_blank = False
    for hdr, hdc in HINT_OFFSETS:
        adr, adc = _rot(hdr, hdc, kind)
        rr, cc = r + adr, c + adc
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        v = int(g[rr, cc])
        if v == bg or v == 0:
            saw_blank = True
            continue
        vals.append(v)
    if not vals:
        return None
    assert not saw_blank, f"partial border hint at corner {(r, c, kind)}"
    colors = set(vals)
    assert len(colors) == 1, f"mixed border hint colors at corner {(r, c, kind)}: {colors}"
    return vals[0]


def _read_legend_and_cells(g, bg):
    """Decode top-edge 2x2 legend blocks and return their occupied cells."""
    H, W = g.shape
    assert H >= 2, "grid too short for top-edge legend"
    legend = {}
    occupied = set()
    top = g[:2, :]
    c = 0
    while c < W:
        v0 = int(top[0, c])
        v1 = int(top[1, c])
        if v0 == bg and v1 == bg:
            c += 1
            continue
        assert v0 != bg and v1 != bg, f"partial legend column at x={c}"
        assert c + 1 < W, "truncated legend block at right edge"
        a0, a1 = int(top[0, c]), int(top[0, c + 1])
        b0, b1 = int(top[1, c]), int(top[1, c + 1])
        assert a0 == a1 and b0 == b1, f"legend block at x={c} must be 2x2 uniform by row"
        assert a0 != bg and b0 != bg, f"legend block at x={c} uses bg color"
        assert a0 != b0, f"legend key color {a0} maps to itself"
        assert a0 not in legend, f"duplicate legend key {a0}"
        legend[a0] = b0
        occupied.update({(0, c), (0, c + 1), (1, c), (1, c + 1)})
        c += 2
        if c < W and (int(top[0, c]) != bg or int(top[1, c]) != bg):
            raise AssertionError(f"legend blocks must be separated by bg column before x={c}")
    assert len(legend) >= 2, "legend must expose at least two color mappings"
    return legend, occupied


def _read_legend(g, bg):
    legend, _ = _read_legend_and_cells(g, bg)
    return legend


# Design analysis for the fused D3/2 + D4/2 variant
# a) Concrete behavior:
#    - Original target cue: one interior marker on the inward diagonal directly exposed the paint color.
#    - Fused target cue: a non-source corner now shows a tiny border L at canonical offsets {(1,0), (0,1)}.
#      That L no longer gives the final paint color directly; it gives a key color.
#    - A separate top-edge legend consists of repeated 2x2 blocks. In each block, the top row is a key
#      color and the bottom row is the actual drawing color. Example trace for a TL target:
#      input border L = color 8, legend block says 8 -> 3, source still provides the glyph shape, so
#      output paints the TL-rotated glyph in color 3 while the border L remains color 8.
# b) Complexity calibration:
#    - The border hint still uses exactly one geometric form, so the target-identification burden stays low.
#    - The legend uses only 2-3 mappings per sample, with repeated target reuse, which is enough to show a
#      nontrivial color code without turning the task into arbitrary palette memorization.
#    - Keys and values are sampled as disjoint color sets, so the fusion does not collapse back into direct
#      color copying and the difference is visually obvious.
# c) Old-assumption audit:
#    - Source detection from outward colored cells still holds because hints are strictly inward on the frame.
#    - Canon extraction from sources still works because sources never receive border hints.
#    - The old target branch assumed a deepest diagonal marker existed; that assumption is removed entirely.
#      Targets are now found via `_read_hint_color`, and their paint colors come from `_read_legend`.
#    - The solver now also assumes hint cells are not part of the canonical glyph. The generator preserves
#      that by keeping the original pattern family, which never uses {(1,0), (0,1)}.
# d) Placement legality:
#    - Frames keep their original sizes and inter-frame gap, but placement uses an asymmetric top margin so
#      the 2-row legend plus one bg separator row fit without touching any frame.
#    - The border L fits even on 5x5 frames because it only uses the first inward cell on each edge.
#    - Legend blocks live on rows 0-1 only, far outside the corner-neighborhood radius used by the solver.
# e) Visual observability:
#    - Across examples, humans can repeatedly see three linked structures: source corners reveal the glyph,
#      target corners expose only coded border colors, and the top legend explains how those code colors turn
#      into the finished pattern colors.
#    - Because keys stay on the border in both input and output, the two-step decode is visible rather than
#      hidden in the generation process.
# f) Visual cleanliness:
#    - The scene is layered: legend at the top edge, frames in the main field, source pattern outside/near
#      corners, and target cue on the frame border. No region needs to carry more than two visual roles.
# g) Color-rule inferability:
#    - The mapping is not numeric or order-based; it is given explicitly by spatial legend pairs.
#    - Each sample uses every legend key on at least one target, so the legend is always behaviorally active.
# 9) New assumptions introduced in `main()` and guaranteed by generator:
#    - Top rows contain only well-formed 2x2 legend blocks separated by bg columns.
#    - Every non-source detected corner has a complete two-cell border hint whose key appears in the legend.
#    - Legend keys differ from their mapped paint colors, preventing identity collapse.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a source corner glyph to every border-coded corner after a legend lookup.

    Input structure:
    - The main objects are still hollow 0-frames on a solid bg, but the top grid edge
      now also contains 2x2 legend blocks whose top row is a key color and bottom row
      is the corresponding paint color.
    - At least one source corner already shows the full glyph in one color, including
      outward cells that distinguish it from an ordinary frame corner.
    - Every other participating corner keeps no interior marker. Instead, the first
      cell on each inward frame edge is recolored, forming a tiny border L in a key color.

    Rule:
    - Recover the shared canonical corner glyph from the source corner(s).
    - For each non-source corner, read its border-L key color, translate that key via
      the top-edge legend, and paint the rotated glyph in the mapped color.

    Key insight: target decoding is now a two-hop chain. The border color near the
    corner is not the final answer by itself; it must be converted through the edge
    legend before the missing corner pattern can be completed.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    legend, legend_cells = _read_legend_and_cells(g, bg)
    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg, skip_cells=legend_cells) for c in corners}

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key, items in per_corner.items():
        outward = [(o, v) for o, v in items if _is_outward(o)]
        if len(outward) >= 2:
            colors = {v for _, v in outward}
            assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
            sources.append(c_key)
    assert sources, "at least one source corner (pre-drawn outward pattern) required"

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for _, v in items).most_common(1)[0][0]
        return frozenset(o for o, v in items if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dc, dr) in canon, (
            f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})"
        )
    for hint in HINT_OFFSETS:
        assert hint not in canon, f"border hint offset {hint} overlaps the pattern"

    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        key_col = _read_hint_color(g, r, c, kind, bg)
        assert key_col is not None, f"non-source corner {(r, c, kind)} is missing its border hint"
        assert key_col in legend, f"hint color {key_col} has no legend mapping"
        paint_col = legend[key_col]
        for off, color in per_corner[c_key]:
            if off in HINT_OFFSETS:
                assert color == key_col, (
                    f"target ({r},{c},{kind}) has conflicting hint color {color} at "
                    f"canonical offset {off}; expected {key_col}"
                )
        targets.append((r, c, kind, paint_col))

    assert targets, "no target corners detected"
    assert len(targets) + len(sources) == len(corners), "unexpected inert corner detected"

    out = g.copy()
    for r, c, kind, paint_col in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == paint_col, (
                f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}) "
                f"color {paint_col}: existing color {v}"
            )
            out[rr, cc] = paint_col

    return out


# --------------- generator (output-first pattern replication (Codex)) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
TOP_MARGIN = 4          # 2 legend rows + 2 bg separator rows
BOTTOM_MARGIN = 1
SIDE_MARGIN = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (5, 8)
SOURCE_MIN_SIZE = 6

# --- legend layout ---
LEGEND_ROWS = 2
LEGEND_KEYS_RANGE = (2, 3)
LEGEND_BLOCK_W = 2
LEGEND_BLOCK_GAP = 1

# --- structural counts ---
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

# --- generation retries ---
MAX_ATTEMPTS = 300

# --- canonical (TL-oriented) pattern templates ---
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (0, 0)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
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


def _choose_pattern() -> list[tuple[int, int]]:
    return list(random.choice(PATTERN_TEMPLATES))


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


def _sample_frame_pair(span: int, before_margin: int, after_margin: int):
    """Pick two frame sizes and their origins along one axis of length `span`."""
    lo, hi = FRAME_SIZE_RANGE
    pair_limit = span - (before_margin + after_margin + FRAME_GAP)
    max_first = min(hi, pair_limit - lo)
    if max_first < lo:
        return None
    s1 = random.randint(lo, max_first)
    s2 = random.randint(lo, min(hi, pair_limit - s1))
    slack = span - (before_margin + after_margin + s1 + FRAME_GAP + s2)
    before, between, _after = _split_total(slack, 3)
    o1 = before_margin + before
    o2 = o1 + s1 + FRAME_GAP + between
    return s1, s2, o1, o2


def _sample_two_frames(H: int, W: int):
    """Place two frames either side-by-side (shared row) or stacked (shared col)."""
    horizontal = random.random() < TWO_FRAME_HORIZONTAL_PROB
    axis = _sample_frame_pair(
        W if horizontal else H,
        SIDE_MARGIN if horizontal else TOP_MARGIN,
        SIDE_MARGIN if horizontal else BOTTOM_MARGIN,
    )
    if axis is None:
        return None
    s1, s2, o1, o2 = axis
    if horizontal:
        max_size = max(s1, s2)
        if TOP_MARGIN > H - max_size - BOTTOM_MARGIN:
            return None
        row = random.randint(TOP_MARGIN, H - max_size - BOTTOM_MARGIN)
        return [{"r": row, "c": o1, "size": s1}, {"r": row, "c": o2, "size": s2}]
    max_size = max(s1, s2)
    if SIDE_MARGIN > W - max_size - SIDE_MARGIN:
        return None
    col = random.randint(SIDE_MARGIN, W - max_size - SIDE_MARGIN)
    return [{"r": o1, "c": col, "size": s1}, {"r": o2, "c": col, "size": s2}]


def _sample_three_frames(H: int, W: int):
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
        if top + FRAME_GAP + bot > H - TOP_MARGIN - BOTTOM_MARGIN:
            continue
        if left + FRAME_GAP + right > W - 2 * SIDE_MARGIN:
            continue
        feasible.append(sizes)
    if not feasible:
        return None

    sizes = random.choice(feasible)
    top, bot, left, right = _bands(sizes)
    top_extra, mid_y, _ = _split_total(H - (TOP_MARGIN + BOTTOM_MARGIN + top + FRAME_GAP + bot), 3)
    left_extra, mid_x, _ = _split_total(W - (2 * SIDE_MARGIN + left + FRAME_GAP + right), 3)
    top_row = TOP_MARGIN + top_extra
    bot_row = top_row + top + FRAME_GAP + mid_y
    left_col = SIDE_MARGIN + left_extra
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


def _paint_hint(grid: np.ndarray, r: int, c: int, kind: str, color: int) -> None:
    for hdr, hdc in HINT_OFFSETS:
        rr, cc = _rot(hdr, hdc, kind)
        grid[r + rr, c + cc] = color


def _legend_width(n_entries: int) -> int:
    return n_entries * LEGEND_BLOCK_W + (n_entries - 1) * LEGEND_BLOCK_GAP


def _sample_legend(bg: int) -> list[tuple[int, int]]:
    n_entries = random.randint(*LEGEND_KEYS_RANGE)
    colors = random.sample([c for c in range(1, 10) if c != bg], 2 * n_entries)
    keys = colors[:n_entries]
    values = colors[n_entries:]
    random.shuffle(values)
    return list(zip(keys, values))


def _draw_legend(grid: np.ndarray, legend_pairs: list[tuple[int, int]]) -> None:
    width = _legend_width(len(legend_pairs))
    _, W = grid.shape
    start = random.randint(0, W - width)
    for i, (key_col, paint_col) in enumerate(legend_pairs):
        c0 = start + i * (LEGEND_BLOCK_W + LEGEND_BLOCK_GAP)
        grid[0, c0:c0 + LEGEND_BLOCK_W] = key_col
        grid[1, c0:c0 + LEGEND_BLOCK_W] = paint_col


def _pick_sources(frames: list[dict]):
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


def _assign_corner_colors(frames: list[dict], legend_pairs: list[tuple[int, int]],
                          sources: set[tuple[int, str]]) -> None:
    legend = dict(legend_pairs)
    keys = list(legend)
    values = list(legend.values())
    targets = [
        (fi, kind)
        for fi, _frame in enumerate(frames)
        for kind in CORNER_KINDS
        if (fi, kind) not in sources
    ]
    assert len(targets) >= len(keys), "not enough targets to expose all legend mappings"

    key_pool = list(keys)
    while len(key_pool) < len(targets):
        key_pool.append(random.choice(keys))
    random.shuffle(key_pool)

    for frame in frames:
        frame["corner_paint_colors"] = {}
        frame["corner_code_colors"] = {}

    for (fi, kind), key_col in zip(targets, key_pool):
        frames[fi]["corner_code_colors"][kind] = key_col
        frames[fi]["corner_paint_colors"][kind] = legend[key_col]

    for fi, kind in sources:
        frames[fi]["corner_code_colors"][kind] = None
        frames[fi]["corner_paint_colors"][kind] = random.choice(values)


def _render_output(base: np.ndarray, frames: list[dict], pattern: list[tuple[int, int]],
                   sources: set[tuple[int, str]]) -> np.ndarray:
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            paint_col = frame["corner_paint_colors"][kind]
            _paint_pattern(out, r, c, kind, pattern, paint_col)
            if (fi, kind) not in sources:
                _paint_hint(out, r, c, kind, frame["corner_code_colors"][kind])
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], sources: set[tuple[int, str]]) -> np.ndarray:
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                inp[r + rr, c + cc] = base[r + rr, c + cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern = _choose_pattern()
    legend_pairs = _sample_legend(bg)

    if _legend_width(len(legend_pairs)) > W:
        return None

    frames = (
        _sample_two_frames(H, W)
        if random.random() < N_FRAMES_2_PROB
        else _sample_three_frames(H, W)
    )
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    _draw_legend(base, legend_pairs)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    sources = _pick_sources(frames)
    if sources is None:
        return None
    _assign_corner_colors(frames, legend_pairs, sources)

    output = _render_output(base, frames, pattern, sources)
    inp = _derive_input(output, base, frames, pattern, sources)

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
