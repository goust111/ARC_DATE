# Combo variant of 4c416de3 (index 2)
# combo_spec: D1/0+D9/0
# Name: 颜色匹配配对+一对一配对

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

# The fused variant only uses templates with |dr|,|dc| <= 2, so a radius of 2 isolates
# same-frame opposite corners while still covering every source/marker cell.
_NEIGH_R = 2

# All source patterns share the same deepest inward-diagonal cell so targets can be
# represented by a single marker regardless of which source they match.
MARKER_CANON = (2, 2)


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical (TL-oriented) offset; self-inverse."""
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
    return off[0] < 0 or off[1] < 0


def _marker_of(pattern):
    diag_ks = [dr for dr, dc in pattern if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal marker anchor"
    k = max(diag_ks)
    return (k, k)


def _extract_source(items, c_key):
    outward = [(off, color) for off, color in items if _is_outward(off)]
    if len(outward) < 2:
        return None
    colors = {color for _, color in outward}
    assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
    color = next(iter(colors))
    assert all(v == color for _, v in items), f"source {c_key} has polluted neighborhood"
    pattern = frozenset(off for off, v in items if v == color)
    for dr, dc in pattern:
        assert (dc, dr) in pattern, f"pattern at {c_key} is not diagonally symmetric"
    return color, pattern, _marker_of(pattern)


# Design analysis for combo D1/0 + D9/0:
# a) Concrete behavior:
#    Original behavior = pick one canonical source pattern, then copy it to every target.
#    New behavior = there are 2-3 source corners with different shapes and unique colors,
#    plus exactly one target marker of each source color. Example: if a red source uses
#    template A and a green source uses template B, the red marker receives only A and the
#    green marker receives only B. Color now selects WHICH pattern to copy, not just what
#    color to paint.
# b) Complexity calibration:
#    Colors are limited to 2-3 pairs and each input color appears exactly twice: once as a
#    full source glyph and once as a lone target marker. Patterns come from a 4-template bank
#    with a shared marker depth, so the pairing is visible without becoming combinatorially noisy.
# c) Broken original assumptions:
#    The old solver assumed all sources agree on one canonical pattern and that every target
#    can be filled with that same canon. That is false here. The new solver instead assumes:
#    source colors are unique, targets are single-cell markers, and all source patterns share
#    the same deepest diagonal marker offset. The generator explicitly enforces those inputs.
# d) Placement legality:
#    Active corners are chosen as opposite-corner pairs on each frame, patterns stay within
#    radius 2, and candidate instances are rejected unless every painted cell is in-bounds,
#    active output patterns are disjoint, and each corner's local neighborhood contains only
#    its intended source glyph or target marker.
# e) Visual observability:
#    Human solvers see several differently-shaped source glyphs and several isolated same-color
#    markers. Because the matching colors are separated across frames, the color pairing is the
#    only stable cue that explains which shape goes where.
# f) Visual cleanliness:
#    Unused corners remain plain 0-frame corners; no extra decorations are added. The small
#    template bank and local-neighborhood rejection keep the grids from becoming cluttered.
# g) Anti-degeneration:
#    Every sample contains at least two distinct source patterns. Therefore any "single global
#    canon" strategy from the original task necessarily paints at least one target with the wrong
#    shape, so the fused behavior does not collapse back to the base task.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fully drawn corner glyph to the lone target marker of the same color.

    Input structure:
    - bg is the most frequent color. The grid contains 2-3 hollow 0-frames, and some frame
      corners are inert 0-corners with no colored pixels nearby.
    - 2-3 source corners are already fully drawn. Each source is monochrome, includes outward
      cells plus inward-diagonal cells, and different sources may use different diagonally
      symmetric canonical patterns.
    - 2-3 target corners contain exactly one colored marker pixel, always at the shared deepest
      inward-diagonal offset. A target's marker color matches exactly one source color.

    Rule:
    - Detect every source corner and recover its own TL-oriented canonical pattern.
    - Detect every target marker, pair it with the unique source of the same color, then paint
      that matched source pattern at the target corner after rotating it to the target's corner kind.

    Key insight:
    - Color is now the pairing key and shape is no longer globally shared. The common marker
      depth tells you which colored singletons are targets; the matched source color tells you
      which pattern to copy.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c_key: _colored_around(g, *c_key, bg) for c_key in corners}

    source_by_color = {}
    source_corners = set()
    marker_canon = None
    for c_key, items in per_corner.items():
        src = _extract_source(items, c_key)
        if src is None:
            continue
        color, pattern, marker = src
        assert color not in source_by_color, f"duplicate source color {color}"
        if marker_canon is None:
            marker_canon = marker
        else:
            assert marker == marker_canon, (
                f"source {c_key} uses marker {marker}, expected {marker_canon}"
            )
        source_by_color[color] = (pattern, c_key)
        source_corners.add(c_key)

    assert len(source_by_color) >= 2, "fused variant expects at least two distinct sources"
    assert marker_canon is not None, "no source marker depth found"

    target_by_color = {}
    for c_key, items in per_corner.items():
        if c_key in source_corners or not items:
            continue
        assert len(items) == 1, f"target {c_key} should have exactly one marker, got {items}"
        (off, color), = items
        assert off == marker_canon, f"target {c_key} marker at {off}, expected {marker_canon}"
        assert color in source_by_color, f"target {c_key} uses unmatched color {color}"
        assert color not in target_by_color, f"duplicate target color {color}"
        target_by_color[color] = c_key

    assert set(target_by_color) == set(source_by_color), (
        f"source colors {sorted(source_by_color)} and target colors {sorted(target_by_color)} differ"
    )

    out = g.copy()
    for color, target_key in target_by_color.items():
        pattern, _source_key = source_by_color[color]
        r, c, kind = target_key
        for cdr, cdc in pattern:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            assert 0 <= rr < H and 0 <= cc < W, f"paint out of bounds at {(rr, cc)}"
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == color, (
                f"paint conflict at ({rr},{cc}) for target {target_key}: "
                f"existing {v}, new {color}"
            )
            out[rr, cc] = color

    return out


# --------------- generator (output-first color-paired one-to-one replication) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (6, 8)

N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
MAX_ATTEMPTS = 400

PATTERN_BANK = (
    frozenset({(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)}),
    frozenset({(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)}),
    frozenset({(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)}),
    frozenset({(-1, 0), (0, -1), (1, 1), (1, 2), (2, 1), (2, 2)}),
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}
_OPPOSITE_PAIRS = (("TL", "BR"), ("TR", "BL"))


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


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


def _sample_frame_pair(span: int):
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


def _sample_two_frames(H: int, W: int):
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


def _sample_three_frames(H: int, W: int):
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(kind for kind in CORNER_KINDS if kind != missing)

    def _bands(sizes: dict[str, int]):
        top = max((sizes[k] for k in present if k[0] == "T"), default=0)
        bot = max((sizes[k] for k in present if k[0] == "B"), default=0)
        left = max((sizes[k] for k in present if k[1] == "L"), default=0)
        right = max((sizes[k] for k in present if k[1] == "R"), default=0)
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
    return [{"r": coords[k][0], "c": coords[k][1], "size": sizes[k]} for k in present]


def _sample_frames(H: int, W: int):
    if random.random() < N_FRAMES_2_PROB:
        return _sample_two_frames(H, W)
    return _sample_three_frames(H, W)


def _pattern_cells(frame: dict, kind: str, pattern):
    r, c = _corner_coord(frame, kind)
    cells = []
    for dr, dc in pattern:
        rr_off, cc_off = _rot(dr, dc, kind)
        cells.append((r + rr_off, c + cc_off))
    return cells


def _paint_pattern(grid: np.ndarray, frame: dict, kind: str, pattern, color: int) -> None:
    for rr, cc in _pattern_cells(frame, kind, pattern):
        grid[rr, cc] = color


def _paint_marker(grid: np.ndarray, frame: dict, kind: str, color: int) -> None:
    r, c = _corner_coord(frame, kind)
    rr_off, cc_off = _rot(*MARKER_CANON, kind)
    grid[r + rr_off, c + cc_off] = color


def _derangement(n: int):
    if n == 2:
        return (1, 0)
    if n == 3:
        return random.choice(((1, 2, 0), (2, 0, 1)))
    raise ValueError(n)


def _choose_roles(frames):
    sources = []
    targets = []
    for fi, _frame in enumerate(frames):
        pair = random.choice(_OPPOSITE_PAIRS)
        if random.random() < 0.5:
            source_kind, target_kind = pair
        else:
            target_kind, source_kind = pair
        sources.append((fi, source_kind))
        targets.append((fi, target_kind))
    return sources, targets


def _build_pair_specs(frames, bg):
    pair_count = len(frames)
    patterns = random.sample(PATTERN_BANK, pair_count)
    colors = random.sample([c for c in range(1, 10) if c != bg], pair_count)
    sources, targets = _choose_roles(frames)
    perm = _derangement(pair_count)
    return [
        {
            "pattern": patterns[i],
            "color": colors[i],
            "source": sources[i],
            "target": targets[perm[i]],
        }
        for i in range(pair_count)
    ]


def _patterns_disjoint(frames, pair_specs):
    painted = {}
    for spec in pair_specs:
        for role in ("source", "target"):
            fi, kind = spec[role]
            for cell in _pattern_cells(frames[fi], kind, spec["pattern"]):
                prev = painted.get(cell)
                if prev is not None:
                    return False
                painted[cell] = spec["color"]
    return True


def _all_in_bounds(shape, frames, pair_specs):
    H, W = shape
    for spec in pair_specs:
        for role in ("source", "target"):
            fi, kind = spec[role]
            for rr, cc in _pattern_cells(frames[fi], kind, spec["pattern"]):
                if not (0 <= rr < H and 0 <= cc < W):
                    return False
    return True


def _local_signature_ok(inp: np.ndarray, frames, pair_specs, bg: int) -> bool:
    expected = {}
    for fi, _frame in enumerate(frames):
        for kind in CORNER_KINDS:
            expected[(fi, kind)] = set()
    for spec in pair_specs:
        expected[spec["source"]] = {(off, spec["color"]) for off in spec["pattern"]}
        expected[spec["target"]] = {(MARKER_CANON, spec["color"])}

    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            got = set(_colored_around(inp, r, c, kind, bg))
            if got != expected[(fi, kind)]:
                return False
    return True


def _render_case(base: np.ndarray, frames, pair_specs):
    inp = base.copy()
    out = base.copy()
    for spec in pair_specs:
        sfi, skind = spec["source"]
        tfi, tkind = spec["target"]
        _paint_pattern(inp, frames[sfi], skind, spec["pattern"], spec["color"])
        _paint_pattern(out, frames[sfi], skind, spec["pattern"], spec["color"])
        _paint_marker(inp, frames[tfi], tkind, spec["color"])
        _paint_pattern(out, frames[tfi], tkind, spec["pattern"], spec["color"])
    return inp, out


def _gen_attempt():
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    frames = _sample_frames(H, W)
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    for _ in range(40):
        pair_specs = _build_pair_specs(frames, bg)
        if len({spec["pattern"] for spec in pair_specs}) < 2:
            continue
        if not _all_in_bounds(base.shape, frames, pair_specs):
            continue
        if not _patterns_disjoint(frames, pair_specs):
            continue
        inp, out = _render_case(base, frames, pair_specs)
        if not _local_signature_ok(inp, frames, pair_specs, bg):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
