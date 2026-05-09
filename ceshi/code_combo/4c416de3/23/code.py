# Combo variant of 4c416de3 (index 23)
# combo_spec: D10/1+D15/1
# Name: 红绿标记系统

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg, draw_line, in_bounds


CORNER_KINDS = ("TL", "TR", "BL", "BR")
SOURCE_MARK = 2
TARGET_MARK = 3
CONTROL_OFFSETS = ((-2, 0), (0, -2))

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

# Radius for scanning a corner's neighborhood. The fused payload/control system
# uses offsets only in [-2, 2], and keeping the radius tight prevents another
# corner of the same small frame from polluting the local readout.
_NEIGH_R = 2


# Design audit for the fused variant:
# a) Concrete behavior:
#    Every active frame-corner now carries a fixed outward 2-pixel control code at
#    canonical offsets (-2,0) and (0,-2). Red means "this corner is a source";
#    green means "this corner is a target". Sources still show the full payload
#    glyph in their own color. Targets keep only the deepest inward-diagonal
#    payload pixel, which supplies the color used in the output.
# b) Why this is not a trivial restatement of the original:
#    The original solver used outward colored cells to detect sources. Here every
#    target also owns two outward colored cells, so that heuristic misfires on all
#    targets. The role colors therefore become necessary rather than decorative.
# c) Hidden assumptions replaced/guarded:
#    Source/target classification no longer depends on pattern completeness. The
#    new solver assumes only that control pixels use fixed colors 2/3 and payload
#    colors avoid 2/3; the generator enforces both. Target payload color is read
#    from the deepest diagonal cell of the canonical glyph, just as before.
# d/e/f/g) Legality, observability, cleanliness:
#    The generator reserves >=2 background cells around the grid border so the
#    control markers stay in-bounds, keeps the inter-frame gap at 4, and rejects
#    any layout whose payload/control footprints overlap frames or each other.
#    Payload colors are sampled from a small 2-4 color pool (excluding red/green)
#    so the copied-shape color rule is visible without turning the grid noisy.


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
                for (odr, odc) in OUT_DIRS[kind]:
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


def _role_of_corner(items):
    vals = {off: color for off, color in items if off in CONTROL_OFFSETS}
    code = tuple(vals.get(off) for off in CONTROL_OFFSETS)
    if code == (SOURCE_MARK, SOURCE_MARK):
        return "source"
    if code == (TARGET_MARK, TARGET_MARK):
        return "target"
    if code == (None, None):
        return None
    raise AssertionError(f"invalid role marker code at corner: {code}")


def _payload_items(items):
    return [
        (off, color)
        for off, color in items
        if off not in CONTROL_OFFSETS and color not in (SOURCE_MARK, TARGET_MARK)
    ]


def _actual_offsets(offsets, kind):
    for dr, dc in offsets:
        yield _rot(dr, dc, kind)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the source corner glyph onto every green-marked target corner.

    Input structure:
    - bg = most frequent color. Hollow 0-frames appear as before.
    - Every active corner has a 2-pixel outward role code: red means source,
      green means target. These role pixels are not part of the glyph payload.
    - Source corners keep the full glyph in a non-red/non-green color. Target
      corners keep only the deepest inward-diagonal payload pixel in their own
      output color.

    Rule:
    - Read the role code first to decide which corners are sources and which are
      targets; do not use pattern completeness for that classification.
    - Extract the shared canonical glyph from the red-marked sources.
    - For each green-marked target, read its deepest diagonal payload pixel and
      paint the full rotated glyph in that color, leaving the red/green role
      markers unchanged.

    Key insight:
    - The outward red/green control pixels deliberately occupy the same semantic
      region that the original task used for source detection, so role decoding
      must happen before glyph extraction. Once control pixels are ignored, the
      remaining payload is the same diagonally symmetric corner glyph as before.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner_all = {c: _colored_around(g, *c, bg) for c in corners}
    roles = {}
    per_corner_payload = {}
    for c_key, items in per_corner_all.items():
        role = _role_of_corner(items)
        if role is None:
            continue
        roles[c_key] = role
        per_corner_payload[c_key] = _payload_items(items)

    sources = [c for c, role in roles.items() if role == "source"]
    targets = [c for c, role in roles.items() if role == "target"]
    assert sources, "expected at least one red-marked source corner"
    assert targets, "expected at least one green-marked target corner"

    def _extract(c_key):
        items = per_corner_payload[c_key]
        assert items, f"source {c_key} has no payload"
        pat_col = Counter(v for (_, v) in items).most_common(1)[0][0]
        canon = frozenset(off for (off, v) in items if v == pat_col)
        return canon, pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dc, dr) in canon, f"pattern not diagonally symmetric: ({dr},{dc})"

    diag_ks = [dr for (dr, dc) in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal payload anchor"
    marker_canon = (max(diag_ks), max(diag_ks))

    out = g.copy()
    for c_key in targets:
        r, c, kind = c_key
        mdr, mdc = _rot(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
        assert 0 <= mr < H and 0 <= mc < W, f"target marker out of bounds for {c_key}"
        tcol = int(g[mr, mc])
        assert tcol not in (bg, 0, SOURCE_MARK, TARGET_MARK), (
            f"target {c_key} lacks a valid payload color marker"
        )
        for off, color in per_corner_payload[c_key]:
            assert off in canon, f"unexpected payload cell {off} at target {c_key}"
            assert color == tcol, f"target {c_key} mixes payload colors {color} and {tcol}"
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            assert 0 <= rr < H and 0 <= cc < W, f"paint out of bounds at ({rr},{cc})"
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f"paint conflict at ({rr},{cc}) for target {c_key}: existing {v}, new {tcol}"
            )
            out[rr, cc] = tcol

    return out


# --------------- generator (output-first pattern replication with role markers) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 2
FRAME_GAP = 4
FRAME_SIZE_RANGE = (5, 8)

# --- structural counts ---
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

# --- generation retries ---
MAX_ATTEMPTS = 400

# --- palette ---
CONTENT_COLORS = [1, 4, 5, 6, 7, 8, 9]

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
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


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


def _sample_frame_pair(span: int) -> tuple[int, int, int, int] | None:
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


def _paint_role_code(grid: np.ndarray, r: int, c: int, kind: str, color: int) -> None:
    for adr, adc in _actual_offsets(CONTROL_OFFSETS, kind):
        grid[r + adr, c + adc] = color


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    palette = [c for c in CONTENT_COLORS if c != bg]
    shared = random.sample(palette, random.randint(2, min(4, len(palette))))
    for frame in frames:
        frame["corner_colors"] = {kind: random.choice(shared) for kind in CORNER_KINDS}


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    eligible = [
        (fi, kind)
        for fi, frame in enumerate(frames)
        for kind in CORNER_KINDS
    ]
    if not eligible:
        return None
    n = random.randint(1, min(N_SOURCES_MAX, len(eligible)))
    return set(random.sample(eligible, n))


def _layout_is_legal(base: np.ndarray, frames: list[dict], pattern: list[tuple[int, int]]) -> bool:
    bg = int(get_bg(base))
    seen = set()
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            for dr, dc in pattern:
                adr, adc = _rot(dr, dc, kind)
                rr, cc = r + adr, c + adc
                if not in_bounds(base.shape, rr, cc):
                    return False
                if int(base[rr, cc]) == 0 and (dr, dc) != (0, 0):
                    return False
                if (rr, cc) in seen:
                    return False
                seen.add((rr, cc))
            for adr, adc in _actual_offsets(CONTROL_OFFSETS, kind):
                rr, cc = r + adr, c + adc
                if not in_bounds(base.shape, rr, cc):
                    return False
                if int(base[rr, cc]) != bg:
                    return False
                if (rr, cc) in seen:
                    return False
                seen.add((rr, cc))
    return True


def _render_output(base: np.ndarray, frames: list[dict],
                   pattern: list[tuple[int, int]], sources: set[tuple[int, str]]) -> np.ndarray:
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            _paint_pattern(out, r, c, kind, pattern, frame["corner_colors"][kind])
            role_color = SOURCE_MARK if (fi, kind) in sources else TARGET_MARK
            _paint_role_code(out, r, c, kind, role_color)
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], k_max: int,
                  sources: set[tuple[int, str]]) -> np.ndarray:
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            mdr, mdc = _rot(k_max, k_max, kind)
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
    bg = random.choice(CONTENT_COLORS)
    pattern, k_max = _choose_pattern()

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

    if not _layout_is_legal(base, frames, pattern):
        return None

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None

    output = _render_output(base, frames, pattern, sources)
    inp = _derive_input(output, base, frames, pattern, k_max, sources)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
