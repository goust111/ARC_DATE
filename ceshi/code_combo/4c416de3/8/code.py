# Combo variant of 4c416de3 (index 8)
# combo_spec: D4/2+D10/0
# Name: 图例颜色映射系统

# Design analysis for the fusion:
# a) Concrete behavior: targets no longer carry their final paint color directly.
#    The left grid edge contains legend rows [key_color, mapped_color]. A source
#    corner still shows the full glyph in a key color; a target keeps only the
#    deepest diagonal marker in a key color. Output recolors each target by first
#    reading the marker key, then looking up the mapped color in the legend.
# b) Complexity calibration: the legend uses only 2-4 colors and forms a deranged
#    permutation, so marker!=output for every key. Corner keys are reused across
#    the whole grid to make the mapping visually learnable rather than noisy.
# c) Broken assumptions from the original task: the original solver assumed the
#    target marker already was the output color, and that any visible target-cell
#    color near canonical offsets must equal that final paint color. Here those
#    cells must agree with the legend key instead, and solving must parse the
#    legend before target painting.
# d) Placement legality: the legend lives in the first two columns only, while
#    frames are shifted right so no corner neighborhood scan of radius 3 can see
#    the legend. This prevents overlap and source-pattern contamination.
# e) Visual observability: the same key colors reappear in source patterns, target
#    markers, and legend-left entries; the mapped colors appear in legend-right
#    entries and completed target patterns. The transform is therefore readable
#    from a few examples by inspection.
# f) Visual cleanliness: the legend is confined to a dedicated edge strip with
#    empty spacer columns, so it does not stack on top of frames or corner glyphs.
# g) Non-degeneracy target: generation forces a no-fixed-point mapping and balanced
#    reuse of key colors, so applying the old "marker color = output color" rule
#    would disagree with the intended output on every target corner.

import random
from collections import Counter

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

# Outward-from-corner directions; cells here must not be 0.
OUT_DIRS = {
    "TL": ((-1, 0), (0, -1)),
    "TR": ((-1, 0), (0, +1)),
    "BL": ((+1, 0), (0, -1)),
    "BR": ((+1, 0), (0, +1)),
}

_NEIGH_R = 3


def _rot(dr: int, dc: int, kind: str) -> tuple[int, int]:
    """Map actual offset <-> canonical TL-oriented offset; self-inverse."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    return -dr, -dc


def _edge_metric(
    g: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    bg: int,
    H: int,
    W: int,
    max_step: int = 10,
) -> tuple[int, bool]:
    """Walk from (r,c) until bg/boundary. Return (non_bg_count, saw_zero)."""
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


def _find_corners(g: np.ndarray, bg: int) -> list[tuple[int, int, str]]:
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


def _colored_around(
    g: np.ndarray, r: int, c: int, kind: str, bg: int
) -> list[tuple[tuple[int, int], int]]:
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


def _read_legend(g: np.ndarray, bg: int) -> dict[int, int]:
    """Read left-edge legend rows of the form [key, mapped]."""
    H, W = g.shape
    assert W >= 3, "legend requires at least 3 columns"
    mapping: dict[int, int] = {}
    rows = []
    for r in range(H):
        key = int(g[r, 0])
        val = int(g[r, 1])
        if key in (bg, 0) or val in (bg, 0):
            continue
        if int(g[r, 2]) != bg:
            continue
        assert key != val, f"legend row {r} is a fixed point"
        assert key not in mapping, f"duplicate legend key {key}"
        mapping[key] = val
        rows.append(r)
    assert len(mapping) >= 2, "legend not found on left edge"
    assert rows == sorted(rows), "legend rows must be ordered"
    return mapping


def _extract_source_pattern(
    per_corner: dict[tuple[int, int, str], list[tuple[tuple[int, int], int]]],
    c_key: tuple[int, int, str],
) -> tuple[frozenset[tuple[int, int]], int]:
    items = per_corner[c_key]
    pat_col = Counter(v for _, v in items).most_common(1)[0][0]
    return frozenset(off for off, v in items if v == pat_col), pat_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a source corner glyph to every marker-bearing corner, but recolor targets by a legend lookup.

    Input structure:
    - Background is the most frequent non-black color. Hollow 0-frames sit away from the left edge.
    - The left edge contains a legend: each active row has a [key_color, mapped_color] pair in columns 0 and 1.
    - Source corners already show the full corner glyph in one key color. Other active corners show only the
      deepest inward-diagonal marker, also in a key color.

    Rule:
    - Detect the shared canonical corner glyph from any full source corner.
    - For each non-source corner with a colored marker, treat the marker color as a legend key, look up the
      mapped color on the left edge, rotate the canonical glyph to that corner, and paint it in the mapped color.

    Key insight: target markers are no longer self-describing colors. They encode a source-side key, and the
    actual output color is determined only after reading the edge legend's key->mapped correspondence.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    legend = _read_legend(g, bg)

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"
    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    def _is_outward(off: tuple[int, int]) -> bool:
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key, items in per_corner.items():
        outward = [(off, v) for off, v in items if _is_outward(off)]
        if len(outward) >= 2:
            colors = {v for _, v in outward}
            assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
            sources.append(c_key)
    assert sources, "at least one source corner required"

    canon, _ = _extract_source_pattern(per_corner, sources[0])
    for s in sources[1:]:
        other, _ = _extract_source_pattern(per_corner, s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dc, dr) in canon, f"pattern not diagonally symmetric: ({dr},{dc})"

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal anchor"
    marker_k = max(diag_ks)
    marker_canon = (marker_k, marker_k)

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
        key = int(g[mr, mc])
        if key == bg or key == 0:
            continue
        assert key in legend, f"marker color {key} missing from legend"
        for off, color in per_corner[c_key]:
            if off in canon:
                assert color == key, (
                    f"target ({r},{c},{kind}) has conflicting key color {color} at {off}; "
                    f"expected marker key {key}"
                )
        targets.append((r, c, kind, key, legend[key]))

    assert targets, "no target corners detected"

    out = g.copy()
    for r, c, kind, key, paint_color in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == paint_color or v == key, (
                f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}); "
                f"existing color {v}, new color {paint_color}"
            )
            out[rr, cc] = paint_color

    return out


# --------------- generator (legend-driven variant) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (5, 8)
N_FRAMES_2_PROB = 0.55
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2
MAX_ATTEMPTS = 400

LEGEND_COLS = 2
FRAME_MIN_C = 5  # keeps legend outside the 3-cell corner scan radius
X_SHIFT = FRAME_MIN_C - BORDER

LEGEND_SIZE_RANGE = (2, 4)

# Only k=1 diagonal depth is used to keep same-frame corner patterns visually separated.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1)],
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


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
    x_span = W - X_SHIFT
    axis = _sample_frame_pair(x_span if horizontal else H)
    if axis is None:
        return None
    s1, s2, o1, o2 = axis
    if horizontal:
        row = random.randint(BORDER, H - max(s1, s2) - BORDER)
        return [
            {"r": row, "c": X_SHIFT + o1, "size": s1},
            {"r": row, "c": X_SHIFT + o2, "size": s2},
        ]
    col_local = random.randint(BORDER, x_span - max(s1, s2) - BORDER)
    return [
        {"r": o1, "c": X_SHIFT + col_local, "size": s1},
        {"r": o2, "c": X_SHIFT + col_local, "size": s2},
    ]


def _sample_three_frames(H: int, W: int) -> list[dict] | None:
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(k for k in CORNER_KINDS if k != missing)
    x_span = W - X_SHIFT

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
        if left + FRAME_GAP + right > x_span - 2 * BORDER:
            continue
        feasible.append(sizes)
    if not feasible:
        return None

    sizes = random.choice(feasible)
    top, bot, left, right = _bands(sizes)
    top_extra, mid_y, _ = _split_total(H - (2 * BORDER + top + FRAME_GAP + bot), 3)
    left_extra, mid_x, _ = _split_total(x_span - (2 * BORDER + left + FRAME_GAP + right), 3)
    top_row = BORDER + top_extra
    bot_row = top_row + top + FRAME_GAP + mid_y
    left_col = X_SHIFT + BORDER + left_extra
    right_col = left_col + left + FRAME_GAP + mid_x
    coords = {
        "TL": (top_row, left_col),
        "TR": (top_row, right_col),
        "BL": (bot_row, left_col),
        "BR": (bot_row, right_col),
    }
    return [{"r": coords[s][0], "c": coords[s][1], "size": sizes[s]} for s in present]


def _paint_pattern(
    grid: np.ndarray,
    r: int,
    c: int,
    kind: str,
    pattern: list[tuple[int, int]],
    color: int,
) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _choose_legend(bg: int) -> tuple[list[int], dict[int, int]]:
    n = random.randint(*LEGEND_SIZE_RANGE)
    palette = [c for c in range(1, 10) if c != bg]
    keys = random.sample(palette, n)
    vals = list(keys)
    while True:
        random.shuffle(vals)
        if all(a != b for a, b in zip(keys, vals)):
            break
    return keys, {k: v for k, v in zip(keys, vals)}


def _assign_corner_keys(frames: list[dict], keys: list[int]) -> None:
    n_corners = len(frames) * len(CORNER_KINDS)
    pool = (keys * ((n_corners + len(keys) - 1) // len(keys)))[:n_corners]
    random.shuffle(pool)
    idx = 0
    for frame in frames:
        frame["corner_keys"] = {}
        for kind in CORNER_KINDS:
            frame["corner_keys"][kind] = pool[idx]
            idx += 1


def _pick_sources(frames: list[dict], keys: list[int]) -> set[tuple[int, str]] | None:
    eligible = [(fi, kind) for fi, _frame in enumerate(frames) for kind in CORNER_KINDS]
    max_sources = min(N_SOURCES_MAX, len(eligible) - 1)
    for _ in range(40):
        n = random.randint(1, max_sources)
        sources = set(random.sample(eligible, n))
        if _check_target_support(frames, sources, keys):
            return sources
    return None


def _legend_rows(H: int, n_rows: int) -> list[int]:
    choices = sorted(random.sample(range(BORDER, H - BORDER), n_rows))
    return choices


def _draw_legend(base: np.ndarray, rows: list[int], legend: dict[int, int], keys: list[int]) -> None:
    for r, key in zip(rows, keys):
        base[r, 0] = key
        base[r, 1] = legend[key]


def _render_output(
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    sources: set[tuple[int, str]],
    legend: dict[int, int],
) -> np.ndarray:
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            key = frame["corner_keys"][kind]
            color = key if (fi, kind) in sources else legend[key]
            _paint_pattern(out, r, c, kind, pattern, color)
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    k_max: int,
    sources: set[tuple[int, str]],
) -> np.ndarray:
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            key = frame["corner_keys"][kind]
            mdr, mdc = _rot(k_max, k_max, kind)
            marker = (r + mdr, c + mdc)
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                rr, cc = r + rr, c + cc
                inp[rr, cc] = base[rr, cc]
            inp[marker] = key
    return inp


def _check_target_support(
    frames: list[dict], sources: set[tuple[int, str]], keys: list[int]
) -> bool:
    target_keys = []
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) not in sources:
                target_keys.append(frame["corner_keys"][kind])
    seen = set(target_keys)
    return (
        len(target_keys) >= 2
        and len(seen) >= 2
        and seen == set(keys)
    )


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    if W - X_SHIFT < 11:
        return None
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frames = (
        _sample_two_frames(H, W)
        if random.random() < N_FRAMES_2_PROB
        else _sample_three_frames(H, W)
    )
    if frames is None:
        return None

    keys, legend = _choose_legend(bg)
    base = np.full((H, W), bg, dtype=int)
    _draw_legend(base, _legend_rows(H, len(keys)), legend, keys)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_keys(frames, keys)
    sources = _pick_sources(frames, keys)
    if sources is None:
        return None

    output = _render_output(base, frames, pattern, sources, legend)
    inp = _derive_input(output, base, frames, pattern, k_max, sources)

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
