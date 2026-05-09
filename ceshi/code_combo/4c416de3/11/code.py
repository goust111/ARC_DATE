# Combo variant of 4c416de3 (index 11)
# combo_spec: D1/0+D11/0
# Name: 颜色匹配配对+不同框架颜色

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line, get_bg


CORNER_KINDS = ("TL", "TR", "BL", "BR")

EDGE_DIRS = {
    "TL": ((+1, 0), (0, +1)),
    "TR": ((+1, 0), (0, -1)),
    "BL": ((-1, 0), (0, +1)),
    "BR": ((-1, 0), (0, -1)),
}

_NEIGH_R = 2


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical TL-oriented offset; self-inverse."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    if kind == "BR":
        return -dr, -dc
    raise ValueError(f"unknown corner kind: {kind}")


def _edge_run(g, r, c, dr, dc, color, H, W, max_step=10):
    """Count consecutive `color` cells from the first inward step onward."""
    cnt = 0
    for k in range(1, max_step + 1):
        rr, cc = r + k * dr, c + k * dc
        if not (0 <= rr < H and 0 <= cc < W):
            break
        if int(g[rr, cc]) != color:
            break
        cnt += 1
    return cnt


def _find_corners(g, bg):
    """
    Find painted frame corners even when the corner cell itself is overwritten by a pattern.

    A valid corner has two inward-adjacent cells that share the same non-bg frame color,
    and that color continues for at least 3 cells along both frame edges.
    """
    H, W = g.shape
    corners = []
    for r in range(H):
        for c in range(W):
            for kind in CORNER_KINDS:
                (d1r, d1c), (d2r, d2c) = EDGE_DIRS[kind]
                r1, c1 = r + d1r, c + d1c
                r2, c2 = r + d2r, c + d2c
                if not (0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W):
                    continue
                col1 = int(g[r1, c1])
                col2 = int(g[r2, c2])
                if col1 == bg or col2 == bg or col1 != col2:
                    continue
                frame_col = col1
                cnt1 = _edge_run(g, r, c, d1r, d1c, frame_col, H, W)
                cnt2 = _edge_run(g, r, c, d2r, d2c, frame_col, H, W)
                if cnt1 >= 3 and cnt2 >= 3:
                    corners.append((r, c, kind, frame_col))
                    break
    return corners


def _colored_around(g, r, c, kind, bg, frame_col):
    """Collect local pattern cells around a corner in canonical TL coordinates."""
    H, W = g.shape
    items = []
    for dr in range(-_NEIGH_R, _NEIGH_R + 1):
        for dc in range(-_NEIGH_R, _NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == frame_col:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


# Design analysis for the fusion:
# a) Concrete behavior trace:
#    - Suppose a blue frame contains a red source with pattern A and a yellow source with
#      pattern B, while a green frame also contains red/yellow sources but swaps them to
#      patterns C/B. A red target inside the blue frame must copy A; a red target inside
#      the green frame must copy C. The solver therefore has to use BOTH cues: first stay
#      inside the same frame-color group, then match the target marker color to the source
#      of that color inside the group.
# b) Complexity calibration:
#    - If a frame had only one source, border colors alone would solve the task and
#      D1/0 would disappear. If source colors were unique globally, marker-color matching
#      would solve it without using frame colors. The generator avoids both degeneracies by
#      giving every frame two source colors that repeat across frames, while allowing the
#      per-frame color->pattern mapping to change.
# c) Implicit assumptions broken from the original solver:
#    - Filtering only `bg` and `0` no longer works because frame edges are colored, so
#      local pattern extraction must ignore `frame_col`.
#    - Pooling all sources into one global `canon` is invalid; multiple sources exist and
#      they must be partitioned by frame color first.
#    - Within a frame, target selection is not "all remaining corners": the solver must
#      choose the source whose paint color matches the target marker.
# d) Placement legality:
#    - Frames remain non-overlapping full squares of size 6-8 with a gap of 4, larger
#      than the pattern scan radius 2, so neither neighboring groups nor adjacent corners
#      within the same frame contaminate local extraction. Patterns stay within
#      |dr|,|dc| <= 1 in generated instances.
# e) Visual observability:
#    - Frame colors are the visible grouping cue, while the repeated source/marker colors
#      across different frames expose the need for the second pairing rule.
# f) Visual cleanliness:
#    - Only the corner cell can overwrite the frame border; the rest of each frame edge
#      stays solid, so every frame remains easy to read. Two sources and two targets per
#      frame keep both branches visible without over-crowding a 5-8 sized square.
# g) New solver assumptions guaranteed by the generator:
#    - Each frame color is unique and contributes exactly four detected corners.
#    - Every frame has exactly two sources and two targets.
#    - The same two source/marker colors are reused across all frames, but at least one
#      frame assigns a different pattern to one of those colors.
#    - Pattern/marker colors never equal the local frame color, so local extraction stays
#      unambiguous.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate each source corner's motif only to same-frame targets whose marker color
    matches that source.

    Input structure:
    - bg = most frequent color. The scene contains 2-3 hollow square frames whose
      borders use distinct non-bg colors; a corner cell itself may be overwritten by
      a source/marker color.
    - Each frame contributes exactly two source corners. Each source shows a diagonally
      symmetric corner pattern in a single paint color, and the two source colors are
      reused across all frames.
    - The other two corners of the same frame are targets. Each target keeps only one
      colored marker, and that marker color tells which same-frame source pattern to copy.

    Rule:
    - Detect corners and group them by frame-border color.
    - Inside each frame-color group, extract every source corner's canonical pattern and
      remember it by the source paint color.
    - For each target corner in that same group, find the source whose paint color matches
      the target marker color, then copy that source pattern to the target orientation.

    Key insight: border color and paint color now both matter. Different colored frames may
    reuse the same paint colors but bind them to different motifs, so solving requires
    first partitioning corners by frame color, then matching target marker color to the
    correct source inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corners detected"

    groups = {}
    for r, c, kind, frame_col in corners:
        groups.setdefault(frame_col, []).append((r, c, kind))

    per_corner = {}
    for r, c, kind, frame_col in corners:
        per_corner[(r, c, kind, frame_col)] = _colored_around(g, r, c, kind, bg, frame_col)

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    out = g.copy()
    total_targets = 0

    for frame_col, frame_corners in groups.items():
        kinds = {kind for _, _, kind in frame_corners}
        assert kinds == set(CORNER_KINDS), f"frame color {frame_col} missing corners: {kinds}"

        source_keys = []
        for r, c, kind in frame_corners:
            c_key = (r, c, kind, frame_col)
            outward = [(off, v) for off, v in per_corner[c_key] if _is_outward(off)]
            if len(outward) >= 2:
                colors = {v for _, v in outward}
                assert len(colors) == 1, (
                    f"source {(r, c, kind)} in frame color {frame_col} has mixed outward colors: {colors}"
                )
                source_keys.append(c_key)

        assert len(source_keys) == 2, f"frame color {frame_col} expected 2 sources, got {len(source_keys)}"

        source_by_color = {}
        marker_by_color = {}
        for source_key in source_keys:
            source_items = per_corner[source_key]
            source_col = Counter(v for _, v in source_items).most_common(1)[0][0]
            canon = frozenset(off for off, v in source_items if v == source_col)

            for dr, dc in canon:
                assert (dc, dr) in canon, (
                    f"frame color {frame_col} pattern not diagonally symmetric at {(dr, dc)}"
                )

            diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
            assert diag_ks, f"frame color {frame_col} pattern lacks inward-diagonal marker cell"
            assert source_col not in source_by_color, (
                f"frame color {frame_col} has duplicate source color {source_col}"
            )
            source_by_color[source_col] = canon
            marker_by_color[source_col] = (max(diag_ks), max(diag_ks))

        assert len({canon for canon in source_by_color.values()}) == len(source_by_color), (
            f"frame color {frame_col} sources collapsed to the same pattern"
        )

        for r, c, kind in frame_corners:
            c_key = (r, c, kind, frame_col)
            if c_key in source_keys:
                continue
            items = per_corner[c_key]
            matches = []
            for source_col, canon in source_by_color.items():
                mdr, mdc = _rot(*marker_by_color[source_col], kind)
                mr, mc = r + mdr, c + mdc
                if not (0 <= mr < H and 0 <= mc < W):
                    continue
                if int(g[mr, mc]) != source_col:
                    continue
                if any(color != source_col for off, color in items if off in canon):
                    continue
                matches.append((source_col, canon))

            assert len(matches) == 1, (
                f"target {(r, c, kind)} in frame {frame_col} has ambiguous source matches: {matches}"
            )
            tcol, canon = matches[0]
            total_targets += 1
            for cdr, cdc in canon:
                adr, adc = _rot(cdr, cdc, kind)
                rr, cc = r + adr, c + adc
                assert 0 <= rr < H and 0 <= cc < W, f"paint out of bounds at {(rr, cc)}"
                v = int(out[rr, cc])
                assert v in (bg, frame_col, tcol), (
                    f"paint conflict at {(rr, cc)} for target {(r, c, kind)}: existing {v}, need {tcol}"
                )
                out[rr, cc] = tcol

    assert total_targets > 0, "no targets found"
    return out


# --------------- generator (frame-color-grouped replication) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (6, 8)
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
MAX_ATTEMPTS = 400

PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}
_TEMPLATE_KMAX = tuple(max(dr for dr, dc in pat if dr == dc and dr > 0) for pat in PATTERN_TEMPLATES)
_TEMPLATE_EFFECTIVE = tuple(
    frozenset(pat) - {(k_max, k_max)}
    for pat, k_max in zip(PATTERN_TEMPLATES, _TEMPLATE_KMAX)
)
# Restrict generation to k_max=1 templates so pattern differences remain visible in the
# output and do not collapse onto the already-preserved input marker.
_GEN_TEMPLATE_IDXS = (0, 1, 3)


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int, color: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=color)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=color)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=color)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=color)


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


def _assign_patterns(frames: list[dict]) -> bool:
    valid_pairs_per_frame = []
    for frame in frames:
        valid = [
            idx for idx, k_max in enumerate(_TEMPLATE_KMAX)
            if idx in _GEN_TEMPLATE_IDXS
            if frame["size"] >= 2 * k_max + 2
        ]
        pairs = [(a, b) for a in valid for b in valid if a != b]
        if not pairs:
            return False
        valid_pairs_per_frame.append(pairs)

    pair_idxs = [random.choice(pairs) for pairs in valid_pairs_per_frame]
    if len(frames) > 1 and len(set(pair_idxs)) == 1:
        for fi, pairs in enumerate(valid_pairs_per_frame):
            alts = [pair for pair in pairs if pair != pair_idxs[fi]]
            if alts:
                pair_idxs[fi] = random.choice(alts)
                break
        if len(set(pair_idxs)) == 1:
            return False

    for frame, pair in zip(frames, pair_idxs):
        frame["pattern_pair"] = tuple(list(PATTERN_TEMPLATES[idx]) for idx in pair)
    return True


def _assign_colors(frames: list[dict], bg: int) -> bool:
    frame_palette = [c for c in range(1, 10) if c != bg]
    if len(frame_palette) < len(frames):
        return False
    frame_colors = random.sample(frame_palette, len(frames))
    blocked = {bg, *frame_colors}
    paint_palette = [c for c in range(1, 10) if c not in blocked]
    if len(paint_palette) < 2:
        return False
    shared_source_colors = random.sample(paint_palette, 2)
    for frame, frame_col in zip(frames, frame_colors):
        frame["frame_color"] = frame_col
        frame["shared_source_colors"] = tuple(shared_source_colors)
    return True


def _decorate_frames(frames: list[dict]) -> bool:
    if not _assign_patterns(frames):
        return False
    for frame in frames:
        source_kinds = tuple(random.sample(CORNER_KINDS, 2))
        target_kinds = [kind for kind in CORNER_KINDS if kind not in source_kinds]
        random.shuffle(target_kinds)
        source_colors = list(frame["shared_source_colors"])
        frame["source_kinds"] = source_kinds
        frame["target_kinds"] = tuple(target_kinds)
        frame["corner_colors"] = {
            source_kinds[0]: source_colors[0],
            source_kinds[1]: source_colors[1],
            target_kinds[0]: source_colors[0],
            target_kinds[1]: source_colors[1],
        }
        frame["pattern_by_color"] = {
            source_colors[0]: frame["pattern_pair"][0],
            source_colors[1]: frame["pattern_pair"][1],
        }
    return True


def _render_output(base: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = base.copy()
    bg = int(get_bg(base))
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            color = frame["corner_colors"][kind]
            pattern = frame["pattern_by_color"][color]
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                rr, cc = r + rr, c + cc
                v = int(out[rr, cc])
                assert v in (bg, frame["frame_color"], color), (
                    f"pattern collision at {(rr, cc)} while rendering frame color {frame['frame_color']}"
                )
            _paint_pattern(out, r, c, kind, pattern, color)
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict]) -> np.ndarray:
    inp = output.copy()
    for frame in frames:
        for kind in CORNER_KINDS:
            if kind in frame["source_kinds"]:
                continue
            r, c = _corner_coord(frame, kind)
            color = frame["corner_colors"][kind]
            pattern = frame["pattern_by_color"][color]
            k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
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
    bg = random.randint(1, 9)

    frames = _sample_two_frames(H, W) if random.random() < N_FRAMES_2_PROB else _sample_three_frames(H, W)
    if frames is None:
        return None

    if not _assign_colors(frames, bg):
        return None
    if not _decorate_frames(frames):
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"], frame["frame_color"])

    output = _render_output(base, frames)
    inp = _derive_input(output, base, frames)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
