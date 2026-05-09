# Variant of 4c416de3
# Dimension: D13 — 角点检测逻辑 (meta)
# Change: 通过检测两条相邻框架边（>=3非背景像素且含0像素）的交汇点来定位角点，外向方向不能有0像素... → 锚点基于框架中心而非角点
#
# Design analysis for the D13 variant
# a) Concrete behavior and trace:
#    - Original task: every replicated glyph is anchored at a frame corner, and the solver
#      searches for corners directly from adjacent frame edges. Targets are identified by a
#      marker on the inward diagonal from that corner.
#    - This variant moves the whole reference system to the geometric center of each hollow
#      square frame. The four "slots" are now the four interior quadrants around the center.
#      A source slot already contains a full motif inside one quadrant; each target slot keeps
#      only the motif's farthest cell from the center as a marker.
#    - Example trace on a 9x9 frame: the original solver would treat the top-left corner cell
#      as the anchor and rotate a corner-tail motif around that point. Here the anchor is the
#      center cell; the canonical TL motif might occupy offsets {(1,1),(2,1),(3,2)} from the
#      center, drawn as actual cells {(-1,-1),(-2,-1),(-3,-2)}. For the BR target, the new
#      rule maps the same magnitudes by sign flip to {(+1,+1),(+2,+1),(+3,+2)} and paints
#      them in the BR marker's color. The corner itself no longer participates at all.
# b) Complexity calibration:
#    - Frames remain 2-3 distinct objects, so the puzzle still has multiple repeated examples.
#    - Motifs are asymmetric within a quadrant and always include a unique farthest marker cell,
#      so quadrant mapping is visible and target detection is unambiguous.
#    - Pattern radius is restricted to 2 or 3 cells and frame sizes are restricted to odd
#      squares with a single center pixel, avoiding excessive freedom.
# c) Hidden assumptions in the original main() that break here:
#    - `_find_corners(...)` assumes usable anchors are intersections of two frame edges.
#      That is false now; anchors are inferred from a full frame component and its bbox center.
#    - `_rot(...)` assumes all offsets are expressed in a corner-oriented canonical system.
#      That is replaced with a center/quadrant sign mapping on positive offset magnitudes.
#    - Source detection via "outward cells" is obsolete because center-anchored motifs never
#      extend outside the frame. The new solver instead detects sources as quadrants holding
#      multiple same-colored cells.
#    - Marker lookup at `(k,k)` from a corner is no longer valid; the marker is the unique
#      farthest canonical offset from the center.
# d) Placement legality:
#    - Only full hollow square 0-frames are generated, with odd sizes 7 or 9 so the center is
#      a real pixel. Motifs stay strictly inside the frame interior and never overwrite 0 edges.
#    - Frame gaps and border margins are kept, so 0-components stay disjoint and frame detection
#      by connected component bbox is stable.
# e) Visual observability:
#    - The change is obvious in samples because the colored action happens around frame centers,
#      not corners. Human solvers can read the four quadrants around each center as the repeated
#      slots that receive mirrored copies.
# f) Visual cleanliness:
#    - Each quadrant contains at most one small motif or one marker pixel, all inside the frame.
#      No corner/border overpainting occurs, so frames remain legible.
# g) Color inference:
#    - Colors are copied directly from each target marker. No sorting, arithmetic, or hidden
#      color-id logic is used; the spatial cue is the quadrant position relative to the center.
# h) Reference-task note:
#    - The requested source task is `b10624e5`. A solver copy was read from
#      `/home/lixinyu/snap/new_code/solvers/arc2/b10624e5.py`; its relevant baseline is that
#      transformations are keyed by quadrant relative to a global structural anchor rather than
#      by local corners. This variant borrows only that anchor-selection idea.
# i) New solver assumptions that the generator must guarantee:
#    - Every usable frame is a full odd square 0-ring with no interior 0 pixels.
#    - Each non-source quadrant contains its marker exactly at the unique farthest canonical
#      offset, and source quadrants all share the same canonical motif.

import random
from collections import Counter
from itertools import product

import numpy as np

from arc_utils import draw_line, get_bg


QUADRANT_KINDS = ("TL", "TR", "BL", "BR")
QUAD_SIGNS = {
    "TL": (-1, -1),
    "TR": (-1, +1),
    "BL": (+1, -1),
    "BR": (+1, +1),
}
SCAN_R = 3


def _quadrant_offset(off: tuple[int, int], kind: str) -> tuple[int, int]:
    sr, sc = QUAD_SIGNS[kind]
    return sr * off[0], sc * off[1]


def _cc4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(g: np.ndarray) -> list[dict]:
    frames = []
    for comp in _cc4(g == 0):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        size = r1 - r0 + 1
        if size != c1 - c0 + 1 or size < 7 or size % 2 == 0:
            continue
        if len(comp) != 4 * (size - 1):
            continue
        if not np.all(g[r0, c0 : c1 + 1] == 0):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == 0):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == 0):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == 0):
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == 0):
            continue
        frames.append(
            {
                "r": int(r0),
                "c": int(c0),
                "size": int(size),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    return frames


def _colored_quadrants(g: np.ndarray, frame: dict, bg: int) -> dict[str, list[tuple[tuple[int, int], int]]]:
    H, W = g.shape
    cr, cc = frame["center"]
    inner_r = (frame["size"] - 1) // 2 - 1
    radius = min(SCAN_R, inner_r)
    items = {kind: [] for kind in QUADRANT_KINDS}
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 or dc == 0:
                continue
            rr, cc2 = cr + dr, cc + dc
            if not (0 <= rr < H and 0 <= cc2 < W):
                continue
            v = int(g[rr, cc2])
            if v == bg or v == 0:
                continue
            kind = ("T" if dr < 0 else "B") + ("L" if dc < 0 else "R")
            items[kind].append(((abs(dr), abs(dc)), v))
    return items


def _marker_from_canon(canon: frozenset[tuple[int, int]]) -> tuple[int, int]:
    best = max(a + b for a, b in canon)
    cands = [off for off in canon if off[0] + off[1] == best]
    assert len(cands) == 1, f"marker offset is ambiguous: {sorted(cands)}"
    return cands[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a pre-drawn quadrant motif from frame centers to every other marker-bearing quadrant.

    Input structure:
    - bg is the most frequent color. The grid contains 2-3 full hollow square 0-frames, each
      with an odd side length so the frame has a unique center pixel.
    - At least one frame quadrant already contains the full motif in a single color; this is a
      source quadrant. The motif lives strictly inside the frame interior, positioned relative to
      the frame center rather than any corner.
    - Other quadrants keep only one colored marker pixel at the motif's unique farthest offset
      from the center. That marker color is the output color for its quadrant.

    Rule:
    - Detect each frame as a standalone 0-ring and use its bbox center as the anchor point.
    - Extract the canonical motif from any source quadrant as positive center offsets in TL form.
    - For every non-source quadrant with a marker, map the same canonical offsets into that
      quadrant by center-based sign flips and paint them in the marker's color.

    Key insight: quadrant identity now comes from the sign of a cell's offset from the frame
    center, so the transformation is a four-quadrant center mapping rather than a corner rotation.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _find_frames(g)
    assert frames, "no full odd square frame detected"

    per_quad: dict[tuple[int, str], list[tuple[tuple[int, int], int]]] = {}
    for fi, frame in enumerate(frames):
        for kind, items in _colored_quadrants(g, frame, bg).items():
            per_quad[(fi, kind)] = items

    sources: list[tuple[int, str]] = []
    for key, items in per_quad.items():
        if len(items) < 2:
            continue
        colors = {v for _, v in items}
        assert len(colors) == 1, f"source {key} has mixed colors: {colors}"
        sources.append(key)
    assert sources, "at least one source quadrant is required"

    def _extract(q_key: tuple[int, str]) -> tuple[frozenset[tuple[int, int]], int]:
        items = per_quad[q_key]
        pat_col = Counter(v for _, v in items).most_common(1)[0][0]
        return frozenset(off for off, v in items if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f"source motifs disagree: {sorted(canon)} vs {sorted(other)}"

    marker_canon = _marker_from_canon(canon)

    src_set = set(sources)
    targets: list[tuple[int, str, int]] = []
    for fi, frame in enumerate(frames):
        cr, cc = frame["center"]
        for kind in QUADRANT_KINDS:
            if (fi, kind) in src_set:
                continue
            mdr, mdc = _quadrant_offset(marker_canon, kind)
            mr, mc = cr + mdr, cc + mdc
            if not (0 <= mr < H and 0 <= mc < W):
                continue
            v = int(g[mr, mc])
            if v == bg or v == 0:
                continue
            tcol = v
            for off, color in per_quad[(fi, kind)]:
                if off in canon:
                    assert color == tcol, (
                        f"target {(fi, kind)} has conflicting color {color} at {off}; expected {tcol}"
                    )
            targets.append((fi, kind, tcol))

    assert targets, "no target quadrants detected"

    out = g.copy()
    for fi, kind, tcol in targets:
        cr, cc = frames[fi]["center"]
        for off in canon:
            dr, dc = _quadrant_offset(off, kind)
            rr, cc2 = cr + dr, cc + dc
            if not (0 <= rr < H and 0 <= cc2 < W):
                continue
            v = int(out[rr, cc2])
            assert v == bg or v == tcol, (
                f"paint conflict at {(rr, cc2)} for target {(fi, kind)} color {tcol}: existing {v}"
            )
            out[rr, cc2] = tcol

    return out


# --------------- generator (output-first center-anchored quadrant replication) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_OPTIONS = (7, 9)

N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2
MAX_ATTEMPTS = 300

PATTERN_TEMPLATES = (
    ((1, 1), (2, 1), (2, 2)),
    ((1, 1), (1, 2), (2, 2)),
    ((1, 1), (1, 2), (2, 1), (2, 2)),
    ((1, 1), (2, 1), (3, 2)),
    ((1, 1), (1, 2), (3, 2)),
    ((1, 1), (1, 2), (2, 1), (3, 2)),
    ((1, 1), (2, 1), (2, 2), (3, 2)),
)


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern() -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, ...]]:
    pattern = list(random.choice(PATTERN_TEMPLATES))
    radius = max(max(a, b) for a, b in pattern)
    size_options = tuple(
        size for size in FRAME_SIZE_OPTIONS if ((size - 1) // 2 - 1) >= radius
    )
    assert size_options
    return pattern, _marker_from_canon(frozenset(pattern)), size_options


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _sample_frame_pair(span: int, size_options: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    feasible = [
        (s1, s2)
        for s1 in size_options
        for s2 in size_options
        if 2 * BORDER + s1 + FRAME_GAP + s2 <= span
    ]
    if not feasible:
        return None
    s1, s2 = random.choice(feasible)
    slack = span - (2 * BORDER + s1 + FRAME_GAP + s2)
    before, between, _after = _split_total(slack, 3)
    o1 = BORDER + before
    o2 = o1 + s1 + FRAME_GAP + between
    return s1, s2, o1, o2


def _sample_two_frames(H: int, W: int, size_options: tuple[int, ...]) -> list[dict] | None:
    horizontal = random.random() < TWO_FRAME_HORIZONTAL_PROB
    axis = _sample_frame_pair(W if horizontal else H, size_options)
    if axis is None:
        return None
    s1, s2, o1, o2 = axis
    if horizontal:
        row = random.randint(BORDER, H - max(s1, s2) - BORDER)
        return [{"r": row, "c": o1, "size": s1}, {"r": row, "c": o2, "size": s2}]
    col = random.randint(BORDER, W - max(s1, s2) - BORDER)
    return [{"r": o1, "c": col, "size": s1}, {"r": o2, "c": col, "size": s2}]


def _sample_three_frames(H: int, W: int, size_options: tuple[int, ...]) -> list[dict] | None:
    missing = random.choice(QUADRANT_KINDS)
    present = tuple(kind for kind in QUADRANT_KINDS if kind != missing)

    def _bands(sizes: dict[str, int]) -> tuple[int, int, int, int]:
        top = max((sizes[k] for k in present if k[0] == "T"), default=0)
        bot = max((sizes[k] for k in present if k[0] == "B"), default=0)
        left = max((sizes[k] for k in present if k[1] == "L"), default=0)
        right = max((sizes[k] for k in present if k[1] == "R"), default=0)
        return top, bot, left, right

    feasible: list[dict[str, int]] = []
    for combo in product(size_options, repeat=len(present)):
        sizes = dict(zip(present, combo))
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


def _frame_center(frame: dict) -> tuple[int, int]:
    half = frame["size"] // 2
    return frame["r"] + half, frame["c"] + half


def _paint_pattern(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    kind: str,
    pattern: list[tuple[int, int]],
    color: int,
) -> None:
    for off in pattern:
        dr, dc = _quadrant_offset(off, kind)
        grid[center_r + dr, center_c + dc] = color


def _assign_quadrant_colors(frames: list[dict], bg: int) -> None:
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["quadrant_colors"] = dict(zip(QUADRANT_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]]:
    eligible = [(fi, kind) for fi in range(len(frames)) for kind in QUADRANT_KINDS]
    n = random.randint(1, min(N_SOURCES_MAX, len(eligible)))
    return set(random.sample(eligible, n))


def _render_output(base: np.ndarray, frames: list[dict], pattern: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    for frame in frames:
        cr, cc = _frame_center(frame)
        for kind in QUADRANT_KINDS:
            _paint_pattern(out, cr, cc, kind, pattern, frame["quadrant_colors"][kind])
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    marker_canon: tuple[int, int],
    sources: set[tuple[int, str]],
) -> np.ndarray:
    inp = output.copy()
    for fi, frame in enumerate(frames):
        cr, cc = _frame_center(frame)
        for kind in QUADRANT_KINDS:
            if (fi, kind) in sources:
                continue
            mdr, mdc = _quadrant_offset(marker_canon, kind)
            marker = (cr + mdr, cc + mdc)
            for off in pattern:
                dr, dc = _quadrant_offset(off, kind)
                rr, cc2 = cr + dr, cc + dc
                if (rr, cc2) != marker:
                    inp[rr, cc2] = base[rr, cc2]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, marker_canon, size_options = _choose_pattern()

    frames = (
        _sample_two_frames(H, W, size_options)
        if random.random() < N_FRAMES_2_PROB
        else _sample_three_frames(H, W, size_options)
    )
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_quadrant_colors(frames, bg)
    sources = _pick_sources(frames)

    output = _render_output(base, frames, pattern)
    inp = _derive_input(output, base, frames, pattern, marker_canon, sources)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
