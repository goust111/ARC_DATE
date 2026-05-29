# Combo variant of 4c416de3 (index 18)
# combo_spec: D7/2+D13/1
# Name: 单框架+中心锚点
#
# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Original 4c416de3 anchors a shared glyph at frame corners. A source corner already holds
#      the full corner-tail pattern, and each target corner keeps only one inward-diagonal marker.
#    - D7/2 collapses the layout from 2-3 separate frames to one large frame, so all four usable
#      slots now belong to the same object. D13/1 simultaneously moves the anchor from corners to
#      the frame center, so the four slots become the four interior quadrants around that center.
#    - Example trace on an  11x11 frame: suppose the canonical TL motif uses positive center
#      offsets {(1,1),(1,2),(3,2)}. In the actual TL quadrant these appear at center-relative
#      cells {(-1,-1),(-1,-2),(-3,-2)}. If BR is only a marker target, it shows just the farthest
#      cell at (+3,+2). The output copies the same three magnitudes into BR by sign flip, producing
#      {(+1,+1),(+1,+2),(+3,+2)} in the BR marker color. No frame corner participates anymore.
# b) Complexity calibration:
#    - Using one frame could over-simplify the puzzle if the motif were too small or symmetric.
#      To avoid that, motifs are asymmetric within a quadrant, have 3-4 cells, and always contain
#      a unique farthest marker cell.
#    - The frame remains visually large by restricting it to odd side lengths 11-15 when feasible,
#      so the center anchor is obvious and the change from corner-based reasoning is easy to see.
#    - There are 1-2 source quadrants and therefore 2-3 targets in the same frame, which gives
#      repeated evidence without introducing extra frames that would blur the fusion idea.
# c) Hidden assumptions from the original solver that no longer hold:
#    - Corner detection from adjacent 0-edges is obsolete; the solver must detect a full hollow
#      square 0-ring and derive a unique bbox center from it.
#    - Corner-oriented offset normalization and outward-tail checks are invalid because motifs now
#      live entirely inside the frame and are classified by quadrant sign relative to the center.
#    - The original generator assumed separate frames could host independent corners. Here the
#      generator must ensure one frame contains all four quadrants, with enough interior radius
#      that every motif cell lies strictly inside the border.
# d) Placement legality:
#    - The generator draws exactly one odd square 0-frame, keeps a border margin from the grid
#      edge, and chooses only motifs whose maximum offset fits within the frame interior.
#    - Motif cells never touch 0 edges or the center axes, so different quadrants stay disjoint
#      and no output paint conflicts occur.
# e) Visual observability:
#    - Across examples, all colored evidence is clustered around one frame center rather than at
#      corners. A human can infer that the four quadrants of one object are the slots being linked.
#    - Each target keeps its own colored marker at the farthest motif cell, making the output color
#      source directly visible without any hidden color rule.
# f) Visual cleanliness:
#    - There is only one large frame and at most one compact motif per quadrant. This avoids the
#      clutter that multiple frames would create after moving the action into frame interiors.
# g) Color inference:
#    - Each quadrant gets a distinct non-background color in generation, so the copied color rule
#      is legible from position alone. The solver still supports repeated colors, but the generator
#      prefers distinct colors to keep the transformation easy to read.
# h) Fusion-specific non-degeneracy target:
#    - Samples must simultaneously show "single frame" and "center anchor". A useful diagnostic is
#      that the grid contains exactly one 0-ring and zero painted cells in the 2-cell corner zones
#      of that ring; a corner-anchored degeneration would violate the second condition.
# i) New solver assumptions that the generator must guarantee:
#    - There is exactly one full odd square 0-frame.
#    - Every source quadrant is monochrome and all source quadrants share the same canonical offset
#      set.
#    - Every non-source quadrant keeps exactly the unique farthest canonical offset as its marker.

import random

import numpy as np

from arc_utils import draw_line, get_bg


QUADRANT_KINDS = ("TL", "TR", "BL", "BR")
QUAD_SIGNS = {
    "TL": (-1, -1),
    "TR": (-1, +1),
    "BL": (+1, -1),
    "BR": (+1, +1),
}

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_SIZE_OPTIONS = (11, 13, 15)
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
        if size != c1 - c0 + 1 or size < 9 or size % 2 == 0:
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


def _colored_quadrants(
    g: np.ndarray, frame: dict, bg: int
) -> dict[str, list[tuple[tuple[int, int], int]]]:
    H, W = g.shape
    cr, cc = frame["center"]
    half = frame["size"] // 2
    items = {kind: [] for kind in QUADRANT_KINDS}
    for dr in range(-(half - 1), half):
        for dc in range(-(half - 1), half):
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
    Complete every marker-bearing quadrant of one large hollow frame from a center-anchored source motif.

    Input structure:
    - The background is the most common color. There is exactly one large hollow square 0-frame
      with odd side length, so it has a single center pixel that acts as the anchor.
    - One or two quadrants already contain the full motif in a single color. The motif lives
      strictly inside the frame interior and is described relative to the frame center.
    - Each remaining active quadrant keeps only one colored marker at the motif's unique farthest
      offset from the center; that marker color is the color to use in the output.

    Rule:
    - Detect the unique square 0-ring, compute its center, and read colored cells by quadrant.
    - Extract one canonical motif from any source quadrant as TL-style positive offsets from the
      center, verifying that all source quadrants share the same offset set.
    - For every non-source quadrant whose marker is present, mirror the canonical offsets into
      that quadrant by sign choice and paint the motif in the marker's color.

    Key insight: the repeated relationship is no longer "same shape at different corners"; it is
    "same offset magnitudes from one shared center" mapped across the four quadrants of one frame.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _find_frames(g)
    assert len(frames) == 1, f"expected one frame, found {len(frames)}"
    frame = frames[0]

    per_quad = _colored_quadrants(g, frame, bg)

    sources: list[str] = []
    for kind, items in per_quad.items():
        if len(items) < 2:
            continue
        colors = {v for _, v in items}
        assert len(colors) == 1, f"source quadrant {kind} has mixed colors: {colors}"
        sources.append(kind)
    assert sources, "at least one source quadrant is required"

    def _extract(kind: str) -> frozenset[tuple[int, int]]:
        return frozenset(off for off, _ in per_quad[kind])

    canon = _extract(sources[0])
    for kind in sources[1:]:
        other = _extract(kind)
        assert other == canon, f"source motifs disagree: {sorted(canon)} vs {sorted(other)}"

    marker_canon = _marker_from_canon(canon)

    cr, cc = frame["center"]
    src_set = set(sources)
    targets: list[tuple[str, int]] = []
    for kind in QUADRANT_KINDS:
        if kind in src_set:
            continue
        mdr, mdc = _quadrant_offset(marker_canon, kind)
        mr, mc = cr + mdr, cc + mdc
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        v = int(g[mr, mc])
        if v == bg or v == 0:
            continue
        tcol = v
        for off, color in per_quad[kind]:
            if off in canon:
                assert color == tcol, (
                    f"target quadrant {kind} has conflicting color {color} at {off}; expected {tcol}"
                )
        targets.append((kind, tcol))

    assert targets, "no target quadrants detected"

    out = g.copy()
    for kind, tcol in targets:
        for off in canon:
            dr, dc = _quadrant_offset(off, kind)
            rr, cc2 = cr + dr, cc + dc
            if not (0 <= rr < H and 0 <= cc2 < W):
                continue
            v = int(out[rr, cc2])
            assert v == bg or v == tcol, (
                f"paint conflict at {(rr, cc2)} for target {kind} color {tcol}: existing {v}"
            )
            out[rr, cc2] = tcol

    return out


def _feasible_frame_sizes(H: int, W: int) -> list[int]:
    return [size for size in FRAME_SIZE_OPTIONS if size <= H - 2 * BORDER and size <= W - 2 * BORDER]


def _choose_pattern() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    pattern = list(random.choice(PATTERN_TEMPLATES))
    return pattern, _marker_from_canon(frozenset(pattern))


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _sample_frame(H: int, W: int, pattern: list[tuple[int, int]]) -> dict | None:
    radius = max(max(a, b) for a, b in pattern)
    feasible = [
        size
        for size in _feasible_frame_sizes(H, W)
        if ((size - 1) // 2 - 1) >= radius
    ]
    if not feasible:
        return None
    size = random.choice(feasible)
    r = random.randint(BORDER, H - size - BORDER)
    c = random.randint(BORDER, W - size - BORDER)
    return {"r": r, "c": c, "size": size, "center": (r + size // 2, c + size // 2)}


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


def _assign_quadrant_colors(bg: int) -> dict[str, int]:
    palette = [c for c in range(1, 10) if c != bg]
    return dict(zip(QUADRANT_KINDS, random.sample(palette, 4)))


def _pick_sources() -> set[str]:
    n = random.randint(1, N_SOURCES_MAX)
    return set(random.sample(QUADRANT_KINDS, n))


def _render_output(base: np.ndarray, frame: dict, pattern: list[tuple[int, int]], colors: dict[str, int]) -> np.ndarray:
    out = base.copy()
    cr, cc = frame["center"]
    for kind in QUADRANT_KINDS:
        _paint_pattern(out, cr, cc, kind, pattern, colors[kind])
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frame: dict,
    pattern: list[tuple[int, int]],
    marker_canon: tuple[int, int],
    colors: dict[str, int],
    sources: set[str],
) -> np.ndarray:
    inp = output.copy()
    cr, cc = frame["center"]
    for kind in QUADRANT_KINDS:
        if kind in sources:
            continue
        mdr, mdc = _quadrant_offset(marker_canon, kind)
        marker = (cr + mdr, cc + mdc)
        for off in pattern:
            dr, dc = _quadrant_offset(off, kind)
            rr, cc2 = cr + dr, cc + dc
            if (rr, cc2) != marker:
                inp[rr, cc2] = base[rr, cc2]
        inp[marker] = colors[kind]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, marker_canon = _choose_pattern()
    frame = _sample_frame(H, W, pattern)
    if frame is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    _draw_frame(base, frame["r"], frame["c"], frame["size"])

    colors = _assign_quadrant_colors(bg)
    sources = _pick_sources()
    output = _render_output(base, frame, pattern, colors)
    inp = _derive_input(output, base, frame, pattern, marker_canon, colors, sources)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
