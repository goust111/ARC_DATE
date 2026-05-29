# Combo variant of 4c416de3 (index 6)
# combo_spec: D7/2+D9/0
# Name: 单框架+多源多目标一对一
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace
# - The original task had 2-3 independent frames and one shared canonical corner pattern
#   that broadcast from 1-2 sources to every marker-bearing target.
# - This fused variant collapses the layout to one large complete square 0-frame. All four
#   corners are active: exactly two adjacent corners are sources and the opposite side's
#   two corners are targets.
# - Each source keeps its own diagonally symmetric canonical pattern. The two source
#   patterns are intentionally different: one has deepest marker (1,1), the other (2,2).
# - Each source drives only its diagonally opposite target. Example with top-side sources:
#   TL source pattern A copies only to BR using BR's marker color; TR source pattern B
#   copies only to BL using BL's marker color. The two targets therefore complete with
#   different shapes, not with a single shared broadcast glyph.
#
# b) Complexity calibration
# - Using one frame alone would simplify the scene too much, so the generator makes all
#   four corners active and forces two different source patterns. That keeps the output
#   visually richer than the original while still easy to parse.
# - Pairing is restricted to the unique diagonal-opposite relation. This avoids arbitrary
#   matching freedom while still making the 1:1 rule visible.
#
# c) Implicit-assumption audit for the original main()
# - Original assumption "all sources share one canonical pattern" is false here. The new
#   solver extracts one pattern per source corner.
# - Original assumption "every non-source marker gets the same pattern" is false here.
#   The new solver paints each target only from its paired opposite source.
# - Original corner detection and canonicalization still remain valid because the frame is
#   still axis-aligned and each corner pattern still uses the original reflection-based
#   orientation mapping.
# - New solver assumptions:
#   1. there is exactly one full frame, hence exactly one detected corner of each kind;
#   2. exactly two source corners exist and they are adjacent, not opposite;
#   3. each source pattern has a deepest inward-diagonal marker cell;
#   4. the two targets are exactly the opposite corners of those two sources.
#   The generator enforces all four.
#
# d) Placement legality
# - The generator draws one complete square 0-frame with size 10-14 and keeps at least one
#   background cell around it, so all outward pattern cells stay in-bounds.
# - Pattern radius stays within 2, far smaller than the frame side, so opposite-corner
#   neighborhoods cannot overlap.
# - Output-first construction followed by `main(inp) == out` catches any accidental
#   collision or bad marker placement.
#
# e) Visual observability
# - The fusion is visible from a few examples because target markers appear at two
#   different depths, matching the two different source patterns.
# - A human can see both cues at once: the source on the opposite diagonal gives the shape,
#   while the local marker gives the target color.
#
# f) Visual cleanliness
# - There is only one frame region, so attention is not split across disconnected subplots.
# - Each corner neighborhood holds one monochrome local glyph or one single marker, which
#   keeps the puzzle tidy even though all four corners are active.
#
# g) Color inferability
# - No hidden color ordering is used. Source colors only identify the visible source
#   exemplars; target outputs still use the target corner's own marker color.
# - The matching rule is positional (diagonal opposite), not based on numeric color ids.

import random

import numpy as np

from arc_utils import draw_line, get_bg


CORNER_KINDS = ("TL", "TR", "BL", "BR")
OPPOSITE_KIND = {"TL": "BR", "TR": "BL", "BL": "TR", "BR": "TL"}
ADJACENT_SOURCE_PAIRS = (
    ("TL", "TR"),
    ("TR", "BR"),
    ("BL", "BR"),
    ("TL", "BL"),
)

EDGE_DIRS = {
    "TL": ((+1, 0), (0, +1)),
    "TR": ((+1, 0), (0, -1)),
    "BL": ((-1, 0), (0, +1)),
    "BR": ((-1, 0), (0, -1)),
}

OUT_DIRS = {
    "TL": ((-1, 0), (0, -1)),
    "TR": ((-1, 0), (0, +1)),
    "BL": ((+1, 0), (0, -1)),
    "BR": ((+1, 0), (0, +1)),
}

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}
_NEIGH_R = 3

# Keep the original diagonal-symmetric corner-glyph language, but split templates by the
# depth of their inward marker so the two source-target pairs are visibly distinct.
PATTERNS_K1 = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1)],
)
PATTERNS_K2 = (
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)

H_RANGE = (18, 24)
W_RANGE = (18, 24)
FRAME_SIZE_RANGE = (10, 14)
BORDER = 1
MAX_ATTEMPTS = 400
_GEN_PAIR_INDEX = 0
_GEN_SHALLOW_INDEX = 0
_GEN_DEEP_INDEX = 0
_GEN_SWAP_FLAG = 0


def _rot(dr: int, dc: int, kind: str) -> tuple[int, int]:
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


def _is_outward(off: tuple[int, int]) -> bool:
    return off[0] < 0 or off[1] < 0


def _edge_metric(
    g: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    bg: int,
    H: int,
    W: int,
    max_step: int = 20,
) -> tuple[int, bool]:
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
    corners: list[tuple[int, int, str]] = []
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
                cnt1, saw_zero1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, saw_zero2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 3 and saw_zero1 and cnt2 >= 3 and saw_zero2):
                    continue
                ok = True
                for odr, odc in OUT_DIRS[kind]:
                    rr, cc = r + odr, c + odc
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == 0:
                        ok = False
                        break
                if ok:
                    corners.append((r, c, kind))
                    break
    return corners


def _colored_around(
    g: np.ndarray, r: int, c: int, kind: str, bg: int
) -> list[tuple[tuple[int, int], int]]:
    H, W = g.shape
    items: list[tuple[tuple[int, int], int]] = []
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


def _extract_source_pattern(
    items: list[tuple[tuple[int, int], int]]
) -> tuple[frozenset[tuple[int, int]], int, tuple[int, int]]:
    outward = [(off, color) for off, color in items if _is_outward(off)]
    colors = {color for _, color in outward}
    assert len(outward) >= 2 and len(colors) == 1, "source must have a monochrome outward tail"
    src_color = next(iter(colors))
    canon = frozenset(off for off, color in items if color == src_color)
    for dr, dc in canon:
        assert (dc, dr) in canon, f"pattern not diagonally symmetric: {(dr, dc)}"
    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "source pattern lacks an inward-diagonal marker"
    k_max = max(diag_ks)
    return canon, src_color, (k_max, k_max)


def _frame_corner_map(corners: list[tuple[int, int, str]]) -> dict[str, tuple[int, int, str]]:
    assert len(corners) == 4, f"expected exactly 4 corners, got {len(corners)}"
    by_kind = {kind: (r, c, kind) for r, c, kind in corners}
    assert set(by_kind) == set(CORNER_KINDS), f"missing corner kinds: {set(CORNER_KINDS) - set(by_kind)}"
    tl, tr, bl, br = by_kind["TL"], by_kind["TR"], by_kind["BL"], by_kind["BR"]
    assert tl[0] == tr[0] and bl[0] == br[0], "top/bottom rows mismatch"
    assert tl[1] == bl[1] and tr[1] == br[1], "left/right cols mismatch"
    assert (tr[1] - tl[1]) == (bl[0] - tl[0]), "frame is not square"
    return by_kind


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete two target corners inside one large frame by copying from their paired opposite sources.

    Input structure:
    - The grid has one hollow square 0-frame on a solid non-zero background, and all four
      corners of that frame are active.
    - Exactly two adjacent corners are sources: each already shows a full monochrome corner
      pattern, and the two source patterns are allowed to differ.
    - The other two corners are targets: each keeps only one colored marker at the deepest
      inward-diagonal cell of its own paired source pattern.

    Rule:
    - Detect the two source corners from their outward colored tail cells and extract one
      canonical TL-oriented pattern from each source independently.
    - Pair every source with its diagonally opposite corner, read that target marker's color,
      and paint only that source's pattern at only that opposite target, rotated to the
      target's corner orientation.

    Key insight:
    - This is no longer a single shared pattern broadcast to every marker. The frame holds
      two separate source exemplars, and each target borrows shape from the source across its
      diagonal while keeping its own local marker color.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corner_map = _frame_corner_map(_find_corners(g, bg))
    per_corner = {
        kind: _colored_around(g, corner_map[kind][0], corner_map[kind][1], kind, bg)
        for kind in CORNER_KINDS
    }

    sources = [kind for kind in CORNER_KINDS if sum(1 for off, _ in per_corner[kind] if _is_outward(off)) >= 2]
    assert len(sources) == 2, f"expected 2 source corners, got {sources}"
    assert OPPOSITE_KIND[sources[0]] != sources[1], "sources must be adjacent, not opposite"

    source_specs: dict[str, tuple[frozenset[tuple[int, int]], tuple[int, int]]] = {}
    for kind in sources:
        canon, _src_color, marker_canon = _extract_source_pattern(per_corner[kind])
        source_specs[kind] = (canon, marker_canon)
    assert len({source_specs[k][0] for k in sources}) == 2, "two distinct source patterns are required"

    out = g.copy()
    for src_kind in sources:
        tgt_kind = OPPOSITE_KIND[src_kind]
        assert tgt_kind not in sources, "paired opposite corner unexpectedly also classified as source"
        canon, marker_canon = source_specs[src_kind]
        tr, tc, _ = corner_map[tgt_kind]
        mdr, mdc = _rot(*marker_canon, tgt_kind)
        mr, mc = tr + mdr, tc + mdc
        assert 0 <= mr < H and 0 <= mc < W, "target marker out of bounds"
        tcol = int(g[mr, mc])
        assert tcol != bg and tcol != 0, f"missing marker at target {tgt_kind}"
        for off, color in per_corner[tgt_kind]:
            if off in canon:
                assert color == tcol, f"target {tgt_kind} has conflicting color at {off}"
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, tgt_kind)
            rr, cc = tr + adr, tc + adc
            assert 0 <= rr < H and 0 <= cc < W, "paint out of bounds"
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, f"paint conflict at {(rr, cc)}"
            out[rr, cc] = tcol

    return out


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    span = frame["size"] - 1
    return frame["r"] + dr * span, frame["c"] + dc * span


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


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


def _choose_frame(H: int, W: int) -> dict | None:
    max_size = min(FRAME_SIZE_RANGE[1], H - 2 * BORDER, W - 2 * BORDER)
    if max_size < FRAME_SIZE_RANGE[0]:
        return None
    size = random.randint(FRAME_SIZE_RANGE[0], max_size)
    r = random.randint(BORDER, H - size - BORDER)
    c = random.randint(BORDER, W - size - BORDER)
    return {"r": r, "c": c, "size": size}


def _choose_pattern_pair() -> dict[str, list[tuple[int, int]]]:
    global _GEN_SHALLOW_INDEX, _GEN_DEEP_INDEX
    p1 = list(PATTERNS_K1[_GEN_SHALLOW_INDEX % len(PATTERNS_K1)])
    p2 = list(PATTERNS_K2[_GEN_DEEP_INDEX % len(PATTERNS_K2)])
    _GEN_SHALLOW_INDEX += 1
    _GEN_DEEP_INDEX += 1
    return {"shallow": p1, "deep": p2}


def _target_marker(pattern: list[tuple[int, int]]) -> tuple[int, int]:
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return k_max, k_max


def _render_output(
    base: np.ndarray,
    frame: dict,
    source_patterns: dict[str, list[tuple[int, int]]],
    corner_colors: dict[str, int],
) -> np.ndarray:
    out = base.copy()
    for kind, pattern in source_patterns.items():
        r, c = _corner_coord(frame, kind)
        _paint_pattern(out, r, c, kind, pattern, corner_colors[kind])
    for src_kind, pattern in source_patterns.items():
        tgt_kind = OPPOSITE_KIND[src_kind]
        r, c = _corner_coord(frame, tgt_kind)
        _paint_pattern(out, r, c, tgt_kind, pattern, corner_colors[tgt_kind])
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frame: dict,
    source_patterns: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    inp = output.copy()
    source_kinds = set(source_patterns)
    for kind in CORNER_KINDS:
        if kind in source_kinds:
            continue
        paired_source = OPPOSITE_KIND[kind]
        pattern = source_patterns[paired_source]
        marker = _target_marker(pattern)
        r, c = _corner_coord(frame, kind)
        mdr, mdc = _rot(*marker, kind)
        keep = (r + mdr, c + mdc)
        for dr, dc in pattern:
            rr, cc = _rot(dr, dc, kind)
            rr += r
            cc += c
            if (rr, cc) != keep:
                inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    global _GEN_PAIR_INDEX, _GEN_SWAP_FLAG
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    frame = _choose_frame(H, W)
    if frame is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    _draw_frame(base, frame["r"], frame["c"], frame["size"])

    source_pair = ADJACENT_SOURCE_PAIRS[_GEN_PAIR_INDEX % len(ADJACENT_SOURCE_PAIRS)]
    _GEN_PAIR_INDEX += 1
    source_patterns = _choose_pattern_pair()
    # Randomize which of the two adjacent sources gets the shallow vs deep pattern.
    pattern_by_kind = {
        source_pair[0]: source_patterns["shallow"],
        source_pair[1]: source_patterns["deep"],
    }
    if _GEN_SWAP_FLAG % 2 == 1:
        pattern_by_kind[source_pair[0]], pattern_by_kind[source_pair[1]] = (
            pattern_by_kind[source_pair[1]],
            pattern_by_kind[source_pair[0]],
        )
    _GEN_SWAP_FLAG += 1

    colors = random.sample([c for c in range(1, 10) if c != bg], 4)
    corner_colors = dict(zip(CORNER_KINDS, colors))

    output = _render_output(base, frame, pattern_by_kind, corner_colors)
    inp = _derive_input(output, base, frame, pattern_by_kind)

    solved = main(inp)
    if not np.array_equal(solved, output):
        return None

    meta = {
        "source_pair": source_pair,
        "pattern_sizes": {k: len(v) for k, v in pattern_by_kind.items()},
        "marker_depths": {k: _target_marker(v)[0] for k, v in pattern_by_kind.items()},
    }
    return inp, output, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        got = _gen_attempt()
        if got is not None:
            return got[0]
    raise RuntimeError("failed to generate a valid input after repeated attempts")
