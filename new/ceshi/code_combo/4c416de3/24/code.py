# Combo variant of 4c416de3 (index 24)
# combo_spec: D1/4+D7/3
# Name: 行列对齐匹配+非方形矩形框架

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line, get_bg


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original task: every target corner copied one global canonical pattern, so the
#    only decision was the target corner orientation. Here that would degenerate:
#    row/column matching would not affect the output if all sources still shared one
#    pattern. The fused task therefore uses two distinct source patterns in each
#    sample. Example for a horizontal layout:
#    - left rectangle has full sources at TR and BR;
#    - right rectangle has only markers at TL and BL;
#    - TL must copy TR because they share the same row;
#    - BL must copy BR because they share the same row.
#    In a vertical layout the same logic switches to shared columns.
# b) Complexity calibration:
#    Two active source corners and two active target corners are enough to make the
#    alignment rule visible without creating a combinatorial matching problem. Using
#    only one source pattern would collapse back to the original task; using many
#    source patterns would make the examples noisy. Two patterns is the smallest
#    non-degenerate choice.
# c) Broken original assumptions:
#    - "all sources share one canonical pattern" is no longer true;
#    - "the deepest diagonal marker is global" is no longer true, because each
#      source pattern can have its own k_max;
#    - "every non-source corner can be filled from orientation alone" is false;
#      a target now needs a unique aligned source first.
#    The solver therefore extracts each source independently, then tests every
#    non-source corner against aligned sources only.
# d) Placement legality:
#    Frames are generated as clean 6x8 or 6x9 hollow rectangles with >=4 cells
#    between frames. This guarantees no overlaps, keeps patterns inside bounds, and
#    leaves enough empty space for outward source strokes and target markers.
# e) Visual observability:
#    The frame shapes are clearly non-square, and only the side facing the other
#    rectangle is active. In horizontal examples, the relevant pairing is obviously
#    top-row to top-row / bottom-row to bottom-row; in vertical examples, it is
#    left-column to left-column / right-column to right-column.
# f) Visual cleanliness:
#    Height is fixed at 6 so top-corner and bottom-corner patterns on the same side
#    do not touch even when a pattern reaches two cells inward. The candidate offset
#    set is also limited to the library used by the generator, preventing accidental
#    capture of distant colors.
# g) Inferability of color behavior:
#    Colors still come from target markers, not from source colors. This stays
#    inferable because alignment decides the SHAPE, while the marker decides the
#    COLOR; every sample contains both cues in separate visible locations.


CORNER_KINDS = ("TL", "TR", "BL", "BR")

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

# Canonical TL-oriented source shapes. All are diagonally symmetric.
PATTERN_LIBRARY = (
    frozenset({(-1, 0), (0, -1), (1, 1)}),
    frozenset({(-1, -1), (-1, 0), (0, -1), (1, 1)}),
    frozenset({(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)}),
    frozenset({(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1)}),
    frozenset({(-1, 0), (0, -1), (1, 1), (1, 2), (2, 1)}),
    frozenset({(-1, -1), (-1, 0), (0, -1), (1, 1), (1, 2), (2, 1)}),
)
PATTERN_SET = set(PATTERN_LIBRARY)
PATTERN_OFFSETS = tuple(sorted({off for pat in PATTERN_LIBRARY for off in pat}))


def _rot(dr: int, dc: int, kind: str) -> tuple[int, int]:
    """Map actual offsets <-> canonical TL offsets; the mapping is self-inverse."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    if kind == "BR":
        return -dr, -dc
    raise ValueError(f"bad corner kind: {kind}")


def _edge_metric(
    g: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    bg: int,
    H: int,
    W: int,
    max_step: int = 12,
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
                cnt1, saw1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, saw2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 4 and saw1 and cnt2 >= 4 and saw2):
                    continue
                blocked = False
                for odr, odc in OUT_DIRS[kind]:
                    rr, cc = r + odr, c + odc
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == 0:
                        blocked = True
                        break
                if blocked:
                    continue
                corners.append((r, c, kind))
                break
    return corners


def _collect_pattern_items(
    g: np.ndarray, r: int, c: int, kind: str, bg: int
) -> list[tuple[tuple[int, int], int]]:
    H, W = g.shape
    items: list[tuple[tuple[int, int], int]] = []
    for cdr, cdc in PATTERN_OFFSETS:
        adr, adc = _rot(cdr, cdc, kind)
        rr, cc = r + adr, c + adc
        if not (0 <= rr < H and 0 <= cc < W):
            continue
        v = int(g[rr, cc])
        if v == bg or v == 0:
            continue
        items.append(((cdr, cdc), v))
    return items


def _is_outward(off: tuple[int, int]) -> bool:
    return off[0] < 0 or off[1] < 0


def _extract_source(
    items: list[tuple[tuple[int, int], int]]
) -> tuple[frozenset[tuple[int, int]], int, int] | None:
    outward = [(off, v) for off, v in items if _is_outward(off)]
    if len(outward) < 2:
        return None
    colors = {v for _, v in outward}
    assert len(colors) == 1, f"source has mixed outward colors: {colors}"
    scol = next(iter(colors))
    pattern = frozenset(off for off, v in items if v == scol)
    assert pattern in PATTERN_SET, f"unexpected source pattern: {sorted(pattern)}"
    diag_ks = [dr for dr, dc in pattern if dr == dc and dr > 0]
    assert diag_ks, "source pattern lacks an inward diagonal marker anchor"
    return pattern, scol, max(diag_ks)


def _aligned_axis(source: tuple[int, int, str], target: tuple[int, int, str]) -> str | None:
    same_row = source[0] == target[0]
    same_col = source[1] == target[1]
    if same_row ^ same_col:
        return "row" if same_row else "col"
    return None


def _marker_cell(
    corner: tuple[int, int, str], marker_k: int
) -> tuple[int, int]:
    adr, adc = _rot(marker_k, marker_k, corner[2])
    return corner[0] + adr, corner[1] + adc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a source-corner glyph to aligned target corners inside non-square rectangles.

    Input structure:
    - Background is the most common color. There are two hollow 0-rectangles whose
      height and width are intentionally different, making row-vs-column alignment
      visually salient.
    - One rectangle contributes the sources on the side facing the other rectangle:
      each source corner already contains a full painted glyph in one color.
    - The opposite rectangle contributes the targets on its facing side: each target
      corner keeps only a colored marker at that source glyph's deepest inward
      diagonal cell.

    Rule:
    - Detect every full source glyph independently; sources do not need to share the
      same canonical shape.
    - For each non-source corner, look only at sources that share its row or its
      column. If exactly one aligned source also explains the visible marker, copy
      that source's glyph to the target corner, rotated to the target corner kind
      and painted in the marker's color.

    Key insight:
    - Orientation still matters locally because the copied glyph must rotate with the
      target corner, but orientation alone is no longer enough to choose the glyph.
      The decisive clue is which source corner lines up with the target by row or by
      column inside the non-square frame layout.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corners detected"

    per_corner = {corner: _collect_pattern_items(g, *corner, bg) for corner in corners}

    source_info: dict[tuple[int, int, str], dict[str, object]] = {}
    for corner, items in per_corner.items():
        src = _extract_source(items)
        if src is None:
            continue
        pattern, scol, marker_k = src
        source_info[corner] = {
            "pattern": pattern,
            "color": scol,
            "marker_k": marker_k,
        }
    assert source_info, "no source corners detected"

    targets: list[tuple[tuple[int, int, str], frozenset[tuple[int, int]], int]] = []
    for corner in corners:
        if corner in source_info:
            continue
        hits: list[tuple[frozenset[tuple[int, int]], int]] = []
        for source_corner, info in source_info.items():
            axis = _aligned_axis(source_corner, corner)
            if axis is None:
                continue
            marker_r, marker_c = _marker_cell(corner, int(info["marker_k"]))
            if not (0 <= marker_r < H and 0 <= marker_c < W):
                continue
            tcol = int(g[marker_r, marker_c])
            if tcol == bg or tcol == 0:
                continue
            pattern = info["pattern"]
            consistent = True
            for off, color in per_corner[corner]:
                if off in pattern and color != tcol:
                    consistent = False
                    break
            if consistent:
                hits.append((pattern, tcol))
        assert len(hits) <= 1, f"ambiguous aligned source for target {corner}"
        if hits:
            pattern, tcol = hits[0]
            targets.append((corner, pattern, tcol))

    assert targets, "no aligned targets detected"

    out = g.copy()
    for (r, c, kind), pattern, tcol in targets:
        for cdr, cdc in pattern:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f"paint conflict at ({rr},{cc}) for target {(r, c, kind)}: {v}"
            )
            out[rr, cc] = tcol

    return out


# --------------- generator (output-first aligned-source replication) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_HEIGHT = 6
FRAME_WIDTHS = (8, 9)
MAX_ATTEMPTS = 300
_LAYOUT_CASES = (
    ("horizontal", True),
    ("horizontal", False),
    ("vertical", True),
    ("vertical", False),
)
_GEN_COUNTER = 0

_CORNER_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _draw_frame(grid: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    draw_line(grid, r, c, end_r=r, end_c=c + w - 1, color=0)
    draw_line(grid, r, c, end_r=r + h - 1, end_c=c, color=0)
    draw_line(grid, r + h - 1, c, end_r=r + h - 1, end_c=c + w - 1, color=0)
    draw_line(grid, r, c + w - 1, end_r=r + h - 1, end_c=c + w - 1, color=0)


def _corner_coord(frame: dict[str, int], kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    return frame["r"] + dr * (frame["h"] - 1), frame["c"] + dc * (frame["w"] - 1)


def _paint_pattern(
    grid: np.ndarray, r: int, c: int, kind: str, pattern: frozenset[tuple[int, int]], color: int
) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _next_layout_case() -> tuple[int, str, bool]:
    global _GEN_COUNTER
    sample_id = _GEN_COUNTER
    _GEN_COUNTER += 1
    axis, source_first = _LAYOUT_CASES[sample_id % len(_LAYOUT_CASES)]
    return sample_id, axis, source_first


def _active_corner_map(axis: str, source_first: bool) -> tuple[tuple[str, str], tuple[str, str]]:
    if axis == "horizontal":
        if source_first:
            return ("TR", "BR"), ("TL", "BL")
        return ("TL", "BL"), ("TR", "BR")
    if source_first:
        return ("BL", "BR"), ("TL", "TR")
    return ("TL", "TR"), ("BL", "BR")


def _place_frames(axis: str, w: int) -> tuple[int, int, list[dict[str, int]]] | None:
    h = FRAME_HEIGHT
    if axis == "horizontal":
        min_h = h + 2 * BORDER
        min_w = 2 * w + FRAME_GAP + 2 * BORDER
        if min_w > W_RANGE[1]:
            return None
        H = random.randint(max(H_RANGE[0], min_h), H_RANGE[1])
        W = random.randint(max(W_RANGE[0], min_w), W_RANGE[1])
        extra_y = H - min_h
        extra_x = W - min_w
        top_extra, _ = _split_total(extra_y, 2)
        left_extra, between_extra, _ = _split_total(extra_x, 3)
        row = BORDER + top_extra
        c0 = BORDER + left_extra
        c1 = c0 + w + FRAME_GAP + between_extra
        return H, W, [{"r": row, "c": c0, "h": h, "w": w}, {"r": row, "c": c1, "h": h, "w": w}]

    min_h = 2 * h + FRAME_GAP + 2 * BORDER
    min_w = w + 2 * BORDER
    if min_h > H_RANGE[1]:
        return None
    H = random.randint(max(H_RANGE[0], min_h), H_RANGE[1])
    W = random.randint(max(W_RANGE[0], min_w), W_RANGE[1])
    extra_y = H - min_h
    extra_x = W - min_w
    top_extra, between_extra, _ = _split_total(extra_y, 3)
    left_extra, _ = _split_total(extra_x, 2)
    col = BORDER + left_extra
    r0 = BORDER + top_extra
    r1 = r0 + h + FRAME_GAP + between_extra
    return H, W, [{"r": r0, "c": col, "h": h, "w": w}, {"r": r1, "c": col, "h": h, "w": w}]


def _gen_attempt() -> np.ndarray | None:
    sample_id, axis, source_first = _next_layout_case()
    frame_w = FRAME_WIDTHS[(sample_id // len(_LAYOUT_CASES)) % len(FRAME_WIDTHS)]
    placed = _place_frames(axis, frame_w)
    if placed is None:
        return None
    H, W, frames = placed

    bg = random.randint(1, 9)
    palette = [c for c in range(1, 10) if c != bg]
    source_patterns = random.sample(list(PATTERN_LIBRARY), 2)
    source_kinds, target_kinds = _active_corner_map(axis, source_first)
    source_frame = frames[0] if source_first else frames[1]
    target_frame = frames[1] if source_first else frames[0]

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["h"], frame["w"])

    output = base.copy()
    colors = random.sample(palette, 4)
    source_colors = colors[:2]
    for kind, pattern, color in zip(source_kinds, source_patterns, source_colors):
        r, c = _corner_coord(source_frame, kind)
        _paint_pattern(output, r, c, kind, pattern, color)

    target_colors = colors[2:]
    target_specs: list[tuple[str, frozenset[tuple[int, int]], int]] = []
    for kind, pattern, color in zip(target_kinds, source_patterns, target_colors):
        r, c = _corner_coord(target_frame, kind)
        _paint_pattern(output, r, c, kind, pattern, color)
        target_specs.append((kind, pattern, color))

    inp = output.copy()
    for kind, pattern, _color in target_specs:
        r, c = _corner_coord(target_frame, kind)
        marker_k = max(dr for dr, dc in pattern if dr == dc and dr > 0)
        marker = _marker_cell((r, c, kind), marker_k)
        for dr, dc in pattern:
            adr, adc = _rot(dr, dc, kind)
            rr, cc = r + adr, c + adc
            if (rr, cc) != marker:
                inp[rr, cc] = base[rr, cc]

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
