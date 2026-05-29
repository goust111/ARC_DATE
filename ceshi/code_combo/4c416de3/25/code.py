# Combo variant of 4c416de3 (index 25)
# combo_spec: D2/2+D13/2
# Name: 边缘精灵+边缘中点锚定
#
# Design analysis
# ----------------
# a) Concrete behavior trace
#    - Original task: a source glyph sits on a frame corner, targets are other corners,
#      and the only surviving hint on a target is the deepest inward diagonal marker.
#    - Fused task: the anchor moves from the corner to the midpoint of a frame side.
#      Example: for a top-side source, a canonical sprite can use
#      {(-1,-1),(-1,0),(-1,1),(0,0),(1,0)} around the top midpoint, where positive
#      dr points inward and dc runs along the edge. A right-side target keeps only
#      the rotated deepest inward marker, then the output restores the full sprite in
#      that target color. The visible difference from the original task is structural:
#      corners stay inert while the four edge midpoints now carry the action.
# b) Complexity calibration
#    - The anchor class is restricted to exactly four side kinds
#      (TOP/RIGHT/BOTTOM/LEFT), mirroring the original four corners.
#    - Sprites are limited to 4-6 cells, are mirrored across the midpoint centerline
#      dc==0, and use only one inward stem cell per depth. This keeps the rule
#      legible while avoiding ambiguous free-form edge clutter.
#    - Each frame gives its four sides distinct colors. That prevents a small frame's
#      nearby markers from contaminating source extraction when local windows overlap.
# c) Hidden-assumption audit versus the original solver
#    - `_find_corners(...)` assumed L-junctions of two frame edges. That is invalid
#      here because anchor significance no longer comes from corners at all.
#    - `_rot(...)` used sign flips around a TL-style diagonal. The fused rule instead
#      needs a side-local basis: inward normal plus tangential edge direction.
#    - `diag_ks` and `(k,k)` markers assumed diagonal inward trails. The new anchor
#      marker is on the side normal, so the preserved target clue is `(k,0)`.
#    - Corner neighborhoods could ignore nearby corners because the original geometry
#      separated them. Midpoint neighborhoods are closer on 5x5 frames, so generator
#      construction must guarantee side colors are distinct within each frame.
# d) Placement legality
#    - Frames are full odd-sized hollow 0-squares only, so every side has a unique
#      midpoint cell. Sizes stay in {5,7}; if any frame is size 5 the generator only
#      uses depth-1 sprites, preventing opposite-side stems from colliding at center.
#    - Border and frame gaps are chosen so outward sprite pixels never leave the grid
#      and sprites from different frames cannot overlap.
# e) Visual observability
#    - In every sample, corners remain plain 0-frame corners while every active hint
#      is centered on a side midpoint. A human can spot that all four sides behave
#      like rotated copies of the same small edge sprite.
# f) Visual cleanliness
#    - There is only one layer of annotation per side: either a full monochrome sprite
#      (source) or a single deepest-stem marker (target). No mixed-color overlays are
#      allowed on one side.
# g) Inferability of changes
#    - The color rule remains simple and local: the copied sprite always uses the
#      target marker's own color. Because the anchor shift is geometric rather than
#      numeric, the rule is visible from position and orientation alone.

from collections import Counter, deque
import random

import numpy as np

from arc_utils import draw_line, get_bg


SIDE_KINDS = ("TOP", "RIGHT", "BOTTOM", "LEFT")

# For each side, canonical coordinates mean:
#   dr > 0: move inward from the frame edge
#   dc > 0: move along the edge in the positive tangential direction
SIDE_BASIS = {
    "TOP": ((+1, 0), (0, +1)),
    "RIGHT": ((0, -1), (+1, 0)),
    "BOTTOM": ((-1, 0), (0, -1)),
    "LEFT": ((0, +1), (-1, 0)),
}

NEIGH_R = 3
OUTWARD_MAX = 2
TANGENTIAL_MAX = 1
FRAME_SIZE_CHOICES = (5, 7)


def _to_actual(dr: int, dc: int, side: str) -> tuple[int, int]:
    """Convert canonical side-local offsets to actual grid offsets."""
    (nr, nc), (tr, tc) = SIDE_BASIS[side]
    return dr * nr + dc * tr, dr * nc + dc * tc


def _to_canon(dr: int, dc: int, side: str) -> tuple[int, int]:
    """Convert actual grid offsets back to canonical top-side coordinates."""
    (nr, nc), (tr, tc) = SIDE_BASIS[side]
    return dr * nr + dc * nc, dr * tr + dc * tc


def _is_side_window(cdr: int, cdc: int) -> bool:
    return -OUTWARD_MAX <= cdr <= OUTWARD_MAX and abs(cdc) <= TANGENTIAL_MAX


def _zero_components(g: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) != 0:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(g: np.ndarray) -> list[dict]:
    """Detect odd-sized hollow 0-squares, allowing colored midpoint cells to be missing."""
    frames = []
    for comp in _zero_components(g):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        size = r1 - r0 + 1
        if size != c1 - c0 + 1:
            continue
        if size not in FRAME_SIZE_CHOICES:
            continue
        if size % 2 == 0:
            continue
        border_only = all(r in (r0, r1) or c in (c0, c1) for r, c in comp)
        if not border_only:
            continue
        # A full ring has 4*size-4 zeros. Up to one midpoint cell per side may be recolored.
        if len(comp) < (4 * size - 8):
            continue
        frames.append({"r": r0, "c": c0, "size": size})
    frames.sort(key=lambda f: (f["r"], f["c"], f["size"]))
    return frames


def _midpoint_coord(frame: dict, side: str) -> tuple[int, int]:
    mid = frame["size"] // 2
    r0, c0 = frame["r"], frame["c"]
    if side == "TOP":
        return r0, c0 + mid
    if side == "RIGHT":
        return r0 + mid, c0 + frame["size"] - 1
    if side == "BOTTOM":
        return r0 + frame["size"] - 1, c0 + mid
    return r0 + mid, c0


def _frame_corners(frame: dict) -> list[tuple[int, int]]:
    r0, c0, s = frame["r"], frame["c"], frame["size"]
    r1, c1 = r0 + s - 1, c0 + s - 1
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _colored_around(g: np.ndarray, r: int, c: int, side: str, bg: int) -> list[tuple[tuple[int, int], int]]:
    H, W = g.shape
    items = []
    for dr in range(-NEIGH_R, NEIGH_R + 1):
        for dc in range(-NEIGH_R, NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            cdr, cdc = _to_canon(dr, dc, side)
            if _is_side_window(cdr, cdc):
                items.append(((cdr, cdc), v))
    return items


def _extract_task(g: np.ndarray) -> dict:
    g = np.asarray(g, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _find_frames(g)
    assert frames, "no midpoint-anchored frame detected"

    anchors = []
    for fi, frame in enumerate(frames):
        for side in SIDE_KINDS:
            r, c = _midpoint_coord(frame, side)
            anchors.append((fi, side, r, c))

    per_anchor = {
        a: _colored_around(g, a[2], a[3], a[1], bg)
        for a in anchors
    }

    def _is_outward(off: tuple[int, int]) -> bool:
        return off[0] < 0

    sources = []
    for anchor, items in per_anchor.items():
        outward = [(off, v) for off, v in items if _is_outward(off)]
        if len(outward) >= 2:
            colors = {v for _, v in outward}
            assert len(colors) == 1, f"source {anchor} has mixed outward colors: {colors}"
            sources.append(anchor)
    assert sources, "at least one source side is required"

    def _extract(anchor: tuple[int, str, int, int]) -> tuple[frozenset[tuple[int, int]], int]:
        items = per_anchor[anchor]
        counts = Counter(v for _, v in items)
        pat_col, _ = counts.most_common(1)[0]
        return frozenset(off for off, v in items if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for anchor in sources[1:]:
        other, _ = _extract(anchor)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dr, -dc) in canon, f"pattern not side-symmetric: ({dr},{dc}) has no mirror"

    stem_ds = [dr for dr, dc in canon if dr > 0 and dc == 0]
    assert stem_ds, "pattern has no inward stem marker"
    marker_canon = (max(stem_ds), 0)

    src_set = set(sources)
    targets = []
    for anchor in anchors:
        if anchor in src_set:
            continue
        _fi, side, r, c = anchor
        mdr, mdc = _to_actual(*marker_canon, side)
        mr, mc = r + mdr, c + mdc
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        v = int(g[mr, mc])
        if v == bg or v == 0:
            continue
        tcol = v
        for off, color in per_anchor[anchor]:
            if off in canon:
                assert color == tcol, (
                    f"target {anchor} has conflicting color {color} at canonical offset {off}; "
                    f"expected {tcol}"
                )
        targets.append((anchor, tcol))

    assert targets, "no target side detected"

    return {
        "grid": g,
        "bg": bg,
        "frames": frames,
        "anchors": anchors,
        "per_anchor": per_anchor,
        "sources": sources,
        "canon": canon,
        "marker_canon": marker_canon,
        "targets": targets,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a midpoint-anchored edge sprite from source sides to other marked sides.

    Input structure:
    - bg is the most common non-zero background color. The structural objects are
      odd-sized hollow 0-squares, and each of the four side midpoints acts like an
      anchor position instead of the corners doing so.
    - One or more source sides already show the full monochrome sprite centered on a
      side midpoint, with some cells outside the frame and a short stem extending
      inward from that midpoint.
    - Other sides keep only a single colored marker at the deepest inward stem cell;
      that marker's color is the color the completed sprite should use on output.

    Rule:
    - Detect every frame, convert each side midpoint neighborhood into a canonical
      side-local coordinate system, and extract the shared sprite from any source side.
    - For every non-source side whose deepest inward marker is colored, paint the same
      sprite at that side midpoint after rotating it into the side's orientation, using
      the marker's own color.

    Key insight:
    - The geometric role once played by a corner now belongs to a side midpoint, so
      the correct local frame is "inward normal + edge tangent" rather than a corner
      diagonal. The sprite is mirrored across the midpoint centerline `dc==0`, which
      makes one canonical edge sprite enough for all four sides.
    """
    info = _extract_task(np.asarray(grid, dtype=int))
    g = info["grid"]
    out = g.copy()

    for anchor, tcol in info["targets"]:
        _fi, side, r, c = anchor
        for cdr, cdc in info["canon"]:
            adr, adc = _to_actual(cdr, cdc, side)
            rr, cc = r + adr, c + adc
            v = int(out[rr, cc])
            assert v == info["bg"] or v == 0 or v == tcol, (
                f"paint conflict at ({rr},{cc}) for target {anchor}: existing color {v}, new {tcol}"
            )
            out[rr, cc] = tcol
    return out


# --------------- generator (output-first midpoint-edge replication) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 2
FRAME_GAP = 4
N_FRAMES_2_PROB = 0.55
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 3
MAX_ATTEMPTS = 500
DEEP_PATTERN_PROB = 0.4

PATTERN_TEMPLATES = {
    1: (
        [(-1, -1), (-1, 1), (0, 0), (1, 0)],
        [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
        [(-2, 0), (-1, -1), (-1, 1), (0, 0), (1, 0)],
        [(-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
    ),
    2: (
        [(-1, -1), (-1, 1), (0, 0), (1, 0), (2, 0)],
        [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0), (2, 0)],
        [(-2, 0), (-1, -1), (-1, 1), (0, 0), (1, 0), (2, 0)],
    ),
}


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern(max_depth: int, force_depth: int | None = None) -> tuple[list[tuple[int, int]], int]:
    if force_depth is not None:
        assert 1 <= force_depth <= max_depth
        depth = force_depth
    else:
        depth_choices = [1] if max_depth == 1 else [1, 2]
        depth = random.choice(depth_choices)
    pattern = list(random.choice(PATTERN_TEMPLATES[depth]))
    k_max = max(dr for dr, dc in pattern if dr > 0 and dc == 0)
    return pattern, k_max


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _sample_two_frames(H: int, W: int,
                       allowed_sizes: tuple[int, ...] = FRAME_SIZE_CHOICES) -> list[dict] | None:
    candidates = []
    for horizontal in (True, False):
        span = W if horizontal else H
        cross = H if horizontal else W
        for s1 in allowed_sizes:
            for s2 in allowed_sizes:
                if 2 * BORDER + s1 + FRAME_GAP + s2 > span:
                    continue
                if 2 * BORDER + max(s1, s2) > cross:
                    continue
                slack_main = span - (2 * BORDER + s1 + FRAME_GAP + s2)
                slack_cross = cross - (2 * BORDER + max(s1, s2))
                candidates.append((horizontal, s1, s2, slack_main, slack_cross))
    if not candidates:
        return None
    horizontal = random.random() < TWO_FRAME_HORIZONTAL_PROB
    orient_candidates = [c for c in candidates if c[0] == horizontal]
    if not orient_candidates:
        orient_candidates = candidates
    horizontal, s1, s2, slack_main, slack_cross = random.choice(orient_candidates)
    before, between, _after = _split_total(slack_main, 3)
    cross_before, _ = _split_total(slack_cross, 2)
    if horizontal:
        row = BORDER + cross_before
        c1 = BORDER + before
        c2 = c1 + s1 + FRAME_GAP + between
        return [{"r": row, "c": c1, "size": s1}, {"r": row, "c": c2, "size": s2}]
    col = BORDER + cross_before
    r1 = BORDER + before
    r2 = r1 + s1 + FRAME_GAP + between
    return [{"r": r1, "c": col, "size": s1}, {"r": r2, "c": col, "size": s2}]


def _sample_three_frames(H: int, W: int,
                         allowed_sizes: tuple[int, ...] = FRAME_SIZE_CHOICES) -> list[dict] | None:
    missing = random.choice(SIDE_KINDS)
    # Reuse the four-corner placement idea but remove one corner slot.
    all_slots = ("TL", "TR", "BL", "BR")
    absent = {
        "TOP": "TL",
        "RIGHT": "TR",
        "BOTTOM": "BR",
        "LEFT": "BL",
    }[missing]
    present = tuple(slot for slot in all_slots if slot != absent)

    def _bands(sizes: dict[str, int]) -> tuple[int, int, int, int]:
        top = max((sizes[s] for s in present if s[0] == "T"), default=0)
        bot = max((sizes[s] for s in present if s[0] == "B"), default=0)
        left = max((sizes[s] for s in present if s[1] == "L"), default=0)
        right = max((sizes[s] for s in present if s[1] == "R"), default=0)
        return top, bot, left, right

    feasible = []
    for combo in np.ndindex(*(len(allowed_sizes) for _ in present)):
        sizes = {slot: allowed_sizes[i] for slot, i in zip(present, combo)}
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


def _sample_frames(H: int, W: int, deep_pattern: bool = False) -> list[dict] | None:
    allowed_sizes = (7,) if deep_pattern else FRAME_SIZE_CHOICES
    if random.random() < N_FRAMES_2_PROB:
        frames = _sample_two_frames(H, W, allowed_sizes)
        if frames is not None:
            return frames
    frames = _sample_three_frames(H, W, allowed_sizes)
    if frames is not None:
        return frames
    return None if deep_pattern else _sample_two_frames(H, W, allowed_sizes)


def _paint_pattern(grid: np.ndarray, r: int, c: int, side: str,
                   pattern: list[tuple[int, int]], color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _to_actual(dr, dc, side)
        grid[r + rr, c + cc] = color


def _assign_side_colors(frames: list[dict], bg: int) -> None:
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        chosen = random.sample(palette, 4)
        frame["side_colors"] = dict(zip(SIDE_KINDS, chosen))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]]:
    # Keep at most one source per frame so recolored midpoint anchors never split a 0-ring.
    n = random.randint(1, min(N_SOURCES_MAX, len(frames)))
    chosen_frames = random.sample(range(len(frames)), n)
    return {(fi, random.choice(SIDE_KINDS)) for fi in chosen_frames}


def _render_output(base: np.ndarray, frames: list[dict],
                   pattern: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    for frame in frames:
        for side in SIDE_KINDS:
            r, c = _midpoint_coord(frame, side)
            _paint_pattern(out, r, c, side, pattern, frame["side_colors"][side])
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], k_max: int,
                  sources: set[tuple[int, str]]) -> np.ndarray:
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for side in SIDE_KINDS:
            if (fi, side) in sources:
                continue
            r, c = _midpoint_coord(frame, side)
            mdr, mdc = _to_actual(k_max, 0, side)
            marker = (r + mdr, c + mdc)
            for dr, dc in pattern:
                adr, adc = _to_actual(dr, dc, side)
                rr, cc = r + adr, c + adc
                if (rr, cc) != marker:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _count_corner_paints(grid: np.ndarray, frames: list[dict], bg: int) -> int:
    total = 0
    for frame in frames:
        for r, c in _frame_corners(frame):
            v = int(grid[r, c])
            if v != bg and v != 0:
                total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)

    want_deep = random.random() < DEEP_PATTERN_PROB
    frames = _sample_frames(H, W, deep_pattern=want_deep)
    if frames is None:
        return None

    max_depth = 1 if any(frame["size"] == 5 for frame in frames) else 2
    force_depth = 2 if want_deep else None
    pattern, k_max = _choose_pattern(max_depth, force_depth=force_depth)

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_side_colors(frames, bg)
    sources = _pick_sources(frames)

    output = _render_output(base, frames, pattern)
    inp = _derive_input(output, base, frames, pattern, k_max, sources)

    if _count_corner_paints(inp, frames, bg) != 0:
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except AssertionError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid midpoint-edge input after {MAX_ATTEMPTS} attempts")
