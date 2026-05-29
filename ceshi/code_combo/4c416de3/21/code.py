# Combo variant of 4c416de3 (index 21)
# combo_spec: D11/2+D13/0
# Name: 开口框架+虚拟方位标记

import random

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


CORNER_KINDS = ("TL", "TR", "BL", "BR")
_CORNER_FACTORS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}

# One colored guide pixel sits in an outside 2x2 block. Its exact slot encodes
# the corner's logical orientation, which may differ from the physical corner.
_GUIDE_SLOTS = {
    "TL": {"TL": (-3, 0), "TR": (-2, 0), "BL": (0, -3), "BR": (0, -2)},
    "TR": {"TL": (-3, 0), "TR": (-2, 0), "BL": (0, 2), "BR": (0, 3)},
    "BL": {"TL": (0, -3), "TR": (0, -2), "BL": (2, 0), "BR": (3, 0)},
    "BR": {"TL": (0, 2), "TR": (0, 3), "BL": (2, 0), "BR": (3, 0)},
}

_PATTERN_R = 2
_ALLOWED_PATTERN_OFFSETS = {
    (-1, -1),
    (-1, 0),
    (0, -1),
    (0, 0),
    (1, 1),
    (2, 2),
}


def _rot(dr, dc, kind):
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    if kind == "BR":
        return -dr, -dc
    raise ValueError(kind)


# -----------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior:
#    - D11/2: every square 0-frame has one deliberate border opening of length
#      1-2, so corner recovery cannot rely on four intact sides.
#    - D13/0: every physical corner has an outside guide pixel whose slot says
#      which logical corner orientation to use when reading or copying the glyph.
#
# b) Complexity calibration:
#    - The opening is short and never touches a corner, so a human can still
#      infer the intended square frame.
#    - The guide is only one pixel, but it lives in a fixed 2x2 codebook around
#      each corner, giving four stable and visually learnable states.
#
# c) Broken original assumptions and fixes:
#    - Local "two adjacent 0-edges" corner tests fail on open frames. The solver
#      instead scans for almost-complete square perimeters and verifies all four
#      corner guide codes.
#    - Physical corner kind no longer determines rotation. The solver decodes the
#      logical kind from the guide before extracting markers or painting targets.
#
# d/e/f/g) Placement and observability:
#    - Guide pixels are outside the frame and outside the copied glyph radius, so
#      semantic roles do not overlap.
#    - Generator layouts keep frames separated and enforce at least one virtual /
#      physical mismatch, so the fused behavior is always visible.
# -----------------------------------------------------------------------------


def _corner_coord(frame, kind):
    fr, fc = _CORNER_FACTORS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + fr * s1, frame["c"] + fc * s1


def _side_cells(frame):
    r0, c0, size = frame["r"], frame["c"], frame["size"]
    r1, c1 = r0 + size - 1, c0 + size - 1
    return {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }


def _perimeter_cells(frame):
    sides = _side_cells(frame)
    coords = []
    seen = set()
    for cells in sides.values():
        for cell in cells:
            if cell not in seen:
                seen.add(cell)
                coords.append(cell)
    return coords


def _read_guide(g, r, c, physical_kind, bg):
    seen = []
    for logical_kind, (dr, dc) in _GUIDE_SLOTS[physical_kind].items():
        rr, cc = r + dr, c + dc
        if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
            continue
        v = int(g[rr, cc])
        if v != bg and v != 0:
            seen.append((logical_kind, v, (rr, cc)))
    assert len(seen) == 1, f"bad guide at {(r, c, physical_kind)}: {seen}"
    logical_kind, color, cell = seen[0]
    return logical_kind, color, {cell}


def _colored_around(g, r, c, logical_kind, bg, banned_cells):
    items = []
    H, W = g.shape
    for dr in range(-_PATTERN_R, _PATTERN_R + 1):
        for dc in range(-_PATTERN_R, _PATTERN_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in banned_cells:
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            canon_off = _rot(dr, dc, logical_kind)
            if canon_off not in _ALLOWED_PATTERN_OFFSETS:
                continue
            items.append((canon_off, v))
    return items


def _gap_cells_for_frame(g, frame, bg):
    gaps = []
    for side_name, cells in _side_cells(frame).items():
        interior = cells[1:-1]
        side_gaps = [cell for cell in interior if int(g[cell]) == bg]
        if side_gaps:
            if len(side_gaps) > 2:
                return None
            idxs = [interior.index(cell) for cell in side_gaps]
            if idxs[-1] - idxs[0] + 1 != len(idxs):
                return None
            gaps.extend((side_name, cell) for cell in side_gaps)
    if len(gaps) not in (1, 2):
        return None
    if len({side_name for side_name, _ in gaps}) != 1:
        return None
    return [cell for _, cell in gaps]


def _candidate_frame(g, bg, r0, c0, size):
    frame = {"r": r0, "c": c0, "size": size}
    inside = g[r0 + 1:r0 + size - 1, c0 + 1:c0 + size - 1]
    if np.count_nonzero(inside == 0):
        return None

    gap_cells = _gap_cells_for_frame(g, frame, bg)
    if gap_cells is None:
        return None

    perim = _perimeter_cells(frame)
    if sum(int(g[cell]) != bg for cell in perim) != len(perim) - len(gap_cells):
        return None

    guides = {}
    for physical_kind in CORNER_KINDS:
        r, c = _corner_coord(frame, physical_kind)
        try:
            logical_kind, color, guide_cells = _read_guide(g, r, c, physical_kind, bg)
        except AssertionError:
            return None
        guides[physical_kind] = {
            "logical": logical_kind,
            "color": color,
            "guide_cells": guide_cells,
        }

    frame["guides"] = guides
    frame["gap_cells"] = gap_cells
    frame["score"] = (len(perim) - len(gap_cells), size)
    return frame


def _boxes_overlap(a, b):
    return not (
        a["r"] + a["size"] - 1 < b["r"]
        or b["r"] + b["size"] - 1 < a["r"]
        or a["c"] + a["size"] - 1 < b["c"]
        or b["c"] + b["size"] - 1 < a["c"]
    )


def _sum_score(frames):
    return sum(fr["score"][0] for fr in frames), sum(fr["score"][1] for fr in frames)


def _find_frames(g, bg):
    H, W = g.shape
    candidates = []
    for size in range(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1] + 1):
        for r0 in range(H - size + 1):
            for c0 in range(W - size + 1):
                frame = _candidate_frame(g, bg, r0, c0, size)
                if frame is not None:
                    candidates.append(frame)

    assert candidates, "no candidate frames found"
    candidates.sort(key=lambda fr: (-fr["score"][0], -fr["score"][1], fr["r"], fr["c"]))

    best = []

    def dfs(idx, chosen):
        nonlocal best
        if idx == len(candidates):
            if len(chosen) > len(best) or (
                len(chosen) == len(best) and _sum_score(chosen) > _sum_score(best)
            ):
                best = chosen[:]
            return
        if len(chosen) + (len(candidates) - idx) < len(best):
            return
        cur = candidates[idx]
        if all(not _boxes_overlap(cur, prev) for prev in chosen):
            chosen.append(cur)
            dfs(idx + 1, chosen)
            chosen.pop()
        dfs(idx + 1, chosen)

    dfs(0, [])
    assert 2 <= len(best) <= 3, f"expected 2-3 frames, got {len(best)}"
    best.sort(key=lambda fr: (fr["r"], fr["c"]))
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover open square frames and use corner guide pixels to decide how the
    shared glyph should be rotated before copying it to target corners.

    Input structure:
    - The background is a single non-zero color.
    - Each object is a hollow square 0-frame with one short missing border
      segment, so the frame is visibly open but still inferable.
    - Every physical corner has exactly one colored guide pixel in a fixed
      outside 2x2 code area. That pixel encodes the corner's logical orientation.
    - Source corners keep the full glyph in their own color. Target corners keep
      only the deepest logical-diagonal marker plus the guide pixel.

    Rule:
    - Detect each open frame from its almost-complete square perimeter.
    - At every physical corner, decode the guide slot to obtain the logical
      corner kind.
    - Extract the shared canonical glyph from the source corner(s), then repaint
      it at each target using that target corner's color and logical orientation.

    Key insight:
    The frame tells you where the corner is, but not how that corner should
    behave. The required rotation is moved into the external guide pixel, so a
    physical TL corner may need to be treated as logical BR.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _find_frames(g, bg)

    corners = []
    for frame in frames:
        for physical_kind in CORNER_KINDS:
            r, c = _corner_coord(frame, physical_kind)
            logical_kind = frame["guides"][physical_kind]["logical"]
            guide_color = frame["guides"][physical_kind]["color"]
            guide_cells = frame["guides"][physical_kind]["guide_cells"]
            items = _colored_around(g, r, c, logical_kind, bg, guide_cells)
            corners.append(
                {
                    "r": r,
                    "c": c,
                    "physical": physical_kind,
                    "logical": logical_kind,
                    "guide_color": guide_color,
                    "guide_cells": guide_cells,
                    "items": items,
                }
            )

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for info in corners:
        outward = [(off, color) for (off, color) in info["items"] if _is_outward(off)]
        if len(outward) >= 2:
            colors = {color for _, color in outward}
            assert len(colors) == 1, f"source color conflict at {(info['r'], info['c'])}"
            sources.append(info)
    assert sources, "no source corner detected"

    def _extract(info):
        pat_col = info["guide_color"]
        canon = frozenset(off for (off, color) in info["items"] if color == pat_col)
        assert canon, f"empty source at {(info['r'], info['c'], info['physical'])}"
        return canon

    canon = _extract(sources[0])
    for src in sources[1:]:
        other = _extract(src)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dc, dr) in canon, f"pattern not diagonal-symmetric at {(dr, dc)}"

    diag_ks = [dr for (dr, dc) in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no diagonal marker"
    marker_k = max(diag_ks)
    marker_canon = (marker_k, marker_k)

    source_ids = {(info["r"], info["c"], info["physical"]) for info in sources}
    targets = []
    for info in corners:
        key = (info["r"], info["c"], info["physical"])
        if key in source_ids:
            continue
        mdr, mdc = _rot(*marker_canon, info["logical"])
        mr, mc = info["r"] + mdr, info["c"] + mdc
        assert 0 <= mr < H and 0 <= mc < W, f"marker out of bounds at {key}"
        tcol = int(g[mr, mc])
        assert tcol == info["guide_color"], f"guide/marker mismatch at {key}"
        visible_offsets = {off for off, _ in info["items"]}
        assert visible_offsets.issubset(canon), f"unexpected target cells at {key}: {visible_offsets - canon}"
        for off, color in info["items"]:
            assert color == tcol, f"target color conflict at {key}"
            assert off == marker_canon, f"target shows non-marker cell at {key}: {off}"
        targets.append((info["r"], info["c"], info["logical"], tcol))

    assert targets, "no target corners detected"

    out = g.copy()
    for r, c, logical_kind, tcol in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, logical_kind)
            rr, cc = r + adr, c + adc
            assert 0 <= rr < H and 0 <= cc < W
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, f"paint conflict at {(rr, cc)}"
            out[rr, cc] = tcol
    return out


# --------------- generator (output-first fused open-frame variant) ---------------

H_RANGE = (24, 24)
W_RANGE = (24, 24)
BORDER = 3
FRAME_SIZE_RANGE = (7, 7)
N_SOURCES_MAX = 2
MAX_ATTEMPTS = 500

PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)


def _choose_pattern():
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid, r, c, size):
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _paint_pattern(grid, r, c, logical_kind, pattern, color):
    for dr, dc in pattern:
        adr, adc = _rot(dr, dc, logical_kind)
        grid[r + adr, c + adc] = color


def _paint_guide(grid, r, c, physical_kind, logical_kind, color):
    dr, dc = _GUIDE_SLOTS[physical_kind][logical_kind]
    grid[r + dr, c + dc] = color


def _assign_corner_colors(frames, bg):
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _assign_virtual_kinds(frames):
    mapping = {}
    mismatches = 0
    for fi, _frame in enumerate(frames):
        perm = list(CORNER_KINDS)
        random.shuffle(perm)
        for physical_kind, logical_kind in zip(CORNER_KINDS, perm):
            mapping[(fi, physical_kind)] = logical_kind
            mismatches += logical_kind != physical_kind
    if mismatches == 0:
        fi = random.randrange(len(frames))
        mapping[(fi, "TL")], mapping[(fi, "TR")] = mapping[(fi, "TR")], mapping[(fi, "TL")]
    return mapping


def _pick_sources(frames):
    frame_ids = list(range(len(frames)))
    n = random.randint(1, min(N_SOURCES_MAX, len(frame_ids)))
    chosen_frames = random.sample(frame_ids, n)
    return {(fi, random.choice(CORNER_KINDS)) for fi in chosen_frames}


def _frame_zone(frame, pattern):
    cells = set(_perimeter_cells(frame))
    for physical_kind in CORNER_KINDS:
        r, c = _corner_coord(frame, physical_kind)
        for logical_kind in CORNER_KINDS:
            gdr, gdc = _GUIDE_SLOTS[physical_kind][logical_kind]
            cells.add((r + gdr, c + gdc))
            for dr, dc in pattern:
                adr, adc = _rot(dr, dc, logical_kind)
                cells.add((r + adr, c + adc))
    return cells


def _frame_cells_in_bounds(frame, H, W, pattern):
    for r, c in _frame_zone(frame, pattern):
        if not (0 <= r < H and 0 <= c < W):
            return False
    return True


def _zones_are_clean(frames, pattern, H, W):
    occupied = set()
    for frame in frames:
        zone = _frame_zone(frame, pattern)
        if not _frame_cells_in_bounds(frame, H, W, pattern):
            return False
        if occupied & zone:
            return False
        occupied |= zone
    return True


def _sample_frames(H, W, pattern):
    size = FRAME_SIZE_RANGE[0]
    top_rows = [BORDER, BORDER + 1]
    left_cols = [BORDER, BORDER + 1]
    bottom_row = H - BORDER - size
    right_col = W - BORDER - size
    layouts = (
        [
            {"r": random.choice(top_rows), "c": random.choice(left_cols), "size": size},
            {"r": bottom_row, "c": right_col, "size": size},
        ],
        [
            {"r": random.choice(top_rows), "c": right_col, "size": size},
            {"r": bottom_row, "c": random.choice(left_cols), "size": size},
        ],
    )
    frames = [dict(frame) for frame in random.choice(layouts)]
    for idx, frame in enumerate(frames):
        frame["index"] = idx
    return frames


def _opening_candidates(frame):
    options = []
    for cells in _side_cells(frame).values():
        interior = cells[1:-1]
        for length in (1, 2):
            for start in range(0, len(interior) - length + 1):
                if start == 0 or start + length == len(interior):
                    continue
                options.append(interior[start:start + length])
    return options


def _apply_openings(base, frames, bg):
    openings = {}
    for frame in frames:
        options = _opening_candidates(frame)
        assert options
        seg = random.choice(options)
        for r, c in seg:
            base[r, c] = bg
        openings[frame["index"]] = seg
    return openings


def _render_guides(base, frames, virtual_kinds):
    for frame in frames:
        fi = frame["index"]
        for physical_kind in CORNER_KINDS:
            r, c = _corner_coord(frame, physical_kind)
            color = frame["corner_colors"][physical_kind]
            logical_kind = virtual_kinds[(fi, physical_kind)]
            _paint_guide(base, r, c, physical_kind, logical_kind, color)


def _build_output(base, frames, virtual_kinds, pattern):
    out = base.copy()
    occupancy = {}
    for frame in frames:
        fi = frame["index"]
        for physical_kind in CORNER_KINDS:
            r, c = _corner_coord(frame, physical_kind)
            logical_kind = virtual_kinds[(fi, physical_kind)]
            color = frame["corner_colors"][physical_kind]
            for dr, dc in pattern:
                adr, adc = _rot(dr, dc, logical_kind)
                cell = (r + adr, c + adc)
                prev = occupancy.get(cell)
                if prev is not None and prev != color:
                    return None
                occupancy[cell] = color
            _paint_pattern(out, r, c, logical_kind, pattern, color)
    return out


def _derive_input(output, base, frames, virtual_kinds, pattern, k_max, sources):
    inp = output.copy()
    for frame in frames:
        fi = frame["index"]
        for physical_kind in CORNER_KINDS:
            if (fi, physical_kind) in sources:
                continue
            r, c = _corner_coord(frame, physical_kind)
            logical_kind = virtual_kinds[(fi, physical_kind)]
            mdr, mdc = _rot(k_max, k_max, logical_kind)
            marker = (r + mdr, c + mdc)
            for dr, dc in pattern:
                adr, adc = _rot(dr, dc, logical_kind)
                cell = (r + adr, c + adc)
                if cell != marker:
                    inp[cell] = base[cell]
    return inp


def _fusion_metrics(frames, virtual_kinds):
    mismatches = sum(
        virtual_kinds[(frame["index"], physical_kind)] != physical_kind
        for frame in frames
        for physical_kind in CORNER_KINDS
    )
    return {
        "n_frames": len(frames),
        "mismatches": mismatches,
        "distinct_logical": len(set(virtual_kinds.values())),
    }


def _gen_attempt(return_meta=False):
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frames = _sample_frames(H, W, pattern)
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frames, bg)
    virtual_kinds = _assign_virtual_kinds(frames)
    openings = _apply_openings(base, frames, bg)
    _render_guides(base, frames, virtual_kinds)

    sources = _pick_sources(frames)
    output = _build_output(base, frames, virtual_kinds, pattern)
    if output is None:
        return None

    inp = _derive_input(output, base, frames, virtual_kinds, pattern, k_max, sources)
    solved = main(inp)
    if not np.array_equal(solved, output):
        return None

    if return_meta:
        return inp, {
            "frames": frames,
            "virtual_kinds": virtual_kinds,
            "openings": openings,
            "sources": sources,
            "metrics": _fusion_metrics(frames, virtual_kinds),
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
