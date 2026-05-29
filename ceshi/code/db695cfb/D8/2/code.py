# Variant of db695cfb
# Dimension: D8 — 触发像素数量与排列 (obstacle)
# Change: 对角线上可能有多个品红像素，每个独立触发延伸... → 品红像素成对触发

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D8 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     - Blue endpoints still define the main diagonal segment exactly as in the seed.
#     - Magenta pixels that lie on that diagonal no longer fire independently.
#     - Instead, magenta pixels are paired by local proximity along the diagonal:
#       repeatedly take the closest adjacent magenta pair on that segment, remove it,
#       and continue. Any leftover single magenta does nothing.
#     - For each paired magenta pixel, emit perpendicular diagonal rays in both
#       directions, but only for a finite length equal to the pair's diagonal gap.
#   concrete_behavior:
#     - Input still contains only endpoint blue pixels plus some magenta pixels on
#       the segment interiors and some distractors elsewhere.
#     - Output still fills the blue diagonal segment, but magenta growth becomes
#       finite and depends on pair spacing instead of "all the way to the border".
#   behavior_trace:
#     - Example segment interior trigger order: indices [2, 5, 11].
#     - Original seed: all three magenta pixels fire infinite perpendicular rays.
#     - New rule: the closest pair is (2, 5), so those two each fire rays of length 3.
#       The magenta at index 11 is left unmatched and stays unchanged.
#     - This changes both which magenta pixels are active and how long their output is.
#   complexity_calibration:
#     - Pair gaps are limited to a small visible range (2-4 cells), so examples are
#       readable and repeated enough for humans to infer the rule.
#     - Multiple trigger patterns are allowed, but dense ambiguous runs are avoided:
#       generator layouts separate intended pairs by larger gaps than within-pair gaps.
#     - Some segments carry a lone magenta alongside one or two pairs so the
#       "unpaired does nothing" branch is observable.
#   implicit_assumptions_checked_against_seed_main:
#     - The seed treated each non-blue pixel on a segment independently; that fails
#       here because activation depends on neighboring magenta positions.
#     - The seed scanned `out` while drawing. That is harmless when every trigger is
#       independent, but pairing must depend on the original input arrangement, so
#       the new solver reads trigger locations from `grid` and writes to `out`.
#     - The seed emitted to the boundary. Here the extension length is variable, so
#       emission must count a bounded number of diagonal steps instead.
#     - The seed generator could tolerate a few accidental magenta alignments. This
#       variant introduces more magenta points, so the generator now avoids accidental
#       cross-segment magenta diagonals that could confuse line-color detection.
#   placement_legality:
#     - Blue segments are placed on distinct diagonals and their endpoints are kept
#       away from diagonal alignment with endpoints from other segments.
#     - Rays are computed output-first; any candidate segment is rejected if its
#       finite magenta rays would overlap existing placed pixels or previously drawn
#       output cells.
#     - Off-diagonal magenta distractors are also kept away from diagonal alignment
#       with true on-segment magentas, so the pairing evidence remains local.
#   visual_observability:
#     - The visible cue is strong: two close magenta pixels yield two short matching
#       perpendicular ray bundles, while a lone distant magenta yields none.
#     - The rule can be inferred from a few samples because pair distance is directly
#       reflected in ray length and unpaired magenta pixels remain inert.
#   cleanliness:
#     - Per-segment trigger counts stay modest and segments do not overlap, so the
#       output keeps the original sparse-diagonal look instead of becoming tangled.
#   color_inference:
#     - No hidden use of color ids is introduced. Blue still marks the carrier line
#       and magenta still marks the reactive pixels and their extensions.
#   solver_assumptions_after_change:
#     - On each usable segment, intended pairs must be the unique closest adjacent
#       magenta gaps. The generator enforces this with wider separations between
#       groups than within pairs.
#     - At least one actual pair must exist in generated tasks; otherwise the sample
#       would collapse to the seed's "just fill the diagonal" behavior.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal endpoint pairs, then let paired trigger pixels emit bounded perpendicular rays.

    Input structure:
    - bg is the most frequent color. One color forms diagonal endpoint pairs, while magenta-style trigger pixels may sit on those diagonals in singles or clustered groups.

    Rule:
    - Draw each detected endpoint diagonal from one extreme endpoint to the other.
    - On that completed diagonal, repeatedly pair the closest adjacent non-line trigger pixels of the same color. Each matched trigger emits perpendicular diagonal rays in both directions, with length equal to the matched pair's spacing. Unmatched trigger pixels do not emit anything.

    Key insight:
    - The trigger effect is no longer per-pixel. The relevant unit is a local pair on the carrier diagonal, and the gap inside that pair controls the ray length.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        dr = 1 if r1 > r0 else -1
        dc = 1 if c1 > c0 else -1
        steps = abs(r1 - r0)
        return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]

    def greedy_adjacent_pairs(
        entries: list[tuple[int, int, int]]
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        remaining = list(entries)
        pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        while len(remaining) >= 2:
            best_i = min(range(len(remaining) - 1), key=lambda i: remaining[i + 1][0] - remaining[i][0])
            pairs.append((remaining[best_i], remaining[best_i + 1]))
            del remaining[best_i : best_i + 2]
        return pairs

    # Choose the "line color": the non-background color with the longest diagonal-aligned pair.
    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    emissions: list[tuple[int, int, int, tuple[tuple[int, int], tuple[int, int]], int]] = []

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant (slope +1)
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant (slope -1)
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = ((dr, -dc), (-dr, dc))

        triggers_by_color: dict[int, list[tuple[int, int, int]]] = {}
        for idx, (r, c) in enumerate(path):
            if int(out[r, c]) == bg:
                out[r, c] = line_color
            original = int(grid[r, c])
            if original != bg and original != line_color:
                triggers_by_color.setdefault(original, []).append((idx, r, c))

        for tcol, entries in triggers_by_color.items():
            for left, right in greedy_adjacent_pairs(entries):
                idx0, r0, c0 = left
                idx1, r1, c1 = right
                span = idx1 - idx0
                if span <= 0:
                    continue
                emissions.append((r0, c0, tcol, perp_dirs, span))
                emissions.append((r1, c1, tcol, perp_dirs, span))

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for r0, c0, tcol, dirs, ray_len in emissions:
        for ddr, ddc in dirs:
            for step in range(1, ray_len + 1):
                r = r0 + ddr * step
                c = c0 + ddc * step
                if not (0 <= r < h and 0 <= c < w):
                    break
                if int(out[r, c]) == bg:
                    out[r, c] = tcol

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)              # grid height range
_W_RANGE = (9, 21)              # grid width range
_LINE_COLOR = 1                 # color for diagonal segment endpoints/fills
_TRIGGER_COLOR = 6              # color for trigger pixels and perpendicular rays
_SEGMENT_COUNT = (1, 4)         # number of diagonal segments
_SEG_LEN_RANGE = (7, 16)        # longer minimum leaves room for pair/lone layouts
_PAIR_GAP_RANGE = (2, 4)        # diagonal gap inside one trigger pair
_GROUP_SEP_MIN = 5              # keep intended pairs visually closer than group gaps

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)  # unpaired line_color distractor pixels
_OFF_DIAG_TRIGGERS = (0, 3)     # off-diagonal trigger_color distractor pixels

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100    # attempts to place a single segment
_TRIGGER_LAYOUT_RETRIES = 60    # attempts to place trigger groups on one segment
_DISTRACTOR_RETRIES = 60        # attempts to place a single distractor pixel
_GEN_RETRIES = 400              # outer generate_input retry limit


def _diag_aligned(p: tuple[int, int], q: tuple[int, int]) -> bool:
    return p[0] != q[0] and abs(p[0] - q[0]) == abs(p[1] - q[1])


def _compute_pair_ray_cells(pair_descriptors, perp_dirs, H, W):
    """Compute all cells touched by bounded perpendicular rays from paired triggers."""
    cells = set()
    for (r0, c0), (r1, c1), span in pair_descriptors:
        for tr, tc in ((r0, c0), (r1, c1)):
            for ddr, ddc in perp_dirs:
                for step in range(1, span + 1):
                    r = tr + ddr * step
                    c = tc + ddc * step
                    if not (0 <= r < H and 0 <= c < W):
                        break
                    cells.add((r, c))
    return cells


def _build_expected(inp, bg, segments):
    """Build the expected output grid by filling diagonals and emitting bounded rays."""
    expected = inp.copy()
    H, W = inp.shape
    for _, _, _, pair_descriptors, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for (r0, c0), (r1, c1), span in pair_descriptors:
            for tr, tc in ((r0, c0), (r1, c1)):
                for ddr, ddc in perp_dirs:
                    for step in range(1, span + 1):
                        r = tr + ddr * step
                        c = tc + ddc * step
                        if not (0 <= r < H and 0 <= c < W):
                            break
                        if expected[r, c] == bg:
                            expected[r, c] = _TRIGGER_COLOR
    return expected


def _try_place_segment(slope_type, seg_len, H, W, occupied, drawn, used_a, used_b, existing_line_pixels):
    """Try to place one diagonal segment, returning (path, dc, diag_id) or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
            diag_id = r0 - c0
            if diag_id in used_a:
                continue
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1
            diag_id = r0 + c0
            if diag_id in used_b:
                continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue

        ep0, ep1 = path[0], path[-1]
        if any(_diag_aligned(ep0, p) or _diag_aligned(ep1, p) for p in existing_line_pixels):
            continue

        return path, dc, diag_id
    return None


def _pattern_min_cells(pattern):
    total = 0
    for kind in pattern:
        total += 1 if kind == "lone" else (_PAIR_GAP_RANGE[0] + 1)
    if len(pattern) >= 2:
        total += _GROUP_SEP_MIN * (len(pattern) - 1)
    return total


def _plan_trigger_layout(interior_len, require_pair):
    patterns = [()]
    for pattern in [
        ("pair",),
        ("pair", "lone"),
        ("lone", "pair"),
        ("pair", "pair"),
    ]:
        if _pattern_min_cells(pattern) <= interior_len:
            patterns.append(pattern)

    if require_pair:
        patterns = [p for p in patterns if "pair" in p]
    if not patterns:
        return None

    for _ in range(_TRIGGER_LAYOUT_RETRIES):
        pattern = random.choice(patterns)
        pair_gaps = []
        widths = []
        for kind in pattern:
            if kind == "lone":
                widths.append(1)
            else:
                gap = random.randint(_PAIR_GAP_RANGE[0], _PAIR_GAP_RANGE[1])
                pair_gaps.append(gap)
                widths.append(gap + 1)

        base_total = sum(widths) + _GROUP_SEP_MIN * max(0, len(pattern) - 1)
        if base_total > interior_len:
            continue

        slack = interior_len - base_total
        buckets = [0] * (len(pattern) + 1)
        for _ in range(slack):
            buckets[random.randint(0, len(buckets) - 1)] += 1

        cursor = buckets[0]
        pair_idx = 0
        trigger_indices = []
        pair_index_spans = []
        for i, kind in enumerate(pattern):
            start = cursor
            if kind == "lone":
                trigger_indices.append(start)
                cursor += 1
            else:
                gap = pair_gaps[pair_idx]
                pair_idx += 1
                trigger_indices.extend([start, start + gap])
                pair_index_spans.append((start, start + gap, gap))
                cursor += gap + 1
            if i < len(pattern) - 1:
                cursor += _GROUP_SEP_MIN + buckets[i + 1]

        if not trigger_indices:
            return [], []

        if len(set(trigger_indices)) != len(trigger_indices):
            continue
        if max(trigger_indices) >= interior_len:
            continue

        return sorted(trigger_indices), pair_index_spans
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()   # input pixel positions
    drawn = set()      # output-only cells (diagonal fills + bounded ray fills)
    segments = []

    used_a = set()
    used_b = set()
    all_line_pixels = []
    all_trigger_positions = []
    need_pair = True

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(
            slope_type,
            seg_len,
            H,
            W,
            occupied,
            drawn,
            used_a,
            used_b,
            all_line_pixels,
        )
        if result is None:
            continue
        path, dc, diag_id = result

        path_interior = path[1:-1]
        layout = _plan_trigger_layout(len(path_interior), require_pair=need_pair)
        if layout is None:
            continue
        trigger_indices, pair_index_spans = layout
        trigger_positions = [path_interior[i] for i in trigger_indices]
        pair_descriptors = [
            (path_interior[i0], path_interior[i1], span) for i0, i1, span in pair_index_spans
        ]

        if any(_diag_aligned(tp, old) for tp in trigger_positions for old in all_trigger_positions):
            continue

        perp_dirs = ((1, -dc), (-1, dc))
        ray_cells = _compute_pair_ray_cells(pair_descriptors, perp_dirs, H, W)

        if pair_descriptors and not ray_cells:
            continue
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, pair_descriptors, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)
        all_line_pixels.extend([ep0, ep1])
        all_trigger_positions.extend(trigger_positions)
        if slope_type == "backslash":
            used_a.add(diag_id)
        else:
            used_b.add(diag_id)
        if pair_descriptors:
            need_pair = False

    if not segments or need_pair:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            cand = (r, c)
            if cand in occupied or cand in drawn:
                continue
            if any(_diag_aligned(cand, lp) for lp in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add(cand)
            all_line_positions.append(cand)
            break

    trigger_positions_now = list(zip(*np.where(inp == _TRIGGER_COLOR)))
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            cand = (r, c)
            if cand in occupied or cand in drawn:
                continue
            if any(_diag_aligned(cand, tp) for tp in trigger_positions_now):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add(cand)
            trigger_positions_now.append(cand)
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
