# Variant of 16de56c4
# Dimension: D14 — 标记点数量约束 (meta)
# Change: 每行每种重复色恰好2个标记点... → 多标记色共存

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D14 variant
# ===================================
# a) Concrete behavior:
#    The original task allowed only one repeated color on each shooting line, so
#    every line had a single arithmetic progression.  This variant keeps the same
#    global shooting edge, but a single active line may now contain 2-3 repeated
#    colors.  Each repeated color still appears exactly twice and defines its own
#    step, yet only the earliest repeated color on that line is forced to touch
#    the shooting edge.  Later repeated colors start deeper inside the same line.
#    Example in left-edge orientation on a row of length 11:
#    - old rule: color 2 at distances 0 and 3 fills 0,3,6,9
#    - new rule: color 2 at 0 and 3 fills 0,3,6,9, while color 4 at 2 and 5
#      independently fills 2,5,8.  The line output is produced by painting the
#      first ray, then the second ray; if both rays hit the same cell, the later
#      start overwrites the earlier paint.
#    This changes the solver at the "per line" stage: it can no longer stop after
#    finding one repeated color, and it can no longer treat every aligned foreign
#    dot as a companion because some aligned foreign dots are now other markers.
#
# b) Complexity calibration:
#    The generator uses 1-3 marker colors per active line, with at least one
#    multi-marker line per grid and usually several.  This is enough to make the
#    new behavior visible without turning each row/column into an unreadable pile
#    of dots.  Companion dots and bystanders are still present, but kept sparse.
#
# c) Old assumptions that break:
#    - "one repeated color per line" is no longer true, so `repeating` must be
#      processed as an ordered list instead of a singleton.
#    - "a companion is any aligned non-marker color" fails, because another
#      repeated color may align to the same modular class.  The new solver only
#      allows single-occurrence colors to serve as companions.
#    - "no-companion fill spans the whole line" was safe only because the first
#      marker always sat on the edge.  Interior-start markers would otherwise
#      paint backwards toward the edge.  The variant therefore fills from the
#      first marker onward in the shooting direction.
#    - "write only on bg or marker cells" breaks overlay order.  With several
#      rays on one line, later rays must be allowed to overwrite earlier paint.
#
# d) Placement legality:
#    The generator works in normalized shooting distances d=0..line_len-1.  On
#    each active line it places marker pairs at (lead, lead+step), optionally one
#    aligned companion farther away, and optionally one bystander that is not on
#    any marker beat.  Input dots are unique per cell, stay in-bounds, and are
#    validated so that each marker sees at most one aligned singleton companion.
#
# e) Visual observability:
#    Every grid contains at least one line with 2+ marker colors, and later
#    markers are chosen so that their future painted beats intersect earlier rays.
#    The new rule is therefore visible in the output as layered progressions on a
#    single row/column rather than as separate one-ray lines that mimic the seed.
#
# f) Cleanliness:
#    The grid still has separated active lines, only a few repeated colors per
#    line, and at most one bystander per line.  Overlap happens along the same
#    shooting line, which is legible; the generator avoids random singleton dots
#    that would create ambiguous or noisy interpretations.
#
# g) Color inference:
#    No color-id ordering is used.  Colors are only roles: repeated color = ray
#    marker, aligned singleton = companion color, unaligned singleton = bystander.
#    Overlay order depends only on visible position from the shooting edge.
#
# h) New solver assumptions and generator guarantees:
#    - Each repeated color occurs exactly twice on a line.  Guaranteed by the
#      generator by using distinct line-local colors for markers.
#    - Only singleton colors can be companions.  Guaranteed because companions
#      and bystanders are line-local singletons.
#    - For a given marker there is at most one aligned singleton beyond its
#      second marker.  The line validator enforces this before accepting a line.
#    - Marker starts on the same line are distinct and ordered from the shooting
#      edge outward.  The generator constructs them that way so overlay order is
#      well-defined and visually recoverable.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several periodic rays that all point away from one dominant grid edge.

    Input structure:
    - bg is the most common color.  Non-bg cells are sparse dots on a subset of
      rows or columns determined by one unique shooting edge.
    - A shooting-direction line may contain multiple repeated colors.  Each such
      color appears exactly twice and defines one ray by its first dot and gap.
    - Only the earliest repeated color on a line is forced to touch the shooting
      edge; later repeated colors may begin farther inside the same line.
    - Single-occurrence colors are either companions (aligned to one ray beyond
      its second marker) or bystanders (not aligned to any ray).

    Rule:
    - Detect the unique shooting edge; left/right means rows are processed,
      top/bottom means columns are processed.
    - On each active line, treat every repeated color independently.  Order the
      repeated colors by where their first marker appears when scanning away from
      the shooting edge.
    - For one repeated color, the distance between its two dots is the step.
      The ray paints that beat starting at its first marker and moving away from
      the shooting edge.  An aligned singleton farther away is a companion: it
      limits the painted span and provides the paint color.  Without a companion,
      the repeated color paints to the far end of the line.
    - Apply all rays on the line in that positional order; later-starting rays
      overwrite earlier paint where their beat positions overlap.  Bystanders are
      untouched because they are not on any ray beat.

    Key insight:
    The seed task's arithmetic progression still exists, but now it must be
    recovered multiple times on the same row/column.  The important separation is
    "repeated colors define rays, singleton colors may stop rays", and overlap is
    resolved by the visible launch order from the common shooting edge.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge_counts = {
        "top": sum(1 for c in range(W) if g[0, c] != bg),
        "bottom": sum(1 for c in range(W) if g[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if g[r, 0] != bg),
        "right": sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    forward = edge in ("left", "top")

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def dist_from_edge(pos: int) -> int:
        return pos if forward else line_len - 1 - pos

    def pos_from_dist(dist: int) -> int:
        return dist if forward else line_len - 1 - dist

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            v = int(g[cell(li, p)])
            if v != bg:
                colors[v].append(p)

        if not colors:
            continue

        repeating = {
            color: sorted(positions, key=dist_from_edge)
            for color, positions in colors.items()
            if len(positions) == 2
        }
        singletons = {
            color: positions[0]
            for color, positions in colors.items()
            if len(positions) == 1
        }
        assert repeating, (
            f"Line {li} ({'row' if by_row else 'col'}) has non-bg dots "
            f"{dict(colors)} but no repeating color"
        )

        rays = []
        for marker, positions in repeating.items():
            lead_pos, trail_pos = positions
            lead_d = dist_from_edge(lead_pos)
            trail_d = dist_from_edge(trail_pos)
            step = trail_d - lead_d
            assert step > 0, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"has invalid positions {positions}"
            )

            candidates = []
            for oc, pos in singletons.items():
                d = dist_from_edge(pos)
                if d > trail_d and (d - lead_d) % step == 0:
                    candidates.append((d, oc))
            assert len(candidates) <= 1, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"has ambiguous companions: {candidates}"
            )

            if candidates:
                stop_d, paint = candidates[0]
            else:
                stop_d, paint = line_len - 1, marker

            rays.append((lead_d, step, stop_d, paint))

        rays.sort(key=lambda ray: ray[0])
        for lead_d, step, stop_d, paint in rays:
            for d in range(lead_d, stop_d + 1, step):
                out[cell(li, pos_from_dist(d))] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_COMPANION = 0.45
_P_BYSTANDER = 0.25
_MAX_RETRIES = 400
_LINE_ATTEMPTS = 300


def _choose_dims(edge: str, n_active: int):
    min_num_lines = 2 * n_active - 1
    by_row = edge in ("left", "right")
    if by_row:
        lo = max(_ROWS_RANGE[0], min_num_lines)
        if lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(lo, _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
        return H, W, H, W
    lo = max(_COLS_RANGE[0], min_num_lines)
    if lo > _COLS_RANGE[1]:
        return None
    W = random.randint(lo, _COLS_RANGE[1])
    H = random.randint(*_ROWS_RANGE)
    return H, W, W, H


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _normalized_to_pos(edge: str, line_len: int, dist: int) -> int:
    return dist if edge in ("left", "top") else line_len - 1 - dist


def _fill_dists(spec: dict[str, int], line_len: int) -> list[int]:
    return list(range(spec["lead"], spec["stop"] + 1, spec["step"]))


def _line_is_valid(marker_specs: list[dict[str, int]], singletons: list[dict[str, int]]) -> bool:
    used_positions = set()
    for spec in marker_specs:
        pair = (spec["lead"], spec["trail"])
        if pair[0] >= pair[1]:
            return False
        if spec["lead"] in used_positions or spec["trail"] in used_positions:
            return False
        used_positions.update(pair)
    for item in singletons:
        if item["dist"] in used_positions:
            return False
        used_positions.add(item["dist"])

    singleton_by_dist = {item["dist"]: item for item in singletons}
    for spec in marker_specs:
        aligned = []
        for dist in singleton_by_dist:
            if dist > spec["trail"] and (dist - spec["lead"]) % spec["step"] == 0:
                aligned.append(dist)
        if spec["companion"] is None:
            if aligned:
                return False
        else:
            if aligned != [spec["companion"]]:
                return False

    return True


def _build_line_spec(
    line_len: int,
    palette: list[int],
    *,
    multi_required: bool,
) -> dict | None:
    max_markers = 3 if line_len >= 12 else 2
    if multi_required:
        choices = [k for k in (2, 3) if k <= max_markers]
        if not choices:
            return None
        n_markers = random.choice(choices)
    else:
        pool = [1]
        if max_markers >= 2:
            pool += [2, 2]
        if max_markers >= 3:
            pool += [3]
        n_markers = random.choice(pool)

    max_colors_needed = 2 * n_markers + 1
    if len(palette) < max_colors_needed - 1:
        return None

    for _ in range(_LINE_ATTEMPTS):
        marker_colors = random.sample(palette, n_markers)
        marker_specs = []
        used_positions = set()
        fill_union = set()
        prev_lead = -1
        ok = True

        for idx in range(n_markers):
            color = marker_colors[idx]
            candidate_pairs = []
            lead_lo = 0 if idx == 0 else prev_lead + 1
            for lead in range(lead_lo, line_len - 1):
                if idx == 0 and lead != 0:
                    continue
                if lead in used_positions:
                    continue
                max_step = min(_STEP_RANGE[1], line_len - 1 - lead)
                for step in range(_STEP_RANGE[0], max_step + 1):
                    trail = lead + step
                    if trail in used_positions:
                        continue
                    fills = set(range(lead, line_len, step))
                    if idx > 0 and not (fills & fill_union):
                        continue
                    candidate_pairs.append((lead, step, trail, fills))
            random.shuffle(candidate_pairs)
            if not candidate_pairs:
                ok = False
                break

            lead, step, trail, fills = candidate_pairs[0]
            marker_specs.append(
                {
                    "marker_color": color,
                    "lead": lead,
                    "step": step,
                    "trail": trail,
                    "companion": None,
                    "paint_color": color,
                    "stop": line_len - 1,
                }
            )
            used_positions.update((lead, trail))
            fill_union |= fills
            prev_lead = lead

        if not ok:
            continue

        line_colors_used = set(marker_colors)
        singletons = []
        shuffled_specs = marker_specs[:]
        random.shuffle(shuffled_specs)
        for spec in shuffled_specs:
            if random.random() >= _P_COMPANION:
                continue
            beats = list(range(spec["trail"] + spec["step"], line_len, spec["step"]))
            random.shuffle(beats)
            chosen_dist = None
            for dist in beats:
                if dist in used_positions:
                    continue
                if any(
                    dist > other["trail"] and (dist - other["lead"]) % other["step"] == 0
                    for other in marker_specs
                    if other is not spec
                ):
                    continue
                chosen_dist = dist
                break
            if chosen_dist is None:
                continue

            available_colors = [c for c in palette if c not in line_colors_used]
            if not available_colors:
                continue
            color = random.choice(available_colors)
            line_colors_used.add(color)
            used_positions.add(chosen_dist)
            spec["companion"] = chosen_dist
            spec["paint_color"] = color
            spec["stop"] = chosen_dist
            singletons.append({"dist": chosen_dist, "color": color, "kind": "companion"})

        if random.random() < _P_BYSTANDER:
            candidates = []
            for dist in range(1, line_len):
                if dist in used_positions:
                    continue
                accidental = False
                for spec in marker_specs:
                    if dist > spec["trail"] and (dist - spec["lead"]) % spec["step"] == 0:
                        accidental = True
                        break
                    if spec["lead"] <= dist <= spec["stop"] and (dist - spec["lead"]) % spec["step"] == 0:
                        accidental = True
                        break
                if not accidental:
                    candidates.append(dist)
            available_colors = [c for c in palette if c not in line_colors_used]
            if candidates and available_colors:
                dist = random.choice(candidates)
                color = random.choice(available_colors)
                line_colors_used.add(color)
                used_positions.add(dist)
                singletons.append({"dist": dist, "color": color, "kind": "bystander"})

        if not _line_is_valid(marker_specs, singletons):
            continue

        dots = []
        for spec in marker_specs:
            dots.append((spec["lead"], spec["marker_color"]))
            dots.append((spec["trail"], spec["marker_color"]))
        for item in singletons:
            dots.append((item["dist"], item["color"]))

        overlaps = 0
        painted_sets = [set(_fill_dists(spec, line_len)) for spec in marker_specs]
        for idx in range(1, len(painted_sets)):
            overlaps += len(painted_sets[idx] & set().union(*painted_sets[:idx]))

        return {
            "dots": dots,
            "markers": marker_specs,
            "singletons": singletons,
            "n_markers": n_markers,
            "overlaps": overlaps,
        }

    return None


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]],
    edge: str,
    H: int,
    W: int,
) -> dict[str, int]:
    boundary = {
        "top": lambda r, c: r == 0,
        "bottom": lambda r, c: r == H - 1,
        "left": lambda r, c: c == 0,
        "right": lambda r, c: c == W - 1,
    }
    counts = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    palette = random.sample(_COLORS, random.randint(6, min(8, len(_COLORS))))
    required_multi = random.randint(1, min(max(1, n_active // 2), n_active))
    multi_lines = set(random.sample(active_lines, required_multi))

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    multi_count = 0
    overlap_total = 0

    for li in active_lines:
        spec = _build_line_spec(line_len, palette, multi_required=li in multi_lines)
        if spec is None:
            return None

        if spec["n_markers"] > 1:
            multi_count += 1
            overlap_total += spec["overlaps"]

        for dist, color in spec["dots"]:
            pos = _normalized_to_pos(edge, line_len, dist)
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

    if multi_count == 0 or overlap_total == 0:
        return None

    shooting_edge_dots = len(active_lines)
    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
