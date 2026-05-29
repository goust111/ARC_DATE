# Combo variant of 16de56c4 (index 28)
# combo_spec: D11/1+D14/0
# Name: 密集排列+多标记色共存

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior:
#    The seed task had one repeated color per active shooting line and those
#    lines were separated by blank rows/cols.  This combo changes both levers:
#    active lines form one consecutive band, and each active line contains
#    several repeated colors that launch independent rays.  Example in a left-
#    edge orientation with line length 11:
#    - seed line: color 2 at distances 0 and 3 fills 0,3,6,9
#    - combo line: color 2 at 0 and 2 fills 0,2,4,6,8,10 while color 4 at
#      3 and 6 fills 3,6,9, and maybe a companion at 8 turns the second ray
#      into color 7 on 3,6.  On a neighboring row the same first-ray geometry
#      may be painted in another color, so stacked rows look like rectangular
#      fill bands even though the rule is still "solve each line independently".
#
# b) Complexity calibration:
#    To make both dimensions visible, the generator uses 4-9 consecutive active
#    lines and gives every active line 2-3 marker colors.  The earliest marker
#    on each line still touches the shooting edge, which keeps the global edge
#    inference stable.  Later markers start inside the line so overlay order and
#    "forward only" painting are genuinely required.
#
# c) Old assumptions that break:
#    - Active-line gaps are gone, so visual grouping can no longer rely on blank
#      separators; the dense band must still be interpretable from local line
#      structure alone.
#    - "Any aligned foreign dot is a companion" is false once repeated colors can
#      coexist on one line.  Only singleton colors may serve as companions.
#    - "A ray without a companion fills the whole line" is unsafe for interior
#      markers, because it would backfill toward the shooting edge.  The combo
#      must paint only from the first marker onward in shooting direction.
#    - "Only bg or marker cells can be overwritten" fails with multiple rays on
#      one line; later rays must overwrite earlier paint at their own beats.
#
# d) Placement legality:
#    Generation is done in normalized distances from the shooting edge.  Every
#    line reserves unique cells for marker pairs and singleton dots, keeps them
#    in bounds, and validates that each repeated color sees at most one aligned
#    singleton beyond its second marker.  Consecutive active lines are chosen as
#    one dense block, usually away from the orthogonal borders to keep the
#    shooting edge unambiguous.
#
# e) Visual observability:
#    Each sample contains a full dense band of adjacent active rows/cols, and
#    every such line has multiple rays.  A shared primary ray geometry across the
#    band creates visible stacked strips/blocks, while secondary interior rays
#    make it clear that several repeated colors on the same line are being solved
#    independently rather than via a simple row fill.
#
# f) Cleanliness:
#    The output can become dense, so the generator limits each line to 2-3 marker
#    colors and at most one bystander.  Companions are chosen to stop only one
#    ray, and later markers are required to intersect earlier paint so overlap is
#    legible rather than random noise.
#
# g) Change inferability:
#    No hidden ordering by color id is used.  Colors only encode roles visible in
#    the input: repeated pairs define rays, aligned singletons stop rays and
#    donate paint colors, unaligned singletons stay unchanged.  Overwrite order
#    depends only on visible distance from the common shooting edge.
#
# h) New solver assumptions and generator guarantees:
#    - Repeated colors appear exactly twice per line; singletons appear once.
#      The generator enforces this with line-local unique positions.
#    - Only the first repeated color on a line touches the shooting edge; later
#      repeated colors begin strictly farther inside.  This gives a recoverable
#      launch order for overwrite handling.
#    - Any singleton aligned past a marker's second dot belongs to at most one
#      marker.  The generator rejects lines that would make a singleton look like
#      two companions at once.
#    - Dense adjacency itself is not a new 2-D rule.  The solver still works
#      line-by-line; the generator guarantees consecutive active lines so the
#      dense rectangular visual effect appears in the examples.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a dense band of periodic rays launched from one dominant boundary.

    Input structure:
    - The background is the most common color. Non-background cells lie on one
      set of shooting-direction lines chosen by a unique edge.
    - The active rows/columns form a consecutive band rather than being spaced
      apart, so many adjacent lines contain ray markers.
    - Each active line contains multiple repeated colors. Every repeated color
      appears exactly twice and defines one ray; only the earliest such pair is
      forced to touch the shooting edge, later pairs may start deeper inside.
    - Single-occurrence colors are either companions aligned to one ray beyond
      its second marker or bystanders that do not align to any ray beat.

    Rule:
    - Detect the unique shooting edge from boundary dot counts; left/right means
      rows are processed, top/bottom means columns are processed.
    - On each non-empty line, separate repeated colors from singleton colors and
      order the repeated colors by their first marker distance from the edge.
    - For one repeated color, the gap between its two dots is the step. Paint
      that arithmetic progression starting at the first marker and moving away
      from the shooting edge. If one aligned singleton lies beyond the second
      marker, it stops the ray and supplies the paint color; otherwise the ray
      continues to the far end.
    - Apply the rays in launch order on that line. Later-starting rays may
      overwrite earlier paint on overlapping beat cells. Bystanders remain as-is.

    Key insight:
    The new 2-D "filled rectangles" are only the visible consequence of solving
    many 1-D progressions on consecutive lines. The real reasoning task is to
    recover several repeated-color marker pairs per line, ignore singleton
    bystanders, and respect forward launch order from the common edge.
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
            value = int(g[cell(li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        repeating: dict[int, list[int]] = {}
        singletons: dict[int, int] = {}
        for color, positions in colors.items():
            ordered = sorted(positions, key=dist_from_edge)
            assert len(ordered) in (1, 2), (
                f"Line {li} ({'row' if by_row else 'col'}) color {color} "
                f"appears {len(ordered)} times: {ordered}"
            )
            if len(ordered) == 2:
                repeating[color] = ordered
            else:
                singletons[color] = ordered[0]

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
            for other_color, pos in singletons.items():
                dist = dist_from_edge(pos)
                if dist > trail_d and (dist - lead_d) % step == 0:
                    candidates.append((dist, other_color))
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
            for dist in range(lead_d, stop_d + 1, step):
                out[cell(li, pos_from_dist(dist))] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_PRIMARY_STEP_POOL = [1, 2, 2, 3]
_P_PRIMARY_COMPANION = 0.8
_P_EXTRA_COMPANION = 0.45
_P_BYSTANDER = 0.20

_MAX_RETRIES = 600
_LINE_ATTEMPTS = 500


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    by_row = edge in ("left", "right")
    if by_row:
        lo = max(_ROWS_RANGE[0], n_active)
        if n_active < _ROWS_RANGE[1]:
            lo = max(lo, n_active + 1)
        if lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(lo, _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
        return H, W, H, W

    lo = max(_COLS_RANGE[0], n_active)
    if n_active < _COLS_RANGE[1]:
        lo = max(lo, n_active + 1)
    if lo > _COLS_RANGE[1]:
        return None
    W = random.randint(lo, _COLS_RANGE[1])
    H = random.randint(*_ROWS_RANGE)
    return H, W, W, H


def _dense_indices(n: int, total: int) -> list[int]:
    starts = list(range(total - n + 1))
    inner = [start for start in starts if start > 0 and start + n < total]
    pool = inner if inner else starts
    start = random.choice(pool)
    return list(range(start, start + n))


def _normalized_to_pos(edge: str, line_len: int, dist: int) -> int:
    return dist if edge in ("left", "top") else line_len - 1 - dist


def _fill_dists(spec: dict[str, int]) -> list[int]:
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

    singleton_dists = sorted(item["dist"] for item in singletons)
    for spec in marker_specs:
        aligned = [
            dist
            for dist in singleton_dists
            if dist > spec["trail"] and (dist - spec["lead"]) % spec["step"] == 0
        ]
        if spec["companion"] is None:
            if aligned:
                return False
        else:
            if aligned != [spec["companion"]]:
                return False

    return True


def _candidate_pairs(
    line_len: int,
    used_positions: set[int],
    reserved_positions: set[int],
    fill_union: set[int],
    prev_lead: int,
) -> list[tuple[int, int, int, set[int]]]:
    candidates = []
    for lead in range(max(prev_lead + 1, 1), line_len - 1):
        if lead in used_positions or lead in reserved_positions:
            continue
        max_step = min(_STEP_RANGE[1], line_len - 1 - lead)
        for step in range(_STEP_RANGE[0], max_step + 1):
            trail = lead + step
            if trail in used_positions or trail in reserved_positions:
                continue
            fills = set(range(lead, line_len, step))
            if not (fills & fill_union):
                continue
            if lead < step:
                continue
            candidates.append((lead, step, trail, fills))
    return candidates


def _pick_available_color(palette: list[int], used_colors: set[int]) -> int | None:
    choices = [color for color in palette if color not in used_colors]
    if not choices:
        return None
    return random.choice(choices)


def _build_line_spec(
    line_len: int,
    palette: list[int],
    *,
    primary_step: int,
    primary_stop: int | None,
) -> dict | None:
    if line_len - 1 < primary_step:
        return None

    n_markers = 2
    if line_len >= 12 and random.random() < 0.45:
        n_markers = 3

    for _ in range(_LINE_ATTEMPTS):
        marker_colors = random.sample(palette, n_markers)
        marker_specs = [
            {
                "marker_color": marker_colors[0],
                "lead": 0,
                "step": primary_step,
                "trail": primary_step,
                "companion": primary_stop,
                "paint_color": marker_colors[0],
                "stop": line_len - 1 if primary_stop is None else primary_stop,
            }
        ]
        used_positions = {0, primary_step}
        reserved_positions = set()
        if primary_stop is not None:
            reserved_positions.add(primary_stop)
        fill_union = set(range(0, line_len, primary_step))
        prev_lead = 0

        ok = True
        for idx in range(1, n_markers):
            candidates = _candidate_pairs(
                line_len,
                used_positions,
                reserved_positions,
                fill_union,
                prev_lead,
            )
            if not candidates:
                ok = False
                break

            random.shuffle(candidates)
            candidates.sort(
                key=lambda item: (
                    -len(item[3] & fill_union),
                    item[0],
                    item[1],
                )
            )
            lead, step, trail, fills = random.choice(candidates[: min(8, len(candidates))])
            marker_specs.append(
                {
                    "marker_color": marker_colors[idx],
                    "lead": lead,
                    "step": step,
                    "trail": trail,
                    "companion": None,
                    "paint_color": marker_colors[idx],
                    "stop": line_len - 1,
                }
            )
            used_positions.update((lead, trail))
            fill_union |= fills
            prev_lead = lead

        if not ok:
            continue

        line_colors_used = set(marker_colors)
        singletons: list[dict[str, int | str]] = []

        if primary_stop is not None:
            color = _pick_available_color(palette, line_colors_used)
            if color is None:
                continue
            line_colors_used.add(color)
            singletons.append({"dist": primary_stop, "color": color, "kind": "companion"})
            marker_specs[0]["paint_color"] = color

        shuffled_specs = marker_specs[1:]
        random.shuffle(shuffled_specs)
        for spec in shuffled_specs:
            if random.random() >= _P_EXTRA_COMPANION:
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

            color = _pick_available_color(palette, line_colors_used)
            if color is None:
                continue
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

            color = _pick_available_color(palette, line_colors_used)
            if candidates and color is not None:
                dist = random.choice(candidates)
                line_colors_used.add(color)
                used_positions.add(dist)
                singletons.append({"dist": dist, "color": color, "kind": "bystander"})

        if not _line_is_valid(marker_specs, singletons):  # type: ignore[arg-type]
            continue

        dots = []
        for spec in marker_specs:
            dots.append((spec["lead"], spec["marker_color"]))
            dots.append((spec["trail"], spec["marker_color"]))
        for item in singletons:
            dots.append((int(item["dist"]), int(item["color"])))

        painted_sets = [set(_fill_dists(spec)) for spec in marker_specs]
        overlaps = 0
        seen: set[int] = set()
        for current in painted_sets:
            overlaps += len(current & seen)
            seen |= current

        return {
            "dots": dots,
            "markers": marker_specs,
            "singletons": singletons,
            "n_markers": n_markers,
            "overlaps": overlaps,
            "coverage": len(seen),
            "primary_uses_companion": primary_stop is not None,
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


def _choose_primary_stop(line_len: int, step: int) -> int | None:
    if random.random() >= _P_PRIMARY_COMPANION:
        return None
    beats = [
        dist
        for dist in range(max(line_len // 2, step * 3), line_len - 1)
        if dist > step and dist % step == 0
    ]
    if not beats:
        return None
    return random.choice(beats)


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _dense_indices(n_active, num_lines)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    palette_size = random.randint(6, min(8, len(_COLORS)))
    palette = random.sample(_COLORS, palette_size)
    primary_step = random.choice([step for step in _PRIMARY_STEP_POOL if step < line_len])
    primary_stop = _choose_primary_stop(line_len, primary_step)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    overlap_total = 0
    coverage_total = 0
    primary_companion_lines = 0

    for li in active_lines:
        spec = _build_line_spec(
            line_len,
            palette,
            primary_step=primary_step,
            primary_stop=primary_stop,
        )
        if spec is None:
            return None

        overlap_total += int(spec["overlaps"])
        coverage_total += int(spec["coverage"])
        primary_companion_lines += int(spec["primary_uses_companion"])

        for dist, color in spec["dots"]:
            pos = _normalized_to_pos(edge, line_len, dist)
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

    if overlap_total == 0:
        return None
    if coverage_total < n_active * max(5, line_len // 2):
        return None
    if primary_stop is not None and primary_companion_lines != n_active:
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

    active_mask = inp != 0
    out_mask = out != 0
    vertical_contacts = 0
    if by_row:
        lo, hi = active_lines[0], active_lines[-1]
        for r in range(lo, hi):
            for c in range(W):
                if out_mask[r, c] and out_mask[r + 1, c] and out[r, c] != out[r + 1, c]:
                    vertical_contacts += 1
    else:
        lo, hi = active_lines[0], active_lines[-1]
        for c in range(lo, hi):
            for r in range(H):
                if out_mask[r, c] and out_mask[r, c + 1] and out[r, c] != out[r, c + 1]:
                    vertical_contacts += 1

    if vertical_contacts < max(4, n_active - 1):
        return None
    if active_mask.sum() <= 2 * n_active:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
