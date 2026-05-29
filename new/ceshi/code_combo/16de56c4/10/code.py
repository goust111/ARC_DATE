# Combo variant of 16de56c4 (index 10)
# combo_spec: D3/2+D14/0
# Name: 颜色循环+多标记色共存

from __future__ import annotations

from collections import defaultdict
import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The seed task had one repeated color per shooting line, so one line meant one
#    arithmetic progression and one paint color (or one companion override).  D3/2
#    replaces the single paint color with a visible color cycle read from the line's
#    non-bg dots in shooting order.  D14/0 allows several repeated colors on the
#    same line, so one line may now hold 2-3 progressions that overlap.
#    The fused rule makes those two changes interact on the same object:
#    - every repeated color still defines one ray by its two marker dots
#    - the color cycle is built once from all non-bg dots on that line, in visible
#      order from the shooting edge outward
#    - each ray consumes that shared cycle starting from its own first marker's
#      position in the cycle, then later-starting rays overwrite earlier paint
#    Small left-edge example on distances 0..10:
#      dots = [(0,A),(2,A),(4,B),(7,B),(9,C)]
#      cycle = [A,A,B,B,C]
#      ray A: step 2 -> beats 0,2,4,6,8,10 -> paints A,A,B,B,C,A
#      ray B: step 3 -> beats 4,7,10 -> starts at B's first dot inside the same
#             cycle -> paints B,B,C and overwrites A at 4 and 10
#    Compared with the seed task, both the set of painted cells and the final color
#    at overlap cells now depend on separating multiple marker pairs and on reading
#    a shared visible cycle.
#
# b) Complexity calibration:
#    The generator keeps 1-3 marker colors per active line, with at least one
#    multi-marker line and at least one actual overlap of painted beats.  That is
#    enough to make the fused behavior visible, but small enough that humans can
#    still inspect a row/column and enumerate its dots.  Singletons remain sparse:
#    at most one extra dot per line, so the cycle stays short and legible.
#
# c) Implicit-assumption audit of the original solver:
#    - "one repeated color per line" is false; we must process every repeated color.
#    - "any aligned foreign dot is the companion" is false; another repeated color
#      may align, so only singleton colors may serve as companions.
#    - "one scalar paint per line" is false; each ray now walks a cycle.
#    - "fill the whole line when no companion exists" is only safe for the earliest
#      marker that touches the edge.  Interior markers must start from their own
#      first dot and move away from the edge, as in the D14 variant.
#    - "write only on bg or marker cells" is false because later rays must be able
#      to overwrite earlier rays at genuine overlaps.
#
# d) Placement legality:
#    The generator works in normalized distances from the shooting edge.  Marker
#    pairs are placed in increasing non-overlapping order `(lead, lead+step)` so
#    every source dot is unique and in-bounds.  Later markers are chosen only when
#    their future beat set intersects earlier beat sets, guaranteeing real overlay.
#    Optional singletons are placed beyond all marker pairs and are validated to be
#    either the unique aligned companion of one ray or a true bystander aligned to
#    no ray.
#
# e) Visual observability:
#    A solver can recover the common shooting edge exactly as before, then inspect a
#    single active line and see several repeated colors.  Because marker pairs are
#    ordered from the edge outward, the shared cycle is visible directly from spatial
#    order; no color-id sorting or hidden metadata is used.  Overlap cells show why
#    both sub-rules matter: a later ray can recolor an earlier beat, but the chosen
#    overwrite color comes from the shared cycle rather than from a fixed marker or
#    companion color.
#
# f) Visual cleanliness:
#    Active lines remain separated by at least one empty line.  Within a line, only
#    2-3 marker pairs and at most one singleton are used, with pairs not interleaved.
#    This keeps the dot sequence readable and avoids dense clutter while still
#    allowing overlay farther from the edge.
#
# g) Change inferability:
#    The palette per grid is limited to a small sampled subset of 1..9, and colors
#    repeat across lines.  This gives enough repeated evidence that the output colors
#    are copied from visible line-local sequences rather than from arbitrary color
#    remapping.  Requiring at least one multi-marker overlap and rejecting samples
#    whose output matches a uniform-fill multi-ray solver prevents collapse back to
#    a simpler rule.
#
# h) Anti-degeneration checks:
#    The generator rejects inputs if the combo output equals either:
#    - the seed solver behavior, or
#    - a multi-marker but uniform-fill solver.
#    It also requires at least one multi-marker line and at least one overlapped
#    painted cell, so both halves of the combo are materially active.
#
# i) New solver assumptions and generator guarantees:
#    - Each repeated color appears exactly twice on a line.  Guaranteed by line-local
#      distinct marker colors.
#    - Singleton colors are unique on the line, so a singleton can be classified as
#      companion/bystander without ambiguity.
#    - Marker pairs are ordered by first appearance from the shooting edge, and their
#      dots are not interleaved.  The solver does not need this for correctness, but
#      the generator enforces it to keep the shared cycle visually easy to read.
#    - Every intended companion is the only singleton aligned beyond its target ray's
#      second marker.  The line validator checks this before accepting the line.


def _detect_edge(g: np.ndarray, bg: int) -> str:
    H, W = g.shape
    edge_counts = {
        "top": sum(1 for c in range(W) if int(g[0, c]) != bg),
        "bottom": sum(1 for c in range(W) if int(g[H - 1, c]) != bg),
        "left": sum(1 for r in range(H) if int(g[r, 0]) != bg),
        "right": sum(1 for r in range(H) if int(g[r, W - 1]) != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    return winners[0]


def _line_tools(edge: str, H: int, W: int):
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    forward = edge in ("left", "top")

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def dist_from_pos(pos: int) -> int:
        return pos if forward else line_len - 1 - pos

    def pos_from_dist(dist: int) -> int:
        return dist if forward else line_len - 1 - dist

    return by_row, num_lines, line_len, cell, dist_from_pos, pos_from_dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several periodic rays from one dominant edge, with each ray walking a shared visible color cycle.

    Input structure:
    - The background is the most frequent color. Non-bg cells are sparse dots lying
      on rows or columns determined by one unique shooting edge.
    - A shooting-direction line may contain several repeated colors; each repeated
      color appears exactly twice and defines one ray by its first dot and gap.
    - Singleton colors on the same line are optional extras: an aligned singleton
      beyond a ray's second marker is that ray's companion; an unaligned singleton
      is just a bystander dot.
    - The line's non-bg dots, read from the shooting edge outward, form a visible
      ordered color list that will be reused as a cycle.

    Rule:
    - Detect the unique shooting edge. Left/right means process rows; top/bottom
      means process columns.
    - On each active line, recover every repeated-color ray independently. Order
      those rays by the position of their first marker from the shooting edge.
    - For one ray, the beat step is the distance between its two markers. Without a
      companion it paints from its first marker to the far end of the line; with a
      companion it paints only up to that aligned singleton.
    - The paint colors are not uniform: use the full ordered non-bg dot list on that
      line as a repeating cycle, and start that cycle at the ray's own first marker.
      Apply rays in launch order, so later rays overwrite earlier ones at overlaps.

    Key insight:
    The geometric part still comes from repeated pairs on a common shooting line, but
    color now depends on a shared line-level sequence rather than on one marker color.
    Solving requires two separations at once: which repeated colors define distinct
    rays, and where each ray enters the common visible cycle.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    _, num_lines, line_len, cell, dist_from_pos, pos_from_dist = _line_tools(edge, H, W)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        dots: list[tuple[int, int]] = []
        for pos in range(line_len):
            value = int(g[cell(li, pos)])
            if value == bg:
                continue
            dist = dist_from_pos(pos)
            colors[value].append(dist)
            dots.append((dist, value))

        if not dots:
            continue

        repeating = {
            color: sorted(dists)
            for color, dists in colors.items()
            if len(dists) == 2
        }
        singletons = {
            color: dists[0]
            for color, dists in colors.items()
            if len(dists) == 1
        }
        bad = {color: dists for color, dists in colors.items() if len(dists) not in (1, 2)}
        assert not bad, f"Line {li} has colors with invalid multiplicity: {bad}"
        assert repeating, f"Line {li} has dots but no repeated marker color: {dict(colors)}"

        ordered_dots = sorted(dots)
        cycle = [color for _, color in ordered_dots]
        cycle_index_by_dist = {dist: idx for idx, (dist, _) in enumerate(ordered_dots)}

        rays = []
        for marker, pair in repeating.items():
            lead_d, trail_d = pair
            step = trail_d - lead_d
            assert step > 0, f"Line {li} marker {marker} has invalid pair {pair}"

            companions = [
                (dist, color)
                for color, dist in singletons.items()
                if dist > trail_d and (dist - lead_d) % step == 0
            ]
            assert len(companions) <= 1, (
                f"Line {li} marker {marker} has ambiguous companions: {companions}"
            )
            stop_d = companions[0][0] if companions else line_len - 1
            rays.append((lead_d, step, stop_d, cycle_index_by_dist[lead_d]))

        rays.sort(key=lambda item: item[0])
        for lead_d, step, stop_d, start_idx in rays:
            beat_idx = 0
            for dist in range(lead_d, stop_d + 1, step):
                paint = int(cycle[(start_idx + beat_idx) % len(cycle)])
                out[cell(li, pos_from_dist(dist))] = paint
                beat_idx += 1

    return out


def _uniform_multi_main(grid: np.ndarray) -> np.ndarray:
    """D14-style multi-ray solver without the shared color cycle."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    _, num_lines, line_len, cell, dist_from_pos, pos_from_dist = _line_tools(edge, H, W)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for pos in range(line_len):
            value = int(g[cell(li, pos)])
            if value != bg:
                colors[value].append(dist_from_pos(pos))

        if not colors:
            continue

        repeating = {
            color: sorted(dists)
            for color, dists in colors.items()
            if len(dists) == 2
        }
        singletons = {
            color: dists[0]
            for color, dists in colors.items()
            if len(dists) == 1
        }
        assert repeating, f"Uniform-check line {li} has no repeated marker color"

        rays = []
        for marker, pair in repeating.items():
            lead_d, trail_d = pair
            step = trail_d - lead_d
            assert step > 0, f"Uniform-check line {li} marker {marker} has invalid pair {pair}"

            companions = [
                (dist, color)
                for color, dist in singletons.items()
                if dist > trail_d and (dist - lead_d) % step == 0
            ]
            assert len(companions) <= 1, (
                f"Uniform-check line {li} marker {marker} has ambiguous companions: {companions}"
            )
            if companions:
                stop_d, paint = companions[0]
            else:
                stop_d, paint = line_len - 1, marker
            rays.append((lead_d, step, stop_d, paint))

        rays.sort(key=lambda item: item[0])
        for lead_d, step, stop_d, paint in rays:
            for dist in range(lead_d, stop_d + 1, step):
                out[cell(li, pos_from_dist(dist))] = int(paint)

    return out


def _seed_main(grid: np.ndarray) -> np.ndarray:
    """Original solver behavior, kept only for anti-degeneration checks."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for pos in range(line_len):
            value = int(g[cell(li, pos)])
            if value != bg:
                colors[value].append(pos)

        if not colors:
            continue

        repeating = {color: poses for color, poses in colors.items() if len(poses) >= 2}
        assert repeating, f"Seed-check line {li} has no repeated color"

        for marker, poses in repeating.items():
            assert len(poses) == 2, f"Seed-check marker {marker} has invalid count {len(poses)}"
            a, b = sorted(poses)
            step = b - a
            offset = a % step

            companion = None
            for color, other_poses in colors.items():
                if color == marker:
                    continue
                if any(pos % step == offset for pos in other_poses):
                    companion = color
                    break

            if companion is not None:
                paint = companion
                all_pos = poses + colors[companion]
                lo, hi = min(all_pos), max(all_pos)
            else:
                paint = marker
                lo, hi = 0, line_len - 1

            for pos in range(lo, hi + 1):
                if pos % step == offset:
                    rc = cell(li, pos)
                    cur = int(out[rc])
                    if cur == bg or cur == marker:
                        out[rc] = int(paint)

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_COMPANION = 0.25
_P_BYSTANDER = 0.30
_MAX_RETRIES = 500
_LINE_ATTEMPTS = 350


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
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


def _pick_marker_count(line_len: int, multi_required: bool) -> int:
    max_markers = 3 if line_len >= 12 else 2
    if multi_required:
        choices = [k for k in (2, 3) if k <= max_markers]
        assert choices, f"Cannot require multi-marker line at length {line_len}"
        return random.choice(choices)

    pool = [1, 1, 2]
    if max_markers >= 3:
        pool.append(3)
    return random.choice(pool)


def _fill_dists(spec: dict[str, int], line_len: int) -> set[int]:
    stop = spec["companion"] if spec["companion"] is not None else line_len - 1
    return set(range(spec["lead"], stop + 1, spec["step"]))


def _line_is_valid(marker_specs: list[dict[str, int]], singletons: list[dict[str, int]]) -> bool:
    used = set()
    for spec in marker_specs:
        if spec["lead"] >= spec["trail"]:
            return False
        if spec["lead"] in used or spec["trail"] in used:
            return False
        used.add(spec["lead"])
        used.add(spec["trail"])
    for item in singletons:
        if item["dist"] in used:
            return False
        used.add(item["dist"])

    for spec in marker_specs:
        aligned = []
        for item in singletons:
            dist = item["dist"]
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
    forced_extra_kind: str | None = None,
) -> dict | None:
    n_markers = _pick_marker_count(line_len, multi_required)
    if len(palette) < n_markers + 1:
        return None

    for _ in range(_LINE_ATTEMPTS):
        marker_colors = random.sample(palette, n_markers)
        marker_specs = []
        fill_union: set[int] = set()
        prev_trail = -1
        ok = True

        for idx, color in enumerate(marker_colors):
            remaining = n_markers - idx - 1
            candidate_pairs = []
            lead_lo = 0 if idx == 0 else prev_trail + 1
            max_lead = line_len - 2 * remaining - 2
            for lead in range(lead_lo, max_lead + 1):
                if idx == 0 and lead != 0:
                    continue
                max_step = min(_STEP_RANGE[1], line_len - 1 - lead - 2 * remaining)
                for step in range(_STEP_RANGE[0], max_step + 1):
                    trail = lead + step
                    fills = set(range(lead, line_len, step))
                    if idx > 0 and not (fills & fill_union):
                        continue
                    candidate_pairs.append((lead, step, trail, fills))

            if not candidate_pairs:
                ok = False
                break

            lead, step, trail, fills = random.choice(candidate_pairs)
            marker_specs.append(
                {
                    "marker_color": color,
                    "lead": lead,
                    "step": step,
                    "trail": trail,
                    "companion": None,
                }
            )
            fill_union |= fills
            prev_trail = trail

        if not ok:
            continue

        used_dists = {spec["lead"] for spec in marker_specs} | {spec["trail"] for spec in marker_specs}
        max_trail = max(spec["trail"] for spec in marker_specs)
        line_colors_used = set(marker_colors)
        singletons: list[dict[str, int]] = []

        kind_pool = ["none", "companion", "bystander"]
        weights = [0.45, _P_COMPANION, _P_BYSTANDER]
        extra_kind = (
            forced_extra_kind
            if forced_extra_kind is not None
            else random.choices(kind_pool, weights=weights, k=1)[0]
        )

        if extra_kind == "companion":
            targets = marker_specs[:]
            random.shuffle(targets)
            placed = False
            for spec in targets:
                candidates = []
                for dist in range(max_trail + 1, line_len):
                    if dist in used_dists:
                        continue
                    if (dist - spec["lead"]) % spec["step"] != 0:
                        continue
                    if any(
                        dist > other["trail"] and (dist - other["lead"]) % other["step"] == 0
                        for other in marker_specs
                        if other is not spec
                    ):
                        continue
                    candidates.append(dist)
                if not candidates:
                    continue
                available_colors = [c for c in palette if c not in line_colors_used]
                if not available_colors:
                    break
                dist = random.choice(candidates)
                color = random.choice(available_colors)
                line_colors_used.add(color)
                used_dists.add(dist)
                spec["companion"] = dist
                singletons.append({"dist": dist, "color": color, "kind": "companion"})
                placed = True
                break
            if forced_extra_kind == "companion" and not placed:
                continue

        elif extra_kind == "bystander":
            candidates = []
            for dist in range(max_trail + 1, line_len):
                if dist in used_dists:
                    continue
                if any((dist - spec["lead"]) % spec["step"] == 0 for spec in marker_specs):
                    continue
                candidates.append(dist)
            available_colors = [c for c in palette if c not in line_colors_used]
            if candidates and available_colors:
                dist = random.choice(candidates)
                color = random.choice(available_colors)
                singletons.append({"dist": dist, "color": color, "kind": "bystander"})
                used_dists.add(dist)
                line_colors_used.add(color)
            elif forced_extra_kind == "bystander":
                continue
        elif forced_extra_kind == "none":
            pass

        if not _line_is_valid(marker_specs, singletons):
            continue

        dots = []
        for spec in marker_specs:
            dots.append((spec["lead"], spec["marker_color"]))
            dots.append((spec["trail"], spec["marker_color"]))
        for item in singletons:
            dots.append((item["dist"], item["color"]))
        dots.sort()

        overlap = 0
        painted_so_far: set[int] = set()
        for spec in marker_specs:
            fills = _fill_dists(spec, line_len)
            overlap += len(fills & painted_so_far)
            painted_so_far |= fills

        return {
            "dots": dots,
            "markers": marker_specs,
            "singletons": singletons,
            "n_markers": n_markers,
            "overlap": overlap,
        }

    return None


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

    palette = random.sample(_COLORS, random.randint(5, min(8, len(_COLORS))))
    required_multi = random.randint(1, min(max(1, n_active // 2), n_active))
    multi_lines = set(random.sample(active_lines, required_multi))
    role_lines = random.sample(active_lines, 3)
    forced_extra_by_line = {
        role_lines[0]: "companion",
        role_lines[1]: "bystander",
        role_lines[2]: "none",
    }

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    multi_count = 0
    overlap_total = 0
    companion_count = 0
    bystander_count = 0
    plain_count = 0

    for li in active_lines:
        spec = _build_line_spec(
            line_len,
            palette,
            multi_required=li in multi_lines,
            forced_extra_kind=forced_extra_by_line.get(li),
        )
        if spec is None:
            return None

        if spec["n_markers"] > 1:
            multi_count += 1
            overlap_total += spec["overlap"]
        singleton_kinds = {item["kind"] for item in spec["singletons"]}
        if "companion" in singleton_kinds:
            companion_count += 1
        if "bystander" in singleton_kinds:
            bystander_count += 1
        if not spec["singletons"]:
            plain_count += 1

        for dist, color in spec["dots"]:
            pos = _normalized_to_pos(edge, line_len, dist)
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

    if multi_count == 0 or overlap_total == 0:
        return None
    if companion_count == 0 or bystander_count == 0 or plain_count == 0:
        return None

    shooting_edge_dots = len(active_lines)
    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
        uniform = _uniform_multi_main(inp)
        seed = _seed_main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, uniform):
        return None
    if np.array_equal(out, seed):
        return None

    cycle_delta = int(np.count_nonzero(out != uniform))
    if cycle_delta < 3:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
