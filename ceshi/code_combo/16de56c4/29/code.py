# Combo variant of 16de56c4 (index 29)
# combo_spec: D14/0+D18/2
# Name: 多标记色共存+颜色层级

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for combo D14/0 + D18/2
# =======================================
# a) Concrete behavior and trace
#    - Original task:
#      - each active row/column had exactly one repeated color;
#      - that pair defined one arithmetic-progression ray;
#      - if another aligned singleton existed, it stopped the ray and supplied
#        the paint color.
#    - D14/0 already breaks the first assumption: one line may now contain
#      several repeated colors, hence several independent rays.
#    - This combo changes the interaction rule again: when two rays cover the
#      same cell, visible launch order no longer decides the winner.  Instead,
#      the final color is chosen by a fixed visible hierarchy
#      green < blue < red, i.e. 3 < 1 < 2 in ARC ids.
#    - Mini trace on a left-edge row of length 11:
#      - red markers at distances 0 and 2 define beats 0,2,4,6,8,10;
#      - green markers at distances 1 and 4 define beats 1,4,7,10;
#      - old D14/0 layering would paint the row in start order, so the later
#        green ray would take cells 4 and 10;
#      - fused rule compares only the paint colors at overlaps, therefore red
#        keeps 4 and 10 even though the green ray starts later.
#    - The generator also keeps some companion lines, so the original "aligned
#      singleton stops the ray and can change its paint color" mechanism still
#      participates in the combined task rather than being dropped.
#
# b) Complexity calibration
#    - Only three priority colors are used for all ray paints: green, blue,
#      red.  This is enough to show a genuine hierarchy without asking the
#      solver to infer an arbitrary long palette order.
#    - Every sample contains three forced multi-ray lines exhibiting the three
#      pairwise contests (blue vs green, red vs blue, red vs green).  That
#      gives repeated evidence for the color order inside a single sample.
#    - Remaining lines are simple one-ray lines, usually with a companion or a
#      bystander, so the scene still resembles the seed task instead of turning
#      into an unrestricted overlap puzzle.
#
# c) Implicit-assumption audit of the original main()
#    - `repeating` can no longer be treated as a singleton; every repeated color
#      on the same line must be parsed into its own ray.
#    - "any aligned foreign dot is the companion" is no longer valid because an
#      aligned foreign color may be another repeated marker pair.  Companions are
#      restricted to singleton colors only.
#    - "later ray overwrites earlier paint" is the D14-only rule and is exactly
#      what D18/2 replaces.  The new solver must collect all claims for a cell
#      and resolve them by paint-color rank.
#    - Because interior marker pairs are now allowed, no-companion rays should
#      fill only from their first marker outward from the shooting edge, never
#      backward toward that edge.
#
# d) Placement legality
#    - Generation uses normalized edge-distance coordinates `d = 0..line_len-1`.
#      The first marker of every active line is always at `d=0`, preserving the
#      single dominant shooting edge cue.
#    - Multi-ray lines are built from searched templates whose four marker cells
#      are all distinct and whose fill sets genuinely overlap.
#    - Companion dots are placed only on their own ray beats beyond the second
#      marker and are rejected if they accidentally align to any other ray.
#    - Bystanders use non-priority colors and are placed off every ray beat, so
#      they remain untouched and visually distinct.
#
# e) Visual observability
#    - The output of a decisive conflict line contains both contestant colors:
#      the lower-priority ray still paints its private beats, but loses exactly
#      the shared beats.  That makes "color decides overlaps" visible by eye.
#    - Because all three pairwise comparisons are forced in each sample, a human
#      can infer the full order green < blue < red from a few train pairs
#      without relying on hidden numeric ids.
#    - Companion lines still reveal the original stopping/recolor behavior, so
#      the sample teaches both the seed rule and the new interaction layer.
#
# f) Visual cleanliness
#    - There are at most two repeated colors on a line.  This keeps rows/columns
#      readable and avoids dense three-way piles of dots.
#    - Only one optional singleton companion and one optional bystander are used
#      on a line, and distractor colors never serve as ray paints.
#    - Active lines remain spaced apart by at least one blank line, matching the
#      seed task's sparse appearance.
#
# g) Non-degeneration metric
#    - The generator compares the fused solver against a D14-only baseline that
#      still uses start-order overwrite.  A sample is accepted only if the two
#      outputs differ on at least two cells.
#    - It also requires at least three multi-marker lines and all three pairwise
#      priority contests, ensuring the sample is not merely the original task or
#      the D14-only variant in disguise.
#
# h) New solver assumptions and generator guarantees
#    - Assumption: every repeated color on a line appears exactly twice.
#      Guarantee: generator uses distinct line-local marker colors and stamps
#      each pair exactly once.
#    - Assumption: every ray paint belongs to the visible priority palette
#      {green(3), blue(1), red(2)}.
#      Guarantee: marker colors and companion colors are chosen only from that
#      palette; bystanders use a disjoint distractor palette.
#    - Assumption: each marker sees at most one aligned singleton companion.
#      Guarantee: candidate companions are validated against all rays on the line.


PRIORITY_COLORS: tuple[int, ...] = (3, 1, 2)  # green < blue < red
PRIORITY_RANK = {color: rank for rank, color in enumerate(PRIORITY_COLORS)}
DISTRACTOR_COLORS: tuple[int, ...] = (4, 5, 6, 7, 8, 9)

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 7)
_STEP_RANGE = (1, 5)
_EDGES = ("top", "bottom", "left", "right")
_MAX_RETRIES = 500

_P_FILLER_COMPANION = 0.75
_P_FILLER_BYSTANDER = 0.45
_P_SPECIAL_BYSTANDER = 0.35


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


def _orientation_helpers(edge: str, shape: tuple[int, int]):
    H, W = map(int, shape)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    forward = edge in ("left", "top")

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (int(line), int(pos)) if by_row else (int(pos), int(line))

    def dist_from_edge(pos: int) -> int:
        return int(pos) if forward else int(line_len - 1 - pos)

    def pos_from_dist(dist: int) -> int:
        return int(dist) if forward else int(line_len - 1 - dist)

    return by_row, num_lines, line_len, cell, dist_from_edge, pos_from_dist


def _extract_line_rays(
    g: np.ndarray,
    bg: int,
    line: int,
    line_len: int,
    cell,
    dist_from_edge,
) -> tuple[dict[int, list[int]], list[dict[str, int]]]:
    colors: dict[int, list[int]] = defaultdict(list)
    for p in range(line_len):
        v = int(g[cell(line, p)])
        if v != bg:
            colors[v].append(int(p))

    if not colors:
        return {}, []

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
        f"Line {line} ({'row' if cell(line, 0)[0] == line else 'col'}) has non-bg dots "
        f"{dict(colors)} but no repeating color"
    )

    rays: list[dict[str, int]] = []
    for marker, positions in repeating.items():
        lead_pos, trail_pos = positions
        lead_d = dist_from_edge(lead_pos)
        trail_d = dist_from_edge(trail_pos)
        step = int(trail_d - lead_d)
        assert step > 0, f"Marker {marker} on line {line} has invalid positions {positions}"

        candidates = []
        for other_color, pos in singletons.items():
            d = dist_from_edge(pos)
            if d > trail_d and (d - lead_d) % step == 0:
                candidates.append((int(d), int(other_color)))
        assert len(candidates) <= 1, (
            f"Line {line} marker {marker} has ambiguous companions: {candidates}"
        )

        if candidates:
            stop_d, paint = candidates[0]
        else:
            stop_d, paint = line_len - 1, int(marker)
        rays.append(
            {
                "marker": int(marker),
                "lead": int(lead_d),
                "trail": int(trail_d),
                "step": int(step),
                "stop": int(stop_d),
                "paint": int(paint),
            }
        )

    rays.sort(key=lambda ray: int(ray["lead"]))
    return {int(k): [int(p) for p in v] for k, v in colors.items()}, rays


def _solve(grid: np.ndarray, *, use_priority: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _detect_edge(g, bg)
    _by_row, num_lines, line_len, cell, dist_from_edge, pos_from_dist = _orientation_helpers(edge, g.shape)

    for line in range(num_lines):
        colors, rays = _extract_line_rays(g, bg, line, line_len, cell, dist_from_edge)
        if not colors:
            continue

        if use_priority:
            claims: dict[int, tuple[int, int]] = {}
            for ray in rays:
                paint = int(ray["paint"])
                rank = PRIORITY_RANK.get(paint)
                assert rank is not None, f"Unknown priority color: {paint}"
                for d in range(int(ray["lead"]), int(ray["stop"]) + 1, int(ray["step"])):
                    prev = claims.get(int(d))
                    if prev is None or rank > prev[0]:
                        claims[int(d)] = (int(rank), paint)
            for d, (_rank, paint) in claims.items():
                out[cell(line, pos_from_dist(d))] = int(paint)
        else:
            for ray in rays:
                for d in range(int(ray["lead"]), int(ray["stop"]) + 1, int(ray["step"])):
                    out[cell(line, pos_from_dist(d))] = int(ray["paint"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct several edge-fired rays, then merge their overlaps by color rank.

    Input structure:
    - Background is the most frequent color.  All non-background dots lie on rows
      or columns determined by one uniquely dominant shooting edge.
    - One shooting-direction line may contain multiple repeated colors.  Each
      repeated color appears exactly twice and defines one periodic ray by its
      start position and gap.
    - Singleton dots are of two kinds: aligned singletons beyond a ray's second
      marker are companions, while unaligned singletons are bystanders.
    - All ray paint colors come from the visible priority palette
      green < blue < red.

    Rule:
    - Detect the shooting edge; left/right means solve row-wise, top/bottom
      means solve column-wise.
    - For every repeated color on one line, compute its step from the two marker
      positions.  The ray fills that beat starting at its first marker and
      continuing away from the shooting edge.
    - If a singleton lies farther along the same beat, it is that ray's
      companion: it limits the fill range and supplies the paint color.
      Otherwise the ray keeps its own marker color and fills to the far end.
    - When several rays cover the same cell, ignore launch order and keep only
      the highest-priority paint color.  Bystanders are left unchanged because
      they do not lie on any active beat.

    Key insight:
    The arithmetic progressions from the seed task still define the rays, but
    the fusion requires two separations at once: repeated colors create rays,
    singleton colors only stop rays, and overlap ownership is decided globally
    by the visible color hierarchy rather than by the order in which rays begin.
    """
    return _solve(grid, use_priority=True)


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
    return [int(picks[i] + i) for i in range(n)]


def _normalized_to_pos(edge: str, line_len: int, dist: int) -> int:
    return int(dist) if edge in ("left", "top") else int(line_len - 1 - dist)


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
    counts: dict[str, int] = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(int(r), int(c)))
    return counts


def _fill_set(ray: dict[str, int]) -> set[int]:
    return set(range(int(ray["lead"]), int(ray["stop"]) + 1, int(ray["step"])))


def _line_metrics(line_spec: dict) -> dict[str, object]:
    rays = [dict(ray) for ray in line_spec["rays"]]
    multi = int(len(rays) >= 2)
    overlap_cells = 0
    pair_tags: set[tuple[int, int]] = set()
    for i in range(len(rays)):
        ai = _fill_set(rays[i])
        for j in range(i + 1, len(rays)):
            bj = _fill_set(rays[j])
            inter = ai & bj
            if inter:
                overlap_cells += len(inter)
                pair_tags.add(tuple(sorted((int(rays[i]["paint"]), int(rays[j]["paint"])))))
    return {
        "multi_marker": multi,
        "overlap_cells": int(overlap_cells),
        "priority_pairs": pair_tags,
        "has_companion": any(ray["paint"] != ray["marker"] for ray in rays),
    }


@lru_cache(maxsize=None)
def _two_marker_templates(line_len: int) -> tuple[dict[str, int], ...]:
    limit = min(_STEP_RANGE[1], line_len - 1)
    templates: list[dict[str, int]] = []
    for step_hi in range(_STEP_RANGE[0], limit + 1):
        trail_hi = step_hi
        hi_fill = set(range(0, line_len, step_hi))
        for lead_lo in range(1, line_len - 1):
            max_step_lo = min(_STEP_RANGE[1], line_len - 1 - lead_lo)
            for step_lo in range(_STEP_RANGE[0], max_step_lo + 1):
                trail_lo = lead_lo + step_lo
                if len({0, trail_hi, lead_lo, trail_lo}) < 4:
                    continue
                lo_fill = set(range(lead_lo, line_len, step_lo))
                overlaps = sorted(int(d) for d in (hi_fill & lo_fill))
                if not overlaps:
                    continue
                if overlaps[0] > line_len - 3:
                    continue
                templates.append(
                    {
                        "step_hi": int(step_hi),
                        "lead_lo": int(lead_lo),
                        "step_lo": int(step_lo),
                        "overlap_count": int(len(overlaps)),
                        "first_overlap": int(overlaps[0]),
                    }
                )
    random_order = sorted(
        templates,
        key=lambda item: (abs(int(item["overlap_count"]) - 2), int(item["first_overlap"]), int(item["lead_lo"])),
    )
    return tuple(random_order)


def _candidate_bystander_dists(
    line_len: int,
    rays: list[dict[str, int]],
    used_dists: set[int],
) -> list[int]:
    beats = set().union(*(_fill_set(ray) for ray in rays))
    blocked = set(used_dists) | beats
    return [d for d in range(1, line_len - 1) if d not in blocked]


def _attach_companion(
    line_len: int,
    rays: list[dict[str, int]],
    used_dists: set[int],
    used_colors: set[int],
) -> bool:
    candidates = []
    for idx, ray in enumerate(rays):
        spare_colors = [c for c in PRIORITY_COLORS if c not in used_colors]
        if not spare_colors:
            continue
        for dist in range(int(ray["trail"]) + int(ray["step"]), line_len - 1, int(ray["step"])):
            if dist in used_dists:
                continue
            accidental = False
            for j, other in enumerate(rays):
                if j == idx:
                    continue
                if dist > int(other["trail"]) and (dist - int(other["lead"])) % int(other["step"]) == 0:
                    accidental = True
                    break
            if accidental:
                continue
            candidates.append((idx, int(dist), list(spare_colors)))
    if not candidates:
        return False
    idx, dist, spare_colors = random.choice(candidates)
    color = int(random.choice(spare_colors))
    rays[idx]["stop"] = int(dist)
    rays[idx]["paint"] = int(color)
    used_dists.add(int(dist))
    used_colors.add(int(color))
    return True


def _make_line_spec(
    line_len: int,
    rays: list[dict[str, int]],
    *,
    allow_companion: bool,
    bystander_prob: float,
) -> dict | None:
    used_dists = {int(ray["lead"]) for ray in rays} | {int(ray["trail"]) for ray in rays}
    used_colors = {int(ray["marker"]) for ray in rays}
    singletons: list[dict[str, int | str]] = []

    if allow_companion and random.random() < _P_FILLER_COMPANION:
        before = [(int(ray["paint"]), int(ray["stop"])) for ray in rays]
        if _attach_companion(line_len, rays, used_dists, used_colors):
            after = [(int(ray["paint"]), int(ray["stop"])) for ray in rays]
            for idx, (prev, cur) in enumerate(zip(before, after)):
                if prev != cur:
                    singletons.append(
                        {
                            "dist": int(rays[idx]["stop"]),
                            "color": int(rays[idx]["paint"]),
                            "kind": "companion",
                        }
                    )
                    break

    if random.random() < bystander_prob:
        candidates = _candidate_bystander_dists(line_len, rays, used_dists)
        if candidates:
            singletons.append(
                {
                    "dist": int(random.choice(candidates)),
                    "color": int(random.choice(DISTRACTOR_COLORS)),
                    "kind": "bystander",
                }
            )

    dots = []
    for ray in rays:
        dots.append((int(ray["lead"]), int(ray["marker"])))
        dots.append((int(ray["trail"]), int(ray["marker"])))
    for item in singletons:
        dots.append((int(item["dist"]), int(item["color"])))

    if len({int(dist) for dist, _color in dots}) != len(dots):
        return None

    line_spec = {
        "rays": [dict(ray) for ray in rays],
        "singletons": singletons,
        "dots": dots,
    }
    line_spec.update(_line_metrics(line_spec))
    return line_spec


def _build_priority_conflict_line(line_len: int, high_color: int, low_color: int) -> dict | None:
    templates = list(_two_marker_templates(line_len))
    if not templates:
        return None
    best = [tpl for tpl in templates if int(tpl["overlap_count"]) >= 2] or templates
    tpl = random.choice(best[: min(10, len(best))])
    rays = [
        {
            "marker": int(high_color),
            "lead": 0,
            "trail": int(tpl["step_hi"]),
            "step": int(tpl["step_hi"]),
            "stop": int(line_len - 1),
            "paint": int(high_color),
        },
        {
            "marker": int(low_color),
            "lead": int(tpl["lead_lo"]),
            "trail": int(tpl["lead_lo"] + tpl["step_lo"]),
            "step": int(tpl["step_lo"]),
            "stop": int(line_len - 1),
            "paint": int(low_color),
        },
    ]
    return _make_line_spec(
        line_len,
        rays,
        allow_companion=False,
        bystander_prob=_P_SPECIAL_BYSTANDER,
    )


def _build_single_ray_line(line_len: int, *, require_companion: bool) -> dict | None:
    attempts = 60
    for _ in range(attempts):
        marker = int(random.choice(PRIORITY_COLORS))
        max_step = min(_STEP_RANGE[1], line_len - 2)
        if max_step < _STEP_RANGE[0]:
            return None
        step = int(random.randint(_STEP_RANGE[0], max_step))
        rays = [
            {
                "marker": marker,
                "lead": 0,
                "trail": int(step),
                "step": int(step),
                "stop": int(line_len - 1),
                "paint": marker,
            }
        ]
        spec = _make_line_spec(
            line_len,
            rays,
            allow_companion=True,
            bystander_prob=_P_FILLER_BYSTANDER,
        )
        if spec is None:
            continue
        if require_companion and not bool(spec["has_companion"]):
            continue
        return spec
    return None


def _sample_metrics_from_grid(inp: np.ndarray) -> dict[str, int]:
    fused = main(inp)
    d14 = _solve(inp, use_priority=False)
    bg = int(get_bg(inp))
    edge = _detect_edge(inp, bg)
    _by_row, num_lines, line_len, cell, dist_from_edge, _pos_from_dist = _orientation_helpers(edge, inp.shape)

    multi_lines = 0
    overlap_cells = 0
    companion_lines = 0
    pair_mask = set()
    for line in range(num_lines):
        colors, rays = _extract_line_rays(inp, bg, line, line_len, cell, dist_from_edge)
        if not colors:
            continue
        if len(rays) >= 2:
            multi_lines += 1
        if any(int(ray["paint"]) != int(ray["marker"]) for ray in rays):
            companion_lines += 1
        for i in range(len(rays)):
            ai = _fill_set(rays[i])
            for j in range(i + 1, len(rays)):
                bj = _fill_set(rays[j])
                inter = ai & bj
                if inter:
                    overlap_cells += len(inter)
                    pair_mask.add(tuple(sorted((int(rays[i]["paint"]), int(rays[j]["paint"])))))

    return {
        "multi_marker_lines": int(multi_lines),
        "overlap_cells": int(overlap_cells),
        "companion_lines": int(companion_lines),
        "priority_pairs": int(len(pair_mask)),
        "priority_conflict_cells": int(np.sum(fused != d14)),
    }


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if len(active_lines) != n_active:
        return None

    pair_specs = [
        (1, 3),  # blue beats green
        (2, 1),  # red beats blue
        (2, 3),  # red beats green
    ]
    if n_active < len(pair_specs) + 1:
        return None

    line_specs: list[dict] = []
    pair_tags = set()
    for high, low in pair_specs:
        spec = _build_priority_conflict_line(line_len, high, low)
        if spec is None:
            return None
        line_specs.append(spec)
        pair_tags |= set(spec["priority_pairs"])

    filler_needed = n_active - len(line_specs)
    companion_seen = any(bool(spec["has_companion"]) for spec in line_specs)
    for idx in range(filler_needed):
        spec = _build_single_ray_line(line_len, require_companion=(not companion_seen and idx == filler_needed - 1))
        if spec is None:
            return None
        companion_seen = companion_seen or bool(spec["has_companion"])
        line_specs.append(spec)

    if not companion_seen:
        return None

    random.shuffle(line_specs)

    by_row = edge in ("left", "right")

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (int(line), int(pos)) if by_row else (int(pos), int(line))

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    multi_lines = 0
    overlap_total = 0
    pair_tags = set()

    for line, spec in zip(active_lines, line_specs):
        if int(spec["multi_marker"]):
            multi_lines += 1
        overlap_total += int(spec["overlap_cells"])
        pair_tags |= set(spec["priority_pairs"])
        for dist, color in spec["dots"]:
            pos = _normalized_to_pos(edge, line_len, int(dist))
            r, c = cell(int(line), int(pos))
            if inp[r, c] != 0:
                return None
            inp[r, c] = int(color)
            all_rc.append((int(r), int(c)))

    if multi_lines < 3 or overlap_total < 4 or len(pair_tags) < 3:
        return None

    shooting_edge_dots = len(active_lines)
    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        fused = main(inp)
        d14 = _solve(inp, use_priority=False)
    except Exception:
        return None

    if np.array_equal(inp, fused):
        return None
    if np.array_equal(fused, d14):
        return None

    metrics = _sample_metrics_from_grid(inp)
    if metrics["multi_marker_lines"] < 3:
        return None
    if metrics["priority_pairs"] < 3:
        return None
    if metrics["companion_lines"] < 1:
        return None
    if metrics["priority_conflict_cells"] < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
