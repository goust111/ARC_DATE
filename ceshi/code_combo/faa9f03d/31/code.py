# Combo variant of faa9f03d (index 31)
# combo_spec: D5/2+D8/1+D11/0
# Name: 交叉棋盘格 + 高密度多色 + 三色交叉

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: after singleton markers are reassigned to line colors and short straight
#      gaps are repaired, each overlap cell is shown as exactly one winning line color.
#    - This combo changes the overlap display itself. For ordinary 2-color crossings, the two
#      participating colors alternate by grid parity, so one crossing cell shows the horizontal
#      claimant and the next parity shows the vertical claimant.
#    - Three-color crossings need a geometrically feasible topology. Three full through-lines
#      cannot share one 4-neighbor cell without overlapping along whole segments, so this
#      variant uses two repeated tri-color motifs instead:
#         1. `V + L + R`: one vertical through-line plus two half-horizontal lines that meet
#            from left and right at the same crossing cell.
#         2. `H + U + D`: one horizontal through-line plus two half-vertical lines that meet
#            from above and below at the same crossing cell.
#      At such a tri-color cell, the visible color follows a 3-cycle:
#         `L -> V -> R` or `U -> H -> D` by `(r + c) % 3`.
#    - Small trace for `V + L + R` at row/col `(6, 8)`:
#      the original logic could only choose one winner at `(6, 8)`. Here the repaired logical
#      claims are still three colors, but the visible output becomes the left stub color if
#      `(6 + 8) % 3 == 0`, the vertical through color if it is `1`, or the right stub color if
#      it is `2`. Input crossings are phase-shifted so the output is not a no-op recoloring.
# b) Complexity calibration
#    - D8/1 is realized as a fixed 18x18 high-density scaffold with 6 line colors. The scene is
#      denser than the 12x12 original but still readable because crossings belong to just four
#      local role families: `H`, `V`, `L/R`, `U/D`.
#    - The tri-color motif is repeated twice per sample, and the 2-color checkerboard motif also
#      appears multiple times, so humans get repeated support for both rules from 2-4 examples.
# c) Hidden-assumption audit against the original solver
#    - Original crossing logic assumed every ambiguous cell could be resolved to one winner using
#      global fragmentation counts. That is false here.
#    - Original code also implicitly assumed every claimant already owned the crossing cell after
#      marker repair. That fails for hidden half-lines in a tri-color crossing; they terminate one
#      step away from the displayed color. The new solver therefore infers hidden claims from
#      adjacent endpoints as well as visible occupancy.
#    - Yellow controls still conceptually override overlaps, so that priority is preserved even
#      though this generator uses red-only inputs for reliability.
# d) Placement legality
#    - The generator constructs the full logical network first, then places isolated red markers
#      away from crossings and their 4-neighbors. That preserves unique marker assignment, avoids
#      overlapping markers, and keeps local crossing roles legible.
#    - Every intended ambiguous cell is one of: `H+V`, `V+L+R`, or `H+U+D`. The scaffold avoids
#      accidental same-axis multi-cell overlaps.
# e) Visual observability
#    - No color-id ordering is used. The output rule depends only on visible local geometry:
#      horizontal-vs-vertical parity for 2-color crossings, and side/through roles for 3-color
#      crossings. The generator forces both ordinary checkerboard and tri-color cycle behavior to
#      appear in every sample.
# f) Visual cleanliness
#    - Even in the larger grid, only straight lines and half-lines are used. Red markers are kept
#      out of crossing neighborhoods, so the overlap pattern remains easy to read.
# g) Inference and anti-degeneracy
#    - A sample is rejected unless at least one repaired crossing differs from the old
#      fragmentation-first winner on the same logical claims, and unless both 2-color and 3-color
#      crossings are present. This prevents accidental collapse back to the original task.
# h) New solver assumptions and generator guarantees
#    - Assumption: every ambiguous cell fits one supported local role signature.
#      Guarantee: the scaffold only builds `H+V`, `V+L+R`, and `H+U+D`.
#    - Assumption: hidden stub claims appear only one cell beyond a visible endpoint.
#      Guarantee: red markers never occupy crossing-adjacent cells.
#    - Assumption: line colors remain non-singleton after input corruption.
#      Guarantee: markers are placed on sufficiently long segments only.


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_LABEL = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}
HORIZONTAL_LABELS = {"L", "R", "H"}
VERTICAL_LABELS = {"U", "D", "V"}

BG = 0
RED = 2
YELLOW = 4
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

H = W = 18
GEN_RETRIES = 200
RED_COUNT_RANGE = (3, 5)
MIN_RED_MARGIN = 1


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _neighbors(shape: tuple[int, int], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            out.append((nr, nc))
    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment(a, b)
        cells.extend(seg[1:] if cells else seg)
    return cells


def _dirs_in(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(dr, dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pts]


def _axis_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r, c - 1) in mask or (r, c + 1) in mask
    return (r - 1, c) in mask or (r + 1, c) in mask


def _perp_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r - 1, c) in mask or (r + 1, c) in mask
    return (r, c - 1) in mask or (r, c + 1) in mask


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _mask_from_pts(shape: tuple[int, int], pts: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return mask


def _degree(pts: set[tuple[int, int]], pt: tuple[int, int]) -> int:
    return len(_dirs_in(pts, pt))


def _label_from_dirs(dirs: list[tuple[int, int]]) -> str | None:
    dirs = sorted(dirs)
    if dirs == [(0, -1)]:
        return "L"
    if dirs == [(0, 1)]:
        return "R"
    if dirs == [(-1, 0)]:
        return "U"
    if dirs == [(1, 0)]:
        return "D"
    if dirs == [(0, -1), (0, 1)]:
        return "H"
    if dirs == [(-1, 0), (1, 0)]:
        return "V"
    return None


def _claim_label(pts: set[tuple[int, int]], pt: tuple[int, int]) -> str | None:
    if pt in pts:
        return _label_from_dirs(_dirs_in(pts, pt))

    r, c = pt
    dirs = [(dr, dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pts]
    label = _label_from_dirs(dirs)
    if label in {"H", "V"}:
        return label
    if label in {"L", "R", "U", "D"}:
        step = dirs[0]
        nbr = (r + step[0], c + step[1])
        nr, nc = nbr
        on_boundary = nr == 0 or nr == H - 1 or nc == 0 or nc == W - 1
        nbr_dirs = _dirs_in(pts, nbr)
        back = (-step[0], -step[1])
        if len(nbr_dirs) == 1 and nbr_dirs[0] == (-back[0], -back[1]) and not on_boundary:
            return label
    return None


def _pick_cross_color(pt: tuple[int, int], labels: dict[int, str], phase: int = 0) -> int:
    r, c = pt
    here = dict(labels)
    if len(here) == 2:
        horiz = [color for color, label in here.items() if label in HORIZONTAL_LABELS]
        vert = [color for color, label in here.items() if label in VERTICAL_LABELS]
        assert len(horiz) == 1 and len(vert) == 1, f"unsupported 2-color crossing at {pt}: {here}"
        cycle = [horiz[0], vert[0]]
        return cycle[(r + c + phase) % 2]

    assert len(here) == 3, f"unsupported crossing arity at {pt}: {here}"
    label_set = set(here.values())
    if label_set == {"L", "R", "V"}:
        cycle = [
            next(color for color, label in here.items() if label == "L"),
            next(color for color, label in here.items() if label == "V"),
            next(color for color, label in here.items() if label == "R"),
        ]
        return cycle[(r + c + phase) % 3]
    if label_set == {"U", "D", "H"}:
        cycle = [
            next(color for color, label in here.items() if label == "U"),
            next(color for color, label in here.items() if label == "H"),
            next(color for color, label in here.items() if label == "D"),
        ]
        return cycle[(r + c + phase) % 3]
    raise AssertionError(f"unsupported 3-color crossing at {pt}: {here}")


def _is_supported_combo(labels: dict[int, str]) -> bool:
    if len(labels) == 2:
        horiz = sum(label in HORIZONTAL_LABELS for label in labels.values())
        vert = sum(label in VERTICAL_LABELS for label in labels.values())
        return horiz == 1 and vert == 1
    if len(labels) == 3:
        label_set = set(labels.values())
        return label_set in ({"L", "R", "V"}, {"U", "D", "H"})
    return len(labels) == 1


def _infer_claims(
    shape: tuple[int, int],
    masks: dict[int, set[tuple[int, int]]],
    control_ray: dict[int, set[tuple[int, int]]] | None = None,
    allow_hidden: bool = True,
) -> tuple[dict[tuple[int, int], dict[int, str]], dict[int, set[tuple[int, int]]]]:
    control_ray = control_ray or {color: set() for color in masks}
    role_claims: dict[tuple[int, int], dict[int, str]] = {}
    logical_masks = {color: set(pts) for color, pts in masks.items()}

    for color, pts in masks.items():
        for r in range(shape[0]):
            for c in range(shape[1]):
                pt = (r, c)
                label = _label_from_dirs(_dirs_in(pts, pt)) if pt in pts else (_claim_label(pts, pt) if allow_hidden else None)
                if label is None:
                    continue
                role_claims.setdefault(pt, {})[color] = label
                if allow_hidden and pt not in pts and (len(_dirs_in(pts, pt)) == 2 or len(_dirs_in(pts, pt)) == 1):
                    logical_masks[color].add(pt)

    for color, pts in control_ray.items():
        for pt in pts:
            role_claims.setdefault(pt, {})
            if color not in role_claims[pt]:
                role_claims[pt][color] = _claim_label(logical_masks[color], pt) or "H"
                logical_masks[color].add(pt)
    return role_claims, logical_masks


def _render_from_masks(
    shape: tuple[int, int],
    bg: int,
    masks: dict[int, set[tuple[int, int]]],
    control_ray: dict[int, set[tuple[int, int]]] | None = None,
    phase: int = 0,
    allow_hidden: bool = True,
) -> tuple[np.ndarray, dict[tuple[int, int], dict[int, str]], dict[int, set[tuple[int, int]]]]:
    role_claims, logical_masks = _infer_claims(shape, masks, control_ray, allow_hidden=allow_hidden)
    normalized_claims: dict[tuple[int, int], dict[int, str]] = {}
    for pt, labels in role_claims.items():
        if _is_supported_combo(labels):
            normalized_claims[pt] = labels
            continue
        visible = {color: label for color, label in labels.items() if pt in masks[color]}
        if visible:
            normalized_claims[pt] = visible
        else:
            keep: dict[int, str] = {}
            if len(labels) == 1:
                keep = labels
            elif len(labels) == 2:
                horiz = {color: label for color, label in labels.items() if label in HORIZONTAL_LABELS}
                vert = {color: label for color, label in labels.items() if label in VERTICAL_LABELS}
                if len(horiz) == 1 and not vert:
                    keep = horiz
                elif len(vert) == 1 and not horiz:
                    keep = vert
            if keep:
                normalized_claims[pt] = keep
    out = np.full(shape, bg, dtype=int)
    for pt, labels in normalized_claims.items():
        forced = [color for color in labels if pt in (control_ray or {}).get(color, set())]
        if forced:
            assert len(forced) == 1, f"ambiguous yellow rays at {pt}: {forced}"
            out[pt] = forced[0]
        elif len(labels) == 1:
            out[pt] = next(iter(labels))
        else:
            out[pt] = _pick_cross_color(pt, labels, phase=phase)
    return out, normalized_claims, logical_masks


def _baseline_fragmented_color(
    pt: tuple[int, int],
    labels: dict[int, str],
    comp_count: dict[int, int],
) -> int:
    return min(labels, key=lambda color: (-comp_count[color], color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a dense one-cell-thick colored line network by recoloring singleton markers, restoring hidden crossing claims, and repainting 2-color and 3-color overlaps with periodic crossing patterns.

    Input structure:
    - The grid is larger and denser than the original task, but it still has a dominant background and 1-cell-thick rectilinear colored line fragments.
    - Red and/or yellow marker colors are exactly the foreground colors whose 4-connected components are all singleton pixels.
    - Ordinary overlaps may involve either two colors or three colors; tri-color cases use one through-line plus two opposite half-lines that terminate at the crossing cell.

    Rule:
    - Assign each singleton marker to the nearest visible line color along a row or column, preferring a sightline that already has support on that axis.
    - Recolor those markers logically, fill supported straight gaps for each color, and apply any yellow arm-flip controls before resolving overlaps.
    - At an ambiguous cell, infer every claimant from local repaired geometry, including hidden half-lines that end one step short of the displayed crossing cell.
    - Then repaint crossings by pattern rather than winner-take-all: 2-color crossings alternate horizontal-vs-vertical by checkerboard parity, while supported 3-color motifs cycle through their side/through/side roles by diagonal phase.

    Key insight:
    - The repair stage still reconstructs the underlying sparse rectilinear network, but crossings are no longer single-color winners. You must first infer which colors truly meet at each overlap cell, then apply the visible parity/cycle pattern dictated by each crossing's local geometric roles.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    shape = g.shape
    H0, W0 = shape

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"

    marker_colors: set[int] = set()
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        assert comps, f"foreground color {color} must have a component"
        if max(len(comp) for comp in comps) == 1:
            marker_colors.add(int(color))

    line_colors = colors - marker_colors
    assert line_colors, "expected at least one line color"

    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[int, int, int, int, int]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < H0 and 0 <= c < W0:
                value = int(g[r, c])
                if value in line_colors:
                    axis = 0 if dr == 0 else 1
                    supported = _axis_support(original[value], r, c, axis)
                    candidates.append((dist, 0 if supported else 1, value, dr, dc))
                    break
                if value == bg or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break
        assert candidates, f"marker at {(r0, c0)} cannot see a line color"
        candidates.sort()
        _dist, _unsupported, color, dr, dc = candidates[0]
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return (r, c) in marker_pts[color] or _axis_support(masks[color], r, c, axis)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        r1, c1 = a
        r2, c2 = b
        gap = c2 - c1 - 1 if axis == 0 else r2 - r1 - 1
        if gap <= 0:
            return False

        a_axis = has_axis_support(color, r1, c1, axis)
        b_axis = has_axis_support(color, r2, c2, axis)
        if gap == 1:
            if a_axis or b_axis:
                return True
            a_perp = _perp_support(masks[color], r1, c1, axis)
            b_perp = _perp_support(masks[color], r2, c2, axis)
            return not (a_perp and b_perp)

        if (r1, c1) in marker_pts[color] and (r2, c2) in marker_pts[color]:
            return False
        return a_axis and b_axis

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(H0):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill(color, (r, c1), (r, c2), axis=0):
                    additions.update((r, c) for c in range(c1 + 1, c2))
        for c in range(W0):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if should_fill(color, (r1, c), (r2, c), axis=1):
                    additions.update((r, c) for r in range(r1 + 1, r2))
        masks[color].update(additions)

    control_ray = {color: set() for color in line_colors}
    if YELLOW in marker_colors:
        for control in _pts(g == YELLOW):
            color = assignment[control]
            dr, dc = assign_dir[control]

            r, c = control[0] + dr, control[1] + dc
            pivot: tuple[int, int] | None = None
            while 0 <= r < H0 and 0 <= c < W0:
                if (r, c) in masks[color]:
                    pivot = (r, c)
                    break
                r += dr
                c += dc
            assert pivot is not None, f"yellow control at {control} has no assigned endpoint"

            old_start = (pivot[0] + dr, pivot[1] + dc)
            if old_start in masks[color]:
                stack = [old_start]
                old_arm: set[tuple[int, int]] = set()
                while stack:
                    q = stack.pop()
                    if q == pivot or q in old_arm or q not in masks[color]:
                        continue
                    old_arm.add(q)
                    qr, qc = q
                    for ndr, ndc in DIRS4:
                        stack.append((qr + ndr, qc + ndc))
                for q in old_arm:
                    masks[color].discard(q)
                    marker_pts[color].discard(q)
                    control_ray[color].discard(q)

            pr, pc = -dr, -dc
            r, c = pivot
            while 0 <= r < H0 and 0 <= c < W0:
                masks[color].add((r, c))
                control_ray[color].add((r, c))
                r += pr
                c += pc

    out, role_claims, logical_masks = _render_from_masks(shape, bg, masks, control_ray, phase=0)

    crossing_colors = set()
    for pt, labels in role_claims.items():
        if len(labels) > 1:
            crossing_colors.update(labels)
            chosen = _pick_cross_color(pt, labels, phase=0) if not any(pt in control_ray[c] for c in labels) else out[pt]
            assert int(out[pt]) == int(chosen)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing in repaired output"

    # Variant-specific endpoint check: interior endpoints are only allowed if they terminate at a crossing.
    crossing_pts = {pt for pt, labels in role_claims.items() if len(labels) > 1}
    for color, pts in logical_masks.items():
        mask = np.zeros(shape, dtype=bool)
        for r, c in pts:
            mask[r, c] = True
        for comp in components_mask(mask, connectivity=4):
            comp_pts = set(map(tuple, comp))
            for r, c in comp_pts:
                nbrs = [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in comp_pts]
                assert len(nbrs) <= 2, f"junction at {(r, c)} for color {color}"
                if len(nbrs) != 1:
                    continue
                if (r, c) in crossing_pts:
                    continue
                on_boundary = (r == 0 or r == H0 - 1 or c == 0 or c == W0 - 1)
                assert on_boundary, f"interior endpoint {(r, c)} for color {color} is not a crossing"

    return out


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[tuple[int, int], dict[int, str]]] | None:
    colors = random.sample(LINE_PALETTE, 6)
    v_color, l_color, r_color, h_color, u_color, d_color = colors

    rows = sorted(random.sample(range(4, H - 4), 2))
    cols = sorted(random.sample(range(4, W - 4), 2))
    if random.random() < 0.5:
        r0, r1 = rows
    else:
        r1, r0 = rows
    if random.random() < 0.5:
        c0, c1 = cols
    else:
        c1, c0 = cols
    if r0 == r1 or c0 == c1:
        return None

    logical = {
        v_color: set(_path([(0, c0), (H - 1, c0)])),
        l_color: set(_path([(r0, 0), (r0, c0)])),
        r_color: set(_path([(r0, c0), (r0, W - 1)])),
        h_color: set(_path([(r1, 0), (r1, W - 1)])),
        u_color: set(_path([(0, c1), (r1, c1)])),
        d_color: set(_path([(r1, c1), (H - 1, c1)])),
    }

    out, role_claims, _logical_masks = _render_from_masks((H, W), BG, logical, phase=0, allow_hidden=False)
    crossings = {pt: labels for pt, labels in role_claims.items() if len(labels) > 1}
    if not any(len(labels) == 3 and set(labels.values()) == {"L", "R", "V"} for labels in crossings.values()):
        return None
    if not any(len(labels) == 3 and set(labels.values()) == {"U", "D", "H"} for labels in crossings.values()):
        return None
    if not any(len(labels) == 2 for labels in crossings.values()):
        return None
    if np.count_nonzero(out) < 60:
        return None
    return logical, crossings


def _candidate_red_positions(
    logical: dict[int, set[tuple[int, int]]],
    crossings: dict[tuple[int, int], dict[int, str]],
) -> list[tuple[int, tuple[int, int]]]:
    blocked = set(crossings)
    for pt in list(crossings):
        blocked.update(_neighbors((H, W), pt))
    candidates: list[tuple[int, tuple[int, int]]] = []
    for color, pts in logical.items():
        for pt in pts:
            if pt in blocked:
                continue
            r, c = pt
            if r in (0, H - 1) or c in (0, W - 1):
                continue
            if _degree(pts, pt) == 2:
                candidates.append((color, pt))
    random.shuffle(candidates)
    return candidates


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    crossings: dict[tuple[int, int], dict[int, str]],
) -> list[tuple[int, tuple[int, int]]]:
    count = random.randint(*RED_COUNT_RANGE)
    chosen: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()
    candidates = _candidate_red_positions(logical, crossings)
    per_color = {color: 0 for color in logical}
    for color, pt in candidates:
        if len(chosen) == count:
            break
        if per_color[color] >= 1 and random.random() < 0.6:
            continue
        neighborhood = {pt, *_neighbors((H, W), pt)}
        if used & neighborhood:
            continue
        trial = set(logical[color])
        trial.discard(pt)
        if _component_count((H, W), trial) > 2:
            continue
        comp_sizes = [len(comp) for comp in components_mask(_mask_from_pts((H, W), trial), connectivity=4)]
        if not comp_sizes or max(comp_sizes) <= 1:
            continue
        chosen.append((color, pt))
        used.update(neighborhood)
        per_color[color] += 1
    return chosen if len(chosen) >= 3 else []


def _apply_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    red_markers: list[tuple[int, tuple[int, int]]],
) -> dict[int, set[tuple[int, int]]]:
    visible = {color: set(pts) for color, pts in logical.items()}
    for color, pt in red_markers:
        visible[color].discard(pt)
    return visible


def _render_input_and_target(
    logical: dict[int, set[tuple[int, int]]],
    crossings: dict[tuple[int, int], dict[int, str]],
    red_markers: list[tuple[int, tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    visible = _apply_red_markers(logical, red_markers)
    target, target_claims, _ = _render_from_masks((H, W), BG, logical, phase=0, allow_hidden=False)
    inp_cross, inp_claims, _ = _render_from_masks((H, W), BG, logical, phase=1, allow_hidden=False)

    inp = np.full((H, W), BG, dtype=int)
    red_by_pt = {pt: color for color, pt in red_markers}
    all_pts = set().union(*logical.values())
    for pt in all_pts:
        if pt in red_by_pt:
            inp[pt] = RED
            continue
        if pt in crossings:
            inp[pt] = inp_cross[pt]
            continue
        for color, pts in visible.items():
            if pt in pts:
                inp[pt] = color
                break

    if any(inp[nb] == RED for pt in red_by_pt for nb in _neighbors((H, W), pt)):
        return None

    visible_plus_markers = {color: set(pts) for color, pts in visible.items()}
    for color, pt in red_markers:
        visible_plus_markers[color].add(pt)
    comp_count = {color: _component_count((H, W), pts) for color, pts in visible_plus_markers.items()}

    changed = 0
    tri_count = 0
    pair_count = 0
    baseline_diff = 0
    for pt, labels in target_claims.items():
        if len(labels) == 1:
            continue
        if len(labels) == 2:
            pair_count += 1
        elif len(labels) == 3:
            tri_count += 1
        if int(inp[pt]) != int(target[pt]):
            changed += 1
        if target[pt] != _baseline_fragmented_color(pt, labels, comp_count):
            baseline_diff += 1

    if tri_count < 2 or pair_count < 2 or changed < 3 or baseline_diff < 1:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, target):
        return None

    metrics = {
        "crossings_changed": changed,
        "triple_crossings": tri_count,
        "pair_crossings": pair_count,
        "baseline_diff": baseline_diff,
        "line_colors": len(logical),
    }
    return inp, target, metrics


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    built = _build_logical()
    if built is None:
        return None
    logical, crossings = built
    red_markers = _choose_red_markers(logical, crossings)
    if not red_markers:
        return None
    rendered = _render_input_and_target(logical, crossings, red_markers)
    if rendered is None:
        return None
    inp, _target, metrics = rendered
    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid fused faa9f03d input")


def generate_input_with_metrics() -> tuple[np.ndarray, dict[str, int]]:
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt
    raise RuntimeError("failed to generate a valid fused faa9f03d input")
