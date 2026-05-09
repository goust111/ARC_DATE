# Combo variant of 21897d95 (index 4)
# combo_spec: D3/1+D5/1
# Name: 十字尺寸编码

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the fused D3/1 + D5/1 variant
# ---------------------------------------------------
# a) Concrete behavior and trace
#    - Original task:
#      * normal marker = 4-cell all-1 T; the protruding arm itself reveals the stem.
#      * colored marker = colored hub + 3 singleton 1s; hub color is the replacement.
#    - Fused task:
#      * normal marker = 5-cell full cross '+': center + the 4 cardinal arms, all color 1.
#      * colored marker = 7-cell expanded cross with a colored hub and 6 color-1 cells.
#        Two expansion styles are supported:
#          1. "wide": the two bar arms extend by one more cell each.
#          2. "long": the stem extends by two more cells.
#      * The source/destination reading still uses the old T logic, but now the stem is no
#        longer read from a protruding tetromino branch. Instead the solver tests each cardinal
#        direction as a possible stem and looks for the unique opposite arm that never touches
#        the source color locally. That arm is the contextual "gap side", and the opposite side
#        is the true stem.
#      * Example with an intended stem = RIGHT:
#          old T: the right branch sticks out, so stem = RIGHT immediately.
#          fused cross: the marker shows a full '+' footprint, so shape alone does not choose
#          RIGHT. The source diagonals at (center + RIGHT + UP/DOWN) still identify the source.
#          The west arm is then forced to be the only arm that has no source contact, so the
#          solver infers gap = LEFT and therefore stem = RIGHT.
#      * Behavior type is no longer distinguished by "does this marker have a colored center?".
#        The decisive cue is size:
#          5 cells -> copy source color into destination.
#          7 cells -> use the center hub color as the destination color.
#
# b) Complexity calibration
#    - The two changes are both visually obvious and tightly coupled:
#      * every marker is cross-shaped rather than T-shaped,
#      * colored behavior only appears on the larger 7-cell version.
#    - The generator forces at least one expanded cross per grid so the size code is observable.
#    - Colored markers use one expansion dialect per grid ("wide" or "long"), avoiding a noisy
#      mix of unrelated large glyphs while still preserving the size distinction.
#    - The contextual gap rule is enforced by construction, so every cross has one unique valid
#      orientation instead of multiple equally plausible readings.
#
# c) Hidden assumptions that break from the original solver
#    - Parsing 4-cell T tetrominoes no longer applies; 5-cell all-1 components must be parsed as
#      symmetric crosses with no built-in direction.
#    - The old colored-marker parser expected three singleton 1-neighbors around a hub. That fails
#      here because the expanded cross has six 1-cells, some of which form 2-cell or 3-cell chains.
#    - The old solver always started the destination walk at step 2. That remains correct for the
#      5-cell cross and the wide 7-cell cross, but it is wrong for the long 7-cell cross, which
#      occupies stem steps 1, 2, and 3, so the walk must start at step 4.
#    - The old orientation inference depended only on marker geometry. Here orientation depends on
#      post-control base components, so parsing and semantic interpretation must be decoupled.
#
# d) Placement legality
#    - The generator checks full marker footprints, not just the base '+' core, against bounds,
#      other markers, and a one-cell buffer neighborhood.
#    - It also rewrites a tiny amount of local region-id context around each marker so the gap side
#      is unique: the intended source diagonals become source-region cells, while the opposite
#      diagonals become two different non-source region ids.
#    - This guarantees that the solver's contextual orientation test matches the generated stem.
#
# e) Visual observability
#    - With only a few examples, a human can notice two repeated regularities:
#      * all controls are plus-like, not T-like,
#      * some controls are strictly larger, and those are exactly the ones whose center color
#        matters for the repaint.
#    - The orientation cue is structural and local: look for the one side of the '+' that lacks
#      source-side support, then take the opposite side as the stem.
#
# f) Visual cleanliness
#    - A marker buffer keeps different crosses and the optional L-rotation cue separated.
#    - Expanded crosses stay compact at 7 cells, so even the long form is still easy to read.
#    - The generator uses a moderate number of regions and markers to avoid overcrowding.
#
# g) Inference of the color change
#    - Region colors remain distinct and few. The only nontrivial color payload is the colored hub
#      on expanded crosses, and its role is accompanied by the repeated geometric cue of size.
#    - Normal and colored markers are both present, so a solver must infer that size selects the
#      repaint rule while the contextual gap selects the direction.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
BIG_CROSS_VARIANTS = ("wide", "long")
BIG_CROSS_DEST_START = {"wide": 2, "long": 4}


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(int(n)))

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-int(d[0]), -int(d[1]))


def _add(cell: tuple[int, int], direction: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return (int(cell[0] + direction[0] * scale), int(cell[1] + direction[1] * scale))


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if int(stem[0]) == 0:
        return [UP, DOWN]
    return [LEFT, RIGHT]


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    return {center} | {_add(center, d) for d in DIRS}


def _cross_arm_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    return {_add(center, d) for d in DIRS}


def _diag_cells(center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    return [_add(_add(center, direction), bar) for bar in _get_bar(direction)]


def _gap_cells(center: tuple[int, int], stem: tuple[int, int]) -> list[tuple[int, int]]:
    return _diag_cells(center, _opp(stem))


def _big_cross_one_cells(
    center: tuple[int, int],
    stem: tuple[int, int],
    variant: str,
) -> set[tuple[int, int]]:
    cells = _cross_arm_cells(center)
    bar = _get_bar(stem)
    if variant == "wide":
        cells.add(_add(center, bar[0], 2))
        cells.add(_add(center, bar[1], 2))
        return cells
    if variant == "long":
        cells.add(_add(center, stem, 2))
        cells.add(_add(center, stem, 3))
        return cells
    raise ValueError(f"unknown expanded-cross variant {variant}")


def _marker_cells(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    variant: str | None = None,
) -> set[tuple[int, int]]:
    if kind == "normal":
        return _cross_cells(center)
    assert variant is not None
    return {center} | _big_cross_one_cells(center, stem, variant)


def _marker_one_cells(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    variant: str | None = None,
) -> set[tuple[int, int]]:
    cells = _marker_cells(kind, center, stem, variant)
    cells.remove(center)
    return cells


def _marker_neighborhood(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    variant: str | None = None,
) -> set[tuple[int, int]]:
    cells = _marker_cells(kind, center, stem, variant)
    neighborhood: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (int(r + dr), int(c + dc))
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _parse_cross(cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    if len(cells) != 5:
        return None
    for r, c in cells:
        center = (int(r), int(c))
        if cells == _cross_cells(center):
            return center
    return None


def _label_base_components(g: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
    comp_areas: list[int] = []
    cid = 0
    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == col) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == k)
            if coords.size == 0:
                continue
            for rr, cc in coords:
                comp_id[int(rr), int(cc)] = cid
            comp_colors.append(int(col))
            comp_areas.append(int(len(coords)))
            cid += 1
    return comp_id, comp_colors, comp_areas


def _supported_diag_parts(
    center: tuple[int, int],
    direction: tuple[int, int],
    comp_id: np.ndarray,
    comp_colors: list[int],
) -> list[int] | None:
    parts: list[int] = []
    for rr, cc in _diag_cells(center, direction):
        if not _in_bounds(comp_id.shape, rr, cc):
            return None
        cid = int(comp_id[rr, cc])
        if cid < 0:
            return None
        parts.append(cid)
    if comp_colors[parts[0]] != comp_colors[parts[1]]:
        return None
    return parts


def _arm_touches_color(
    center: tuple[int, int],
    direction: tuple[int, int],
    comp_id: np.ndarray,
    comp_colors: list[int],
    color: int,
) -> bool:
    ar, ac = _add(center, direction)
    for dr, dc in DIRS:
        rr, cc = ar + dr, ac + dc
        if (rr, cc) == center or not _in_bounds(comp_id.shape, rr, cc):
            continue
        cid = int(comp_id[rr, cc])
        if cid >= 0 and comp_colors[cid] == int(color):
            return True
    return False


def _infer_cross_orientation(
    center: tuple[int, int],
    comp_id: np.ndarray,
    comp_colors: list[int],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[int]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int], list[int]]] = []
    for stem in DIRS:
        source_parts = _supported_diag_parts(center, stem, comp_id, comp_colors)
        if source_parts is None:
            continue
        source_color = int(comp_colors[source_parts[0]])
        gap = _opp(stem)
        if _arm_touches_color(center, gap, comp_id, comp_colors, source_color):
            continue
        if not all(
            _arm_touches_color(center, direction, comp_id, comp_colors, source_color)
            for direction in DIRS
            if direction != gap
        ):
            continue
        candidates.append((gap, stem, source_parts))

    assert len(candidates) == 1, "cross marker must have one unique gap side"
    gap, stem, source_parts = candidates[0]
    return gap, stem, _get_bar(stem), source_parts


def _parse_big_cross_at(
    g: np.ndarray,
    center: tuple[int, int],
    used_ones: set[tuple[int, int]],
) -> dict | None:
    if int(g[center]) == 1:
        return None

    arms = _cross_arm_cells(center)
    if not all(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1 and (rr, cc) not in used_ones for rr, cc in arms):
        return None

    candidates: list[dict] = []

    wide_specs = [
        (LEFT, RIGHT, {UP, DOWN}),
        (UP, DOWN, {LEFT, RIGHT}),
    ]
    for d0, d1, allowed_stems in wide_specs:
        ext0 = _add(center, d0, 2)
        ext1 = _add(center, d1, 2)
        if not all(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1 and (rr, cc) not in used_ones for rr, cc in (ext0, ext1)):
            continue
        forbidden = [
            _add(center, s, 2)
            for s in DIRS
            if s not in {d0, d1}
        ] + [_add(center, d0, 3), _add(center, d1, 3)]
        if any(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1 for rr, cc in forbidden):
            continue
        one_cells = arms | {ext0, ext1}
        candidates.append(
            {
                "variant": "wide",
                "one_cells": one_cells,
                "cells": one_cells | {center},
                "dest_start": int(BIG_CROSS_DEST_START["wide"]),
                "allowed_stems": {tuple(map(int, d)) for d in allowed_stems},
            }
        )

    for stem in DIRS:
        ext2 = _add(center, stem, 2)
        ext3 = _add(center, stem, 3)
        if not all(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1 and (rr, cc) not in used_ones for rr, cc in (ext2, ext3)):
            continue
        bar = _get_bar(stem)
        forbidden = [_add(center, _opp(stem), 2), _add(center, bar[0], 2), _add(center, bar[1], 2)]
        if any(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1 for rr, cc in forbidden):
            continue
        one_cells = arms | {ext2, ext3}
        candidates.append(
            {
                "variant": "long",
                "one_cells": one_cells,
                "cells": one_cells | {center},
                "dest_start": int(BIG_CROSS_DEST_START["long"]),
                "stem_hint": tuple(map(int, stem)),
            }
        )

    unique: dict[frozenset[tuple[int, int]], dict] = {}
    for candidate in candidates:
        unique[frozenset(candidate["one_cells"])] = candidate
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        k: {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == k)}
        for k in range(1, int(n_1) + 1)
    }

    markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)

    for cells in one_components.values():
        center = _parse_cross(cells)
        if center is None:
            continue
        markers.append({"kind": "normal", "center": center, "cells": cells, "dest_start": 2})
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            center = (int(r), int(c))
            parsed = _parse_big_cross_at(g, center, used_ones)
            if parsed is None:
                continue
            markers.append(
                {
                    "kind": "colored",
                    "center": center,
                    "variant": parsed["variant"],
                    "cells": parsed["cells"],
                    "dest_start": parsed["dest_start"],
                    "target_color": int(g[center]),
                    "allowed_stems": parsed.get("allowed_stems"),
                    "stem_hint": parsed.get("stem_hint"),
                }
            )
            used_ones |= parsed["one_cells"]
            for rr, cc in parsed["cells"]:
                control[rr, cc] = True

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a cross control marker"

    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == col) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == k)
            if len(coords) > 5:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if len(coords) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
                continue
            cells = {tuple(map(int, xy)) for xy in coords}
            markers.append({"kind": "small", "cells": cells})
            for rr, cc in cells:
                control[rr, cc] = True

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"
    dsu = _DSU(len(comp_colors))

    def first_component_from(center: tuple[int, int], direction: tuple[int, int], start: int) -> int | None:
        step = int(start)
        while True:
            rr, cc = _add(center, direction, step)
            if not _in_bounds(g.shape, rr, cc):
                return None
            if comp_id[rr, cc] >= 0:
                return int(comp_id[rr, cc])
            step += 1

    def adjacent_component_counts(cells: set[tuple[int, int]]) -> Counter:
        counts: Counter = Counter()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0:
                    counts[int(comp_id[rr, cc])] += 1
        return counts

    for marker in markers:
        if marker["kind"] in {"normal", "colored"}:
            center = marker["center"]
            gap, stem, bar, source_parts = _infer_cross_orientation(center, comp_id, comp_colors)
            if marker["kind"] == "colored":
                allowed_stems = marker.get("allowed_stems")
                stem_hint = marker.get("stem_hint")
                if allowed_stems is not None:
                    assert tuple(map(int, stem)) in allowed_stems, "wide expanded cross inferred impossible stem"
                if stem_hint is not None:
                    assert tuple(map(int, stem)) == tuple(map(int, stem_hint)), "long expanded cross inferred wrong stem"
            dest = first_component_from(center, stem, start=int(marker["dest_start"]))
            assert dest is not None, "cross marker must point at a destination region"

            source = int(source_parts[0])
            for part in source_parts[1:]:
                assert comp_colors[part] == comp_colors[source], "source diagonals must share a color"
                dsu.union(source, part)

            for comp in adjacent_component_counts(marker["cells"]):
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
            marker["gap"] = gap
            marker["stem"] = stem
            marker["bar"] = bar
            marker["dest_comp"] = dest
            marker["source_comp"] = source
        else:
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "small marker must touch its owner region"
            owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
            for comp in counts:
                if comp_colors[comp] == comp_colors[owner]:
                    dsu.union(owner, comp)
            marker["source_comp"] = owner

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    final_color = dict(root_color)
    for marker in markers:
        if marker["kind"] not in {"normal", "colored"}:
            marker["source"] = dsu.find(marker["source_comp"])
            continue
        dest = dsu.find(marker["dest_comp"])
        source = dsu.find(marker["source_comp"])
        marker["source"] = source
        replacement = int(marker["target_color"]) if marker["kind"] == "colored" else int(root_color[source])
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting cross controls for one region"
        final_color[dest] = replacement

    out = np.full(g.shape, -1, dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[dsu.find(cid)])

    for marker in markers:
        color = int(final_color[marker["source"]])
        for r, c in marker["cells"]:
            out[r, c] = color

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def _find_l9_orientation(g: np.ndarray) -> str | None:
    """Find a 5-cell L-shaped color-9 marker and return its corner position."""
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    for k in range(1, int(n) + 1):
        coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == k)}
        if len(coords) != 5:
            continue
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        for cr, cc in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
            h_nb = (cr, cc + 1 if cc == c0 else cc - 1)
            v_nb = (cr + 1 if cr == r0 else cr - 1, cc)
            if h_nb in coords and v_nb in coords:
                if cr == r0 and cc == c1:
                    return "top_right"
                if cr == r0 and cc == c0:
                    return "top_left"
                assert False, f"unobserved L9 corner at ({cr},{cc})"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint regions from size-coded cross controls, erase the controls, and optionally rotate from a color-9 L cue.

    Input structure:
    - The board is partitioned into irregular monochrome regions.
    - Every recoloring control is cross-shaped rather than T-shaped.
    - A normal control is a 5-cell all-1 '+'.
    - A colored control is a 7-cell expanded cross: the hub is a literal non-1 color and the six remaining cells are color 1.
    - An optional 5-cell L marker in color 9 requests a 90-degree rotation after recoloring.

    Rule:
    - For any cross, test each side as a possible stem by looking at the two diagonal cells one step beyond it.
    - The true source side is the one whose diagonals belong to the same region color while the opposite arm is the unique arm that lacks any local contact with that source; that opposite arm is the contextual gap, so its opposite side is the stem.
    - The stem points toward the destination region.
    - Size chooses the repaint mode:
      5-cell cross -> destination becomes the source region's original color.
      7-cell expanded cross -> destination becomes the hub cell's literal color.
    - Every marker cell is finally filled with the source-side region's final color.
    - The color-9 L marker still rotates the finished board: top-right corner means CCW 90 degrees, top-left means CW 90 degrees.

    Key insight:
    - You must read two independent structural cues at once: the symmetric '+' needs contextual gap inference to recover direction, and the total marker size decides whether to copy the source color or use the hub color.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (l9rot-cross-sizecoded) ---------------

# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [7, 8]
REGION_WEIGHTS = [1, 3]

# Marker counts. At least one expanded cross is required so the size code is visible.
MARKER_MIN = 4
MARKER_MAX = 4
NORMAL_MAX = 3
COLORED_MIN = 1
COLORED_MAX = 1

# Color pool: exclude 1 for marker cells and 9 for the L rotation cue.
COLOR_POOL = [c for c in range(10) if c not in {1, 9}]

# L9 rotation marker
L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

# Generation retry budget
SEED_ATTEMPTS = 200
GEN_RETRIES = 300
SEARCH_PAIR_LIMIT = 6
SEARCH_PLACEMENT_LIMIT = 8
PATCH_SIZE = 11
PATCH_CENTER = 5
BOARD_SIZE = 27
PATCH_ANCHORS = [(1, 1), (1, 15), (15, 1), (15, 15)]

_CANON_SOURCE = {
    (dr, dc)
    for dr in range(-5, 0)
    for dc in range(-5, 6)
    if not (dr <= -2 and abs(dc) <= 1)
}
_CANON_DEST = {(dr, dc) for dr in range(-5, -1) for dc in range(-1, 2)}
_CANON_GAP_LEFT = {(dr, dc) for dr in range(1, 6) for dc in range(-5, 0)}
_CANON_GAP_RIGHT = {(dr, dc) for dr in range(1, 6) for dc in range(1, 6)}


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    """Grow n_regions from random seeds via randomized BFS; returns region-id array or None."""
    region = np.full((H, W), -1, dtype=int)
    seeds = set()
    for _ in range(SEED_ATTEMPTS):
        if len(seeds) >= n_regions:
            break
        seeds.add((random.randint(1, H - 2), random.randint(1, W - 2)))
    if len(seeds) < n_regions:
        return None

    frontier = []
    for i, (r, c) in enumerate(list(seeds)):
        region[r, c] = i
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and region[nr, nc] == -1:
                frontier.append((nr, nc, i))
    random.shuffle(frontier)

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if region[r, c] != -1:
            continue
        region[r, c] = rid
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and region[nr, nc] == -1:
                frontier.append((nr, nc, rid))

    for rid in range(n_regions):
        mask = region == rid
        if mask.sum() < MIN_REGION_CELLS:
            return None
        _, n = label_mask(mask, connectivity=4)
        if n != 1:
            return None
    return region


def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    """Return all directed (src, dst) pairs between adjacent regions."""
    H, W = region.shape
    undirected = set()
    for r in range(H):
        for c in range(W):
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    a, b = int(region[r, c]), int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _choose_marker_counts(n_regions: int) -> tuple[int, int, int] | None:
    """Return (n_markers, n_normal, n_colored) or None if infeasible."""
    max_possible = min(MARKER_MAX, n_regions)
    if max_possible < MARKER_MIN:
        return None
    n_markers = MARKER_MIN
    n_colored = COLORED_MIN
    n_normal = n_markers - n_colored
    if n_normal > NORMAL_MAX:
        n_colored = n_markers - NORMAL_MAX
        n_normal = NORMAL_MAX
        if n_colored < COLORED_MIN or n_colored > COLORED_MAX:
            return None
    return n_markers, n_normal, n_colored


def _marker_destination_start(kind: str, variant: str | None) -> int:
    if kind == "normal":
        return 2
    assert variant is not None
    return int(BIG_CROSS_DEST_START[variant])


def _has_supported_pair(
    region: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
    src: int,
) -> bool:
    coords = _diag_cells(center, direction)
    return all(
        _in_bounds(region.shape, rr, cc) and int(region[rr, cc]) == int(src)
        for rr, cc in coords
    )


def _gap_replacements(
    region: np.ndarray,
    center: tuple[int, int],
    stem: tuple[int, int],
    src: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    cells = _gap_cells(center, stem)
    cross = _cross_cells(center)
    options: list[list[int]] = []
    for r, c in cells:
        ids: set[int] = set()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(region.shape, nr, nc) or (nr, nc) in cross:
                continue
            rid = int(region[nr, nc])
            if rid != int(src):
                ids.add(rid)
        current = int(region[r, c])
        if current != int(src):
            ids.add(current)
        if not ids:
            return None
        options.append(sorted(ids))

    for left in options[0]:
        for right in options[1]:
            if left != right:
                return cells, (int(left), int(right))
    return None


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    kind: str,
    variant: str | None,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[list[tuple[int, int]], tuple[int, int]]]]:
    """Find valid (center, stem, forced-context) placements for one marker."""
    candidates = []
    for r in range(H):
        for c in range(W):
            center = (int(r), int(c))
            for stem in DIRS:
                cells = _marker_cells(kind, center, stem, variant)
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                if cells & occupied_cells:
                    continue
                if _marker_neighborhood(kind, center, stem, variant) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue
                if not _has_supported_pair(region, center, stem, src):
                    continue

                step = _marker_destination_start(kind, variant)
                found = False
                while True:
                    nr, nc = _add(center, stem, step)
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied_cells or (nr, nc) in cells:
                        step += 1
                        continue
                    found = int(region[nr, nc]) == int(dst)
                    break
                if not found:
                    continue

                forced = _gap_replacements(region, center, stem, src)
                if forced is not None:
                    candidates.append((center, tuple(map(int, stem)), forced))
    return candidates


def _collect_feasible_defs(
    region: np.ndarray,
    kind: str,
    variant: str | None,
    used_dests: set[int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[dict]:
    feasible: list[dict] = []
    for src, dst in _find_adjacent_pairs(region):
        if int(dst) in used_dests:
            continue
        placements = _find_valid_placements(
            region,
            int(src),
            int(dst),
            kind,
            variant,
            occupied_cells,
            occupied_neighbors,
            H,
            W,
        )
        if placements:
            feasible.append(
                {
                    "src": int(src),
                    "dst": int(dst),
                    "placements": placements,
                    "score": int(len(placements)),
                }
            )
    return feasible


def _search_marker_layout(
    region: np.ndarray,
    marker_plan: list[str],
    input_colors: list[int],
    pool: list[int],
    H: int,
    W: int,
) -> list[dict] | None:
    def dfs(
        index: int,
        occupied_cells: set[tuple[int, int]],
        occupied_neighbors: set[tuple[int, int]],
        used_dests: set[int],
    ) -> list[dict] | None:
        if index >= len(marker_plan):
            return []

        marker_kind = marker_plan[index]
        variants = list(BIG_CROSS_VARIANTS) if marker_kind == "colored" else [None]
        random.shuffle(variants)

        options: list[tuple[str | None, dict, list[tuple[tuple[int, int], tuple[int, int], tuple[list[tuple[int, int]], tuple[int, int]]]]]] = []
        for variant in variants:
            feasible = _collect_feasible_defs(
                region,
                marker_kind,
                variant,
                used_dests,
                occupied_cells,
                occupied_neighbors,
                H,
                W,
            )
            random.shuffle(feasible)
            feasible.sort(key=lambda item: (item["score"], random.random()))
            for item in feasible[:SEARCH_PAIR_LIMIT]:
                placements = item["placements"][:]
                random.shuffle(placements)
                options.append((variant, item, placements[:SEARCH_PLACEMENT_LIMIT]))
            if marker_kind == "colored" and options:
                break

        for variant, item, placements in options:
            target_color = None
            if marker_kind == "colored":
                cands = [c for c in pool if c not in {input_colors[item["src"]], input_colors[item["dst"]]}]
                if not cands:
                    continue
                target_color = int(random.choice(cands))
            for center, stem, forced in placements:
                placed = {
                    "kind": marker_kind,
                    "src": int(item["src"]),
                    "dst": int(item["dst"]),
                    "variant": variant,
                    "target_color": target_color,
                    "center": center,
                    "stem": stem,
                    "forced_context": forced,
                }
                next_occupied_cells = set(occupied_cells)
                next_occupied_neighbors = set(occupied_neighbors)
                _register_marker(placed, next_occupied_cells, next_occupied_neighbors)
                tail = dfs(
                    index + 1,
                    next_occupied_cells,
                    next_occupied_neighbors,
                    used_dests | {int(item["dst"])},
                )
                if tail is not None:
                    return [placed] + tail
        return None

    return dfs(0, set(), set(), set())


def _register_marker(
    marker: dict,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
    """Update occupied_cells and occupied_neighbors after placing a marker."""
    cells = _marker_cells(marker["kind"], marker["center"], marker["stem"], marker.get("variant"))
    occupied_cells |= cells
    occupied_neighbors |= (
        _marker_neighborhood(marker["kind"], marker["center"], marker["stem"], marker.get("variant")) - cells
    )


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    """Map region ids to colors via vectorized lookup."""
    lut = np.array(colors, dtype=int)
    return lut[region]


def _apply_cross_context(
    region: np.ndarray,
    center: tuple[int, int],
    stem: tuple[int, int],
    src: int,
    forced: tuple[list[tuple[int, int]], tuple[int, int]],
) -> None:
    for r, c in _diag_cells(center, stem):
        region[r, c] = int(src)
    cells, values = forced
    for (r, c), rid in zip(cells, values):
        region[r, c] = int(rid)


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[dict]) -> None:
    """Write cross markers onto the input grid."""
    for marker in placed_markers:
        if marker["kind"] == "normal":
            for cr, cc in _marker_cells(marker["kind"], marker["center"], marker["stem"], marker.get("variant")):
                grid[cr, cc] = 1
        else:
            grid[marker["center"]] = int(marker["target_color"])
            for cr, cc in _marker_one_cells(marker["kind"], marker["center"], marker["stem"], marker.get("variant")):
                grid[cr, cc] = 1


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[dict], out_colors: list[int]) -> None:
    """Fill all marker cells in the output grid with the source region's final color."""
    for marker in placed_markers:
        color = int(out_colors[marker["src"]])
        for cr, cc in _marker_cells(marker["kind"], marker["center"], marker["stem"], marker.get("variant")):
            grid[cr, cc] = color


def _rotate_offset(offset: tuple[int, int], stem: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, offset)
    if stem == UP:
        return dr, dc
    if stem == RIGHT:
        return dc, -dr
    if stem == DOWN:
        return -dr, -dc
    if stem == LEFT:
        return -dc, dr
    raise ValueError(f"unknown stem direction {stem}")


def _sample_distinct_colors(n: int, banned: set[int]) -> list[int]:
    pool = [c for c in COLOR_POOL if c not in banned]
    random.shuffle(pool)
    if len(pool) >= n:
        return pool[:n]
    return [random.choice(pool) for _ in range(n)]


def _paint_patch(
    grid: np.ndarray,
    anchor: tuple[int, int],
    stem: tuple[int, int],
    marker: dict,
    bg_color: int,
) -> None:
    ar, ac = anchor
    for r in range(ar, ar + PATCH_SIZE):
        for c in range(ac, ac + PATCH_SIZE):
            grid[r, c] = int(bg_color)

    region_sets = [
        (_CANON_SOURCE, int(marker["source_color"])),
        (_CANON_DEST, int(marker["dest_color"])),
        (_CANON_GAP_LEFT, int(marker["gap_left_color"])),
        (_CANON_GAP_RIGHT, int(marker["gap_right_color"])),
    ]
    for offsets, color in region_sets:
        for offset in offsets:
            dr, dc = _rotate_offset(offset, stem)
            grid[ar + PATCH_CENTER + dr, ac + PATCH_CENTER + dc] = color

    _stamp_markers_input(grid, [marker])


def _place_l9_direct(grid: np.ndarray) -> None:
    if random.random() >= L9_PROBABILITY:
        return
    orient = random.choice(list(L9_OFFSETS))
    base_r, base_c = 12, 12
    for dr, dc in L9_OFFSETS[orient]:
        grid[base_r + dr, base_c + dc] = 9


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[dict],
    occupied_neighbors: set[tuple[int, int]],
    input_grid: np.ndarray,
) -> str | None:
    """Attempt to place a 5-cell L9 marker; return orientation string or None."""
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for marker in placed_markers:
        all_marker_cells |= _marker_cells(marker["kind"], marker["center"], marker["stem"], marker.get("variant"))

    candidates = []
    for r in range(H):
        for c in range(W):
            cells = [(r + dr, c + dc) for dr, dc in offsets]
            if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if any((cr, cc) in all_marker_cells for cr, cc in cells):
                continue
            if any((cr, cc) in occupied_neighbors for cr, cc in cells):
                continue
            candidates.append((r, c, rid))

    if not candidates:
        return None
    lr, lc, _rid = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[lr + dr, lc + dc] = 9
    return orient


def _gen_attempt() -> np.ndarray | None:
    input_grid = np.full((BOARD_SIZE, BOARD_SIZE), random.choice(COLOR_POOL), dtype=int)

    stems = list(DIRS)
    random.shuffle(stems)
    colored_index = random.randrange(len(PATCH_ANCHORS))
    colored_variant = random.choice(BIG_CROSS_VARIANTS)

    placed_markers: list[dict] = []
    for idx, anchor in enumerate(PATCH_ANCHORS):
        stem = tuple(map(int, stems[idx]))
        is_colored = idx == colored_index
        kind = "colored" if is_colored else "normal"
        variant = colored_variant if is_colored else None
        needed = 5 if is_colored else 4
        bg_color = int(input_grid[anchor])
        colors = _sample_distinct_colors(needed, banned={1, 9, bg_color})
        marker = {
            "kind": kind,
            "center": (anchor[0] + PATCH_CENTER, anchor[1] + PATCH_CENTER),
            "stem": stem,
            "variant": variant,
            "source_color": int(colors[0]),
            "dest_color": int(colors[1]),
            "gap_left_color": int(colors[2]),
            "gap_right_color": int(colors[3]),
            "target_color": int(colors[4]) if is_colored else None,
        }
        _paint_patch(input_grid, anchor, stem, marker, bg_color=bg_color)
        placed_markers.append(marker)

    _place_l9_direct(input_grid)

    try:
        result = main(input_grid)
    except Exception:
        return None
    if np.array_equal(result, input_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
