# Combo variant of 21897d95 (index 10)
# combo_spec: D4/1+D5/2
# Name: 朝向双重编码

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace:
#    - Every control is now the same visible object: a 4-cell color-1 T tetromino.
#    - We first recover the geometric protrusion p, then invert it as in D4/1:
#      target_dir = -p. This means the flat head, not the lone protruding arm,
#      selects the destination side and the source witness diagonals live at
#      center + target_dir + each perpendicular direction.
#    - We then apply the D5/2 behavior split on that absolute target_dir:
#      target_dir vertical (flat head up/down) => normal copy-source behavior;
#      target_dir horizontal (flat head left/right) => colored behavior, but with
#      no colored center. Instead the replacement is the color of the unique third
#      region touching the marker.
#    - Example: if the T has branches at down/left/right, then the protrusion is
#      down and the flat head faces up. The original task would have treated down
#      as the target side; this fused variant treats up as the target side. Since
#      that target_dir is vertical, the marker is normal and copies the source
#      color from the two up-side diagonals. If the same shape is rotated so the
#      flat head faces right, it still targets the flat-head side, but now it is
#      horizontal and therefore uses the third touched region color instead.
# b) Complexity calibration:
#    - Using only one of the two changes would be too weak: direction reversal
#      alone still allows colored-center markers, while orientation coding alone
#      still uses the original pointed side. Here both cues must be read together.
#    - To keep the rule visible in 2-4 examples, the generator always includes at
#      least one horizontal-target marker and at least two vertical-target markers.
#      This prevents samples from collapsing into a pure D4/1 instance.
#    - Horizontal-target markers are restricted to touch exactly three regions
#      {source, destination, third}; vertical-target markers touch exactly two
#      {source, destination}. This makes the behavioral split human-readable.
# c) Implicit-assumption review:
#    - The original solver assumed some markers had colored centers; that scan is
#      removed. All color-1 cells must now belong to 4-cell T tetrominoes.
#    - The original solver assumed the unmatched branch direction is the
#      destination ray; after D4/1 it is only a geometric protrusion, and the
#      usable direction is its opposite.
#    - The original D5/2 single variant derived normal/colored from the original
#      stem orientation. In the fusion, the discriminator must use target_dir
#      after inversion, otherwise the two dimensions are not actually combined.
#    - The colored replacement can no longer come from a literal center color.
#      The solver must infer a unique extra adjacent root besides source/dest.
# d) Placement legality:
#    - Markers may not overlap, touch each other, or touch the L9 marker buffer.
#    - The two source witness diagonals must lie in the chosen source region.
#    - The destination ray must agree with solver parsing: the first base
#      component at step >= 2 along target_dir is the chosen destination.
#    - For readability, the immediate notch cell at center + target_dir is also
#      forced to belong to the destination, so the affected side visibly touches
#      the T even though the solver still confirms destination from step 2.
#    - Horizontal-target markers additionally require a third region behind the
#      protruding arm, at center + 2 * protrusion, and no further touched colors.
# e) Visual observability:
#    - Solvers can compare the same T shape under different absolute rotations:
#      the flat head always chooses the affected side, and only horizontal
#      flat-head markers invoke the third-color rule.
#    - The destination remains visually on the flat-head side because the notch
#      cell is part of the destination region; the third color is visible behind
#      the protrusion for horizontal-target markers.
# f) Visual cleanliness:
#    - Tight adjacency constraints keep each T locally interpretable: vertical
#      markers show a clean source/destination dichotomy, while horizontal markers
#      show a controlled three-region contact.
#    - Region colors stay distinct, markers are buffered apart, and optional L9
#      markers are embedded away from those buffers.
# g) Change inferability and new solver contract:
#    - Replacement colors are always structurally signaled by source or third
#      regions, never by hidden color ordering.
#    - main() now assumes every T component is valid, every vertical target T has
#      no extra adjacent root, and every horizontal target T has exactly one extra
#      adjacent root. The generator explicitly enforces those conditions.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)


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


def _protrusion_from_branches(branches: list[tuple[int, int]]) -> tuple[int, int]:
    assert len(branches) == 3
    branch_set = set(branches)
    for d in branches:
        if _opp(d) not in branch_set:
            return d
    raise AssertionError("T marker has no single protrusion")


def _parse_tetromino(
    cells: set[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    if len(cells) != 4:
        return None
    for r, c in cells:
        branches = [(dr, dc) for dr, dc in DIRS if (r + dr, c + dc) in cells]
        if len(branches) != 3:
            continue
        protrusion = _protrusion_from_branches(branches)
        expected = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        expected.remove((r - protrusion[0], c - protrusion[1]))
        if cells == expected:
            target_dir = _opp(protrusion)
            span = [d for d in branches if d != protrusion]
            assert len(span) == 2 and span[0] == _opp(span[1])
            return (int(r), int(c)), target_dir, span
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
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, target_dir, span = parsed
        kind = "normal" if target_dir[1] == 0 else "colored"
        markers.append(
            {"kind": kind, "center": center, "target_dir": target_dir, "span": span, "cells": cells}
        )
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a T control marker"

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
        r, c = center
        step = int(start)
        while True:
            rr = r + direction[0] * step
            cc = c + direction[1] * step
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
            target_dir = marker["target_dir"]
            dest = first_component_from(center, target_dir, start=2)
            assert dest is not None, "T marker must point at a destination region"

            source_parts: list[int] = []
            for span_dir in marker["span"]:
                rr = center[0] + target_dir[0] + span_dir[0]
                cc = center[1] + target_dir[1] + span_dir[1]
                assert _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0
                source_parts.append(int(comp_id[rr, cc]))
            source = source_parts[0]
            for part in source_parts[1:]:
                assert comp_colors[part] == comp_colors[source], "T source diagonals must share a color"
                dsu.union(source, part)

            adj_counts = adjacent_component_counts(marker["cells"])
            for comp in adj_counts:
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
            marker["dest_comp"] = dest
            marker["source_comp"] = source
            marker["adjacent_comps"] = list(adj_counts)
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
        if marker["kind"] == "normal":
            extra_roots = {
                dsu.find(comp)
                for comp in marker["adjacent_comps"]
                if dsu.find(comp) not in {source, dest}
            }
            assert not extra_roots, "vertical flat-head T must not touch an extra region"
            replacement = int(root_color[source])
        else:
            extra_roots = {
                dsu.find(comp)
                for comp in marker["adjacent_comps"]
                if dsu.find(comp) not in {source, dest}
            }
            assert len(extra_roots) == 1, "horizontal flat-head T must touch exactly one third region"
            replacement = int(root_color[next(iter(extra_roots))])
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting T controls for one region"
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
    Repaint a region mosaic from uniform T markers whose flat-head side chooses the affected region and whose absolute head orientation chooses the instruction family.

    Input structure:
    - The board is divided into irregular monochrome regions, with color 1 reserved for T-shaped controls.
    - Every control is a 4-cell color-1 T tetromino; there is no special colored-center form in this variant.
    - The T's lone protruding arm is only geometric. The meaningful facing direction is the opposite side, namely the flat head.
    - An optional 5-cell color-9 L marker still requests a 90-degree rotation of the finished output.

    Rule:
    - For each T, reverse the usual reading of the shape: the flat-head side points toward the destination region, and the two diagonals on that same side identify the source region.
    - If that flat-head direction is vertical (up or down), the destination is repainted to the source region's original color.
    - If that flat-head direction is horizontal (left or right), the destination is repainted to the color of the unique third region touching the T marker.
    - After recoloring, every T cell is filled with the final color of its source-side region, and the L marker rotates the completed board: ┐ means 90 degrees counterclockwise, ┌ means 90 degrees clockwise.

    Key insight:
    - You must decode two layers at once: where the T acts comes from the flat head instead of the protrusion, and what it does comes from whether that flat head is vertical or horizontal.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (stable four-gadget fused coverage) ---------------

GRID_SIZE = 19
PATCH_SIZE = 7
PATCH_SLOTS = [(2, 2), (2, 10), (10, 2), (10, 10)]
BACKGROUND_COLOR = 0
PATCH_COLOR_POOL = [2, 3, 4, 5, 6, 7, 8]
L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}
L9_ANCHOR = {"top_left": (0, 0), "top_right": (0, GRID_SIZE - 1)}
GEN_RETRIES = 20

PATCH_SPECS = {
    "normal_up": {
        "template": (
            "SSSDSSS",
            "SSSDSSS",
            "SSSDSSS",
            "SSTTTSS",
            "SSSTSSS",
            "SSSSSSS",
            "SSSSSSS",
        ),
        "output_map": {"S": "S", "D": "S", "T": "S"},
        "roles": ("S", "D"),
    },
    "normal_down": {
        "template": (
            "SSSSSSS",
            "SSSSSSS",
            "SSSTSSS",
            "SSTTTSS",
            "SSSDSSS",
            "SSSDSSS",
            "SSSDSSS",
        ),
        "output_map": {"S": "S", "D": "S", "T": "S"},
        "roles": ("S", "D"),
    },
    "colored_right": {
        "template": (
            "CCCCSSS",
            "CCCCSSS",
            "CCCTSSS",
            "CCTTDDS",
            "CCCTSSS",
            "CCCCSSS",
            "CCCCSSS",
        ),
        "output_map": {"C": "C", "D": "C", "S": "S", "T": "S"},
        "roles": ("S", "D", "C"),
    },
    "colored_left": {
        "template": (
            "SSSCCCC",
            "SSSCCCC",
            "SSSTCCC",
            "SDDTTCC",
            "SSSTCCC",
            "SSSCCCC",
            "SSSCCCC",
        ),
        "output_map": {"C": "C", "D": "C", "S": "S", "T": "S"},
        "roles": ("S", "D", "C"),
    },
}


def _choose_role_colors(roles: tuple[str, ...]) -> dict[str, int]:
    picked = random.sample(PATCH_COLOR_POOL, len(roles))
    return {role: color for role, color in zip(roles, picked)}


def _render_patch(spec: dict[str, object], role_colors: dict[str, int], *, output: bool) -> np.ndarray:
    rows = spec["template"]
    output_map = spec["output_map"]
    patch = np.full((PATCH_SIZE, PATCH_SIZE), BACKGROUND_COLOR, dtype=int)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if output:
                patch[r, c] = role_colors[output_map[ch]]
            else:
                patch[r, c] = 1 if ch == "T" else role_colors[ch]
    return patch


def _blit_patch(board: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    board[top : top + PATCH_SIZE, left : left + PATCH_SIZE] = patch


def _stamp_l9(input_grid: np.ndarray, output_grid: np.ndarray) -> str | None:
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(["top_left", "top_right"])
    anchor_r, anchor_c = L9_ANCHOR[orient]
    for dr, dc in L9_OFFSETS[orient]:
        input_grid[anchor_r + dr, anchor_c + dc] = 9
        output_grid[anchor_r + dr, anchor_c + dc] = BACKGROUND_COLOR
    return orient


def _gen_attempt() -> np.ndarray | None:
    input_grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    output_grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)

    slot_order = PATCH_SLOTS[:]
    random.shuffle(slot_order)
    patch_kinds = ["normal_up", "normal_down", "colored_left", "colored_right"]

    for kind, (top, left) in zip(patch_kinds, slot_order):
        spec = PATCH_SPECS[kind]
        role_colors = _choose_role_colors(spec["roles"])
        _blit_patch(input_grid, _render_patch(spec, role_colors, output=False), top, left)
        _blit_patch(output_grid, _render_patch(spec, role_colors, output=True), top, left)

    l9_orient = _stamp_l9(input_grid, output_grid)
    if l9_orient == "top_right":
        output_grid = np.rot90(output_grid, 1)
    elif l9_orient == "top_left":
        output_grid = np.rot90(output_grid, 3)

    try:
        result = main(input_grid)
    except Exception:
        return None
    return input_grid if np.array_equal(result, output_grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
