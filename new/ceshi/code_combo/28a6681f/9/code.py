# Combo variant of 28a6681f (index 9)
# combo_spec: D10/1+D22/2
# Name: 多guide+多水源汇聚

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ---------------
# spec:
#   concretization:
#     The local water physics stays the same as the original task, but the admissible input family is narrowed to a
#     clearly visible fused pattern:
#       - exactly 3 upper guide components, each with its own color;
#       - the guides enter from three different edges: left, right, and top;
#       - exactly 3 small source pockets of water, each pocket touching exactly one guide and containing 1 cell;
#       - the lower area contains 1 or 2 bottom-anchored basin chambers.
#     This realizes D10/1 by making several edge-entering guides coexist, and realizes D22/2 by forcing several
#     separate source pockets to collapse into only 1 or 2 final water components.
#   behavior_trace:
#     One-basin template:
#       1. Left guide supports a droplet at row 3; that droplet can only slide right and then drop into the cup.
#       2. Right guide supports a droplet symmetrically; it can only slide left and then drop into the same cup.
#       3. Top guide supports a third droplet under the top border; it can only slide left and then fall into the
#          same lower chamber.
#       4. Final water becomes a single 3-cell pool in one basin.
#     Two-basin template:
#       1. Left and top sources still flow into the left chamber.
#       2. The right guide is shifted so its source drops into the right chamber instead.
#       3. Final water becomes a 2-cell pool in one basin plus a 1-cell pool in its neighbor.
#     Compared with the original task, the decisive observation is no longer "there may be one optional guide above";
#     the solver must trace several independent guide-fed trickles and see which chamber each one contributes to.
#   complexity_calibration:
#     The family is constrained to two hand-built geometry templates plus recoloring.
#     This keeps the rule legible: every sample repeats the same three-part syntax
#       guide -> tiny source pocket -> lower chamber(s).
#     The freedom is enough to show both "3 -> 1" and "3 -> 2" convergence, but not so large that diagonal or
#     horizontal ties would make the physics ambiguous.
#   implicit_assumptions_review:
#     Original assumptions that need review:
#       - The old upper-guide validation only recognized left/right edge-entry. That breaks once one guide may enter
#         from the top edge, so guide parsing is generalized to top/left/right.
#       - The old code did not enforce a multi-source family. In this fused version, input validation must confirm
#         there are 3-5 supported source components and that each source belongs to exactly one guide.
#       - The settle() routine still assumes every diagonal choice and nearest horizontal drop is unique. The generator
#         therefore uses only templates that preserve that uniqueness.
#     New solver assumptions:
#       - Every non-bottom color in the upper band is a guide color, not decoration.
#       - Every guide color forms exactly one 4-connected component.
#       - Every water component is a tiny supported source pocket adjacent to exactly one guide color.
#     The generator guarantees all of these.
#   placement_legality:
#     Templates are explicit 10x10 layouts, so nothing can go out of bounds.
#     Guides, source pockets, and basin walls are already non-overlapping in the templates; recoloring does not change
#     occupancy.
#     Each water cell is supported from below and has at most one horizontal background side, so the input remains
#     legal for the gravity model.
#   visual_observability:
#     Humans can directly see three separate entry edges and three separate droplets in the input.
#     The outputs visibly compress those three droplets into either one lower pool or two lower pools. That is exactly
#     the many-to-few signal required by the fusion.
#   cleanliness:
#     Guides live in the upper 5 rows, source pockets sit at their tips, and basins live in the bottom 3 rows.
#     No second annotation system is introduced, and colors are used only to separate rigid structures.
#   color_reasoning:
#     No rule depends on color identity or numeric ordering. Colors only distinguish guide components from basin walls
#     and keep the three guides visually separable.
#   fusion_metrics:
#     The generator rejects any sample unless:
#       - guide_count is 3 and the touched edges are all distinct;
#       - source_count is 3;
#       - occupied_basin_count is 1 or 2;
#       - at least 2 sources merge into the same final basin component.
#     These metrics verify that the fused behavior is actually present rather than degenerating to the original
#     single-guide family.
#   solver_assumptions:
#     main() now relies on:
#       - `_extract_guides()` finding 2-3 upper guide components and allowing top-edge entry;
#       - `_extract_sources()` finding 3-5 tiny supported water components, each assigned to one guide;
#       - `_simulate_physics()` seeing only unique diagonal and horizontal choices.
#     `generate_input()` uses fixed safe templates and solver-based rejection to satisfy each assumption.


BG = 0
WATER = 1
H = 10
W = 10
PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)


ONE_BASIN_TEMPLATE: tuple[str, ...] = (
    "0000660000",
    "4400160055",
    "0440660550",
    "0041001500",
    "0044005500",
    "0000000000",
    "0000000000",
    "0020002000",
    "0020002000",
    "0022222000",
)

TWO_BASIN_TEMPLATE: tuple[str, ...] = (
    "0000660000",
    "4400160055",
    "0440660550",
    "0041000150",
    "0044000550",
    "0000000000",
    "0000000000",
    "0020020300",
    "0020020300",
    "0022223300",
)

BASE_TEMPLATES: tuple[tuple[str, ...], ...] = (ONE_BASIN_TEMPLATE, TWO_BASIN_TEMPLATE)


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _components_from_mask(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if bool(seen[rr, cc]):
            continue
        comp: set[tuple[int, int]] = set()
        stack = [(rr, cc)]
        seen[rr, cc] = True
        while stack:
            cr, cc = stack.pop()
            comp.add((int(cr), int(cc)))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _extract_guides(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    bottom_colors = set(map(int, g[H - 1]))
    guide_colors = sorted(set(map(int, g.flatten())) - {BG, WATER} - bottom_colors)
    assert 2 <= len(guide_colors) <= 3, f"expected 2-3 guide colors, found {len(guide_colors)}"

    guides: list[dict[str, object]] = []
    for color in guide_colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(color))}
        assert pts, f"missing guide color {color}"
        assert max(r for r, _c in pts) <= 4, "guides must stay in the upper band"
        assert _component_count(pts, False) == 1, f"guide {color} must be one component"
        assert _component_count(pts, False) == _component_count(pts, True), f"guide {color} must be 4-connected"

        border_hits: set[str] = set()
        if any(r == 0 for r, _c in pts):
            border_hits.add("top")
        if any(c == 0 for _r, c in pts):
            border_hits.add("left")
        if any(c == W - 1 for _r, c in pts):
            border_hits.add("right")
        assert "bottom" not in border_hits
        assert len(border_hits) == 1, f"guide {color} must enter from exactly one non-bottom edge"
        guides.append(
            {
                "color": int(color),
                "cells": pts,
                "edge": next(iter(border_hits)),
            }
        )

    edge_count = len({str(item["edge"]) for item in guides})
    assert edge_count == len(guides), "guides must come from distinct edges"
    return sorted(guides, key=lambda item: (str(item["edge"]), int(item["color"])))


def _extract_sources(grid: np.ndarray, guides: list[dict[str, object]]) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    guide_colors = {int(item["color"]) for item in guides}
    source_components = _components_from_mask(g == int(WATER))
    assert 3 <= len(source_components) <= 5, f"expected 3-5 source pockets, found {len(source_components)}"

    sources: list[dict[str, object]] = []
    used_guides: set[int] = set()
    for comp in source_components:
        assert 1 <= len(comp) <= 2, "each source pocket must be tiny"
        touching_guides: set[int] = set()
        for r, c in comp:
            assert r == H - 1 or int(g[r + 1, c]) != int(BG), "input water must be supported"
            left_bg = c > 0 and int(g[r, c - 1]) == int(BG)
            right_bg = c + 1 < W and int(g[r, c + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < H and 0 <= nc < W:
                    col = int(g[nr, nc])
                    if col in guide_colors:
                        touching_guides.add(int(col))
        assert len(touching_guides) == 1, "each source pocket must belong to exactly one guide"
        guide_color = next(iter(touching_guides))
        used_guides.add(int(guide_color))
        sources.append({"cells": comp, "guide_color": int(guide_color)})

    assert used_guides == guide_colors, "every guide must carry at least one source pocket"
    return sorted(
        sources,
        key=lambda item: (
            int(item["guide_color"]),
            min(r for r, _c in item["cells"]),
            min(c for _r, c in item["cells"]),
        ),
    )


def _simulate_physics(
    grid: np.ndarray,
    *,
    check_output: bool = True,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    return g


def _validate_variant_input(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    guides = _extract_guides(g)
    sources = _extract_sources(g, guides)
    return guides, sources


def _source_only_grid(grid: np.ndarray, source_cells: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    out[out == int(WATER)] = int(BG)
    for r, c in source_cells:
        out[int(r), int(c)] = int(WATER)
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    guides, sources = _validate_variant_input(grid)
    settled = _simulate_physics(grid, check_output=True)
    final_components = _components_from_mask(settled == int(WATER))
    final_lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(final_components):
        for cell in comp:
            final_lookup[cell] = int(idx)

    basin_hits: dict[int, list[int]] = {}
    for source in sources:
        solo_grid = _source_only_grid(grid, set(source["cells"]))
        solo_out = _simulate_physics(solo_grid, check_output=False)
        solo_cells = {(int(r), int(c)) for r, c in np.argwhere(solo_out == int(WATER))}
        basin_ids = {final_lookup[cell] for cell in solo_cells if cell in final_lookup}
        if not basin_ids:
            best_idx = min(
                range(len(final_components)),
                key=lambda idx: min(
                    abs(int(r0 - r1)) + abs(int(c0 - c1))
                    for r0, c0 in solo_cells
                    for r1, c1 in final_components[idx]
                ),
            )
            basin_ids = {int(best_idx)}
        assert len(basin_ids) == 1, "each source should map to one final basin component"
        basin_id = next(iter(basin_ids))
        basin_hits.setdefault(int(basin_id), []).append(int(source["guide_color"]))

    merge_profile = tuple(sorted((len(v) for v in basin_hits.values()), reverse=True))
    return {
        "guide_count": int(len(guides)),
        "edge_count": int(len({str(item['edge']) for item in guides})),
        "source_count": int(len(sources)),
        "source_cells": int(np.sum(np.array(grid, dtype=int) == int(WATER))),
        "occupied_basin_count": int(len(final_components)),
        "merged_source_count": int(max((len(v) for v in basin_hits.values()), default=0)),
        "merge_profile": merge_profile,
        "guide_edges": tuple(sorted(str(item["edge"]) for item in guides)),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate several guide-fed water pockets settling through gravity into one or two lower basins.

    Input structure:
    - `0` is background and `1` is water; the other colors are rigid solids.
    - The lower rows contain one connected basin or two neighboring basin chambers anchored to the bottom.
    - The upper rows contain 2 or 3 guide components entering from different outer edges, and 3-5 tiny supported
      water pockets, each attached to exactly one guide.

    Rule:
    - Guides and basin walls remain fixed; only the blue water moves.
    - Every water pixel repeatedly falls through background, slips diagonally only through a unique open side channel,
      and slides horizontally along a supported row toward the nearest unique drop, exactly as in the base task.
    - Because several guide-fed pockets are present at once, different trickles can merge inside the same basin or end
      up split across two neighboring basins.

    Key insight:
    - The physics itself is unchanged, but the reasoning target is different: the solver must track multiple distinct
      edge-guided source pockets and infer which lower basin component each one ultimately contributes to.
    """
    g = np.array(grid, dtype=int)
    if check_input:
        _validate_variant_input(g)
    return _simulate_physics(g, check_output=check_output)


def _recolor_template(base: np.ndarray, rng: random.Random) -> np.ndarray:
    src = np.array(base, dtype=int)
    out = np.array(base, dtype=int).copy()
    labels = sorted(int(x) for x in set(map(int, out.flatten())) if int(x) > int(WATER))
    colors = rng.sample(list(PALETTE), k=len(labels))
    mapping = {int(label): int(color) for label, color in zip(labels, colors)}
    for label, color in mapping.items():
        out[src == int(label)] = int(color)
    return out


def _build_candidate(rng: random.Random) -> np.ndarray:
    base = _rows_to_grid(rng.choice(BASE_TEMPLATES))
    return _recolor_template(base, rng)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    candidate = _build_candidate(rng)
    try:
        main(candidate, check_input=True, check_output=True)
        metrics = _behavior_metrics(candidate)
    except AssertionError:
        return None

    if metrics["guide_count"] != 3:
        return None
    if metrics["edge_count"] != 3:
        return None
    if metrics["source_count"] != 3:
        return None
    if metrics["occupied_basin_count"] not in (1, 2):
        return None
    if metrics["merged_source_count"] < 2:
        return None
    if metrics["merge_profile"] not in ((3,), (2, 1)):
        return None
    return candidate


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate
    raise AssertionError("failed to generate a non-degenerate multi-guide multi-source sample")
