# Combo variant of 28a6681f (index 35)
# combo_spec: D1/3+D10/1+D22/2
# Name: 多guide+多水源汇聚+BFS扩散

from __future__ import annotations

import random

import numpy as np


DESIGN_ANALYSIS = """
Design analysis
===============

a) Concrete fused behavior and traced example
- The source task used one gravity-driven water supply that eventually settled into adjacent bottom
  basins. This combo keeps the same visible object families: `0` is background, `1` is water, and
  2-3 other colors are static solids.
- The fused change is that there are now three edge-fed source reservoirs at once: one on the left
  edge, one on the right edge, and one at the top edge. Each source is a supported water pocket
  with exactly one visible outlet cell into the shared cavity above the basins.
- Solver trace on a typical sample:
  * old rule: each blue pixel leaves its pocket individually, falls, slips diagonally, and keeps
    searching for deeper drops until it settles in the lower basin floors;
  * new rule: remove all source water first, mark the three outlet cells as simultaneous BFS seeds,
    then fill background cells layer by layer in 4-neighbor distance order;
  * visible difference: the new output creates a single merged high-water front in the upper basin
    corridor before it reaches the deeper pockets that gravity would have filled first.

b) Complexity calibration
- The full spec allows 2-3 guides, but the generator fixes the strong three-guide case. That is
  still inside the allowed range and makes the "many sources into few basins" interaction visible
  in every sample instead of occasionally collapsing to a weak two-source scene.
- The basin layout only varies along a few controlled axes: source heights, partition height,
  whether a connected shelf is present, and left-right mirroring. This keeps the scenes readable
  while still producing repeated evidence for the fused rule.
- Total water is limited to 10-12 pixels. That is enough for three BFS layers and a visible merged
  component, but not enough to flood the entire cavity and erase the multi-source structure.

c) Hidden assumption review against the original main()
- The original `settle()` assumed each water pixel could be simulated independently. That fails
  here because the output depends on the joint frontier of all outlets at once, so the combo solver
  computes one multi-source BFS plan instead of per-pixel settling.
- The original outer fixed-point loop assumed water moves until local stability under gravity.
  The combo solver no longer searches for a stable resting arrangement; it spends a fixed water
  budget on successive BFS layers.
- The original output check required each water cell to be supported on left/below/right. That is
  invalid here because BFS is allowed to leave unsupported or high-positioned water in the shared
  corridor.
- The original guide parsing used upper-row colors and diagonal-link heuristics. In this combo that
  would be brittle, because the important signal is not a specific guide color but the existence of
  multiple edge-bound source pockets with unique outlets. The new solver therefore infers sources
  from the water components themselves.

d) Placement legality
- Every template is built on a fixed 10x10 grid with explicit coordinates. The bottom basin network
  is one bottom-anchored connected solid component; the source pockets sit well above it.
- Each water component is surrounded so it has exactly one adjacent background outlet, and every
  water cell is supported from below by either another water cell or a solid cell.
- Post-generation validation reruns the combo solver and a gravity baseline. Any candidate with
  invalid exits, disconnected basin structure, or ambiguous gravity behavior is rejected.

e) Visual observability
- A human can read the three sources directly from the pixels: blue pockets appear on three
  different edges, and each opens into the same basin corridor.
- Accepted samples must reduce three inputs to fewer output components, so at least one pair of
  wave fronts visibly meets and merges in the shared basin region. They must also differ strongly
  from the gravity baseline, so the solver does not degenerate back to the original task.

f) Visual cleanliness
- The scene is partitioned into three readable zones: source pockets along the edges, a middle
  corridor where the fronts merge, and a bottom basin network. No auxiliary markers or hidden
  labels are introduced.
- Only 2-3 solid colors are used. Colors never control ordering or matching; geometry does.

g) Inferability of the change
- The only hidden state the solver uses is BFS distance from the visible outlet cells. That is
  recoverable from the pixel geometry and repeated across samples.
- Because all accepted outputs merge at least some of the source fronts while still sitting much
  higher than the gravity baseline, the training pairs give a clear "multi-source flood fill"
  signal rather than a palette-dependent trick.

9) New solver assumptions and generator guarantees
- Assumption: there are 2-3 edge-bound source components. Guarantee: the generator always makes
  three, and `main()` accepts the full 2-3 range.
- Assumption: each source component has exactly one outlet background cell. Guarantee: the source
  pockets are hard-coded around a single mouth and are validated before use.
- Assumption: source reservoirs are only bookkeeping for water amount and do not refill. Guarantee:
  the diffusion planner marks all original source cells as blocked before the BFS expands.
- Assumption: the basin network is a single bottom-anchored solid component. Guarantee: the
  generator builds it explicitly and rejects candidates that violate this.
"""


BG = 0
WATER = 1
GRID_SHAPE = (10, 10)
SOLID_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400

ABSTRACT_BASIN = 2
ABSTRACT_SIDE_GUIDE = 3
ABSTRACT_TOP_GUIDE = 4


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((int(rr), int(cc)))
            for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                nxt = (int(nr), int(nc))
                if bool(mask[int(nr), int(nc)]) and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(sorted(comp))
    out.sort(key=lambda comp: comp[0])
    return out


def _water_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _components(np.asarray(grid) == int(WATER))


def _solid_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    solids = (np.asarray(grid) != int(BG)) & (np.asarray(grid) != int(WATER))
    return _components(solids)


def _component_edge_labels(comp: list[tuple[int, int]], h: int, w: int) -> set[str]:
    labels: set[str] = set()
    if any(r == 0 for r, _c in comp):
        labels.add("top")
    if any(c == 0 for _r, c in comp):
        labels.add("left")
    if any(c == w - 1 for _r, c in comp):
        labels.add("right")
    if any(r == h - 1 for r, _c in comp):
        labels.add("bottom")
    return labels


def _bottom_anchored_solid_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    out: list[list[tuple[int, int]]] = []
    for comp in _solid_components(grid):
        labels = _component_edge_labels(comp, h, w)
        if "bottom" in labels:
            out.append(comp)
    return out


def _basin_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    anchored = _bottom_anchored_solid_components(grid)
    assert len(anchored) == 1, "expected exactly one bottom-anchored solid basin network"
    pts = np.array(anchored[0], dtype=int)
    r0 = int(np.min(pts[:, 0]))
    c0 = int(np.min(pts[:, 1]))
    r1 = int(np.max(pts[:, 0]))
    c1 = int(np.max(pts[:, 1]))
    return int(r0), int(c0), int(r1), int(c1)


def _source_exits(
    grid: np.ndarray,
    *,
    check_input: bool,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], list[str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _water_components(g)
    if check_input:
        assert 2 <= len(comps) <= 3, "expected 2-3 edge-bound source reservoirs"

    exits: list[tuple[int, int]] = []
    source_edges: list[str] = []
    for comp in comps:
        if check_input:
            for r, c in comp:
                assert r == h - 1 or int(g[r + 1, c]) != int(BG), "input water must be supported"

        labels = _component_edge_labels(comp, h, w) - {"bottom"}
        if check_input:
            assert len(labels) == 1, "each source reservoir must clearly enter from one edge"
        edge_label = sorted(labels)[0] if labels else "internal"

        adj_bg: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                if int(g[nr, nc]) == int(BG):
                    adj_bg.add((int(nr), int(nc)))

        ordered_adj = sorted(adj_bg)
        if check_input:
            assert len(ordered_adj) == 1, f"source reservoir must have exactly one outlet, got {ordered_adj}"
        assert ordered_adj, "source reservoir must touch background through at least one outlet"

        exits.append(ordered_adj[0])
        source_edges.append(edge_label)

    if check_input:
        assert len(set(source_edges)) == len(source_edges), "source reservoirs must come from different edges"
        assert len(set(source_edges)) >= 2, "expected sources from at least two edges"
        assert len(set(exits)) == len(exits), "source outlets must be distinct"

    return comps, exits, source_edges


def _diffusion_plan(
    grid: np.ndarray,
    *,
    check_input: bool,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]], bool, int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    source_components, exits, _source_edges = _source_exits(g, check_input=check_input)

    source_cells = {cell for comp in source_components for cell in comp}
    remaining = len(source_cells)
    assert remaining >= len(exits), "water count must be at least the number of source outlets"

    seen = np.zeros((h, w), dtype=bool)
    for r, c in source_cells:
        seen[r, c] = True

    frontier = sorted(exits)
    for r, c in frontier:
        seen[r, c] = True

    filled: list[tuple[int, int]] = []
    layers: list[list[tuple[int, int]]] = []
    partial_last_layer = False

    if frontier:
        if remaining >= len(frontier):
            chosen = list(frontier)
            remaining -= len(chosen)
        else:
            chosen = frontier[:remaining]
            frontier = []
            remaining = 0
            partial_last_layer = True
        layers.append(chosen)
        filled.extend(chosen)

    while frontier and remaining > 0:
        layer: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c, h, w):
                if seen[nr, nc]:
                    continue
                if int(g[nr, nc]) != int(BG):
                    continue
                seen[nr, nc] = True
                layer.add((int(nr), int(nc)))

        ordered = sorted(layer)
        if not ordered:
            break

        if remaining >= len(ordered):
            chosen = ordered
            frontier = ordered
            remaining -= len(chosen)
        else:
            chosen = ordered[:remaining]
            frontier = []
            remaining = 0
            partial_last_layer = True

        layers.append(chosen)
        filled.extend(chosen)

    return filled, layers, partial_last_layer, remaining, exits


def _output_component_count(grid: np.ndarray) -> int:
    return len(_components(np.asarray(grid) == int(WATER)))


def _gravity_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < h
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < h
                and cc + 1 < w
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                if len(opts) != 1:
                    raise AssertionError(f"ambiguous diagonal moves: {opts}")
                rr, cc = opts[0]
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < w and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                if len(nearest) != 1:
                    raise AssertionError(f"ambiguous nearest drop columns: {nearest}")
                best = nearest[0]
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                if int(g[rr, nx]) != int(BG):
                    raise AssertionError("horizontal slide blocked")
                cc = nx
                continue

            break
        else:
            raise AssertionError("gravity baseline did not settle a water pixel")

        return int(rr), int(cc)

    for _pass in range(h * w):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("gravity baseline did not converge")

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Redistribute several edge-fed source reservoirs by a shared breadth-first flood fill.

    Input structure:
    - bg = 0. Water pixels = color 1 (blue).
    - Other colors are static solids: one bottom-anchored basin network plus 2-3 edge guides that
      hold supported source reservoirs against different borders.
    - Each source reservoir has one visible outlet cell from its guide pocket into the common basin
      corridor.

    Rule:
    - Ignore gravity-style per-pixel settling. Count every input water pixel as one unit of water,
      then remove the reservoirs from the grid.
    - Use the outlet of every source reservoir as a simultaneous 4-neighbor BFS seed and expand all
      wave fronts layer by layer through background cells only.
    - Fill each full BFS layer at once. If the final layer is only partly needed, keep the result
      deterministic by taking reading order within that layer.

    Key insight: The answer is the nearest reachable background prefix around multiple guide outlets
    at once. The important structure is where the different BFS fronts meet and merge, not where
    gravity would eventually drop individual pixels.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    assert (h, w) == GRID_SHAPE, f"expected {GRID_SHAPE}, got {(h, w)}"

    if check_input:
        anchored = _bottom_anchored_solid_components(g)
        assert len(anchored) == 1, "expected one bottom-anchored basin network"
        _source_exits(g, check_input=True)

    source_count = int(np.sum(g == int(WATER)))
    out = g.copy()
    out[out == int(WATER)] = int(BG)

    filled, _layers, _partial_last_layer, remaining, _exits = _diffusion_plan(g, check_input=check_input)
    assert remaining == 0, "not enough reachable background cells for multi-source BFS diffusion"

    for r, c in filled:
        out[r, c] = int(WATER)

    if check_output:
        assert int(np.sum(out == int(WATER))) == source_count, "water count must be conserved"
        solid_mask = (g != int(BG)) & (g != int(WATER))
        assert np.array_equal(out[solid_mask], g[solid_mask]), "solid structure must stay unchanged"
        for r, c in np.argwhere(out == int(WATER)):
            assert int(g[int(r), int(c)]) == int(BG), "output water must occupy former background cells"

    return out


def _count_unsupported(out: np.ndarray) -> int:
    h, _w = out.shape
    count = 0
    for r, c in np.argwhere(np.asarray(out) == int(WATER)):
        rr = int(r)
        cc = int(c)
        if rr + 1 < h and int(out[rr + 1, cc]) == int(BG):
            count += 1
    return count


def _distance_map(
    grid: np.ndarray,
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dist: dict[tuple[int, int], int] = {start: 0}
    queue = [start]
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for nr, nc in _neighbors4(r, c, h, w):
            nxt = (int(nr), int(nc))
            if nxt in blocked or nxt in dist:
                continue
            if int(g[nr, nc]) != int(BG):
                continue
            dist[nxt] = int(dist[(r, c)] + 1)
            queue.append(nxt)
    return dist


def _merge_boundary_contacts(grid: np.ndarray, out: np.ndarray) -> int:
    source_components, exits, _source_edges = _source_exits(grid, check_input=False)
    blocked = {cell for comp in source_components for cell in comp}
    dist_maps = [_distance_map(grid, exit_cell, blocked) for exit_cell in exits]
    filled = {(int(r), int(c)) for r, c in np.argwhere(np.asarray(out) == int(WATER))}

    labels: dict[tuple[int, int], int] = {}
    for cell in filled:
        options = [(dist_map[cell], idx) for idx, dist_map in enumerate(dist_maps) if cell in dist_map]
        if not options:
            continue
        labels[cell] = min(options)[1]

    contacts = 0
    for r, c in filled:
        here = labels.get((r, c))
        if here is None:
            continue
        for nr, nc in _neighbors4(r, c, *grid.shape):
            other = labels.get((int(nr), int(nc)))
            if other is None or other == here:
                continue
            contacts += 1
    return int(contacts // 2)


def _behavior_metrics(grid: np.ndarray) -> dict[str, float | int | bool]:
    new_out = main(grid)
    old_out = _gravity_baseline(grid)
    filled, layers, partial_last_layer, remaining, exits = _diffusion_plan(grid, check_input=False)
    assert remaining == 0

    new_rows = [int(r) for r, _c in np.argwhere(new_out == int(WATER))]
    old_rows = [int(r) for r, _c in np.argwhere(old_out == int(WATER))]
    basin_r0, basin_c0, basin_r1, basin_c1 = _basin_bbox(grid)

    basin_water = 0
    for r, c in np.argwhere(new_out == int(WATER)):
        rr = int(r)
        cc = int(c)
        if basin_r0 <= rr <= basin_r1 and basin_c0 <= cc <= basin_c1:
            basin_water += 1

    return {
        "source_count": int(len(exits)),
        "delta_vs_gravity": int(np.sum(new_out != old_out)),
        "unsupported_water": int(_count_unsupported(new_out)),
        "height_gap": float(np.mean(old_rows) - np.mean(new_rows)),
        "layer_count": int(len(layers)),
        "partial_last_layer": bool(partial_last_layer),
        "output_components": int(_output_component_count(new_out)),
        "basin_water": int(basin_water),
        "merge_contacts": int(_merge_boundary_contacts(grid, new_out)),
        "filled_cells": int(len(filled)),
    }


def _build_abstract_candidate(
    *,
    left_h: int,
    right_h: int,
    top_h: int,
    partition_top: int,
    shelf: bool,
    mirror: bool,
) -> np.ndarray:
    assert left_h in (3, 4)
    assert right_h in (3, 4)
    assert top_h == 4
    assert partition_top in (6, 7)

    g = np.zeros(GRID_SHAPE, dtype=int)

    # Bottom basin network: two neighboring basins split by a short partition.
    g[9, 1:9] = ABSTRACT_BASIN
    g[5:10, 1] = ABSTRACT_BASIN
    g[5:10, 8] = ABSTRACT_BASIN
    g[partition_top:10, 5] = ABSTRACT_BASIN
    if shelf:
        g[partition_top + 1, 6] = ABSTRACT_BASIN

    # Left source pocket.
    g[1:4, 1] = ABSTRACT_SIDE_GUIDE
    left_top = 5 - left_h
    g[left_top - 1, 0] = ABSTRACT_SIDE_GUIDE
    g[5, 0] = ABSTRACT_SIDE_GUIDE
    g[left_top:5, 0] = WATER

    # Right source pocket.
    g[1:4, 8] = ABSTRACT_SIDE_GUIDE
    right_top = 5 - right_h
    g[right_top - 1, 9] = ABSTRACT_SIDE_GUIDE
    g[5, 9] = ABSTRACT_SIDE_GUIDE
    g[right_top:5, 9] = WATER

    # Top source pocket.
    g[0:5, 4] = ABSTRACT_TOP_GUIDE
    g[0:3, 6] = ABSTRACT_TOP_GUIDE
    g[4, 5] = ABSTRACT_TOP_GUIDE
    top_start = 4 - top_h
    if top_start - 1 >= 0:
        g[top_start - 1, 5] = ABSTRACT_TOP_GUIDE
    g[top_start:4, 5] = WATER

    if mirror:
        g = np.fliplr(g).copy()

    return g


def _colorize_abstract_candidate(abstract: np.ndarray) -> np.ndarray:
    palette = random.sample(list(SOLID_POOL), 3)
    basin_color = palette[0]
    side_color = palette[1]
    top_color = palette[2] if random.random() < 0.7 else side_color

    out = np.array(abstract, dtype=int).copy()
    out[out == ABSTRACT_BASIN] = int(basin_color)
    out[out == ABSTRACT_SIDE_GUIDE] = int(side_color)
    out[out == ABSTRACT_TOP_GUIDE] = int(top_color)
    return out


def generate_input() -> np.ndarray:
    last_error: Exception | None = None
    for _ in range(GEN_RETRIES):
        abstract = _build_abstract_candidate(
            left_h=random.choice((3, 4)),
            right_h=random.choice((3, 4)),
            top_h=4,
            partition_top=random.choice((6, 7)),
            shelf=bool(random.random() < 0.7),
            mirror=bool(random.random() < 0.5),
        )
        candidate = _colorize_abstract_candidate(abstract)

        try:
            metrics = _behavior_metrics(candidate)
        except Exception as exc:
            last_error = exc
            continue

        if metrics["source_count"] != 3:
            continue
        if metrics["layer_count"] < 3:
            continue
        if not (1 <= metrics["output_components"] < metrics["source_count"]):
            continue
        if metrics["delta_vs_gravity"] < 14:
            continue
        if metrics["height_gap"] < 3.0:
            continue
        if metrics["unsupported_water"] < 4:
            continue
        if metrics["basin_water"] < 1:
            continue
        if metrics["merge_contacts"] < 2:
            continue

        return candidate

    if last_error is not None:
        raise RuntimeError(f"failed to generate a valid fused BFS multi-source sample: {last_error}")
    raise RuntimeError("failed to generate a valid fused BFS multi-source sample")
