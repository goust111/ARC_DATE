# Combo variant of 21897d95 (index 13)
# combo_spec: D3/3+D18/0
# Name: 极简单像素标记

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_HERE = Path(__file__).resolve()
for _parent in (_HERE.parent, *_HERE.parents):
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break

from arc_utils import label_mask


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original markers were 4-cell T controls. Here a marker is a single color-1 pixel
#    embedded one cell inside the destination region. In the intended local picture:
#        D D D
#        S 1 D
#    the unique direct S neighbor is the source, while the three D neighbors show that
#    the control pixel sits on the destination side of the boundary. This replaces
#    "stem + two source diagonals" with "one source side + three destination sides".
# b) Complexity calibration:
#    A singleton pixel is much harder to notice than a T, so the generator keeps 4-8
#    markers and forces the local 3-versus-1 exposure pattern. That makes the new cue
#    visually repeat across samples instead of degenerating into arbitrary noise.
# c) Broken original assumptions:
#    Any code path that expected 4-cell color-1 components or colored T centers is no
#    longer valid. The solver now asserts that every color-1 component is a singleton
#    and reads source/destination only from its four adjacent base cells. The old
#    diagonal-source test is replaced entirely; only one direct source contact remains.
# d) Placement legality:
#    The generator only places a singleton marker on an interior destination cell whose
#    four neighbors are exactly three destination cells and one source cell. Markers are
#    buffered so they do not merge into multi-pixel color-1 components or corrupt each
#    other's local neighborhoods. Optional L9 markers also avoid these buffered cells.
# e) Visual observability:
#    Across examples, every control is the same minimal blue singleton. Humans can still
#    infer the rule because the pixel is always tucked into the region that changes, and
#    the touching region on the lone opposite side supplies the copied color.
# f) Clean visuals:
#    Regions remain large and connected, region colors exclude 1 and 9, markers are not
#    adjacent, and L9 occupies its own small footprint. This avoids layered clutter.
# g) Inferability of color changes:
#    Region input colors are unique, so a changed destination can be traced back to the
#    one touching source side of its marker without relying on any numeric color order.
# h) Coverage guardrails:
#    The generator keeps the no-rotation / clockwise / counterclockwise branches close
#    to balanced in expectation and rejects over-collapsed outputs with too few final
#    colors, so 2-4 examples still expose both the control logic and the optional L9 cue.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


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


def _singleton_markers(g: np.ndarray) -> tuple[list[dict], np.ndarray]:
    markers: list[dict] = []
    control = np.zeros(g.shape, dtype=bool)
    labels, n_labels = label_mask(g == 1, connectivity=4)
    for k in range(1, int(n_labels) + 1):
        coords = np.argwhere(labels == k)
        assert len(coords) == 1, "each color-1 marker must be a singleton"
        r, c = map(int, coords[0])
        markers.append({"kind": "normal", "cell": (r, c), "cells": {(r, c)}})
        control[r, c] = True
    assert markers, "at least one singleton control pixel is required"
    return markers, control


def _find_small_controls(g: np.ndarray, control: np.ndarray, markers: list[dict]) -> None:
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


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
    markers, control = _singleton_markers(g)
    _find_small_controls(g, control, markers)

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"
    dsu = _DSU(len(comp_colors))

    def adjacent_component_counts(cells: set[tuple[int, int]]) -> Counter:
        counts: Counter = Counter()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0:
                    counts[int(comp_id[rr, cc])] += 1
        return counts

    for marker in markers:
        if marker["kind"] == "small":
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "small marker must touch its owner region"
            owner = max(counts, key=lambda cid: (counts[cid], comp_areas[cid]))
            for comp in counts:
                if comp_colors[comp] == comp_colors[owner]:
                    dsu.union(owner, comp)
            marker["source_comp"] = owner
            continue

        r, c = marker["cell"]
        assert 0 < r < g.shape[0] - 1 and 0 < c < g.shape[1] - 1, "singleton marker must be interior"

        color_counts: Counter = Counter()
        color_to_comps: dict[int, set[int]] = {}
        color_to_unique_comp: dict[int, int] = {}
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            cid = int(comp_id[rr, cc])
            assert cid >= 0, "singleton marker must be surrounded by base regions"
            col = int(comp_colors[cid])
            color_counts[col] += 1
            color_to_comps.setdefault(col, set()).add(cid)
            color_to_unique_comp.setdefault(col, cid)

        assert sum(color_counts.values()) == 4
        assert len(color_counts) == 2, "singleton marker must separate exactly two region colors"
        source_colors = [col for col, cnt in color_counts.items() if cnt == 1]
        dest_colors = [col for col, cnt in color_counts.items() if cnt == 3]
        assert len(source_colors) == 1 and len(dest_colors) == 1, "marker must show a 1-vs-3 source/destination split"

        source_color = int(source_colors[0])
        dest_color = int(dest_colors[0])
        source_comp = int(color_to_unique_comp[source_color])
        dest_comps = list(color_to_comps[dest_color])
        dest_comp = int(dest_comps[0])
        for other in dest_comps[1:]:
            dsu.union(dest_comp, int(other))

        marker["source_comp"] = source_comp
        marker["dest_comp"] = dest_comp

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    final_color = dict(root_color)
    for marker in markers:
        source = dsu.find(marker["source_comp"])
        marker["source"] = source
        if marker["kind"] == "small":
            continue
        dest = dsu.find(marker["dest_comp"])
        replacement = int(root_color[source])
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting controls for one region"
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
        for cr, cc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            h_nb = (cr, cc + 1 if cc == c0 else cc - 1)
            v_nb = (cr + 1 if cr == r0 else cr - 1, cc)
            if h_nb in coords and v_nb in coords:
                if cr == r0 and cc == c1:
                    return "top_right"
                if cr == r0 and cc == c0:
                    return "top_left"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint boundary regions from singleton control pixels, erase the markers, and optionally rotate via an L9 cue.

    Input structure:
    - The grid is partitioned into large monochrome regions using colors other than 1.
    - Each control is a single color-1 pixel placed on the destination side of a source/destination boundary.
    - Around each control pixel, exactly one 4-neighbor belongs to the source region and the other three 4-neighbors belong to the destination region.
    - An optional 5-cell L-shaped color-9 marker may be embedded inside one region.

    Rule:
    - For every singleton color-1 marker, recolor the surrounding destination region to the original color of the lone source-side neighbor region.
    - All recolorings happen simultaneously, so a source contributes its original region color even if that source is also repainted elsewhere.
    - Replace each control pixel itself with the same final color used for its source side.
    - If an L9 marker is present, rotate the completed output: top-right corner L means 90 degrees counterclockwise, top-left corner L means 90 degrees clockwise.

    Key insight:
    - The marker is no longer a T to decode; the whole instruction is contained in one pixel's local 1-versus-3 neighborhood split.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (singleton-boundary variant) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [2, 3, 3, 2, 1]

MARKER_MIN = 4
MARKER_MAX = 8

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 2.0 / 3.0
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 240
GEN_RETRIES = 400


def _grow_regions(height: int, width: int, n_regions: int) -> np.ndarray | None:
    region = np.full((height, width), -1, dtype=int)
    seeds = set()
    for _ in range(SEED_ATTEMPTS):
        if len(seeds) >= n_regions:
            break
        seeds.add((random.randint(1, height - 2), random.randint(1, width - 2)))
    if len(seeds) < n_regions:
        return None

    frontier: list[tuple[int, int, int]] = []
    for rid, (r, c) in enumerate(list(seeds)):
        region[r, c] = rid
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and region[rr, cc] == -1:
                frontier.append((rr, cc, rid))
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
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and region[rr, cc] == -1:
                frontier.append((rr, cc, rid))

    for rid in range(n_regions):
        mask = region == rid
        if int(mask.sum()) < MIN_REGION_CELLS:
            return None
        _, n_parts = label_mask(mask, connectivity=4)
        if int(n_parts) != 1:
            return None
    return region


def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    height, width = region.shape
    undirected: set[tuple[int, int]] = set()
    for r in range(height):
        for c in range(width):
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    a = int(region[r, c])
                    b = int(region[rr, cc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed: list[tuple[int, int]] = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _choose_marker_count(n_regions: int) -> int | None:
    max_possible = min(MARKER_MAX, n_regions)
    if max_possible < MARKER_MIN:
        return None
    return random.randint(MARKER_MIN, max_possible)


def _marker_halo(cell: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = cell
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }


def _region_remains_connected(region: np.ndarray, rid: int, removed: tuple[int, int]) -> bool:
    mask = region == int(rid)
    if not mask[removed]:
        return False
    mask = mask.copy()
    mask[removed] = False
    if not mask.any():
        return False
    _, n_parts = label_mask(mask, connectivity=4)
    return int(n_parts) == 1


def _find_valid_marker_cells(
    region: np.ndarray,
    src: int,
    dst: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = region.shape
    candidates: list[tuple[int, int]] = []
    for r, c in map(tuple, np.argwhere(region == int(dst))):
        r = int(r)
        c = int(c)
        if not (0 < r < height - 1 and 0 < c < width - 1):
            continue
        halo = _marker_halo((r, c))
        if halo & forbidden:
            continue

        counts: Counter = Counter()
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            counts[int(region[rr, cc])] += 1
        if counts[int(src)] != 1 or counts[int(dst)] != 3 or len(counts) != 2:
            continue
        if not _region_remains_connected(region, dst, (r, c)):
            continue
        candidates.append((r, c))
    return candidates


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    return np.asarray(colors, dtype=int)[region]


def _has_sufficient_output_diversity(out_colors: list[int]) -> bool:
    return len(set(map(int, out_colors))) >= 3


def _stamp_markers_input(grid: np.ndarray, markers: list[tuple[tuple[int, int], int, int]]) -> None:
    for (r, c), _src, _dst in markers:
        grid[r, c] = 1


def _stamp_markers_output(grid: np.ndarray, markers: list[tuple[tuple[int, int], int, int]], out_colors: list[int]) -> None:
    for (r, c), src, _dst in markers:
        grid[r, c] = int(out_colors[src])


def _try_place_l9(
    region: np.ndarray,
    markers: list[tuple[tuple[int, int], int, int]],
    forbidden: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
) -> str | None:
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    height, width = region.shape

    marker_cells = {cell for cell, _src, _dst in markers}
    candidates: list[tuple[int, int, int]] = []
    for r in range(height):
        for c in range(width):
            cells = [(r + dr, c + dc) for dr, dc in offsets]
            if not all(0 <= rr < height and 0 <= cc < width for rr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[rr, cc]) == rid for rr, cc in cells):
                continue
            if any((rr, cc) in marker_cells for rr, cc in cells):
                continue
            if any((rr, cc) in forbidden for rr, cc in cells):
                continue
            candidates.append((r, c, rid))

    if not candidates:
        return None
    r0, c0, rid = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[r0 + dr, c0 + dc] = 9
        output_grid[r0 + dr, c0 + dc] = int(out_colors[rid])
    return orient


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    height = random.randint(min_size, GRID_SIZE_MAX)
    width = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(height, width, n_regions)
    if region is None:
        return None

    directed_pairs = _find_adjacent_pairs(region)
    n_markers = _choose_marker_count(n_regions)
    if n_markers is None:
        return None

    random.shuffle(directed_pairs)
    dest_used: set[int] = set()
    marker_pairs: list[tuple[int, int]] = []
    for src, dst in directed_pairs:
        if dst in dest_used:
            continue
        marker_pairs.append((src, dst))
        dest_used.add(dst)
        if len(marker_pairs) >= n_markers:
            break
    if len(marker_pairs) < MARKER_MIN:
        return None

    input_colors = random.sample(COLOR_POOL, n_regions)
    out_colors = list(input_colors)
    for src, dst in marker_pairs:
        out_colors[dst] = int(input_colors[src])
    if not _has_sufficient_output_diversity(out_colors):
        return None

    markers: list[tuple[tuple[int, int], int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    random.shuffle(marker_pairs)
    for src, dst in marker_pairs:
        candidates = _find_valid_marker_cells(region, src, dst, forbidden)
        if not candidates:
            return None
        cell = random.choice(candidates)
        markers.append((cell, src, dst))
        forbidden |= _marker_halo(cell)

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)
    _stamp_markers_input(input_grid, markers)
    _stamp_markers_output(output_grid, markers, out_colors)

    l9_orient = _try_place_l9(region, markers, forbidden, input_grid, output_grid, out_colors)
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
