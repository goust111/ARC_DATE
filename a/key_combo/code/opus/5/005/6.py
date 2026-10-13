import math
import random
from collections import Counter, deque
from itertools import combinations

import numpy as np


Coord = tuple[int, int]

BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400

# All prototypes are 4-connected, asymmetric, and include the bbox center pixel.
PROTOTYPES = [
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
]


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _normalize_coords(coords: list[Coord] | np.ndarray) -> tuple[Coord, ...]:
    arr = np.asarray(coords, dtype=int)
    r0, c0, _, _ = bbox_coords(arr)
    items = sorted((int(r - r0), int(c - c0)) for r, c in arr)
    return tuple(items)


def _anchor_from_bbox(bbox: tuple[int, int, int, int]) -> Coord:
    r0, c0, r1, c1 = bbox
    return (r0 + (r1 - r0) // 2, c0 + (c1 - c0) // 2)


def _find_band(grid: np.ndarray, bg: int, grey: int) -> tuple[int, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    for r in range(h):
        row = [int(v) for v in grid[r]]
        k = 0
        while k < w and row[k] != bg:
            k += 1
        if k == 0:
            continue
        colors = row[:k]
        if any(v in (bg, grey) for v in colors):
            continue
        if len(set(colors)) != len(colors):
            continue
        if any(v != bg for v in row[k:]):
            continue
        candidates.append((r, colors))
    assert candidates, "expected a horizontal color band"
    return candidates[-1]


def _bresenham(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    cells: list[Coord] = []
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def _make_line_records(anchors: list[Coord]) -> list[dict]:
    records: list[dict] = []
    for i, j in combinations(range(len(anchors)), 2):
        a = anchors[i]
        b = anchors[j]
        dr = a[0] - b[0]
        dc = a[1] - b[1]
        d2 = dr * dr + dc * dc
        records.append(
            {
                "i": i,
                "j": j,
                "d2": int(d2),
                "cells": _bresenham(a, b),
            }
        )
    records.sort(key=lambda rec: (rec["d2"], rec["i"], rec["j"]))
    return records


def _grid_from_cells(shape: tuple[int, int], cells: list[Coord], color: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in cells:
        out[r, c] = int(color)
    return out


def _is_asymmetric(shape_cells: tuple[Coord, ...]) -> bool:
    patch = _grid_from_cells((3, 3), list(shape_cells), 1)
    variants = [
        patch,
        np.flipud(patch),
        np.fliplr(patch),
        np.rot90(patch, 1),
        np.rot90(patch, 2),
        np.rot90(patch, 3),
        np.transpose(patch),
        np.fliplr(np.transpose(patch)),
    ]
    base = _normalize_coords(np.argwhere(patch == 1))
    return all(_normalize_coords(np.argwhere(v == 1)) != base for v in variants[1:])


def _sprite_cells_for_anchor(anchor: Coord, shape_cells: tuple[Coord, ...], local_anchor: Coord) -> list[Coord]:
    ar, ac = anchor
    lr, lc = local_anchor
    return [(ar + dr - lr, ac + dc - lc) for dr, dc in shape_cells]


def _render_input_from_layout(
    shape: tuple[int, int],
    anchors: list[Coord],
    shape_cells: tuple[Coord, ...],
    local_anchor: Coord,
    band_colors: list[int],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for anchor in anchors:
        for r, c in _sprite_cells_for_anchor(anchor, shape_cells, local_anchor):
            grid[r, c] = GREY
    grid[h - 1, : len(band_colors)] = np.array(band_colors, dtype=int)
    return grid


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = get_bg(g)
    band_row, band_colors = _find_band(g, bg, GREY)
    sprite_comps = _find_components_of_color(g, GREY)
    assert 4 <= len(sprite_comps) <= 5, "expected 4 or 5 grey sprites"

    shape_sigs = {_normalize_coords(comp["coords"]) for comp in sprite_comps}
    assert len(shape_sigs) == 1, "grey sprites must share one shape"
    shape_sig = next(iter(shape_sigs))
    assert 4 <= len(shape_sig) <= 6, "sprite area must be between 4 and 6"
    assert _is_asymmetric(shape_sig), "sprite shape must be asymmetric"

    anchors: list[Coord] = []
    for comp in sprite_comps:
        anchor = _anchor_from_bbox(comp["bbox"])
        assert anchor in set(map(tuple, comp["coords"].tolist())), "bbox center must lie on the sprite"
        anchors.append(anchor)

    edge_records = _make_line_records(anchors)
    d2s = [rec["d2"] for rec in edge_records]
    assert len(d2s) == len(set(d2s)), "pairwise anchor distances must be unique"

    sprite_sets = [set(map(tuple, comp["coords"].tolist())) for comp in sprite_comps]
    return {
        "grid": g,
        "shape": (h, w),
        "bg": bg,
        "band_row": band_row,
        "band_colors": band_colors,
        "sprites": sprite_comps,
        "sprite_sets": sprite_sets,
        "anchors": anchors,
        "edges": edge_records,
    }


def _solve_from_analysis(info: dict) -> np.ndarray:
    g = info["grid"]
    bg = int(info["bg"])
    band_row = int(info["band_row"])
    band_colors = list(map(int, info["band_colors"]))
    sprites = info["sprites"]
    edges = info["edges"]
    keep = edges[: len(band_colors)]

    out = g.copy()
    out[band_row, : len(band_colors)] = bg

    best_color: list[int | None] = [None] * len(sprites)
    for rank, edge in enumerate(keep):
        color = band_colors[rank]
        for r, c in edge["cells"]:
            out[r, c] = color
        if best_color[edge["i"]] is None:
            best_color[edge["i"]] = color
        if best_color[edge["j"]] is None:
            best_color[edge["j"]] = color

    for idx, comp in enumerate(sprites):
        color = GREY if best_color[idx] is None else int(best_color[idx])
        for r, c in comp["coords"]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the shortest anchor-to-anchor links in legend-ranked colors, then recolor each grey sprite by its shortest kept link.

    Input structure:
    - Black background surrounds 4 or 5 identical grey 4-connected sprites, each with the bbox-center pixel present as its anchor.
    - A one-row horizontal color band sits at the bottom, flush with the left border, with distinct colors and background to its right.
    - The grey sprites are separated in the main field, while the band is the only non-grey colored object in the input.

    Rule:
    - Find every grey sprite and take its anchor as the floor-centered pixel of its bounding box.
    - Draw Bresenham 8-connected segments between every anchor pair, compute each pair's Euclidean length, and sort all segments from shortest to longest.
    - Read the bottom color band left-to-right; if it has K colors, keep only the first K ranked segments and color rank k with band color k.
    - Recolor each grey sprite to the color of its shortest kept incident segment; sprites with no kept incident segment stay grey.
    - Erase the input color band, and whenever a segment overlaps a sprite, keep the sprite color on top.

    Key insight:
    - The color band does not describe sprite identities directly; it encodes a rank-to-color map for globally sorted anchor distances, and sprite recoloring depends on which short links survive that filter.
    """
    info = _analyze_input(np.asarray(grid, dtype=int))
    return _solve_from_analysis(info)


def _min_pixel_gap(cells_a: list[Coord], cells_b: list[Coord]) -> int:
    best = 10 ** 9
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            best = min(best, abs(ra - rb) + abs(ca - cb))
    return int(best)


def _line_intersection_is_clear(keep: list[dict], anchors: list[Coord]) -> bool:
    for idx_a in range(len(keep)):
        for idx_b in range(idx_a + 1, len(keep)):
            edge_a = keep[idx_a]
            edge_b = keep[idx_b]
            shared = set(edge_a["cells"]) & set(edge_b["cells"])
            if not shared:
                continue
            allowed: set[Coord] = set()
            for v in (edge_a["i"], edge_a["j"]):
                if v in (edge_b["i"], edge_b["j"]):
                    allowed.add(anchors[v])
            if shared - allowed:
                return False
    return True


def _validate_layout(
    shape: tuple[int, int],
    anchors: list[Coord],
    shape_cells: tuple[Coord, ...],
    local_anchor: Coord,
    band_colors: list[int],
    *,
    require_filtered: bool = True,
) -> bool:
    h, w = shape
    all_sprite_cells: list[list[Coord]] = []
    sprite_sets: list[set[Coord]] = []
    for anchor in anchors:
        cells = _sprite_cells_for_anchor(anchor, shape_cells, local_anchor)
        if any(not (0 <= r < h - 1 and 0 <= c < w) for r, c in cells):
            return False
        all_sprite_cells.append(cells)
        sprite_sets.append(set(cells))

    occupied: set[Coord] = set()
    for cells in sprite_sets:
        if occupied & cells:
            return False
        occupied |= cells

    for i, j in combinations(range(len(all_sprite_cells)), 2):
        if _min_pixel_gap(all_sprite_cells[i], all_sprite_cells[j]) < 5:
            return False

    edges = _make_line_records(anchors)
    if len({edge["d2"] for edge in edges}) != len(edges):
        return False
    if require_filtered and len(band_colors) >= len(edges):
        return False
    keep = edges[: len(band_colors)]

    if not _line_intersection_is_clear(keep, anchors):
        return False

    for edge in keep:
        cells = set(edge["cells"])
        for sprite_idx, sprite in enumerate(sprite_sets):
            if sprite_idx in (edge["i"], edge["j"]):
                continue
            if cells & sprite:
                return False

    return True


def _sample_four_anchor_layout(
    shape: tuple[int, int],
    shape_cells: tuple[Coord, ...],
    local_anchor: Coord,
    k: int,
    *,
    require_filtered: bool,
) -> list[Coord] | None:
    h, w = shape
    row_candidates = range(3, h - 5)
    col_candidates = range(3, w - 3)
    points = [(r, c) for r in row_candidates for c in col_candidates]
    if len(points) < 4:
        return None

    for _ in range(500):
        anchors = random.sample(points, 4)
        if _validate_layout(shape, anchors, shape_cells, local_anchor, [1] * k, require_filtered=require_filtered):
            return anchors
    return None


def _sample_extra_anchor(
    shape: tuple[int, int],
    base_anchors: list[Coord],
    shape_cells: tuple[Coord, ...],
    local_anchor: Coord,
    k: int,
) -> Coord | None:
    h, w = shape
    base_edges = _make_line_records(base_anchors)
    max_kept = base_edges[k - 1]["d2"]
    center_r = sum(r for r, _ in base_anchors) / len(base_anchors)
    center_c = sum(c for _, c in base_anchors) / len(base_anchors)
    candidates: list[tuple[float, Coord]] = []
    for r in range(3, h - 5):
        for c in range(3, w - 3):
            cand = (r, c)
            d2s = [((r - ar) ** 2 + (c - ac) ** 2) for ar, ac in base_anchors]
            if min(d2s) <= max_kept:
                continue
            score = math.hypot(r - center_r, c - center_c)
            candidates.append((-score, cand))
    random.shuffle(candidates)
    candidates.sort()

    for _, cand in candidates:
        anchors = base_anchors + [cand]
        if _validate_layout(shape, anchors, shape_cells, local_anchor, [1] * k):
            return cand
    return None


def _sample_layout(shape: tuple[int, int], shape_cells: tuple[Coord, ...], local_anchor: Coord) -> tuple[list[Coord], list[int]] | None:
    if random.random() < 0.45:
        k = random.randint(3, 6)
        base = _sample_four_anchor_layout(shape, shape_cells, local_anchor, k, require_filtered=False)
        if base is None:
            return None
        extra = _sample_extra_anchor(shape, base, shape_cells, local_anchor, k)
        if extra is None:
            return None
        anchors = base + [extra]
    else:
        k = random.randint(3, 5)
        anchors = _sample_four_anchor_layout(shape, shape_cells, local_anchor, k, require_filtered=True)
        if anchors is None:
            return None

    band_colors = random.sample(PALETTE, k)
    if not _validate_layout(shape, anchors, shape_cells, local_anchor, band_colors):
        return None
    return anchors, band_colors


def _sprite_shape_meta(shape_cells: tuple[Coord, ...]) -> tuple[tuple[Coord, ...], Coord]:
    arr = np.array(shape_cells, dtype=int)
    bbox = bbox_coords(arr)
    local_anchor = _anchor_from_bbox(bbox)
    return shape_cells, local_anchor


def _validate_instance(inp: np.ndarray) -> bool:
    try:
        info = _analyze_input(inp)
    except AssertionError:
        return False

    band_colors = info["band_colors"]
    anchors = info["anchors"]
    sprites = info["sprites"]
    sprite_sets = info["sprite_sets"]
    edges = info["edges"]
    keep = edges[: len(band_colors)]

    if len(band_colors) < 3 or len(band_colors) > 6:
        return False
    if len(edges) <= len(band_colors):
        return False
    if len(sprites) not in (4, 5):
        return False

    for edge in keep:
        cells = set(edge["cells"])
        for sprite_idx, sprite in enumerate(sprite_sets):
            if sprite_idx in (edge["i"], edge["j"]):
                continue
            if cells & sprite:
                return False

    if not _line_intersection_is_clear(keep, anchors):
        return False

    if len(sprites) == 5:
        touched = {edge["i"] for edge in keep} | {edge["j"] for edge in keep}
        if len(touched) >= len(sprites):
            return False

    out = main(inp)
    if np.array_equal(inp, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(19, 26), random.randint(19, 26))
    shape_cells = random.choice(PROTOTYPES)
    if not _is_asymmetric(shape_cells):
        return None
    shape_cells, local_anchor = _sprite_shape_meta(shape_cells)
    sampled = _sample_layout(shape, shape_cells, local_anchor)
    if sampled is None:
        return None
    anchors, band_colors = sampled
    inp = _render_input_from_layout(shape, anchors, shape_cells, local_anchor, band_colors)
    if not _validate_instance(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
