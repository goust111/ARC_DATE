from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _frame_infos(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perim = set(_perimeter_coords(r0, c0, r1, c1))
        comp_set = set(comp)
        if comp_set != perim:
            continue
        colors = [int(grid[r, c]) for r, c in comp]
        counts = Counter(colors)
        border_color, _ = counts.most_common(1)[0]
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "perimeter": perim,
                "border_color": int(border_color),
            }
        )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def _copy_tiles(
    h: int,
    w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    motif_rel: list[tuple[int, int]],
    side: str,
    occupied: set[tuple[int, int]] | None = None,
) -> list[list[tuple[int, int]]]:
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    ir0, ic0 = r0 + 1, c0 + 1
    if side == "right":
        tr, tc = ir0, c1 + 1
        step_r, step_c = 0, iw
    elif side == "left":
        tr, tc = ir0, c0 - iw
        step_r, step_c = 0, -iw
    elif side == "bottom":
        tr, tc = r1 + 1, ic0
        step_r, step_c = ih, 0
    else:
        tr, tc = r0 - ih, ic0
        step_r, step_c = -ih, 0

    tiles: list[list[tuple[int, int]]] = []
    while 0 <= tr and 0 <= tc and tr + ih <= h and tc + iw <= w:
        tile = [(tr + dr, tc + dc) for dr, dc in motif_rel]
        if occupied is not None and any(cell in occupied for cell in tile):
            break
        tiles.append(tile)
        tr += step_r
        tc += step_c
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each rectangle's inner motif through the border gap selected by matching color, then tile it outside.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular frames appear in non-black colors; each frame is mostly one border color but has one contiguous 2-4 pixel border segment recolored into a distinct gap color.
    - Inside each frame, the background dominates except for a tiny 1-4 pixel motif whose color matches that frame's gap segment.

    Rule:
    - For each rectangular frame, read the motif color C from its interior and find the border segment of color C to determine whether the gap is on the top, bottom, left, or right side.
    - Fill the full interior of that frame with the frame's main border color, leaving the recolored gap segment unchanged.
    - Copy the interior motif outside the frame in the gap direction, repeating it in interior-sized tiles until the next copy would leave the grid or overlap an existing non-background pixel.

    Key insight:
    - The interior color is only an index: it first identifies the matching colored gap on the border, and that border location then determines the extrusion direction for the repeated motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[counts.argmax()])

    frames = _frame_infos(g, bg)
    out = g.copy()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        border_color = int(frame["border_color"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        motif_points = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in np.argwhere(inner != bg)]
        if not motif_points:
            continue
        motif_colors = [int(g[r, c]) for r, c in motif_points]
        motif_color, _ = Counter(motif_colors).most_common(1)[0]
        motif_points = [(r, c) for r, c in motif_points if int(g[r, c]) == motif_color]
        gap_points = [(r, c) for r, c in frame["perimeter"] if int(g[r, c]) == motif_color]
        if not gap_points:
            continue

        if all(r == r0 for r, _ in gap_points):
            side = "top"
        elif all(r == r1 for r, _ in gap_points):
            side = "bottom"
        elif all(c == c0 for _, c in gap_points):
            side = "left"
        elif all(c == c1 for _, c in gap_points):
            side = "right"
        else:
            continue

        out[r0 + 1 : r1, c0 + 1 : c1] = border_color
        motif_rel = [(r - (r0 + 1), c - (c0 + 1)) for r, c in motif_points]
        occupied = {(rr, cc) for rr, cc in np.argwhere(out != bg)}
        for tile in _copy_tiles(h, w, r0, c0, r1, c1, motif_rel, side, occupied):
            for rr, cc in tile:
                out[rr, cc] = motif_color
                occupied.add((rr, cc))

    return out


MOTIF_TEMPLATES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 1), (1, 1), (2, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
]


def _template_bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _sample_motif(ih: int, iw: int, rng: random.Random) -> list[tuple[int, int]]:
    fits = [tpl for tpl in MOTIF_TEMPLATES if _template_bbox(tpl)[0] <= ih and _template_bbox(tpl)[1] <= iw]
    template = rng.choice(fits)
    th, tw = _template_bbox(template)
    off_r = rng.randint(0, ih - th)
    off_c = rng.randint(0, iw - tw)
    return sorted((off_r + r, off_c + c) for r, c in template)


def _solid_bbox_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expanded_bbox_coords(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1)
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1)
    }


def _render_input(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        border_color = int(spec["border_color"])
        gap_color = int(spec["gap_color"])
        for r, c in _perimeter_coords(r0, c0, r1, c1):
            grid[r, c] = border_color
        for r, c in spec["gap_cells"]:
            grid[r, c] = gap_color
        for dr, dc in spec["motif_rel"]:
            grid[r0 + 1 + dr, c0 + 1 + dc] = gap_color
    return grid


def _render_output(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = _render_input(h, w, specs)
    out = grid.copy()
    for spec in sorted(specs, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        r0, c0, r1, c1 = spec["bbox"]
        border_color = int(spec["border_color"])
        gap_color = int(spec["gap_color"])
        out[r0 + 1 : r1, c0 + 1 : c1] = border_color
        occupied = {(rr, cc) for rr, cc in np.argwhere(out != BG)}
        for tile in _copy_tiles(h, w, r0, c0, r1, c1, spec["motif_rel"], spec["side"], occupied):
            for rr, cc in tile:
                out[rr, cc] = gap_color
                occupied.add((rr, cc))
    return out


def _gap_cells_for_side(r0: int, c0: int, r1: int, c1: int, side: str, start: int, length: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c0 + start + k) for k in range(length)]
    if side == "bottom":
        return [(r1, c0 + start + k) for k in range(length)]
    if side == "left":
        return [(r0 + start + k, c0) for k in range(length)]
    return [(r0 + start + k, c1) for k in range(length)]


def _placement_options(
    h: int,
    w: int,
    oh: int,
    ow: int,
    side: str,
    motif_rel: list[tuple[int, int]],
    reserved_out: set[tuple[int, int]],
    reserved_frame_pad: set[tuple[int, int]],
) -> list[dict]:
    options: list[dict] = []
    for r0 in range(h - oh + 1):
        for c0 in range(w - ow + 1):
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            bbox_cells = _solid_bbox_coords(r0, c0, r1, c1)
            if bbox_cells & reserved_out:
                continue
            if _expanded_bbox_coords(r0, c0, r1, c1, h, w) & reserved_frame_pad:
                continue
            max_tiles = _copy_tiles(h, w, r0, c0, r1, c1, motif_rel, side, None)
            if not max_tiles:
                continue
            actual_tiles = _copy_tiles(h, w, r0, c0, r1, c1, motif_rel, side, reserved_out)
            if len(actual_tiles) != len(max_tiles):
                continue
            out_cells = set(bbox_cells)
            for tile in max_tiles:
                out_cells.update(tile)
            options.append({"bbox": (r0, c0, r1, c1), "out_cells": out_cells})
    return options


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    n_rects = rng.randint(1, 3)
    palette = rng.sample(list(range(1, 10)), 2 * n_rects)
    border_colors = palette[:n_rects]
    gap_colors = palette[n_rects:]

    reserved_out: set[tuple[int, int]] = set()
    reserved_frame_pad: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for border_color, gap_color in zip(border_colors, gap_colors, strict=False):
        placed = False
        for _ in range(200):
            side = rng.choice(SIDES)
            oh = rng.randint(5, min(9, h - 1))
            ow = rng.randint(5, min(9, w - 1))
            ih = oh - 2
            iw = ow - 2
            motif_rel = _sample_motif(ih, iw, rng)
            options = _placement_options(h, w, oh, ow, side, motif_rel, reserved_out, reserved_frame_pad)
            if not options:
                continue
            choice = rng.choice(options)
            r0, c0, r1, c1 = choice["bbox"]
            side_len = (ow - 2) if side in {"top", "bottom"} else (oh - 2)
            gap_len = rng.randint(2, min(4, side_len))
            gap_start = rng.randint(1, side_len - gap_len + 1)
            spec = {
                "bbox": (r0, c0, r1, c1),
                "border_color": int(border_color),
                "gap_color": int(gap_color),
                "side": side,
                "motif_rel": motif_rel,
                "gap_cells": _gap_cells_for_side(r0, c0, r1, c1, side, gap_start, gap_len),
            }
            specs.append(spec)
            reserved_out.update(choice["out_cells"])
            reserved_frame_pad.update(_expanded_bbox_coords(r0, c0, r1, c1, h, w))
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(h, w, specs)
    expected = _render_output(h, w, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        candidate = _generate_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid sample")
