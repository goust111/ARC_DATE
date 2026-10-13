import random
from collections import defaultdict

import numpy as np


BG = 0
GEN_RETRIES = 400
MIN_CANVAS = 18
MAX_CANVAS = 36


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _component_infos(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    comp_id = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "id": comp_id,
                    "color": color,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(cells),
                }
            )
            comp_id += 1
    return comps


def _mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["cells"]:
        mask[r - r0, c - c0] = True
    return mask


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start_r, start_c = map(int, pts[0])
    stack = [(start_r, start_c)]
    seen = np.zeros((h, w), dtype=bool)
    seen[start_r, start_c] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _detect_palette(comps: list[dict], width: int) -> list[dict]:
    edge_singletons = [comp for comp in comps if comp["area"] == 1 and comp["bbox"][1] in (0, width - 1)]
    by_col: dict[int, list[dict]] = defaultdict(list)
    for comp in edge_singletons:
        by_col[int(comp["bbox"][1])].append(comp)
    assert by_col, "expected a vertical edge palette made of single cells"
    palette = max(by_col.values(), key=len)
    palette = sorted(palette, key=lambda comp: int(comp["bbox"][0]))
    assert len(palette) >= 2
    return palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor scattered ring fragments by hole count, then stack them concentrically.

    Input structure:
    - bg = black.
    - Several scattered monochrome ring-like components with one central hole and
      optionally extra enclosed holes in their band.
    - A vertical edge palette of single pixels maps hole counts 1, 2, 3... to colors.

    Rule:
    - Count each fragment's enclosed black regions inside its bbox.
    - Recolor the whole fragment with the palette color for that hole count.
    - Sort fragments by bbox area descending and center-align them inside the
      largest bbox, producing a concentric nested output.
    """
    g = np.asarray(grid, dtype=int)
    comps = _component_infos(g, bg=BG)
    palette = _detect_palette(comps, g.shape[1])
    palette_ids = {comp["id"] for comp in palette}
    color_by_holes = {idx + 1: int(comp["color"]) for idx, comp in enumerate(palette)}

    fragments = []
    for comp in comps:
        if comp["id"] in palette_ids:
            continue
        mask = _mask_from_component(comp)
        hole_count = _count_holes(mask)
        if hole_count == 0:
            continue
        h, w = mask.shape
        fragments.append(
            {
                "mask": mask,
                "holes": hole_count,
                "h": h,
                "w": w,
                "area": h * w,
            }
        )

    assert fragments, "expected at least one ring fragment"
    fragments.sort(key=lambda frag: (-int(frag["area"]), -int(frag["h"]), -int(frag["w"])))

    out_h = int(fragments[0]["h"])
    out_w = int(fragments[0]["w"])
    out = np.zeros((out_h, out_w), dtype=int)

    for frag in fragments:
        assert int(frag["holes"]) in color_by_holes
        r0 = (out_h - int(frag["h"])) // 2
        c0 = (out_w - int(frag["w"])) // 2
        block = out[r0 : r0 + int(frag["h"]), c0 : c0 + int(frag["w"])]
        block[frag["mask"]] = int(color_by_holes[int(frag["holes"])])

    return out


def _choose_counts(num_fragments: int, legend_len: int) -> list[int]:
    active_n = random.randint(2, min(num_fragments, legend_len))
    active = random.sample(list(range(1, legend_len + 1)), active_n)
    counts = active[:]
    while len(counts) < num_fragments:
        counts.append(random.choice(active))
    random.shuffle(counts)
    return counts


def _eligible_hole_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w)):
                cells.append((r, c))
    return cells


def _pick_isolated_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    for _ in range(40):
        pool = cells[:]
        random.shuffle(pool)
        chosen = []
        for r, c in pool:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _make_ring_patch(h: int, w: int, thickness: int, hole_count: int, color: int) -> np.ndarray | None:
    if h <= 2 * thickness or w <= 2 * thickness:
        return None
    patch = np.full((h, w), int(color), dtype=int)
    patch[thickness : h - thickness, thickness : w - thickness] = BG

    if hole_count > 1:
        base_mask = patch == int(color)
        eligible = _eligible_hole_cells(base_mask)
        chosen = _pick_isolated_cells(eligible, hole_count - 1)
        if chosen is None:
            return None
        for r, c in chosen:
            patch[r, c] = BG

    mask = patch == int(color)
    if not _is_connected(mask):
        return None
    if _count_holes(mask) != hole_count:
        return None
    return patch


def _build_layers(num_fragments: int, counts: list[int]) -> list[dict] | None:
    max_side = {2: 24, 3: 22, 4: 20}[num_fragments]
    max_area = {2: 480, 3: 420, 4: 360}[num_fragments]

    counts_by_size = counts[:]
    random.shuffle(counts_by_size)

    inner_h = random.randint(1, 4)
    inner_w = random.randint(1, 6)
    layers_small = []
    last_area = -1

    for hole_count in counts_by_size:
        thickness = random.choice([2, 2, 3])
        outer_h = inner_h + 2 * thickness
        outer_w = inner_w + 2 * thickness
        area = outer_h * outer_w
        if area <= last_area:
            return None
        layers_small.append(
            {
                "holes": int(hole_count),
                "thickness": int(thickness),
                "h": int(outer_h),
                "w": int(outer_w),
                "area": int(area),
            }
        )
        last_area = area
        gap_h = random.randint(0, 1)
        gap_w = random.randint(0, 1)
        inner_h = outer_h + 2 * gap_h
        inner_w = outer_w + 2 * gap_w

    if layers_small[-1]["h"] > max_side or layers_small[-1]["w"] > max_side:
        return None
    if layers_small[-1]["area"] > max_area:
        return None

    return list(reversed(layers_small))


def _render_expected(layers: list[dict], palette_colors: list[int]) -> np.ndarray:
    out_h = int(layers[0]["h"])
    out_w = int(layers[0]["w"])
    out = np.zeros((out_h, out_w), dtype=int)
    mapping = {idx + 1: int(color) for idx, color in enumerate(palette_colors)}
    for layer in layers:
        r0 = (out_h - int(layer["h"])) // 2
        c0 = (out_w - int(layer["w"])) // 2
        block = out[r0 : r0 + int(layer["h"]), c0 : c0 + int(layer["w"])]
        block[layer["patch"] != BG] = int(mapping[int(layer["holes"])])
    return out


def _candidate_positions(occ: np.ndarray, h: int, w: int, c_min: int, c_max: int) -> list[tuple[int, int]]:
    grid_h, grid_w = occ.shape
    positions = []
    for r0 in range(1, grid_h - h):
        for c0 in range(c_min, c_max + 1):
            if r0 + h >= grid_h - 1:
                continue
            if c0 + w >= grid_w - 1:
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if not occ[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _sample_canvas(layers: list[dict], legend_len: int) -> tuple[int, int] | None:
    largest_h = int(layers[0]["h"])
    largest_w = int(layers[0]["w"])
    total_area = sum(int(layer["area"]) for layer in layers)
    side = int(np.ceil(np.sqrt(total_area * 2.0)))
    min_h = max(MIN_CANVAS, largest_h + 3, 2 * legend_len + 3, side)
    min_w = max(MIN_CANVAS, largest_w + 5, side + 3)
    if min_h > MAX_CANVAS or min_w > MAX_CANVAS:
        return None
    h = random.randint(min_h, MAX_CANVAS)
    w = random.randint(min_w, MAX_CANVAS)
    return h, w


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    block = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BG
    block[mask] = patch[mask]


def _build_input() -> np.ndarray | None:
    num_fragments = random.randint(2, 4)
    legend_len = random.randint(2, 4)
    counts = _choose_counts(num_fragments, legend_len)
    layers = _build_layers(num_fragments, counts)
    if layers is None:
        return None

    colors = random.sample(list(range(1, 10)), legend_len + num_fragments)
    palette_colors = colors[:legend_len]
    fragment_colors = colors[legend_len:]

    for layer, color in zip(layers, fragment_colors):
        patch = _make_ring_patch(
            int(layer["h"]),
            int(layer["w"]),
            int(layer["thickness"]),
            int(layer["holes"]),
            int(color),
        )
        if patch is None:
            return None
        layer["patch"] = patch

    canvas_shape = _sample_canvas(layers, legend_len)
    if canvas_shape is None:
        return None
    grid_h, grid_w = canvas_shape
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    palette_side = random.choice(["left", "right"])
    palette_col = 0 if palette_side == "left" else grid_w - 1
    buffer_col = 1 if palette_side == "left" else grid_w - 2
    palette_height = 2 * legend_len - 1
    palette_top = random.randint(1, grid_h - palette_height - 1)
    for idx, color in enumerate(palette_colors):
        grid[palette_top + 2 * idx, palette_col] = int(color)
    occ[:, palette_col] = True
    occ[:, buffer_col] = True

    shuffled_layers = layers[:]
    random.shuffle(shuffled_layers)
    c_min = 2 if palette_side == "left" else 1
    c_max = grid_w - 2
    if palette_side == "right":
        c_max = grid_w - 3

    for layer in sorted(shuffled_layers, key=lambda x: int(x["area"]), reverse=True):
        h = int(layer["patch"].shape[0])
        w = int(layer["patch"].shape[1])
        positions = _candidate_positions(occ, h, w, c_min, c_max - w + 1)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paste(grid, layer["patch"], r0, c0)
        occ[max(0, r0 - 1) : min(grid_h, r0 + h + 1), max(0, c0 - 1) : min(grid_w, c0 + w + 1)] = True

    expected = _render_expected(sorted(layers, key=lambda x: -int(x["area"])), palette_colors)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
