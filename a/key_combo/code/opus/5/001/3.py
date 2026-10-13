from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg, object_interior


BACKGROUND = 0
COLORS = tuple(range(1, 10))
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 26)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 240


def _mask(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


PLAIN_SHAPES = (
    _mask(("##.", "#..", "###")),
    _mask(("###", ".#.", "##.")),
    _mask(("####", ".#..", ".###")),
    _mask((".##.", "##..", ".###")),
    _mask(("###.", "..#.", "###.")),
    _mask(("###", "##.", ".#.")),
)

SMALL_HOLE_SHAPES = (
    _mask(("###", "#.#", "###")),
    _mask(("####", "#..#", "####")),
    _mask(("###", "#.#", "#.#", "###")),
)

LARGE_HOLE_SHAPES = (
    _mask(("#####", "#...#", "#####")),
    _mask(("####", "#..#", "#..#", "####")),
    _mask(("#####", "#...#", "#...#", "#####")),
)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_shape(kind: str) -> np.ndarray:
    if kind == "plain":
        base = random.choice(PLAIN_SHAPES)
    elif kind == "small":
        base = random.choice(SMALL_HOLE_SHAPES)
    else:
        base = random.choice(LARGE_HOLE_SHAPES)
    shape = base.copy()
    if random.random() < 0.5:
        shape = np.fliplr(shape)
    shape = np.rot90(shape, k=random.randint(0, 3))
    shape = _tight_crop(shape)
    assert len(components_mask(shape, connectivity=4)) == 1
    area = int(shape.sum())
    assert 5 <= area <= 15
    return shape


def _extract_legend(grid: np.ndarray, bg: int) -> list[dict]:
    col = grid[:, 0]
    entries: list[dict] = []
    r = 0
    while r < grid.shape[0]:
        if int(col[r]) == bg:
            r += 1
            continue
        r0 = r
        while r < grid.shape[0] and int(col[r]) != bg:
            r += 1
        r1 = r - 1
        colors = [int(v) for v in col[r0 : r1 + 1]]
        assert len(colors) in (3, 4), "legend entries must contain one source and a 2-3 color palette"
        source = colors[0]
        palette = colors[1:]
        assert source not in {entry["source"] for entry in entries}
        entries.append({"source": source, "palette": palette, "row": int(r0), "order": len(entries)})
    assert 2 <= len(entries) <= 3
    return entries


def _collect_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict], list[dict], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    legend = _extract_legend(g, bg)
    work = g.copy()
    work[:, :2] = bg

    objects: list[dict] = []
    source_pixels: dict[int, np.ndarray] = {}
    for entry in legend:
        source = int(entry["source"])
        comps = find_components(
            work,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[source],
            return_format="dict",
        )
        assert comps, "each legend source color must label at least one object"
        source_pixels[source] = np.argwhere(work == source)
        for comp in comps:
            objects.append(
                {
                    "source": source,
                    "palette": list(entry["palette"]),
                    "order": int(entry["order"]),
                    "coords": comp["coords"],
                    "bbox": tuple(map(int, comp["bbox"])),
                    "area": int(comp["area"]),
                }
            )
    assert 2 <= len(objects) <= 4
    return work, bg, legend, objects, source_pixels


def _object_holes(obj: dict, bg: int) -> list[np.ndarray]:
    r0, c0, r1, c1 = obj["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = obj["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(obj["source"])
    hole_mask = object_interior(patch, background=bg) & (patch == int(bg))
    holes: list[np.ndarray] = []
    for comp in components_mask(hole_mask, connectivity=4):
        global_coords = comp.copy()
        global_coords[:, 0] += int(r0)
        global_coords[:, 1] += int(c0)
        holes.append(global_coords)
    return holes


def _large_hole_mask(shape: tuple[int, int], objects: list[dict], bg: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for obj in objects:
        for hole in _object_holes(obj, bg):
            if int(hole.shape[0]) >= 3:
                mask[hole[:, 0], hole[:, 1]] = True
    return mask


def _distance_arrays(shape: tuple[int, int], legend: list[dict], source_pixels: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    rr, cc = np.indices(shape)
    dists: dict[int, np.ndarray] = {}
    for entry in legend:
        coords = source_pixels[int(entry["source"])]
        dist = np.abs(rr[..., None] - coords[:, 0]) + np.abs(cc[..., None] - coords[:, 1])
        dists[int(entry["source"])] = dist.min(axis=2)
    return dists


def _expected_by_rules(grid: np.ndarray) -> np.ndarray:
    work, bg, legend, objects, source_pixels = _collect_scene(grid)
    H, W = work.shape
    out = work.copy()
    out[:, :2] = bg
    keep_black = _large_hole_mask((H, W), objects, bg)
    fillable = (out == bg)
    fillable[:, :2] = False
    fillable &= ~keep_black

    for r in range(H):
        for c in range(2, W):
            if not fillable[r, c]:
                continue
            best: tuple[int, int, int] | None = None
            for entry in legend:
                coords = source_pixels[int(entry["source"])]
                distance = min(abs(int(r) - int(pr)) + abs(int(c) - int(pc)) for pr, pc in coords)
                palette = entry["palette"]
                color = int(palette[(distance - 1) % len(palette)])
                cand = (int(distance), int(entry["order"]), color)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            assert best is not None
            out[r, c] = int(best[2])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the left legend and paint cyclic Manhattan-distance halos around matching objects while preserving large enclosed holes.

    Input structure:
    - The dominant background is black, with 2-3 legend entries stacked in column 0 and column 1 left blank.
    - Each legend entry is a vertical run whose top pixel is a source color and whose lower 2-3 pixels form that source's cyclic halo palette.
    - From column 2 onward there are 2-4 irregular 4-connected monochrome objects using the legend source colors; some may contain enclosed background holes.

    Rule:
    - Clear the legend area (columns 0 and 1) to black in the output.
    - For every remaining background cell, measure Manhattan distance to the nearest pixel of each source-color object set, using object pixels rather than object centers.
    - Color the cell by the nearest source's palette, cycling by distance so distance 1 uses palette[0], distance 2 uses palette[1], and so on; ties between different sources go to the legend entry whose top pixel is higher.
    - Keep object pixels unchanged, and keep any enclosed background hole of area at least 3 black; smaller enclosed gaps are treated as ordinary background and receive halo colors.

    Key insight:
    - The left legend maps each object color to an ordered multi-color cycle, so the visible ring color is determined jointly by nearest-object Manhattan distance and by that source's palette period.
    """
    work, bg, legend, objects, source_pixels = _collect_scene(grid)
    H, W = work.shape
    out = work.copy()
    out[:, :2] = bg
    keep_black = _large_hole_mask((H, W), objects, bg)
    fillable = (out == bg)
    fillable[:, :2] = False
    fillable &= ~keep_black

    best_distance = np.full((H, W), 10**9, dtype=int)
    best_order = np.full((H, W), 10**9, dtype=int)
    best_color = np.full((H, W), int(bg), dtype=int)

    dists = _distance_arrays((H, W), legend, source_pixels)
    for entry in legend:
        source = int(entry["source"])
        palette = np.asarray(entry["palette"], dtype=int)
        dist = dists[source]
        color_idx = (dist - 1) % len(palette)
        candidate_color = palette[color_idx]
        better = fillable & (
            (dist < best_distance) |
            ((dist == best_distance) & (int(entry["order"]) < best_order))
        )
        best_distance[better] = dist[better]
        best_order[better] = int(entry["order"])
        best_color[better] = candidate_color[better]

    out[fillable] = best_color[fillable]
    return out


def _min_gap(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> int:
    best = 10**9
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            dist = abs(int(ra) - int(rb)) + abs(int(ca) - int(cb))
            if dist < best:
                best = int(dist)
    return best


def _translated_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [(int(r + top), int(c + left)) for r, c in coords]


def _make_palette(source: int, length: int, banned_first: set[int]) -> list[int]:
    pool = [color for color in COLORS if color != int(source)]
    random.shuffle(pool)
    first_choices = [color for color in pool if color not in banned_first]
    first = random.choice(first_choices or pool)
    rest_pool = [color for color in pool if color != first]
    rest = random.sample(rest_pool, length - 1)
    return [int(first), *map(int, rest)]


def _make_legend() -> list[dict]:
    num_sources = random.choice((2, 2, 3))
    sources = random.sample(COLORS, num_sources)
    lengths = [2, 3]
    if num_sources == 3:
        lengths.append(random.choice((2, 3)))
    random.shuffle(lengths)
    if len(set(lengths)) == 1:
        lengths[0] = 2
        lengths[1] = 3

    used_first_colors: set[int] = set()
    legend: list[dict] = []
    for order, (source, length) in enumerate(zip(sources, lengths)):
        palette = _make_palette(int(source), int(length), used_first_colors)
        used_first_colors.add(int(palette[0]))
        legend.append({"source": int(source), "palette": palette, "order": int(order)})
    return legend


def _choose_features() -> dict:
    need_large = random.random() < 0.7
    need_small = random.random() < 0.55
    if not need_large and not need_small:
        if random.random() < 0.5:
            need_large = True
        else:
            need_small = True
    return {
        "large_hole": need_large,
        "small_hole": need_small,
        "duplicate": random.random() < 0.5,
        "tie": True,
    }


def _assign_shape_kinds(count: int, expect: dict) -> list[str]:
    kinds = ["plain"] * count
    indices = list(range(count))
    if expect["large_hole"]:
        idx = random.choice(indices)
        kinds[idx] = "large"
    if expect["small_hole"]:
        choices = [idx for idx in indices if kinds[idx] == "plain"]
        if not choices:
            choices = indices
        idx = random.choice(choices)
        kinds[idx] = "small"
    return kinds


def _place_legend(grid: np.ndarray, legend: list[dict]) -> None:
    H = grid.shape[0]
    heights = [1 + len(entry["palette"]) for entry in legend]
    required = sum(heights) + 2 * (len(legend) - 1)
    slack = H - required
    assert slack >= 0

    extras = [0] * (len(legend) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1

    top = extras[0]
    for idx, (entry, height) in enumerate(zip(legend, heights)):
        entry["row"] = int(top)
        colors = [int(entry["source"]), *map(int, entry["palette"])]
        grid[top : top + height, 0] = colors
        top += height
        if idx != len(legend) - 1:
            top += 2 + extras[idx + 1]


def _is_valid_placement(
    cells: list[tuple[int, int]],
    source: int,
    placed: list[dict],
    H: int,
    W: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < H and 3 <= c < W):
            return False
    for other in placed:
        gap = _min_gap(cells, other["cells"])
        required = 4 if int(source) != int(other["source"]) else 2
        if gap < required:
            return False
    return True


def _try_place_objects(H: int, W: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []

    first = specs[0]
    second = specs[1]
    gap = random.randint(4, 7)
    w_need = first["mask"].shape[1] + second["mask"].shape[1] + gap
    if 3 + w_need >= W:
        return None
    left = random.randint(3, W - w_need - 1)
    row = random.randint(1, max(1, H - max(first["mask"].shape[0], second["mask"].shape[0]) - 1))
    first_cells = _translated_cells(first["mask"], row, left)
    second_cells = _translated_cells(second["mask"], row, left + first["mask"].shape[1] + gap)
    if _min_gap(first_cells, second_cells) < 4:
        return None
    placed.append({**first, "top": int(row), "left": int(left), "cells": first_cells})
    placed.append(
        {
            **second,
            "top": int(row),
            "left": int(left + first["mask"].shape[1] + gap),
            "cells": second_cells,
        }
    )

    for spec in specs[2:]:
        sibling_cells = None
        for other in placed:
            if int(other["source"]) == int(spec["source"]):
                sibling_cells = other["cells"]
                break
        placed_one = False
        for _ in range(PLACEMENT_ATTEMPTS):
            if sibling_cells is not None and random.random() < 0.65:
                anchor_r = min(r for r, _ in sibling_cells)
                anchor_c = max(c for _, c in sibling_cells)
                top_lo = max(1, anchor_r - 3)
                top_hi = min(H - spec["mask"].shape[0] - 1, anchor_r + 4)
                left_lo = max(3, anchor_c + 2)
                left_hi = min(W - spec["mask"].shape[1] - 1, anchor_c + 7)
                if top_lo > top_hi or left_lo > left_hi:
                    continue
                top = random.randint(top_lo, top_hi)
                left = random.randint(left_lo, left_hi)
            else:
                top = random.randint(1, H - spec["mask"].shape[0] - 1)
                left = random.randint(3, W - spec["mask"].shape[1] - 1)
            cells = _translated_cells(spec["mask"], top, left)
            if not _is_valid_placement(cells, int(spec["source"]), placed, H, W):
                continue
            placed.append({**spec, "top": int(top), "left": int(left), "cells": cells})
            placed_one = True
            break
        if not placed_one:
            return None
    return placed


def _render_grid(H: int, W: int, legend: list[dict], placed: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    _place_legend(grid, legend)
    for item in placed:
        for r, c in item["cells"]:
            grid[r, c] = int(item["source"])
    return grid


def _hole_summary(objects: list[dict], bg: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    small: list[np.ndarray] = []
    large: list[np.ndarray] = []
    for obj in objects:
        for hole in _object_holes(obj, bg):
            if int(hole.shape[0]) >= 3:
                large.append(hole)
            else:
                small.append(hole)
    return small, large


def _has_tie_effect(work: np.ndarray, legend: list[dict], source_pixels: dict[int, np.ndarray], keep_black: np.ndarray) -> bool:
    dists = _distance_arrays(work.shape, legend, source_pixels)
    fillable = (work == BACKGROUND)
    fillable[:, :2] = False
    fillable &= ~keep_black

    for r, c in np.argwhere(fillable):
        min_dist = min(int(dists[int(entry["source"])][r, c]) for entry in legend)
        winners = [entry for entry in legend if int(dists[int(entry["source"])][r, c]) == min_dist]
        if len(winners) < 2:
            continue
        colors = {int(entry["palette"][(min_dist - 1) % len(entry["palette"])]) for entry in winners}
        if len(colors) > 1:
            return True
    return False


def _semantic_check(grid: np.ndarray, expect: dict) -> bool:
    try:
        actual = main(grid)
        expected = _expected_by_rules(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    work, bg, legend, objects, source_pixels = _collect_scene(grid)
    if not np.all(actual[:, :2] == int(bg)):
        return False
    if len({len(entry["palette"]) for entry in legend}) < 2:
        return False

    source_counts = Counter(int(obj["source"]) for obj in objects)
    if expect["duplicate"] and not any(count >= 2 for count in source_counts.values()):
        return False
    if not all(source_counts[int(entry["source"])] >= 1 for entry in legend):
        return False

    for i, obj_a in enumerate(objects):
        bbox = obj_a["bbox"]
        area = int(obj_a["area"])
        if area == (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1):
            return False
        for obj_b in objects[i + 1 :]:
            if int(obj_a["source"]) == int(obj_b["source"]):
                continue
            gap = _min_gap(
                [(int(r), int(c)) for r, c in obj_a["coords"]],
                [(int(r), int(c)) for r, c in obj_b["coords"]],
            )
            if gap < 4:
                return False

    small_holes, large_holes = _hole_summary(objects, bg)
    if expect["small_hole"]:
        if not small_holes:
            return False
        if not all(np.all(actual[hole[:, 0], hole[:, 1]] != int(bg)) for hole in small_holes):
            return False
    if expect["large_hole"]:
        if not large_holes:
            return False
        if not all(np.all(actual[hole[:, 0], hole[:, 1]] == int(bg)) for hole in large_holes):
            return False

    keep_black = _large_hole_mask(work.shape, objects, bg)
    if expect["tie"] and not _has_tie_effect(work, legend, source_pixels, keep_black):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    legend = _make_legend()
    expect = _choose_features()
    num_objects = len(legend) + (1 if expect["duplicate"] else 0)
    num_objects = min(4, num_objects)
    kinds = _assign_shape_kinds(num_objects, expect)

    sources = [int(entry["source"]) for entry in legend]
    object_sources = list(sources)
    while len(object_sources) < num_objects:
        object_sources.append(int(random.choice(sources)))

    specs: list[dict] = []
    for source, kind in zip(object_sources, kinds[:num_objects]):
        mask = _random_shape(kind)
        specs.append({"source": int(source), "mask": mask, "kind": kind})

    H_min = max(GRID_H_RANGE[0], sum(1 + len(entry["palette"]) for entry in legend) + 2 * (len(legend) - 1))
    W_min = max(GRID_W_RANGE[0], 3 + specs[0]["mask"].shape[1] + specs[1]["mask"].shape[1] + 6)
    if H_min > GRID_H_RANGE[1] or W_min > GRID_W_RANGE[1]:
        return None

    H = random.randint(H_min, GRID_H_RANGE[1])
    W = random.randint(W_min, GRID_W_RANGE[1])
    placed = _try_place_objects(H, W, specs)
    if placed is None:
        return None

    grid = _render_grid(H, W, legend, placed)
    if not _semantic_check(grid, expect):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cyclic-halo ARC input")
