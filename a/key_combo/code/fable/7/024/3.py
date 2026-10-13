from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, find_components, get_bg, paste


DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OFF_PATH_BLOCK_SHAPES = [
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
]


def _touches_border(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(1, grid.shape[0] - 1):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            candidates.append((r, int(row[0])))
    assert len(candidates) == 1, "expected a unique full-width separator row"
    return candidates[0]


def _extract_sprite(grid: np.ndarray, separator_row: int, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    legend = grid[separator_row + 1 :, :]
    objects = find_components(
        legend,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="objects",
    )
    assert len(objects) == 1, "expected one legend sprite"
    sprite_object = objects[0]
    sprite_bbox = bbox(sprite_object != bg)
    assert sprite_bbox is not None
    sprite = crop_bbox(sprite_object, sprite_bbox)
    anchor_coords = np.argwhere(sprite == Color.GREY)
    assert anchor_coords.shape == (1, 2), "legend sprite must contain exactly one grey anchor"
    assert np.count_nonzero(sprite != bg) >= 4, "legend sprite must be nontrivial"
    return sprite, tuple(map(int, anchor_coords[0]))


def _find_source_arrow(main_region: np.ndarray, bg: int) -> dict:
    components = find_components(
        main_region,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    candidates: list[dict] = []
    for comp in components:
        if int(comp["area"]) != 2:
            continue
        coords = comp["coords"].astype(int, copy=False)
        dr = int(coords[1, 0] - coords[0, 0])
        dc = int(coords[1, 1] - coords[0, 1])
        if abs(dr) != 1 or abs(dc) != 1:
            continue
        cells = [tuple(map(int, cell)) for cell in coords.tolist()]
        border_cells = [cell for cell in cells if _touches_border(main_region.shape, cell)]
        if len(border_cells) != 1:
            continue
        edge = border_cells[0]
        inner = cells[0] if cells[1] == edge else cells[1]
        if _touches_border(main_region.shape, inner):
            continue
        candidates.append(
            {
                "color": int(comp["color"]),
                "edge": edge,
                "inner": inner,
                "direction": (int(inner[0] - edge[0]), int(inner[1] - edge[1])),
                "coords": {edge, inner},
            }
        )
    assert len(candidates) == 1, "expected a unique border-mounted two-pixel diagonal arrow"
    return candidates[0]


def _simulate_bounce_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    blocker_mask: np.ndarray | None = None,
) -> dict:
    h, w = shape
    r, c = map(int, start)
    dr, dc = map(int, direction)
    seen: set[tuple[int, int, int, int]] = set()
    path: list[tuple[int, int]] = []
    events: list[dict] = []
    total_reflections = 0

    while (r, c, dr, dc) not in seen:
        seen.add((r, c, dr, dc))
        nr = r + dr
        nc = c + dc
        reflected_axes = 0
        if nr < 0 or nr >= h:
            dr = -dr
            nr = r + dr
            reflected_axes += 1
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
            reflected_axes += 1
        total_reflections += reflected_axes
        event = {
            "cell": (int(nr), int(nc)),
            "direction": (int(dr), int(dc)),
            "reflected_axes": int(reflected_axes),
            "total_reflections": int(total_reflections),
        }
        events.append(event)
        if blocker_mask is not None and bool(blocker_mask[nr, nc]):
            return {
                "hit": (int(nr), int(nc)),
                "path": path,
                "events": events,
                "total_reflections": int(total_reflections),
            }
        path.append((int(nr), int(nc)))
        r, c = int(nr), int(nc)

    if blocker_mask is None:
        return {
            "hit": None,
            "path": path,
            "events": events,
            "total_reflections": int(total_reflections),
        }
    raise AssertionError("ray must hit a blocker before repeating its state")


def _parse_and_trace(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 8 and g.shape[1] >= 8

    bg = int(get_bg(g))
    separator_row, separator_color = _find_separator_row(g, bg)
    main_region = g[:separator_row, :]
    assert main_region.size > 0

    sprite, anchor = _extract_sprite(g, separator_row, bg)
    arrow = _find_source_arrow(main_region, bg)

    main_colors = colors_present(main_region, exclude={bg})
    assert len(main_colors) == 2 and arrow["color"] in main_colors
    blocker_color = next(color for color in main_colors if color != arrow["color"])
    blocker_mask = main_region == int(blocker_color)
    assert bool(blocker_mask.any()), "expected blockers in the main region"

    trace = _simulate_bounce_path(main_region.shape, arrow["inner"], arrow["direction"], blocker_mask)
    hit = trace["hit"]

    top = int(hit[0] - anchor[0])
    left = int(hit[1] - anchor[1])
    assert top >= 0 and left >= 0
    assert top + sprite.shape[0] <= separator_row and left + sprite.shape[1] <= g.shape[1]

    return {
        "grid": g,
        "bg": bg,
        "separator_row": int(separator_row),
        "separator_color": int(separator_color),
        "main_region": main_region,
        "sprite": sprite,
        "anchor": tuple(map(int, anchor)),
        "arrow": arrow,
        "source_color": int(arrow["color"]),
        "blocker_color": int(blocker_color),
        "hit": tuple(map(int, hit)),
        "path": [tuple(map(int, cell)) for cell in trace["path"]],
        "total_reflections": int(trace["total_reflections"]),
    }


def _render_from_parsed(parsed: dict) -> np.ndarray:
    out = parsed["grid"].copy()
    bg = int(parsed["bg"])
    separator_row = int(parsed["separator_row"])
    source_color = int(parsed["source_color"])
    hit = tuple(map(int, parsed["hit"]))
    anchor_r, anchor_c = map(int, parsed["anchor"])

    out[separator_row + 1 :, :] = bg
    for r, c in parsed["path"]:
        if (int(r), int(c)) != hit:
            out[int(r), int(c)] = source_color

    paste(out, parsed["sprite"], hit[0] - anchor_r, hit[1] - anchor_c, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bounce the diagonal ray inside the main area and stamp the legend sprite onto the first blocker it hits.

    Input structure:
    - The grid is split by one full-width horizontal separator line, with a legend area below it.
    - The legend area contains one multicolor sprite with exactly one grey anchor pixel.
    - Above the separator, a background field contains several same-color blockers and one two-pixel diagonal source arrow whose outer pixel sits on the boundary.

    Rule:
    - Read the firing direction from the border pixel toward the inner pixel of the arrow.
    - Advance a one-pixel diagonal ray through the main area, reflecting off the left, right, top, and separator boundaries as a mirror would.
    - Stop at the first blocker pixel the ray reaches; color the traversed non-blocker cells with the source color.
    - Copy the legend sprite into the main area so its grey anchor lands exactly on that hit blocker pixel, then clear the legend area back to background while keeping the separator line.

    Key insight:
    - The target blocker is not chosen by proximity or alignment alone; it is the unique first collision produced by simulating the reflected diagonal path.
    """
    parsed = _parse_and_trace(np.asarray(grid, dtype=int))
    return _render_from_parsed(parsed)


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in CARDINALS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
    for r, c in cells:
        mask[r - min(rs), c - min(cs)] = True
    return mask


def _make_sprite(sprite_colors: list[int], bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    height = random.randint(3, 5)
    width = random.randint(3, 6)
    max_area = min(height * width, 9)
    min_area = min(max_area, max(4, (height * width) // 2))
    area = random.randint(min_area, max_area)
    mask = _grow_connected_mask(height, width, area)
    coords = [tuple(map(int, cell)) for cell in np.argwhere(mask).tolist()]
    anchor = random.choice(coords)
    non_anchor = [cell for cell in coords if cell != anchor]
    random.shuffle(non_anchor)

    color_count = random.randint(2, min(3, len(sprite_colors)))
    chosen = sprite_colors[:color_count]
    sprite = np.full(mask.shape, bg, dtype=int)
    sprite[anchor] = Color.GREY

    for idx, cell in enumerate(non_anchor):
        sprite[cell] = int(chosen[idx % color_count])

    assert len({int(sprite[cell]) for cell in non_anchor}) >= 2
    return sprite, anchor


def _arrow_options(height: int, width: int, direction: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    dr, dc = direction
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if dr == 1 and dc == 1:
        options.extend([((0, c), (1, c + 1)) for c in range(1, width - 2)])
        options.extend([((r, 0), (r + 1, 1)) for r in range(1, height - 2)])
    elif dr == 1 and dc == -1:
        options.extend([((0, c), (1, c - 1)) for c in range(2, width - 1)])
        options.extend([((r, width - 1), (r + 1, width - 2)) for r in range(1, height - 2)])
    elif dr == -1 and dc == 1:
        options.extend([((height - 1, c), (height - 2, c + 1)) for c in range(1, width - 2)])
        options.extend([((r, 0), (r - 1, 1)) for r in range(2, height - 1)])
    else:
        options.extend([((height - 1, c), (height - 2, c - 1)) for c in range(2, width - 1)])
        options.extend([((r, width - 1), (r - 1, width - 2)) for r in range(2, height - 1)])
    return options


def _stamp_box(hit: tuple[int, int], anchor: tuple[int, int], sprite_shape: tuple[int, int]) -> set[tuple[int, int]]:
    top = int(hit[0] - anchor[0])
    left = int(hit[1] - anchor[1])
    return {
        (r, c)
        for r in range(top, top + sprite_shape[0])
        for c in range(left, left + sprite_shape[1])
    }


def _can_place_block(
    cells: set[tuple[int, int]],
    height: int,
    width: int,
    forbidden: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if (r, c) in forbidden or (r, c) in occupied:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _build_output_from_spec(spec: dict) -> np.ndarray:
    out = spec["input"].copy()
    out[spec["separator_row"] + 1 :, :] = spec["bg"]
    for r, c in spec["ray_path"]:
        out[int(r), int(c)] = spec["source_color"]
    paste(
        out,
        spec["sprite"],
        spec["target_hit"][0] - spec["anchor"][0],
        spec["target_hit"][1] - spec["anchor"][1],
        transparent=spec["bg"],
    )
    return out


def _generate_case() -> tuple[np.ndarray, dict]:
    for _ in range(400):
        palette = random.sample([c for c in range(10) if c != Color.GREY], 8)
        bg, separator_color, source_color, blocker_color = palette[:4]
        sprite_palette = palette[4:]

        main_height = random.randint(10, 18)
        width = random.randint(10, 18)

        sprite, anchor = _make_sprite(sprite_palette, bg)
        legend_height = random.randint(sprite.shape[0] + 2, min(8, 29 - main_height))
        total_height = main_height + 1 + legend_height
        if total_height > 30:
            continue

        direction = random.choice(DIAGONALS)
        arrow_choices = _arrow_options(main_height, width, direction)
        if not arrow_choices:
            continue
        edge, inner = random.choice(arrow_choices)

        empty_trace = _simulate_bounce_path((main_height, width), inner, direction, None)
        min_reflections = random.choice([1, 1, 2])
        candidates: list[dict] = []
        for step_index, event in enumerate(empty_trace["events"]):
            hit = tuple(map(int, event["cell"]))
            if step_index < 4 or int(event["total_reflections"]) < min_reflections:
                continue
            top = hit[0] - anchor[0]
            left = hit[1] - anchor[1]
            if top < 0 or left < 0:
                continue
            if top + sprite.shape[0] > main_height or left + sprite.shape[1] > width:
                continue
            stamp_cells = _stamp_box(hit, anchor, sprite.shape)
            if edge in stamp_cells or inner in stamp_cells:
                continue
            candidates.append(
                {
                    "hit": hit,
                    "step_index": step_index,
                    "stamp_cells": stamp_cells,
                    "reflections": int(event["total_reflections"]),
                }
            )
        if not candidates:
            continue

        target = random.choice(candidates)
        target_hit = target["hit"]
        step_index = int(target["step_index"])
        stamp_cells = target["stamp_cells"]
        ray_path = [tuple(map(int, event["cell"])) for event in empty_trace["events"][:step_index] if tuple(map(int, event["cell"])) != target_hit]
        prefix_cells = set(ray_path)
        future_cells = [tuple(map(int, event["cell"])) for event in empty_trace["events"][step_index + 1 :]]
        obstacle_cells = {target_hit}
        arrow_cells = {edge, inner}
        forbidden = prefix_cells | arrow_cells | stamp_cells

        future_candidates = [
            cell
            for cell in future_cells
            if _can_place_block({cell}, main_height, width, forbidden, obstacle_cells)
        ]
        if future_candidates:
            obstacle_cells.add(random.choice(future_candidates))

        wanted_decoys = random.randint(2, 5)
        for _ in range(wanted_decoys):
            shape = random.choice(OFF_PATH_BLOCK_SHAPES)
            placed = False
            for _ in range(60):
                base_r = random.randint(0, main_height - 1)
                base_c = random.randint(0, width - 1)
                cells = {(base_r + dr, base_c + dc) for dr, dc in shape}
                if _can_place_block(cells, main_height, width, forbidden, obstacle_cells):
                    obstacle_cells |= cells
                    placed = True
                    break
            if not placed:
                break

        if len(obstacle_cells) < 3:
            continue

        grid = np.full((total_height, width), bg, dtype=int)
        grid[main_height, :] = separator_color
        grid[edge] = source_color
        grid[inner] = source_color
        for r, c in obstacle_cells:
            grid[int(r), int(c)] = blocker_color

        legend_patch = np.full((legend_height, width), bg, dtype=int)
        sprite_row = random.randint(1, legend_height - sprite.shape[0])
        sprite_col = random.randint(1, width - sprite.shape[1] - 1)
        paste(legend_patch, sprite, sprite_row, sprite_col, transparent=bg)
        grid[main_height + 1 :, :] = legend_patch

        spec = {
            "input": grid,
            "bg": int(bg),
            "separator_row": int(main_height),
            "separator_color": int(separator_color),
            "source_color": int(source_color),
            "blocker_color": int(blocker_color),
            "edge": tuple(map(int, edge)),
            "inner": tuple(map(int, inner)),
            "direction": tuple(map(int, direction)),
            "target_hit": tuple(map(int, target_hit)),
            "ray_path": [tuple(map(int, cell)) for cell in ray_path],
            "sprite": sprite.copy(),
            "anchor": tuple(map(int, anchor)),
            "total_reflections": int(target["reflections"]),
        }
        expected = _build_output_from_spec(spec)
        spec["expected"] = expected

        parsed = _parse_and_trace(grid)
        if parsed["hit"] != spec["target_hit"]:
            continue
        if parsed["arrow"]["edge"] != spec["edge"] or parsed["arrow"]["inner"] != spec["inner"]:
            continue
        if parsed["arrow"]["direction"] != spec["direction"]:
            continue
        if parsed["path"] != spec["ray_path"]:
            continue
        if not np.array_equal(_render_from_parsed(parsed), expected):
            continue
        if np.array_equal(grid, expected):
            continue
        return grid, spec

    raise RuntimeError("failed to generate a valid bounce-and-stamp ARC input")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
