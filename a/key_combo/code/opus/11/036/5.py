import random
from collections import Counter

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MARKER_COLORS = (GREEN, YELLOW, ORANGE)
CARDINAL4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CARDINAL8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    deltas = CARDINAL4 if connectivity == 4 else CARDINAL8
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _neighbors8_of_cell(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in CARDINAL8]


def _expand8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors8_of_cell(r, c):
            if in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _generate_templates() -> list[dict]:
    bases = [
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (-1, 0), (0, -1), (0, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 0), (0, 1), (1, 1), (1, 2)},
        {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)},
        {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
    ]
    seen = set()
    templates = []
    for base in bases:
        coords = list(base)
        for _ in range(4):
            coords = [(c, -r) for r, c in coords]
            key = tuple(sorted(coords))
            if key in seen:
                continue
            seen.add(key)
            grey = sorted(pt for pt in coords if pt != (0, 0))
            templates.append({"cells": sorted(coords), "greys": grey})
    return templates


TEMPLATES = _generate_templates()


def _seed_info(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int], int]:
    blue = [(int(r), int(c)) for r, c in np.argwhere(grid == BLUE)]
    assert 2 <= len(blue) <= 3, "seed must contain 2-3 blue cells"
    blue_set = set(blue)
    endpoints = []
    for r, c in blue:
        neigh = [(nr, nc) for nr, nc in _neighbors8_of_cell(r, c) if (nr, nc) in blue_set]
        assert len(neigh) in (1, 2), "seed must be a simple diagonal chain"
        if len(neigh) == 1:
            endpoints.append(((r, c), neigh[0]))
        for nr, nc in neigh:
            assert abs(nr - r) == 1 and abs(nc - c) == 1, "seed must be diagonal"
    assert len(endpoints) == 2, "seed must have two endpoints"

    outward = []
    for endpoint, neighbor in endpoints:
        dr, dc = neighbor[0] - endpoint[0], neighbor[1] - endpoint[1]
        back = (endpoint[0] - dr, endpoint[1] - dc)
        outward.append((not in_bounds(grid.shape, back[0], back[1]), endpoint, (dr, dc)))
    starters = [item for item in outward if item[0]]
    assert len(starters) == 1, "seed must expose exactly one outer endpoint"
    _, start, direction = starters[0]
    return blue_set, start, direction, len(blue_set)


def _mirror_orientations(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirrors = {}
    for comp in components_from_mask(grid == RED, connectivity=4):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) == 1:
            orient = "horizontal"
        elif len(cols) == 1:
            orient = "vertical"
        else:
            raise AssertionError("red mirror must be straight")
        for rc in comp:
            mirrors[rc] = orient
    assert mirrors, "expected at least one mirror"
    return mirrors


def _trace_ray(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    blue_set, start, (dr, dc), _ = _seed_info(grid)
    mirrors = _mirror_orientations(grid)
    out = np.asarray(grid, dtype=int).copy()
    path = [start]
    reflections = []
    seen_states: set[tuple[int, int, int, int]] = set()
    r, c = start
    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "ray must not loop"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            break
        if int(grid[nr, nc]) == RED:
            orient = mirrors[(nr, nc)]
            if orient == "horizontal":
                dr = -dr
            else:
                dc = -dc
            reflections.append(len(path) - 1)
            continue

        r, c = nr, nc
        path.append((r, c))
        if int(grid[r, c]) not in (BLUE, RED):
            out[r, c] = BLUE
    return out, path, reflections


def _sprite_and_control_data(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    mask = np.isin(grid, [GREY, GREEN, YELLOW, ORANGE])
    sprites = []
    controls: dict[int, tuple[int, int]] = {}
    for comp in components_from_mask(mask, connectivity=4):
        if len(comp) == 1:
            r, c = comp[0]
            color = int(grid[r, c])
            assert color in MARKER_COLORS, "control must use a marker color"
            assert color not in controls, "control colors must be unique"
            controls[color] = (r, c)
            continue
        colors = Counter(int(grid[r, c]) for r, c in comp)
        assert GREY in colors, "sprite must contain grey body cells"
        marker_colors = [color for color in colors if color != GREY]
        assert len(marker_colors) == 1, "sprite must have one marker color"
        marker_color = marker_colors[0]
        marker_cells = [(r, c) for r, c in comp if int(grid[r, c]) == marker_color]
        assert len(marker_cells) == 1, "sprite marker must be unique"
        grey_cells = [(r, c) for r, c in comp if int(grid[r, c]) == GREY]
        assert 3 <= len(grey_cells) <= 4, "sprite should expose 3-4 grey body cells"
        sprites.append(
            {
                "coords": [(int(r), int(c)) for r, c in comp],
                "marker": marker_cells[0],
                "marker_color": int(marker_color),
                "grey_cells": grey_cells,
            }
        )
    assert len(sprites) == 3, "expected exactly three sprites"
    assert set(controls) == set(MARKER_COLORS), "expected one control per marker color"
    return sprites, controls


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    ray_out, path, reflections = _trace_ray(g)
    path_index = {cell: idx for idx, cell in enumerate(path)}
    sprites, controls = _sprite_and_control_data(g, bg)

    selected = []
    for sprite in sprites:
        hits = [path_index[cell] for cell in sprite["grey_cells"] if cell in path_index]
        if hits:
            sprite = dict(sprite)
            sprite["first_hit"] = min(hits)
            selected.append(sprite)
    selected.sort(key=lambda item: item["first_hit"])
    control_hits = [cell for cell in controls.values() if cell in path_index]

    return {
        "bg": bg,
        "ray_out": ray_out,
        "path": path,
        "path_index": path_index,
        "reflections": reflections,
        "sprites": sprites,
        "controls": controls,
        "selected": selected,
        "control_hits": control_hits,
    }


def _apply_moves(grid: np.ndarray, analysis: dict, chosen: list[dict]) -> np.ndarray:
    out = analysis["ray_out"].copy()
    bg = analysis["bg"]
    controls = analysis["controls"]
    for sprite in chosen:
        for r, c in sprite["coords"]:
            out[r, c] = bg
        tr, tc = controls[sprite["marker_color"]]
        out[tr, tc] = bg
        mr, mc = sprite["marker"]
        dr, dc = tr - mr, tc - mc
        for r, c in sprite["coords"]:
            nr, nc = r + dr, c + dc
            if in_bounds(out.shape, nr, nc):
                out[nr, nc] = int(grid[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend a reflected diagonal ray, then move only the sprites whose grey bodies are crossed onto same-colored controls.

    Input structure:
    - A black background contains one short blue diagonal seed near a corner and 1-2 red straight mirror segments.
    - Three small 4-connected sprites are built from grey body pixels plus one embedded marker pixel colored green, yellow, or orange.
    - Three isolated single-pixel controls of those same marker colors sit on the border and are not touched by the ray.

    Rule:
    - Start from the seed endpoint whose backward diagonal step leaves the grid, and continue in the seed's inward diagonal direction.
    - When the next cell is red, reflect without painting it: a horizontal mirror flips the row direction, and a vertical mirror flips the column direction.
    - Paint every visited non-red non-blue cell blue until the ray exits the grid.
    - Identify sprites whose grey body is crossed by the ray, order them by first contact, then translate each whole sprite so its marker lands on the isolated control of the same color; clear the old sprite and consume the used control, clipping pasted cells at the border if needed.

    Key insight:
    - The reflected ray is both a drawing process and a selector: only sprites whose grey bodies lie on the traced path are activated, and each activated sprite's own marker color tells you where it must move.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    return _apply_moves(np.asarray(grid, dtype=int), analysis, analysis["selected"])


def _plan_ray_layout(rng: random.Random, shape: tuple[int, int]) -> tuple[np.ndarray, dict] | None:
    h, w = shape
    corner = rng.choice(["tl", "tr", "bl", "br"])
    seed_len = rng.choice([2, 3])
    reflection_count = rng.choice([1, 2])

    if corner == "tl":
        start = rng.choice([(0, 0), (0, 1), (1, 0)])
        dr, dc = 1, 1
    elif corner == "tr":
        start = rng.choice([(0, w - 1), (0, w - 2), (1, w - 1)])
        dr, dc = 1, -1
    elif corner == "bl":
        start = rng.choice([(h - 1, 0), (h - 2, 0), (h - 1, 1)])
        dr, dc = -1, 1
    else:
        start = rng.choice([(h - 1, w - 1), (h - 2, w - 1), (h - 1, w - 2)])
        dr, dc = -1, -1

    seed_cells = [(start[0] + k * dr, start[1] + k * dc) for k in range(seed_len)]
    if not all(in_bounds(shape, r, c) for r, c in seed_cells):
        return None

    grid = np.zeros(shape, dtype=int)
    for r, c in seed_cells:
        grid[r, c] = BLUE

    occupied = set(seed_cells)
    path_cells = set(seed_cells)
    current = seed_cells[-1]
    cur_dir = (dr, dc)

    for reflection_idx in range(reflection_count):
        seg_min = 2
        seg_max = 6
        placed = False
        seg_lengths = list(range(seg_min, seg_max + 1))
        rng.shuffle(seg_lengths)
        orientations = ["horizontal", "vertical"]
        rng.shuffle(orientations)
        for steps in seg_lengths:
            seg = [(current[0] + i * cur_dir[0], current[1] + i * cur_dir[1]) for i in range(1, steps + 1)]
            hit = (current[0] + (steps + 1) * cur_dir[0], current[1] + (steps + 1) * cur_dir[1])
            if not all(in_bounds(shape, r, c) for r, c in seg) or not in_bounds(shape, hit[0], hit[1]):
                continue
            if any(cell in occupied for cell in seg) or hit in occupied:
                continue
            if any(cell in path_cells for cell in seg):
                continue
            for orient in orientations:
                next_dir = (-cur_dir[0], cur_dir[1]) if orient == "horizontal" else (cur_dir[0], -cur_dir[1])
                lengths = [3, 4, 5]
                rng.shuffle(lengths)
                for bar_len in lengths:
                    offsets = list(range(bar_len))
                    rng.shuffle(offsets)
                    for offset in offsets:
                        if orient == "horizontal":
                            bar = [(hit[0], hit[1] - offset + j) for j in range(bar_len)]
                        else:
                            bar = [(hit[0] - offset + j, hit[1]) for j in range(bar_len)]
                        if not all(in_bounds(shape, r, c) for r, c in bar):
                            continue
                        if any(cell in occupied or cell in path_cells for cell in bar):
                            continue
                        if orient == "horizontal" and (min(c for _, c in bar) == 0 or max(c for _, c in bar) == w - 1):
                            continue
                        if orient == "vertical" and (min(r for r, _ in bar) == 0 or max(r for r, _ in bar) == h - 1):
                            continue
                        for r, c in bar:
                            grid[r, c] = RED
                        occupied.update(bar)
                        path_cells.update(seg)
                        current = seg[-1]
                        cur_dir = next_dir
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    try:
        traced, path, reflections = _trace_ray(grid)
    except AssertionError:
        return None
    if len(reflections) != reflection_count:
        return None
    if len(path) < 11 or len(path) > 24:
        return None
    if len(set(path)) != len(path):
        return None
    return grid, {"path": path, "reflections": reflections, "seed_len": seed_len, "traced": traced}


def _place_sprite_on_hit(
    rng: random.Random,
    grid: np.ndarray,
    hit: tuple[int, int],
    path_set: set[tuple[int, int]],
    occupied_cells: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
    color: int,
) -> dict | None:
    templates = TEMPLATES[:]
    rng.shuffle(templates)
    for template in templates:
        hits = template["greys"][:]
        rng.shuffle(hits)
        for hit_rel in hits:
            mr, mc = hit[0] - hit_rel[0], hit[1] - hit_rel[1]
            coords = [(mr + r, mc + c) for r, c in template["cells"]]
            if not all(in_bounds(grid.shape, r, c) for r, c in coords):
                continue
            coord_set = set(coords)
            if any(cell in occupied_cells for cell in coord_set):
                continue
            if any(cell in occupied_buffer for cell in coord_set):
                continue
            path_overlap = coord_set & path_set
            if path_overlap != {hit}:
                continue
            marker = (mr, mc)
            if marker in path_set:
                continue
            sprite = {"marker": marker, "cells": coords, "template": template, "marker_color": color}
            return sprite
    return None


def _candidate_edge_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for c in range(w):
        cells.append((0, c))
        cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        cells.append((r, w - 1))
    seen = []
    used = set()
    for cell in cells:
        if cell not in used:
            used.add(cell)
            seen.append(cell)
    return seen


def _translated_visible(sprite: dict, control: tuple[int, int], shape: tuple[int, int]) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    mr, mc = sprite["marker"]
    tr, tc = control
    dr, dc = tr - mr, tc - mc
    visible = set()
    color_map = {}
    for r, c in sprite["cells"]:
        nr, nc = r + dr, c + dc
        if in_bounds(shape, nr, nc):
            visible.add((nr, nc))
            color_map[(nr, nc)] = sprite["marker_color"] if (r, c) == sprite["marker"] else GREY
    return visible, color_map


def _place_control_for_sprite(
    rng: random.Random,
    sprite: dict,
    path_set: set[tuple[int, int]],
    input_reserved: set[tuple[int, int]],
    output_reserved: set[tuple[int, int]],
    shape: tuple[int, int],
    require_clip: bool,
    control_buffer: set[tuple[int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    candidates = _candidate_edge_cells(shape)
    rng.shuffle(candidates)
    for cell in candidates:
        if cell in path_set or cell in input_reserved or cell in control_buffer:
            continue
        visible, _ = _translated_visible(sprite, cell, shape)
        clipped = len(visible) < len(sprite["cells"])
        if require_clip and not clipped:
            continue
        if not require_clip and clipped and rng.random() < 0.7:
            continue
        if len(visible) < 2:
            continue
        if visible & output_reserved:
            continue
        return cell, visible
    return None


def _draw_sprite(grid: np.ndarray, sprite: dict) -> None:
    for r, c in sprite["cells"]:
        grid[r, c] = sprite["marker_color"] if (r, c) == sprite["marker"] else GREY


def _place_unselected_sprite(
    rng: random.Random,
    grid: np.ndarray,
    path_set: set[tuple[int, int]],
    occupied_cells: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
    output_reserved: set[tuple[int, int]],
    color: int,
) -> tuple[dict, tuple[int, int]] | None:
    h, w = grid.shape
    anchors = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
    rng.shuffle(anchors)
    templates = TEMPLATES[:]
    rng.shuffle(templates)
    edge_cells = _candidate_edge_cells(grid.shape)
    rng.shuffle(edge_cells)
    for template in templates:
        for mr, mc in anchors:
            coords = [(mr + r, mc + c) for r, c in template["cells"]]
            coord_set = set(coords)
            if not all(in_bounds(grid.shape, r, c) for r, c in coords):
                continue
            if coord_set & path_set:
                continue
            if coord_set & occupied_cells or coord_set & occupied_buffer:
                continue
            marker = (mr, mc)
            sprite = {"marker": marker, "cells": coords, "template": template, "marker_color": color}
            for control in edge_cells:
                if control in path_set or control in occupied_cells or control in occupied_buffer:
                    continue
                visible, _ = _translated_visible(sprite, control, grid.shape)
                if visible & output_reserved:
                    continue
                if control in output_reserved:
                    continue
                return sprite, control
    return None


def _all_moved_output(grid: np.ndarray) -> np.ndarray:
    analysis = _analyze(grid)
    return _apply_moves(grid, analysis, analysis["sprites"])


def _instance_is_valid(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze(grid)
        actual = main(grid)
        move_all = _all_moved_output(grid)
    except AssertionError:
        return False

    if analysis["control_hits"]:
        return False
    if len(analysis["selected"]) != 2:
        return False
    if len(analysis["reflections"]) not in (1, 2):
        return False
    first_bounce = analysis["reflections"][0]
    hit_indices = [sprite["first_hit"] for sprite in analysis["selected"]]
    if not (hit_indices[0] <= first_bounce < hit_indices[1]):
        return False
    if np.array_equal(actual, grid):
        return False
    if np.array_equal(actual, move_all):
        return False

    remaining_controls = []
    used_colors = {sprite["marker_color"] for sprite in analysis["selected"]}
    for color, cell in analysis["controls"].items():
        if color not in used_colors:
            remaining_controls.append((color, cell))
    if len(remaining_controls) != 1:
        return False
    color, (r, c) = remaining_controls[0]
    if int(actual[r, c]) != color:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(16, 19), rng.randint(16, 19))
    layout = _plan_ray_layout(rng, shape)
    if layout is None:
        return None
    grid, info = layout
    path = info["path"]
    reflections = info["reflections"]
    seed_len = info["seed_len"]
    path_set = set(path)

    first_bounce = reflections[0]
    pre_candidates = [path[i] for i in range(seed_len, first_bounce + 1) if 2 <= path[i][0] < shape[0] - 2 and 2 <= path[i][1] < shape[1] - 2]
    post_candidates = [path[i] for i in range(first_bounce + 1, len(path)) if 2 <= path[i][0] < shape[0] - 2 and 2 <= path[i][1] < shape[1] - 2]
    if not pre_candidates or len(post_candidates) < 1:
        return None

    marker_colors = list(MARKER_COLORS)
    rng.shuffle(marker_colors)
    selected_colors = marker_colors[:2]
    unselected_color = marker_colors[2]

    occupied_cells = {(int(r), int(c)) for r, c in np.argwhere(grid != 0)}
    occupied_buffer = _expand8(occupied_cells, shape)

    first_hit = rng.choice(pre_candidates)
    second_hit = rng.choice(post_candidates)

    sprite_a = _place_sprite_on_hit(rng, grid, first_hit, path_set, occupied_cells, occupied_buffer, selected_colors[0])
    if sprite_a is None:
        return None
    occupied_cells.update(sprite_a["cells"])
    occupied_buffer |= _expand8(set(sprite_a["cells"]), shape)

    sprite_b = _place_sprite_on_hit(rng, grid, second_hit, path_set, occupied_cells, occupied_buffer, selected_colors[1])
    if sprite_b is None:
        return None
    occupied_cells.update(sprite_b["cells"])
    occupied_buffer |= _expand8(set(sprite_b["cells"]), shape)

    output_reserved = set(path)
    output_reserved.update((int(r), int(c)) for r, c in np.argwhere(grid == RED))
    output_reserved.update((int(r), int(c)) for r, c in np.argwhere(grid == BLUE))
    control_buffer: set[tuple[int, int]] = set()

    clip_required = rng.random() < 0.4
    clip_target = rng.choice([0, 1]) if clip_required else None
    selected_controls = []
    selected_sprites = [sprite_a, sprite_b]
    for idx, sprite in enumerate(selected_sprites):
        placed = _place_control_for_sprite(
            rng,
            sprite,
            path_set,
            occupied_cells,
            output_reserved,
            shape,
            require_clip=clip_target == idx,
            control_buffer=control_buffer,
        )
        if placed is None:
            return None
        control, visible = placed
        selected_controls.append(control)
        output_reserved.update(visible)
        occupied_cells.add(control)
        control_buffer |= _expand8({control}, shape)

    unselected = _place_unselected_sprite(
        rng,
        grid,
        path_set,
        occupied_cells,
        occupied_buffer | control_buffer,
        output_reserved,
        unselected_color,
    )
    if unselected is None:
        return None
    sprite_c, control_c = unselected

    _draw_sprite(grid, sprite_a)
    _draw_sprite(grid, sprite_b)
    _draw_sprite(grid, sprite_c)
    for color, cell in zip(selected_colors, selected_controls):
        grid[cell] = color
    grid[control_c] = unselected_color

    if not _instance_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
