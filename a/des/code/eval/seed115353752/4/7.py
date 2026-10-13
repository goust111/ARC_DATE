from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ARROW_COLOR = 5
FRAME_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_l_shape(coords: np.ndarray) -> bool:
    if len(coords) != 3:
        return False
    pts = {tuple(map(int, pt)) for pt in coords}
    degrees = []
    for r, c in pts:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        degrees.append(deg)
    degrees.sort()
    return degrees == [1, 1, 2]


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue
            if len(comp) != 2 * (h + w) - 4:
                continue
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            occ = np.zeros((h, w), dtype=bool)
            occ[comp[:, 0] - r0, comp[:, 1] - c0] = True
            if not np.array_equal(occ, border):
                continue
            frames.append({"color": color, "bbox": (r0, c0, r1, c1), "cells": comp})
    assert frames, "expected at least one colored frame"
    return frames


def _parse_arrow(interior: np.ndarray, bg: int) -> dict:
    comps = _components(interior != bg)
    arrows = [comp for comp in comps if _is_l_shape(comp)]
    assert len(arrows) == 1, "each frame interior must contain exactly one L-shaped arrow"
    coords = arrows[0]
    h, w = interior.shape
    pts = {tuple(map(int, pt)) for pt in coords}

    degrees: dict[tuple[int, int], int] = {}
    for r, c in pts:
        degrees[(r, c)] = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
    bend = next(cell for cell, deg in degrees.items() if deg == 2)
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "L-shape must have exactly two endpoints"

    top = int(coords[:, 0].max()) <= 1
    bottom = int(coords[:, 0].min()) >= h - 2
    left = int(coords[:, 1].max()) <= 1
    right = int(coords[:, 1].min()) >= w - 2
    assert top ^ bottom, "arrow must sit on exactly one vertical side"
    assert left ^ right, "arrow must sit on exactly one horizontal side"
    start_top = bool(top)
    start_left = bool(left)

    def corner_distance(cell: tuple[int, int]) -> int:
        r, c = cell
        dr = r if start_top else (h - 1 - r)
        dc = c if start_left else (w - 1 - c)
        return int(dr + dc)

    start = min(endpoints, key=corner_distance)
    direction = (int(bend[0] - start[0]), int(bend[1] - start[1]))
    assert direction in {(0, 1), (0, -1), (1, 0), (-1, 0)}, "arrow long arm must be cardinal"
    return {
        "coords": coords,
        "start_top": start_top,
        "start_left": start_left,
        "direction": direction,
    }


def _extract_templates(grid: np.ndarray, bg: int, frames: list[dict]) -> dict[int, dict]:
    reserved = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        reserved[frame["cells"][:, 0], frame["cells"][:, 1]] = True
        r0, c0, r1, c1 = frame["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        arrow = _parse_arrow(interior, bg)
        frame["arrow"] = arrow
        abs_coords = arrow["coords"] + np.array([r0 + 1, c0 + 1], dtype=int)
        reserved[abs_coords[:, 0], abs_coords[:, 1]] = True

    comps = _components((grid != bg) & ~reserved)
    templates: dict[int, dict] = {}
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        values = [int(grid[r, c]) for r, c in comp]
        colors, counts = np.unique(np.array(values, dtype=int), return_counts=True)
        order = np.argsort(-counts)
        dominant = int(colors[order[0]])
        assert len(order) == 1 or int(counts[order[0]]) > int(counts[order[1]]), "template dominant color must be unique"
        assert dominant not in templates, "template dominant colors must be unique"
        templates[dominant] = {"pattern": sub, "coords": comp}
    assert templates, "expected at least one template"
    return templates


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int], start_top: bool, start_left: bool) -> np.ndarray:
    h, w = map(int, shape)
    ph, pw = map(int, pattern.shape)
    row_origin = 0 if start_top else h - 1
    col_origin = 0 if start_left else w - 1
    row_idx = (np.arange(h) - row_origin) % ph
    col_idx = (np.arange(w) - col_origin) % pw
    return pattern[row_idx[:, None], col_idx[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each colored frame with a repeated external stamp matched by dominant color.

    The grey L-shaped arrow in each frame marks a start corner and a principal direction.
    The final tiled crop is anchored at that corner; the source templates outside the frames
    are removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    templates = _extract_templates(g, bg, frames)

    out = g.copy()
    for template in templates.values():
        comp = template["coords"]
        out[comp[:, 0], comp[:, 1]] = bg

    for frame in frames:
        color = int(frame["color"])
        assert color in templates, "each frame color must have a matching template"
        pattern = templates[color]["pattern"]
        r0, c0, r1, c1 = frame["bbox"]
        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        arrow = frame["arrow"]
        filled = _tile_pattern(pattern, (interior_h, interior_w), arrow["start_top"], arrow["start_left"])
        out[r0 + 1 : r1, c0 + 1 : c1] = filled
    return out


def _sample_template(dominant: int, accent_pool: list[int]) -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    patch = np.full((h, w), dominant, dtype=int)
    area = h * w
    max_accent_count = min(3, area - 2)
    assert max_accent_count >= 1, "template must leave at least two dominant-color cells"
    for _ in range(100):
        patch[:, :] = dominant
        accent_count = random.randint(1, max_accent_count)
        positions = random.sample(range(area), accent_count)
        accent_colors = random.sample(accent_pool, k=random.randint(1, min(2, len(accent_pool))))
        flat = patch.reshape(-1)
        for idx, pos in enumerate(positions):
            flat[pos] = accent_colors[idx % len(accent_colors)]
        values, counts = np.unique(flat, return_counts=True)
        dominant_count = int(counts[np.where(values == dominant)[0][0]])
        others = [int(count) for value, count in zip(values, counts, strict=False) if int(value) != dominant]
        if all(dominant_count > count for count in others):
            return patch.copy()
    raise AssertionError("failed to sample a template with a unique dominant color")


def _sample_direction_and_corner() -> tuple[tuple[int, int], tuple[bool, bool]]:
    direction = random.choice(((0, 1), (0, -1), (1, 0), (-1, 0)))
    if direction == (0, 1):
        start_top = bool(random.getrandbits(1))
        start_left = True
    elif direction == (0, -1):
        start_top = bool(random.getrandbits(1))
        start_left = False
    elif direction == (1, 0):
        start_top = True
        start_left = bool(random.getrandbits(1))
    else:
        start_top = False
        start_left = bool(random.getrandbits(1))
    return direction, (start_top, start_left)


def _arrow_cells(h: int, w: int, direction: tuple[int, int], start_top: bool, start_left: bool) -> list[tuple[int, int]]:
    sr = 0 if start_top else h - 1
    sc = 0 if start_left else w - 1
    mr = sr + direction[0]
    mc = sc + direction[1]
    corner_target = (0 if start_top else h - 1, 0 if start_left else w - 1)

    def corner_distance(cell: tuple[int, int]) -> int:
        return int(abs(cell[0] - corner_target[0]) + abs(cell[1] - corner_target[1]))

    perp_options = [(1, 0), (-1, 0)] if direction[1] != 0 else [(0, 1), (0, -1)]
    tip = None
    for pr, pc in perp_options:
        candidate = (mr + pr, mc + pc)
        if not (0 <= candidate[0] < h and 0 <= candidate[1] < w):
            continue
        if corner_distance(candidate) > corner_distance((mr, mc)):
            tip = candidate
            break
    assert tip is not None, "arrow short arm must point inward"
    cells = [(sr, sc), (mr, mc), tip]
    assert all(0 <= r < h and 0 <= c < w for r, c in cells), "arrow must fit inside the frame interior"
    return cells


def _render_frame(color: int, template: np.ndarray) -> tuple[np.ndarray, dict]:
    inner_h = random.randint(4, 9)
    inner_w = random.randint(4, 9)
    direction, (start_top, start_left) = _sample_direction_and_corner()
    patch = np.full((inner_h + 2, inner_w + 2), BACKGROUND, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    arrow_cells = _arrow_cells(inner_h, inner_w, direction, start_top, start_left)
    for r, c in arrow_cells:
        patch[1 + r, 1 + c] = ARROW_COLOR
    info = {
        "color": color,
        "template": template,
        "inner_shape": (inner_h, inner_w),
        "start_top": start_top,
        "start_left": start_left,
    }
    return patch, info


def _place_patches(shape: tuple[int, int], patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    for idx in order:
        ph, pw = map(int, patches[idx].shape)
        if ph > h - 2 or pw > w - 2:
            return None
        row_choices = list(range(1, h - ph))
        col_choices = list(range(1, w - pw))
        random.shuffle(row_choices)
        random.shuffle(col_choices)
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.choice(row_choices)
            c0 = random.choice(col_choices)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _expected_output(inp: np.ndarray, template_positions: list[tuple[np.ndarray, tuple[int, int]]], frame_positions: list[tuple[dict, tuple[int, int]]]) -> np.ndarray:
    out = inp.copy()
    for template, (r0, c0) in template_positions:
        h, w = template.shape
        out[r0 : r0 + h, c0 : c0 + w] = BACKGROUND
    for info, (r0, c0) in frame_positions:
        inner_h, inner_w = info["inner_shape"]
        tiled = _tile_pattern(info["template"], (inner_h, inner_w), info["start_top"], info["start_left"])
        out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = tiled
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(20, 30)
    grid_w = random.randint(20, 30)
    n_frames = random.randint(1, 3)
    if n_frames > len(FRAME_COLOR_POOL):
        return None

    frame_colors = random.sample(list(FRAME_COLOR_POOL), n_frames)
    accent_pool = [color for color in FRAME_COLOR_POOL if color not in frame_colors and color != ARROW_COLOR]
    if len(accent_pool) < 2:
        return None

    templates = [_sample_template(color, accent_pool) for color in frame_colors]
    frame_patches: list[np.ndarray] = []
    frame_infos: list[dict] = []
    for color, template in zip(frame_colors, templates, strict=False):
        patch, info = _render_frame(color, template)
        frame_patches.append(patch)
        frame_infos.append(info)

    all_patches = templates + frame_patches
    positions = _place_patches((grid_h, grid_w), all_patches)
    if positions is None or len(positions) != len(all_patches):
        return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    template_positions: list[tuple[np.ndarray, tuple[int, int]]] = []
    frame_positions: list[tuple[dict, tuple[int, int]]] = []
    for idx, (patch, (r0, c0)) in enumerate(zip(all_patches, positions, strict=False)):
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        if idx < n_frames:
            template_positions.append((patch, (r0, c0)))
        else:
            frame_positions.append((frame_infos[idx - n_frames], (r0, c0)))

    if _most_common_color(grid) != BACKGROUND:
        return None

    expected = _expected_output(grid, template_positions, frame_positions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
