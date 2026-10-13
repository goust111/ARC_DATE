from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
PAIR_COUNT_CHOICES = (2, 3, 4)
FRAGMENT_SPAN_MAX = {2: 6, 3: 5, 4: 4}
SLACK_CHOICES = {2: (0, 1, 2), 3: (0, 1, 2), 4: (0, 1)}
GRID_RANGE = {2: (18, 28), 3: (20, 30), 4: (22, 30)}
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 150
GENERATION_RETRIES = 500
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        visited[rr, cc] = True
        queue: deque[tuple[int, int]] = deque([(rr, cc)])
        points: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            points.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        components.append(points)
    return components


def _detect_frame(grid: np.ndarray, points: list[tuple[int, int]]) -> dict[str, object] | None:
    colors = {int(grid[r, c]) for r, c in points}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    if color == BG:
        return None

    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    rel_mask = np.zeros((h, w), dtype=bool)
    for r, c in points:
        rel_mask[r - r0, c - c0] = True

    frame_mask = np.zeros((h, w), dtype=bool)
    frame_mask[0, :] = True
    frame_mask[-1, :] = True
    frame_mask[:, 0] = True
    frame_mask[:, -1] = True
    if not np.array_equal(rel_mask, frame_mask):
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub[1:-1, 1:-1] != BG):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "anchor": (r0 + 1, c0 + 1),
        "interior_shape": (h - 2, w - 2),
    }


def _parse_fragment(
    grid: np.ndarray,
    points: list[tuple[int, int]],
    template_colors: set[int],
) -> dict[str, object]:
    counts = Counter(int(grid[r, c]) for r, c in points)
    assert len(counts) == 2, "each fragment must use one body color and one locator color"

    singleton_colors = [color for color, count in counts.items() if count == 1]
    assert singleton_colors, "fragment must contain a unique locator pixel"
    locator_matches = [color for color in singleton_colors if color in template_colors]
    assert len(locator_matches) == 1, "locator color must uniquely match one template color"
    locator_color = locator_matches[0]

    body_colors = [color for color in counts if color != locator_color]
    assert len(body_colors) == 1, "fragment body color must be unique"
    body_color = body_colors[0]

    locator = next((r, c) for r, c in points if int(grid[r, c]) == locator_color)
    cells: list[tuple[int, int, int]] = []
    for r, c in points:
        dr = r - locator[0]
        dc = c - locator[1]
        assert dr >= 0 and dc >= 0, "locator pixel must be the fragment anchor"
        value = locator_color if (r, c) == locator else int(grid[r, c])
        cells.append((dr, dc, value))

    return {
        "locator_color": locator_color,
        "body_color": body_color,
        "locator": locator,
        "points": points,
        "cells": cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move each two-color fragment into the hollow template selected by its locator color.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular hollow templates appear in distinct border colors.
    - Separate 4-connected fragments sit elsewhere; each fragment uses one body color plus exactly one locator pixel in a different color.

    Rule:
    - Detect each fragment's unique locator pixel color and pair the fragment with the template whose border has that same color.
    - Clear the fragment from its original position.
    - Paste the fragment into the matched template so the locator pixel aligns with the template interior's top-left cell.
    - Keep the template border unchanged and set the pasted locator pixel to the template border color while leaving the fragment body color unchanged.

    Key insight:
    - The locator pixel is both the color-based pairing key and the placement anchor inside the matching template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _connected_components(g != BG)
    assert components, "grid must contain non-background objects"

    frames: list[dict[str, object]] = []
    fragment_points: list[list[tuple[int, int]]] = []
    for points in components:
        frame = _detect_frame(g, points)
        if frame is None:
            fragment_points.append(points)
        else:
            frames.append(frame)

    assert frames, "expected at least one hollow template"
    assert fragment_points, "expected at least one fragment"

    template_by_color: dict[int, dict[str, object]] = {}
    for frame in frames:
        color = int(frame["color"])
        assert color not in template_by_color, "template border colors must be unique"
        template_by_color[color] = frame

    template_colors = set(template_by_color)
    fragments: dict[int, dict[str, object]] = {}
    for points in fragment_points:
        fragment = _parse_fragment(g, points, template_colors)
        locator_color = int(fragment["locator_color"])
        assert locator_color not in fragments, "each template should match exactly one fragment"
        fragments[locator_color] = fragment

    assert set(fragments) == template_colors, "templates and fragments must pair one-to-one"

    out = g.copy()
    for fragment in fragments.values():
        for r, c in fragment["points"]:
            out[r, c] = BG

    for color, frame in template_by_color.items():
        fragment = fragments[color]
        anchor_r, anchor_c = frame["anchor"]
        inner_h, inner_w = frame["interior_shape"]
        for dr, dc, value in fragment["cells"]:
            assert 0 <= dr < inner_h and 0 <= dc < inner_w, "fragment must fit inside the template interior"
            out[anchor_r + dr, anchor_c + dc] = color if (dr, dc) == (0, 0) else value

    return out


def _grow_fragment_cells(rng: random.Random, max_h: int, max_w: int) -> list[tuple[int, int]]:
    target = rng.randint(3, max_h * max_w)
    cells = {(0, 0)}
    starters = []
    if max_w > 1:
        starters.append((0, 1))
    if max_h > 1:
        starters.append((1, 0))
    cells.add(rng.choice(starters))

    while len(cells) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < max_h and 0 <= nc < max_w and nr >= 0 and nc >= 0 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    max_row = max(rows)
    max_col = max(cols)
    return sorted((r, c) for r, c in cells if r <= max_row and c <= max_col)


def _build_pair_specs(rng: random.Random, pair_count: int) -> list[dict[str, object]]:
    colors = rng.sample(COLOR_POOL, 2 * pair_count)
    border_colors = colors[:pair_count]
    body_colors = colors[pair_count:]

    specs: list[dict[str, object]] = []
    max_span = FRAGMENT_SPAN_MAX[pair_count]
    slack_choices = SLACK_CHOICES[pair_count]

    for border_color, body_color in zip(border_colors, body_colors):
        raw_h = rng.randint(2, max_span)
        raw_w = rng.randint(2, max_span)
        cells = _grow_fragment_cells(rng, raw_h, raw_w)
        frag_h = max(r for r, _ in cells) + 1
        frag_w = max(c for _, c in cells) + 1
        inner_h = frag_h + rng.choice(slack_choices)
        inner_w = frag_w + rng.choice(slack_choices)

        specs.append(
            {
                "border_color": border_color,
                "body_color": body_color,
                "fragment_cells": cells,
                "frag_shape": (frag_h, frag_w),
                "frame_shape": (inner_h + 2, inner_w + 2),
                "interior_shape": (inner_h, inner_w),
            }
        )
    return specs


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    h_total, w_total = occupied.shape
    r0 = max(0, r - PLACEMENT_PADDING)
    c0 = max(0, c - PLACEMENT_PADDING)
    r1 = min(h_total, r + h + PLACEMENT_PADDING)
    c1 = min(w_total, c + w + PLACEMENT_PADDING)
    return not np.any(occupied[r0:r1, c0:c1])


def _mark_occupied(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    h_total, w_total = occupied.shape
    r0 = max(0, r - PLACEMENT_PADDING)
    c0 = max(0, c - PLACEMENT_PADDING)
    r1 = min(h_total, r + h + PLACEMENT_PADDING)
    c1 = min(w_total, c + w + PLACEMENT_PADDING)
    occupied[r0:r1, c0:c1] = True


def _place_objects(
    rng: random.Random,
    grid_shape: tuple[int, int],
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    grid_h, grid_w = grid_shape
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict[str, object]] = []

    objects: list[tuple[str, int, int, int, dict[str, object]]] = []
    for idx, spec in enumerate(specs):
        frame_h, frame_w = spec["frame_shape"]
        frag_h, frag_w = spec["frag_shape"]
        objects.append(("frame", idx, frame_h, frame_w, spec))
        objects.append(("fragment", idx, frag_h, frag_w, spec))
    objects.sort(key=lambda item: item[2] * item[3], reverse=True)

    placements: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, obj_h, obj_w, _spec in objects:
        if obj_h > grid_h or obj_w > grid_w:
            return None
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r = rng.randint(0, grid_h - obj_h)
            c = rng.randint(0, grid_w - obj_w)
            if _can_place(occupied, r, c, obj_h, obj_w):
                _mark_occupied(occupied, r, c, obj_h, obj_w)
                placements[(kind, idx)] = (r, c)
                success = True
                break
        if not success:
            return None

    for idx, spec in enumerate(specs):
        frame_pos = placements[("frame", idx)]
        fragment_pos = placements[("fragment", idx)]
        placed_spec = dict(spec)
        placed_spec["frame_pos"] = frame_pos
        placed_spec["fragment_pos"] = fragment_pos
        placed.append(placed_spec)
    return placed


def _render_input(grid_shape: tuple[int, int], placed_specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)

    for spec in placed_specs:
        frame_r, frame_c = spec["frame_pos"]
        frame_h, frame_w = spec["frame_shape"]
        border_color = int(spec["border_color"])
        grid[frame_r, frame_c : frame_c + frame_w] = border_color
        grid[frame_r + frame_h - 1, frame_c : frame_c + frame_w] = border_color
        grid[frame_r : frame_r + frame_h, frame_c] = border_color
        grid[frame_r : frame_r + frame_h, frame_c + frame_w - 1] = border_color

        frag_r, frag_c = spec["fragment_pos"]
        body_color = int(spec["body_color"])
        for r, c in spec["fragment_cells"]:
            grid[frag_r + r, frag_c + c] = body_color
        grid[frag_r, frag_c] = border_color

    return grid


def _render_expected(grid: np.ndarray, placed_specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in placed_specs:
        frag_r, frag_c = spec["fragment_pos"]
        for r, c in spec["fragment_cells"]:
            out[frag_r + r, frag_c + c] = BG

    for spec in placed_specs:
        frame_r, frame_c = spec["frame_pos"]
        body_color = int(spec["body_color"])
        border_color = int(spec["border_color"])
        anchor_r = frame_r + 1
        anchor_c = frame_c + 1
        for r, c in spec["fragment_cells"]:
            out[anchor_r + r, anchor_c + c] = body_color
        out[anchor_r, anchor_c] = border_color

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    low, high = GRID_RANGE[pair_count]
    grid_shape = (rng.randint(low, high), rng.randint(low, high))
    specs = _build_pair_specs(rng, pair_count)
    placed_specs = _place_objects(rng, grid_shape, specs)
    if placed_specs is None:
        return None

    grid = _render_input(grid_shape, placed_specs)
    expected = _render_expected(grid, placed_specs)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
