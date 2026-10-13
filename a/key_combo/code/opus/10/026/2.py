from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DOWN = (1, 0)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_of_pixels(pixels: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pixels]
    cols = [c for _, c in pixels]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_pixels(pixels: set[tuple[int, int]]) -> bool:
    if not pixels:
        return False
    start = next(iter(pixels))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pixels and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(pixels)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pixels = {(r, c)}
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
                        pixels.add((nr, nc))
            comps.append({"color": color, "pixels": pixels, "bbox": _bbox_of_pixels(pixels)})
    return comps


def _normalize_pixels(pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox_of_pixels(pixels)
    return {(r - r0, c - c0) for r, c in pixels}


def _mask_from_pixels(pixels: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_pixels(pixels)
    h = 1 + max(r for r, _ in norm)
    w = 1 + max(c for _, c in norm)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _is_rectangle(pixels: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox_of_pixels(pixels)
    return len(pixels) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _shape_has_nontrivial_symmetry(pixels: set[tuple[int, int]]) -> bool:
    mask = _mask_from_pixels(pixels)
    variants = [
        np.rot90(mask, k) for k in range(4)
    ] + [
        np.fliplr(np.rot90(mask, k)) for k in range(4)
    ]
    for idx, other in enumerate(variants):
        if idx == 0:
            continue
        if other.shape == mask.shape and np.array_equal(other, mask):
            return True
    return False


def _same_color_neighbors(pixel: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = pixel
    return [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in pixels]


def _perpendicular_neighbors(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    components = _components_by_color(g, bg)
    singletons = {next(iter(comp["pixels"])): int(comp["color"]) for comp in components if len(comp["pixels"]) == 1}
    multi = [comp for comp in components if len(comp["pixels"]) > 1]

    objects = []
    for comp in multi:
        color = int(comp["color"])
        pixels = set(comp["pixels"])
        marker_hits = []
        for pixel in pixels:
            for nr, nc in _neighbors4(*pixel):
                if (nr, nc) in singletons and int(singletons[(nr, nc)]) != color:
                    marker_hits.append((pixel, (nr, nc), int(singletons[(nr, nc)])))

        candidates = []
        seen_keys = set()
        for endpoint, marker_coord, marker_color in marker_hits:
            neighbors = _same_color_neighbors(endpoint, pixels)
            if len(neighbors) != 1:
                continue
            prev = neighbors[0]
            back = (prev[0] - endpoint[0], prev[1] - endpoint[1])
            direction = (-back[0], -back[1])
            segment = [endpoint]
            cur = endpoint
            attach = None
            while True:
                nxt = (cur[0] + back[0], cur[1] + back[1])
                if nxt not in pixels:
                    attach = None
                    break
                nxt_neighbors = _same_color_neighbors(nxt, pixels)
                expected = {cur, (nxt[0] + back[0], nxt[1] + back[1])}
                extra = [cell for cell in nxt_neighbors if cell not in expected]
                if extra or len(nxt_neighbors) >= 3:
                    attach = nxt
                    break
                segment.append(nxt)
                cur = nxt

            if attach is None:
                continue

            ok = True
            for idx, cell in enumerate(segment):
                expected = set()
                if idx > 0:
                    expected.add(segment[idx - 1])
                if idx + 1 < len(segment):
                    expected.add(segment[idx + 1])
                else:
                    expected.add(attach)
                if set(_same_color_neighbors(cell, pixels)) != expected:
                    ok = False
                    break
            if not ok:
                continue

            stem_pixels = set(segment)
            body_pixels = pixels - stem_pixels
            if len(segment) < 2 or not body_pixels or not _connected_pixels(body_pixels):
                continue

            body_bbox = _bbox_of_pixels(body_pixels)
            r0, c0, r1, c1 = body_bbox
            local_offsets = sorted((r - r0, c - c0) for r, c in body_pixels)
            key = (tuple(sorted(local_offsets)), direction, len(segment), marker_color)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            candidates.append(
                {
                    "body_color": color,
                    "marker_color": marker_color,
                    "direction": direction,
                    "period": len(segment),
                    "body_pixels": body_pixels,
                    "stem_pixels": stem_pixels,
                    "marker_coord": marker_coord,
                    "body_bbox": body_bbox,
                    "body_offsets": local_offsets,
                    "body_shape": (r1 - r0 + 1, c1 - c0 + 1),
                }
            )

        assert len(candidates) == 1, f"expected exactly one valid body/stem split for color {color}"
        objects.append(candidates[0])

    objects.sort(key=lambda obj: (obj["body_bbox"][0], obj["body_bbox"][1]))
    return objects


def _patch_intersects_grid(top: int, left: int, shape: tuple[int, int], grid_shape: tuple[int, int]) -> bool:
    h, w = shape
    gh, gw = grid_shape
    return top < gh and left < gw and top + h > 0 and left + w > 0


def _stride_for(obj: dict, direction: tuple[int, int]) -> int:
    h, w = obj["body_shape"]
    return h + 1 if direction[0] != 0 else w + 1


def _sequence_tops(obj: dict, direction: tuple[int, int] | None = None, period: int | None = None):
    del period
    use_direction = obj["direction"] if direction is None else direction
    r0, c0, _r1, _c1 = obj["body_bbox"]
    stride = _stride_for(obj, use_direction)
    dr, dc = use_direction
    k = 0
    while True:
        top = r0 + k * dr * stride
        left = c0 + k * dc * stride
        yield k + 1, top, left
        k += 1


def _render_from_specs(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj in sorted(specs, key=lambda item: (item["body_bbox"][0], item["body_bbox"][1])):
        period = int(obj["period"])
        for index, top, left in _sequence_tops(obj):
            if not _patch_intersects_grid(top, left, obj["body_shape"], shape):
                break
            color = int(obj["marker_color"]) if index % period == 0 else int(obj["body_color"])
            for dr0, dc0 in obj["body_offsets"]:
                rr = top + dr0
                cc = left + dc0
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    out[rr, cc] = color
    return out


def _visible_cells(obj: dict, shape: tuple[int, int], direction: tuple[int, int] | None = None) -> tuple[set[tuple[int, int]], list[int]]:
    use_direction = obj["direction"] if direction is None else direction
    cells = set()
    visible_indices = []
    for index, top, left in _sequence_tops(obj, direction=use_direction):
        if not _patch_intersects_grid(top, left, obj["body_shape"], shape):
            break
        visible_indices.append(index)
        for dr0, dc0 in obj["body_offsets"]:
            rr = top + dr0
            cc = left + dc0
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                cells.add((rr, cc))
    return cells, visible_indices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each sprite's encoded connector, then tile the sprite along that connector direction while recoloring every L-th element by the connector's terminal marker color.

    Input structure:
    - A black background contains 2 or 3 monochrome sprite objects, each 4-connected and made of an irregular body plus a same-color one-cell-wide straight connector stem.
    - Each stem ends next to a single-pixel marker of a different color, and each marker color is unique.
    - The stem direction is one of the four cardinal directions, and the stem length is between 2 and 4 cells.

    Rule:
    - For each monochrome object, detect the straight stem that protrudes from the body and read its direction and length L.
    - Read the color of the single-pixel marker adjacent to the stem's far end.
    - Erase the stem and marker, keep the original body as element 1, then place further body copies along the stem direction.
    - The copy stride is the body's bbox size along that direction plus one blank cell, copies are clipped at the grid edge, and every element whose 1-based index is a multiple of L is recolored to the marker color.
    - If tiled sequences overlap, later objects in row-major bbox order overwrite earlier ones.

    Key insight:
    - One geometric accessory encodes three independent parameters at once: the stem direction says where to tile, the stem length says the recolor period, and the terminal marker says which color appears on each period hit.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    specs = _extract_objects(g)
    return _render_from_specs(specs, g.shape, bg)


def _random_shape() -> set[tuple[int, int]]:
    for _ in range(200):
        target = random.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if -3 <= nr <= 3 and -3 <= nc <= 3 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))
        cells = _normalize_pixels(cells)
        r0, c0, r1, c1 = _bbox_of_pixels(cells)
        if max(r1 - r0 + 1, c1 - c0 + 1) > 4:
            continue
        if _is_rectangle(cells):
            continue
        if _shape_has_nontrivial_symmetry(cells):
            continue
        return cells
    raise RuntimeError("failed to sample an asymmetric non-rectangular sprite body")


def _clear_stem_corridor(
    body: set[tuple[int, int]],
    attach: tuple[int, int],
    direction: tuple[int, int],
    period: int,
) -> bool:
    perp_a, perp_b = _perpendicular_neighbors(direction)
    for step in range(1, period + 1):
        cell = (attach[0] + direction[0] * step, attach[1] + direction[1] * step)
        if cell in body:
            return False
        if (cell[0] + perp_a[0], cell[1] + perp_a[1]) in body:
            return False
        if (cell[0] + perp_b[0], cell[1] + perp_b[1]) in body:
            return False
    return True


def _attachment_candidates(
    body: set[tuple[int, int]],
    direction: tuple[int, int],
    period: int,
) -> list[tuple[int, int]]:
    perp_a, perp_b = _perpendicular_neighbors(direction)
    candidates = []
    for r, c in body:
        out = (r + direction[0], c + direction[1])
        if out in body:
            continue
        if (r + perp_a[0], c + perp_a[1]) in body or (r + perp_b[0], c + perp_b[1]) in body:
            attach = (r, c)
            if _clear_stem_corridor(body, attach, direction, period):
                candidates.append(attach)
    return candidates


def _choose_body_and_attach(
    direction: tuple[int, int],
    period: int,
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        body = _random_shape()
        candidates = _attachment_candidates(body, direction, period)
        if candidates:
            return body, random.choice(candidates)
    raise RuntimeError("failed to find a body with a unique-looking stem attachment")


def _object_input_cells(
    body: set[tuple[int, int]],
    attach_local: tuple[int, int],
    top_left: tuple[int, int],
    direction: tuple[int, int],
    period: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    r0, c0 = top_left
    body_global = {(r0 + r, c0 + c) for r, c in body}
    ar = r0 + attach_local[0]
    ac = c0 + attach_local[1]
    stem = {(ar + direction[0] * step, ac + direction[1] * step) for step in range(1, period + 1)}
    marker = (ar + direction[0] * (period + 1), ac + direction[1] * (period + 1))
    return body_global, stem, marker


def _visible_sequences_for_layout(
    body: set[tuple[int, int]],
    top_left: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], list[int], tuple[int, int]]:
    r0, c0 = top_left
    br0, bc0, br1, bc1 = _bbox_of_pixels(body)
    body_offsets = sorted((r - br0, c - bc0) for r, c in body)
    body_shape = (br1 - br0 + 1, bc1 - bc0 + 1)
    stride = body_shape[0] + 1 if direction[0] != 0 else body_shape[1] + 1
    gh, gw = shape
    cells = set()
    indices = []
    k = 0
    while True:
        top = r0 + k * direction[0] * stride
        left = c0 + k * direction[1] * stride
        if not _patch_intersects_grid(top, left, body_shape, shape):
            break
        indices.append(k + 1)
        for dr0, dc0 in body_offsets:
            rr = top + dr0
            cc = left + dc0
            if 0 <= rr < gh and 0 <= cc < gw:
                cells.add((rr, cc))
        k += 1
    return cells, indices, body_shape


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    gh, gw = shape
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < gh and 0 <= nc < gw:
                    out.add((nr, nc))
    return out


def _build_input_from_layout(layouts: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for item in layouts:
        for r, c in item["body_global"] | item["stem"]:
            grid[r, c] = int(item["body_color"])
        mr, mc = item["marker"]
        grid[mr, mc] = int(item["marker_color"])
    return grid


def _mutate_spec(obj: dict, **updates) -> dict:
    mutated = dict(obj)
    mutated.update(updates)
    return mutated


def _semantic_checks(grid: np.ndarray) -> bool:
    specs = _extract_objects(grid)
    if not (2 <= len(specs) <= 3):
        return False
    if all(tuple(obj["direction"]) == DOWN for obj in specs):
        return False
    if len({obj["body_color"] for obj in specs}) != len(specs):
        return False
    if len({obj["marker_color"] for obj in specs}) != len(specs):
        return False

    bg = _get_bg(grid)
    baseline = _render_from_specs(specs, grid.shape, bg)
    if not np.array_equal(baseline, main(grid)):
        return False

    occupied = set()
    for obj in specs:
        cells, visible_indices = _visible_cells(obj, grid.shape)
        if any(cell in occupied for cell in cells):
            return False
        occupied |= cells
        if not any(index % int(obj["period"]) == 0 for index in visible_indices):
            return False

        alt_periods = [value for value in (2, 3, 4) if value != int(obj["period"])]
        if not any(
            np.any(
                _render_from_specs(
                    [_mutate_spec(spec, period=value) if spec is obj else spec for spec in specs],
                    grid.shape,
                    bg,
                )
                != baseline
            )
            for value in alt_periods
        ):
            return False

        other_dirs = [direction for direction in DIRS4 if direction != tuple(obj["direction"])]
        if not any(
            np.any(
                _render_from_specs(
                    [_mutate_spec(spec, direction=direction) if spec is obj else spec for spec in specs],
                    grid.shape,
                    bg,
                )
                != baseline
            )
            for direction in other_dirs
        ):
            return False

        if not np.any(
            _render_from_specs(
                [_mutate_spec(spec, marker_color=int(obj["body_color"])) if spec is obj else spec for spec in specs],
                grid.shape,
                bg,
            )
            != baseline
        ):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    count = random.choice((2, 3))
    directions = random.sample(DIRS4, count)
    if all(direction == DOWN for direction in directions):
        return None

    palette = random.sample(range(1, 10), 2 * count)
    body_colors = palette[:count]
    marker_colors = palette[count:]
    periods = [random.randint(2, 4) for _ in range(count)]
    if count == 3 and len(set(periods)) == 1:
        periods[-1] = 2 if periods[-1] != 2 else 3

    input_forbidden = set()
    output_occupied = set()
    layouts = []

    for idx in range(count):
        period = periods[idx]
        body, attach_local = _choose_body_and_attach(directions[idx], period)
        candidates = []
        body_bbox = _bbox_of_pixels(body)
        body_h = body_bbox[2] - body_bbox[0] + 1
        body_w = body_bbox[3] - body_bbox[1] + 1

        for top in range(0, shape[0] - body_h + 1):
            for left in range(0, shape[1] - body_w + 1):
                body_global, stem, marker = _object_input_cells(body, attach_local, (top, left), directions[idx], period)
                all_input = set(body_global) | set(stem) | {marker}
                if not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in all_input):
                    continue
                if any(cell in input_forbidden for cell in all_input):
                    continue

                output_cells, visible_indices, body_shape = _visible_sequences_for_layout(body, (top, left), directions[idx], shape)
                if any(cell in output_occupied for cell in output_cells):
                    continue
                if not any(index % period == 0 for index in visible_indices):
                    continue

                candidates.append(
                    {
                        "body_color": body_colors[idx],
                        "marker_color": marker_colors[idx],
                        "period": period,
                        "direction": directions[idx],
                        "body": body,
                        "body_global": body_global,
                        "body_bbox": (top, left, top + body_shape[0] - 1, left + body_shape[1] - 1),
                        "body_offsets": sorted(body),
                        "body_shape": body_shape,
                        "stem": stem,
                        "marker": marker,
                        "output_cells": output_cells,
                    }
                )

        if not candidates:
            return None

        choice = random.choice(candidates)
        layouts.append(choice)
        input_forbidden |= _expanded(choice["body_global"] | choice["stem"] | {choice["marker"]}, shape)
        output_occupied |= choice["output_cells"]

    grid = _build_input_from_layout(layouts, shape)
    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector-coded periodic tiling instance")
