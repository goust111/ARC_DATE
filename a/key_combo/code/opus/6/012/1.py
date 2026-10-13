from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RAY = 2
ORANGE = 7
SPRITE_COLORS = (1, 3, 4, 5)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 400

SHAPES = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            count += 1
            q = deque([(r, c)])
            labels[r, c] = count
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = count
                        q.append((nr, nc))
    return labels, count


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    labels, n = _label_mask(mask)
    return [np.argwhere(labels == idx) for idx in range(1, n + 1)]


def _normalized(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _sprite_components(grid: np.ndarray) -> list[dict[str, object]]:
    sprites: list[dict[str, object]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in (BG, RAY)):
        for coords in _component_coords(grid == color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords.astype(int),
                    "top": int(rows.min()),
                    "bottom": int(rows.max()),
                    "left": int(cols.min()),
                    "right": int(cols.max()),
                    "cols": frozenset(int(c) for c in np.unique(cols)),
                    "shape": _normalized(coords),
                }
            )
    return sprites


def _region_parity(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    open_mask = grid != RAY
    labels, n_regions = _label_mask(open_mask)
    assert n_regions >= 3, "rays must split the grid into at least three regions"
    start = int(labels[0, 0])
    assert start > 0, "top-left corner must lie inside a region"

    graph: dict[int, set[int]] = {idx: set() for idx in range(1, n_regions + 1)}
    h, w = grid.shape
    ray_cells = np.argwhere(grid == RAY)
    assert len(ray_cells) > 0, "expected red ray pixels"
    for r0, c0 in ray_cells:
        r = int(r0)
        c = int(c0)
        if 0 < r < h - 1:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)
        if 0 < c < w - 1:
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)

    parity = {start: 0}
    q = deque([start])
    while q:
        region = q.popleft()
        for nxt in sorted(graph[region]):
            want = 1 - parity[region]
            if nxt in parity:
                assert parity[nxt] == want, "region graph must be bipartite"
            else:
                parity[nxt] = want
                q.append(nxt)
    assert len(parity) == n_regions, "all regions must be connected through ray adjacencies"
    return labels, parity


def _touches_ray(coords: np.ndarray, ray_mask: np.ndarray) -> bool:
    for r0, c0 in coords:
        r = int(r0)
        c = int(c0)
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(ray_mask.shape, nr, nc) and ray_mask[nr, nc]:
                return True
    return False


def _drop_sprite(
    coords: np.ndarray,
    direction: int,
    ray_mask: np.ndarray,
    occupied: np.ndarray,
) -> np.ndarray:
    moved = coords.copy()
    while True:
        candidate = moved.copy()
        candidate[:, 0] += direction
        rows = candidate[:, 0]
        cols = candidate[:, 1]
        if np.any(rows < 0) or np.any(rows >= ray_mask.shape[0]):
            break
        if np.any(ray_mask[rows, cols]) or np.any(occupied[rows, cols]):
            break
        moved = candidate
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the space with red edge rays, then apply parity-controlled vertical gravity and recolor sprites that land against a ray.

    Input structure:
    - The background is black, and red pixels form full-length horizontal and vertical rays that start from the grid edges.
    - Non-red foreground pixels are monochrome 4-connected sprites, each wholly inside one ray-bounded region.
    - The rays partition the non-ray cells into multiple 4-connected regions, with the top-left region serving as the parity anchor.

    Rule:
    - Keep every red ray pixel unchanged.
    - Build the region-adjacency graph across the rays, 2-color it from the region containing (0,0), and send parity-A sprites downward while parity-B sprites move upward.
    - Move each sprite as a rigid shape until its next step would hit a ray, the grid boundary, or an already stopped sprite.
    - After moving, recolor the whole sprite orange iff any of its pixels is 4-neighbor adjacent to a red ray; otherwise keep the sprite's original color.

    Key insight:
    - The red rays are not drawn paths to copy: they define a bipartite region map, and that hidden parity decides each sprite's gravity direction before the final ray-contact recolor is tested.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    ray_mask = g == RAY
    region_labels, parity = _region_parity(g)
    sprites = _sprite_components(g)
    assert sprites, "expected at least one sprite"

    planned: list[tuple[int, int, dict[str, object]]] = []
    for sprite in sprites:
        coords = sprite["coords"]
        region = int(region_labels[int(coords[0, 0]), int(coords[0, 1])])
        assert region > 0, "sprite must lie in a non-ray region"
        assert np.all(region_labels[coords[:, 0], coords[:, 1]] == region), "sprite may not cross ray boundaries"
        direction = 1 if parity[region] == 0 else -1
        measure = -int(sprite["bottom"]) if direction == 1 else int(sprite["top"])
        planned.append((direction, measure, sprite))

    planned.sort(key=lambda item: (item[0], item[1], int(item[2]["left"])))

    out = np.where(ray_mask, RAY, BG)
    occupied = np.zeros(g.shape, dtype=bool)
    for _direction, _measure, sprite in planned:
        direction = _direction
        coords = sprite["coords"]
        landed = _drop_sprite(coords, direction, ray_mask, occupied)
        color = ORANGE if _touches_ray(landed, ray_mask) else int(sprite["color"])
        out[landed[:, 0], landed[:, 1]] = color
        occupied[landed[:, 0], landed[:, 1]] = True
    return out


def _split_total(total: int, parts: int, min_size: int) -> list[int]:
    extra = total - parts * min_size
    assert extra >= 0
    cuts = sorted(random.sample(range(extra + parts - 1), parts - 1)) if parts > 1 else []
    widths: list[int] = []
    last = -1
    for cut in cuts + [extra + parts - 1]:
        widths.append(cut - last)
        last = cut
    return [min_size + w - 1 for w in widths]


def _line_positions(total: int, n_lines: int, min_band: int) -> tuple[list[int], list[tuple[int, int]]]:
    bands = _split_total(total - n_lines, n_lines + 1, min_band)
    positions: list[int] = []
    start = 0
    intervals: list[tuple[int, int]] = []
    for idx, band in enumerate(bands):
        intervals.append((start, start + band - 1))
        start += band
        if idx < n_lines:
            positions.append(start)
            start += 1
    return positions, intervals


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    return h, w


def _place_shape(shape: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _band_safe_cols(
    col_band: tuple[int, int],
    col_idx: int,
    n_bands: int,
) -> tuple[int, int]:
    c0, c1 = col_band
    left = c0 + (1 if col_idx > 0 else 0)
    right = c1 - (1 if col_idx < n_bands - 1 else 0)
    return left, right


def _top_candidates(
    row_band: tuple[int, int],
    row_idx: int,
    n_bands: int,
    height: int,
    direction: int,
    must_touch: bool,
) -> tuple[int, int]:
    r0, r1 = row_band
    if direction == 1:
        final_bottom = r1 if must_touch else r1
        final_top = final_bottom - height + 1
        safe_top = r0 + 1
        return final_top, final_top - safe_top
    final_top = 0 if (row_idx == 0 and not must_touch) else r0
    final_bottom = final_top + height - 1
    safe_bottom = r1 - 1
    return final_top, safe_bottom - final_bottom


def _behavior_region_candidates(
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    parity: int,
    must_touch: bool,
) -> dict[str, list[tuple[int, int]]]:
    top_row = 0
    bottom_row = len(row_bands) - 1
    top: list[tuple[int, int]] = []
    bottom: list[tuple[int, int]] = []
    for i in range(len(row_bands)):
        for j in range(len(col_bands)):
            if (i + j) % 2 != parity:
                continue
            if i == top_row:
                if parity == 0 and must_touch:
                    top.append((i, j))
                if parity == 1 and not must_touch:
                    top.append((i, j))
            if i == bottom_row:
                if parity == 0 and not must_touch:
                    bottom.append((i, j))
                if parity == 1 and must_touch:
                    bottom.append((i, j))
    return {"top": top, "bottom": bottom}


def _pick_shape(max_width: int) -> tuple[tuple[int, int], ...]:
    choices = [shape for shape in SHAPES if _shape_dims(shape)[1] <= max_width]
    return random.choice(choices)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(20, 24)
    w = random.randint(20, 24)
    n_h = random.randint(1, 2)
    n_v = random.randint(1, 2)
    h_lines, row_bands = _line_positions(h, n_h, min_band=6)
    v_lines, col_bands = _line_positions(w, n_v, min_band=6)

    grid_in = np.zeros((h, w), dtype=int)
    for r in h_lines:
        grid_in[r, :] = RAY
    for c in v_lines:
        grid_in[:, c] = RAY

    behaviors = [
        {"color": 1, "direction": 1, "touch": True},
        {"color": 3, "direction": -1, "touch": False},
        {"color": 4, "direction": 1, "touch": False},
        {"color": 5, "direction": -1, "touch": True},
    ]
    random.shuffle(behaviors)

    options: list[tuple[int, int]] = []
    bottom_idx = len(row_bands) - 1
    even_top = [j for j in range(len(col_bands)) if j % 2 == 0]
    odd_top = [j for j in range(len(col_bands)) if j % 2 == 1]
    even_bottom = [j for j in range(len(col_bands)) if (bottom_idx + j) % 2 == 0]
    odd_bottom = [j for j in range(len(col_bands)) if (bottom_idx + j) % 2 == 1]
    if not (even_top and odd_top and even_bottom and odd_bottom):
        return None
    options.append((0, random.choice(even_top)))
    options.append((0, random.choice(odd_top)))
    options.append((bottom_idx, random.choice(even_bottom)))
    options.append((bottom_idx, random.choice(odd_bottom)))

    band_counts = {j: 0 for j in range(len(col_bands))}
    for _, j in options:
        band_counts[j] += 1

    used_cols_by_band: dict[int, list[tuple[int, int]]] = {j: [] for j in range(len(col_bands))}
    placed_input: list[tuple[np.ndarray, int]] = []
    placed_output: list[tuple[np.ndarray, int]] = []

    for spec, (row_idx, col_idx) in zip(behaviors, options):
        band_left, band_right = _band_safe_cols(col_bands[col_idx], col_idx, len(col_bands))
        capacity = band_right - band_left + 1
        max_width = 2 if band_counts[col_idx] >= 2 else 3
        max_width = min(max_width, capacity)
        if max_width < 1:
            return None
        shape = _pick_shape(max_width=max_width)
        shape_h, shape_w = _shape_dims(shape)
        top_final, max_shift = _top_candidates(
            row_bands[row_idx],
            row_idx,
            len(row_bands),
            shape_h,
            int(spec["direction"]),
            bool(spec["touch"]),
        )
        if max_shift < 1:
            return None

        starts: list[int] = []
        for left in range(band_left, band_right - shape_w + 2):
            interval = (left, left + shape_w - 1)
            if all(interval[1] < a or b < interval[0] for a, b in used_cols_by_band[col_idx]):
                starts.append(left)
        if not starts:
            return None
        left = random.choice(starts)
        used_cols_by_band[col_idx].append((left, left + shape_w - 1))

        final_coords = _place_shape(shape, top_final, left)
        shift = random.randint(1, max_shift)
        init_top = top_final - shift if int(spec["direction"]) == 1 else top_final + shift
        init_coords = _place_shape(shape, init_top, left)
        placed_input.append((init_coords, int(spec["color"])))

        touch = _touches_ray(final_coords, grid_in == RAY)
        if touch != bool(spec["touch"]):
            return None
        out_color = ORANGE if touch else int(spec["color"])
        placed_output.append((final_coords, out_color))

    for coords, color in placed_input:
        grid_in[coords[:, 0], coords[:, 1]] = color

    grid_out = np.where(grid_in == RAY, RAY, BG)
    for coords, color in placed_output:
        if np.any(grid_out[coords[:, 0], coords[:, 1]] != BG):
            return None
        grid_out[coords[:, 0], coords[:, 1]] = color

    return grid_in, grid_out


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    ray_mask = inp == RAY
    ray_rows = [r for r in range(inp.shape[0]) if np.all(inp[r, :] == RAY)]
    ray_cols = [c for c in range(inp.shape[1]) if np.all(inp[:, c] == RAY)]
    if not (1 <= len(ray_rows) <= 2 and 1 <= len(ray_cols) <= 2):
        return False

    try:
        labels, parity = _region_parity(inp)
    except AssertionError:
        return False

    sprites = _sprite_components(inp)
    if len(sprites) != 4 or len({int(s["color"]) for s in sprites}) != 4:
        return False

    used_columns: set[int] = set()
    for sprite in sprites:
        coords = sprite["coords"]
        region = int(labels[int(coords[0, 0]), int(coords[0, 1])])
        if region <= 0 or not np.all(labels[coords[:, 0], coords[:, 1]] == region):
            return False
        if _touches_ray(coords, ray_mask):
            return False
        cols = set(int(c) for c in sprite["cols"])
        if used_columns & cols:
            return False
        used_columns |= cols

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    out_sprites = _sprite_components(solved)
    if len(out_sprites) != 4:
        return False
    matched: set[int] = set()
    saw_orange = False
    saw_original = False
    for sprite in sprites:
        region = int(labels[int(sprite["coords"][0, 0]), int(sprite["coords"][0, 1])])
        direction = 1 if parity[region] == 0 else -1
        candidates = []
        for idx, out_sprite in enumerate(out_sprites):
            if idx in matched:
                continue
            if sprite["cols"] == out_sprite["cols"] and sprite["shape"] == out_sprite["shape"]:
                candidates.append((idx, out_sprite))
        if len(candidates) != 1:
            return False
        idx, out_sprite = candidates[0]
        matched.add(idx)
        delta = int(out_sprite["top"]) - int(sprite["top"])
        if direction == 1 and delta <= 0:
            return False
        if direction == -1 and delta >= 0:
            return False
        landed = out_sprite["coords"]
        touch = _touches_ray(landed, ray_mask)
        if touch:
            if int(out_sprite["color"]) != ORANGE:
                return False
            saw_orange = True
        else:
            if int(out_sprite["color"]) != int(sprite["color"]):
                return False
            saw_original = True
    return saw_orange and saw_original


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if _validate_case(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid ray-partition gravity input")
