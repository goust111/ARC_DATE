from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ORANGE = 7
FRAME_COLORS = tuple(c for c in range(10) if c not in (BG, ORANGE))
MAX_MARKERS = 5
GEN_RETRIES = 400


def _border_area(height: int, width: int) -> int:
    return 2 * (int(height) + int(width)) - 4


def _draw_rect_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _square_bbox(center_r2: int, center_c2: int, radius: int) -> tuple[int, int, int, int]:
    top = (int(center_r2) - 2 * int(radius) + 1) // 2
    left = (int(center_c2) - 2 * int(radius) + 1) // 2
    bottom = (int(center_r2) + 2 * int(radius)) // 2
    right = (int(center_c2) + 2 * int(radius)) // 2
    return int(top), int(left), int(bottom), int(right)


def _draw_square_ring(grid: np.ndarray, center_r2: int, center_c2: int, radius: int, color: int) -> None:
    height, width = map(int, grid.shape)
    top, left, bottom, right = _square_bbox(center_r2, center_c2, radius)
    if bottom < 0 or right < 0 or top >= height or left >= width:
        return

    row0 = max(0, top)
    row1 = min(height - 1, bottom)
    col0 = max(0, left)
    col1 = min(width - 1, right)
    if row0 > row1 or col0 > col1:
        return

    if 0 <= top < height:
        grid[top, col0 : col1 + 1] = int(color)
    if 0 <= bottom < height:
        grid[bottom, col0 : col1 + 1] = int(color)
    if 0 <= left < width:
        grid[row0 : row1 + 1, left] = int(color)
    if 0 <= right < width:
        grid[row0 : row1 + 1, right] = int(color)


def _ring_mask(height: int, width: int, center_r2: int, center_c2: int, start_radius: int, layers: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for radius in range(int(start_radius), int(start_radius) + int(layers)):
        _draw_square_ring(mask, center_r2, center_c2, radius, True)
    return mask


def _render_group_output(
    grid: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    colors_outer_to_inner: list[int],
    marker_count: int,
) -> None:
    r0, c0, r1, c1 = map(int, outer_bbox)
    center_r2 = int(r0 + r1)
    center_c2 = int(c0 + c1)
    for offset, color in enumerate(reversed(colors_outer_to_inner)):
        _draw_square_ring(grid, center_r2, center_c2, int(marker_count) + offset, int(color))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(grid: np.ndarray) -> list[dict[str, object]]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict[str, object]] = []

    for r0 in range(height):
        for c0 in range(width):
            color = int(grid[r0, c0])
            if color == BG or seen[r0, c0]:
                continue

            seen[r0, c0] = True
            q: deque[tuple[int, int]] = deque([(int(r0), int(c0))])
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr})

    return comps


def _frame_candidates(grid: np.ndarray) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for comp in _components_by_color(grid):
        color = int(comp["color"])
        coords = comp["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 3 or width < 3:
            continue

        border_ok = bool(np.all(grid[r0, c0 : c1 + 1] == color))
        border_ok = border_ok and bool(np.all(grid[r1, c0 : c1 + 1] == color))
        border_ok = border_ok and bool(np.all(grid[r0 : r1 + 1, c0] == color))
        border_ok = border_ok and bool(np.all(grid[r0 : r1 + 1, c1] == color))
        if not border_ok:
            continue

        area = int(coords.shape[0])
        border_area = _border_area(height, width)
        if color == ORANGE:
            if area < border_area or area > border_area + MAX_MARKERS:
                continue
        else:
            if area != border_area:
                continue

        candidates.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
            }
        )

    return candidates


def _extract_groups(grid: np.ndarray) -> list[dict[str, object]]:
    frames = _frame_candidates(grid)
    by_bbox = {tuple(frame["bbox"]): frame for frame in frames}

    groups: list[dict[str, object]] = []
    outer_frames = []
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        parent_bbox = (r0 - 2, c0 - 2, r1 + 2, c1 + 2)
        if parent_bbox not in by_bbox:
            outer_frames.append(frame)

    for outer in sorted(outer_frames, key=lambda f: tuple(f["bbox"])):
        chain = [outer]
        current = outer
        while True:
            r0, c0, r1, c1 = map(int, current["bbox"])
            child_bbox = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
            child = by_bbox.get(child_bbox)
            if child is None:
                break
            chain.append(child)
            current = child

        innermost_bbox = tuple(chain[-1]["bbox"])
        ir0, ic0, ir1, ic1 = map(int, innermost_bbox)
        marker_window = grid[ir0 + 1 : ir1, ic0 + 1 : ic1]
        marker_count = int(np.count_nonzero(marker_window == ORANGE))
        assert 1 <= marker_count <= MAX_MARKERS, "innermost frame must contain 1..5 orange markers"

        groups.append(
            {
                "outer_bbox": tuple(chain[0]["bbox"]),
                "colors": [int(frame["color"]) for frame in chain],
                "marker_count": marker_count,
            }
        )

    assert groups, "expected at least one nested-frame group"
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Strip each nested frame stack and redraw it as concentric square rings.

    Input structure:
    - Background is black.
    - Each object group is a set of nested 1-pixel rectangular frames with a
      1-pixel black gap between consecutive layers.
    - The innermost frame contains 1..5 orange marker pixels.

    Rule:
    - For each group, read frame colors from outer to inner and count the orange
      pixels inside the innermost frame.
    - Clear the whole group.
    - At the same center, draw tight concentric square rings whose innermost
      half-side length equals the orange marker count.
    - Colors reverse: innermost output ring uses the original innermost frame
      color, then proceed outward.
    - Clip any ring pixels that extend beyond the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full_like(g, BG)
    for group in _extract_groups(g):
        _render_group_output(
            out,
            tuple(group["outer_bbox"]),
            list(group["colors"]),
            int(group["marker_count"]),
        )
    return out


# --------------- generator ---------------

GRID_H_RANGE = (15, 30)
GRID_W_RANGE = (15, 30)
GROUP_COUNT_RANGE = (1, 4)
LAYER_COUNT_RANGE = (1, 4)
INNER_SIZE_RANGE = (3, 8)
PLACEMENT_TRIES = 250
EDGE_BIAS_PROB = 0.35


def _sample_marker_cells(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if count == 1:
        return [random.choice(cells)]

    for _ in range(40):
        picked = random.sample(cells, count)
        if all(abs(r1 - r2) + abs(c1 - c2) != 1 for i, (r1, c1) in enumerate(picked) for r2, c2 in picked[i + 1 :]):
            return picked
    return random.sample(cells, count)


def _sample_group_spec(height: int, width: int) -> dict[str, object] | None:
    max_layers = min(LAYER_COUNT_RANGE[1], len(FRAME_COLORS))
    layers = random.randint(LAYER_COUNT_RANGE[0], max_layers)
    marker_count = random.randint(1, MAX_MARKERS)

    max_inner_h = min(INNER_SIZE_RANGE[1], int(height) - 4 * (layers - 1))
    max_inner_w = min(INNER_SIZE_RANGE[1], int(width) - 4 * (layers - 1))
    if max_inner_h < INNER_SIZE_RANGE[0] or max_inner_w < INNER_SIZE_RANGE[0]:
        return None

    choices = []
    for inner_frame_h in range(INNER_SIZE_RANGE[0], max_inner_h + 1):
        for inner_frame_w in range(INNER_SIZE_RANGE[0], max_inner_w + 1):
            inner_h = inner_frame_h - 2
            inner_w = inner_frame_w - 2
            if inner_h * inner_w >= marker_count:
                choices.append((inner_frame_h, inner_frame_w))
    if not choices:
        return None

    inner_frame_h, inner_frame_w = random.choice(choices)
    outer_h = int(inner_frame_h + 4 * (layers - 1))
    outer_w = int(inner_frame_w + 4 * (layers - 1))
    colors = random.sample(list(FRAME_COLORS), layers)
    markers = _sample_marker_cells(inner_frame_h - 2, inner_frame_w - 2, marker_count)

    return {
        "layers": layers,
        "marker_count": marker_count,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "inner_frame_h": inner_frame_h,
        "inner_frame_w": inner_frame_w,
        "colors": colors,
        "markers": markers,
    }


def _render_group_input(grid: np.ndarray, r0: int, c0: int, spec: dict[str, object]) -> None:
    layers = int(spec["layers"])
    colors = list(spec["colors"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])

    for layer in range(layers):
        bbox = (
            int(r0 + 2 * layer),
            int(c0 + 2 * layer),
            int(r0 + outer_h - 1 - 2 * layer),
            int(c0 + outer_w - 1 - 2 * layer),
        )
        _draw_rect_border(grid, bbox, int(colors[layer]))

    inner_r0 = int(r0 + 2 * (layers - 1) + 1)
    inner_c0 = int(c0 + 2 * (layers - 1) + 1)
    for mr, mc in spec["markers"]:
        grid[inner_r0 + int(mr), inner_c0 + int(mc)] = ORANGE


def _group_output_mask(height: int, width: int, r0: int, c0: int, spec: dict[str, object]) -> np.ndarray:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    marker_count = int(spec["marker_count"])
    layers = int(spec["layers"])
    center_r2 = int(2 * r0 + outer_h - 1)
    center_c2 = int(2 * c0 + outer_w - 1)
    return _ring_mask(height, width, center_r2, center_c2, marker_count, layers)


def _output_clips(height: int, width: int, r0: int, c0: int, spec: dict[str, object]) -> bool:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    radius = int(spec["marker_count"]) + int(spec["layers"]) - 1
    bbox = _square_bbox(int(2 * r0 + outer_h - 1), int(2 * c0 + outer_w - 1), radius)
    top, left, bottom, right = bbox
    return top < 0 or left < 0 or bottom >= height or right >= width


def _render_expected_output(grid: np.ndarray, r0: int, c0: int, spec: dict[str, object]) -> None:
    bbox = (int(r0), int(c0), int(r0 + int(spec["outer_h"]) - 1), int(c0 + int(spec["outer_w"]) - 1))
    _render_group_output(grid, bbox, list(spec["colors"]), int(spec["marker_count"]))


def _bbox_blocked(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, halo: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, int(r0) - halo)
    cc0 = max(0, int(c0) - halo)
    rr1 = min(h, int(r1) + halo + 1)
    cc1 = min(w, int(c1) + halo + 1)
    return bool(occupied[rr0:rr1, cc0:cc1].any())


def _edge_biased_position(limit: int, span: int) -> int:
    if limit <= 0:
        return 0
    edge_span = min(limit, max(0, span))
    if edge_span == 0:
        return random.randint(0, limit)
    if random.random() < 0.5:
        return random.randint(0, edge_span)
    return random.randint(limit - edge_span, limit)


def _place_group(
    inp: np.ndarray,
    out: np.ndarray,
    occupied_in: np.ndarray,
    occupied_out: np.ndarray,
    spec: dict[str, object],
) -> bool:
    height, width = map(int, inp.shape)
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    max_r0 = height - outer_h
    max_c0 = width - outer_w
    if max_r0 < 0 or max_c0 < 0:
        return False

    clip_target = random.random() < EDGE_BIAS_PROB
    edge_radius = int(spec["marker_count"]) + int(spec["layers"]) - 1

    fallback: tuple[int, int] | None = None
    for _ in range(PLACEMENT_TRIES):
        if clip_target:
            r0 = _edge_biased_position(max_r0, edge_radius)
            c0 = _edge_biased_position(max_c0, edge_radius)
        else:
            r0 = random.randint(0, max_r0)
            c0 = random.randint(0, max_c0)

        r1 = int(r0 + outer_h - 1)
        c1 = int(c0 + outer_w - 1)
        if _bbox_blocked(occupied_in, r0, c0, r1, c1, halo=1):
            continue

        mask = _group_output_mask(height, width, r0, c0, spec)
        if not mask.any() or bool((mask & occupied_out).any()):
            continue

        clipped = _output_clips(height, width, r0, c0, spec)
        if not clip_target and clipped:
            if fallback is None:
                fallback = (int(r0), int(c0))
            continue
        if clip_target and not clipped and random.random() < 0.7:
            if fallback is None:
                fallback = (int(r0), int(c0))
            continue

        _render_group_input(inp, r0, c0, spec)
        _render_expected_output(out, r0, c0, spec)
        occupied_in[r0 : r1 + 1, c0 : c1 + 1] = True
        occupied_out |= mask
        return True

    if fallback is None:
        return False

    r0, c0 = fallback
    r1 = int(r0 + outer_h - 1)
    c1 = int(c0 + outer_w - 1)
    mask = _group_output_mask(height, width, r0, c0, spec)
    if not mask.any() or _bbox_blocked(occupied_in, r0, c0, r1, c1, halo=1) or bool((mask & occupied_out).any()):
        return False

    _render_group_input(inp, r0, c0, spec)
    _render_expected_output(out, r0, c0, spec)
    occupied_in[r0 : r1 + 1, c0 : c1 + 1] = True
    occupied_out |= mask
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    target_groups = random.randint(*GROUP_COUNT_RANGE)

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    occupied_in = np.zeros((height, width), dtype=bool)
    occupied_out = np.zeros((height, width), dtype=bool)

    for _ in range(target_groups):
        spec = _sample_group_spec(height, width)
        if spec is None:
            return None
        if not _place_group(inp, out, occupied_in, occupied_out, spec):
            return None

    if np.array_equal(inp, out):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
