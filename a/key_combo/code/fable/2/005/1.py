from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TAIL_COLOR = 1
HEAD_COLOR = 2
RING_PALETTE = (3, 4, 5, 6, 7, 8, 9)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
GEN_ATTEMPTS = 1200
SCENARIOS = ("mixed", "obstacle_corner", "boundary_corner")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _cell_kind(shape: tuple[int, int], obstacle_bbox: tuple[int, int, int, int], r: int, c: int) -> str | None:
    h, w = shape
    if not (0 <= r < h and 0 <= c < w):
        return "boundary"
    r0, c0, r1, c1 = obstacle_bbox
    if r0 <= r <= r1 and c0 <= c <= c1:
        return "obstacle"
    return None


def _trace_ray(
    shape: tuple[int, int],
    obstacle_bbox: tuple[int, int, int, int],
    head: tuple[int, int],
    direction: tuple[int, int],
    segment_count: int,
) -> tuple[list[list[tuple[int, int]]], list[dict[str, object]]]:
    h, w = shape
    r, c = map(int, head)
    dr, dc = map(int, direction)
    assert 0 <= r < h and 0 <= c < w

    segments: list[list[tuple[int, int]]] = []
    collisions: list[dict[str, object]] = []
    step_budget = h * w * max(8, segment_count * 4)
    used_steps = 0

    for _ in range(segment_count):
        segment: list[tuple[int, int]] = []
        while True:
            used_steps += 1
            if used_steps > step_budget:
                raise RuntimeError("ray simulation exceeded budget")

            row_kind = _cell_kind(shape, obstacle_bbox, r + dr, c)
            col_kind = _cell_kind(shape, obstacle_bbox, r, c + dc)
            diag_kind = _cell_kind(shape, obstacle_bbox, r + dr, c + dc)
            row_hit = row_kind is not None
            col_hit = col_kind is not None
            diag_hit = diag_kind is not None

            if row_hit or col_hit or diag_hit:
                corner = False
                if row_hit and col_hit:
                    dr, dc = -dr, -dc
                    corner = True
                elif row_hit:
                    dr = -dr
                elif col_hit:
                    dc = -dc
                else:
                    dr, dc = -dr, -dc
                    corner = True
                sources = {kind for kind in (row_kind, col_kind, diag_kind) if kind is not None}
                collisions.append(
                    {
                        "position": (r, c),
                        "sources": sources,
                        "corner": corner,
                    }
                )
                break

            r += dr
            c += dc
            segment.append((r, c))

        segments.append(segment)

    return segments, collisions


def _parse_scene(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    tail_coords = np.argwhere(g == TAIL_COLOR)
    head_coords = np.argwhere(g == HEAD_COLOR)
    assert tail_coords.shape == (1, 2), "expected one launcher tail pixel"
    assert head_coords.shape == (1, 2), "expected one launcher head pixel"

    tail = tuple(map(int, tail_coords[0]))
    head = tuple(map(int, head_coords[0]))
    dr = int(np.sign(head[0] - tail[0]))
    dc = int(np.sign(head[1] - tail[1]))
    assert abs(head[0] - tail[0]) == 1 and abs(head[1] - tail[1]) == 1, "launcher pixels must be diagonal neighbors"

    obstacle_mask = (g != BG) & (g != TAIL_COLOR) & (g != HEAD_COLOR)
    comps = _components(obstacle_mask, connectivity=8)
    assert len(comps) == 1, "expected exactly one concentric block"
    obstacle_bbox = _bbox(obstacle_mask)
    assert obstacle_bbox is not None
    r0, c0, r1, c1 = obstacle_bbox
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w >= 3, "block must be a square"
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub != BG), "block must fill its bbox"

    side = side_h
    layer_count = (side + 1) // 2
    rr, cc = np.indices(sub.shape)
    shell_index = np.minimum.reduce([rr, cc, side - 1 - rr, side - 1 - cc])
    outer_to_inner: list[int] = []
    for d in range(layer_count):
        shell_vals = sub[shell_index == d]
        color = int(shell_vals[0])
        assert np.all(shell_vals == color), "each ring must be monochromatic"
        assert color not in (BG, TAIL_COLOR, HEAD_COLOR), "ring colors must not reuse launcher colors"
        outer_to_inner.append(color)

    return tail, head, (dr, dc), obstacle_bbox, outer_to_inner[::-1]


def _paint_segments(
    grid: np.ndarray,
    segments: list[list[tuple[int, int]]],
    layer_colors_inner_to_outer: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for color, segment in zip(layer_colors_inner_to_outer, segments):
        for r, c in segment:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color successive reflected ray segments using the concentric square's layers from inner to outer.

    Input structure:
    - The background is black.
    - One filled axis-aligned square is partitioned into nested one-cell-thick monochromatic rings.
    - One launcher is a diagonally adjacent two-pixel domino made of fixed tail and head colors.

    Rule:
    - Start from the head pixel and move in the diagonal direction from tail to head.
    - Treat the concentric square as a solid reflecting obstacle, and also reflect at the grid border:
      a flat hit flips one direction component, while a corner hit flips both.
    - Color the cells visited before the 1st reflection with the innermost ring color, the next segment
      with the next ring color outward, and continue until the number of colored segments equals the
      number of rings.
    - Keep the launcher pixels unchanged.

    Key insight:
    - The square's outer edge controls the billiard path, while the square's inner-to-outer layer order
      is a separate visible program tape that determines both segment count and segment colors.
    """
    g = np.asarray(grid, dtype=int)
    tail, head, direction, obstacle_bbox, layer_colors = _parse_scene(g)
    segments, _collisions = _trace_ray(g.shape, obstacle_bbox, head, direction, len(layer_colors))
    out = _paint_segments(g, segments, layer_colors)
    out[tail] = TAIL_COLOR
    out[head] = HEAD_COLOR
    return out


def _build_block(side: int, layer_colors_inner_to_outer: list[int]) -> np.ndarray:
    block = np.zeros((side, side), dtype=int)
    for depth, color in enumerate(reversed(layer_colors_inner_to_outer)):
        block[depth : side - depth, depth : side - depth] = int(color)
    return block


def _launcher_ok(obstacle_bbox: tuple[int, int, int, int], tail: tuple[int, int], head: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = obstacle_bbox
    for r, c in (tail, head):
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
            return False
    return True


def _collision_summary(collisions: list[dict[str, object]]) -> dict[str, object]:
    obstacle_hits = sum("obstacle" in info["sources"] for info in collisions)
    boundary_hits = sum("boundary" in info["sources"] for info in collisions)
    obstacle_corner = any(info["corner"] and "obstacle" in info["sources"] for info in collisions)
    boundary_corner = any(info["corner"] and "boundary" in info["sources"] for info in collisions)
    return {
        "obstacle_hits": obstacle_hits,
        "boundary_hits": boundary_hits,
        "obstacle_corner": obstacle_corner,
        "boundary_corner": boundary_corner,
    }


def _candidate_launchers(
    shape: tuple[int, int],
    obstacle_bbox: tuple[int, int, int, int],
    layer_count: int,
    scenario: str,
) -> list[tuple[tuple[int, int], tuple[int, int], list[list[tuple[int, int]]], list[dict[str, object]]]]:
    h, w = shape
    candidates: list[tuple[tuple[int, int], tuple[int, int], list[list[tuple[int, int]]], list[dict[str, object]]]] = []
    for dr, dc in DIAGONALS:
        for hr in range(h):
            for hc in range(w):
                tail = (hr - dr, hc - dc)
                head = (hr, hc)
                if not (0 <= tail[0] < h and 0 <= tail[1] < w):
                    continue
                if _cell_kind(shape, obstacle_bbox, *tail) is not None:
                    continue
                if _cell_kind(shape, obstacle_bbox, *head) is not None:
                    continue
                if not _launcher_ok(obstacle_bbox, tail, head):
                    continue
                try:
                    segments, collisions = _trace_ray(shape, obstacle_bbox, head, (dr, dc), layer_count)
                except RuntimeError:
                    continue
                if not all(segment for segment in segments):
                    continue
                flat_total = sum(1 for info in collisions if not info["corner"])
                if flat_total == 0:
                    continue
                path_cells = [cell for segment in segments for cell in segment]
                unique_cells = len(set(path_cells))
                if unique_cells < max(7, 2 * layer_count):
                    continue
                if unique_cells * 10 < len(path_cells) * 6:
                    continue
                summary = _collision_summary(collisions)
                if summary["obstacle_hits"] == 0 or summary["boundary_hits"] == 0:
                    continue
                if scenario == "obstacle_corner" and not summary["obstacle_corner"]:
                    continue
                if scenario == "boundary_corner" and not summary["boundary_corner"]:
                    continue
                candidates.append((tail, head, segments, collisions))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(3, 6)
    side = 2 * layer_count - random.choice((0, 1))
    min_size = max(side + 8, 14)
    h = random.randint(min_size, 30)
    w = random.randint(min_size, 30)
    if h - side < 5 or w - side < 5:
        return None

    top = random.randint(2, h - side - 3)
    left = random.randint(2, w - side - 3)
    obstacle_bbox = (top, left, top + side - 1, left + side - 1)
    scenario = random.choice(SCENARIOS)

    candidates = _candidate_launchers((h, w), obstacle_bbox, layer_count, scenario)
    if not candidates:
        return None

    tail, head, _segments, _collisions = random.choice(candidates)
    ring_colors = random.sample(RING_PALETTE, layer_count)
    block = _build_block(side, ring_colors)

    grid = np.zeros((h, w), dtype=int)
    grid[top : top + side, left : left + side] = block
    grid[tail] = TAIL_COLOR
    grid[head] = HEAD_COLOR

    parsed_tail, parsed_head, direction, parsed_bbox, parsed_colors = _parse_scene(grid)
    assert parsed_tail == tail and parsed_head == head
    assert parsed_bbox == obstacle_bbox
    assert parsed_colors == ring_colors
    segments, collisions = _trace_ray(grid.shape, parsed_bbox, parsed_head, direction, len(parsed_colors))
    summary = _collision_summary(collisions)
    if summary["obstacle_hits"] == 0 or summary["boundary_hits"] == 0:
        return None
    if scenario == "obstacle_corner" and not summary["obstacle_corner"]:
        return None
    if scenario == "boundary_corner" and not summary["boundary_corner"]:
        return None
    if not all(segment for segment in segments):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid concentric-ray puzzle input")
