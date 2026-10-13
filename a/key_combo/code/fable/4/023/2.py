from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEAL = 8

Coord = tuple[int, int]
Direction = tuple[int, int]

DIR_UP: Direction = (-1, 0)
DIR_DOWN: Direction = (1, 0)
DIR_LEFT: Direction = (0, -1)
DIR_RIGHT: Direction = (0, 1)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _bbox(cells: list[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(cells: list[Coord] | tuple[Coord, ...]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _normalize_cells(cells: list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    return tuple(sorted((int(r), int(c)) for r, c in cells))


def _shift_cells(cells: tuple[Coord, ...], direction: Direction, distance: int) -> tuple[Coord, ...]:
    dr, dc = direction
    return _normalize_cells([(r + dr * distance, c + dc * distance) for r, c in cells])


def _distance_to_edge(cells: tuple[Coord, ...], direction: Direction, shape: tuple[int, int]) -> int:
    h, w = shape
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if direction == DIR_RIGHT:
        return w - 1 - max(cols)
    if direction == DIR_LEFT:
        return min(cols)
    if direction == DIR_DOWN:
        return h - 1 - max(rows)
    if direction == DIR_UP:
        return min(rows)
    raise ValueError("unsupported direction")


def _distance_to_obstacle(
    moving_cells: tuple[Coord, ...],
    obstacle_cells: tuple[Coord, ...],
    direction: Direction,
) -> int | None:
    best: int | None = None
    if direction == DIR_RIGHT:
        obstacle_rows: dict[int, list[int]] = {}
        for r, c in obstacle_cells:
            obstacle_rows.setdefault(r, []).append(c)
        for r, c in moving_cells:
            for oc in obstacle_rows.get(r, []):
                dist = oc - c - 1
                if dist > 0 and (best is None or dist < best):
                    best = dist
    elif direction == DIR_LEFT:
        obstacle_rows = {}
        for r, c in obstacle_cells:
            obstacle_rows.setdefault(r, []).append(c)
        for r, c in moving_cells:
            for oc in obstacle_rows.get(r, []):
                dist = c - oc - 1
                if dist > 0 and (best is None or dist < best):
                    best = dist
    elif direction == DIR_DOWN:
        obstacle_cols: dict[int, list[int]] = {}
        for r, c in obstacle_cells:
            obstacle_cols.setdefault(c, []).append(r)
        for r, c in moving_cells:
            for orow in obstacle_cols.get(c, []):
                dist = orow - r - 1
                if dist > 0 and (best is None or dist < best):
                    best = dist
    elif direction == DIR_UP:
        obstacle_cols = {}
        for r, c in obstacle_cells:
            obstacle_cols.setdefault(c, []).append(r)
        for r, c in moving_cells:
            for orow in obstacle_cols.get(c, []):
                dist = r - orow - 1
                if dist > 0 and (best is None or dist < best):
                    best = dist
    else:
        raise ValueError("unsupported direction")
    return best


def _infer_red_direction(cells: list[Coord]) -> Direction:
    assert len(cells) >= 3, "red object must contain a rectangle body and one protrusion"
    cell_set = set(cells)

    candidates: list[Direction] = []
    for tail in cells:
        body = [cell for cell in cells if cell != tail]
        if not _is_solid_rectangle(body):
            continue
        r0, c0, r1, c1 = _bbox(body)
        tr, tc = tail
        if c0 <= tc <= c1 and tr == r0 - 1:
            candidates.append(DIR_DOWN)
        elif c0 <= tc <= c1 and tr == r1 + 1:
            candidates.append(DIR_UP)
        elif r0 <= tr <= r1 and tc == c0 - 1:
            candidates.append(DIR_RIGHT)
        elif r0 <= tr <= r1 and tc == c1 + 1:
            candidates.append(DIR_LEFT)

    assert len(candidates) == 1, f"expected one unique protrusion interpretation, got {len(candidates)}"
    return candidates[0]


def _find_anchor(grid: np.ndarray) -> tuple[Coord, ...]:
    anchor_components = _components_of_color(grid, TEAL)
    matches = []
    for comp in anchor_components:
        if len(comp) != 4 or not _is_solid_rectangle(comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 == 1 and c1 - c0 == 1:
            matches.append(_normalize_cells(comp))
    assert len(matches) == 1, "expected exactly one teal 2x2 anchor block"
    return matches[0]


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[int], tuple[Coord, ...]]:
    anchor = _find_anchor(grid)

    pieces: list[dict] = []
    red_ids: list[int] = []

    for comp in _components_of_color(grid, RED):
        direction = _infer_red_direction(comp)
        piece = {
            "id": len(pieces),
            "color": RED,
            "cells": _normalize_cells(comp),
            "direction": direction,
        }
        pieces.append(piece)
        red_ids.append(piece["id"])

    for comp in _components_of_color(grid, GREY):
        assert _is_solid_rectangle(comp), "grey objects must be solid rectangles"
        pieces.append(
            {
                "id": len(pieces),
                "color": GREY,
                "cells": _normalize_cells(comp),
                "direction": None,
            }
        )

    assert red_ids, "expected at least one red object"
    return pieces, red_ids, anchor


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each tailed red object opposite its protrusion and relay the motion through every object it hits.

    Input structure:
    - The background is black.
    - Red objects are 4-connected shapes made of one solid rectangular body plus exactly one
      single-cell side protrusion.
    - Grey objects are solid rectangles with no protrusion, and one teal 2x2 block is a fixed anchor.

    Rule:
    - For each red object, read the protrusion side and move the whole object in the opposite direction.
    - A moving object stops as soon as it would become 4-neighbor adjacent to the next object on that lane;
      the struck object then continues in the same direction from its own original position.
    - The last relayed object either stops adjacent to the teal anchor or, if nothing blocks it, slides until
      it is flush with the grid edge. All other untouched objects stay where they started.

    Key insight:
    - The protrusion is a tail flame, so it points backward; motion passes down the lane like a Newton's-cradle
      relay, and only the final object uses the remaining open distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pieces, red_ids, anchor = _parse_scene(g)
    final_cells: dict[int, tuple[Coord, ...]] = {}
    claimed: set[int] = set()

    for red_id in red_ids:
        if red_id in claimed:
            continue

        direction = pieces[red_id]["direction"]
        assert direction is not None
        current_id = red_id
        chain_seen: set[int] = set()

        while True:
            chain_seen.add(current_id)
            moving_cells = pieces[current_id]["cells"]
            candidates: list[tuple[int, str, int | None]] = []

            for piece in pieces:
                other_id = piece["id"]
                if other_id in chain_seen:
                    continue
                dist = _distance_to_obstacle(moving_cells, piece["cells"], direction)
                if dist is not None:
                    candidates.append((dist, "piece", other_id))

            anchor_dist = _distance_to_obstacle(moving_cells, anchor, direction)
            if anchor_dist is not None:
                candidates.append((anchor_dist, "anchor", None))

            if not candidates:
                edge_dist = _distance_to_edge(moving_cells, direction, g.shape)
                final_cells[current_id] = _shift_cells(moving_cells, direction, edge_dist)
                claimed.add(current_id)
                break

            best_dist = min(dist for dist, _kind, _target in candidates)
            best = [entry for entry in candidates if entry[0] == best_dist]
            assert len(best) == 1, "collision order must be unique"
            dist, kind, target = best[0]
            final_cells[current_id] = _shift_cells(moving_cells, direction, dist)
            claimed.add(current_id)

            if kind == "anchor":
                break
            assert target is not None
            current_id = target

    out = np.full_like(g, BLACK)
    for r, c in anchor:
        out[r, c] = TEAL

    for piece in pieces:
        cells = final_cells.get(piece["id"], piece["cells"])
        for r, c in cells:
            assert out[r, c] == BLACK, "final objects must not overlap"
            out[r, c] = int(piece["color"])

    return out


def _make_red_piece(top: int, height: int, full_left: int, full_width: int, sign: int) -> tuple[Coord, ...]:
    body_width = full_width - 1
    tail_row = top + random.randrange(height)
    cells: list[Coord] = []

    if sign == 1:
        body_left = full_left + 1
        tail_col = full_left
    else:
        body_left = full_left
        tail_col = full_left + body_width

    for r in range(top, top + height):
        for c in range(body_left, body_left + body_width):
            cells.append((r, c))
    cells.append((tail_row, tail_col))
    return _normalize_cells(cells)


def _make_grey_piece(top: int, height: int, left: int, width: int) -> tuple[Coord, ...]:
    return _normalize_cells([(r, c) for r in range(top, top + height) for c in range(left, left + width)])


def _lane_intervals(
    width: int,
    chain_len: int,
    sign: int,
    target_mode: str,
    anchor_left: int,
    anchor_right: int,
) -> tuple[list[tuple[int, int]], list[int]] | None:
    red_full_width = random.randint(3, 5)
    widths = [red_full_width] + [random.randint(2, 4) for _ in range(chain_len - 1)]
    intervals: list[tuple[int, int] | None] = [None] * chain_len

    if sign == 1:
        cursor = anchor_left if target_mode == "anchor" else width
        for i in range(chain_len - 1, -1, -1):
            gap = random.randint(2, 5 if target_mode == "anchor" else 6)
            right = cursor - gap - 1
            left = right - widths[i] + 1
            intervals[i] = (left, right)
            cursor = left
    else:
        cursor = anchor_right if target_mode == "anchor" else -1
        for i in range(chain_len - 1, -1, -1):
            gap = random.randint(2, 5 if target_mode == "anchor" else 6)
            left = cursor + gap + 1
            right = left + widths[i] - 1
            intervals[i] = (left, right)
            cursor = right

    placed = [interval for interval in intervals if interval is not None]
    if any(left < 0 or right >= width for left, right in placed):
        return None
    return placed, widths


def _build_horizontal_lane(
    top: int,
    height: int,
    width: int,
    sign: int,
    target_mode: str,
    anchor_left: int,
    anchor_right: int,
    chain_len: int,
) -> list[tuple[int, tuple[Coord, ...]] | None] | None:
    sampled = _lane_intervals(width, chain_len, sign, target_mode, anchor_left, anchor_right)
    if sampled is None:
        return None
    intervals, widths = sampled

    pieces: list[tuple[int, tuple[Coord, ...]] | None] = []
    for i, (left, _right) in enumerate(intervals):
        if i == 0:
            pieces.append((RED, _make_red_piece(top, height, left, widths[i], sign)))
        else:
            pieces.append((GREY, _make_grey_piece(top, height, left, widths[i])))
    return pieces


def _pick_lane_bands(height: int, anchor_top: int, lane_count: int) -> list[tuple[int, int]] | None:
    occupied = [(anchor_top, anchor_top + 1)]
    bands = [(anchor_top, 2)]

    for _ in range(lane_count - 1):
        choices: list[tuple[int, int]] = []
        for lane_height in (2, 3):
            for top in range(0, height - lane_height + 1):
                bottom = top + lane_height - 1
                if all(bottom < other_top - 1 or top > other_bottom + 1 for other_top, other_bottom in occupied):
                    choices.append((top, lane_height))
        if not choices:
            return None
        top, lane_height = random.choice(choices)
        occupied.append((top, top + lane_height - 1))
        bands.append((top, lane_height))

    return bands


def _paint_cells(grid: np.ndarray, cells: tuple[Coord, ...], color: int) -> bool:
    for r, c in cells:
        if grid[r, c] != BLACK:
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _build_canonical_input() -> np.ndarray | None:
    height = random.randint(16, 26)
    width = random.randint(22, 30)
    anchor_top = random.randint(2, height - 4)
    anchor_left = random.randint(7, width - 9)
    lane_count = random.randint(2, 3)

    lane_bands = _pick_lane_bands(height, anchor_top, lane_count)
    if lane_bands is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[anchor_top : anchor_top + 2, anchor_left : anchor_left + 2] = TEAL

    lane_specs = []
    toward_sign = random.choice((1, -1))
    lane_specs.append((lane_bands[0], toward_sign, "anchor", random.randint(3, 4)))

    for top, lane_height in lane_bands[1:]:
        side = random.choice(("left", "right"))
        sign = -1 if side == "left" else 1
        lane_specs.append(((top, lane_height), sign, "edge", random.randint(2, 3)))

    for (top, lane_height), sign, target_mode, chain_len in lane_specs:
        pieces = _build_horizontal_lane(
            top=top,
            height=lane_height,
            width=width,
            sign=sign,
            target_mode=target_mode,
            anchor_left=anchor_left,
            anchor_right=anchor_left + 1,
            chain_len=chain_len,
        )
        if pieces is None:
            return None
        for piece in pieces:
            assert piece is not None
            color, cells = piece
            if not _paint_cells(grid, cells, color):
                return None

    return grid


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _valid_generated_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if len(_components_of_color(grid, RED)) < 2:
        return False
    if len(_components_of_color(grid, GREY)) < 2:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return not np.array_equal(grid, out)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_canonical_input()
        if grid is None:
            continue
        grid = _random_symmetry(grid)
        if _valid_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
