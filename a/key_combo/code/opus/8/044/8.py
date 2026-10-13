from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
OBSTACLE = 5

GEN_ATTEMPTS = 600
GEN_HEIGHT_RANGE = (12, 20)
GEN_WIDTH_RANGE = (15, 24)


def _periodic_positions(seed: int, step: int, limit: int) -> list[int]:
    positions: set[int] = set()
    value = int(seed)
    while value >= 0:
        positions.add(value)
        value -= step
    value = int(seed) + step
    while value < limit:
        positions.add(value)
        value += step
    return sorted(positions)


def _label_obstacle_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    height, width = mask.shape
    comp_ids = np.zeros((height, width), dtype=int)
    components: dict[int, dict[str, object]] = {}
    next_id = 1

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or comp_ids[r, c] != 0:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            comp_ids[r, c] = next_id
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and comp_ids[nr, nc] == 0:
                        comp_ids[nr, nc] = next_id
                        q.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert r1 - r0 + 1 <= 2 and c1 - c0 + 1 <= 2, "obstacle blocks must be at most 2x2"
            assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1), "each obstacle must be a solid rectangle"
            components[next_id] = {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "cells": cells,
            }
            next_id += 1

    return comp_ids, components


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    allowed = {BG, BLUE, RED, OBSTACLE}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected colors in grid"

    blue_coords = np.argwhere(g == BLUE)
    red_coords = np.argwhere(g == RED)
    assert len(blue_coords) == 1 and len(red_coords) == 1, "expected exactly one blue seed and one red seed"

    blue = tuple(map(int, blue_coords[0]))
    red = tuple(map(int, red_coords[0]))
    obstacle_mask = g == OBSTACLE
    assert not obstacle_mask[blue] and not obstacle_mask[red], "seeds must not overlap obstacles"
    return g, blue, red, obstacle_mask


def _infer_layout(
    shape: tuple[int, int], blue: tuple[int, int], red: tuple[int, int]
) -> tuple[str, int, list[tuple[int, int]]]:
    height, width = map(int, shape)
    horizontal = blue[1] == red[1] or blue[0] in (0, height - 1) or red[0] in (0, height - 1)
    if horizontal:
        distance = abs(int(blue[0]) - int(red[0]))
        assert distance > 0, "horizontal stripes require distinct seed rows"
        step = 2 * distance
        tracks = [(row, BLUE) for row in _periodic_positions(blue[0], step, height)]
        tracks += [(row, RED) for row in _periodic_positions(red[0], step, height)]
        return "horizontal", distance, sorted(tracks)

    distance = abs(int(blue[1]) - int(red[1]))
    assert distance > 0, "vertical stripes require distinct seed columns"
    step = 2 * distance
    tracks = [(col, BLUE) for col in _periodic_positions(blue[1], step, width)]
    tracks += [(col, RED) for col in _periodic_positions(red[1], step, width)]
    return "vertical", distance, sorted(tracks)


def _paint_cell(out: np.ndarray, r: int, c: int, color: int) -> None:
    current = int(out[r, c])
    assert current in (BG, color), "stripe collision should not occur in valid inputs"
    out[r, c] = color


def _trace_horizontal(
    out: np.ndarray,
    obstacle_mask: np.ndarray,
    comp_ids: np.ndarray,
    components: dict[int, dict[str, object]],
    row: int,
    color: int,
) -> list[int]:
    height, width = out.shape
    del height
    cur_row = int(row)
    hits: list[int] = []

    for col in range(width):
        assert not obstacle_mask[cur_row, col], "a deflected stripe must never occupy an obstacle cell"
        _paint_cell(out, cur_row, col, color)
        if col == width - 1 or not obstacle_mask[cur_row, col + 1]:
            continue

        comp_id = int(comp_ids[cur_row, col + 1])
        hits.append(comp_id)
        center_row = float(components[comp_id]["center"][0])  # type: ignore[index]
        shift = -1 if cur_row < center_row else 1
        next_row = cur_row + shift
        assert 0 <= next_row < out.shape[0], "deflection must stay in bounds"
        assert not obstacle_mask[next_row, col], "connector cell must stay clear"
        _paint_cell(out, next_row, col, color)
        cur_row = next_row

    return hits


def _trace_vertical(
    out: np.ndarray,
    obstacle_mask: np.ndarray,
    comp_ids: np.ndarray,
    components: dict[int, dict[str, object]],
    col: int,
    color: int,
) -> list[int]:
    height, width = out.shape
    del width
    cur_col = int(col)
    hits: list[int] = []

    for row in range(height):
        assert not obstacle_mask[row, cur_col], "a deflected stripe must never occupy an obstacle cell"
        _paint_cell(out, row, cur_col, color)
        if row == height - 1 or not obstacle_mask[row + 1, cur_col]:
            continue

        comp_id = int(comp_ids[row + 1, cur_col])
        hits.append(comp_id)
        center_col = float(components[comp_id]["center"][1])  # type: ignore[index]
        shift = -1 if cur_col < center_col else 1
        next_col = cur_col + shift
        assert 0 <= next_col < out.shape[1], "deflection must stay in bounds"
        assert not obstacle_mask[row, next_col], "connector cell must stay clear"
        _paint_cell(out, row, next_col, color)
        cur_col = next_col

    return hits


def _render_with_details(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g, blue, red, obstacle_mask = _parse_grid(grid)
    height, width = g.shape
    comp_ids, components = _label_obstacle_components(obstacle_mask)
    assert len(components) <= 4, "expected at most 4 obstacle blocks"

    orientation, period, tracks = _infer_layout(g.shape, blue, red)
    out = np.zeros((height, width), dtype=int)
    out[obstacle_mask] = OBSTACLE

    hits_by_track: dict[int, list[int]] = {}
    encountered: list[int] = []

    for track, color in tracks:
        if orientation == "horizontal":
            hits = _trace_horizontal(out, obstacle_mask, comp_ids, components, track, color)
        else:
            hits = _trace_vertical(out, obstacle_mask, comp_ids, components, track, color)
        hits_by_track[int(track)] = hits
        encountered.extend(hits)

    out[blue] = BLUE
    out[red] = RED
    details = {
        "orientation": orientation,
        "period": period,
        "tracks": tracks,
        "hits_by_track": hits_by_track,
        "encountered": encountered,
        "obstacle_count": len(components),
    }
    return out, details


def _render_without_obstacles(shape: tuple[int, int], blue: tuple[int, int], red: tuple[int, int]) -> np.ndarray:
    orientation, _period, tracks = _infer_layout(shape, blue, red)
    out = np.zeros(shape, dtype=int)

    for track, color in tracks:
        if orientation == "horizontal":
            out[track, :] = color
        else:
            out[:, track] = color

    out[blue] = BLUE
    out[red] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw periodic seed-defined stripes and bend each stripe one cell around gray obstacle blocks.

    Input structure:
    - A black background contains exactly one blue seed, exactly one red seed, and 2-4 gray obstacle blocks.
    - Each obstacle block is a solid 1x1, 1x2, 2x1, or 2x2 rectangle that does not overlap a seed.
    - If the seeds share a column, or either seed lies on the top or bottom row, the target stripes are horizontal rows; otherwise they are vertical columns.

    Rule:
    - Use the seed rows for horizontal stripes or the seed columns for vertical stripes, and let the absolute seed separation on that axis be the period offset D.
    - Extend a blue stripe through the blue seed track and a red stripe through the red seed track, then repeat both tracks every 2D cells along the perpendicular axis.
    - Scan each stripe along its length from left to right or top to bottom; when the next stripe cell would enter a gray obstacle, shift the stripe by one cell away from that obstacle's center, break the corner with a one-cell connector, and keep following the new shifted track.
    - Gray obstacles remain unchanged, while every traversed background cell becomes the stripe color.

    Key insight:
    - The two seeds jointly encode stripe orientation and period, while each obstacle only changes the stripe locally by forcing a permanent one-cell sidestep away from the obstacle's center.
    """
    out, _details = _render_with_details(grid)
    return out


def _sample_seed_layout() -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], str] | None:
    branch = random.choice(("horizontal_samecol", "horizontal_edge", "vertical"))

    for _ in range(80):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)

        if branch == "horizontal_samecol":
            max_d = min(6, height - 3)
            if max_d < 2:
                continue
            distance = random.randint(2, max_d)
            row1 = random.randint(1, height - distance - 2)
            row2 = row1 + distance
            col = random.randint(1, max(2, width // 2))
            blue, red = (row1, col), (row2, col)
        elif branch == "horizontal_edge":
            max_d = min(6, height - 2)
            if max_d < 2:
                continue
            distance = random.randint(2, max_d)
            edge_top = random.choice((True, False))
            if edge_top:
                rows = (0, distance)
            else:
                rows = (height - 1 - distance, height - 1)
            col1 = random.randint(1, max(2, width // 2))
            col2 = random.randint(col1 + 1, width - 2) if col1 + 1 <= width - 2 else width - 1
            blue, red = (rows[0], col1), (rows[1], col2)
        else:
            max_d = min(6, width - 3)
            if max_d < 2:
                continue
            distance = random.randint(2, max_d)
            col1 = random.randint(1, width - distance - 2)
            col2 = col1 + distance
            row1 = random.randint(1, max(2, height // 2))
            row2 = random.randint(1, max(2, height // 2))
            blue, red = (row1, col1), (row2, col2)

        orientation, period, tracks = _infer_layout((height, width), blue, red)
        if branch.startswith("horizontal") and orientation != "horizontal":
            continue
        if branch == "vertical" and orientation != "vertical":
            continue
        if period < 2 or len(tracks) < 4:
            continue
        return (height, width), blue, red, branch

    return None


def _expanded_overlaps(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r1 + pad + 1)
    cc1 = min(reserved.shape[1], c1 + pad + 1)
    return bool(reserved[rr0:rr1, cc0:cc1].any())


def _choose_tracks_for_obstacles(tracks: list[tuple[int, int]]) -> list[int]:
    blue_tracks = [track for track, color in tracks if color == BLUE]
    red_tracks = [track for track, color in tracks if color == RED]
    count = random.randint(2, min(4, len(tracks)))

    chosen = [random.choice(blue_tracks), random.choice(red_tracks)]
    remaining = [track for track, _color in tracks if track not in chosen]
    extra_needed = count - len(chosen)
    if extra_needed > 0:
        chosen.extend(random.sample(remaining, extra_needed))
    random.shuffle(chosen)
    return chosen


def _horizontal_obstacle_bbox(
    track: int, shift: int, height_choice: int, width_choice: int, hit_col: int
) -> tuple[int, int, int, int]:
    if shift == -1:
        row0 = track
    elif height_choice == 1:
        row0 = track
    else:
        row0 = track - 1
    return row0, hit_col, row0 + height_choice - 1, hit_col + width_choice - 1


def _vertical_obstacle_bbox(
    track: int, shift: int, width_choice: int, height_choice: int, hit_row: int
) -> tuple[int, int, int, int]:
    if shift == -1:
        col0 = track
    elif width_choice == 1:
        col0 = track
    else:
        col0 = track - 1
    return hit_row, col0, hit_row + height_choice - 1, col0 + width_choice - 1


def _sample_obstacles(
    shape: tuple[int, int],
    blue: tuple[int, int],
    red: tuple[int, int],
) -> list[tuple[int, int, int, int]] | None:
    orientation, _period, tracks = _infer_layout(shape, blue, red)
    base_tracks = {track for track, _ in tracks}
    chosen_tracks = _choose_tracks_for_obstacles(tracks)
    height, width = shape
    reserved = np.zeros(shape, dtype=bool)
    shifted_tracks: set[int] = set()
    obstacles: list[tuple[int, int, int, int]] = []
    seed_cells = {blue, red}
    seed_pos_by_track: dict[int, int] = {}
    if orientation == "horizontal":
        seed_pos_by_track[blue[0]] = blue[1]
        seed_pos_by_track[red[0]] = red[1]
    else:
        seed_pos_by_track[blue[1]] = blue[0]
        seed_pos_by_track[red[1]] = red[0]

    for track in chosen_tracks:
        if orientation == "horizontal":
            candidate_shifts = [
                shift
                for shift in (-1, 1)
                if 0 <= track + shift < height and track + shift not in base_tracks and track + shift not in shifted_tracks
            ]
        else:
            candidate_shifts = [
                shift
                for shift in (-1, 1)
                if 0 <= track + shift < width and track + shift not in base_tracks and track + shift not in shifted_tracks
            ]
        if not candidate_shifts:
            return None
        random.shuffle(candidate_shifts)

        placed = False
        for shift in candidate_shifts:
            for _ in range(40):
                if orientation == "horizontal":
                    height_choice = 2 if shift == -1 else random.choices([1, 2], weights=[3, 2])[0]
                    width_choice = random.choice((1, 2))
                    extra_row = track + 1 if shift == -1 else (track - 1 if height_choice == 2 else None)
                    if extra_row is not None and extra_row in shifted_tracks:
                        continue
                    low = 2
                    high = width - width_choice - 2
                    if track in seed_pos_by_track:
                        low = max(low, seed_pos_by_track[track] + 2)
                    if low > high:
                        continue
                    hit_col = random.randint(low, high)
                    bbox = _horizontal_obstacle_bbox(track, shift, height_choice, width_choice, hit_col)
                    connector = (track + shift, hit_col - 1)
                else:
                    width_choice = 2 if shift == -1 else random.choices([1, 2], weights=[3, 2])[0]
                    height_choice = random.choice((1, 2))
                    extra_col = track + 1 if shift == -1 else (track - 1 if width_choice == 2 else None)
                    if extra_col is not None and extra_col in shifted_tracks:
                        continue
                    low = 2
                    high = height - height_choice - 2
                    if track in seed_pos_by_track:
                        low = max(low, seed_pos_by_track[track] + 2)
                    if low > high:
                        continue
                    hit_row = random.randint(low, high)
                    bbox = _vertical_obstacle_bbox(track, shift, width_choice, height_choice, hit_row)
                    connector = (hit_row - 1, track + shift)

                r0, c0, r1, c1 = bbox
                if _expanded_overlaps(reserved, bbox):
                    continue
                cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                if cells & seed_cells or connector in seed_cells:
                    continue
                if not (0 <= connector[0] < height and 0 <= connector[1] < width):
                    continue

                obstacles.append(bbox)
                reserved[r0 : r1 + 1, c0 : c1 + 1] = True
                shifted_tracks.add(track + shift)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return obstacles


def _build_input_grid(
    shape: tuple[int, int],
    blue: tuple[int, int],
    red: tuple[int, int],
    obstacles: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in obstacles:
        grid[r0 : r1 + 1, c0 : c1 + 1] = OBSTACLE
    grid[blue] = BLUE
    grid[red] = RED
    return grid


def _obstacle_masks(grid: np.ndarray) -> list[np.ndarray]:
    obstacle_mask = grid == OBSTACLE
    comp_ids, components = _label_obstacle_components(obstacle_mask)
    masks: list[np.ndarray] = []
    for comp_id in sorted(components):
        mask = comp_ids == comp_id
        masks.append(mask)
    return masks


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        out, details = _render_with_details(inp)
    except AssertionError:
        return False

    _g, blue, red, _obstacle_mask = _parse_grid(inp)
    base = _render_without_obstacles(inp.shape, blue, red)
    if np.array_equal(out, base):
        return False

    encountered = list(map(int, details["encountered"]))
    obstacle_count = int(details["obstacle_count"])
    if not 2 <= obstacle_count <= 4:
        return False
    if len(encountered) != obstacle_count or set(encountered) != set(range(1, obstacle_count + 1)):
        return False

    hits_by_track: dict[int, list[int]] = details["hits_by_track"]  # type: ignore[assignment]
    if any(len(hits) > 1 for hits in hits_by_track.values()):
        return False

    if int(np.count_nonzero(out == BLUE)) <= 1 or int(np.count_nonzero(out == RED)) <= 1:
        return False

    diff_mask = (out != base) & (inp != OBSTACLE)
    if int(np.count_nonzero(diff_mask)) < obstacle_count * 2:
        return False

    for obstacle_mask in _obstacle_masks(inp):
        variant = inp.copy()
        variant[obstacle_mask] = BG
        if np.array_equal(main(variant), out):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    layout = _sample_seed_layout()
    if layout is None:
        return None
    shape, blue, red, _branch = layout
    obstacles = _sample_obstacles(shape, blue, red)
    if obstacles is None:
        return None

    inp = _build_input_grid(shape, blue, red, obstacles)
    if not _semantic_checks(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-deflection puzzle input")
