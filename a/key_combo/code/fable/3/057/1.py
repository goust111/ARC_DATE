from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GENERATION_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _marker_lane(coord: tuple[int, int], shape: tuple[int, int]) -> dict:
    r, c = map(int, coord)
    h, w = map(int, shape)
    side_hits = int(r == 0) + int(r == h - 1) + int(c == 0) + int(c == w - 1)
    if side_hits != 1:
        raise ValueError("marker must lie on exactly one border side")

    if c == 0:
        return {"key": ("row", r), "line_axis": 0, "line": r, "move_axis": 1, "sign": 1}
    if c == w - 1:
        return {"key": ("row", r), "line_axis": 0, "line": r, "move_axis": 1, "sign": -1}
    if r == 0:
        return {"key": ("col", c), "line_axis": 1, "line": c, "move_axis": 0, "sign": 1}
    return {"key": ("col", c), "line_axis": 1, "line": c, "move_axis": 0, "sign": -1}


def _component_on_lane(comp: dict, lane: dict) -> bool:
    coords = np.asarray(comp["coords"], dtype=int)
    return bool(np.any(coords[:, int(lane["line_axis"])] == int(lane["line"])))


def _sort_key(comp: dict, lane: dict) -> int:
    coords = np.asarray(comp["coords"], dtype=int)
    axis = int(lane["move_axis"])
    sign = int(lane["sign"])
    vals = coords[:, axis]
    return int(vals.min()) if sign > 0 else -int(vals.max())


def _steps_to_next(curr: dict, nxt: dict, lane: dict) -> int:
    axis = int(lane["move_axis"])
    sign = int(lane["sign"])
    curr_vals = np.asarray(curr["coords"], dtype=int)[:, axis]
    next_vals = np.asarray(nxt["coords"], dtype=int)[:, axis]
    if sign > 0:
        return int(next_vals.min() - curr_vals.max() - 1)
    return int(curr_vals.min() - next_vals.max() - 1)


def _steps_to_boundary(comp: dict, lane: dict, shape: tuple[int, int]) -> int:
    axis = int(lane["move_axis"])
    sign = int(lane["sign"])
    vals = np.asarray(comp["coords"], dtype=int)[:, axis]
    limit = int(shape[axis] - 1)
    if sign > 0:
        return int(limit - vals.max())
    return int(vals.min())


def _shift_coords(coords: np.ndarray, lane: dict, steps: int) -> np.ndarray:
    shifted = np.asarray(coords, dtype=int).copy()
    shifted[:, int(lane["move_axis"])] += int(lane["sign"]) * int(steps)
    return shifted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each border-launched shove through the objects on its row or column, handing off the gaps one by one.

    Input structure:
    - The background is black.
    - One-pixel launcher markers sit on the grid border, each selecting one row or one column and a direction into the grid.
    - Several monochromatic solid foreground objects lie on each launched row or column, while other off-lane objects act as stationary distractors.

    Rule:
    - Keep every launcher marker fixed and identify the objects intersecting its row or column.
    - Along each launched lane, order those objects from nearest to farthest from the marker, then move each whole object away from the marker.
    - Every non-final object advances exactly until it becomes edge-adjacent to the next object's original position, and the final object advances to the grid boundary; all off-lane objects stay unchanged.

    Key insight:
    - The lane does not simply compress toward the wall: each move copies the next gap in the chain, so the sequence of inter-object gaps shifts one step outward and only the last gap becomes the boundary gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components = _find_components(g, bg)

    lanes: list[dict] = []
    marker_ids: set[int] = set()
    used_keys: set[tuple[str, int]] = set()
    for idx, comp in enumerate(components):
        if int(comp["area"]) != 1:
            continue
        r, c = map(int, np.asarray(comp["coords"], dtype=int)[0])
        if r not in (0, g.shape[0] - 1) and c not in (0, g.shape[1] - 1):
            continue
        lane = _marker_lane((r, c), g.shape)
        if lane["key"] in used_keys:
            raise ValueError("multiple launcher markers cannot select the same lane")
        lane["marker_index"] = idx
        lanes.append(lane)
        marker_ids.add(idx)
        used_keys.add(lane["key"])

    if not lanes:
        raise ValueError("expected at least one border launcher marker")

    moving_ids: set[int] = set()
    lane_members: list[list[dict]] = [[] for _ in lanes]
    for idx, comp in enumerate(components):
        if idx in marker_ids:
            continue
        matches = [lane_idx for lane_idx, lane in enumerate(lanes) if _component_on_lane(comp, lane)]
        if len(matches) > 1:
            raise ValueError("foreground object intersects multiple launch lanes")
        if len(matches) == 1:
            lane_members[matches[0]].append(comp)
            moving_ids.add(idx)

    out = g.copy()
    for idx in moving_ids:
        coords = np.asarray(components[idx]["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg

    placements: list[tuple[int, np.ndarray]] = []
    for lane, chain in zip(lanes, lane_members):
        if not chain:
            continue
        ordered = sorted(chain, key=lambda comp: _sort_key(comp, lane))
        for i, comp in enumerate(ordered):
            if i + 1 < len(ordered):
                steps = _steps_to_next(comp, ordered[i + 1], lane)
            else:
                steps = _steps_to_boundary(comp, lane, g.shape)
            if steps < 0:
                raise ValueError("launch lane objects must be separated in launch order")
            placements.append((int(comp["color"]), _shift_coords(comp["coords"], lane, steps)))

    for color, coords in placements:
        if not np.all(out[coords[:, 0], coords[:, 1]] == bg):
            raise ValueError("moved objects must not collide with markers or distractors")
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _sample_spaced_positions(rng: random.Random, values: list[int], count: int, min_gap: int) -> list[int] | None:
    shuffled = values[:]
    rng.shuffle(shuffled)
    chosen: list[int] = []
    for val in shuffled:
        if all(abs(val - other) >= min_gap for other in chosen):
            chosen.append(int(val))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _sample_chain_segments(rng: random.Random, span: int, count: int, max_len: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        lengths = [rng.randint(1, max_len) for _ in range(count)]
        gaps = [rng.randint(1, 4) for _ in range(count + 1)]
        used = sum(lengths) + sum(gaps)
        if used > span:
            continue
        extra = span - used
        for _ in range(extra):
            gaps[rng.randrange(len(gaps))] += 1

        cursor = 1 + gaps[0]
        segments: list[tuple[int, int]] = []
        for idx, length in enumerate(lengths):
            start = cursor
            end = start + length - 1
            segments.append((start, end))
            cursor = end + 1 + gaps[idx + 1]

        if segments[-1][1] <= span:
            return segments
    return None


def _mirror_segment(segment: tuple[int, int], edge_index: int) -> tuple[int, int]:
    start, end = map(int, segment)
    return int(edge_index - end), int(edge_index - start)


def _paint_bar(grid: np.ndarray, orientation: str, line: int, segment: tuple[int, int], color: int) -> None:
    start, end = map(int, segment)
    if orientation == "horizontal":
        grid[int(line), start : end + 1] = int(color)
    else:
        grid[start : end + 1, int(line)] = int(color)


def _padding_clear(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _place_distractors(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    orientation: str,
    active_lines: list[int],
    colors: list[int],
) -> bool:
    h, w = grid.shape
    forbidden = set(map(int, active_lines))
    for color in colors:
        placed = False
        for _ in range(120):
            rh = rng.randint(1, 3)
            rw = rng.randint(1, 3)
            if rh >= h - 2 or rw >= w - 2:
                continue
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            rows = set(range(r0, r0 + rh))
            cols = set(range(c0, c0 + rw))
            if orientation == "horizontal" and rows & forbidden:
                continue
            if orientation == "vertical" and cols & forbidden:
                continue
            if not _padding_clear(occupied, r0, c0, rh, rw, pad=1):
                continue
            grid[r0 : r0 + rh, c0 : c0 + rw] = int(color)
            occupied[r0 : r0 + rh, c0 : c0 + rw] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _choose_object_counts(rng: random.Random, lane_count: int) -> list[int]:
    min_total = 2 * lane_count
    max_total = min(7, 4 if lane_count == 1 else 3 * lane_count)
    total = rng.randint(min_total, max_total)
    counts = [2] * lane_count
    extra = total - min_total
    per_lane_cap = 4 if lane_count == 1 else 3
    while extra > 0:
        idx = rng.randrange(lane_count)
        if counts[idx] < per_lane_cap:
            counts[idx] += 1
            extra -= 1
    rng.shuffle(counts)
    return counts


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    orientation = rng.choice(("horizontal", "vertical"))
    h = rng.randint(11, 22)
    w = rng.randint(11, 22)
    lane_count = rng.randint(1, 3)

    if orientation == "horizontal":
        lines = _sample_spaced_positions(rng, list(range(1, h - 1)), lane_count, min_gap=2)
        span = w - 1
        max_len = 4
    else:
        lines = _sample_spaced_positions(rng, list(range(1, w - 1)), lane_count, min_gap=2)
        span = h - 1
        max_len = 4
    if lines is None:
        return None

    counts = _choose_object_counts(rng, lane_count)
    moving_total = sum(counts)
    max_distractors = min(2, 8 - moving_total)
    distractor_count = rng.randint(0, max_distractors) if max_distractors > 0 else 0

    needed_colors = 1 + moving_total + distractor_count
    if needed_colors > len(PALETTE):
        return None

    colors = list(rng.sample(PALETTE, needed_colors))
    marker_color = colors[0]
    moving_colors = colors[1 : 1 + moving_total]
    distractor_colors = colors[1 + moving_total :]

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    color_cursor = 0
    for line, count in zip(lines, counts):
        side = rng.choice(("positive", "negative"))
        segments = _sample_chain_segments(rng, span, count, max_len=max_len)
        if segments is None:
            return None

        if orientation == "horizontal":
            if side == "positive":
                marker = (int(line), 0)
                actual_segments = segments
            else:
                marker = (int(line), w - 1)
                actual_segments = [_mirror_segment(seg, w - 1) for seg in segments]
        else:
            if side == "positive":
                marker = (0, int(line))
                actual_segments = segments
            else:
                marker = (h - 1, int(line))
                actual_segments = [_mirror_segment(seg, h - 1) for seg in segments]

        if occupied[marker]:
            return None
        grid[marker] = marker_color
        occupied[marker] = True

        for segment in actual_segments:
            color = moving_colors[color_cursor]
            color_cursor += 1
            _paint_bar(grid, orientation, int(line), segment, color)
            start, end = map(int, segment)
            if orientation == "horizontal":
                occupied[int(line), start : end + 1] = True
            else:
                occupied[start : end + 1, int(line)] = True

    if not _place_distractors(rng, grid, occupied, orientation, lines, distractor_colors):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the chain-transfer launcher task")
