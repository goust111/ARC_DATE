from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
SIDE_ORDER = ("up", "down", "left", "right")
SIDE_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
MIN_GRID = 15
MAX_GRID = 21
MIN_CLEAR_RAY = 2

# Each arrow is a 6-cell 4-connected component: five grey pixels plus one colored marker.
ARROW_SPECS = {
    "right": {
        "shape": (3, 4),
        "grey": {(1, 0), (1, 2), (1, 3), (0, 1), (2, 1)},
        "marker": (1, 1),
        "tip": (1, 3),
    },
    "left": {
        "shape": (3, 4),
        "grey": {(1, 0), (1, 1), (1, 3), (0, 2), (2, 2)},
        "marker": (1, 2),
        "tip": (1, 0),
    },
    "down": {
        "shape": (4, 3),
        "grey": {(0, 1), (2, 1), (3, 1), (1, 0), (1, 2)},
        "marker": (1, 1),
        "tip": (3, 1),
    },
    "up": {
        "shape": (4, 3),
        "grey": {(0, 1), (1, 1), (3, 1), (2, 0), (2, 2)},
        "marker": (2, 1),
        "tip": (0, 1),
    },
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
                for dr, dc in SIDE_TO_VEC.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "expected a grey frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"
    assert np.all(grid[r0, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r1, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c0] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c1] == FRAME)
    return r0, c0, r1, c1


def _read_legends(grid: np.ndarray, frame: tuple[int, int, int, int]) -> dict[str, dict[str, object]]:
    r0, c0, r1, c1 = frame
    h, w = grid.shape
    assert 0 < r0 < r1 < h - 1 and 0 < c0 < c1 < w - 1, "frame must leave space for outer legends"

    legends: dict[str, dict[str, object]] = {}

    top_cols = [c for c in range(c0, c1 + 1) if int(grid[r0 - 1, c]) != BG]
    assert len(top_cols) == 2 and top_cols[1] == top_cols[0] + 1
    legends["up"] = {
        "replacement": int(grid[r0 - 1, top_cols[0]]),
        "source": int(grid[r0 - 1, top_cols[1]]),
        "coords": [(r0 - 1, top_cols[0]), (r0 - 1, top_cols[1])],
    }

    bottom_cols = [c for c in range(c0, c1 + 1) if int(grid[r1 + 1, c]) != BG]
    assert len(bottom_cols) == 2 and bottom_cols[1] == bottom_cols[0] + 1
    legends["down"] = {
        "replacement": int(grid[r1 + 1, bottom_cols[1]]),
        "source": int(grid[r1 + 1, bottom_cols[0]]),
        "coords": [(r1 + 1, bottom_cols[0]), (r1 + 1, bottom_cols[1])],
    }

    left_rows = [r for r in range(r0, r1 + 1) if int(grid[r, c0 - 1]) != BG]
    assert len(left_rows) == 2 and left_rows[1] == left_rows[0] + 1
    legends["left"] = {
        "replacement": int(grid[left_rows[0], c0 - 1]),
        "source": int(grid[left_rows[1], c0 - 1]),
        "coords": [(left_rows[0], c0 - 1), (left_rows[1], c0 - 1)],
    }

    right_rows = [r for r in range(r0, r1 + 1) if int(grid[r, c1 + 1]) != BG]
    assert len(right_rows) == 2 and right_rows[1] == right_rows[0] + 1
    legends["right"] = {
        "replacement": int(grid[right_rows[1], c1 + 1]),
        "source": int(grid[right_rows[0], c1 + 1]),
        "coords": [(right_rows[0], c1 + 1), (right_rows[1], c1 + 1)],
    }

    sources = [int(legends[side]["source"]) for side in SIDE_ORDER]
    assert len(set(sources)) == 4, "legend source colors must all differ"
    return legends


def _decode_arrow(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    assert len(coords) == 6, "each arrow component must have 6 cells"
    colors = [int(grid[r, c]) for r, c in coords]
    marker_colors = {color for color in colors if color != FRAME}
    assert len(marker_colors) == 1, "each arrow must contain exactly one marker color"
    source = int(next(iter(marker_colors)))
    marker_cells = [(r, c) for r, c in coords if int(grid[r, c]) == source]
    assert len(marker_cells) == 1

    pts = [(int(r), int(c)) for r, c in coords]
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c in pts:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    long_rows = [r for r, n in row_counts.items() if n == 4]
    long_cols = [c for c, n in col_counts.items() if n == 4]
    cross_rows = [r for r, n in row_counts.items() if n == 3]
    cross_cols = [c for c, n in col_counts.items() if n == 3]

    if len(long_rows) == 1 and len(cross_cols) == 1 and not long_cols and not cross_rows:
        row = int(long_rows[0])
        cross_c = int(cross_cols[0])
        line_cols = sorted(c for rr, c in pts if rr == row)
        cross_rs = sorted(rr for rr, cc in pts if cc == cross_c)
        assert line_cols == list(range(line_cols[0], line_cols[0] + 4))
        assert cross_rs == [row - 1, row, row + 1]
        left_dist = cross_c - line_cols[0]
        right_dist = line_cols[-1] - cross_c
        assert {left_dist, right_dist} == {1, 2}
        if left_dist < right_dist:
            return {"direction": "right", "source": source, "tip": (row, line_cols[-1])}
        return {"direction": "left", "source": source, "tip": (row, line_cols[0])}

    assert len(long_cols) == 1 and len(cross_rows) == 1 and not long_rows and not cross_cols
    col = int(long_cols[0])
    cross_r = int(cross_rows[0])
    line_rows = sorted(r for r, cc in pts if cc == col)
    cross_cs = sorted(cc for rr, cc in pts if rr == cross_r)
    assert line_rows == list(range(line_rows[0], line_rows[0] + 4))
    assert cross_cs == [col - 1, col, col + 1]
    up_dist = cross_r - line_rows[0]
    down_dist = line_rows[-1] - cross_r
    assert {up_dist, down_dist} == {1, 2}
    if up_dist < down_dist:
        return {"direction": "down", "source": source, "tip": (line_rows[-1], col)}
    return {"direction": "up", "source": source, "tip": (line_rows[0], col)}


def _trace_endpoint(grid: np.ndarray, tip: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = SIDE_TO_VEC[direction]
    rr, cc = tip[0] + dr, tip[1] + dc
    while 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
        if int(grid[rr, cc]) == FRAME:
            return int(rr), int(cc)
        assert int(grid[rr, cc]) == BG, "ray path must be empty until the frame"
        rr += dr
        cc += dc
    raise AssertionError("ray did not hit the frame")


def main(grid: np.ndarray) -> np.ndarray:
    """Use each arrow's facing direction to choose both a frame endpoint and an edge legend recolor.

    Input structure:
    - A grey one-pixel rectangular frame surrounds a black interior, with four two-pixel legends just outside the four sides.
    - Inside the frame are 2 or 3 asymmetric T-shaped arrows made of grey pixels plus one unique colored marker pixel each.
    - Each legend stores a replacement/source pair, and each arrow marker color matches the source color on the legend for the side it faces.

    Rule:
    - Decode each arrow direction from which bar end is farther from the crossbar; the crossbar is closer to the tail, so the arrow points the other way.
    - Cast a ray forward from the tip of the main bar to the grey frame cell it first hits, and read the legend on that same side for a replacement/source pair.
    - Recolor the hit frame cell with the legend replacement color, then erase every arrow pixel and every legend pixel back to black while leaving the rest of the frame unchanged.

    Key insight:
    - The arrow orientation is a fork node that simultaneously selects where to paint on the frame and which outer legend supplies the replacement color; both branches are needed to determine the final recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame = _frame_bbox(g)
    legends = _read_legends(g, frame)
    r0, c0, r1, c1 = frame

    interior_mask = np.zeros_like(g, dtype=bool)
    interior_mask[r0 + 1 : r1, c0 + 1 : c1] = g[r0 + 1 : r1, c0 + 1 : c1] != BG
    components = _components(interior_mask)
    assert 2 <= len(components) <= 3, "expected 2 or 3 interior arrows"

    out = g.copy()
    for side in SIDE_ORDER:
        for rr, cc in legends[side]["coords"]:
            out[int(rr), int(cc)] = BG

    seen_directions: set[str] = set()
    for coords in components:
        arrow = _decode_arrow(g, coords)
        direction = str(arrow["direction"])
        assert direction not in seen_directions, "arrow directions must be distinct"
        seen_directions.add(direction)

        legend = legends[direction]
        assert int(legend["source"]) == int(arrow["source"]), "arrow marker must match its side legend source color"
        endpoint = _trace_endpoint(g, arrow["tip"], direction)

        for rr, cc in coords:
            out[int(rr), int(cc)] = BG
        out[endpoint] = int(legend["replacement"])

    return out


def _legend_layout(frame: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame
    top_start = c0 + (c1 - c0 - 1) // 2
    side_start = r0 + (r1 - r0 - 1) // 2
    return {
        "up": [(r0 - 1, top_start), (r0 - 1, top_start + 1)],
        "down": [(r1 + 1, top_start), (r1 + 1, top_start + 1)],
        "left": [(side_start, c0 - 1), (side_start + 1, c0 - 1)],
        "right": [(side_start, c1 + 1), (side_start + 1, c1 + 1)],
    }


def _render_arrow(
    grid: np.ndarray,
    *,
    direction: str,
    top: int,
    left: int,
    marker_color: int,
) -> set[tuple[int, int]]:
    spec = ARROW_SPECS[direction]
    coords: set[tuple[int, int]] = set()
    for dr, dc in spec["grey"]:
        rr, cc = top + dr, left + dc
        grid[rr, cc] = FRAME
        coords.add((rr, cc))
    mr, mc = spec["marker"]
    grid[top + mr, left + mc] = marker_color
    coords.add((top + mr, left + mc))
    return coords


def _occupied_with_padding(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    padded = set(coords)
    for r, c in list(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    padded.add((rr, cc))
    return padded


def _candidate_positions(
    shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    direction: str,
) -> list[dict[str, object]]:
    grid_h, grid_w = shape
    r0, c0, r1, c1 = frame
    spec = ARROW_SPECS[direction]
    ph, pw = spec["shape"]
    tip_r, tip_c = spec["tip"]
    dr, dc = SIDE_TO_VEC[direction]
    candidates: list[dict[str, object]] = []

    for top in range(r0 + 1, r1 - ph + 1):
        for left in range(c0 + 1, c1 - pw + 1):
            coords = {(top + rr, left + cc) for rr, cc in spec["grey"] | {spec["marker"]}}
            tip = (top + tip_r, left + tip_c)
            endpoint = (tip[0] + dr, tip[1] + dc)
            clear_cells: list[tuple[int, int]] = []
            while r0 < endpoint[0] < r1 and c0 < endpoint[1] < c1:
                clear_cells.append(endpoint)
                endpoint = (endpoint[0] + dr, endpoint[1] + dc)

            if endpoint[0] not in (r0, r1) and endpoint[1] not in (c0, c1):
                continue
            if endpoint in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
                continue
            if len(clear_cells) < MIN_CLEAR_RAY:
                continue
            if not all(0 <= rr < grid_h and 0 <= cc < grid_w for rr, cc in coords):
                continue

            candidates.append(
                {
                    "top": top,
                    "left": left,
                    "coords": coords,
                    "padded": _occupied_with_padding(coords, shape),
                    "tip": tip,
                    "clear": set(clear_cells),
                    "endpoint": endpoint,
                }
            )
    return candidates


def _choose_directions() -> list[str]:
    if random.random() < 0.55:
        return list(random.choice(
            [
                ("up", "down", "left"),
                ("up", "down", "right"),
                ("up", "left", "right"),
                ("down", "left", "right"),
            ]
        ))
    return list(random.choice(
        [
            ("up", "left"),
            ("up", "right"),
            ("down", "left"),
            ("down", "right"),
            ("up", "down"),
            ("left", "right"),
        ]
    ))


def _render_input_from_plan(
    shape: tuple[int, int],
    source_by_side: dict[str, int],
    replacement_by_side: dict[str, int],
    arrows: list[dict[str, object]],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    grid[1 : h - 1, 1] = FRAME
    grid[1 : h - 1, w - 2] = FRAME
    grid[1, 1 : w - 1] = FRAME
    grid[h - 2, 1 : w - 1] = FRAME

    frame = (1, 1, h - 2, w - 2)
    legend_coords = _legend_layout(frame)
    for side in SIDE_ORDER:
        cells = legend_coords[side]
        replacement = int(replacement_by_side[side])
        source = int(source_by_side[side])
        if side == "up":
            values = [replacement, source]
        elif side == "down":
            values = [source, replacement]
        elif side == "left":
            values = [replacement, source]
        else:
            values = [source, replacement]
        for (rr, cc), value in zip(cells, values):
            grid[rr, cc] = value

    for arrow in arrows:
        _render_arrow(
            grid,
            direction=str(arrow["direction"]),
            top=int(arrow["top"]),
            left=int(arrow["left"]),
            marker_color=int(arrow["source"]),
        )
    return grid


def _oracle_output(
    inp: np.ndarray,
    replacement_by_side: dict[str, int],
    arrows: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    frame = (1, 1, inp.shape[0] - 2, inp.shape[1] - 2)
    for cells in _legend_layout(frame).values():
        for rr, cc in cells:
            out[rr, cc] = BG
    for arrow in arrows:
        for rr, cc in arrow["coords"]:
            out[int(rr), int(cc)] = BG
        out[tuple(arrow["endpoint"])] = int(replacement_by_side[str(arrow["direction"])])
    return out


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    source_by_side: dict[str, int],
    replacement_by_side: dict[str, int],
    arrows: list[dict[str, object]],
) -> bool:
    try:
        assert inp.ndim == 2 and out.ndim == 2
        assert max(inp.shape) <= 30 and max(out.shape) <= 30
        assert not np.array_equal(inp, out)
        assert len(arrows) in (2, 3)
        assert len({a["direction"] for a in arrows}) == len(arrows)
        assert len(set(source_by_side.values())) == 4
        assert len(set(replacement_by_side.values())) == 4
        assert set(source_by_side.values()).isdisjoint(replacement_by_side.values())

        frame = _frame_bbox(inp)
        assert frame == (1, 1, inp.shape[0] - 2, inp.shape[1] - 2)
        legends = _read_legends(inp, frame)

        interior_mask = np.zeros_like(inp, dtype=bool)
        interior_mask[2 : inp.shape[0] - 2, 2 : inp.shape[1] - 2] = inp[2 : inp.shape[0] - 2, 2 : inp.shape[1] - 2] != BG
        comps = _components(interior_mask)
        assert len(comps) == len(arrows)

        used_sources = set()
        seen_endpoints = set()
        for comp in comps:
            arrow = _decode_arrow(inp, comp)
            direction = str(arrow["direction"])
            endpoint = _trace_endpoint(inp, arrow["tip"], direction)
            used_sources.add(int(arrow["source"]))
            seen_endpoints.add(endpoint)
            assert int(legends[direction]["source"]) == int(arrow["source"]) == int(source_by_side[direction])
            assert int(legends[direction]["replacement"]) == int(replacement_by_side[direction])
        assert len(seen_endpoints) == len(arrows)

        unused_sides = set(SIDE_ORDER) - {str(a["direction"]) for a in arrows}
        assert unused_sides, "at least one legend must be unused"
        for side in unused_sides:
            assert int(source_by_side[side]) not in used_sources

        expected = _oracle_output(inp, replacement_by_side, arrows)
        assert np.array_equal(expected, out)
        assert np.array_equal(main(inp), out)
        return True
    except Exception:
        return False


def _sample_color_mapping() -> tuple[dict[str, int], dict[str, int]]:
    colors = random.sample(PALETTE, len(PALETTE))
    source_by_side = {side: colors[i] for i, side in enumerate(random.sample(SIDE_ORDER, len(SIDE_ORDER)))}
    replacement_by_side = {side: colors[i + 4] for i, side in enumerate(random.sample(SIDE_ORDER, len(SIDE_ORDER)))}
    return source_by_side, replacement_by_side


def _place_arrows(shape: tuple[int, int], directions: list[str], source_by_side: dict[str, int]) -> list[dict[str, object]] | None:
    frame = (1, 1, shape[0] - 2, shape[1] - 2)
    by_dir = {direction: _candidate_positions(shape, frame, direction) for direction in directions}
    if any(not cands for cands in by_dir.values()):
        return None

    order = sorted(directions, key=lambda side: len(by_dir[side]))
    occupied = set()
    blocked = set()
    arrows: list[dict[str, object]] = []
    vertical_cols: set[int] = set()
    horizontal_rows: set[int] = set()

    for direction in order:
        cands = by_dir[direction][:]
        random.shuffle(cands)
        chosen = None
        for cand in cands:
            coords = set(cand["coords"])
            if coords & blocked:
                continue
            if set(cand["clear"]) & occupied:
                continue
            endpoint = tuple(cand["endpoint"])
            if endpoint in {tuple(a["endpoint"]) for a in arrows}:
                continue
            tip = tuple(cand["tip"])
            if direction in ("up", "down"):
                if tip[1] in vertical_cols:
                    continue
            else:
                if tip[0] in horizontal_rows:
                    continue
            chosen = cand
            break
        if chosen is None:
            return None

        tip = tuple(chosen["tip"])
        if direction in ("up", "down"):
            vertical_cols.add(tip[1])
        else:
            horizontal_rows.add(tip[0])

        occupied |= set(chosen["coords"])
        blocked |= set(chosen["padded"])
        arrows.append(
            {
                "direction": direction,
                "top": int(chosen["top"]),
                "left": int(chosen["left"]),
                "coords": set(chosen["coords"]),
                "tip": tip,
                "endpoint": tuple(chosen["endpoint"]),
                "source": int(source_by_side[direction]),
            }
        )

    return arrows


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    directions = _choose_directions()
    source_by_side, replacement_by_side = _sample_color_mapping()
    arrows = _place_arrows((h, w), directions, source_by_side)
    if arrows is None:
        return None

    inp = _render_input_from_plan((h, w), source_by_side, replacement_by_side, arrows)
    out = _oracle_output(inp, replacement_by_side, arrows)
    if not _validate_instance(inp, out, source_by_side, replacement_by_side, arrows):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/10/027/7")
