from __future__ import annotations

import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _corner_starts(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for r0, c0 in _corner_starts(grid.shape):
        if r0 < 0 or c0 < 0 or r0 + 2 > h or c0 + 2 > w:
            continue
        legend = grid[r0 : r0 + 2, c0 : c0 + 2]
        if legend.shape != (2, 2) or np.any(legend == bg):
            continue

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r0 + 2)
        cc1 = min(w - 1, c0 + 2)
        ok = True
        for rr in range(rr0, rr1 + 1):
            for cc in range(cc0, cc1 + 1):
                if r0 <= rr <= r0 + 1 and c0 <= cc <= c0 + 1:
                    continue
                if int(grid[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((r0, c0, r0 + 1, c0 + 1))

    assert len(candidates) == 1, "expected one separated 2x2 legend in a corner"
    return candidates[0]


def _legend_pairs(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> dict[int, int]:
    r0, c0, r1, c1 = legend_bbox
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert legend.shape == (2, 2)
    assert len(set(map(int, legend.flat))) == 4, "legend colors must be distinct"

    pairs: dict[int, int] = {}
    for row in range(2):
        left = int(legend[row, 0])
        right = int(legend[row, 1])
        assert left != right
        pairs[left] = right
        pairs[right] = left
    return pairs


def _legend_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _marker_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    directions = []
    if r == 0:
        directions.append((1, 0))
    if r == h - 1:
        directions.append((-1, 0))
    if c == 0:
        directions.append((0, 1))
    if c == w - 1:
        directions.append((0, -1))
    assert len(directions) == 1, "markers must lie on exactly one border side"
    return directions[0]


def _walk_ray(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    cells = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _border_markers(
    grid: np.ndarray,
    bg: int,
    legend_bbox: tuple[int, int, int, int],
) -> list[dict]:
    h, w = grid.shape
    legend = _legend_mask(grid.shape, legend_bbox)
    border = np.zeros(grid.shape, dtype=bool)
    border[0, :] = True
    border[h - 1, :] = True
    border[:, 0] = True
    border[:, w - 1] = True

    coords = np.argwhere((grid != bg) & border & ~legend)
    markers = []
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        direction = _marker_direction(grid.shape, r, c)
        markers.append({"coord": (r, c), "color": int(grid[r, c]), "direction": direction})
    assert markers, "expected at least one edge marker"
    return markers


def _ray_reports(
    grid: np.ndarray,
    bg: int,
    markers: list[dict],
    mate: dict[int, int],
) -> list[dict]:
    reports = []
    for marker in markers:
        color = int(marker["color"])
        stop_color = int(mate[color])
        hits = []
        painted = []
        stop = None
        for rr, cc in _walk_ray(grid.shape, marker["coord"], marker["direction"]):
            val = int(grid[rr, cc])
            if val == bg:
                painted.append((rr, cc))
                continue
            blocking = val == stop_color
            hits.append({"coord": (rr, cc), "color": val, "blocking": blocking})
            if blocking:
                stop = (rr, cc)
                break
        reports.append(
            {
                "marker": marker["coord"],
                "color": color,
                "hits": hits,
                "stop": stop,
                "painted": painted,
            }
        )
    return reports


def main(grid: np.ndarray) -> np.ndarray:
    """Extend edge-marker rays inward, stopping only at the marker's legend-row partner color.

    Input structure:
    - One corner contains a separated 2x2 legend whose two rows define two color pairs.
    - Single-pixel markers sit on the outer border, and single-pixel interior blockers use legend colors.
    - The legend, markers, and blockers are surrounded by a dominant background.

    Rule:
    - Each border marker emits a straight ray perpendicular to its side and considers the original input as the reference state.
    - Paint the marker color onto background cells along that ray.
    - When the ray meets a non-background cell, stop only if that cell has the other color from the marker's legend row; otherwise skip over it and continue, leaving all original foreground cells unchanged.

    Key insight:
    - Blocking is defined by the 2x2 legend's row pairing rather than by geometry alone: the same visible obstacle can be a wall for one ray and transparent for another.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 4
    bg = get_bg(g)
    legend_bbox = _find_legend(g, bg)
    mate = _legend_pairs(g, legend_bbox)
    markers = _border_markers(g, bg, legend_bbox)

    legend = _legend_mask(g.shape, legend_bbox)
    border = np.zeros(g.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    fg_mask = g != bg
    marker_mask = np.zeros(g.shape, dtype=bool)
    for marker in markers:
        r, c = marker["coord"]
        marker_mask[r, c] = True
        assert marker["color"] in mate

    blocker_mask = fg_mask & ~legend & ~marker_mask
    if np.any(blocker_mask):
        assert np.all(~border[blocker_mask]), "blockers must be interior pixels"
        assert set(map(int, g[blocker_mask])) <= set(mate), "blockers must use legend colors"

    planned: dict[tuple[int, int], int] = {}
    for marker in markers:
        color = int(marker["color"])
        stop_color = int(mate[color])
        for rr, cc in _walk_ray(g.shape, marker["coord"], marker["direction"]):
            val = int(g[rr, cc])
            if val == bg:
                prev = planned.get((rr, cc))
                assert prev is None or prev == color, "conflicting ray paints are not allowed"
                planned[(rr, cc)] = color
                continue
            if val == stop_color:
                break

    out = g.copy()
    for (rr, cc), color in planned.items():
        out[rr, cc] = color
    return out


GRID_MIN = 13
GRID_MAX = 18
BG = 0
PALETTE = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _side_coord(side: str, pos: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return 0, pos
    if side == "bottom":
        return h - 1, pos
    if side == "left":
        return pos, 0
    assert side == "right"
    return pos, w - 1


def _side_direction(side: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[side]


def _legend_array(colors: list[int]) -> np.ndarray:
    return np.array(colors, dtype=int).reshape(2, 2)


def _put_legend(grid: np.ndarray, legend: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    starts = {
        "tl": (0, 0),
        "tr": (0, w - 2),
        "bl": (h - 2, 0),
        "br": (h - 2, w - 2),
    }
    r0, c0 = starts[corner]
    grid[r0 : r0 + 2, c0 : c0 + 2] = legend
    return (r0, c0, r0 + 1, c0 + 1)


def _choose_positions(shape: tuple[int, int]) -> dict[str, int]:
    h, w = shape
    top_col = random.randint(4, w - 5)
    bottom_choices = [c for c in range(4, w - 4) if c != top_col]
    bottom_col = random.choice(bottom_choices)

    left_row = random.randint(4, h - 7)
    right_row = random.randint(left_row + 2, h - 5)
    return {"top": top_col, "bottom": bottom_col, "left": left_row, "right": right_row}


def _trace_cells(shape: tuple[int, int], side: str, pos: int) -> list[tuple[int, int]]:
    return _walk_ray(shape, _side_coord(side, pos, shape), _side_direction(side))


def _nonblocking_colors(color: int, mate: dict[int, int]) -> list[int]:
    return [c for c in mate if c != mate[color]]


def _make_scene() -> tuple[np.ndarray, dict] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)
    grid = np.full(shape, BG, dtype=int)

    colors = random.sample(PALETTE, 4)
    legend = _legend_array(colors)
    mate = {
        int(legend[0, 0]): int(legend[0, 1]),
        int(legend[0, 1]): int(legend[0, 0]),
        int(legend[1, 0]): int(legend[1, 1]),
        int(legend[1, 1]): int(legend[1, 0]),
    }

    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_bbox = _put_legend(grid, legend, corner)
    positions = _choose_positions(shape)

    top_color = random.choice(colors)
    if random.random() < 0.5:
        left_color = mate[top_color]
    else:
        left_color = random.choice([c for c in colors if c not in {top_color, mate[top_color]}])
    remaining = [c for c in colors if c not in {top_color, left_color}]
    random.shuffle(remaining)
    bottom_color, right_color = remaining
    marker_colors = {
        "top": top_color,
        "bottom": bottom_color,
        "left": left_color,
        "right": right_color,
    }

    for side, pos in positions.items():
        r, c = _side_coord(side, pos, shape)
        grid[r, c] = marker_colors[side]

    occupied = {
        (r, c)
        for r in range(legend_bbox[0], legend_bbox[2] + 1)
        for c in range(legend_bbox[1], legend_bbox[3] + 1)
    }
    occupied.update(_side_coord(side, pos, shape) for side, pos in positions.items())

    cross = (positions["left"], positions["top"])
    if cross in occupied:
        return None

    top_nonblock_row_choices = [r for r in range(2, positions["left"]) if r != positions["right"]]
    if not top_nonblock_row_choices:
        return None
    top_nonblock = (random.choice(top_nonblock_row_choices), positions["top"])

    bottom_stop_row_choices = [r for r in range(positions["right"] + 1, h - 2)]
    if not bottom_stop_row_choices:
        return None
    bottom_stop = (random.choice(bottom_stop_row_choices), positions["bottom"])

    right_cols = [
        c
        for c in range(2, w - 2)
        if c not in {positions["top"], positions["bottom"]}
    ]
    if len(right_cols) < 2:
        return None
    random.shuffle(right_cols)
    right_nonblocks = [(positions["right"], right_cols[0]), (positions["right"], right_cols[1])]

    blocker_cells = [cross, top_nonblock, bottom_stop, *right_nonblocks]
    if len(set(blocker_cells)) != len(blocker_cells):
        return None
    if any(cell in occupied for cell in blocker_cells):
        return None

    grid[cross] = mate[top_color]
    top_pass_choices = [c for c in _nonblocking_colors(top_color, mate) if c != mate[top_color]]
    grid[top_nonblock] = random.choice(top_pass_choices)
    grid[bottom_stop] = mate[bottom_color]

    right_first_choices = [right_color, top_color, bottom_color, left_color]
    right_first_choices = [c for c in right_first_choices if c != mate[right_color]]
    grid[right_nonblocks[0]] = random.choice(right_first_choices)

    right_second_choices = [c for c in colors if c != mate[right_color] and c != int(grid[right_nonblocks[0]])]
    if not right_second_choices:
        right_second_choices = [c for c in colors if c != mate[right_color]]
    grid[right_nonblocks[1]] = random.choice(right_second_choices)

    reports = _ray_reports(grid, BG, _border_markers(grid, BG, legend_bbox), mate)
    report_by_side = {}
    for report in reports:
        r, c = report["marker"]
        if r == 0:
            report_by_side["top"] = report
        elif r == h - 1:
            report_by_side["bottom"] = report
        elif c == 0:
            report_by_side["left"] = report
        else:
            report_by_side["right"] = report

    if set(report_by_side) != {"top", "bottom", "left", "right"}:
        return None

    if report_by_side["top"]["stop"] != cross:
        return None
    if not report_by_side["top"]["hits"] or report_by_side["top"]["hits"][0]["coord"] != top_nonblock:
        return None
    if report_by_side["bottom"]["stop"] != bottom_stop:
        return None
    if report_by_side["left"]["stop"] is not None:
        return None
    if report_by_side["right"]["stop"] is not None:
        return None

    if len(report_by_side["left"]["hits"]) < 1 or report_by_side["left"]["hits"][0]["coord"] != cross:
        return None
    if len(report_by_side["right"]["hits"]) < 2:
        return None

    return grid, {"legend_bbox": legend_bbox, "mate": mate}


def _validate_generated_scene(grid: np.ndarray) -> bool:
    try:
        bg = get_bg(grid)
        legend_bbox = _find_legend(grid, bg)
        mate = _legend_pairs(grid, legend_bbox)
        markers = _border_markers(grid, bg, legend_bbox)
        reports = _ray_reports(grid, bg, markers, mate)
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    by_cell: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    blocked = 0
    passed = 0
    pass_then_block = 0
    for report in reports:
        if report["stop"] is not None:
            blocked += 1
        if any(not hit["blocking"] for hit in report["hits"]):
            passed += 1
        if report["stop"] is not None and any(not hit["blocking"] for hit in report["hits"]):
            pass_then_block += 1
        for hit in report["hits"]:
            by_cell.setdefault(hit["coord"], []).append((report["color"], hit["blocking"]))

    shared_selective = False
    for visits in by_cell.values():
        if len(visits) >= 2 and len({blocking for _color, blocking in visits}) >= 2:
            shared_selective = True
            break

    painted = int(np.sum(out != grid))
    marker_colors = {report["color"] for report in reports}
    legend_colors = set(mate)
    if marker_colors != legend_colors:
        return False
    if blocked < 2 or passed < 3 or pass_then_block < 1:
        return False
    if not shared_selective:
        return False
    if painted < 8:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _make_scene()
        if scene is None:
            continue
        grid, _meta = scene
        if _validate_generated_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid selective-blocking ray scene")
