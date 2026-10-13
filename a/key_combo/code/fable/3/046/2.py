from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

_DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_OPPOSITE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    out: list[list[Coord]] = []

    for sr, sc in np.argwhere(grid == int(color)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[Coord] = deque([(sr, sc)])
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(coords)
    return out


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[Coord]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _gap_side(box: tuple[int, int, int, int], gap: Coord) -> str | None:
    r0, c0, r1, c1 = box
    r, c = gap
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _frame_from_component(grid: np.ndarray, color: int, coords: list[Coord]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    cells = set(coords)
    border = _border_cells(r0, c0, r1, c1)
    if any(cell not in border for cell in cells):
        return None
    gaps = list(border - cells)
    if len(gaps) != 1:
        return None
    gap = gaps[0]
    side = _gap_side((r0, c0, r1, c1), gap)
    if side is None:
        return None
    if cells != border - {gap}:
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == int(color)):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "cells": cells,
        "gap": gap,
        "side": side,
        "dir": _DIR_BY_SIDE[side],
    }


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates: list[tuple[int, list[dict]]] = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        comps = _components_of_color(grid, color)
        frames: list[dict] = []
        ok = bool(comps)
        for comp in comps:
            frame = _frame_from_component(grid, color, comp)
            if frame is None:
                ok = False
                break
            frames.append(frame)
        if ok:
            frames.sort(key=lambda f: f["bbox"])
            candidates.append((int(color), frames))

    assert len(candidates) == 1, "expected exactly one color whose components are one-gap rectangular frames"
    return candidates[0]


def _inside_frame(frame: dict, r: int, c: int) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    return r0 < int(r) < r1 and c0 < int(c) < c1


def _border_lookup(frames: list[dict]) -> dict[Coord, int]:
    lookup: dict[Coord, int] = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        for cell in _border_cells(r0, c0, r1, c1):
            assert cell not in lookup, "frame borders must not overlap"
            lookup[cell] = idx
    return lookup


def _locate_marbles(grid: np.ndarray, bg: int, frame_color: int, frames: list[dict]) -> list[dict]:
    marbles: list[dict] = []
    occupied_frames: set[int] = set()
    marble_cells = np.argwhere((grid != int(bg)) & (grid != int(frame_color)))
    assert len(marble_cells) > 0, "expected at least one marble"

    for r0, c0 in marble_cells:
        r, c = int(r0), int(c0)
        owners = [idx for idx, frame in enumerate(frames) if _inside_frame(frame, r, c)]
        assert len(owners) == 1, "each marble must lie inside exactly one frame"
        frame_idx = owners[0]
        assert frame_idx not in occupied_frames, "a frame may contain at most one marble"
        occupied_frames.add(frame_idx)

        frame = frames[frame_idx]
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["dir"]
        if dr:
            assert c == gap_c, "vertical marble path must align with its frame gap"
        else:
            assert r == gap_r, "horizontal marble path must align with its frame gap"
        marbles.append(
            {
                "start": (r, c),
                "color": int(grid[r, c]),
                "dir": (dr, dc),
                "frame": frame_idx,
            }
        )
    return marbles


def _trace(shape: tuple[int, int], frames: list[dict], lookup: dict[Coord, int], start: Coord, direction: Coord) -> Coord:
    h, w = shape
    r, c = start
    dr, dc = direction

    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return r, c

        frame_idx = lookup.get((nr, nc))
        if frame_idx is not None:
            if (nr, nc) == frames[frame_idx]["gap"]:
                r, c = nr, nc
                continue
            return r, c

        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marble straight out through its own frame gap until a wall blocks it or a door lets it pass.

    Input structure:
    - The background is the most common color.
    - One shared frame color forms several separated hollow rectangular frames; every frame border has exactly one non-corner missing cell.
    - Some frame interiors contain one differently colored marble pixel, and that marble is row- or column-aligned with its own frame gap.
    - Marble rays are cardinal and do not intersect each other.

    Rule:
    - Erase every marble from its original cell.
    - Move each marble from its original cell along the line through its own frame gap and outward from that side.
    - A frame border cell blocks the marble, so it stops on the previous cell; a missing border gap is a door and the marble continues through it.
    - If the ray reaches the grid boundary, the marble stops on the last in-bounds cell. All frames and empty cells otherwise stay unchanged.

    Key insight:
    - The missing pixel is a door only when the ray is aligned with it; all other border pixels are walls, so the landing cell is determined by wall blocking versus gap passage.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, frames = _find_frames(g, bg)
    lookup = _border_lookup(frames)
    marbles = _locate_marbles(g, bg, frame_color, frames)

    out = g.copy()
    starts = {marble["start"] for marble in marbles}
    for r, c in starts:
        out[r, c] = int(bg)

    targets: dict[Coord, int] = {}
    for marble in marbles:
        end = _trace(g.shape, frames, lookup, marble["start"], marble["dir"])
        assert end not in starts, "a marble must move away from its original cell"
        assert end not in targets, "marble landing cells must be distinct"
        assert int(out[end]) == int(bg), "marble landing cell must be empty"
        targets[end] = int(marble["color"])

    for (r, c), color in targets.items():
        out[r, c] = int(color)
    return out


# --------------- generator ---------------

_GEN_COLORS = tuple(range(10))
_GEN_MIN_SIZE = 20
_GEN_MAX_SIZE = 30
_GEN_BRANCHES = ("edge", "wall", "door")
_GEN_MAX_ATTEMPTS = 500


def _gap_cell(r0: int, c0: int, height: int, width: int, side: str, offset: int) -> Coord:
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r0 + height - 1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    assert side == "right"
    return r0 + offset, c0 + width - 1


def _draw_frame(grid: np.ndarray, color: int, bg: int, r0: int, c0: int, height: int, width: int, side: str, offset: int) -> dict:
    r1, c1 = r0 + height - 1, c0 + width - 1
    assert 0 <= r0 < r1 < grid.shape[0] and 0 <= c0 < c1 < grid.shape[1]
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(region == int(bg)), "generated frames must not overlap"

    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    gap = _gap_cell(r0, c0, height, width, side, offset)
    grid[gap] = int(bg)

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "dir": _DIR_BY_SIDE[side],
    }


def _band_starts(total: int, sizes: list[int], min_gap: int = 1) -> list[int] | None:
    need = sum(sizes) + min_gap * (len(sizes) + 1)
    if need > total:
        return None
    gaps = [min_gap] * (len(sizes) + 1)
    for _ in range(total - need):
        gaps[random.randrange(len(gaps))] += 1

    starts: list[int] = []
    pos = gaps[0]
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size + gaps[i + 1]
    return starts


def _line_path(start: Coord, end: Coord, direction: Coord) -> list[Coord]:
    r, c = start
    er, ec = end
    dr, dc = direction
    path: list[Coord] = []
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))
    return path


def _wall_gap_horizontal(near_side: str, lane_offset: int, height: int, width: int) -> tuple[str, int]:
    choices: list[tuple[str, int]] = [
        (_OPPOSITE[near_side], random.randint(1, height - 2)),
        ("top", random.randint(1, width - 2)),
        ("bottom", random.randint(1, width - 2)),
    ]
    other_offsets = [offset for offset in range(1, height - 1) if offset != lane_offset]
    if other_offsets:
        choices.append((near_side, random.choice(other_offsets)))
    return random.choice(choices)


def _wall_gap_vertical(near_side: str, lane_offset: int, height: int, width: int) -> tuple[str, int]:
    choices: list[tuple[str, int]] = [
        (_OPPOSITE[near_side], random.randint(1, width - 2)),
        ("left", random.randint(1, height - 2)),
        ("right", random.randint(1, height - 2)),
    ]
    other_offsets = [offset for offset in range(1, width - 1) if offset != lane_offset]
    if other_offsets:
        choices.append((near_side, random.choice(other_offsets)))
    return random.choice(choices)


def _add_horizontal_track(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    marble_color: int,
    top: int,
    height: int,
    branch: str,
) -> dict | None:
    h, w = grid.shape
    source_w = random.randint(4, 6)
    obstacle_w = random.randint(4, 6)
    lane_offset = random.randint(1, height - 2)
    lane_r = top + lane_offset
    direction_name = random.choice(("left", "right"))

    if branch == "edge":
        if direction_name == "right":
            max_c0 = w - source_w - 4
            if max_c0 < 1:
                return None
            source_c0 = random.randint(1, max_c0)
            end = (lane_r, w - 1)
        else:
            max_c0 = w - source_w - 1
            if max_c0 < 3:
                return None
            source_c0 = random.randint(3, max_c0)
            end = (lane_r, 0)
        _draw_frame(grid, frame_color, bg, top, source_c0, height, source_w, direction_name, lane_offset)
    else:
        gap = random.randint(2, 5)
        total = source_w + obstacle_w + gap
        if total + 2 > w:
            return None
        if direction_name == "right":
            source_c0 = random.randint(1, w - total - 1)
            obstacle_c0 = source_c0 + source_w + gap
            near_side = "left"
            if branch == "wall":
                end = (lane_r, obstacle_c0 - 1)
            else:
                end = (lane_r, obstacle_c0 + obstacle_w - 2)
        else:
            obstacle_c0 = random.randint(1, w - total - 1)
            source_c0 = obstacle_c0 + obstacle_w + gap
            near_side = "right"
            if branch == "wall":
                end = (lane_r, obstacle_c0 + obstacle_w)
            else:
                end = (lane_r, obstacle_c0 + 1)

        _draw_frame(grid, frame_color, bg, top, source_c0, height, source_w, direction_name, lane_offset)
        if branch == "door":
            obstacle_side, obstacle_offset = near_side, lane_offset
        else:
            obstacle_side, obstacle_offset = _wall_gap_horizontal(near_side, lane_offset, height, obstacle_w)
        _draw_frame(grid, frame_color, bg, top, obstacle_c0, height, obstacle_w, obstacle_side, obstacle_offset)

    sr0, sc0, sr1, sc1 = top, source_c0, top + height - 1, source_c0 + source_w - 1
    marble_c = random.randint(sc0 + 1, sc1 - 1)
    if direction_name == "right":
        direction = (0, 1)
    else:
        direction = (0, -1)
    start = (lane_r, marble_c)
    grid[start] = int(marble_color)
    return {
        "start": start,
        "end": end,
        "dir": direction,
        "color": int(marble_color),
        "branch": branch,
        "path": _line_path(start, end, direction),
    }


def _add_vertical_track(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    marble_color: int,
    left: int,
    width: int,
    branch: str,
) -> dict | None:
    h, w = grid.shape
    source_h = random.randint(4, 6)
    obstacle_h = random.randint(4, 6)
    lane_offset = random.randint(1, width - 2)
    lane_c = left + lane_offset
    direction_name = random.choice(("top", "bottom"))

    if branch == "edge":
        if direction_name == "bottom":
            max_r0 = h - source_h - 4
            if max_r0 < 1:
                return None
            source_r0 = random.randint(1, max_r0)
            end = (h - 1, lane_c)
        else:
            max_r0 = h - source_h - 1
            if max_r0 < 3:
                return None
            source_r0 = random.randint(3, max_r0)
            end = (0, lane_c)
        _draw_frame(grid, frame_color, bg, source_r0, left, source_h, width, direction_name, lane_offset)
    else:
        gap = random.randint(2, 5)
        total = source_h + obstacle_h + gap
        if total + 2 > h:
            return None
        if direction_name == "bottom":
            source_r0 = random.randint(1, h - total - 1)
            obstacle_r0 = source_r0 + source_h + gap
            near_side = "top"
            if branch == "wall":
                end = (obstacle_r0 - 1, lane_c)
            else:
                end = (obstacle_r0 + obstacle_h - 2, lane_c)
        else:
            obstacle_r0 = random.randint(1, h - total - 1)
            source_r0 = obstacle_r0 + obstacle_h + gap
            near_side = "bottom"
            if branch == "wall":
                end = (obstacle_r0 + obstacle_h, lane_c)
            else:
                end = (obstacle_r0 + 1, lane_c)

        _draw_frame(grid, frame_color, bg, source_r0, left, source_h, width, direction_name, lane_offset)
        if branch == "door":
            obstacle_side, obstacle_offset = near_side, lane_offset
        else:
            obstacle_side, obstacle_offset = _wall_gap_vertical(near_side, lane_offset, obstacle_h, width)
        _draw_frame(grid, frame_color, bg, obstacle_r0, left, obstacle_h, width, obstacle_side, obstacle_offset)

    sr0, sc0, sr1, _sc1 = source_r0, left, source_r0 + source_h - 1, left + width - 1
    marble_r = random.randint(sr0 + 1, sr1 - 1)
    if direction_name == "bottom":
        direction = (1, 0)
    else:
        direction = (-1, 0)
    start = (marble_r, lane_c)
    grid[start] = int(marble_color)
    return {
        "start": start,
        "end": end,
        "dir": direction,
        "color": int(marble_color),
        "branch": branch,
        "path": _line_path(start, end, direction),
    }


def _expected_from_specs(grid: np.ndarray, bg: int, specs: list[dict]) -> np.ndarray | None:
    out = grid.copy()
    starts = {spec["start"] for spec in specs}
    ends: set[Coord] = set()
    for r, c in starts:
        out[r, c] = int(bg)
    for spec in specs:
        end = spec["end"]
        if end in starts or end in ends:
            return None
        if int(out[end]) != int(bg):
            return None
        ends.add(end)
        out[end] = int(spec["color"])
    return out


def _paths_are_disjoint(specs: list[dict]) -> bool:
    used: set[Coord] = set()
    for spec in specs:
        path = set(spec["path"])
        if used & path:
            return False
        used |= path
    return True


def _semantic_generated_ok(inp: np.ndarray, expected: np.ndarray, bg: int, specs: list[dict]) -> bool:
    if max(inp.shape) > 30:
        return False
    if _get_bg(inp) != int(bg):
        return False
    if set(spec["branch"] for spec in specs) != set(_GEN_BRANCHES):
        return False
    if not _paths_are_disjoint(specs):
        return False
    if np.array_equal(inp, expected):
        return False

    h, w = inp.shape
    for spec in specs:
        er, ec = spec["end"]
        dr, dc = spec["dir"]
        branch = spec["branch"]
        if branch == "edge" and er not in (0, h - 1) and ec not in (0, w - 1):
            return False
        if branch == "wall":
            nr, nc = er + dr, ec + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return False
        if branch == "door":
            if er in (0, h - 1) or ec in (0, w - 1):
                return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(_GEN_MIN_SIZE, _GEN_MAX_SIZE)
    width = random.randint(_GEN_MIN_SIZE, _GEN_MAX_SIZE)
    bg = random.choice(_GEN_COLORS)
    palette = [color for color in _GEN_COLORS if color != bg]
    frame_color = random.choice(palette)
    marble_palette = [color for color in palette if color != frame_color]

    branches = list(_GEN_BRANCHES)
    if random.random() < 0.45:
        branches.append(random.choice(_GEN_BRANCHES))
    random.shuffle(branches)

    grid = np.full((height, width), int(bg), dtype=int)
    specs: list[dict] = []
    orientation = random.choice(("horizontal", "vertical"))
    marble_colors = random.sample(marble_palette, len(branches))

    if orientation == "horizontal":
        sizes = [random.randint(4, 6) for _ in branches]
        starts = _band_starts(height, sizes)
        if starts is None:
            return None
        for top, track_h, branch, marble_color in zip(starts, sizes, branches, marble_colors):
            spec = _add_horizontal_track(grid, bg, frame_color, marble_color, top, track_h, branch)
            if spec is None:
                return None
            specs.append(spec)
    else:
        sizes = [random.randint(4, 6) for _ in branches]
        starts = _band_starts(width, sizes)
        if starts is None:
            return None
        for left, track_w, branch, marble_color in zip(starts, sizes, branches, marble_colors):
            spec = _add_vertical_track(grid, bg, frame_color, marble_color, left, track_w, branch)
            if spec is None:
                return None
            specs.append(spec)

    expected = _expected_from_specs(grid, bg, specs)
    if expected is None:
        return None
    if not _semantic_generated_ok(grid, expected, bg, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid door-and-marble grid")
