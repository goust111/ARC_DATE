from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
SIDE_FROM_DELTA = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}
GEN_RETRIES = 400


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    kind: str
    gaps: tuple[str, ...]
    gap_pos: dict[str, tuple[int, int]]
    center: tuple[int, int]
    coords: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class SignalTrace:
    outcome: str
    final_pos: tuple[int, int] | None
    final_color: int | None
    pipes: tuple[int, ...]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[tuple[int, set[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, set[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "U"
    if r == r1:
        return "D"
    if c == c0:
        return "L"
    if c == c1:
        return "R"
    raise AssertionError("gap must lie on the border")


def _is_corner(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return (r in (r0, r1)) and (c in (c0, c1))


def _detect_frames(grid: np.ndarray, bg: int) -> tuple[list[Frame], dict[tuple[int, int], Frame], set[tuple[int, int]]]:
    frames: list[Frame] = []
    gap_map: dict[tuple[int, int], Frame] = {}
    walls: set[tuple[int, int]] = set()
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        cells = {tuple(map(int, rc)) for rc in coords}
        if len(cells) < 8:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if (r1 - r0) % 2 != 0 or (c1 - c0) % 2 != 0:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if not cells <= border:
            continue
        gaps = tuple(sorted(border - cells))
        if len(gaps) not in (1, 2):
            continue
        if any(_is_corner(gap, (r0, c0, r1, c1)) for gap in gaps):
            continue
        sides = tuple(_gap_side(gap, (r0, c0, r1, c1)) for gap in gaps)
        if len(gaps) == 2:
            if len(set(sides)) != 2:
                continue
            if set(sides) in ({"U", "D"}, {"L", "R"}):
                continue
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        kind = "room" if len(gaps) == 1 else "pipe"
        frame = Frame(
            color=int(color),
            bbox=(r0, c0, r1, c1),
            kind=kind,
            gaps=tuple(sides),
            gap_pos={side: gap for side, gap in zip(sides, gaps)},
            center=center,
            coords=frozenset(cells),
        )
        frames.append(frame)
        walls |= cells
        for gap in gaps:
            assert gap not in gap_map, "gaps must be unique"
            gap_map[gap] = frame
    assert frames, "expected at least one rectangular frame"
    return frames, gap_map, walls


def _find_beacons(grid: np.ndarray, bg: int, frames: list[Frame]) -> list[tuple[Frame, int]]:
    frame_cells = set().union(*(frame.coords for frame in frames))
    centers = {frame.center: frame for frame in frames if frame.kind == "room"}
    beacons: list[tuple[Frame, int]] = []
    for pos, frame in centers.items():
        value = int(grid[pos])
        if value == bg:
            continue
        assert pos not in frame_cells
        assert value != frame.color
        beacons.append((frame, value))
    foreground = {(r, c) for r, c in zip(*np.where(grid != bg))}
    allowed = frame_cells | {frame.center for frame, _color in beacons}
    assert foreground <= allowed, "unexpected non-background pixels outside frames or room centers"
    assert beacons, "expected at least one beacon"
    return beacons


def _trace_signal(
    grid: np.ndarray,
    gap_map: dict[tuple[int, int], Frame],
    walls: set[tuple[int, int]],
    source_room: Frame,
    start_color: int,
) -> SignalTrace:
    h, w = grid.shape
    gap_side = source_room.gaps[0]
    dr, dc = DIRS[gap_side]
    gap_r, gap_c = source_room.gap_pos[gap_side]
    cur_r, cur_c = gap_r + dr, gap_c + dc
    color = int(start_color)
    pipes: list[int] = []

    while True:
        nr, nc = cur_r + dr, cur_c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return SignalTrace("delete", None, None, tuple(pipes))
        next_pos = (nr, nc)
        frame = gap_map.get(next_pos)
        if frame is not None:
            seen_side = SIDE_FROM_DELTA[(-dr, -dc)]
            if seen_side in frame.gap_pos and frame.gap_pos[seen_side] == next_pos:
                if frame.kind == "room":
                    return SignalTrace("room", frame.center, color, tuple(pipes))
                pipes.append(frame.color)
                color = frame.color
                other_side = next(side for side in frame.gaps if side != seen_side)
                dr, dc = DIRS[other_side]
                gap_r, gap_c = frame.gap_pos[other_side]
                cur_r, cur_c = gap_r + dr, gap_c + dc
                if not (0 <= cur_r < h and 0 <= cur_c < w):
                    return SignalTrace("delete", None, None, tuple(pipes))
                continue
        if next_pos in walls:
            return SignalTrace("wall", (cur_r, cur_c), color, tuple(pipes))
        cur_r, cur_c = nr, nc


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], SignalTrace]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    frames, gap_map, walls = _detect_frames(g, bg)
    beacons = _find_beacons(g, bg, frames)

    out = g.copy()
    traces: dict[tuple[int, int], SignalTrace] = {}
    finals: dict[tuple[int, int], int] = {}

    for room, _color in beacons:
        out[room.center] = bg

    for room, color in beacons:
        trace = _trace_signal(g, gap_map, walls, room, color)
        traces[room.center] = trace
        if trace.final_pos is None:
            continue
        assert trace.final_pos not in finals, "two beacons cannot finish in the same cell"
        finals[trace.final_pos] = int(trace.final_color)

    for pos, color in finals.items():
        assert out[pos] == bg, "final beacon must land on background"
        out[pos] = color
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each room-center beacon out through its room gap, relay it through bend pipes, and stop it at the first valid destination.

    Input structure:
    - The grid has a dominant background plus several one-cell-thick monochrome rectangular frames.
    - Each frame is either a room with exactly one non-corner gap or a bend pipe with exactly two non-corner gaps on adjacent sides.
    - Some room centers contain a single beacon pixel whose color differs from that room frame; other room centers are empty.

    Rule:
    - Every beacon leaves its source room through that room's gap and travels straight in the gap's outward direction.
    - If the next encountered border cell is a wall, the beacon stops on the background cell immediately outside that wall.
    - If the beacon reaches a frame through one of its gaps, it enters that frame: entering a pipe recolors the beacon to the pipe color and sends it out through the other gap, while entering a room stops it at that room's center.
    - A beacon that exits the grid disappears; all frames stay unchanged, source centers are cleared, and only the final beacon positions remain.

    Key insight:
    - A gap is a directional doorway, not a leak: single-gap frames are launch or landing rooms, while two-gap frames both turn the travel direction and overwrite the beacon color.
    """
    out, _traces = _solve(grid)
    return out


def _port_cell(center: tuple[int, int], side: str, offset: int, *, size: int = 5) -> tuple[int, int]:
    half = size // 2
    cr, cc = center
    if side == "U":
        return cr - half, cc + offset
    if side == "D":
        return cr + half, cc + offset
    if side == "L":
        return cr + offset, cc - half
    if side == "R":
        return cr + offset, cc + half
    raise AssertionError("unknown side")


def _draw_frame_gaps(
    grid: np.ndarray,
    center: tuple[int, int],
    color: int,
    ports: tuple[tuple[str, int], ...],
    *,
    size: int = 5,
) -> None:
    half = size // 2
    cr, cc = center
    r0, c0 = cr - half, cc - half
    r1, c1 = cr + half, cc + half
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    for side, offset in ports:
        grid[_port_cell(center, side, offset, size=size)] = 0


def _transform_grid(grid: np.ndarray, rot: int, mirror: bool) -> np.ndarray:
    out = np.rot90(grid, rot)
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _pad_grid(grid: np.ndarray, top: int, bottom: int, left: int, right: int, *, bg: int = 0) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h + top + bottom, w + left + right), bg, dtype=int)
    out[top : top + h, left : left + w] = grid
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    base_h = base_w = 26
    grid = np.zeros((base_h, base_w), dtype=int)
    palette = random.sample(range(1, 10), 9)
    beacon_color = palette[-1]
    frame_colors = palette[:-1]
    centers = {
        "source_wall": (10, 2),
        "sink_chain": (10, 10),
        "source_delete": (4, 18),
        "pipe1": (17, 2),
        "pipe2": (19, 10),
        "sink_direct": (14, 18),
        "source_chain": (23, 2),
        "source_direct": (23, 18),
    }
    room_ports = {
        "source_wall": (("R", 0),),
        "sink_chain": (("D", -1),),
        "source_delete": (("U", 0),),
        "sink_direct": (("D", 0),),
        "source_chain": (("U", 1),),
        "source_direct": (("U", 0),),
    }
    room_color_map = {
        "source_wall": frame_colors[0],
        "sink_chain": frame_colors[1],
        "source_delete": frame_colors[2],
        "sink_direct": frame_colors[3],
        "source_chain": frame_colors[4],
        "source_direct": frame_colors[5],
    }
    beacons = {
        "source_wall": beacon_color,
        "source_delete": beacon_color,
        "source_chain": beacon_color,
        "source_direct": beacon_color,
    }
    pipe_ports = {
        "pipe1": (("D", 1), ("R", 1)),
        "pipe2": (("L", -1), ("U", -1)),
    }

    for role, ports in room_ports.items():
        _draw_frame_gaps(grid, centers[role], room_color_map[role], ports)
    for role, ports in pipe_ports.items():
        _draw_frame_gaps(grid, centers[role], frame_colors[6] if role == "pipe1" else frame_colors[7], ports)
    for role, color in beacons.items():
        grid[centers[role]] = color

    expected = grid.copy()
    for role in beacons:
        expected[centers[role]] = 0
    expected[centers["sink_chain"]] = frame_colors[7]
    expected[centers["sink_direct"]] = beacons["source_direct"]
    expected[10, 7] = beacons["source_wall"]

    rot = random.randint(0, 3)
    mirror = bool(random.randint(0, 1))
    grid = _transform_grid(grid, rot, mirror)
    expected = _transform_grid(expected, rot, mirror)

    h = random.randint(26, 29)
    w = random.randint(26, 29)
    extra_h = h - base_h
    extra_w = w - base_w
    top = random.randint(0, extra_h)
    bottom = extra_h - top
    left = random.randint(0, extra_w)
    right = extra_w - left
    grid = _pad_grid(grid, top, bottom, left, right)
    expected = _pad_grid(expected, top, bottom, left, right)
    return grid, expected


def _semantic_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved, traces = _solve(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    by_outcome = Counter(trace.outcome for trace in traces.values())
    if by_outcome != Counter({"room": 2, "wall": 1, "delete": 1}):
        return False

    pipe_lens = sorted(len(trace.pipes) for trace in traces.values())
    if pipe_lens != [0, 0, 0, 2]:
        return False

    if len({trace.final_pos for trace in traces.values() if trace.final_pos is not None}) != 3:
        return False

    room_traces = [trace for trace in traces.values() if trace.outcome == "room"]
    wall_traces = [trace for trace in traces.values() if trace.outcome == "wall"]
    if len(room_traces) != 2 or len(wall_traces) != 1:
        return False
    direct_room = [trace for trace in room_traces if len(trace.pipes) == 0]
    relayed_room = [trace for trace in room_traces if len(trace.pipes) == 2]
    if len(direct_room) != 1 or len(relayed_room) != 1:
        return False
    if wall_traces[0].final_color != direct_room[0].final_color:
        return False
    if relayed_room[0].final_color == direct_room[0].final_color:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, expected = _build_case()
        if _semantic_ok(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid gate-pipe puzzle input")
