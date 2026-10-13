from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIN_FRAME_SPAN = 5
FRAME_SPANS = (5, 7, 9)
GRID_H_RANGE = (16, 27)
GRID_W_RANGE = (16, 27)
FRAME_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
MIN_SEG_LEN = 2
FINAL_MIN_RUN = 2
FUTURE_MIN_RUN = MIN_SEG_LEN + MIN_FRAME_SPAN
MAX_GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class FrameSpec:
    bbox: tuple[int, int, int, int]
    color: int
    center: tuple[int, int]
    pointer: tuple[int, int]
    pointer_dir: tuple[int, int]


@dataclass(frozen=True)
class CaseSpec:
    shape: tuple[int, int]
    bg: int
    bar_color: int
    control_color: int
    pointer_color: int
    bar_cells: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    center: tuple[int, int]
    control: tuple[int, int]
    init_dir: tuple[int, int]
    frames: tuple[FrameSpec, ...]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _opposite(step: tuple[int, int]) -> tuple[int, int]:
    return -step[0], -step[1]


def _bar_cells(center: tuple[int, int], step: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    r, c = center
    if step[0] == 0:
        return ((r - 1, c), (r, c), (r + 1, c))
    return ((r, c - 1), (r, c), (r, c + 1))


def _find_marker_setup(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]]:
    candidates: list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for comp in _components_of_color(grid, color):
            if len(comp) != 3:
                continue
            rows = sorted(r for r, _ in comp)
            cols = sorted(c for _, c in comp)
            if len(set(rows)) == 1 and cols == list(range(cols[0], cols[0] + 3)):
                center = (rows[0], cols[1])
            elif len(set(cols)) == 1 and rows == list(range(rows[0], rows[0] + 3)):
                center = (rows[1], cols[0])
            else:
                continue
            controls = []
            for nr, nc in neighbors4(*center):
                if not in_bounds(grid.shape, nr, nc):
                    continue
                if (nr, nc) in comp:
                    continue
                val = int(grid[nr, nc])
                if val != bg and val != color:
                    controls.append((nr, nc))
            if len(controls) == 1:
                control = controls[0]
                step = (center[0] - control[0], center[1] - control[1])
                candidates.append((color, center, control, step))
    assert len(candidates) == 1, "expected a unique marker bar and control"
    return candidates[0]


def _detect_frames(grid: np.ndarray, bg: int) -> list[FrameSpec]:
    bar_color, _center, _control, _step = _find_marker_setup(grid, bg)
    frames: list[FrameSpec] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg, bar_color}):
        for comp in _components_of_color(grid, color):
            if len(comp) < 16:
                continue
            bbox = bbox_coords(comp)
            r0, c0, r1, c1 = bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < MIN_FRAME_SPAN or w < MIN_FRAME_SPAN or h % 2 == 0 or w % 2 == 0:
                continue
            border = _border_cells(bbox)
            if set(comp) != border:
                continue
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            interior_marks = [
                (r, c)
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
                if int(grid[r, c]) != bg and int(grid[r, c]) != color
            ]
            assert len(interior_marks) == 1, "each frame must contain exactly one pointer pixel"
            pointer = interior_marks[0]
            assert pointer != center, "pointer cannot sit at the frame center"
            assert (pointer[0] == center[0]) ^ (pointer[1] == center[1]), "pointer must be cardinally aligned"
            if pointer[0] == center[0]:
                assert c0 < pointer[1] < c1, "pointer must remain inside the frame"
                pointer_dir = (0, 1 if pointer[1] > center[1] else -1)
            else:
                assert r0 < pointer[0] < r1, "pointer must remain inside the frame"
                pointer_dir = (1 if pointer[0] > center[0] else -1, 0)
            frames.append(
                FrameSpec(
                    bbox=bbox,
                    color=color,
                    center=center,
                    pointer=pointer,
                    pointer_dir=pointer_dir,
                )
            )
    assert frames, "expected at least one router frame"
    return frames


def _entry_cell(anchor: tuple[int, int], step: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    ar, ac = anchor
    if step == (0, 1):
        return ar, c0
    if step == (0, -1):
        return ar, c1
    if step == (1, 0):
        return r0, ac
    if step == (-1, 0):
        return r1, ac
    raise AssertionError("invalid step")


def _corridor_cells(anchor: tuple[int, int], step: tuple[int, int], stop_before: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = anchor[0] + step[0], anchor[1] + step[1]
    while (r, c) != stop_before:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _frame_exit_border(frame: FrameSpec, step: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame.bbox
    r, c = frame.center
    while True:
        nr, nc = r + step[0], c + step[1]
        if not _inside_bbox((nr, nc), frame.bbox):
            return r, c
        r, c = nr, nc


def _ray_cells_to_edge(anchor: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = anchor[0] + step[0], anchor[1] + step[1]
    while in_bounds(shape, r, c):
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _free_run(anchor: tuple[int, int], step: tuple[int, int], shape: tuple[int, int], blocked: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> int:
    run = 0
    for cell in _ray_cells_to_edge(anchor, step, shape):
        if cell in blocked or cell in reserved:
            break
        run += 1
    return run


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, tuple[int, int], tuple[int, int], list[FrameSpec]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bar_color, center, control, init_dir = _find_marker_setup(g, bg)
    frames = _detect_frames(g, bg)

    border_map: dict[tuple[int, int], int] = {}
    for i, frame in enumerate(frames):
        for cell in _border_cells(frame.bbox):
            assert cell not in border_map, "frame borders must not overlap"
            border_map[cell] = i

    h, w = g.shape
    pos = center
    step = init_dir
    seen_states = set()
    seen_frames: list[int] = []
    while True:
        state = (pos, step)
        assert state not in seen_states, "route loop detected"
        seen_states.add(state)
        nr, nc = pos[0] + step[0], pos[1] + step[1]
        if not in_bounds((h, w), nr, nc):
            break
        if int(g[nr, nc]) == bg:
            pos = (nr, nc)
            continue
        idx = border_map.get((nr, nc))
        assert idx is not None, "the ray may only hit frame borders"
        seen_frames.append(idx)
        frame = frames[idx]
        pos = _frame_exit_border(frame, frame.pointer_dir)
        step = frame.pointer_dir
    assert seen_frames, "expected the route to hit at least one frame"
    assert len(set(seen_frames)) == len(seen_frames), "route should visit frames at most once"

    ordered = [frames[i] for i in seen_frames]
    used_borders = set()
    for frame in ordered:
        for cell in _border_cells(frame.bbox):
            assert cell not in used_borders
        used_borders.update(_border_cells(frame.bbox))
    return bg, bar_color, center, init_dir, ordered


def _simulate_from_frames(
    grid: np.ndarray,
    bar_color: int,
    center: tuple[int, int],
    init_dir: tuple[int, int],
    frames: list[FrameSpec],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    h, w = g.shape

    border_to_frame: dict[tuple[int, int], FrameSpec] = {}
    for frame in frames:
        for cell in _border_cells(frame.bbox):
            border_to_frame[cell] = frame

    pos = center
    step = init_dir
    color = bar_color
    for _ in range(h * w * 8):
        nr, nc = pos[0] + step[0], pos[1] + step[1]
        if not in_bounds((h, w), nr, nc):
            return out
        if int(g[nr, nc]) == bg:
            out[nr, nc] = color
            pos = (nr, nc)
            continue
        frame = border_to_frame.get((nr, nc))
        assert frame is not None, "route can only enter a router frame"
        color = frame.color
        pos = _frame_exit_border(frame, frame.pointer_dir)
        step = frame.pointer_dir
    raise AssertionError("simulation exceeded step limit")


def _expected_from_case(case: CaseSpec) -> np.ndarray:
    grid = np.full(case.shape, case.bg, dtype=int)
    for r, c in case.bar_cells:
        grid[r, c] = case.bar_color
    grid[case.control] = case.control_color
    for frame in case.frames:
        for r, c in _border_cells(frame.bbox):
            grid[r, c] = frame.color
        grid[frame.pointer] = case.pointer_color
    return _simulate_from_frames(grid, case.bar_color, case.center, case.init_dir, list(case.frames))


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Cast a ray from the marker bar, reroute it through hollow frames, and recolor each segment by the frame that relays it.

    Input structure:
    - The background is black.
    - One three-cell straight bar has a differently colored control pixel touching its center; the launch direction is away from that control.
    - Several non-overlapping hollow rectangular frames appear elsewhere, each with a single interior pointer pixel on a cardinal line from the frame center.

    Rule:
    - Start from the bar center and paint background cells along the launch ray using the bar color until the ray hits a frame border.
    - When the ray enters a frame, stop painting, jump to that frame center, change direction to the frame pointer's relative direction, and change the segment color to the frame color.
    - Leave the frame without altering the frame, pointer, bar, or control, and keep relaying until the ray exits the grid.

    Key insight:
    - Each hollow frame is a router rather than an obstacle: its border marks the interception point, while its interior marker decides the next outgoing direction and the next segment color.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, bar_color, center, init_dir, ordered_frames = _analyze_grid(g)
    _ = bg
    return _simulate_from_frames(g, bar_color, center, init_dir, ordered_frames)


def _bar_setup(shape: tuple[int, int], bar_color: int, control_color: int) -> tuple[tuple[tuple[int, int], ...], tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = shape
    init_dir = random.choice(DIRS)
    if init_dir == (0, 1):
        center = (random.randint(3, h - 4), random.randint(2, 4))
    elif init_dir == (0, -1):
        center = (random.randint(3, h - 4), random.randint(w - 5, w - 3))
    elif init_dir == (1, 0):
        center = (random.randint(2, 4), random.randint(3, w - 4))
    else:
        center = (random.randint(h - 5, h - 3), random.randint(3, w - 4))
    control = (center[0] - init_dir[0], center[1] - init_dir[1])
    bar_cells = _bar_cells(center, init_dir)
    assert all(in_bounds(shape, r, c) for r, c in bar_cells)
    assert in_bounds(shape, *control)
    return bar_cells, center, control, init_dir


def _candidate_boxes(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    step: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    out: list[tuple[int, int, int, int]] = []
    spans = [(hh, ww) for hh in FRAME_SPANS for ww in FRAME_SPANS if hh <= h - 2 and ww <= w - 2]
    random.shuffle(spans)
    for fh, fw in spans:
        if step == (0, 1):
            for c0 in range(anchor[1] + MIN_SEG_LEN + 1, w - fw):
                low = max(1, anchor[0] - fh + 1)
                high = min(anchor[0], h - fh - 1)
                for r0 in range(low, high + 1):
                    out.append((r0, c0, r0 + fh - 1, c0 + fw - 1))
        elif step == (0, -1):
            for c0 in range(1, anchor[1] - MIN_SEG_LEN - fw):
                low = max(1, anchor[0] - fh + 1)
                high = min(anchor[0], h - fh - 1)
                for r0 in range(low, high + 1):
                    out.append((r0, c0, r0 + fh - 1, c0 + fw - 1))
        elif step == (1, 0):
            for r0 in range(anchor[0] + MIN_SEG_LEN + 1, h - fh):
                low = max(1, anchor[1] - fw + 1)
                high = min(anchor[1], w - fw - 1)
                for c0 in range(low, high + 1):
                    out.append((r0, c0, r0 + fh - 1, c0 + fw - 1))
        else:
            for r0 in range(1, anchor[0] - MIN_SEG_LEN - fh):
                low = max(1, anchor[1] - fw + 1)
                high = min(anchor[1], w - fw - 1)
                for c0 in range(low, high + 1):
                    out.append((r0, c0, r0 + fh - 1, c0 + fw - 1))
    random.shuffle(out)
    return out


def _pick_pointer(center: tuple[int, int], bbox: tuple[int, int, int, int], step: tuple[int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    if step[0] != 0:
        span = (r1 - r0) // 2 - 1
    else:
        span = (c1 - c0) // 2 - 1
    if span < 1:
        return None
    dist = random.randint(1, span)
    return center[0] + step[0] * dist, center[1] + step[1] * dist


def _choose_frame(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    incoming: tuple[int, int],
    color: int,
    blocked: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    remaining_after: int,
    require_turn: bool,
) -> FrameSpec | None:
    out_dirs = [step for step in DIRS if step != _opposite(incoming)]
    if require_turn:
        out_dirs = [step for step in out_dirs if step != incoming]
    random.shuffle(out_dirs)
    for bbox in _candidate_boxes(shape, anchor, incoming):
        bbox_cells = _bbox_cells(bbox)
        if bbox_cells & forbidden:
            continue
        if bbox_cells & reserved:
            continue
        corridor = _corridor_cells(anchor, incoming, _entry_cell(anchor, incoming, bbox))
        if len(corridor) < MIN_SEG_LEN:
            continue
        if any(cell in blocked or cell in reserved for cell in corridor):
            continue
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        for step in out_dirs:
            pointer = _pick_pointer(center, bbox, step)
            if pointer is None:
                continue
            candidate = FrameSpec(bbox=bbox, color=color, center=center, pointer=pointer, pointer_dir=step)
            exit_border = _frame_exit_border(candidate, step)
            need_run = FUTURE_MIN_RUN if remaining_after > 0 else FINAL_MIN_RUN
            run = _free_run(exit_border, step, shape, blocked | bbox_cells, reserved)
            if run < need_run:
                continue
            return candidate
    return None


def _build_case() -> CaseSpec | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)
    bg = 0
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    total_colors = n_frames + 3
    palette = random.sample(range(1, 10), total_colors)
    bar_color = palette[0]
    control_color = palette[1]
    pointer_color = palette[2]
    frame_colors = palette[3:]

    bar_cells, center, control, init_dir = _bar_setup(shape, bar_color, control_color)
    blocked: set[tuple[int, int]] = set(bar_cells)
    blocked.add(control)
    forbidden = _expand_cells(blocked, shape, radius=1)
    reserved: set[tuple[int, int]] = set()

    anchor = center
    direction = init_dir
    frames: list[FrameSpec] = []
    turned = False

    for i, color in enumerate(frame_colors):
        require_turn = False
        if not turned and i == n_frames - 1:
            require_turn = True
        elif not turned and random.random() < 0.45:
            require_turn = True
        frame = _choose_frame(
            shape=shape,
            anchor=anchor,
            incoming=direction,
            color=color,
            blocked=blocked,
            forbidden=forbidden,
            reserved=reserved,
            remaining_after=n_frames - i - 1,
            require_turn=require_turn,
        )
        if frame is None:
            return None
        corridor = _corridor_cells(anchor, direction, _entry_cell(anchor, direction, frame.bbox))
        reserved.update(corridor)
        blocked.update(_bbox_cells(frame.bbox))
        forbidden.update(_expand_cells(_bbox_cells(frame.bbox), shape, radius=1))
        if frame.pointer_dir != direction:
            turned = True
        frames.append(frame)
        anchor = _frame_exit_border(frame, frame.pointer_dir)
        direction = frame.pointer_dir

    final_cells = _ray_cells_to_edge(anchor, direction, shape)
    if len(final_cells) < FINAL_MIN_RUN:
        return None
    if any(cell in blocked or cell in reserved for cell in final_cells):
        return None
    reserved.update(final_cells)

    case = CaseSpec(
        shape=shape,
        bg=bg,
        bar_color=bar_color,
        control_color=control_color,
        pointer_color=pointer_color,
        bar_cells=bar_cells,
        center=center,
        control=control,
        init_dir=init_dir,
        frames=tuple(frames),
    )

    grid = _render_case(case)
    expected = _expected_from_case(case)
    analyzed_bg, analyzed_bar, analyzed_center, analyzed_dir, ordered = _analyze_grid(grid)
    if analyzed_bg != bg or analyzed_bar != bar_color or analyzed_center != center or analyzed_dir != init_dir:
        return None
    if len(ordered) != len(frames):
        return None
    for a, b in zip(ordered, frames):
        if (a.bbox, a.color, a.pointer, a.pointer_dir) != (b.bbox, b.color, b.pointer, b.pointer_dir):
            return None
    if not np.array_equal(main(grid), expected):
        return None
    if len(set(frame.pointer_dir for frame in frames)) == 1 and all(frame.pointer_dir == init_dir for frame in frames):
        return None
    return case


def _render_case(case: CaseSpec) -> np.ndarray:
    grid = np.full(case.shape, case.bg, dtype=int)
    for r, c in case.bar_cells:
        grid[r, c] = case.bar_color
    grid[case.control] = case.control_color
    for frame in case.frames:
        for r, c in _border_cells(frame.bbox):
            grid[r, c] = frame.color
        grid[frame.pointer] = case.pointer_color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return _render_case(case)
    raise RuntimeError("failed to generate a valid routing-frame relay instance")
