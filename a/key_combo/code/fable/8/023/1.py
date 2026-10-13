from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


BG = Color.BLACK
RAY = Color.YELLOW
FRAME_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = [UP, DOWN, LEFT, RIGHT]
DIR_FROM_SIDE = {"U": UP, "D": DOWN, "L": LEFT, "R": RIGHT}
SIDE_FROM_DIR = {UP: "U", DOWN: "D", LEFT: "L", RIGHT: "R"}
OPPOSITE_SIDE = {"U": "D", "D": "U", "L": "R", "R": "L"}

GEN_SIZE_RANGE = (18, 30)
GEN_FRAME_HEIGHTS = (4, 5, 6)
GEN_FRAME_WIDTHS = (4, 5, 6)
GEN_GAP_RANGE = (2, 4)
GEN_MARGIN_RANGE = (1, 3)
GEN_ACTIVE_WEIGHTS = [(1, 1), (2, 3), (3, 5), (4, 4)]
GEN_MAX_ATTEMPTS = 400


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    h: int
    w: int
    color: int
    opening_side: str
    opening_offset: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r1, self.c1

    @property
    def opening_dir(self) -> tuple[int, int]:
        return DIR_FROM_SIDE[self.opening_side]

    @property
    def opening(self) -> tuple[int, int]:
        if self.opening_side == "U":
            return self.r0, self.c0 + self.opening_offset
        if self.opening_side == "D":
            return self.r1, self.c0 + self.opening_offset
        if self.opening_side == "L":
            return self.r0 + self.opening_offset, self.c0
        return self.r0 + self.opening_offset, self.c1

    @property
    def interior_bbox(self) -> tuple[int, int, int, int]:
        return self.r0 + 1, self.c0 + 1, self.r1 - 1, self.c1 - 1


def _paint_frame(grid: np.ndarray, frame: Frame) -> None:
    r0, c0, r1, c1 = frame.bbox
    grid[r0, c0 : c1 + 1] = frame.color
    grid[r1, c0 : c1 + 1] = frame.color
    grid[r0 : r1 + 1, c0] = frame.color
    grid[r0 : r1 + 1, c1] = frame.color
    orow, ocol = frame.opening
    grid[orow, ocol] = BG


def _fill_interior(grid: np.ndarray, frame: Frame, color: int) -> None:
    r0, c0, r1, c1 = frame.interior_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _relay_exit(frame: Frame) -> tuple[int, int]:
    orow, ocol = frame.opening
    if frame.opening_side == "L":
        return orow, frame.c1
    if frame.opening_side == "R":
        return orow, frame.c0
    if frame.opening_side == "U":
        return frame.r1, ocol
    return frame.r0, ocol


def _interior_contains(frame: Frame, coord: tuple[int, int]) -> bool:
    r, c = coord
    ir0, ic0, ir1, ic1 = frame.interior_bbox
    return ir0 <= r <= ir1 and ic0 <= c <= ic1


def _reserve_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _weighted_active_count() -> int:
    values = [item[0] for item in GEN_ACTIVE_WEIGHTS]
    weights = [item[1] for item in GEN_ACTIVE_WEIGHTS]
    return random.choices(values, weights=weights, k=1)[0]


def _sample_axis_frame(
    fixed_line: int,
    grid_span: int,
    horizontal_travel: bool,
    opening_side: str,
    color: int,
) -> Frame | None:
    for _ in range(60):
        h = random.choice(GEN_FRAME_HEIGHTS)
        w = random.choice(GEN_FRAME_WIDTHS)
        if horizontal_travel:
            if opening_side not in {"L", "R"}:
                continue
            offset = random.randint(1, h - 2)
            r0 = fixed_line - offset
            if 1 <= r0 and r0 + h - 1 <= grid_span - 2:
                return Frame(r0, 0, h, w, color, opening_side, offset)
        else:
            if opening_side not in {"U", "D"}:
                continue
            offset = random.randint(1, w - 2)
            c0 = fixed_line - offset
            if 1 <= c0 and c0 + w - 1 <= grid_span - 2:
                return Frame(0, c0, h, w, color, opening_side, offset)
    return None


def _shift_frame(frame: Frame, dr: int, dc: int) -> Frame:
    return Frame(frame.r0 + dr, frame.c0 + dc, frame.h, frame.w, frame.color, frame.opening_side, frame.opening_offset)


def _trace_from_frame(
    input_grid: np.ndarray,
    out: np.ndarray,
    frames: list[Frame],
    opening_map: dict[tuple[int, int], int],
    idx: int,
    direction: tuple[int, int],
    *,
    use_entry_port: bool,
) -> int | None:
    dr, dc = direction
    port = frames[idx].opening if use_entry_port else _relay_exit(frames[idx])
    prow, pcol = port
    out[prow, pcol] = RAY
    rr, cc = prow + dr, pcol + dc
    while in_bounds(input_grid.shape, rr, cc):
        target = opening_map.get((rr, cc))
        if target is not None and target != idx and frames[target].opening_dir == (-dr, -dc):
            out[rr, cc] = RAY
            return target
        if int(input_grid[rr, cc]) != BG:
            return None
        out[rr, cc] = RAY
        rr += dr
        cc += dc
    return None


def _parse_frames(grid: np.ndarray, bg: int) -> list[Frame]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    frames: list[Frame] = []
    for comp in comps:
        color = int(comp["color"])
        if color in {bg, RAY}:
            continue
        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            raise AssertionError("frame bbox must have non-empty interior")
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        if any(coord not in border for coord in coords):
            raise AssertionError("frame component must lie on its bbox border")
        missing = sorted(border - coords)
        if len(missing) != 1:
            raise AssertionError("expected exactly one opening cell")
        orow, ocol = missing[0]
        side_hits = [
            ("U", orow == r0),
            ("D", orow == r1),
            ("L", ocol == c0),
            ("R", ocol == c1),
        ]
        side_names = [name for name, ok in side_hits if ok]
        if len(side_names) != 1:
            raise AssertionError("opening must be on exactly one non-corner side")
        side = side_names[0]
        if side in {"U", "D"}:
            offset = ocol - c0
            if offset in {0, w - 1}:
                raise AssertionError("opening may not be on a corner")
        else:
            offset = orow - r0
            if offset in {0, h - 1}:
                raise AssertionError("opening may not be on a corner")
        expected_area = len(border) - 1
        if int(comp["area"]) != expected_area:
            raise AssertionError("component must be a 1-cell-thick hollow rectangle with one opening")
        frames.append(Frame(r0, c0, h, w, color, side, offset))
    if not frames:
        raise AssertionError("expected at least one frame")
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light the seed frame and relay the signal through every aligned port chain.

    Input structure:
    - The background is black and contains several separated monochrome hollow rectangular frames.
    - Each frame is one cell thick and has exactly one one-cell opening on one side.
    - Exactly one frame contains a single yellow seed pixel in its interior; every other frame interior is empty.

    Rule:
    - The seed frame activates first, fills its interior yellow, and emits a yellow ray through its visible opening in that opening's outward direction.
    - A ray paints every background cell it crosses; if it reaches another frame exactly at that frame's opening from the outside, that frame activates, fills yellow, and the signal continues straight through the box to the opposite wall and onward in the same direction.
    - If the ray would hit a wall before reaching a usable opening, it stops there; unactivated frames stay unchanged.

    Key insight:
    - This is a directed reachability task on openings: geometric alignment alone is insufficient because the first wall on the ray blocks anything farther away, and activated boxes behave like straight-through relays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    seed_coords = np.argwhere(g == RAY)
    if seed_coords.shape != (1, 2):
        raise AssertionError("expected exactly one yellow seed pixel")
    seed = tuple(map(int, seed_coords[0]))

    frames = _parse_frames(g, bg)
    source_indices = [i for i, frame in enumerate(frames) if _interior_contains(frame, seed)]
    if len(source_indices) != 1:
        raise AssertionError("seed must lie inside exactly one frame interior")
    source = source_indices[0]

    out = g.copy()
    opening_map = {frame.opening: i for i, frame in enumerate(frames)}
    if len(opening_map) != len(frames):
        raise AssertionError("frame openings must be distinct")

    active = {source}
    emitted: set[int] = set()
    queue: deque[tuple[int, tuple[int, int], bool]] = deque([(source, frames[source].opening_dir, True)])

    while queue:
        idx, direction, use_entry_port = queue.popleft()
        if idx in emitted:
            continue
        emitted.add(idx)
        _fill_interior(out, frames[idx], RAY)
        target = _trace_from_frame(
            g,
            out,
            frames,
            opening_map,
            idx,
            direction,
            use_entry_port=use_entry_port,
        )
        if target is not None and target not in active:
            active.add(target)
            queue.append((target, direction, False))

    return out


def _random_frame_colors(count: int) -> list[int]:
    if count <= len(FRAME_COLORS):
        return random.sample(FRAME_COLORS, count)
    return [random.choice(FRAME_COLORS) for _ in range(count)]


def _build_linear_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    horizontal = random.random() < 0.5
    direction = random.choice([LEFT, RIGHT]) if horizontal else random.choice([UP, DOWN])
    active_count = _weighted_active_count()
    if active_count >= 4:
        stop_mode = "edge"
        add_shadow_target = False
        extra_frames = random.randint(0, 1)
    elif active_count == 3:
        stop_mode = random.choice(["edge", "wall"])
        add_shadow_target = stop_mode == "wall" and random.random() < 0.5
        extra_frames = random.randint(0, 1)
    else:
        stop_mode = random.choice(["edge", "wall"])
        add_shadow_target = stop_mode == "wall" and random.random() < 0.6
        extra_frames = random.randint(0, 2)

    total_colors = active_count + 1 + extra_frames + int(add_shadow_target)
    colors = _random_frame_colors(total_colors)
    color_i = 0

    canonical_dir = RIGHT if horizontal else DOWN
    source_side = SIDE_FROM_DIR[canonical_dir]
    receiver_side = SIDE_FROM_DIR[(-canonical_dir[0], -canonical_dir[1])]

    if horizontal:
        height = random.randint(18, 26)
        cross_line = random.randint(4, height - 5)
        axis_span = height
        width = 0
    else:
        width = random.randint(18, 26)
        cross_line = random.randint(4, width - 5)
        axis_span = width
        height = 0

    chain: list[Frame] = []
    for i in range(active_count):
        side = source_side if i == 0 else receiver_side
        frame = _sample_axis_frame(cross_line, axis_span, horizontal, side, colors[color_i])
        color_i += 1
        if frame is None:
            return None
        chain.append(frame)

    stopper_side_choices = [side for side in "UDLR" if side != receiver_side]
    stopper = _sample_axis_frame(
        cross_line,
        axis_span,
        horizontal,
        random.choice(stopper_side_choices),
        colors[color_i],
    )
    color_i += 1
    if stopper is None:
        return None

    shadow = None
    if add_shadow_target:
        shadow = _sample_axis_frame(cross_line, axis_span, horizontal, receiver_side, colors[color_i])
        color_i += 1
        if shadow is None:
            return None

    if horizontal:
        start_margin = random.randint(*GEN_MARGIN_RANGE)
        gaps = [random.randint(*GEN_GAP_RANGE) for _ in range(max(0, active_count - 1))]
        end_gap = random.randint(*GEN_GAP_RANGE)
        shadow_gap = random.randint(*GEN_GAP_RANGE) if shadow is not None else 0
        tail_margin = random.randint(*GEN_MARGIN_RANGE)
        total = start_margin + sum(frame.w for frame in chain) + sum(gaps) + tail_margin
        if stop_mode == "wall":
            total += end_gap + stopper.w
            if shadow is not None:
                total += shadow_gap + shadow.w
        if total > 30:
            return None
        width = random.randint(total, 30)
        frames: list[Frame] = []
        cursor = start_margin
        for i, frame in enumerate(chain):
            placed = _shift_frame(frame, 0, cursor)
            frames.append(placed)
            cursor = placed.c1 + 1
            if i < len(gaps):
                cursor += gaps[i]
        if stop_mode == "wall":
            placed_stopper = _shift_frame(stopper, 0, cursor + end_gap)
            frames.append(placed_stopper)
            cursor = placed_stopper.c1 + 1
            if shadow is not None:
                placed_shadow = _shift_frame(shadow, 0, cursor + shadow_gap)
                frames.append(placed_shadow)
    else:
        start_margin = random.randint(*GEN_MARGIN_RANGE)
        gaps = [random.randint(*GEN_GAP_RANGE) for _ in range(max(0, active_count - 1))]
        end_gap = random.randint(*GEN_GAP_RANGE)
        shadow_gap = random.randint(*GEN_GAP_RANGE) if shadow is not None else 0
        tail_margin = random.randint(*GEN_MARGIN_RANGE)
        total = start_margin + sum(frame.h for frame in chain) + sum(gaps) + tail_margin
        if stop_mode == "wall":
            total += end_gap + stopper.h
            if shadow is not None:
                total += shadow_gap + shadow.h
        if total > 30:
            return None
        height = random.randint(total, 30)
        frames = []
        cursor = start_margin
        for i, frame in enumerate(chain):
            placed = _shift_frame(frame, cursor, 0)
            frames.append(placed)
            cursor = placed.r1 + 1
            if i < len(gaps):
                cursor += gaps[i]
        if stop_mode == "wall":
            placed_stopper = _shift_frame(stopper, cursor + end_gap, 0)
            frames.append(placed_stopper)
            cursor = placed_stopper.r1 + 1
            if shadow is not None:
                placed_shadow = _shift_frame(shadow, cursor + shadow_gap, 0)
                frames.append(placed_shadow)

    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    for frame in frames:
        expanded = np.zeros_like(reserved)
        _reserve_bbox(expanded, frame.bbox, pad=1)
        if np.any(expanded & reserved):
            return None
        _reserve_bbox(reserved, frame.bbox, pad=1)
        _paint_frame(grid, frame)

    active_frames = frames[:active_count]
    source = active_frames[0]
    ir0, ic0, ir1, ic1 = source.interior_bbox
    grid[random.randint(ir0, ir1), random.randint(ic0, ic1)] = RAY

    corridor_mask = np.zeros((height, width), dtype=bool)
    if horizontal:
        corridor_mask[source.opening[0], source.opening[1] : width] = True
    else:
        corridor_mask[source.opening[0] : height, source.opening[1]] = True

    distractors: list[Frame] = []
    for _ in range(extra_frames):
        placed = None
        for _ in range(80):
            side = random.choice(list(DIR_FROM_SIDE))
            h = random.choice(GEN_FRAME_HEIGHTS)
            w = random.choice(GEN_FRAME_WIDTHS)
            offset = random.randint(1, (h - 2) if side in {"L", "R"} else (w - 2))
            r0 = random.randint(1, height - h - 1)
            c0 = random.randint(1, width - w - 1)
            frame = Frame(r0, c0, h, w, colors[color_i], side, offset)
            expanded = np.zeros_like(reserved)
            _reserve_bbox(expanded, frame.bbox, pad=1)
            if np.any(expanded & reserved):
                continue
            rr0, cc0, rr1, cc1 = frame.bbox
            if np.any(corridor_mask[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue
            placed = frame
            break
        if placed is None:
            return None
        distractors.append(placed)
        _reserve_bbox(reserved, placed.bbox, pad=1)
        _paint_frame(grid, placed)
        color_i += 1

    all_frames = active_frames + frames[active_count:] + distractors
    expected = grid.copy()
    opening_map = {frame.opening: i for i, frame in enumerate(all_frames)}
    for idx, frame in enumerate(active_frames):
        _fill_interior(expected, frame, RAY)
        next_idx = _trace_from_frame(
            grid,
            expected,
            all_frames,
            opening_map,
            idx,
            canonical_dir,
            use_entry_port=(idx == 0),
        )
        if idx < active_count - 1 and next_idx != idx + 1:
            return None
        if idx == active_count - 1 and stop_mode == "edge" and next_idx is not None:
            return None

    if direction == LEFT:
        grid = np.fliplr(grid).copy()
        expected = np.fliplr(expected).copy()
    elif direction == UP:
        grid = np.flipud(grid).copy()
        expected = np.flipud(expected).copy()

    meta = {
        "direction": direction,
        "active_count": active_count,
        "stop_mode": stop_mode,
        "shadow": shadow is not None,
        "frame_count": len(all_frames),
    }
    return grid, expected, meta


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_linear_case()
        if built is None:
            continue
        grid, expected, meta = built
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid, expected, meta
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
