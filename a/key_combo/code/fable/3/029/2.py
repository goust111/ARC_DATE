from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
FRAME = 1
WATER = 8


@dataclass(frozen=True)
class Frame:
    bbox: tuple[int, int, int, int]
    gaps: tuple[tuple[int, int], ...]

    @property
    def r0(self) -> int:
        return self.bbox[0]

    @property
    def c0(self) -> int:
        return self.bbox[1]

    @property
    def r1(self) -> int:
        return self.bbox[2]

    @property
    def c1(self) -> int:
        return self.bbox[3]

    @property
    def top_gap_cols(self) -> set[int]:
        return {c for r, c in self.gaps if r == self.r0}


def _border_coords(bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = bbox
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def _gap_side(frame: Frame, gap: tuple[int, int]) -> str:
    r, c = gap
    if r == frame.r0:
        return "top"
    if r == frame.r1:
        return "bottom"
    if c == frame.c0:
        return "left"
    if c == frame.c1:
        return "right"
    raise AssertionError("gap is not on the frame border")


def _detect_frames(g: np.ndarray) -> list[Frame]:
    colors = set(map(int, g.ravel())) - {BG, WATER}
    assert colors == {FRAME}, "expected only blue frame pixels besides black and teal"

    h, w = g.shape
    candidates: list[Frame] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r0 in range(1, h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(0, w - 2):
                for c1 in range(c0 + 2, w):
                    if (r0, c0, r1, c1) in seen:
                        continue
                    if not (
                        g[r0, c0] == FRAME
                        and g[r0, c1] == FRAME
                        and g[r1, c0] == FRAME
                        and g[r1, c1] == FRAME
                    ):
                        continue
                    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != BG):
                        continue

                    gaps: list[tuple[int, int]] = []
                    ok = True
                    for r, c in _border_coords((r0, c0, r1, c1)):
                        is_corner = r in (r0, r1) and c in (c0, c1)
                        val = int(g[r, c])
                        if val == FRAME:
                            continue
                        if val == BG and not is_corner:
                            gaps.append((r, c))
                            continue
                        ok = False
                        break
                    if not ok or len(gaps) not in (1, 2):
                        continue
                    candidates.append(Frame((r0, c0, r1, c1), tuple(sorted(gaps))))
                    seen.add((r0, c0, r1, c1))

    assert candidates, "expected at least one rectangular frame"

    cover = np.zeros(g.shape, dtype=int)
    for frame in candidates:
        for r, c in _border_coords(frame.bbox):
            if g[r, c] == FRAME:
                cover[r, c] += 1
    assert np.all(cover[g == FRAME] == 1), "frame pixels must belong to exactly one frame"

    return sorted(candidates, key=lambda f: (f.r0, f.c0, f.r1, f.c1))


def _next_frame_on_column(frames: list[Frame], start_r: int, col: int) -> Frame | None:
    choices = [
        frame
        for frame in frames
        if frame.r0 >= start_r and frame.c0 <= col <= frame.c1
    ]
    if not choices:
        return None
    return min(choices, key=lambda frame: frame.r0)


def _paint_vertical(out: np.ndarray, start_r: int, end_r: int, col: int) -> None:
    h, w = out.shape
    if not (0 <= col < w):
        return
    for r in range(max(0, start_r), min(h - 1, end_r) + 1):
        if out[r, col] != FRAME:
            out[r, col] = WATER


def _simulate_water(g: np.ndarray, frames: list[Frame]) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    sources = [int(c) for c in np.flatnonzero(g[0] == WATER)]
    assert sources, "expected at least one top-row water source"

    def inject(frame: Frame) -> list[tuple[int, int]]:
        other_gaps = [gap for gap in frame.gaps if gap[0] != frame.r0]
        if not other_gaps:
            out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1] = WATER
            return []

        assert len(other_gaps) == 1, "a two-gap frame must have one overflow gap"
        gap = other_gaps[0]
        side = _gap_side(frame, gap)
        out[gap] = WATER
        if side in ("left", "right"):
            gr, gc = gap
            out[gr + 1 : frame.r1, frame.c0 + 1 : frame.c1] = WATER
            spill_c = frame.c0 - 1 if side == "left" else frame.c1 + 1
            return [(gr, spill_c)]
        if side == "bottom":
            gr, gc = gap
            return [(gr + 1, gc)]
        raise AssertionError("top overflow gap is not a valid second gap")

    pending = [(0, c) for c in sources]
    guard = 0
    while pending:
        start_r, col = pending.pop(0)
        guard += 1
        assert guard <= h * max(1, len(frames) + len(sources)), "water chain did not terminate"
        if start_r >= h or not (0 <= col < w):
            continue

        frame = _next_frame_on_column(frames, start_r, col)
        if frame is None:
            _paint_vertical(out, start_r, h - 1, col)
            continue

        _paint_vertical(out, start_r, frame.r0 - 1, col)
        if col in frame.top_gap_cols:
            out[frame.r0, col] = WATER
            pending.extend(inject(frame))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate teal water falling through top gaps and overflowing through chained frame gaps.

    Input structure:
    - The grid has a black background, teal source pixels on the top row, and blue one-pixel-thick rectangular frames below.
    - Each frame has one or two missing non-corner border cells; interiors and gap cells are black in the input.
    - Frames are separated, and only a top-edge gap can accept a falling water column.

    Rule:
    - From each top-row teal source, paint a vertical teal trail downward until the grid bottom or the next frame top edge.
    - A column hitting a non-gap top-edge frame pixel stops above it; a column aligned with a top gap enters the frame.
    - A one-gap entered frame has its whole interior filled; a side overflow gap fills only interior rows strictly below that gap and continues downward from the outside cell; a bottom overflow gap fills no interior and continues below the gap.

    Key insight:
    - The visible gap is a port in a downward chain: a frame's fill level is determined by whether water reaches its top gap and where its second gap releases the overflow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    frames = _detect_frames(g)
    return _simulate_water(g, frames)


# ----------------------------- generator -----------------------------


def _draw_frame(grid: np.ndarray, frame: Frame) -> None:
    for r, c in _border_coords(frame.bbox):
        grid[r, c] = FRAME
    for r, c in frame.gaps:
        grid[r, c] = BG


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _can_place(bbox: tuple[int, int, int, int], frames: list[Frame], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if r0 < 2 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    return all(_separated(bbox, frame.bbox) for frame in frames)


def _top_gap_frame(r0: int, c0: int, height: int, width: int, top_col: int) -> Frame:
    return Frame((r0, c0, r0 + height - 1, c0 + width - 1), ((r0, top_col),))


def _side_frame(r0: int, c0: int, height: int, width: int, top_col: int, side: str, gap_row: int) -> Frame:
    c1 = c0 + width - 1
    side_col = c0 if side == "left" else c1
    return Frame((r0, c0, r0 + height - 1, c1), ((r0, top_col), (gap_row, side_col)))


def _bottom_frame(r0: int, c0: int, height: int, width: int, top_col: int, bottom_col: int) -> Frame:
    return Frame((r0, c0, r0 + height - 1, c0 + width - 1), ((r0, top_col), (r0 + height - 1, bottom_col)))


def _place_lower_frame(
    frames: list[Frame],
    shape: tuple[int, int],
    min_r: int,
    spill_col: int,
    *,
    width_range: tuple[int, int] = (5, 7),
    height_range: tuple[int, int] = (4, 6),
) -> Frame | None:
    h, w = shape
    for _ in range(100):
        fh = random.randint(*height_range)
        fw = random.randint(*width_range)
        max_r0 = h - fh - 2
        if min_r > max_r0:
            return None
        r0 = random.randint(min_r, max_r0)
        offsets = [off for off in range(1, fw - 1) if 1 <= spill_col - off and spill_col - off + fw - 1 < w - 1]
        random.shuffle(offsets)
        for off in offsets:
            c0 = spill_col - off
            frame = _top_gap_frame(r0, c0, fh, fw, spill_col)
            if _can_place(frame.bbox, frames, shape):
                return frame
    return None


def _frame_interior_has_water(out: np.ndarray, frame: Frame) -> bool:
    return bool(np.any(out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1] == WATER))


def _frame_is_empty(out: np.ndarray, frame: Frame) -> bool:
    if np.any(out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1] == WATER):
        return False
    return all(out[r, c] == BG for r, c in frame.gaps)


def _try_add_distractor(frames: list[Frame], sources: set[int], shape: tuple[int, int]) -> Frame | None:
    h, w = shape
    path_cols = set(sources)
    for frame in frames:
        for r, c in frame.gaps:
            side = _gap_side(frame, (r, c))
            if side == "left":
                path_cols.add(frame.c0 - 1)
            elif side == "right":
                path_cols.add(frame.c1 + 1)
            elif side == "bottom":
                path_cols.add(c)

    for _ in range(200):
        fh = random.randint(4, 6)
        fw = random.randint(5, 8)
        r0 = random.randint(3, h - fh - 2)
        c0 = random.randint(1, w - fw - 2)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if not _can_place(bbox, frames, shape):
            continue
        top_choices = [c for c in range(c0 + 1, c0 + fw - 1) if c not in path_cols]
        if random.random() < 0.5 and top_choices:
            gap = (r0, random.choice(top_choices))
            return Frame(bbox, (gap,))
        if random.random() < 0.5:
            gr = random.randint(r0 + 1, r0 + fh - 2)
            side_col = random.choice([c0, c0 + fw - 1])
            return Frame(bbox, ((gr, side_col),))
        bc = random.choice([c for c in range(c0 + 1, c0 + fw - 1) if c not in path_cols] or list(range(c0 + 1, c0 + fw - 1)))
        return Frame(bbox, ((r0 + fh - 1, bc),))
    return None


def _build_attempt() -> np.ndarray | None:
    h = random.randint(25, 30)
    w = random.randint(25, 30)
    frames: list[Frame] = []
    sources: set[int] = set()

    # Left chain: top injection -> side overflow -> lower one-gap full fill.
    fh = random.randint(5, 7)
    fw = random.randint(5, 7)
    r0 = random.randint(4, 6)
    max_left_c0 = max(2, w // 2 - fw - 2)
    c0 = random.randint(2, max_left_c0)
    top_col = random.randint(c0 + 1, c0 + fw - 2)
    side_options = []
    if c0 - 1 >= 1:
        side_options.append("left")
    if c0 + fw < w - 1:
        side_options.append("right")
    side = random.choice(side_options)
    gap_row = random.randint(r0 + 1, r0 + fh - 3)
    side_frame = _side_frame(r0, c0, fh, fw, top_col, side, gap_row)
    if not _can_place(side_frame.bbox, frames, (h, w)):
        return None
    frames.append(side_frame)
    sources.add(top_col)
    spill_col = side_frame.c0 - 1 if side == "left" else side_frame.c1 + 1
    lower = _place_lower_frame(frames, (h, w), side_frame.r1 + random.randint(4, 6), spill_col)
    if lower is None:
        return None
    frames.append(lower)

    # Right chain: top injection -> bottom overflow -> lower one-gap full fill.
    fh = random.randint(4, 6)
    fw = random.randint(5, 8)
    r0 = random.randint(3, 6)
    min_right_c0 = max(w // 2 + 1, 1)
    max_right_c0 = w - fw - 2
    if min_right_c0 > max_right_c0:
        return None
    c0 = random.randint(min_right_c0, max_right_c0)
    top_col = random.randint(c0 + 1, c0 + fw - 2)
    bottom_choices = [c for c in range(c0 + 1, c0 + fw - 1) if c != top_col]
    bottom_col = random.choice(bottom_choices)
    bottom_frame = _bottom_frame(r0, c0, fh, fw, top_col, bottom_col)
    if not _can_place(bottom_frame.bbox, frames, (h, w)):
        return None
    frames.append(bottom_frame)
    sources.add(top_col)
    stop_choices = [c for c in range(c0 + 1, c0 + fw - 1) if c != top_col]
    sources.add(random.choice(stop_choices))
    lower = _place_lower_frame(frames, (h, w), bottom_frame.r1 + random.randint(4, 6), bottom_col)
    if lower is None:
        return None
    frames.append(lower)

    distractor = _try_add_distractor(frames, sources, (h, w))
    if distractor is None:
        return None
    frames.append(distractor)

    grid = np.full((h, w), BG, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)
    for c in sources:
        grid[0, c] = WATER

    try:
        expected = _simulate_water(grid, frames)
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if not _frame_interior_has_water(actual, frames[1]):
        return None
    if not _frame_interior_has_water(actual, frames[3]):
        return None
    if not _frame_is_empty(actual, distractor):
        return None

    side_gap = [gap for gap in side_frame.gaps if gap[0] != side_frame.r0][0]
    if actual[side_gap] != WATER:
        return None
    if not np.any(actual[side_gap[0] + 1 : side_frame.r1, side_frame.c0 + 1 : side_frame.c1] == WATER):
        return None
    if np.any(actual[side_frame.r0 + 1 : side_gap[0] + 1, side_frame.c0 + 1 : side_frame.c1] == WATER):
        return None

    bottom_gap = [gap for gap in bottom_frame.gaps if gap[0] == bottom_frame.r1][0]
    if actual[bottom_gap] != WATER:
        return None
    if np.any(actual[bottom_frame.r0 + 1 : bottom_frame.r1, bottom_frame.c0 + 1 : bottom_frame.c1] == WATER):
        return None

    if np.array_equal(grid, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid water-chain input")
