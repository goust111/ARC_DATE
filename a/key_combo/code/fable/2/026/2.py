from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int
    gap_r: int
    gap_c: int
    gap_side: str


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_frames(grid: np.ndarray, frame_color: int, bg: int) -> list[Frame]:
    frames: list[Frame] = []
    for comp in _components(grid == int(frame_color)):
        r0, c0, r1, c1 = _bbox(comp)
        coords = set(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        assert coords <= border
        gaps = sorted(border - coords)
        assert len(gaps) == 1
        gap_r, gap_c = gaps[0]
        if gap_r == r0:
            assert c0 < gap_c < c1
            gap_side = "top"
        elif gap_c == c0:
            assert r0 < gap_r < r1
            gap_side = "left"
        elif gap_c == c1:
            assert r0 < gap_r < r1
            gap_side = "right"
        else:
            raise AssertionError("frame gap must lie on the top or a side border")
        if r1 - r0 > 1 and c1 - c0 > 1:
            assert bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == int(bg)))
        frames.append(Frame(r0, c0, r1, c1, gap_r, gap_c, gap_side))
    frames.sort(key=lambda f: (f.r0, f.c0, f.r1, f.c1, f.gap_side, f.gap_r, f.gap_c))
    assert frames
    return frames


def _paint_water(out: np.ndarray, static: np.ndarray, frame_color: int, water: int, r: int, c: int) -> None:
    if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(static[r, c]) != int(frame_color):
        out[r, c] = int(water)


def _paint_row(out: np.ndarray, static: np.ndarray, frame_color: int, water: int, row: int, c0: int, c1: int) -> None:
    if not (0 <= row < out.shape[0]):
        return
    for c in range(max(0, c0), min(out.shape[1] - 1, c1) + 1):
        _paint_water(out, static, frame_color, water, row, c)


def _fill_full_frame(out: np.ndarray, frame: Frame, water: int) -> None:
    out[frame.gap_r, frame.gap_c] = int(water)
    if frame.r1 - frame.r0 > 1 and frame.c1 - frame.c0 > 1:
        out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1] = int(water)


def _fill_side_frame(out: np.ndarray, frame: Frame, water: int) -> None:
    out[frame.gap_r, frame.gap_c] = int(water)
    if frame.r1 - frame.r0 > 1 and frame.c1 - frame.c0 > 1:
        out[frame.gap_r : frame.r1, frame.c0 + 1 : frame.c1] = int(water)


def _simulate(grid: np.ndarray, want_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert 0 < height <= 30 and 0 < width <= 30

    bg = _get_bg(g)
    top_colors = sorted({int(v) for v in g[0] if int(v) != int(bg)})
    assert len(top_colors) == 1
    water = int(top_colors[0])
    non_bg = sorted({int(v) for v in np.unique(g)} - {int(bg)})
    frame_colors = [color for color in non_bg if int(color) != int(water)]
    assert len(frame_colors) == 1
    frame_color = int(frame_colors[0])

    frames = _parse_frames(g, frame_color, bg)

    top_gap_map: dict[tuple[int, int], int] = {}
    top_shelf_map: dict[tuple[int, int], int] = {}
    side_gap_map: dict[tuple[int, int], list[int]] = {}
    for idx, frame in enumerate(frames):
        for c in range(frame.c0, frame.c1 + 1):
            key = (frame.r0, c)
            if frame.gap_side == "top" and c == frame.gap_c:
                assert key not in top_gap_map and key not in top_shelf_map
                top_gap_map[key] = idx
            else:
                assert key not in top_gap_map and key not in top_shelf_map
                top_shelf_map[key] = idx
        if frame.gap_side == "left" and frame.c0 > 0:
            side_gap_map.setdefault((frame.gap_r, frame.c0 - 1), []).append(idx)
        elif frame.gap_side == "right" and frame.c1 + 1 < width:
            side_gap_map.setdefault((frame.gap_r, frame.c1 + 1), []).append(idx)

    out = g.copy()
    seen: set[tuple[int, int]] = set()
    q = deque((0, c) for c in range(width) if int(g[0, c]) == int(water))

    trace = {
        "bg": bg,
        "water": water,
        "frame_color": frame_color,
        "frames": frames,
        "top_gap": set(),
        "top_shelf": set(),
        "side_fill": set(),
        "activated": set(),
    }

    while q:
        rr, cc = q.popleft()
        rr = int(rr)
        cc = int(cc)
        while 0 <= rr < height and 0 <= cc < width:
            if (rr, cc) in seen:
                break
            seen.add((rr, cc))
            _paint_water(out, g, frame_color, water, rr, cc)

            for frame_idx in side_gap_map.get((rr, cc), ()):
                _fill_side_frame(out, frames[int(frame_idx)], water)
                trace["side_fill"].add(int(frame_idx))
                trace["activated"].add(int(frame_idx))

            nr = rr + 1
            if nr >= height:
                break

            if (nr, cc) in top_gap_map:
                frame_idx = int(top_gap_map[(nr, cc)])
                frame = frames[frame_idx]
                _fill_full_frame(out, frame, water)
                _paint_row(out, g, frame_color, water, rr, frame.c0 - 1, frame.c1 + 1)
                for next_c in (frame.c0 - 1, frame.c1 + 1):
                    if 0 <= next_c < width:
                        q.append((rr, int(next_c)))
                trace["top_gap"].add(frame_idx)
                trace["activated"].add(frame_idx)
                break

            if (nr, cc) in top_shelf_map:
                frame_idx = int(top_shelf_map[(nr, cc)])
                frame = frames[frame_idx]
                _paint_row(out, g, frame_color, water, rr, frame.c0 - 1, frame.c1 + 1)
                for next_c in (frame.c0 - 1, frame.c1 + 1):
                    if 0 <= next_c < width:
                        q.append((rr, int(next_c)))
                trace["top_shelf"].add(frame_idx)
                trace["activated"].add(frame_idx)
                break

            rr = nr

    return (out, trace) if want_trace else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate a valve-controlled waterfall through one-gap rectangular frames.

    Input structure:
    - The background is black, and one distinct water color appears only as source pixels on the top row.
    - Every other non-background pixel belongs to a same-colored hollow rectangle border with exactly one missing non-corner border cell.
    - Each frame gap lies either on the top edge or on one of the side edges.

    Rule:
    - Water falls straight down from every source, painting only the traversed background cells.
    - Hitting a closed top edge turns that frame into a shelf: water spreads across the whole top span, overflows one cell past both ends, then both overflow cells continue falling.
    - Falling through a top-edge gap fills that frame's entire interior, then the water overflows back out of the gap and spreads across the top exactly like a shelf.
    - Falling outside a side wall and passing the side gap pours into the frame laterally, filling only from the gap row downward while the main vertical stream keeps falling.

    Key insight:
    - The single missing border cell is a valve: top gaps create full-container fill plus overflow, side gaps create level-limited side fill, and closed tops act only as branching shelves.
    """
    return _simulate(grid, want_trace=False)


def _separated(a: Frame, b: Frame, margin: int = 1) -> bool:
    return (
        a.r1 + margin < b.r0
        or b.r1 + margin < a.r0
        or a.c1 + margin < b.c0
        or b.c1 + margin < a.c0
    )


def _frame_fits(frame: Frame, height: int, width: int) -> bool:
    return (
        0 <= frame.r0 < frame.r1 < height
        and 1 <= frame.c0 < frame.c1 < width - 1
        and frame.gap_side in {"top", "left", "right"}
    )


def _frame_clear(candidate: Frame, frames: list[Frame], height: int, width: int) -> bool:
    if not _frame_fits(candidate, height, width):
        return False
    return all(_separated(candidate, other, margin=1) for other in frames)


def _draw_frame(grid: np.ndarray, frame: Frame, frame_color: int) -> None:
    grid[frame.r0, frame.c0 : frame.c1 + 1] = int(frame_color)
    grid[frame.r1, frame.c0 : frame.c1 + 1] = int(frame_color)
    grid[frame.r0 : frame.r1 + 1, frame.c0] = int(frame_color)
    grid[frame.r0 : frame.r1 + 1, frame.c1] = int(frame_color)
    grid[frame.gap_r, frame.gap_c] = 0


def _render_scene(height: int, width: int, water: int, frame_color: int, source_cols: list[int], frames: list[Frame]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for c in source_cols:
        grid[0, int(c)] = int(water)
    for frame in frames:
        _draw_frame(grid, frame, frame_color)
    return grid


def _sample_top_gap_frame(height: int, width: int, stream_col: int, row_low: int, row_high: int, frames: list[Frame]) -> Frame | None:
    if row_low > row_high:
        return None
    for _ in range(80):
        box_h = random.randint(4, 6)
        box_w = random.randint(5, min(8, width - 2))
        offset = random.randint(1, box_w - 2)
        c0 = int(stream_col) - offset
        r0 = random.randint(row_low, row_high)
        frame = Frame(r0, c0, r0 + box_h - 1, c0 + box_w - 1, r0, int(stream_col), "top")
        if _frame_clear(frame, frames, height, width):
            return frame
    return None


def _sample_side_gap_frame(
    height: int,
    width: int,
    stream_col: int,
    row_low: int,
    row_high: int,
    frames: list[Frame],
) -> Frame | None:
    if row_low > row_high:
        return None
    for _ in range(120):
        box_h = random.randint(4, 6)
        box_w = random.randint(5, min(8, width - 2))
        top_offset = random.randint(1, box_w - 2)
        c0 = int(stream_col) - top_offset
        c1 = c0 + box_w - 1
        r0 = random.randint(row_low, row_high)
        gap_side = random.choice(("left", "right"))
        gap_c = c0 if gap_side == "left" else c1
        gap_r = random.randint(r0 + 1, r0 + box_h - 2)
        frame = Frame(r0, c0, r0 + box_h - 1, c1, gap_r, gap_c, gap_side)
        if _frame_clear(frame, frames, height, width):
            return frame
    return None


def _valid_semantics(inp: np.ndarray) -> bool:
    try:
        out, trace = _simulate(inp, want_trace=True)
    except AssertionError:
        return False

    frames: list[Frame] = trace["frames"]
    water = int(trace["water"])
    bg = int(trace["bg"])
    top_gap = set(trace["top_gap"])
    top_shelf = set(trace["top_shelf"])
    side_fill = set(trace["side_fill"])

    if len(frames) < 3:
        return False
    if len(trace["activated"]) != len(frames):
        return False
    if not top_gap or not top_shelf or not side_fill:
        return False
    if not any(idx in top_shelf and idx in side_fill for idx, frame in enumerate(frames) if frame.gap_side != "top"):
        return False

    full_fill_ok = False
    for idx in top_gap:
        frame = frames[int(idx)]
        interior = out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1]
        if interior.size and bool(np.all(interior == water)):
            full_fill_ok = True
            break
    if not full_fill_ok:
        return False

    partial_fill_ok = False
    for idx in side_fill:
        frame = frames[int(idx)]
        if frame.gap_side == "top":
            continue
        interior = out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1]
        if interior.size == 0:
            continue
        split = frame.gap_r - (frame.r0 + 1)
        above = interior[:split, :]
        below = interior[split:, :]
        if below.size and bool(np.all(below == water)) and (above.size == 0 or bool(np.all(above == bg))):
            partial_fill_ok = True
            break
    if not partial_fill_ok:
        return False

    spread_ok = False
    for idx in top_gap | top_shelf:
        frame = frames[int(idx)]
        row = frame.r0 - 1
        if row < 0:
            continue
        left = max(0, frame.c0 - 1)
        right = min(inp.shape[1] - 1, frame.c1 + 1)
        band = out[row, left : right + 1]
        if int(np.count_nonzero(band == water)) >= frame.c1 - frame.c0 + 2:
            spread_ok = True
            break
    if not spread_ok:
        return False

    added = int(np.count_nonzero(out == water) - np.count_nonzero(inp == water))
    return added >= 18


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(20, 29)
    width = random.randint(18, 29)
    water, frame_color = random.sample(range(1, 10), 2)

    source_main = random.randint(4, width - 5)
    frames: list[Frame] = []

    f1 = _sample_top_gap_frame(height, width, source_main, 2, min(5, height - 15), frames)
    if f1 is None:
        return None
    frames.append(f1)

    branch1 = f1.c0 - 1 if random.random() < 0.5 else f1.c1 + 1
    row2_low = f1.r1 + 3
    row2_high = min(f1.r1 + 7, height - 10)
    f2 = _sample_side_gap_frame(height, width, branch1, row2_low, row2_high, frames)
    if f2 is None:
        return None
    frames.append(f2)

    if random.random() < 0.65:
        branch2 = f2.c0 - 1 if f2.gap_side == "left" else f2.c1 + 1
    else:
        branch2 = f2.c1 + 1 if f2.gap_side == "left" else f2.c0 - 1

    row3_low = f2.r1 + 3
    row3_high = height - 6
    if row3_low > row3_high:
        return None
    if random.random() < 0.55:
        f3 = _sample_top_gap_frame(height, width, branch2, row3_low, row3_high, frames)
    else:
        f3 = _sample_side_gap_frame(height, width, branch2, row3_low, row3_high, frames)
    if f3 is None:
        return None
    frames.append(f3)

    source_cols = {int(source_main)}
    extra_source_count = random.choice((0, 1, 1, 2))
    candidates = [
        c
        for c in range(width)
        if c not in source_cols
        and all(abs(c - other) >= 2 for other in source_cols)
        and all(not (frame.c0 - 1 <= c <= frame.c1 + 1 and frame.r0 <= 4) for frame in frames)
    ]
    random.shuffle(candidates)
    for c in candidates[:extra_source_count]:
        source_cols.add(int(c))

    inp = _render_scene(height, width, water, frame_color, sorted(source_cols), frames)
    if not _valid_semantics(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid valve-waterfall input")
