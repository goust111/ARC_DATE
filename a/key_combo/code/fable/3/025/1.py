from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 800


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    h: int
    w: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        out.append(np.array(coords, dtype=int))
    return out


def _extract_rectangles(grid: np.ndarray) -> list[Rect]:
    rects: list[Rect] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            assert len(coords) == h * w, "each foreground object must fill its bbox"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "object must be a solid rectangle"
            rects.append(Rect(color, r0, c0, h, w))
    assert rects, "expected at least one foreground rectangle"
    return rects


def _overlap_cols(a: Rect, b: Rect) -> bool:
    return max(a.c0, b.c0) <= min(a.c1, b.c1)


def _assert_unambiguous_bottom_ties(rects: list[Rect]) -> None:
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert a.r1 != b.r1 or not _overlap_cols(a, b), "same-level overlapping columns are ambiguous"


def _ordered(rects: list[Rect]) -> list[Rect]:
    _assert_unambiguous_bottom_ties(rects)
    return sorted(rects, key=lambda r: (-r.r1, r.c0, r.r0, r.w, r.h))


def _settle(rects: list[Rect], shape: tuple[int, int], trace: bool = False) -> tuple[np.ndarray, list[dict]]:
    h_grid, w_grid = shape
    out = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    records: list[dict] = []

    for rect in _ordered(rects):
        assert 0 <= rect.r0 and 0 <= rect.c0
        assert rect.r1 < h_grid and rect.c1 < w_grid
        assert not np.any(occupied[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1]), (
            "starting rectangle may not overlap an already settled obstacle"
        )

        r = rect.r0
        steps = 0
        while True:
            nr = r + rect.h
            cause = ""
            gap = -1
            if nr + rect.h > h_grid:
                cause = "ground"
                gap = h_grid - (r + rect.h)
            else:
                hit = occupied[nr : nr + rect.h, rect.c0 : rect.c1 + 1]
                if np.any(hit):
                    cause = "object"
                    gap = int(np.argwhere(hit)[:, 0].min())

            if cause:
                break
            r = nr
            steps += 1

        assert not np.any(occupied[r : r + rect.h, rect.c0 : rect.c1 + 1]), "settled rectangles must not overlap"
        out[r : r + rect.h, rect.c0 : rect.c1 + 1] = rect.color
        occupied[r : r + rect.h, rect.c0 : rect.c1 + 1] = True

        if trace:
            records.append(
                {
                    "color": rect.color,
                    "height": rect.h,
                    "width": rect.w,
                    "start_r": rect.r0,
                    "final_r": r,
                    "c0": rect.c0,
                    "c1": rect.c1,
                    "steps": steps,
                    "cause": cause,
                    "gap": gap,
                }
            )

    return out, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each solid rectangle downward in jumps equal to its own height.

    Input structure:
    - A black background contains several separated, solid, axis-aligned rectangles.
    - Rectangles have distinct visible colors, and their column ranges may overlap vertically.
    - Rectangles with the same bottom row do not also overlap in columns, so bottom-up order is unambiguous.

    Rule:
    - Clear the original rectangles, then settle them from lowest bottom edge to highest.
    - A rectangle repeatedly moves down by exactly its own height.
    - It stops at the last position before the next whole-height jump would hit the floor or an already settled rectangle.

    Key insight:
    - This is quantized gravity: a rectangle usually leaves a sub-height gap above its support, and lower rectangles must settle before they become obstacles for upper ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    rects = _extract_rectangles(g)
    out, _records = _settle(rects, g.shape, trace=False)
    return out


# ----------------------------- generator -----------------------------


def _rects_overlap(a: Rect, b: Rect) -> bool:
    rows = max(a.r0, b.r0) <= min(a.r1, b.r1)
    cols = max(a.c0, b.c0) <= min(a.c1, b.c1)
    return rows and cols


def _paint(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for rect in rects:
        if rect.r0 < 0 or rect.c0 < 0 or rect.r1 >= shape[0] or rect.c1 >= shape[1]:
            return None
        area = occupied[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1]
        if np.any(area):
            return None
        grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = rect.color
        area[:, :] = True
    return grid


def _make_bands(width: int, count: int) -> list[tuple[int, int]] | None:
    band_widths = [random.randint(3, 8) for _ in range(count)]
    gaps = [random.randint(2, 4) for _ in range(count - 1)]
    span = sum(band_widths) + sum(gaps)
    if span > width:
        return None
    c = random.randint(0, width - span)
    bands = []
    for i, band_w in enumerate(band_widths):
        bands.append((c, c + band_w - 1))
        if i < len(gaps):
            c += band_w + gaps[i]
    return bands


def _choose_column_interval(band: tuple[int, int]) -> tuple[int, int]:
    left, right = band
    band_w = right - left + 1
    core = random.randint(left, right)
    width = random.randint(2, min(7, band_w))
    c_min = max(left, core - width + 1)
    c_max = min(core, right - width + 1)
    c0 = random.randint(c_min, c_max)
    return c0, c0 + width - 1


def _build_stack(
    shape: tuple[int, int],
    band: tuple[int, int],
    colors: list[int],
    require_multistep: bool,
) -> tuple[list[Rect], list[Rect]] | None:
    h_grid, _w_grid = shape
    n = len(colors)

    for _ in range(120):
        heights = [random.randint(2, 5) for _ in range(n)]
        gaps = [random.randint(0, heights[0] - 1)]
        gaps.extend(random.randint(0, heights[i] - 1) for i in range(1, n))

        final_rects: list[Rect] = []
        bottom_r = h_grid - gaps[0] - heights[0]
        if bottom_r < heights[0]:
            continue

        for i, (color, rh) in enumerate(zip(colors, heights)):
            c0, c1 = _choose_column_interval(band)
            if i == 0:
                r0 = bottom_r
            else:
                support_top = final_rects[-1].r0
                r0 = support_top - gaps[i] - rh
            if r0 < rh:
                break
            final_rects.append(Rect(int(color), int(r0), int(c0), int(rh), int(c1 - c0 + 1)))
        if len(final_rects) != n:
            continue

        start_rects: list[Rect] = []
        for i, rect in enumerate(final_rects):
            drops = 1
            if i == n - 1 and require_multistep and rect.r0 >= 2 * rect.h:
                drops = 2
            elif i == n - 1 and not require_multistep and rect.r0 >= 2 * rect.h and random.random() < 0.35:
                drops = 2
            start_rects.append(Rect(rect.color, rect.r0 - drops * rect.h, rect.c0, rect.h, rect.w))

        if require_multistep and not any(fr.r0 - sr.r0 >= 2 * sr.h for sr, fr in zip(start_rects, final_rects)):
            continue
        if any(_rects_overlap(a, b) for i, a in enumerate(start_rects) for b in start_rects[i + 1 :]):
            continue
        if any(_rects_overlap(a, b) for i, a in enumerate(final_rects) for b in final_rects[i + 1 :]):
            continue
        try:
            _assert_unambiguous_bottom_ties(start_rects)
        except AssertionError:
            continue
        return start_rects, final_rects
    return None


def _continuous_settle(rects: list[Rect], shape: tuple[int, int]) -> np.ndarray | None:
    h_grid, w_grid = shape
    out = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for rect in _ordered(rects):
        r = rect.r0
        while True:
            nr = r + 1
            if nr + rect.h > h_grid:
                break
            if np.any(occupied[nr : nr + rect.h, rect.c0 : rect.c1 + 1]):
                break
            r = nr
        if r < 0 or rect.c0 < 0 or r + rect.h > h_grid or rect.c1 >= w_grid:
            return None
        if np.any(occupied[r : r + rect.h, rect.c0 : rect.c1 + 1]):
            return None
        out[r : r + rect.h, rect.c0 : rect.c1 + 1] = rect.color
        occupied[r : r + rect.h, rect.c0 : rect.c1 + 1] = True
    return out


def _ground_only_quantum_settle(rects: list[Rect], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for rect in _ordered(rects):
        r = rect.r0
        while r + 2 * rect.h <= shape[0]:
            r += rect.h
        out[r : r + rect.h, rect.c0 : rect.c1 + 1] = rect.color
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, records: list[dict]) -> bool:
    if len(records) < 3:
        return False
    if len({rec["height"] for rec in records}) < 2:
        return False
    if not any(rec["steps"] >= 2 for rec in records):
        return False
    if not any(rec["cause"] == "object" for rec in records):
        return False
    if not any(rec["cause"] == "object" and rec["gap"] > 0 for rec in records):
        return False
    if not any(rec["gap"] == 0 for rec in records):
        return False
    if not any(rec["gap"] > 0 for rec in records):
        return False

    rects = _extract_rectangles(inp)
    continuous = _continuous_settle(rects, inp.shape)
    if continuous is None or np.array_equal(continuous, expected):
        return False
    if np.array_equal(_ground_only_quantum_settle(rects, inp.shape), expected):
        return False
    return True


def _sample_lengths(stack_count: int) -> list[int] | None:
    if stack_count == 1:
        return [random.randint(3, 5)]
    lengths = [random.randint(2, 3) for _ in range(stack_count)]
    if sum(lengths) > len(PALETTE):
        return None
    return lengths


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(14, 30)
    max_stacks = min(3, max(1, width // 6))
    stack_count = random.randint(1, max_stacks)

    lengths = _sample_lengths(stack_count)
    if lengths is None:
        return None
    total_rects = sum(lengths)
    colors = random.sample(PALETTE, total_rects)

    bands = _make_bands(width, stack_count)
    if bands is None:
        return None

    starts: list[Rect] = []
    finals: list[Rect] = []
    pos = 0
    multistep_stack = random.randrange(stack_count)
    for stack_i, (band, n) in enumerate(zip(bands, lengths)):
        stack_colors = colors[pos : pos + n]
        pos += n
        built = _build_stack((height, width), band, stack_colors, require_multistep=(stack_i == multistep_stack))
        if built is None:
            return None
        s_rects, f_rects = built
        starts.extend(s_rects)
        finals.extend(f_rects)

    if any(_rects_overlap(a, b) for i, a in enumerate(starts) for b in starts[i + 1 :]):
        return None
    if any(_rects_overlap(a, b) for i, a in enumerate(finals) for b in finals[i + 1 :]):
        return None

    inp = _paint((height, width), starts)
    expected_constructed = _paint((height, width), finals)
    if inp is None or expected_constructed is None:
        return None

    try:
        expected, records = _settle(starts, inp.shape, trace=True)
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(expected, expected_constructed):
        return None
    if not np.array_equal(solved, expected):
        return None
    if not _semantic_checks(inp, expected, records):
        return None
    if int(np.count_nonzero(inp == BG)) <= int(np.count_nonzero(inp != BG)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid quantum-gravity input")
