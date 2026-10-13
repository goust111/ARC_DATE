from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 15
GRID_MAX = 30
FRAME_COUNT_RANGE = (1, 4)
EXIT_TOTAL_RANGE = (2, 6)
FRAME_HALF_RANGE = (1, 5)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _side_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }


def _collect_exit_sides(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> list[str]:
    exit_sides: list[str] = []
    for side, coords in _side_positions(r0, c0, r1, c1).items():
        values = [int(grid[r, c]) for r, c in coords]
        if any(v not in (BLACK, GREY) for v in values):
            return []
        exit_sides.extend(side for v in values if v == BLACK)
    return exit_sides


def _find_frame(grid: np.ndarray, sr: int, sc: int) -> tuple[int, int, int, int, list[str]]:
    h, w = grid.shape
    max_half_h = min(sr, h - 1 - sr)
    max_half_w = min(sc, w - 1 - sc)
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int, list[str]]]] = []

    for half_h in range(1, max_half_h + 1):
        r0 = sr - half_h
        r1 = sr + half_h
        for half_w in range(1, max_half_w + 1):
            c0 = sc - half_w
            c1 = sc + half_w

            if any(int(grid[r, c]) != GREY for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
                continue

            interior = np.array(grid[r0 + 1 : r1, c0 + 1 : c1], copy=True)
            center_r = sr - (r0 + 1)
            center_c = sc - (c0 + 1)
            interior[center_r, center_c] = BLACK
            if interior.size and np.any(interior != BLACK):
                continue

            exit_sides = _collect_exit_sides(grid, r0, c0, r1, c1)
            if not exit_sides:
                continue

            border_gray = 4
            for coords in _side_positions(r0, c0, r1, c1).values():
                border_gray += sum(int(grid[r, c]) == GREY for r, c in coords)

            score = (border_gray, 2 * (half_h + half_w), -len(exit_sides))
            candidates.append((score, (r0, c0, r1, c1, exit_sides)))

    assert candidates, "expected each source pixel to sit inside a pierced grey frame"
    candidates.sort(reverse=True)
    return candidates[0][1]


def _paint_ray(out: np.ndarray, sr: int, sc: int, side: str, length: int, color: int) -> None:
    dr, dc = SIDE_TO_STEP[side]
    h, w = out.shape
    for step in range(1, length + 1):
        rr = sr + dr * step
        cc = sc + dc * step
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot colored rays from each framed source pixel, with ray length equal to the frame's total exit count.

    Input structure:
    - The background is black.
    - Several one-pixel-thick grey rectangular frames appear in the grid, each with one or more missing border pixels acting as exits.
    - Each frame contains exactly one non-black, non-grey source pixel at its geometric center, and the rest of its interior is black.

    Rule:
    - For each frame, count all black exit cells on its border.
    - Each exit contributes a direction according to the side it lies on: top, bottom, left, or right.
    - From the center source pixel, draw a straight ray of that source color for exactly N cells in every exit direction, where N is the total number of exits on that frame.
    - Rays overwrite any grey border cells they cross, while all unrelated cells stay unchanged.

    Key insight:
    - The exits play a double role: they determine which directions fire, and their total count simultaneously sets the length of every ray from that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sources = [
        (int(r), int(c))
        for r, c in np.argwhere((g != BLACK) & (g != GREY))
    ]
    assert sources, "expected at least one colored source pixel"

    out = g.copy()
    for sr, sc in sources:
        color = int(g[sr, sc])
        _r0, _c0, _r1, _c1, exit_sides = _find_frame(g, sr, sc)
        ray_length = len(exit_sides)
        for side in exit_sides:
            _paint_ray(out, sr, sc, side, ray_length, color)

    return out


def _sample_exit_counts() -> dict[str, int] | None:
    total = random.randint(*EXIT_TOTAL_RANGE)
    active_count = random.randint(1, min(len(SIDES), total))
    active_sides = random.sample(SIDES, active_count)
    counts = {side: 0 for side in SIDES}
    for side in active_sides:
        counts[side] = 1
    for _ in range(total - active_count):
        counts[random.choice(active_sides)] += 1
    return counts


def _sample_half_sizes(counts: dict[str, int]) -> tuple[int, int] | None:
    total = sum(counts.values())
    min_half_h = max(1, (max(counts["left"], counts["right"]) + 2) // 2)
    min_half_w = max(1, (max(counts["top"], counts["bottom"]) + 2) // 2)

    max_half_h = FRAME_HALF_RANGE[1]
    max_half_w = FRAME_HALF_RANGE[1]
    if counts["top"] or counts["bottom"]:
        max_half_h = min(max_half_h, total - 1)
    if counts["left"] or counts["right"]:
        max_half_w = min(max_half_w, total - 1)

    if min_half_h > max_half_h or min_half_w > max_half_w:
        return None

    half_h = random.randint(min_half_h, max_half_h)
    half_w = random.randint(min_half_w, max_half_w)
    return half_h, half_w


def _sample_exit_offsets(half_h: int, half_w: int, counts: dict[str, int]) -> dict[str, list[int]]:
    positions = {
        "top": list(range(-half_w + 1, half_w)),
        "bottom": list(range(-half_w + 1, half_w)),
        "left": list(range(-half_h + 1, half_h)),
        "right": list(range(-half_h + 1, half_h)),
    }
    return {
        side: sorted(random.sample(positions[side], counts[side]))
        for side in SIDES
    }


def _unique_dirs(counts: dict[str, int]) -> set[str]:
    return {side for side, count in counts.items() if count > 0}


def _reach(half_size: int, total_exits: int, used: bool) -> int:
    return max(half_size, total_exits if used else 0)


def _reserve_box(spec: dict[str, object], sr: int, sc: int, h: int, w: int) -> tuple[int, int, int, int]:
    counts = spec["counts"]
    half_h = int(spec["half_h"])
    half_w = int(spec["half_w"])
    total = int(spec["total"])
    up = _reach(half_h, total, counts["top"] > 0)
    down = _reach(half_h, total, counts["bottom"] > 0)
    left = _reach(half_w, total, counts["left"] > 0)
    right = _reach(half_w, total, counts["right"] > 0)
    return (
        max(0, sr - up - 1),
        max(0, sc - left - 1),
        min(h - 1, sr + down + 1),
        min(w - 1, sc + right + 1),
    )


def _place_specs(h: int, w: int, specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []

    specs = sorted(specs, key=lambda spec: (int(spec["total"]), int(spec["half_h"]) + int(spec["half_w"])), reverse=True)
    for spec in specs:
        counts = spec["counts"]
        half_h = int(spec["half_h"])
        half_w = int(spec["half_w"])
        total = int(spec["total"])
        up = _reach(half_h, total, counts["top"] > 0)
        down = _reach(half_h, total, counts["bottom"] > 0)
        left = _reach(half_w, total, counts["left"] > 0)
        right = _reach(half_w, total, counts["right"] > 0)

        if up + down + 1 > h or left + right + 1 > w:
            return None

        center_rows = list(range(up, h - down))
        center_cols = list(range(left, w - right))
        random.shuffle(center_rows)
        random.shuffle(center_cols)

        found = None
        for sr in center_rows[:]:
            for sc in center_cols[:]:
                rr0, cc0, rr1, cc1 = _reserve_box(spec, sr, sc, h, w)
                if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                found = (sr, sc, rr0, cc0, rr1, cc1)
                break
            if found is not None:
                break

        if found is None:
            return None

        sr, sc, rr0, cc0, rr1, cc1 = found
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

        spec = dict(spec)
        spec["center"] = (sr, sc)
        placed.append(spec)

    return placed


def _render_input(h: int, w: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for spec in specs:
        sr, sc = spec["center"]
        half_h = int(spec["half_h"])
        half_w = int(spec["half_w"])
        color = int(spec["color"])
        exits = spec["exit_offsets"]

        r0, r1 = sr - half_h, sr + half_h
        c0, c1 = sc - half_w, sc + half_w
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY

        for offset in exits["top"]:
            grid[r0, sc + offset] = BLACK
        for offset in exits["bottom"]:
            grid[r1, sc + offset] = BLACK
        for offset in exits["left"]:
            grid[sr + offset, c0] = BLACK
        for offset in exits["right"]:
            grid[sr + offset, c1] = BLACK

        grid[sr, sc] = color
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for spec in specs:
        sr, sc = spec["center"]
        color = int(spec["color"])
        total = int(spec["total"])
        for side, offsets in spec["exit_offsets"].items():
            for _offset in offsets:
                _paint_ray(out, sr, sc, side, total, color)
    return out


def _build_spec(color: int) -> dict[str, object] | None:
    counts = _sample_exit_counts()
    dims = _sample_half_sizes(counts)
    if dims is None:
        return None
    half_h, half_w = dims
    exit_offsets = _sample_exit_offsets(half_h, half_w, counts)
    return {
        "color": color,
        "counts": counts,
        "exit_offsets": exit_offsets,
        "half_h": half_h,
        "half_w": half_w,
        "total": sum(counts.values()),
    }


def _attempt_generate() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    h = random.randint(max(GRID_MIN, 12 + 2 * frame_count), GRID_MAX)
    w = random.randint(max(GRID_MIN, 12 + 2 * frame_count), GRID_MAX)
    colors = random.sample(SOURCE_COLORS, frame_count)

    specs: list[dict[str, object]] = []
    for color in colors:
        spec = _build_spec(color)
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_specs(h, w, specs)
    if placed is None:
        return None

    inp = _render_input(h, w, placed)
    expected = _expected_output(inp, placed)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
