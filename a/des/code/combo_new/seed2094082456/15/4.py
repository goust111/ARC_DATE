from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS = ("up", "down", "left", "right")
OUTER_SIZES = (3, 5)
GRID_MIN = 14
GRID_MAX = 30
OBJECT_COUNT_CHOICES = (2, 3)
GEN_RETRIES = 400
PLACE_RETRIES = 200
OCCUPANCY_PAD = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _gap_map(r0: int, c0: int, r1: int, c1: int) -> dict[tuple[int, int], str]:
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    return {
        (r0, mid_c): "up",
        (r1, mid_c): "down",
        (mid_r, c0): "left",
        (mid_r, c1): "right",
    }


def _paste_block(out: np.ndarray, block: np.ndarray, top: int, left: int) -> None:
    h, w = out.shape
    bh, bw = block.shape
    for r in range(bh):
        rr = top + r
        if not (0 <= rr < h):
            continue
        for c in range(bw):
            cc = left + c
            if 0 <= cc < w:
                out[rr, cc] = int(block[r, c])


def _apply_frame_rule(out: np.ndarray, bbox: tuple[int, int, int, int], pattern: np.ndarray, directions: list[str]) -> None:
    r0, c0, r1, c1 = bbox
    ph, pw = pattern.shape
    copies = len(directions)

    for direction in directions:
        for step in range(1, copies + 1):
            if direction == "up":
                _paste_block(out, pattern, r0 - step * ph, c0 + 1)
            elif direction == "down":
                _paste_block(out, pattern, r1 + 1 + (step - 1) * ph, c0 + 1)
            elif direction == "left":
                _paste_block(out, pattern, r0 + 1, c0 - step * pw)
            else:
                _paste_block(out, pattern, r0 + 1, c1 + 1 + (step - 1) * pw)


def _candidate_frame(
    grid: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int
) -> dict[str, object] | None:
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3 or height % 2 == 0 or width % 2 == 0:
        return None

    border = int(grid[r0, c0])
    if border == bg:
        return None
    if not (
        int(grid[r0, c1]) == border
        and int(grid[r1, c0]) == border
        and int(grid[r1, c1]) == border
    ):
        return None

    gap_lookup = _gap_map(r0, c0, r1, c1)
    directions: list[str] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if not on_border:
                continue
            value = int(grid[r, c])
            if (r, c) in gap_lookup:
                if value == bg:
                    directions.append(gap_lookup[(r, c)])
                elif value != border:
                    return None
            elif value != border:
                return None

    if len(directions) not in (1, 2):
        return None

    interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    if interior.size == 0:
        return None
    if np.any(interior == border):
        return None
    if not np.any(interior != bg):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border": border,
        "directions": directions,
        "pattern": interior.copy(),
        "area": height * width,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    candidates: list[dict[str, object]] = []
    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            if (r1 - r0) % 2 == 1:
                continue
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    if (c1 - c0) % 2 == 1:
                        continue
                    cand = _candidate_frame(grid, bg, r0, c0, r1, c1)
                    if cand is not None:
                        candidates.append(cand)

    candidates.sort(
        key=lambda item: (
            int(item["area"]),
            len(item["directions"]),
            -int(item["bbox"][0]),
            -int(item["bbox"][1]),
        ),
        reverse=True,
    )

    used = np.zeros_like(grid, dtype=bool)
    frames: list[dict[str, object]] = []
    for cand in candidates:
        r0, c0, r1, c1 = cand["bbox"]
        if used[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        used[r0 : r1 + 1, c0 : c1 + 1] = True
        frames.append(cand)

    frames.sort(key=lambda item: tuple(item["bbox"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each framed mini-pattern outward by copying its interior along gap-indicated directions.

    Input structure:
    - The background is the dominant black color.
    - The grid contains several odd-sized one-pixel-thick colored rectangular frames.
    - Each frame has exactly one or two centered border gaps, and its interior contains a non-empty pattern whose colors differ from the frame color.

    Rule:
    - Detect every rectangular frame and read each gap position as a copy direction: top, bottom, left, or right.
    - Let the total number of gaps on that frame be N; copy the whole interior pattern, without the border, N times along every gap direction.
    - Place each copy flush against the previous one outside the frame, clipping anything that falls beyond the grid.

    Key insight:
    - Gap position and gap count encode different things: the side of each gap gives the direction, while the total number of gaps gives the replication count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        _apply_frame_rule(
            out,
            frame["bbox"],
            np.asarray(frame["pattern"], dtype=int),
            list(frame["directions"]),
        )

    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    pts = []
    if r > 0:
        pts.append((r - 1, c))
    if r + 1 < h:
        pts.append((r + 1, c))
    if c > 0:
        pts.append((r, c - 1))
    if c + 1 < w:
        pts.append((r, c + 1))
    return pts


def _random_connected_pattern(height: int, width: int, color: int, rng: random.Random) -> np.ndarray:
    area = height * width
    target = rng.randint(1, area)
    cells = [(r, c) for r in range(height) for c in range(width)]
    start = rng.choice(cells)
    chosen = {start}

    while len(chosen) < target:
        frontier = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c, height, width):
                if (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            break
        chosen.add(rng.choice(frontier))

    block = np.zeros((height, width), dtype=int)
    for r, c in chosen:
        block[r, c] = int(color)
    return block


def _render_input(grid: np.ndarray, specs: list[dict[str, object]]) -> None:
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        border = int(spec["border"])
        grid[r0, c0 : c1 + 1] = border
        grid[r1, c0 : c1 + 1] = border
        grid[r0 : r1 + 1, c0] = border
        grid[r0 : r1 + 1, c1] = border
        for gap_r, gap_c in spec["gaps"]:
            grid[gap_r, gap_c] = 0
        _paste_block(grid, np.asarray(spec["pattern"], dtype=int), r0 + 1, c0 + 1)


def _expected_from_specs(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        _apply_frame_rule(
            out,
            spec["bbox"],
            np.asarray(spec["pattern"], dtype=int),
            list(spec["directions"]),
        )
    return out


def _copy_extents(spec: dict[str, object]) -> tuple[int, int, int, int]:
    pattern = np.asarray(spec["pattern"], dtype=int)
    ph, pw = pattern.shape
    copies = len(spec["directions"])
    directions = set(spec["directions"])
    up = ph * copies if "up" in directions else 0
    down = ph * copies if "down" in directions else 0
    left = pw * copies if "left" in directions else 0
    right = pw * copies if "right" in directions else 0
    return up, down, left, right


def _reserved_bbox(spec: dict[str, object]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = spec["bbox"]
    up, down, left, right = _copy_extents(spec)
    return r0 - up, c0 - left, r1 + down, c1 + right


def _choose_position(
    occupancy: np.ndarray, spec: dict[str, object], rng: random.Random
) -> tuple[int, int] | None:
    h, w = occupancy.shape
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    up, down, left, right = _copy_extents(spec)

    r_min = up
    r_max = h - outer_h - down
    c_min = left
    c_max = w - outer_w - right
    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(r_min, r_max)
        c0 = rng.randint(c_min, c_max)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        rr0 = max(0, r0 - up - OCCUPANCY_PAD)
        rr1 = min(h - 1, r1 + down + OCCUPANCY_PAD)
        cc0 = max(0, c0 - left - OCCUPANCY_PAD)
        cc1 = min(w - 1, c1 + right + OCCUPANCY_PAD)
        if occupancy[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        occupancy[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return r0, c0

    return None


def _build_spec(border: int, pattern_color: int, rng: random.Random) -> dict[str, object]:
    outer_h = rng.choice(OUTER_SIZES)
    outer_w = rng.choice(OUTER_SIZES)
    inner_h = outer_h - 2
    inner_w = outer_w - 2
    gap_count = rng.choice((1, 2))
    directions = list(rng.sample(DIRS, gap_count))
    pattern = _random_connected_pattern(inner_h, inner_w, pattern_color, rng)

    gap_lookup = _gap_map(0, 0, outer_h - 1, outer_w - 1)
    gaps = [pos for pos, direction in gap_lookup.items() if direction in directions]
    gaps.sort()

    return {
        "border": border,
        "pattern": pattern,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "directions": directions,
        "gaps": gaps,
    }


def _place_specs(grid_h: int, grid_w: int, specs: list[dict[str, object]], rng: random.Random) -> list[dict[str, object]] | None:
    occupancy = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict[str, object]] = []
    order = sorted(
        range(len(specs)),
        key=lambda idx: (
            (_reserved_bbox({**specs[idx], "bbox": (0, 0, specs[idx]["outer_h"] - 1, specs[idx]["outer_w"] - 1)})[2] + 1)
            * (_reserved_bbox({**specs[idx], "bbox": (0, 0, specs[idx]["outer_h"] - 1, specs[idx]["outer_w"] - 1)})[3] + 1)
        ),
        reverse=True,
    )

    for idx in order:
        spec = dict(specs[idx])
        pos = _choose_position(occupancy, spec, rng)
        if pos is None:
            return None
        r0, c0 = pos
        r1 = r0 + int(spec["outer_h"]) - 1
        c1 = c0 + int(spec["outer_w"]) - 1
        spec["bbox"] = (r0, c0, r1, c1)
        spec["gaps"] = [tuple((r0 + gr, c0 + gc)) for gr, gc in spec["gaps"]]
        placed.append(spec)

    placed.sort(key=lambda item: tuple(item["bbox"]))
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_MIN, GRID_MAX)
    grid_w = rng.randint(GRID_MIN, GRID_MAX)
    object_count = rng.choice(OBJECT_COUNT_CHOICES)

    if 2 * object_count > 9:
        return None

    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_colors = colors[:object_count]
    pattern_colors = colors[object_count : 2 * object_count]

    raw_specs = [
        _build_spec(border_colors[i], pattern_colors[i], rng)
        for i in range(object_count)
    ]
    placed_specs = _place_specs(grid_h, grid_w, raw_specs, rng)
    if placed_specs is None:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    _render_input(inp, placed_specs)

    detected = _find_frames(inp, 0)
    if len(detected) != object_count:
        return None

    expected = _expected_from_specs(inp, placed_specs)
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted generation attempts")
