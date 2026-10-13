from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 18
GRID_MAX = 24
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _parse_legend_a(grid: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    _, width = grid.shape
    c = 1
    while c < width:
        if int(grid[0, c]) == BLACK:
            c += 1
            continue
        assert c + 1 < width and int(grid[0, c + 1]) != BLACK
        src = int(grid[0, c])
        mid = int(grid[0, c + 1])
        mapping[src] = mid
        c += 2
    assert mapping
    return mapping


def _parse_legend_b(grid: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    height, _ = grid.shape
    r = 1
    while r < height:
        if int(grid[r, 0]) == BLACK:
            r += 1
            continue
        mid = int(grid[r, 0])
        assert mid != GREY
        rr = r + 1
        length = 0
        while rr < height and int(grid[rr, 0]) == GREY:
            length += 1
            rr += 1
        assert 1 <= length <= 3
        mapping[mid] = length
        r = rr
    assert mapping
    return mapping


def _draw_hollow_square(out: np.ndarray, r: int, c: int, radius: int, color: int) -> None:
    height, width = out.shape
    top = r - radius
    bottom = r + radius
    left = c - radius
    right = c + radius

    c0 = max(0, left)
    c1 = min(width - 1, right)
    r0 = max(0, top)
    r1 = min(height - 1, bottom)

    if 0 <= top < height:
        out[top, c0 : c1 + 1] = color
    if 0 <= bottom < height:
        out[bottom, c0 : c1 + 1] = color
    if 0 <= left < width:
        out[r0 : r1 + 1, left] = color
    if 0 <= right < width:
        out[r0 : r1 + 1, right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain two legends: source color X -> middle color Y -> side radius L,
    then draw a Y-colored hollow square of side 2L+1 around each X point.
    """
    g = np.asarray(grid, dtype=int)
    legend_a = _parse_legend_a(g)
    legend_b = _parse_legend_b(g)

    out = g.copy()
    out[0, :] = BLACK
    out[:, 0] = BLACK

    for rr, cc in np.argwhere(g[1:, 1:] != BLACK):
        r = int(rr + 1)
        c = int(cc + 1)
        src = int(g[r, c])
        mid = int(legend_a[src])
        length = int(legend_b[mid])
        _draw_hollow_square(out, r, c, length, mid)
        out[r, c] = src

    return out


def _random_split(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _place_legend_a(grid: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    width = grid.shape[1]
    count = len(pairs)
    base = 2 * count + (count - 1)
    extra = (width - 1) - base
    assert extra >= 0
    gaps = _random_split(extra, count + 1)
    c = 1 + gaps[0]
    for i, (src, mid) in enumerate(pairs):
        grid[0, c] = int(src)
        grid[0, c + 1] = int(mid)
        c += 2
        if i + 1 < count:
            c += 1 + gaps[i + 1]


def _place_legend_b(grid: np.ndarray, entries: list[tuple[int, int]]) -> None:
    height = grid.shape[0]
    count = len(entries)
    base = sum(1 + length for _mid, length in entries) + (count - 1)
    extra = (height - 1) - base
    assert extra >= 0
    gaps = _random_split(extra, count + 1)
    r = 1 + gaps[0]
    for i, (mid, length) in enumerate(entries):
        grid[r, 0] = int(mid)
        grid[r + 1 : r + 1 + length, 0] = GREY
        r += 1 + length
        if i + 1 < count:
            r += 1 + gaps[i + 1]


def _box_rect(height: int, width: int, r: int, c: int, radius: int) -> tuple[int, int, int, int]:
    return (
        max(1, r - radius),
        max(1, c - radius),
        min(height - 1, r + radius),
        min(width - 1, c + radius),
    )


def _too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_center(height: int, width: int, radius: int, mode: str) -> tuple[int, int] | None:
    top_safe = radius + 1
    left_safe = radius + 1
    bottom_safe = height - 1 - radius
    right_safe = width - 1 - radius

    if mode == "normal":
        if top_safe > bottom_safe or left_safe > right_safe:
            return None
        return random.randint(top_safe, bottom_safe), random.randint(left_safe, right_safe)
    if mode == "bottom":
        lo = max(top_safe, height - radius)
        if lo > height - 1 or left_safe > right_safe:
            return None
        return random.randint(lo, height - 1), random.randint(left_safe, right_safe)
    if mode == "right":
        lo = max(left_safe, width - radius)
        if top_safe > bottom_safe or lo > width - 1:
            return None
        return random.randint(top_safe, bottom_safe), random.randint(lo, width - 1)
    if mode == "corner":
        row_lo = max(top_safe, height - radius)
        col_lo = max(left_safe, width - radius)
        if row_lo > height - 1 or col_lo > width - 1:
            return None
        return random.randint(row_lo, height - 1), random.randint(col_lo, width - 1)
    raise ValueError(f"unknown mode: {mode}")


def _place_points(
    height: int,
    width: int,
    items: list[dict[str, int | str]],
) -> list[dict[str, int]]:
    placed: list[dict[str, int]] = []
    for item in items:
        radius = int(item["L"])
        mode = str(item["mode"])
        success = False
        for _ in range(PLACEMENT_RETRIES):
            center = _sample_center(height, width, radius, mode)
            if center is None:
                break
            r, c = center
            rect = _box_rect(height, width, r, c, radius)
            if any(_too_close(rect, other["rect"]) for other in placed):
                continue
            placed.append(
                {
                    "src": int(item["src"]),
                    "mid": int(item["mid"]),
                    "L": radius,
                    "r": int(r),
                    "c": int(c),
                    "rect": rect,
                }
            )
            success = True
            break
        if not success:
            return []
    return placed


def _validate_generated(
    grid: np.ndarray,
    entries: list[dict[str, int]],
    placed: list[dict[str, int]],
) -> bool:
    if grid.shape[0] < GRID_MIN or grid.shape[1] < GRID_MIN:
        return False

    legend_a = _parse_legend_a(grid)
    legend_b = _parse_legend_b(grid)

    if len(entries) != len(legend_a) or len(entries) != len(legend_b):
        return False
    for entry in entries:
        if legend_a.get(int(entry["src"])) != int(entry["mid"]):
            return False
        if legend_b.get(int(entry["mid"])) != int(entry["L"]):
            return False

    field = np.argwhere(grid[1:, 1:] != BLACK)
    if len(field) != len(placed):
        return False

    seen: list[tuple[int, int, int, int]] = []
    for item in placed:
        r = int(item["r"])
        c = int(item["c"])
        radius = int(item["L"])
        if int(grid[r, c]) != int(item["src"]):
            return False
        if r - radius < 1 or c - radius < 1:
            return False
        rect = _box_rect(grid.shape[0], grid.shape[1], r, c, radius)
        if rect[0] < 1 or rect[1] < 1:
            return False
        if any(_too_close(rect, other) for other in seen):
            return False
        seen.append(rect)

    out = main(grid)
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BLACK, dtype=int)

    count = random.choice((2, 3))
    lengths = random.sample((1, 2, 3), count)
    colors = random.sample(COLOR_POOL, 2 * count)
    src_colors = colors[:count]
    mid_colors = colors[count:]

    entries = [
        {"src": int(src), "mid": int(mid), "L": int(length)}
        for src, mid, length in zip(src_colors, mid_colors, lengths)
    ]

    pairs = [(int(entry["src"]), int(entry["mid"])) for entry in entries]
    legend_b_entries = [(int(entry["mid"]), int(entry["L"])) for entry in entries]
    random.shuffle(pairs)
    random.shuffle(legend_b_entries)

    _place_legend_a(grid, pairs)
    _place_legend_b(grid, legend_b_entries)

    planned: list[dict[str, int | str]] = []
    for entry in entries:
        planned.append({**entry, "mode": "normal"})

    extra_points = random.randint(0, 3)
    for _ in range(extra_points):
        chosen = random.choice(entries)
        planned.append({**chosen, "mode": "normal"})

    if random.random() < 0.5:
        random.choice(planned)["mode"] = random.choice(("bottom", "right", "corner"))

    random.shuffle(planned)
    planned.sort(key=lambda item: (str(item["mode"]) == "normal", -int(item["L"])))

    placed = _place_points(height, width, planned)
    if not placed:
        return None

    for item in placed:
        grid[int(item["r"]), int(item["c"])] = int(item["src"])

    if not _validate_generated(grid, entries, placed):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
