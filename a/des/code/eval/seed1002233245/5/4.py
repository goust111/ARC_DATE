from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
SIDES = ("up", "down", "left", "right")
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("up", "down")

GEN_RETRIES = 400
PLACE_RETRIES = 80
GRID_MIN = 18
GRID_MAX = 30
RECT_COUNT_OPTIONS = (2, 3, 4)
SMALL_INNER = (3, 5)
PRIMARY_INNER = (5, 7)
CLEARANCE = 1


@dataclass(frozen=True)
class Rect:
    r0: int
    c0: int
    r1: int
    c1: int
    border_color: int
    seed_color: int
    seed_r: int
    seed_c: int
    gap_side: str
    ray_length: int

    @property
    def gap(self) -> tuple[int, int]:
        if self.gap_side == "up":
            return self.r0, (self.c0 + self.c1) // 2
        if self.gap_side == "down":
            return self.r1, (self.c0 + self.c1) // 2
        if self.gap_side == "left":
            return (self.r0 + self.r1) // 2, self.c0
        return (self.r0 + self.r1) // 2, self.c1

    @property
    def direction(self) -> tuple[int, int]:
        return DIRS[self.gap_side]

    @property
    def center_row(self) -> int:
        return (self.r0 + self.r1) // 2

    @property
    def center_col(self) -> int:
        return (self.c0 + self.c1) // 2


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _border_points_full(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _border_points(rect: Rect) -> list[tuple[int, int]]:
    gap = rect.gap
    return [p for p in _border_points_full(rect.r0, rect.c0, rect.r1, rect.c1) if p != gap]


def _gap_side_from_missing(
    r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]
) -> str | None:
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    candidates: list[str] = []
    if width % 2 == 1:
        mid_c = (c0 + c1) // 2
        if gap == (r0, mid_c):
            candidates.append("up")
        if gap == (r1, mid_c):
            candidates.append("down")
    if height % 2 == 1:
        mid_r = (r0 + r1) // 2
        if gap == (mid_r, c0):
            candidates.append("left")
        if gap == (mid_r, c1):
            candidates.append("right")
    return candidates[0] if len(candidates) == 1 else None


def _detect_rectangles(grid: np.ndarray) -> list[Rect]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    rects: list[Rect] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        if coords.shape[0] < 7:
            continue

        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        actual = {(int(r), int(c)) for r, c in coords}
        expected = _border_points_full(r0, c0, r1, c1)
        if actual - expected:
            continue
        missing = expected - actual
        if len(missing) != 1:
            continue

        gap = next(iter(missing))
        gap_side = _gap_side_from_missing(r0, c0, r1, c1, gap)
        if gap_side is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        seed_coords = np.argwhere(inner != bg)
        if seed_coords.shape[0] != 1:
            continue

        sr, sc = map(int, seed_coords[0])
        seed_r = r0 + 1 + sr
        seed_c = c0 + 1 + sc
        seed_color = int(g[seed_r, seed_c])
        if seed_color in (bg, color):
            continue

        rects.append(
            Rect(
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                border_color=color,
                seed_color=seed_color,
                seed_r=seed_r,
                seed_c=seed_c,
                gap_side=gap_side,
                ray_length=int(np.sum(inner == bg)),
            )
        )

    rects.sort(key=lambda rect: (rect.r0, rect.c0, rect.r1, rect.c1))
    return rects


def _simulate_rays(grid: np.ndarray, rects: list[Rect]) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = _background_color(g)
    out = g.copy()

    border_lookup: dict[tuple[int, int], Rect] = {}
    for rect in rects:
        for pos in _border_points(rect):
            if pos in border_lookup:
                raise AssertionError("overlapping borders are ambiguous")
            border_lookup[pos] = rect

    seed_positions = {(rect.seed_r, rect.seed_c) for rect in rects}
    claims: dict[tuple[int, int], int] = {}
    ray_crossings: list[int] = []
    collision = False
    seed_hit = False

    for idx, rect in enumerate(rects):
        r, c = rect.gap
        dr, dc = rect.direction
        current_color = rect.seed_color
        remaining = rect.ray_length
        local_crossings = 0

        while remaining > 0 and 0 <= r < h and 0 <= c < w:
            pos = (r, c)
            hit_rect = border_lookup.get(pos)
            if hit_rect is not None:
                current_color = hit_rect.seed_color
                local_crossings += 1
            else:
                if int(g[r, c]) == bg:
                    if pos in claims and claims[pos] != idx:
                        collision = True
                    claims[pos] = idx
                    if int(out[r, c]) == bg:
                        out[r, c] = current_color
                    remaining -= 1
                elif pos in seed_positions and pos != (rect.seed_r, rect.seed_c):
                    seed_hit = True
            r += dr
            c += dc

        ray_crossings.append(local_crossings)

    stats = {
        "crossed_any": any(count > 0 for count in ray_crossings),
        "collision": collision,
        "seed_hit": seed_hit,
        "ray_crossings": ray_crossings,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect gapped rectangular frames, then project each seed color outward from
    the frame gap. When a ray crosses another frame border, it keeps going and
    inherits the crossed frame's seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _detect_rectangles(g)
    assert rects, "expected at least one gapped rectangle"
    out, _stats = _simulate_rays(g, rects)
    return out


def _make_rect(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    gap_side: str,
    border_color: int,
    seed_color: int,
    seed_pos: tuple[int, int],
) -> Rect:
    return Rect(
        r0=r0,
        c0=c0,
        r1=r0 + inner_h + 1,
        c1=c0 + inner_w + 1,
        border_color=border_color,
        seed_color=seed_color,
        seed_r=int(seed_pos[0]),
        seed_c=int(seed_pos[1]),
        gap_side=gap_side,
        ray_length=inner_h * inner_w - 1,
    )


def _seed_choices(rect: Rect, avoid_axis: str | None = None) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for r in range(rect.r0 + 1, rect.r1):
        for c in range(rect.c0 + 1, rect.c1):
            if avoid_axis == "row" and r == rect.center_row:
                continue
            if avoid_axis == "col" and c == rect.center_col:
                continue
            choices.append((r, c))
    return choices


def _candidate_rect(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    gap_side: str,
    border_color: int,
    seed_color: int,
    avoid_axis: str | None = None,
) -> Rect | None:
    probe = _make_rect(
        r0,
        c0,
        inner_h,
        inner_w,
        gap_side,
        border_color,
        seed_color,
        (r0 + 1, c0 + 1),
    )
    choices = _seed_choices(probe, avoid_axis=avoid_axis)
    if not choices:
        return None
    return _make_rect(
        r0,
        c0,
        inner_h,
        inner_w,
        gap_side,
        border_color,
        seed_color,
        random.choice(choices),
    )


def _rect_clear(candidate: Rect, placed: list[Rect], h: int, w: int, pad: int = CLEARANCE) -> bool:
    if candidate.r0 < 1 or candidate.c0 < 1 or candidate.r1 > h - 2 or candidate.c1 > w - 2:
        return False

    for other in placed:
        separated = (
            candidate.r1 + pad < other.r0 - pad
            or other.r1 + pad < candidate.r0 - pad
            or candidate.c1 + pad < other.c0 - pad
            or other.c1 + pad < candidate.c0 - pad
        )
        if not separated:
            return False
    return True


def _render_grid(h: int, w: int, rects: list[Rect]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for rect in rects:
        grid[rect.r0, rect.c0 : rect.c1 + 1] = rect.border_color
        grid[rect.r1, rect.c0 : rect.c1 + 1] = rect.border_color
        grid[rect.r0 : rect.r1 + 1, rect.c0] = rect.border_color
        grid[rect.r0 : rect.r1 + 1, rect.c1] = rect.border_color
        gr, gc = rect.gap
        grid[gr, gc] = BG
        grid[rect.seed_r, rect.seed_c] = rect.seed_color
    return grid


def _rect_signature(rect: Rect) -> tuple[object, ...]:
    return (
        rect.r0,
        rect.c0,
        rect.r1,
        rect.c1,
        rect.border_color,
        rect.seed_color,
        rect.seed_r,
        rect.seed_c,
        rect.gap_side,
        rect.ray_length,
    )


def _place_random_rectangles(
    h: int,
    w: int,
    border_colors: list[int],
    seed_colors: list[int],
    existing: list[Rect] | None = None,
) -> list[Rect] | None:
    placed = list(existing or [])
    extras: list[Rect] = []

    for border_color, seed_color in zip(border_colors, seed_colors):
        rect = None
        for _ in range(PLACE_RETRIES):
            inner_h = random.choice((3, 5, 5, 7))
            inner_w = random.choice((3, 5, 5, 7))
            gap_side = random.choice(SIDES)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            max_r0 = h - outer_h - 1
            max_c0 = w - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                return None
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            candidate = _candidate_rect(r0, c0, inner_h, inner_w, gap_side, border_color, seed_color)
            if candidate is None or not _rect_clear(candidate, placed, h, w):
                continue
            rect = candidate
            break
        if rect is None:
            return None
        placed.append(rect)
        extras.append(rect)

    return extras


def _place_crossing_chain(
    h: int,
    w: int,
    direction: str,
    border_colors: list[int],
    seed_colors: list[int],
) -> list[Rect] | None:
    target_count = len(border_colors) - 1
    primary_h = random.choice(PRIMARY_INNER)
    primary_w = random.choice(PRIMARY_INNER)
    target_dims = [(random.choice(SMALL_INNER), random.choice(SMALL_INNER)) for _ in range(target_count)]

    if direction in HORIZONTAL_SIDES:
        outer_heights = [primary_h + 2] + [inner_h + 2 for inner_h, _inner_w in target_dims]
        half_span = max(height // 2 for height in outer_heights)
        min_center = 1 + half_span
        max_center = h - 2 - half_span
        if min_center > max_center:
            return None
        center = random.randint(min_center, max_center)

        widths = [primary_w + 2] + [inner_w + 2 for _inner_h, inner_w in target_dims]
        spaces = [random.randint(2, 4) for _ in range(target_count)]
        total_width = sum(widths) + sum(spaces)
        if total_width > w - 2:
            return None
        start_c = random.randint(1, w - total_width - 1)

        target_side_pool = VERTICAL_SIDES
        target_axis = "row"
        left_to_right_specs: list[tuple[str, int, int, int, int]] = []
        if direction == "right":
            left_to_right_specs.append(("right", primary_h, primary_w, border_colors[0], seed_colors[0]))
            left_to_right_specs.extend(
                (random.choice(target_side_pool), inner_h, inner_w, border_colors[i + 1], seed_colors[i + 1])
                for i, (inner_h, inner_w) in enumerate(target_dims)
            )
        else:
            left_to_right_specs.extend(
                (random.choice(target_side_pool), inner_h, inner_w, border_colors[i + 1], seed_colors[i + 1])
                for i, (inner_h, inner_w) in enumerate(target_dims)
            )
            left_to_right_specs.append(("left", primary_h, primary_w, border_colors[0], seed_colors[0]))

        rects: list[Rect] = []
        c = start_c
        for idx, (gap_side, inner_h, inner_w, border_color, seed_color) in enumerate(left_to_right_specs):
            r0 = center - (inner_h + 2) // 2
            avoid_axis = None
            if idx != 0 or direction == "left":
                if gap_side in target_side_pool:
                    avoid_axis = target_axis
            candidate = _candidate_rect(r0, c, inner_h, inner_w, gap_side, border_color, seed_color, avoid_axis)
            if candidate is None:
                return None
            rects.append(candidate)
            c += inner_w + 2
            if idx < len(spaces):
                c += spaces[idx]
        return rects

    outer_widths = [primary_w + 2] + [inner_w + 2 for _inner_h, inner_w in target_dims]
    half_span = max(width // 2 for width in outer_widths)
    min_center = 1 + half_span
    max_center = w - 2 - half_span
    if min_center > max_center:
        return None
    center = random.randint(min_center, max_center)

    heights = [primary_h + 2] + [inner_h + 2 for inner_h, _inner_w in target_dims]
    spaces = [random.randint(2, 4) for _ in range(target_count)]
    total_height = sum(heights) + sum(spaces)
    if total_height > h - 2:
        return None
    start_r = random.randint(1, h - total_height - 1)

    target_side_pool = HORIZONTAL_SIDES
    target_axis = "col"
    top_to_bottom_specs: list[tuple[str, int, int, int, int]] = []
    if direction == "down":
        top_to_bottom_specs.append(("down", primary_h, primary_w, border_colors[0], seed_colors[0]))
        top_to_bottom_specs.extend(
            (random.choice(target_side_pool), inner_h, inner_w, border_colors[i + 1], seed_colors[i + 1])
            for i, (inner_h, inner_w) in enumerate(target_dims)
        )
    else:
        top_to_bottom_specs.extend(
            (random.choice(target_side_pool), inner_h, inner_w, border_colors[i + 1], seed_colors[i + 1])
            for i, (inner_h, inner_w) in enumerate(target_dims)
        )
        top_to_bottom_specs.append(("up", primary_h, primary_w, border_colors[0], seed_colors[0]))

    rects = []
    r = start_r
    for idx, (gap_side, inner_h, inner_w, border_color, seed_color) in enumerate(top_to_bottom_specs):
        c0 = center - (inner_w + 2) // 2
        avoid_axis = None
        if idx != 0 or direction == "up":
            if gap_side in target_side_pool:
                avoid_axis = target_axis
        candidate = _candidate_rect(r, c0, inner_h, inner_w, gap_side, border_color, seed_color, avoid_axis)
        if candidate is None:
            return None
        rects.append(candidate)
        r += inner_h + 2
        if idx < len(spaces):
            r += spaces[idx]
    return rects


def _validate_generated_grid(grid: np.ndarray, rects: list[Rect], require_crossing: bool) -> bool:
    if _background_color(grid) != BG:
        return False

    detected = _detect_rectangles(grid)
    if sorted(_rect_signature(rect) for rect in detected) != sorted(_rect_signature(rect) for rect in rects):
        return False

    expected, stats = _simulate_rays(grid, detected)
    if bool(stats["crossed_any"]) != require_crossing:
        return False
    if bool(stats["collision"]) or bool(stats["seed_hit"]):
        return False
    if np.array_equal(expected, grid):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    rect_count = random.choice(RECT_COUNT_OPTIONS)
    require_crossing = random.choice([False, True])

    palette = random.sample(range(1, 10), 2 * rect_count)
    border_colors = palette[:rect_count]
    seed_colors = palette[rect_count:]

    if require_crossing:
        target_count = random.randint(1, min(2, rect_count - 1))
        direction = random.choice(SIDES)
        chain = _place_crossing_chain(
            h,
            w,
            direction,
            border_colors[: target_count + 1],
            seed_colors[: target_count + 1],
        )
        if chain is None:
            return None
        extras = _place_random_rectangles(
            h,
            w,
            border_colors[target_count + 1 :],
            seed_colors[target_count + 1 :],
            existing=chain,
        )
        if extras is None:
            return None
        rects = chain + extras
    else:
        rects = _place_random_rectangles(h, w, border_colors, seed_colors)
        if rects is None:
            return None

    grid = _render_grid(h, w, rects)
    if not _validate_generated_grid(grid, rects, require_crossing=require_crossing):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-ray puzzle")
