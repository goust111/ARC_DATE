from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

PALETTE = tuple(range(10))
SIDE_CHOICES = (5, 7)
FENCE_COUNT_CHOICES = (1, 2, 3, 4)
FENCE_COUNT_WEIGHTS = (1, 3, 3, 2)
MAX_GENERATION_ATTEMPTS = 500
BOX_PLACEMENT_TRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_legend(row: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    width = row.shape[0]

    while c < width:
        while c < width and int(row[c]) == bg:
            c += 1
        if c >= width:
            break
        assert c + 1 < width, "legend pairs must have width 2"
        src = int(row[c])
        dst = int(row[c + 1])
        assert src != bg and dst != bg, "legend pairs must be non-background"
        assert src not in mapping, "legend source colors must be unique"
        mapping[src] = dst
        c += 2
        if c < width:
            assert int(row[c]) == bg or c == width, "legend pairs must be separated by background"

    assert mapping, "expected at least one legend pair on the bottom row"
    return mapping


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _midpoints(r0: int, c0: int, side: int) -> dict[str, tuple[int, int]]:
    half = side // 2
    return {
        "up": (r0, c0 + half),
        "down": (r0 + side - 1, c0 + half),
        "left": (r0 + half, c0),
        "right": (r0 + half, c0 + side - 1),
    }


def _center_of_box(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _collect_fences(body: np.ndarray, mapping: dict[int, int], bg: int) -> tuple[int, list[dict[str, object]]]:
    non_bg_colors = {int(v) for v in np.unique(body) if int(v) != bg}
    fence_candidates = sorted(color for color in non_bg_colors if color not in mapping)
    assert len(fence_candidates) == 1, "expected exactly one fence color"
    fence_color = fence_candidates[0]

    comps = _components(body == fence_color)
    assert comps, "expected at least one fence"

    fences: list[dict[str, object]] = []
    side_lengths = set()
    seen_seeds = set()
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1, "fence bbox must be square"
        assert side % 2 == 1 and side >= 3, "fence size must be odd"
        side_lengths.add(side)

        mids = _midpoints(r0, c0, side)
        gap_sides = [name for name, rc in mids.items() if int(body[rc]) != fence_color]
        assert len(gap_sides) == 1, "each fence must have exactly one side-midpoint gap"
        gap_side = gap_sides[0]

        center = (r0 + side // 2, c0 + side // 2)
        seed_color = int(body[center])
        assert seed_color in mapping, "fence center must contain a legend source color"
        assert seed_color not in seen_seeds, "seed colors must be unique"
        seen_seeds.add(seed_color)

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border:
                    expected = bg if (r, c) == mids[gap_side] else fence_color
                    assert int(body[r, c]) == expected, "fence border must be solid except for one midpoint gap"
                elif (r, c) == center:
                    assert int(body[r, c]) == seed_color
                else:
                    assert int(body[r, c]) == bg, "fence interior must be empty except for the center seed"

        fences.append(
            {
                "box": (r0, c0, r1, c1),
                "center": center,
                "seed": seed_color,
                "gap_side": gap_side,
            }
        )

    assert len(side_lengths) == 1, "all fences must have the same size"
    assert seen_seeds == set(mapping), "legend source colors must match fence seed colors"
    return fence_color, fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow mapped-color rays straight through each square fence gap.

    Input structure:
    - The background is the most common color.
    - The main area contains several equal-sized odd square fences of one color, each a 1-pixel hollow frame with exactly one missing side-midpoint cell.
    - Every fence has one distinct seed pixel at its exact center.
    - The bottom row is a legend made of adjacent source/target color pairs, separated by background gaps.

    Rule:
    - Read the bottom legend as a mapping from each seed color to its ray color.
    - For every fence, detect which side has the missing midpoint; that side gives the seed's growth direction.
    - From the center seed, paint a straight ray in that direction through the fence interior and gap, continuing across background cells until the path leaves the grid or the next cell is another fence border.
    - Keep all fence pixels, seed pixels, and legend pixels unchanged; only background cells on the ray path are recolored.

    Key insight:
    - The missing midpoint is not merely a hole to copy: it encodes the ray direction, while the actual ray color is supplied independently by the bottom-row legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _get_bg(g)
    mapping = _parse_legend(g[-1], bg)
    fence_color, fences = _collect_fences(g[:-1], mapping, bg)

    out = g.copy()
    h, w = g.shape
    for fence in fences:
        dr, dc = DIRS[str(fence["gap_side"])]
        r, c = fence["center"]
        r += dr
        c += dc
        ray_color = mapping[int(fence["seed"])]

        while 0 <= r < h and 0 <= c < w:
            if int(g[r, c]) == fence_color:
                break
            assert int(g[r, c]) == bg, "ray path must cross only background cells outside fences"
            out[r, c] = ray_color
            r += dr
            c += dc

    return out


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _random_box(existing: list[tuple[int, int, int, int]], body_h: int, width: int, side: int) -> tuple[int, int, int, int] | None:
    for _ in range(BOX_PLACEMENT_TRIES):
        r0 = random.randint(0, body_h - side)
        c0 = random.randint(0, width - side)
        box = (r0, c0, r0 + side - 1, c0 + side - 1)
        if any(_boxes_conflict(box, other) for other in existing):
            continue
        return box
    return None


def _place_boxes(
    body_h: int,
    width: int,
    side: int,
    count: int,
    scenario: str,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int] | None]:
    boxes: list[tuple[int, int, int, int]] = []
    blocker_pair: tuple[int, int] | None = None

    if scenario == "mixed":
        if random.random() < 0.5:
            if width < 2 * side + 2:
                return [], None
            r0 = random.randint(0, body_h - side)
            left0 = random.randint(0, width - 2 * side - 2)
            max_gap = width - left0 - 2 * side
            gap = random.randint(2, max_gap)
            right0 = left0 + side + gap
            box_a = (r0, left0, r0 + side - 1, left0 + side - 1)
            box_b = (r0, right0, r0 + side - 1, right0 + side - 1)
        else:
            if body_h < 2 * side + 2:
                return [], None
            c0 = random.randint(0, width - side)
            top0 = random.randint(0, body_h - 2 * side - 2)
            max_gap = body_h - top0 - 2 * side
            gap = random.randint(2, max_gap)
            bottom0 = top0 + side + gap
            box_a = (top0, c0, top0 + side - 1, c0 + side - 1)
            box_b = (bottom0, c0, bottom0 + side - 1, c0 + side - 1)
        boxes.extend([box_a, box_b])
        blocker_pair = (0, 1)

    while len(boxes) < count:
        box = _random_box(boxes, body_h, width, side)
        if box is None:
            return [], None
        boxes.append(box)

    return boxes, blocker_pair


def _direction_toward(source: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> str:
    sr, sc = _center_of_box(source)
    tr, tc = _center_of_box(target)
    if sr == tr:
        return "right" if tc > sc else "left"
    assert sc == tc
    return "down" if tr > sr else "up"


def _assign_directions(
    boxes: list[tuple[int, int, int, int]],
    scenario: str,
    blocker_pair: tuple[int, int] | None,
) -> list[str]:
    directions = [random.choice(tuple(DIRS)) for _ in boxes]
    if scenario != "mixed" or blocker_pair is None:
        return directions

    first, second = blocker_pair
    shooter, blocker = (first, second) if random.random() < 0.5 else (second, first)
    toward = _direction_toward(boxes[shooter], boxes[blocker])
    directions[shooter] = toward

    allowed = [name for name in DIRS if name != OPPOSITE[toward]]
    directions[blocker] = random.choice(allowed)
    return directions


def _draw_fence(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    direction: str,
    fence_color: int,
    bg: int,
    seed_color: int,
) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = fence_color
    grid[r0 : r1 + 1, c1] = fence_color
    grid[r0, c0 : c1 + 1] = fence_color
    grid[r1, c0 : c1 + 1] = fence_color

    gap_r, gap_c = _midpoints(r0, c0, r1 - r0 + 1)[direction]
    grid[gap_r, gap_c] = bg

    center_r, center_c = _center_of_box(box)
    grid[center_r, center_c] = seed_color


def _trace_path(
    grid: np.ndarray,
    center: tuple[int, int],
    direction: str,
    fence_color: int,
    bg: int,
) -> tuple[list[tuple[int, int]], str] | None:
    dr, dc = DIRS[direction]
    h, w = grid.shape
    r = center[0] + dr
    c = center[1] + dc
    path: list[tuple[int, int]] = []

    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value == fence_color:
            return path, "blocked"
        if value != bg:
            return None
        path.append((r, c))
        r += dr
        c += dc

    return path, "edge"


def _pick_legend_starts(width: int, count: int, blocked_cols: set[int]) -> list[int] | None:
    starts = list(range(width - 1))
    random.shuffle(starts)

    def backtrack(chosen: list[int], min_start: int) -> list[int] | None:
        if len(chosen) == count:
            return chosen.copy()

        candidates = [c for c in starts if c >= min_start]
        random.shuffle(candidates)
        for start in candidates:
            if start in blocked_cols or start + 1 in blocked_cols:
                continue
            if chosen and start < chosen[-1] + 3:
                continue
            result = backtrack(chosen + [start], start + 3)
            if result is not None:
                return result
        return None

    return backtrack([], 0)


def _generate_attempt() -> np.ndarray | None:
    count = random.choices(FENCE_COUNT_CHOICES, weights=FENCE_COUNT_WEIGHTS, k=1)[0]
    scenario = "mixed" if count >= 2 and random.random() < 0.5 else "edge"
    side = random.choice(SIDE_CHOICES)

    legend_len = 3 * count - 1
    min_body_h = max(side + 4, 2 * side + 2 if scenario == "mixed" else side + 2)
    min_width = max(legend_len + 4, 2 * side + 2)
    if min_body_h > 29 or min_width > 30:
        return None

    body_h = random.randint(min_body_h, 29)
    width = random.randint(min_width, 30)

    boxes, blocker_pair = _place_boxes(body_h, width, side, count, scenario)
    if len(boxes) != count:
        return None

    if 2 * count + 2 > len(PALETTE):
        return None
    colors = list(PALETTE)
    random.shuffle(colors)
    bg = colors[0]
    fence_color = colors[1]
    seed_colors = colors[2 : 2 + count]
    target_colors = colors[2 + count : 2 + 2 * count]

    directions = _assign_directions(boxes, scenario, blocker_pair)
    grid = np.full((body_h + 1, width), bg, dtype=int)
    fence_infos: list[dict[str, object]] = []
    for box, direction, seed_color, target_color in zip(boxes, directions, seed_colors, target_colors):
        _draw_fence(grid, box, direction, fence_color, bg, seed_color)
        fence_infos.append(
            {
                "box": box,
                "center": _center_of_box(box),
                "direction": direction,
                "seed": seed_color,
                "target": target_color,
            }
        )

    all_paths: list[list[tuple[int, int]]] = []
    modes: list[str] = []
    occupied = set()
    for info in fence_infos:
        traced = _trace_path(grid, info["center"], str(info["direction"]), fence_color, bg)
        if traced is None:
            return None
        path, mode = traced
        cells = set(path)
        if occupied & cells:
            return None
        occupied |= cells
        all_paths.append(path)
        modes.append(mode)

    if scenario == "edge" and any(mode != "edge" for mode in modes):
        return None
    if scenario == "mixed" and not ("edge" in modes and "blocked" in modes):
        return None

    bottom_row = body_h
    blocked_cols = {c for r, c in occupied if r == bottom_row}
    legend_pairs = list(zip(seed_colors, target_colors))
    random.shuffle(legend_pairs)
    starts = _pick_legend_starts(width, count, blocked_cols)
    if starts is None:
        return None

    for start, (src, dst) in zip(sorted(starts), legend_pairs):
        grid[bottom_row, start] = src
        grid[bottom_row, start + 1] = dst

    expected = grid.copy()
    mapping = {src: dst for src, dst in legend_pairs}
    for info, path in zip(fence_infos, all_paths):
        ray_color = mapping[int(info["seed"])]
        for r, c in path:
            expected[r, c] = ray_color

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    if _get_bg(grid) != bg:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fence-ray sample")
