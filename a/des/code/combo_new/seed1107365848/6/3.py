from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (14, 30)
GRID_WIDTH_RANGE = (14, 30)
RECT_COUNT_OPTIONS = [1, 2, 2, 3, 3, 4]
INNER_HEIGHT_RANGE = (2, 6)
INNER_WIDTH_RANGE = (2, 7)
GAP_WIDTH_OPTIONS = (1, 3, 5)
PLACEMENT_TRIES = 140
GENERATION_RETRIES = 400

DIRECTION_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _sorted_gap_cells(side: str, gap_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if side in {"up", "down"}:
        return sorted(gap_cells, key=lambda p: p[1])
    return sorted(gap_cells, key=lambda p: p[0])


def _detect_frame(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    comp = {tuple(map(int, rc)) for rc in coords}
    border = _border_cells(r0, c0, r1, c1)
    if not comp <= border:
        return None

    missing = border - comp
    if not missing:
        return None

    matches: list[tuple[str, list[tuple[int, int]]]] = []
    if all(r == r0 for r, _c in missing):
        gap = _sorted_gap_cells("up", missing)
        cols = [c for _r, c in gap]
        if cols == list(range(cols[0], cols[-1] + 1)):
            matches.append(("up", gap))
    if all(r == r1 for r, _c in missing):
        gap = _sorted_gap_cells("down", missing)
        cols = [c for _r, c in gap]
        if cols == list(range(cols[0], cols[-1] + 1)):
            matches.append(("down", gap))
    if all(c == c0 for _r, c in missing):
        gap = _sorted_gap_cells("left", missing)
        rows = [r for r, _c in gap]
        if rows == list(range(rows[0], rows[-1] + 1)):
            matches.append(("left", gap))
    if all(c == c1 for _r, c in missing):
        gap = _sorted_gap_cells("right", missing)
        rows = [r for r, _c in gap]
        if rows == list(range(rows[0], rows[-1] + 1)):
            matches.append(("right", gap))

    if len(matches) != 1:
        return None

    side, gap_cells = matches[0]
    return {
        "bbox": (r0, c0, r1, c1),
        "gap_side": side,
        "gap_cells": gap_cells,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            frame = _detect_frame(coords)
            if frame is None:
                continue
            frame["color"] = color
            frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot an outward ray from each gapped rectangular frame using its interior seed count and winning seed color.

    Input structure:
    - The background is black, and one-pixel-thick rectangular frames sit on it.
    - Each frame is monochrome, has a single contiguous gap on exactly one side, and encloses several colored single-pixel seeds.
    - Seed colors differ from the frame and background colors, and multiple frames may appear in the same grid.

    Rule:
    - Detect each rectangular frame and determine which side contains its gap.
    - Count all seed pixels inside the frame to get the ray length N.
    - Choose the ray color by majority vote among the interior seed colors; if multiple colors tie, use the tied color whose nearest seed is closest to the gap.
    - Remove all interior seeds, keep the gap open, and draw a straight ray of length N just outside the gap in the outward normal direction.

    Key insight:
    - The gap encodes direction, the total number of seeds encodes ray length, and the seed-color histogram encodes the ray color, with proximity to the gap resolving vote ties.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BACKGROUND if bool((g == BACKGROUND).any()) else int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    out = g.copy()

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = map(int, frame["bbox"])
        frame_color = int(frame["color"])
        gap_cells = list(frame["gap_cells"])
        gap_anchor = gap_cells[len(gap_cells) // 2]
        dr, dc = DIRECTION_STEPS[str(frame["gap_side"])]

        seed_positions: list[tuple[int, int]] = []
        counts: Counter[int] = Counter()
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(g[r, c])
                if value in (bg, frame_color):
                    continue
                seed_positions.append((r, c))
                counts[value] += 1

        if not seed_positions:
            continue

        max_count = max(counts.values())
        candidates = [color for color, cnt in counts.items() if int(cnt) == int(max_count)]

        def color_key(color: int) -> tuple[int, int, int]:
            best_seed = min(
                (pos for pos in seed_positions if int(g[pos]) == int(color)),
                key=lambda pos: (_distance(pos, gap_anchor), pos[0], pos[1]),
            )
            return (_distance(best_seed, gap_anchor), best_seed[0], best_seed[1])

        ray_color = min(candidates, key=color_key)

        for r, c in seed_positions:
            out[r, c] = bg

        for step in range(1, len(seed_positions) + 1):
            rr = int(gap_anchor[0] + dr * step)
            cc = int(gap_anchor[1] + dc * step)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = int(ray_color)

    return out


def _choose_gap_width(side: str, inner_h: int, inner_w: int) -> int:
    span = inner_w if side in {"up", "down"} else inner_h
    options = [width for width in GAP_WIDTH_OPTIONS if int(width) <= int(span)]
    return int(random.choice(options))


def _local_frame_spec() -> dict:
    inner_h = random.randint(*INNER_HEIGHT_RANGE)
    inner_w = random.randint(*INNER_WIDTH_RANGE)
    side = random.choice(list(DIRECTION_STEPS))
    gap_width = _choose_gap_width(side, inner_h, inner_w)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    frame_cells = _border_cells(0, 0, outer_h - 1, outer_w - 1)

    if side in {"up", "down"}:
        gap_offset = random.randint(0, inner_w - gap_width)
        row = 0 if side == "up" else outer_h - 1
        gap_cells = [(row, 1 + gap_offset + i) for i in range(gap_width)]
    else:
        gap_offset = random.randint(0, inner_h - gap_width)
        col = 0 if side == "left" else outer_w - 1
        gap_cells = [(1 + gap_offset + i, col) for i in range(gap_width)]

    for cell in gap_cells:
        frame_cells.remove(cell)

    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "side": side,
        "gap_cells": gap_cells,
        "gap_anchor": gap_cells[len(gap_cells) // 2],
        "frame_cells": frame_cells,
    }


def _seed_plan(spec: dict, branch: str) -> tuple[list[tuple[int, int, int]], int] | None:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    gap_anchor = tuple(map(int, spec["gap_anchor"]))

    cells = [(r, c) for r in range(1, inner_h + 1) for c in range(1, inner_w + 1)]
    ordered = sorted(cells, key=lambda pos: (_distance(pos, gap_anchor), pos[0], pos[1]))

    if branch == "tie":
        if len(cells) < 2:
            return None
        anchor = ordered[0]
        anchor_dist = _distance(anchor, gap_anchor)
        farther = [cell for cell in ordered[1:] if _distance(cell, gap_anchor) > anchor_dist]
        feasible_pairs = [
            k
            for k in range(1, min(3, len(cells) // 2) + 1)
            if len(farther) >= k and len(cells) >= 2 * k
        ]
        if not feasible_pairs:
            return None

        pair_count = random.choice(feasible_pairs)
        winner_color, loser_color = random.sample(SEED_COLORS, 2)

        loser_positions = random.sample(farther, pair_count)
        winner_pool = [cell for cell in cells if cell not in loser_positions and cell != anchor]
        winner_positions = [anchor]
        if pair_count > 1:
            winner_positions.extend(random.sample(winner_pool, pair_count - 1))

        seeds = [(r, c, winner_color) for r, c in winner_positions]
        seeds.extend((r, c, loser_color) for r, c in loser_positions)
        return seeds, int(winner_color)

    max_total = min(6, len(cells))
    if max_total <= 0:
        return None

    if max_total < 3:
        total = max_total
        winner_color = random.choice(SEED_COLORS)
        positions = random.sample(cells, total)
        return [(r, c, winner_color) for r, c in positions], int(winner_color)

    total = random.randint(3, max_total)
    winner_color = random.choice(SEED_COLORS)
    remaining_palette = [color for color in SEED_COLORS if int(color) != int(winner_color)]
    positions = random.sample(cells, total)

    if random.random() < 0.2:
        winner_count = total
    else:
        winner_count = random.randint(total // 2 + 1, total - 1)

    seeds = [(r, c, winner_color) for r, c in positions[:winner_count]]
    others = positions[winner_count:]
    if others:
        loser_colors = random.sample(remaining_palette, min(2, len(remaining_palette)))
        for r, c in others:
            seeds.append((r, c, random.choice(loser_colors)))
    return seeds, int(winner_color)


def _build_object(branch: str) -> dict | None:
    spec = _local_frame_spec()
    made = _seed_plan(spec, branch)
    if made is None:
        return None
    seeds, ray_color = made

    dr, dc = DIRECTION_STEPS[str(spec["side"])]
    ray_length = len(seeds)
    ray_cells = [
        (
            int(spec["gap_anchor"][0] + dr * step),
            int(spec["gap_anchor"][1] + dc * step),
        )
        for step in range(1, ray_length + 1)
    ]

    spec["seeds"] = seeds
    spec["ray_color"] = int(ray_color)
    spec["ray_cells"] = ray_cells
    return spec


def _translate_points(points: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r + r0), int(c + c0)) for r, c in points]


def _translate_seed_points(points: list[tuple[int, int, int]], r0: int, c0: int) -> list[tuple[int, int, int]]:
    return [(int(r + r0), int(c + c0), int(color)) for r, c, color in points]


def _candidate_ranges(spec: dict, height: int, width: int) -> tuple[int, int, int, int] | None:
    all_cells = list(spec["frame_cells"]) + [(r, c) for r, c, _color in spec["seeds"]] + list(spec["ray_cells"])
    min_r = min(r for r, _c in all_cells)
    max_r = max(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    max_c = max(c for _r, c in all_cells)

    r_low = max(1, -min_r)
    r_high = min(height - int(spec["outer_h"]) - 1, height - 1 - max_r)
    c_low = max(1, -min_c)
    c_high = min(width - int(spec["outer_w"]) - 1, width - 1 - max_c)
    if r_low > r_high or c_low > c_high:
        return None
    return int(r_low), int(r_high), int(c_low), int(c_high)


def _fits(points: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    height, width = blocked.shape
    for r, c in points:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if bool(blocked[r, c]):
            return False
    return True


def _place_object(spec: dict, height: int, width: int, blocked: np.ndarray) -> dict | None:
    ranges = _candidate_ranges(spec, height, width)
    if ranges is None:
        return None
    r_low, r_high, c_low, c_high = ranges

    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(r_low, r_high)
        c0 = random.randint(c_low, c_high)
        frame_cells = _translate_points(list(spec["frame_cells"]), r0, c0)
        seeds = _translate_seed_points(list(spec["seeds"]), r0, c0)
        ray_cells = _translate_points(list(spec["ray_cells"]), r0, c0)

        footprint = frame_cells + [(r, c) for r, c, _color in seeds] + ray_cells
        if not _fits(footprint, blocked):
            continue

        placed = {
            "frame_cells": frame_cells,
            "gap_cells": _translate_points(list(spec["gap_cells"]), r0, c0),
            "gap_anchor": (int(spec["gap_anchor"][0] + r0), int(spec["gap_anchor"][1] + c0)),
            "seeds": seeds,
            "ray_cells": ray_cells,
            "ray_color": int(spec["ray_color"]),
            "bbox": (r0, c0, r0 + int(spec["outer_h"]) - 1, c0 + int(spec["outer_w"]) - 1),
        }
        return placed
    return None


def _mark_blocked(blocked: np.ndarray, points: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    blocked[rr, cc] = True


def _render_scene(height: int, width: int, objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BACKGROUND, dtype=int)
    out = np.full((height, width), BACKGROUND, dtype=int)

    for obj in objects:
        for r, c in obj["frame_cells"]:
            inp[r, c] = FRAME_COLOR
            out[r, c] = FRAME_COLOR
        for r, c, color in obj["seeds"]:
            inp[r, c] = int(color)
        for r, c in obj["ray_cells"]:
            out[r, c] = int(obj["ray_color"])
    return inp, out


def _branches_for_count(count: int) -> list[str]:
    branches = [random.choice(["majority", "tie"]) for _ in range(count)]
    if count >= 2 and random.random() < 0.65:
        branches[0] = "majority"
        branches[1] = "tie"
        random.shuffle(branches)
    return branches


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    max_rects = 3 if min(height, width) < 18 else 4
    target_count = min(random.choice(RECT_COUNT_OPTIONS), max_rects)

    blocked = np.zeros((height, width), dtype=bool)
    objects: list[dict] = []

    for branch in _branches_for_count(target_count):
        placed = None
        for _ in range(PLACEMENT_TRIES):
            obj = _build_object(branch)
            if obj is None:
                continue
            placed = _place_object(obj, height, width, blocked)
            if placed is None:
                continue
            points = list(placed["frame_cells"]) + [(r, c) for r, c, _color in placed["seeds"]] + list(placed["ray_cells"])
            _mark_blocked(blocked, points)
            objects.append(placed)
            break
        if placed is None:
            return None

    inp, expected = _render_scene(height, width, objects)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ray-from-gap ARC input")
