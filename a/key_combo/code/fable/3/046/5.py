from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_COLORS = tuple(range(10))
MAX_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_candidate(g: np.ndarray, bg: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if int(coords.shape[0]) != h * w:
        return None

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros(patch.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_values = set(map(int, patch[border].reshape(-1)))
    if len(border_values) != 1:
        return None
    fill = next(iter(border_values))
    if fill == int(bg):
        return None

    marker_mask = patch != int(fill)
    marker_mask[border] = False
    marker_count = int(marker_mask.sum())
    if marker_count < 1 or marker_count > 4:
        return None
    if not bool(np.all(patch[border] == int(fill))):
        return None

    marker_colors = set(map(int, patch[marker_mask].reshape(-1)))
    if len(marker_colors) != 1:
        return None
    marker_color = next(iter(marker_colors))
    if marker_color in (int(bg), int(fill)):
        return None

    interior_non_marker = (~border) & (~marker_mask)
    if not bool(np.all(patch[interior_non_marker] == int(fill))):
        return None

    markers = [
        (int(r0 + rr), int(c0 + cc))
        for rr, cc in np.argwhere(marker_mask)
    ]
    return {
        "bbox": (r0, c0, r1, c1),
        "fill_color": int(fill),
        "marker_color": int(marker_color),
        "markers": markers,
        "count": int(marker_count),
        "coords": [(int(r), int(c)) for r, c in coords],
    }


def _instruction_candidates(g: np.ndarray, bg: int, coords: np.ndarray) -> list[dict]:
    pts = {(int(r), int(c)) for r, c in coords}
    if not (2 <= len(pts) <= 5):
        return []
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 > 3 or c1 - c0 + 1 > 3:
        return []

    candidates: list[dict] = []
    for cr, cc in pts:
        dirs: list[tuple[int, int]] = []
        arm_colors: list[int] = []
        ok = True
        for r, c in pts:
            if (r, c) == (cr, cc):
                continue
            dr, dc = r - cr, c - cc
            if abs(dr) + abs(dc) != 1 or (dr, dc) not in DIRS:
                ok = False
                break
            dirs.append((dr, dc))
            arm_colors.append(int(g[r, c]))
        if not ok or not (1 <= len(dirs) <= 4):
            continue
        if len(set(dirs)) != len(dirs) or len(set(arm_colors)) != 1:
            continue
        center_color = int(g[cr, cc])
        arm_color = int(arm_colors[0])
        if center_color == int(bg) or arm_color == int(bg):
            continue
        candidates.append(
            {
                "center": (int(cr), int(cc)),
                "center_color": center_color,
                "arm_color": arm_color,
                "dirs": tuple(sorted(dirs)),
                "count": int(len(dirs)),
                "cells": sorted(pts),
            }
        )
    return candidates


def _select_instructions(candidate_lists: list[list[dict]]) -> list[dict]:
    assert candidate_lists, "expected at least one instruction cross"

    common_center_colors = set(c["center_color"] for c in candidate_lists[0])
    for cands in candidate_lists[1:]:
        common_center_colors &= set(c["center_color"] for c in cands)

    assignments: list[list[dict]] = []
    for center_color in common_center_colors:
        chosen: list[dict] = []
        ok = True
        for cands in candidate_lists:
            matches = [c for c in cands if int(c["center_color"]) == int(center_color)]
            if len(matches) != 1:
                ok = False
                break
            chosen.append(matches[0])
        if not ok:
            continue
        counts = [int(c["count"]) for c in chosen]
        if len(counts) == len(set(counts)):
            assignments.append(chosen)

    if len(assignments) == 1:
        return assignments[0]

    if not assignments and all(len(cands) == 1 for cands in candidate_lists):
        chosen = [cands[0] for cands in candidate_lists]
        counts = [int(c["count"]) for c in chosen]
        if len(counts) == len(set(counts)):
            return chosen

    raise AssertionError("instruction centers or arm counts are ambiguous")


def _parse_input(g: np.ndarray) -> tuple[int, list[dict], dict[int, dict]]:
    bg = _get_bg(g)
    rectangles: list[dict] = []
    raw_instruction_candidates: list[list[dict]] = []

    for comp in _components(g != int(bg)):
        rect = _rectangle_candidate(g, bg, comp)
        if rect is not None:
            rectangles.append(rect)
            continue
        cands = _instruction_candidates(g, bg, comp)
        assert cands, "unexpected non-background component"
        raw_instruction_candidates.append(cands)

    assert rectangles, "expected at least one marked rectangle"
    instructions = _select_instructions(raw_instruction_candidates)
    by_count: dict[int, dict] = {}
    for instr in instructions:
        count = int(instr["count"])
        assert count not in by_count, "instruction arm counts must be unique"
        by_count[count] = instr

    for rect in rectangles:
        assert int(rect["count"]) in by_count, "every rectangle marker count must match an instruction arm count"

    return bg, rectangles, by_count


def _apply_rule(g: np.ndarray, bg: int, rectangles: list[dict], instructions: dict[int, dict]) -> np.ndarray:
    out = g.copy()

    for instr in instructions.values():
        for r, c in instr["cells"]:
            out[int(r), int(c)] = int(bg)

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        instr = instructions[int(rect["count"])]
        arm_color = int(instr["arm_color"])
        center_color = int(instr["center_color"])

        for mr, mc in rect["markers"]:
            for dr, dc in instr["dirs"]:
                r, c = int(mr) + int(dr), int(mc) + int(dc)
                while r0 <= r <= r1 and c0 <= c <= c1:
                    out[r, c] = arm_color
                    r += int(dr)
                    c += int(dc)

        for mr, mc in rect["markers"]:
            out[int(mr), int(mc)] = center_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair rectangles to crosshair instructions by equal counts, then stamp and extend the selected arms.

    Input structure:
    - The background is the most common color.
    - Several solid axis-aligned rectangles are filled by one color, with one to four
      interior marker pixels of a single different color in each rectangle.
    - Outside the rectangles are small crosshair instruction components: one center
      pixel plus one to four adjacent arm pixels, with unique arm counts across the
      instruction set; the shared center color makes one-arm directions unambiguous.

    Rule:
    - For each rectangle, choose the instruction whose number of arms equals that
      rectangle's number of marker pixels; distance and colors do not participate.
    - Erase all original instruction crosses outside the rectangles.
    - At every marker, copy the chosen instruction center and extend each enabled
      arm direction in the instruction arm color until it reaches that rectangle's edge.

    Key insight:
    - The pairing key is count equality between marker pixels and instruction arms;
      the instruction's direction set, not its position or color, decides how each
      marker grows inside its own rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    bg, rectangles, instructions = _parse_input(g)
    return _apply_rule(g, bg, rectangles, instructions)


# ----------------------------- generator -----------------------------


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _pad_rect(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _place_box(
    rows: int,
    cols: int,
    height: int,
    width: int,
    occupied: list[tuple[int, int, int, int]],
    *,
    pad: int,
) -> tuple[int, int] | None:
    candidates = [(r, c) for r in range(rows - height + 1) for c in range(cols - width + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rect = (r, c, r + height - 1, c + width - 1)
        padded = _pad_rect(rect, pad)
        if any(_rects_overlap(padded, other) for other in occupied):
            continue
        return int(r), int(c)
    return None


def _direction_set(count: int) -> tuple[tuple[int, int], ...]:
    if count == 4:
        return DIRS
    dirs = random.sample(list(DIRS), int(count))
    return tuple(sorted(dirs))


def _choose_counts() -> list[int]:
    if random.random() < 0.55:
        counts = [1, 2, 3, 4]
    else:
        n = random.randint(2, 4)
        counts = random.sample([1, 2, 3, 4], n)
        if 1 not in counts and random.random() < 0.65:
            counts[random.randrange(len(counts))] = 1
    random.shuffle(counts)
    return counts


def _choose_colors(counts: list[int]) -> tuple[int, int, dict[int, int], dict[int, int], dict[int, int]]:
    bg = random.choice(ALL_COLORS)
    palette = [c for c in ALL_COLORS if c != bg]
    center_color = random.choice(palette)
    remaining = [c for c in palette if c != center_color]

    fill_samples = random.sample(remaining, len(counts))
    fill_colors = dict(zip(counts, fill_samples))

    marker_pool = [c for c in remaining if c not in set(fill_samples)]
    if len(marker_pool) < len(counts):
        marker_pool = [c for c in palette if c != center_color]
    marker_samples = random.sample(marker_pool, len(counts))
    marker_colors = dict(zip(counts, marker_samples))

    shifted_counts = counts[1:] + counts[:1]
    arm_colors = {
        count: int(marker_colors[shifted])
        for count, shifted in zip(counts, shifted_counts)
    }
    return int(bg), int(center_color), fill_colors, marker_colors, arm_colors


def _marker_positions(r0: int, c0: int, h: int, w: int, count: int) -> list[tuple[int, int]]:
    rows = random.sample(range(r0 + 1, r0 + h - 1), int(count))
    cols = random.sample(range(c0 + 1, c0 + w - 1), int(count))
    random.shuffle(cols)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def _draw_rectangles(
    grid: np.ndarray,
    counts: list[int],
    fill_colors: dict[int, int],
    marker_colors: dict[int, int],
) -> list[dict] | None:
    rows, cols = grid.shape
    occupied: list[tuple[int, int, int, int]] = []
    rects: list[dict] = []
    for count in counts:
        min_side = max(5, int(count) + 2)
        h = random.randint(min_side, 9)
        w = random.randint(min_side, 9)
        loc = _place_box(rows, cols, h, w, occupied, pad=2)
        if loc is None:
            return None
        r0, c0 = loc
        r1, c1 = r0 + h - 1, c0 + w - 1
        fill = int(fill_colors[count])
        marker_color = int(marker_colors[count])
        grid[r0 : r1 + 1, c0 : c1 + 1] = fill
        markers = _marker_positions(r0, c0, h, w, int(count))
        for r, c in markers:
            grid[r, c] = marker_color
        rect = {
            "bbox": (r0, c0, r1, c1),
            "fill_color": fill,
            "marker_color": marker_color,
            "markers": markers,
            "count": int(count),
            "coords": [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)],
        }
        rects.append(rect)
        occupied.append((r0, c0, r1, c1))
    return rects


def _cross_cells(center: tuple[int, int], dirs: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    r, c = center
    return [(int(r), int(c))] + [(int(r + dr), int(c + dc)) for dr, dc in dirs]


def _can_place_cells(grid: np.ndarray, bg: int, cells: list[tuple[int, int]], pad: int = 1) -> bool:
    rows, cols = grid.shape
    for r, c in cells:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < rows and 0 <= cc < cols and int(grid[rr, cc]) != int(bg):
                    return False
    return True


def _rect_center(rect: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    return (r0 + r1) // 2, (c0 + c1) // 2


def _place_cross(
    grid: np.ndarray,
    bg: int,
    count: int,
    dirs: tuple[tuple[int, int], ...],
    center_color: int,
    arm_color: int,
    prefer: tuple[int, int] | None,
) -> dict | None:
    rows, cols = grid.shape
    candidates: list[tuple[int, int, float]] = []
    for r in range(rows):
        for c in range(cols):
            cells = _cross_cells((r, c), dirs)
            if not _can_place_cells(grid, bg, cells, pad=1):
                continue
            score = random.random()
            if prefer is not None:
                score += abs(r - prefer[0]) + abs(c - prefer[1])
            candidates.append((r, c, score))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2])
    r, c, _ = random.choice(candidates[: min(10, len(candidates))])
    grid[r, c] = int(center_color)
    for dr, dc in dirs:
        grid[r + dr, c + dc] = int(arm_color)
    return {
        "center": (int(r), int(c)),
        "center_color": int(center_color),
        "arm_color": int(arm_color),
        "dirs": tuple(sorted(dirs)),
        "count": int(count),
        "cells": sorted(_cross_cells((int(r), int(c)), dirs)),
    }


def _draw_instructions(
    grid: np.ndarray,
    bg: int,
    counts: list[int],
    rects: list[dict],
    center_color: int,
    arm_colors: dict[int, int],
) -> dict[int, dict] | None:
    instructions: dict[int, dict] = {}
    count_to_rect = {int(rect["count"]): rect for rect in rects}
    for i, count in enumerate(counts):
        wrong_count = counts[(i + 1) % len(counts)]
        prefer = _rect_center(count_to_rect[int(wrong_count)])
        dirs = _direction_set(int(count))
        instr = _place_cross(
            grid,
            bg,
            int(count),
            dirs,
            center_color,
            int(arm_colors[count]),
            prefer,
        )
        if instr is None:
            return None
        instructions[int(count)] = instr
    return instructions


def _distance_to_rect(point: tuple[int, int], rect: dict) -> int:
    r, c = point
    r0, c0, r1, c1 = rect["bbox"]
    if r < r0:
        dr = r0 - r
    elif r > r1:
        dr = r - r1
    else:
        dr = 0
    if c < c0:
        dc = c0 - c
    elif c > c1:
        dc = c - c1
    else:
        dc = 0
    return int(dr + dc)


def _has_nearest_mismatch(rects: list[dict], instructions: dict[int, dict]) -> bool:
    for rect in rects:
        distances = sorted(
            (_distance_to_rect(instr["center"], rect), int(count))
            for count, instr in instructions.items()
        )
        if distances and distances[0][1] != int(rect["count"]):
            return True
    return False


def _has_long_extension(rects: list[dict], instructions: dict[int, dict]) -> bool:
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        instr = instructions[int(rect["count"])]
        for mr, mc in rect["markers"]:
            for dr, dc in instr["dirs"]:
                if dr == -1:
                    dist = mr - r0
                elif dr == 1:
                    dist = r1 - mr
                elif dc == -1:
                    dist = mc - c0
                else:
                    dist = c1 - mc
                if int(dist) >= 2:
                    return True
    return False


def _has_color_lure(rects: list[dict], instructions: dict[int, dict]) -> bool:
    for rect in rects:
        for count, instr in instructions.items():
            if int(count) != int(rect["count"]) and int(instr["arm_color"]) == int(rect["marker_color"]):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    counts = _choose_counts()
    if len(counts) == 4:
        rows = random.randint(26, 30)
        cols = random.randint(26, 30)
    elif len(counts) == 3:
        rows = random.randint(22, 30)
        cols = random.randint(22, 30)
    else:
        rows = random.randint(18, 30)
        cols = random.randint(18, 30)

    bg, center_color, fill_colors, marker_colors, arm_colors = _choose_colors(counts)
    grid = np.full((rows, cols), int(bg), dtype=int)

    rects = _draw_rectangles(grid, counts, fill_colors, marker_colors)
    if rects is None:
        return None
    instructions = _draw_instructions(grid, bg, counts, rects, center_color, arm_colors)
    if instructions is None:
        return None

    if _get_bg(grid) != int(bg):
        return None
    if not _has_nearest_mismatch(rects, instructions):
        return None
    if not _has_color_lure(rects, instructions):
        return None
    if not _has_long_extension(rects, instructions):
        return None

    expected = _apply_rule(grid, bg, rects, instructions)
    if np.array_equal(expected, grid):
        return None
    for instr in instructions.values():
        for r, c in instr["cells"]:
            if int(expected[r, c]) != int(bg):
                return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid count-paired crosshair stamping grid")
