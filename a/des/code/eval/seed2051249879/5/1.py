from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
LEGEND_BORDER = 5
LEGEND_ARROW = 1
LEGEND_SIDE = 7
LEGEND_GAP = 2
GEN_RETRIES = 400

DIR_TO_QUARTERS = {
    (-1, 0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (0, -1): 3,
}

QUARTERS_TO_DIR = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_ring_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox_of(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(coords) == expected


def _find_ring_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in components(grid == color):
            if not _is_ring_component(comp):
                continue
            r0, c0, r1, c1 = bbox_of(comp)
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "side_h": r1 - r0 + 1,
                    "side_w": c1 - c0 + 1,
                }
            )
    return frames


def _majority_color(values: list[int]) -> int:
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in values:
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1
    best = order[0]
    for value in order[1:]:
        if counts[value] > counts[best]:
            best = value
    return best


def _rot_cw(pos: tuple[int, int], side: int) -> tuple[int, int]:
    r, c = pos
    return c, side - 1 - r


def _rotation_orbits(side: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    for r in range(side):
        for c in range(side):
            if (r, c) in seen:
                continue
            orbit: list[tuple[int, int]] = []
            cur = (r, c)
            for _ in range(4):
                if cur not in orbit:
                    orbit.append(cur)
                cur = _rot_cw(cur, side)
            for item in orbit:
                seen.add(item)
            out.append(orbit)
    return out


def _complete_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    side_h, side_w = patch.shape
    assert side_h == side_w
    side = side_h
    out = patch.copy()
    for r in range(side):
        for c in range(side):
            if int(patch[r, c]) != bg:
                continue
            cur = (r, c)
            candidates: list[int] = []
            for _ in range(3):
                cur = _rot_cw(cur, side)
                value = int(patch[cur[0], cur[1]])
                if value != bg:
                    candidates.append(value)
            if candidates:
                out[r, c] = _majority_color(candidates)
    return out


def _parse_legend(frame: dict, grid: np.ndarray, bg: int) -> tuple[int, int] | None:
    r0, c0, r1, c1 = frame["bbox"]
    h = frame["side_h"]
    w = frame["side_w"]
    if h != w or h % 2 == 0 or h < 5:
        return None

    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    arrow_cells = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in zip(*np.where(inner != bg))]
    if len(arrow_cells) != 3:
        return None
    if center not in arrow_cells:
        return None

    arrow_colors = {int(grid[r, c]) for r, c in arrow_cells}
    if len(arrow_colors) != 1:
        return None
    arrow_color = next(iter(arrow_colors))
    if arrow_color == frame["color"]:
        return None

    dir_candidates = []
    for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        nr = center[0] + direction[0]
        nc = center[1] + direction[1]
        if (nr, nc) in arrow_cells:
            dir_candidates.append(direction)
    if len(dir_candidates) != 1:
        return None
    direction = dir_candidates[0]

    if direction[0] == 0:
        perps = [(-1, 0), (1, 0)]
    else:
        perps = [(0, -1), (0, 1)]
    arm = (center[0] + direction[0], center[1] + direction[1])
    foots = {
        (arm[0] + perps[0][0], arm[1] + perps[0][1]),
        (arm[0] + perps[1][0], arm[1] + perps[1][1]),
    }
    extras = set(arrow_cells) - {center, arm}
    if len(extras) != 1 or next(iter(extras)) not in foots:
        return None

    outside_positions: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        outside_positions.extend([(r0 - 1, c), (r1 + 1, c)])
    for r in range(r0, r1 + 1):
        outside_positions.extend([(r, c0 - 1), (r, c1 + 1)])
    label_cells = []
    for rr, cc in outside_positions:
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            value = int(grid[rr, cc])
            if value not in (bg, frame["color"], arrow_color):
                label_cells.append((rr, cc, value))
    if len(label_cells) != 1:
        return None
    return label_cells[0][2], DIR_TO_QUARTERS[direction]


def _extract_legends(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int, int, int]]]:
    legends: dict[int, int] = {}
    legend_boxes: set[tuple[int, int, int, int]] = set()
    for frame in _find_ring_frames(grid, bg):
        parsed = _parse_legend(frame, grid, bg)
        if parsed is None:
            continue
        label_color, quarters = parsed
        legends[label_color] = quarters
        legend_boxes.add(frame["bbox"])
    return legends, legend_boxes


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legends, legend_boxes = _extract_legends(g, bg)
    out = g.copy()
    for frame in _find_ring_frames(g, bg):
        if frame["bbox"] in legend_boxes:
            continue
        if frame["color"] not in legends:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        if frame["side_h"] != frame["side_w"] or frame["side_h"] < 8:
            continue
        patch = g[r0 + 1 : r1, c0 + 1 : c1]
        completed = _complete_patch(patch, bg)
        rotated = np.rot90(completed, -legends[frame["color"]])
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct fragmentary patterns inside colored square frames using rotational counterparts,
    then rotate each completed patch according to the matching border legend.
    """
    return _transform(grid)


def _legend_arrow_cells(direction: tuple[int, int]) -> list[tuple[int, int]]:
    center = (3, 3)
    dr, dc = direction
    arm = (center[0] + dr, center[1] + dc)
    if direction == (-1, 0):
        foot = (arm[0], arm[1] + 1)
    elif direction == (0, 1):
        foot = (arm[0] - 1, arm[1])
    elif direction == (1, 0):
        foot = (arm[0], arm[1] - 1)
    else:
        foot = (arm[0] + 1, arm[1])
    return [center, arm, foot]


def _make_legend(direction: tuple[int, int]) -> np.ndarray:
    arr = np.full((LEGEND_SIDE, LEGEND_SIDE), BG, dtype=int)
    arr[0, :] = LEGEND_BORDER
    arr[-1, :] = LEGEND_BORDER
    arr[:, 0] = LEGEND_BORDER
    arr[:, -1] = LEGEND_BORDER
    for r, c in _legend_arrow_cells(direction):
        arr[r, c] = LEGEND_ARROW
    return arr


def _make_pattern(side: int, colors: list[int]) -> np.ndarray | None:
    orbits = _rotation_orbits(side)
    non_center = [orbit for orbit in orbits if len(orbit) == 4]
    if len(non_center) < 4:
        return None

    for _ in range(120):
        patch = np.full((side, side), BG, dtype=int)
        orbit_count = random.randint(max(4, len(orbits) // 3), max(5, (2 * len(orbits)) // 3))
        chosen = random.sample(orbits, orbit_count)
        partial_orbits = 0
        for orbit in chosen:
            if len(orbit) == 1:
                color = random.choice(colors)
                patch[orbit[0]] = color
                continue

            keep = random.choices([1, 2, 3, 4], weights=[2, 4, 4, 1], k=1)[0]
            if keep < 4:
                partial_orbits += 1
            if keep == 1:
                visible_colors = [random.choice(colors)]
            elif keep == 2:
                color = random.choice(colors)
                visible_colors = [color, color]
            elif keep == 3:
                major = random.choice(colors)
                if len(colors) >= 2 and random.random() < 0.75:
                    minor_choices = [c for c in colors if c != major]
                    visible_colors = [major, major, random.choice(minor_choices)]
                    random.shuffle(visible_colors)
                else:
                    visible_colors = [major, major, major]
            else:
                visible_colors = [random.choice(colors) for _ in range(4)]
                if len(set(visible_colors)) == 1 and len(colors) >= 2:
                    alt_choices = [c for c in colors if c != visible_colors[0]]
                    visible_colors[random.randrange(4)] = random.choice(alt_choices)

            slots = random.sample(range(4), keep)
            for idx, slot in enumerate(slots):
                rr, cc = orbit[slot]
                patch[rr, cc] = visible_colors[idx]

        completed = _complete_patch(patch, BG)
        if partial_orbits == 0:
            continue
        if not np.any((patch == BG) & (completed != BG)):
            continue
        if int(np.count_nonzero(completed)) < side * side // 4:
            continue
        if np.array_equal(completed, patch):
            continue
        if any(np.array_equal(completed, np.rot90(completed, k)) for k in (1, 2, 3)):
            continue
        return patch
    return None


def _placeable(occupied: np.ndarray, top: int, left: int, side: int) -> bool:
    h, w = occupied.shape
    if top < 1 or left < 1 or top + side > h - 1 or left + side > w - 1:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + side + 1)
    c1 = min(w, left + side + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _reserve(occupied: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    side_h, side_w = patch.shape
    occupied[top : top + side_h, left : left + side_w] = True


def _place_legends(grid: np.ndarray, occupied: np.ndarray, border_colors: list[int]) -> list[int]:
    n = len(border_colors)
    h, w = grid.shape
    group_width = n * LEGEND_SIDE + (n - 1) * LEGEND_GAP
    if group_width + 3 > w:
        raise ValueError("legend strip does not fit")

    corner = random.choice(["tl", "tr", "bl", "br"])
    top = 1 if corner in {"tl", "tr"} else h - LEGEND_SIDE - 2
    starts: list[int] = []
    if corner in {"tl", "bl"}:
        base = 2
        for i in range(n):
            starts.append(base + i * (LEGEND_SIDE + LEGEND_GAP))
        label_dc = -1
    else:
        base = w - LEGEND_SIDE - 2
        for i in range(n):
            starts.append(base - i * (LEGEND_SIDE + LEGEND_GAP))
        label_dc = LEGEND_SIDE

    rotations: list[int] = []
    for color, left in zip(border_colors, starts):
        quarters = random.randrange(4)
        rotations.append(quarters)
        patch = _make_legend(QUARTERS_TO_DIR[quarters])
        _reserve(occupied, top, left, patch)
        grid[top : top + LEGEND_SIDE, left : left + LEGEND_SIDE] = patch
        label_r = top + LEGEND_SIDE // 2
        label_c = left + label_dc
        grid[label_r, label_c] = color
        occupied[label_r, label_c] = True
    return rotations


def _place_regions(grid: np.ndarray, occupied: np.ndarray, border_colors: list[int], pattern_colors: list[int]) -> bool:
    h, w = grid.shape
    sides = [random.randint(8, 11) for _ in border_colors]
    random.shuffle(sides)
    for color, side in sorted(zip(border_colors, sides), key=lambda item: -item[1]):
        patch_colors = random.sample(pattern_colors, random.randint(2, min(4, len(pattern_colors))))
        inside = _make_pattern(side - 2, patch_colors)
        if inside is None:
            return False
        placed = False
        for _ in range(250):
            top = random.randint(1, h - side - 1)
            left = random.randint(1, w - side - 1)
            if not _placeable(occupied, top, left, side):
                continue
            grid[top, left : left + side] = color
            grid[top + side - 1, left : left + side] = color
            grid[top : top + side, left] = color
            grid[top : top + side, left + side - 1] = color
            grid[top + 1 : top + side - 1, left + 1 : left + side - 1] = inside
            occupied[top : top + side, left : left + side] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choice([2, 3])
    h = random.randint(26, 34)
    w = random.randint(30, 38)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    border_pool = [2, 3, 4, 6, 7, 8, 9]
    border_colors = random.sample(border_pool, n_regions)
    pattern_pool = [c for c in [2, 3, 4, 6, 7, 8, 9] if c not in border_colors]
    if len(pattern_pool) < 2:
        return None

    try:
        _place_legends(grid, occupied, border_colors)
    except ValueError:
        return None
    if not _place_regions(grid, occupied, border_colors, pattern_pool):
        return None

    try:
        transformed = main(grid)
    except Exception:
        return None
    if transformed.shape != grid.shape:
        return None
    if np.array_equal(transformed, grid):
        return None
    legends, legend_boxes = _extract_legends(grid, BG)
    if len(legends) != n_regions or len(legend_boxes) != n_regions:
        return None
    region_frames = [frame for frame in _find_ring_frames(grid, BG) if frame["bbox"] not in legend_boxes and frame["color"] in legends]
    if len(region_frames) != n_regions:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after retries")
