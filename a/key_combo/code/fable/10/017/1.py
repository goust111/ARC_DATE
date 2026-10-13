import random
from collections import Counter

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def turn_dir(direction: int, chirality: int) -> int:
    return (direction + chirality) % 4


def segment_color(direction: int, h_color: int, v_color: int) -> int:
    dr, _ = DIRS[direction]
    return h_color if dr == 0 else v_color


def build_spiral_segments(
    center: tuple[int, int], first_dir: int, chirality: int, num_segments: int
) -> list[list[tuple[int, int]]]:
    r, c = center
    direction = first_dir
    segments: list[list[tuple[int, int]]] = []
    for seg_idx in range(num_segments):
        dr, dc = DIRS[direction]
        length = seg_idx + 2
        cells: list[tuple[int, int]] = []
        for _ in range(length):
            r += dr
            c += dc
            cells.append((r, c))
        segments.append(cells)
        direction = turn_dir(direction, chirality)
    return segments


def infer_center_and_colors(grid: np.ndarray):
    bg = get_bg(grid)
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    counts = {color: int(np.sum(grid == color)) for color in colors}
    assert len(colors) == 3, "expected exactly three non-background colors"

    for center in np.argwhere(grid != bg):
        center_pos = tuple(map(int, center))
        center_color = int(grid[center_pos])
        if counts[center_color] != 1:
            continue

        residue_colors = sorted({int(grid[tuple(coord)]) for coord in np.argwhere(grid != bg) if tuple(map(int, coord)) != center_pos})
        if len(residue_colors) != 2 or center_color in residue_colors:
            continue

        r0, c0 = center_pos
        adjacent = []
        for direction, (dr, dc) in enumerate(DIRS):
            nr, nc = r0 + dr, c0 + dc
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != bg:
                adjacent.append((direction, int(grid[nr, nc])))
        if len(adjacent) != 1:
            continue

        first_dir, first_color = adjacent[0]
        other_color = residue_colors[0] if residue_colors[1] == first_color else residue_colors[1]
        if DIRS[first_dir][0] == 0:
            h_color, v_color = first_color, other_color
        else:
            h_color, v_color = other_color, first_color
        if h_color != v_color:
            return bg, center_pos, center_color, first_dir, h_color, v_color

    raise AssertionError("failed to identify a valid center marker")


def validate_candidate(
    grid: np.ndarray,
    center: tuple[int, int],
    first_dir: int,
    chirality: int,
    h_color: int,
    v_color: int,
    bg: int,
    center_color: int,
):
    evidence = {
        tuple(map(int, coord)): int(grid[tuple(coord)])
        for coord in np.argwhere((grid != bg) & (grid != center_color))
    }
    found: set[tuple[int, int]] = set()
    max_hit_segment = -1
    r, c = center
    direction = first_dir

    for seg_idx in range(64):
        dr, dc = DIRS[direction]
        color = segment_color(direction, h_color, v_color)
        seg_has_evidence = False
        length = seg_idx + 2
        for _ in range(length):
            nr, nc = r + dr, c + dc
            if not in_bounds(grid.shape, nr, nc):
                if seg_has_evidence or len(found) != len(evidence):
                    return None
                return {"chirality": chirality, "last_segment": max_hit_segment}
            r, c = nr, nc
            if (r, c) in evidence:
                if evidence[(r, c)] != color or (r, c) in found:
                    return None
                found.add((r, c))
                seg_has_evidence = True
                max_hit_segment = seg_idx
        if len(found) == len(evidence) and max_hit_segment == seg_idx:
            return {"chirality": chirality, "last_segment": max_hit_segment}
        direction = turn_dir(direction, chirality)
    return None


def draw_output(
    shape: tuple[int, int],
    bg: int,
    center: tuple[int, int],
    center_color: int,
    first_dir: int,
    chirality: int,
    h_color: int,
    v_color: int,
    last_segment: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    r, c = center
    out[r, c] = center_color
    direction = first_dir
    for seg_idx in range(last_segment + 1):
        dr, dc = DIRS[direction]
        color = segment_color(direction, h_color, v_color)
        for _ in range(seg_idx + 2):
            r += dr
            c += dc
            assert in_bounds(shape, r, c)
            out[r, c] = color
        direction = turn_dir(direction, chirality)
    return out


def solve_instance(grid: np.ndarray):
    bg, center, center_color, first_dir, h_color, v_color = infer_center_and_colors(grid)
    candidates = []
    for chirality in (-1, 1):
        candidate = validate_candidate(
            grid,
            center,
            first_dir,
            chirality,
            h_color,
            v_color,
            bg,
            center_color,
        )
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected a unique spiral handedness"
    candidate = candidates[0]
    return {
        "bg": bg,
        "center": center,
        "center_color": center_color,
        "first_dir": first_dir,
        "chirality": candidate["chirality"],
        "h_color": h_color,
        "v_color": v_color,
        "last_segment": candidate["last_segment"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the hidden square spiral from its sparse two-color residue around a lone center marker.

    Input structure:
    - The background is dominant.
    - Exactly one isolated center marker uses a unique color.
    - All other foreground pixels use exactly two colors and lie sparsely on one
      unseen square spiral whose segment lengths are 2, 3, 4, ... .
    - Horizontal spiral segments use one residue color, vertical segments use the
      other, and the first step away from the center is visibly occupied.

    Rule:
    - Use the residue adjacent to the center to determine the first segment
      direction.
    - Use all residue pixels to determine the unique turning handedness of the
      spiral.
    - Redraw every segment from the center through the complete segment that
      contains the farthest residue pixel, coloring horizontal segments with the
      horizontal residue color and vertical segments with the vertical residue
      color.
    - Keep the center marker unchanged.

    Key insight:
    - The scattered pixels are not independent dots; together they are samples of
      one invisible spiral, and the start direction, handedness, and stopping
      segment are all encoded by where those samples appear.
    """
    g = np.asarray(grid, dtype=int)
    params = solve_instance(g)
    return draw_output(
        g.shape,
        params["bg"],
        params["center"],
        params["center_color"],
        params["first_dir"],
        params["chirality"],
        params["h_color"],
        params["v_color"],
        params["last_segment"],
    )


def king_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def choose_index(
    segment: list[tuple[int, int]],
    selected: set[tuple[int, int]],
    center: tuple[int, int],
    rng: random.Random,
    *,
    must_be_first: bool = False,
    forbid_last: bool = False,
) -> int | None:
    if must_be_first:
        candidates = [0]
    else:
        interior = list(range(1, len(segment) - 1))
        endpoints = [0, len(segment) - 1]
        rng.shuffle(interior)
        rng.shuffle(endpoints)
        candidates = interior + endpoints
    if forbid_last:
        candidates = [idx for idx in candidates if idx != len(segment) - 1]
    for idx in candidates:
        cell = segment[idx]
        if cell != center and king_adjacent(cell, center) and idx != 0:
            continue
        if any(king_adjacent(cell, other) for other in selected):
            continue
        return idx
    return None


def sample_residues(
    segments: list[list[tuple[int, int]]],
    center: tuple[int, int],
    rng: random.Random,
):
    selected: set[tuple[int, int]] = set()
    chosen: dict[int, list[int]] = {}
    for seg_idx, segment in enumerate(segments):
        idx = choose_index(
            segment,
            selected,
            center,
            rng,
            must_be_first=(seg_idx == 0),
            forbid_last=(seg_idx == len(segments) - 1),
        )
        if idx is None:
            return None
        chosen.setdefault(seg_idx, []).append(idx)
        selected.add(segment[idx])

    extra_segments = [i for i, seg in enumerate(segments) if len(seg) >= 6 and i != len(segments) - 1]
    rng.shuffle(extra_segments)
    for seg_idx in extra_segments:
        if rng.random() < 0.45:
            segment = segments[seg_idx]
            used = set(chosen[seg_idx])
            candidates = list(range(1, len(segment) - 1))
            rng.shuffle(candidates)
            for idx in candidates:
                if idx in used:
                    continue
                if all(abs(idx - other) >= 2 for other in used):
                    cell = segment[idx]
                    if not any(king_adjacent(cell, other) for other in selected):
                        chosen[seg_idx].append(idx)
                        selected.add(cell)
                        break
    return chosen


def validate_generated_instance(
    grid: np.ndarray,
    expected_output: np.ndarray,
    center: tuple[int, int],
    segments: list[list[tuple[int, int]]],
    chosen: dict[int, list[int]],
    first_dir: int,
    chirality: int,
) -> bool:
    non_bg = np.argwhere(grid != get_bg(grid))
    cells = [tuple(map(int, coord)) for coord in non_bg if tuple(map(int, coord)) != center]
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            if king_adjacent(a, b):
                return False

    last_seg = len(segments) - 1
    if max(chosen[last_seg]) == len(segments[last_seg]) - 1:
        return False

    horizontal_interior = False
    vertical_interior = False
    for seg_idx, idxs in chosen.items():
        direction = (first_dir + chirality * seg_idx) % 4
        is_horizontal = DIRS[direction][0] == 0
        for idx in idxs:
            if 0 < idx < len(segments[seg_idx]) - 1:
                horizontal_interior |= is_horizontal
                vertical_interior |= not is_horizontal
    if not (horizontal_interior and vertical_interior):
        return False

    params = solve_instance(grid)
    solved = draw_output(
        grid.shape,
        params["bg"],
        params["center"],
        params["center_color"],
        params["first_dir"],
        params["chirality"],
        params["h_color"],
        params["v_color"],
        params["last_segment"],
    )
    if not np.array_equal(solved, expected_output):
        return False
    if params["chirality"] != chirality or params["last_segment"] != last_seg:
        return False

    alt = draw_output(
        grid.shape,
        params["bg"],
        params["center"],
        params["center_color"],
        params["first_dir"],
        params["chirality"],
        params["h_color"],
        params["v_color"],
        max(0, params["last_segment"] - 1),
    )
    if np.array_equal(alt, expected_output):
        return False
    return True


def generate_attempt(rng: random.Random) -> np.ndarray | None:
    num_segments = rng.randint(5, 9)
    first_dir = rng.randrange(4)
    chirality = rng.choice([-1, 1])

    relative_segments = build_spiral_segments((0, 0), first_dir, chirality, num_segments)
    coords = [(0, 0)] + [cell for segment in relative_segments for cell in segment]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    base_h = max_r - min_r + 1
    base_w = max_c - min_c + 1
    if base_h > 28 or base_w > 28:
        return None

    extra_h = rng.randint(2, min(10, 30 - base_h))
    extra_w = rng.randint(2, min(10, 30 - base_w))
    top = rng.randint(1, extra_h - 1)
    bottom = extra_h - top
    left = rng.randint(1, extra_w - 1)
    right = extra_w - left
    height = base_h + top + bottom
    width = base_w + left + right
    center = (-min_r + top, -min_c + left)
    segments = build_spiral_segments(center, first_dir, chirality, num_segments)

    chosen = sample_residues(segments, center, rng)
    if chosen is None:
        return None

    colors = rng.sample(list(range(1, 10)), 3)
    center_color, h_color, v_color = colors
    grid = np.zeros((height, width), dtype=int)
    grid[center] = center_color

    for seg_idx, idxs in chosen.items():
        direction = (first_dir + chirality * seg_idx) % 4
        color = segment_color(direction, h_color, v_color)
        for idx in idxs:
            cell = segments[seg_idx][idx]
            grid[cell] = color

    expected_output = draw_output(
        grid.shape,
        0,
        center,
        center_color,
        first_dir,
        chirality,
        h_color,
        v_color,
        num_segments - 1,
    )
    if not validate_generated_instance(
        grid,
        expected_output,
        center,
        segments,
        chosen,
        first_dir,
        chirality,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-residue instance")
