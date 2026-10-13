import random
from collections import Counter, deque

import numpy as np


SIDES = ("top", "right", "bottom", "left")
DIRS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
FRAME_SIZE_CHOICES = (5, 7, 9)
GEN_ATTEMPTS = 400


def _is_connected(points: tuple[tuple[int, int], ...]) -> bool:
    point_set = set(points)
    start = next(iter(point_set))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS.values():
            nxt = (r + dr, c + dc)
            if nxt in point_set and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(point_set)


def _build_pattern_shapes() -> list[tuple[tuple[int, int], ...]]:
    coords = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)]
    shapes: list[tuple[tuple[int, int], ...]] = []
    for mask in range(1, 1 << len(coords)):
        points = tuple(coords[i] for i in range(len(coords)) if (mask >> i) & 1)
        rows = [r for r, _ in points]
        cols = [c for _, c in points]
        if max(rows) - min(rows) + 1 not in (1, 3):
            continue
        if max(cols) - min(cols) + 1 not in (1, 3):
            continue
        if min(rows) + max(rows) != 0 or min(cols) + max(cols) != 0:
            continue
        if len(points) > 5:
            continue
        if not _is_connected(points):
            continue
        shapes.append(tuple(sorted(points)))
    shapes.sort(key=lambda pts: (len(pts), pts))
    return shapes


PATTERN_SHAPES = _build_pattern_shapes()


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    if np.any(values == 0):
        return 0
    return int(values[int(np.argmax(counts))])


def _gap_position(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    if side == "top":
        return r0, cc
    if side == "right":
        return cr, c1
    if side == "bottom":
        return r1, cc
    return cr, c0


def _frame_border_cells(
    bbox: tuple[int, int, int, int], gaps: set[str] | None = None
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gaps = gaps or set()
    gap_cells = {_gap_position(bbox, side) for side in gaps}
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        if (r0, c) not in gap_cells:
            cells.append((r0, c))
        if (r1, c) not in gap_cells and r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        if (r, c0) not in gap_cells:
            cells.append((r, c0))
        if (r, c1) not in gap_cells and c1 != c0:
            cells.append((r, c1))
    return cells


def _validate_frame(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int, bg: int
) -> dict | None:
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return None

    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    gaps: list[str] = []

    for c in range(c0, c1 + 1):
        top = int(grid[r0, c])
        bottom = int(grid[r1, c])
        if c == cc:
            if top == bg:
                gaps.append("top")
            elif top != color:
                return None
            if bottom == bg:
                gaps.append("bottom")
            elif bottom != color:
                return None
        else:
            if top != color or bottom != color:
                return None

    for r in range(r0 + 1, r1):
        left = int(grid[r, c0])
        right = int(grid[r, c1])
        if r == cr:
            if left == bg:
                gaps.append("left")
            elif left != color:
                return None
            if right == bg:
                gaps.append("right")
            elif right != color:
                return None
        else:
            if left != color or right != color:
                return None

    if not (1 <= len(gaps) <= 3):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or np.any(inner == color):
        return None
    points = np.argwhere(inner != bg)
    if points.size == 0:
        return None

    pr0, pc0 = points.min(axis=0)
    pr1, pc1 = points.max(axis=0)
    if int(pr1 - pr0 + 1) > 3 or int(pc1 - pc0 + 1) > 3:
        return None

    pr0 += r0 + 1
    pr1 += r0 + 1
    pc0 += c0 + 1
    pc1 += c0 + 1
    if int(pr0 + pr1) != int(r0 + r1) or int(pc0 + pc1) != int(c0 + c1):
        return None

    pattern: list[tuple[int, int, int]] = []
    for rr, cc2 in points:
        gr = int(r0 + 1 + rr)
        gc = int(c0 + 1 + cc2)
        pattern.append((gr - cr, gc - cc, int(grid[gr, gc])))
    pattern.sort()

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(color),
        "pattern": pattern,
        "gaps": tuple(gaps),
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    bg = _bg_color(grid)
    height, width = grid.shape
    frames: dict[tuple[int, int, int, int], dict] = {}
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        mask = grid == int(color)
        if not np.any(mask):
            continue
        row_hits = [np.flatnonzero(mask[r]) for r in range(height)]
        col_hits = [np.flatnonzero(mask[:, c]) for c in range(width)]
        points = np.argwhere(mask)
        top_lefts: list[tuple[int, int]] = []
        bottom_rights: list[tuple[int, int]] = []
        for r_raw, c_raw in points:
            r = int(r_raw)
            c = int(c_raw)
            row = row_hits[r]
            col = col_hits[c]
            if row.size and col.size and int(row[-1]) > c and int(col[-1]) > r:
                top_lefts.append((r, c))
            if row.size and col.size and int(row[0]) < c and int(col[0]) < r:
                bottom_rights.append((r, c))

        for r0, c0 in top_lefts:
            for r1, c1 in bottom_rights:
                if r1 <= r0 + 1 or c1 <= c0 + 1:
                    continue
                if (r1 - r0) % 2 != 0 or (c1 - c0) % 2 != 0:
                    continue
                bbox = (int(r0), int(c0), int(r1), int(c1))
                if bbox in frames:
                    continue
                info = _validate_frame(grid, r0, c0, r1, c1, color, bg)
                if info is not None:
                    frames[bbox] = info

    return [frames[bbox] for bbox in sorted(frames)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each frame's centered inner pattern outward through the frame-edge gaps.

    Input structure:
    - The background is black.
    - The grid contains one-pixel-thick rectangular frames, each drawn in a single non-black color.
    - Every frame encloses a small centered pattern made of other colors, and 1-3 frame-edge midpoint cells are missing as gaps.

    Rule:
    - Detect each rectangular frame, its midpoint gaps, and the interior pattern pixels relative to the frame center.
    - For every gap, copy the interior pattern outward in the gap's facing direction.
    - The copied pattern keeps the same center-relative offsets, and its center is placed at the gap translated by the frame side length in that direction.
    - Keep the original frame, pattern, and gaps unchanged; discard copied pixels that fall outside the grid.

    Key insight:
    - A missing midpoint on a frame edge is an emission port, and the travel distance is not constant: it equals that frame's side length for the gapped edge.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = g.shape

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        for side in frame["gaps"]:
            gap_r, gap_c = _gap_position(frame["bbox"], side)
            dist = w if side in ("top", "bottom") else h
            dr, dc = DIRS[side]
            anchor_r = int(gap_r + dr * dist)
            anchor_c = int(gap_c + dc * dist)
            for off_r, off_c, color in frame["pattern"]:
                rr = int(anchor_r + off_r)
                cc = int(anchor_c + off_c)
                if 0 <= rr < height and 0 <= cc < width:
                    out[rr, cc] = int(color)
    return out


def _sample_pattern(pattern_pool: list[int]) -> list[tuple[int, int, int]]:
    shape = random.choice(PATTERN_SHAPES)
    n_colors = random.randint(1, min(3, len(shape), len(pattern_pool)))
    palette = random.sample(pattern_pool, n_colors)
    assignments = list(palette)
    while len(assignments) < len(shape):
        assignments.append(random.choice(palette))
    random.shuffle(assignments)
    return [
        (int(dr), int(dc), int(color))
        for (dr, dc), color in zip(shape, assignments)
    ]


def _sample_spec(frame_color: int, pattern_pool: list[int]) -> dict:
    gap_count = random.randint(1, 3)
    return {
        "frame_color": int(frame_color),
        "height": int(random.choice(FRAME_SIZE_CHOICES)),
        "width": int(random.choice(FRAME_SIZE_CHOICES)),
        "gaps": tuple(sorted(random.sample(SIDES, gap_count), key=SIDES.index)),
        "pattern": _sample_pattern(pattern_pool),
    }


def _render_input(
    height: int, width: int, placements: list[tuple[dict, int, int]]
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec, r0, c0 in placements:
        bbox = (int(r0), int(c0), int(r0 + spec["height"] - 1), int(c0 + spec["width"] - 1))
        for rr, cc in _frame_border_cells(bbox, set(spec["gaps"])):
            grid[rr, cc] = int(spec["frame_color"])
        cr = (bbox[0] + bbox[2]) // 2
        cc_mid = (bbox[1] + bbox[3]) // 2
        for off_r, off_c, color in spec["pattern"]:
            grid[int(cr + off_r), int(cc_mid + off_c)] = int(color)
    return grid


def _render_output(
    height: int, width: int, placements: list[tuple[dict, int, int]]
) -> np.ndarray:
    out = _render_input(height, width, placements)
    for spec, r0, c0 in placements:
        bbox = (int(r0), int(c0), int(r0 + spec["height"] - 1), int(c0 + spec["width"] - 1))
        h = int(spec["height"])
        w = int(spec["width"])
        for side in spec["gaps"]:
            gap_r, gap_c = _gap_position(bbox, side)
            dist = w if side in ("top", "bottom") else h
            dr, dc = DIRS[side]
            anchor_r = int(gap_r + dr * dist)
            anchor_c = int(gap_c + dc * dist)
            for off_r, off_c, color in spec["pattern"]:
                rr = int(anchor_r + off_r)
                cc = int(anchor_c + off_c)
                if 0 <= rr < height and 0 <= cc < width:
                    out[rr, cc] = int(color)
    return out


def _visible_output_cells(
    spec: dict, r0: int, c0: int, height: int, width: int
) -> tuple[set[tuple[int, int]], bool, bool]:
    bbox = (int(r0), int(c0), int(r0 + spec["height"] - 1), int(c0 + spec["width"] - 1))
    cells: set[tuple[int, int]] = set(_frame_border_cells(bbox, set(spec["gaps"])))
    cr = (bbox[0] + bbox[2]) // 2
    cc_mid = (bbox[1] + bbox[3]) // 2
    for off_r, off_c, _color in spec["pattern"]:
        cells.add((int(cr + off_r), int(cc_mid + off_c)))

    clipped = False
    all_visible = True
    for side in spec["gaps"]:
        gap_r, gap_c = _gap_position(bbox, side)
        dist = int(spec["width"] if side in ("top", "bottom") else spec["height"])
        dr, dc = DIRS[side]
        anchor_r = int(gap_r + dr * dist)
        anchor_c = int(gap_c + dc * dist)
        visible_here = 0
        for off_r, off_c, _color in spec["pattern"]:
            rr = int(anchor_r + off_r)
            cc = int(anchor_c + off_c)
            if 0 <= rr < height and 0 <= cc < width:
                visible_here += 1
                cells.add((rr, cc))
            else:
                clipped = True
        if visible_here == 0:
            all_visible = False
    return cells, clipped, all_visible


def _expand_cells(
    cells: set[tuple[int, int]], height: int, width: int, pad: int = 1
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(height - 1, r + pad) + 1):
            for cc in range(max(0, c - pad), min(width - 1, c + pad) + 1):
                expanded.add((int(rr), int(cc)))
    return expanded


def _place_specs(height: int, width: int, specs: list[dict]) -> list[tuple[dict, int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[dict, int, int]] = []
    clip_target = random.randrange(len(specs)) if random.random() < 0.45 else None

    for idx, spec in enumerate(sorted(specs, key=lambda s: s["height"] * s["width"], reverse=True)):
        candidates: list[tuple[int, int, set[tuple[int, int]], bool]] = []
        for r0 in range(0, height - spec["height"] + 1):
            for c0 in range(0, width - spec["width"] + 1):
                cells, clipped, all_visible = _visible_output_cells(spec, r0, c0, height, width)
                if not all_visible:
                    continue
                halo = _expand_cells(cells, height, width, pad=1)
                if any(occupied[r, c] for r, c in halo):
                    continue
                candidates.append((int(r0), int(c0), halo, clipped))
        if not candidates:
            return None

        if clip_target == idx:
            clipped_candidates = [cand for cand in candidates if cand[3]]
            if clipped_candidates:
                chosen = random.choice(clipped_candidates)
            else:
                chosen = random.choice(candidates)
        else:
            chosen = random.choice(candidates)

        r0, c0, halo, _clipped = chosen
        for rr, cc in halo:
            occupied[rr, cc] = True
        placements.append((spec, int(r0), int(c0)))

    return placements


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 1])[0]
    base_min = min(30, 12 + 4 * n_objects)
    if base_min > 30:
        return None
    height = random.randint(base_min, 30)
    width = random.randint(base_min, 30)

    palette = list(range(1, 10))
    if n_objects >= len(palette):
        return None
    frame_colors = random.sample(palette, n_objects)
    pattern_pool = [c for c in palette if c not in frame_colors]
    if not pattern_pool:
        return None

    specs = [_sample_spec(color, pattern_pool) for color in frame_colors]
    placements = _place_specs(height, width, specs)
    if placements is None:
        return None

    inp = _render_input(height, width, placements)
    expected = _render_output(height, width, placements)

    if len(_find_frames(inp)) != len(specs):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
