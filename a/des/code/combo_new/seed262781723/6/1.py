import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
FILL_COLORS = [1, 3, 4, 6, 7, 8, 9]

SIDE_ORDER = ("U", "R", "D", "L")
ROTATION_BY_SIDES = {
    frozenset({"U"}): 0,
    frozenset({"R"}): 1,
    frozenset({"D"}): 2,
    frozenset({"L"}): 3,
    frozenset({"U", "D"}): 2,
    frozenset({"L", "R"}): 2,
    frozenset({"U", "R"}): 1,
    frozenset({"R", "D"}): 2,
    frozenset({"D", "L"}): 3,
    frozenset({"L", "U"}): 0,
    frozenset({"U", "R", "D"}): 1,
    frozenset({"R", "D", "L"}): 2,
    frozenset({"D", "L", "U"}): 3,
    frozenset({"L", "U", "R"}): 0,
    frozenset({"U", "R", "D", "L"}): 0,
}

LEGEND_CORNERS = {
    "tl": (0, 0, 1, 1),
    "tr": (0, -1, 1, -1),
    "bl": (-1, 0, -1, 1),
    "br": (-1, -1, -1, -1),
}

BRANCH_PATTERNS = {
    "single": [("U",), ("R",), ("D",), ("L",)],
    "adjacent": [("U", "R"), ("R", "D"), ("D", "L"), ("L", "U")],
    "opposite": [("U", "D"), ("L", "R")],
    "triple": [
        ("U", "R", "D"),
        ("R", "D", "L"),
        ("D", "L", "U"),
        ("L", "U", "R"),
    ],
    "all": [("U", "R", "D", "L")],
}

PATTERNS = {
    3: [
        np.array(
            [
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
    ],
    5: [
        np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=bool,
        ),
    ],
}

GEN_RETRIES = 300
RNG = random.Random()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate and recolor each framed red motif using a corner legend and the frame's missing-edge pattern.

    Input structure:
    - The background is black.
    - A corner legend contains several rows; each row has 1 to 4 grey cells followed by one colored cell, defining a gap-count-to-color mapping.
    - Elsewhere, several odd-sized grey square frames have exactly zero or one missing border cell on each side and contain a red motif inside.

    Rule:
    - Read the corner legend as "number of missing sides -> fill color", then clear the whole legend area to black.
    - For each grey frame, detect which of the four sides contain the one-cell border gap and count those gapped sides.
    - Rotate the interior motif from its default upward orientation toward the gap distribution: single sides point directly, opposite pairs rotate 180 degrees, adjacent pairs round their diagonal clockwise to the nearest quarter-turn, three-sided sets point opposite the missing side, and perfectly balanced four-sided gaps keep the default orientation.
    - Repaint the rotated motif from red to the legend color chosen by the number of gapped sides; keep the grey frame and all remaining black cells unchanged.

    Key insight:
    - The puzzle combines two independent cues: the legend maps gap counts to colors, while the spatial arrangement of the gaps determines how far the interior motif must rotate.
    """
    g = np.asarray(grid, dtype=int)
    legend = _parse_legend(g)

    work = g.copy()
    lr0, lc0, lr1, lc1 = legend["bbox"]
    work[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG

    frames = _find_frames(work)
    out = g.copy()
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        patch = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        turns = _rotation_from_sides(frame["sides"])
        rotated = np.rot90(patch, -turns)
        fill_color = legend["mapping"][len(frame["sides"])]
        rotated[rotated == RED] = fill_color
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


def _parse_legend(grid: np.ndarray) -> dict:
    h, w = grid.shape
    best = None

    for corner, (r_seed, c_seed, dr, dc) in LEGEND_CORNERS.items():
        r0 = 0 if r_seed == 0 else h - 1
        c0 = 0 if c_seed == 0 else w - 1
        if int(grid[r0, c0]) != GREY:
            continue

        mapping = {}
        rows = []
        valid = True

        for row_idx in range(4):
            r = r0 + row_idx * 2 * dr
            if not (0 <= r < h):
                valid = False
                break
            if int(grid[r, c0]) != GREY:
                break

            c = c0
            run = 0
            while 0 <= c < w and int(grid[r, c]) == GREY:
                run += 1
                c += dc

            if not (1 <= run <= 4):
                valid = False
                break
            if not (0 <= c < w):
                valid = False
                break

            color = int(grid[r, c])
            if color in (BG, RED, GREY) or run in mapping or color in mapping.values():
                valid = False
                break

            next_c = c + dc
            if 0 <= next_c < w and int(grid[r, next_c]) != BG:
                valid = False
                break

            spacer_r = r + dr
            if row_idx < 3 and 0 <= spacer_r < h:
                c_end = c0 + dc * run
                lo = min(c0, c_end)
                hi = max(c0, c_end)
                if np.any(grid[spacer_r, lo : hi + 1] != BG):
                    valid = False
                    break

            mapping[run] = color
            rows.append((r, run))

        if not valid or not mapping:
            continue

        last_run = max(run for _, run in rows)
        if dc > 0:
            col_min = c0
            col_max = c0 + last_run
        else:
            col_min = c0 - last_run
            col_max = c0
        row_vals = [r for r, _ in rows]
        bbox = (min(row_vals), col_min, max(row_vals), col_max)
        score = (len(mapping), len(set(mapping)))

        candidate = {"corner": corner, "mapping": mapping, "bbox": bbox, "score": score}
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        raise AssertionError("legend not found")
    return best


def _find_frames(grid: np.ndarray) -> list[dict]:
    mask = grid == GREY
    frames = []

    for points in _components(mask):
        rows = [r for r, _ in points]
        cols = [c for _, c in points]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        size_h = r1 - r0 + 1
        size_w = c1 - c0 + 1

        if size_h != size_w or size_h < 5 or size_h % 2 == 0:
            continue

        point_set = set(points)
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in point_set):
            continue

        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if not corners.issubset(point_set):
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        non_bg = set(map(int, np.unique(interior))) - {BG}
        if not non_bg or not non_bg.issubset({RED}):
            continue

        sides = []
        valid = True
        side_specs = {
            "U": [(r0, c) for c in range(c0 + 1, c1)],
            "R": [(r, c1) for r in range(r0 + 1, r1)],
            "D": [(r1, c) for c in range(c0 + 1, c1)],
            "L": [(r, c0) for r in range(r0 + 1, r1)],
        }
        for side, cells in side_specs.items():
            missing = [(r, c) for r, c in cells if int(grid[r, c]) != GREY]
            if len(missing) > 1:
                valid = False
                break
            if len(missing) == 1:
                sides.append(side)

        if valid and sides:
            frames.append({"bbox": (r0, c0, r1, c1), "sides": tuple(sides)})

    if not frames:
        raise AssertionError("no frames found")
    return frames


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            points = []
            while q:
                rr, cc = q.popleft()
                points.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(points)

    return out


def _rotation_from_sides(sides: tuple[str, ...]) -> int:
    key = frozenset(sides)
    if key not in ROTATION_BY_SIDES:
        raise AssertionError(f"unsupported gap pattern: {sorted(key)}")
    return ROTATION_BY_SIDES[key]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)

    n_frames = rng.choice([3, 4])
    count2_branch = rng.choice(["adjacent", "opposite"])
    categories = ["single", count2_branch, "triple"]
    if n_frames == 4:
        categories.append("all")
    rng.shuffle(categories)

    frame_specs = []
    rotations = []
    for category in categories:
        sides = rng.choice(BRANCH_PATTERNS[category])
        size = rng.choice([5, 7])
        pattern = rng.choice(PATTERNS[size - 2]).copy()
        frame_specs.append({"sides": tuple(sides), "size": size, "pattern": pattern})
        rotations.append(_rotation_from_sides(tuple(sides)))

    if len(set(rotations)) < 2:
        return None

    used_counts = sorted({len(spec["sides"]) for spec in frame_specs})
    legend_counts = list(used_counts)
    remaining = [count for count in (1, 2, 3, 4) if count not in used_counts]
    if remaining and rng.random() < 0.5:
        legend_counts.append(rng.choice(remaining))
        legend_counts.sort()

    colors = rng.sample(FILL_COLORS, k=len(legend_counts))
    mapping = {count: color for count, color in zip(legend_counts, colors)}
    corner = rng.choice(list(LEGEND_CORNERS))
    legend_bbox = _draw_legend(grid, legend_counts, mapping, corner)

    reserved = [legend_bbox]
    placed = []
    for spec in frame_specs:
        rect = _place_frame_bbox(h, w, spec["size"], reserved, rng)
        if rect is None:
            return None
        reserved.append(rect)
        placed.append(rect)

    for rect, spec in zip(placed, frame_specs):
        _draw_frame(grid, rect, spec, rng)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or not np.any(out != grid):
        return None
    if np.any(out[legend_bbox[0] : legend_bbox[2] + 1, legend_bbox[1] : legend_bbox[3] + 1] != BG):
        return None

    return grid


def _draw_legend(
    grid: np.ndarray, legend_counts: list[int], mapping: dict[int, int], corner: str
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    r_seed, c_seed, dr, dc = LEGEND_CORNERS[corner]
    r0 = 0 if r_seed == 0 else h - 1
    c0 = 0 if c_seed == 0 else w - 1

    row_positions = []
    col_positions = []

    for row_idx, count in enumerate(legend_counts):
        r = r0 + row_idx * 2 * dr
        color_c = c0 + count * dc
        row_positions.append(r)
        col_positions.extend([c0, color_c])

        for offset in range(count):
            c = c0 + offset * dc
            grid[r, c] = GREY
        grid[r, color_c] = mapping[count]

    return (min(row_positions), min(col_positions), max(row_positions), max(col_positions))


def _place_frame_bbox(
    h: int,
    w: int,
    size: int,
    reserved: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    if h - size - 1 < 1 or w - size - 1 < 1:
        return None

    for _ in range(200):
        r0 = rng.randint(1, h - size - 1)
        c0 = rng.randint(1, w - size - 1)
        rect = (r0, c0, r0 + size - 1, c0 + size - 1)
        if all(_rect_gap(rect, other) >= 2 for other in reserved):
            return rect
    return None


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    if ar1 < br0:
        row_gap = br0 - ar1 - 1
    elif br1 < ar0:
        row_gap = ar0 - br1 - 1
    else:
        row_gap = 0

    if ac1 < bc0:
        col_gap = bc0 - ac1 - 1
    elif bc1 < ac0:
        col_gap = ac0 - bc1 - 1
    else:
        col_gap = 0

    if row_gap == 0 and col_gap == 0:
        return 0
    return max(row_gap, col_gap)


def _draw_frame(grid: np.ndarray, rect: tuple[int, int, int, int], spec: dict, rng: random.Random) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    size = spec["size"]
    if "U" in spec["sides"]:
        grid[r0, c0 + rng.randint(1, size - 2)] = BG
    if "R" in spec["sides"]:
        grid[r0 + rng.randint(1, size - 2), c1] = BG
    if "D" in spec["sides"]:
        grid[r1, c0 + rng.randint(1, size - 2)] = BG
    if "L" in spec["sides"]:
        grid[r0 + rng.randint(1, size - 2), c0] = BG

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[:, :] = BG
    interior[spec["pattern"]] = RED

