import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIZE_OPTIONS = (5, 7)
GAP_RANGE = (2, 3)
PAD_MAX = 2
GEN_RETRIES = 300

DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

RING_EDGES = [
    ((0, 0), "right", (0, 1)),
    ((0, 1), "right", (0, 2)),
    ((0, 2), "down", (1, 2)),
    ((1, 2), "down", (2, 2)),
    ((2, 2), "left", (2, 1)),
    ((2, 1), "left", (2, 0)),
    ((2, 0), "up", (1, 0)),
    ((1, 0), "up", (0, 0)),
]

CENTER_TARGETS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0 + 1, r1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _is_rect_frame(coords: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int] | None]:
    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False, None
    expected = _frame_border(bbox)
    actual = set(coords)
    if actual != expected:
        return False, None
    return True, bbox


def _seed_direction(
    seed: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    r, c = seed
    r0, c0, r1, c1 = bbox
    dists = {
        "up": r - r0,
        "down": r1 - r,
        "left": c - c0,
        "right": c1 - c,
    }
    best = min(dists.values())
    winners = [name for name, dist in dists.items() if dist == best]
    assert len(winners) == 1, f"seed at {seed} must have a unique nearest side"
    side = winners[0]
    if side == "up":
        opening = (r0, c)
    elif side == "down":
        opening = (r1, c)
    elif side == "left":
        opening = (r, c0)
    else:
        opening = (r, c1)
    return side, opening, DIR_VECS[side]


def _on_border(pt: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = pt
    r0, c0, r1, c1 = bbox
    return (r == r0 or r == r1 or c == c0 or c == c1) and r0 <= r <= r1 and c0 <= c <= c1


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {BG}):
        for comp in _components(grid == color):
            ok, bbox = _is_rect_frame(comp)
            if ok:
                frames.append({"color": color, "bbox": bbox})
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open frame borders according to enclosed seed pixels, then shoot matching-color rays outward toward other frames.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular frames of distinct non-black colors are scattered in the grid.
    - Each frame encloses 1-3 singleton seed pixels on black interior cells, and each seed color matches the border color of some other frame.

    Rule:
    - Detect every rectangular border component and read the non-black interior pixels as seeds for that frame.
    - For each seed, find its unique nearest side of the enclosing frame, open the aligned border pixel on that side, and cast a straight ray outward in the same direction.
    - Paint the ray with the seed color through black cells until it reaches the grid edge or the first non-black obstacle; if that obstacle is the matching target frame border, stop there and open that hit border pixel too.

    Key insight:
    - A seed pixel simultaneously encodes which other frame to connect to by color and which side to exit from by its offset inside the enclosing rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    frames = _find_frames(g)
    assert len(frames) >= 2, "expected multiple rectangular frames"
    frame_by_color: dict[int, dict] = {}
    for frame in frames:
        color = int(frame["color"])
        assert color not in frame_by_color, "frame border colors must be unique"
        frame_by_color[color] = frame

    ops: list[dict] = []
    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = [
            (r0 + 1 + rr, c0 + 1 + cc)
            for rr, cc in np.argwhere(interior != BG)
        ]
        assert 1 <= len(seeds) <= 3, f"frame {frame['bbox']} must contain 1-3 seeds"
        seen_colors = set()
        for seed in seeds:
            seed_color = int(g[seed])
            assert seed_color in frame_by_color, f"seed color {seed_color} must match another frame"
            assert seed_color != int(frame["color"]), "seed cannot target its own frame color"
            assert seed_color not in seen_colors, "seed colors inside one frame must be distinct"
            seen_colors.add(seed_color)
            _side, opening, step = _seed_direction(seed, frame["bbox"])
            ops.append(
                {
                    "opening": opening,
                    "step": step,
                    "color": seed_color,
                    "target_bbox": frame_by_color[seed_color]["bbox"],
                }
            )
            out[opening] = BG

    for op in ops:
        rr = op["opening"][0] + op["step"][0]
        cc = op["opening"][1] + op["step"][1]
        while 0 <= rr < h and 0 <= cc < w:
            value = int(g[rr, cc])
            if value == BG:
                out[rr, cc] = int(op["color"])
                rr += op["step"][0]
                cc += op["step"][1]
                continue
            if value == int(op["color"]) and _on_border((rr, cc), op["target_bbox"]):
                out[rr, cc] = BG
            break

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    for r, c in _frame_border(bbox):
        grid[r, c] = color


def _seed_pos(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "up":
        return r0 + 1, mid_c
    if direction == "down":
        return r1 - 1, mid_c
    if direction == "left":
        return mid_r, c0 + 1
    return mid_r, c1 - 1


def _gen_attempt(rng: random.Random) -> np.ndarray:
    row_heights = [rng.choice(SIZE_OPTIONS) for _ in range(3)]
    col_widths = [rng.choice(SIZE_OPTIONS) for _ in range(3)]
    row_gaps = [rng.randint(*GAP_RANGE) for _ in range(2)]
    col_gaps = [rng.randint(*GAP_RANGE) for _ in range(2)]

    base_h = sum(row_heights) + sum(row_gaps)
    base_w = sum(col_widths) + sum(col_gaps)
    if base_h > 30 or base_w > 30:
        return None

    extra_h = min(PAD_MAX * 2, 30 - base_h)
    extra_w = min(PAD_MAX * 2, 30 - base_w)
    top_pad = rng.randint(0, min(PAD_MAX, extra_h))
    bottom_pad = rng.randint(0, min(PAD_MAX, extra_h - top_pad))
    left_pad = rng.randint(0, min(PAD_MAX, extra_w))
    right_pad = rng.randint(0, min(PAD_MAX, extra_w - left_pad))

    h = base_h + top_pad + bottom_pad
    w = base_w + left_pad + right_pad
    if h > 30 or w > 30:
        return None

    row_starts = [top_pad]
    row_starts.append(row_starts[0] + row_heights[0] + row_gaps[0])
    row_starts.append(row_starts[1] + row_heights[1] + row_gaps[1])
    col_starts = [left_pad]
    col_starts.append(col_starts[0] + col_widths[0] + col_gaps[0])
    col_starts.append(col_starts[1] + col_widths[1] + col_gaps[1])

    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i in range(3):
        for j in range(3):
            r0 = row_starts[i]
            c0 = col_starts[j]
            bboxes[(i, j)] = (r0, c0, r0 + row_heights[i] - 1, c0 + col_widths[j] - 1)

    grid = np.zeros((h, w), dtype=int)
    frame_colors = {}
    for pos, color in zip(sorted(bboxes), rng.sample(COLORS, 9)):
        frame_colors[pos] = color
        _draw_frame(grid, bboxes[pos], color)

    used_seed_cells: set[tuple[int, int]] = set()
    for src, direction, dst in RING_EDGES:
        seed = _seed_pos(bboxes[src], direction)
        if seed in used_seed_cells:
            return None
        used_seed_cells.add(seed)
        grid[seed] = frame_colors[dst]

    center_dirs = rng.sample(list(CENTER_TARGETS), k=rng.randint(1, 3))
    for direction in center_dirs:
        seed = _seed_pos(bboxes[(1, 1)], direction)
        if seed in used_seed_cells:
            return None
        used_seed_cells.add(seed)
        grid[seed] = frame_colors[CENTER_TARGETS[direction]]

    solved = main(grid)
    assert solved.shape == grid.shape
    assert not np.array_equal(solved, grid), "generator must create a nontrivial instance"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
