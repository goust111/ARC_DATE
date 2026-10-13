from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 400
PLACE_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cells.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        cells.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cells.append((r, c0))
    return cells


def _noncorner_perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r, c in _perimeter_cells(r0, c0, r1, c1) if r not in (r0, r1) or c not in (c0, c1)]


def _layer_cells(r0: int, c0: int, r1: int, c1: int, depth: int, side: str) -> tuple[list[tuple[int, int]], bool]:
    top = r0 + depth
    left = c0 + depth
    bottom = r1 - depth
    right = c1 - depth
    if top > bottom or left > right:
        return [], True

    if top == bottom and left == right:
        return [(top, left)], True
    if top == bottom:
        return [(top, c) for c in range(left, right + 1)], True
    if left == right:
        return [(r, left) for r in range(top, bottom + 1)], True

    if side == "top":
        cells = [(top, c) for c in range(left, right + 1)]
        cells += [(r, right) for r in range(top + 1, bottom + 1)]
        cells += [(bottom, c) for c in range(right - 1, left - 1, -1)]
        cells += [(r, left) for r in range(bottom - 1, top, -1)]
        return cells, False
    if side == "right":
        cells = [(r, right) for r in range(top, bottom + 1)]
        cells += [(bottom, c) for c in range(right - 1, left - 1, -1)]
        cells += [(r, left) for r in range(bottom - 1, top - 1, -1)]
        cells += [(top, c) for c in range(left + 1, right)]
        return cells, False
    if side == "bottom":
        cells = [(bottom, c) for c in range(right, left - 1, -1)]
        cells += [(r, left) for r in range(bottom - 1, top - 1, -1)]
        cells += [(top, c) for c in range(left + 1, right + 1)]
        cells += [(r, right) for r in range(top + 1, bottom)]
        return cells, False

    cells = [(r, left) for r in range(bottom, top - 1, -1)]
    cells += [(top, c) for c in range(left + 1, right + 1)]
    cells += [(r, right) for r in range(top + 1, bottom + 1)]
    cells += [(bottom, c) for c in range(right - 1, left, -1)]
    return cells, False


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        pts = {(int(r), int(c)) for r, c in coords}
        perim = 2 * (h + w) - 4
        if len(pts) < perim - 4:
            continue
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
            continue
        frames.append(
            {
                "frame_color": color,
                "bbox": (r0, c0, r1, c1),
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_seed_band(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> tuple[int, str]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.size > 0, "frame must have an interior"
    pts = np.argwhere(inner != bg)
    assert pts.size > 0, "frame must contain a seed band"
    seed_colors = {int(inner[r, c]) for r, c in pts}
    assert len(seed_colors) == 1, "interior non-bg pixels must belong to one seed band"
    seed_color = seed_colors.pop()

    abs_pts = [(int(r0 + 1 + r), int(c0 + 1 + c)) for r, c in pts]
    rows = sorted({r for r, _ in abs_pts})
    cols = sorted({c for _, c in abs_pts})

    sides = []
    if len(rows) == 1 and cols == list(range(c0 + 1, c1)):
        if rows[0] == r0 + 1:
            sides.append("top")
        if rows[0] == r1 - 1:
            sides.append("bottom")
    if len(cols) == 1 and rows == list(range(r0 + 1, r1)):
        if cols[0] == c0 + 1:
            sides.append("left")
        if cols[0] == c1 - 1:
            sides.append("right")

    assert sides, "seed band must fill one full inner side of the frame"
    return seed_color, sides[0]


def _paint_frame(out: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int, seed_color: int, side: str, rings: int) -> None:
    r0, c0, r1, c1 = bbox
    for r, c in _perimeter_cells(r0, c0, r1, c1):
        out[r, c] = frame_color

    for depth in range(1, int(rings) + 1):
        cells, terminal = _layer_cells(r0, c0, r1, c1, depth, side)
        for r, c in cells:
            out[r, c] = seed_color
        if terminal:
            break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each inner seed band into a clockwise spiral whose depth is encoded by border gaps.

    Input structure:
    - bg = most common color.
    - Several non-overlapping one-pixel-thick rectangular frames, each with its own border color.
    - Inside each frame, one full inner side is occupied by a monochrome seed band of a different color.
    - Each frame border has 1 to 4 missing border pixels replaced by bg.

    Rule:
    - For each frame, count how many border positions are missing; that gap count is the spiral depth N.
    - Repair the border gaps back to the frame color.
    - Starting from the seed band, paint clockwise inner-wall layers of the seed color for N rings, moving one cell inward after each completed ring.
    - If the rectangle becomes too small for another full ring, paint the remaining reachable center line or point and stop.

    Key insight:
    - The border gaps are not damage; they encode how many inward spiral layers the seed color should occupy after the frame is repaired.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frames = _frame_candidates(g, bg)
    assert frames, "expected at least one rectangular frame"

    for frame in frames:
        bbox = frame["bbox"]
        frame_color = int(frame["frame_color"])
        seed_color, side = _find_seed_band(g, bg, bbox)
        gaps = sum(int(g[r, c]) != frame_color for r, c in _perimeter_cells(*bbox))
        _paint_frame(out, bbox, frame_color, seed_color, side, gaps)

    return out


def _draw_input(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        frame_color = int(spec["frame_color"])
        seed_color = int(spec["seed_color"])
        side = spec["side"]

        for r, c in _perimeter_cells(r0, c0, r1, c1):
            grid[r, c] = frame_color
        for r, c in spec["gaps_pos"]:
            grid[r, c] = bg

        if side == "top":
            grid[r0 + 1, c0 + 1 : c1] = seed_color
        elif side == "bottom":
            grid[r1 - 1, c0 + 1 : c1] = seed_color
        elif side == "left":
            grid[r0 + 1 : r1, c0 + 1] = seed_color
        else:
            grid[r0 + 1 : r1, c1 - 1] = seed_color

    return grid


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        _paint_frame(
            out,
            spec["bbox"],
            int(spec["frame_color"]),
            int(spec["seed_color"]),
            spec["side"],
            int(spec["gap_count"]),
        )
    return out


def _sample_inner_size(gap_count: int) -> tuple[int, int]:
    mode = "full"
    if gap_count >= 2 and random.random() < 0.5:
        mode = "tight"

    if mode == "full":
        lo = max(3, 2 * gap_count)
        inner_h = random.randint(lo, min(10, max(lo, 10)))
        inner_w = random.randint(lo, min(10, max(lo, 10)))
        return inner_h, inner_w

    tight_dim = random.randint(3, 2 * gap_count - 1)
    other_dim = random.randint(4, 10)
    if random.random() < 0.5:
        return tight_dim, other_dim
    return other_dim, tight_dim


def _sample_specs(shape: tuple[int, int], bg: int) -> list[dict] | None:
    h, w = shape
    n_frames = random.choice([1, 2, 2, 3])
    if n_frames * 2 > len([c for c in range(10) if c != bg]):
        return None

    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    color_pairs = [(palette[2 * i], palette[2 * i + 1]) for i in range(n_frames)]

    base_specs = []
    for frame_color, seed_color in color_pairs:
        gap_count = random.randint(1, 4)
        inner_h, inner_w = _sample_inner_size(gap_count)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h > h or outer_w > w:
            return None
        base_specs.append(
            {
                "frame_color": int(frame_color),
                "seed_color": int(seed_color),
                "gap_count": int(gap_count),
                "side": random.choice(SIDES),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
            }
        )

    base_specs.sort(key=lambda item: item["outer_h"] * item["outer_w"], reverse=True)

    blocked = np.zeros(shape, dtype=bool)
    placed = []
    for spec in base_specs:
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        placed_here = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(0, h - outer_h)
            c0 = random.randint(0, w - outer_w)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r1 + 2)
            cc1 = min(w, c1 + 2)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue

            gap_candidates = _noncorner_perimeter_cells(r0, c0, r1, c1)
            if len(gap_candidates) < spec["gap_count"]:
                continue

            gaps_pos = random.sample(gap_candidates, spec["gap_count"])
            blocked[rr0:rr1, cc0:cc1] = True
            placed.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "frame_color": spec["frame_color"],
                    "seed_color": spec["seed_color"],
                    "gap_count": spec["gap_count"],
                    "gaps_pos": [(int(r), int(c)) for r, c in gaps_pos],
                    "side": spec["side"],
                }
            )
            placed_here = True
            break
        if not placed_here:
            return None

    return placed


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(vals, counts)}
    bg_count = count_map.get(int(bg), 0)
    return all(bg_count > cnt for color, cnt in count_map.items() if color != bg)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        shape = (random.randint(18, 30), random.randint(18, 30))
        bg = random.choice([0, 1])
        specs = _sample_specs(shape, bg)
        if specs is None:
            continue

        inp = _draw_input(shape, bg, specs)
        if not _bg_is_dominant(inp, bg):
            continue

        expected = _expected_output(inp, specs)
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a valid input grid")
