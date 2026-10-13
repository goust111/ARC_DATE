from __future__ import annotations

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 8
TRANSPOSED_GAP_TO_FILL = {
    "top": 1,
    "right": 2,
    "bottom": 3,
    "left": 4,
}
ORIGINAL_TO_TRANSPOSED_SIDE = {
    "top": "left",
    "right": "bottom",
    "bottom": "right",
    "left": "top",
}
CARDINAL4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "right", "bottom", "left")
GRID_MIN = 13
GRID_MAX = 15
FRAME_COUNT_CHOICES = (3, 4)
FRAME_SIZE_MIN = 4
FRAME_SIZE_MAX = 6
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in CARDINAL4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _flood_fill(grid: np.ndarray, start: tuple[int, int], color: int) -> None:
    h, w = grid.shape
    r0, c0 = map(int, start)
    if not (0 <= r0 < h and 0 <= c0 < w):
        return
    old = int(grid[r0, c0])
    new = int(color)
    if old == new:
        return
    stack = [(r0, c0)]
    while stack:
        r, c = stack.pop()
        if not (0 <= r < h and 0 <= c < w):
            continue
        if int(grid[r, c]) != old:
            continue
        grid[r, c] = new
        for dr, dc in CARDINAL4:
            stack.append((r + dr, c + dc))


def _frame_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _analyze_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, FRAME_COLOR}), "input must contain only black background and cyan frames"

    frames: list[dict] = []
    for coords in _components(g == FRAME_COLOR):
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert FRAME_SIZE_MIN <= h <= FRAME_SIZE_MAX, "frame height out of range"
        assert FRAME_SIZE_MIN <= w <= FRAME_SIZE_MAX, "frame width out of range"

        coord_set = {tuple(map(int, rc)) for rc in coords}
        border = _frame_border_cells(r0, c0, r1, c1)
        expected_area = len(border) - 1
        assert len(coord_set) == expected_area, "frame must be a rectangular border with exactly one missing border cell"

        gaps = list(border - coord_set)
        assert len(gaps) == 1, "frame must have exactly one gap"
        gap = gaps[0]
        gap_r, gap_c = gap

        if gap_r == r0:
            assert c0 < gap_c < c1, "gap on top side must be non-corner"
            side = "top"
        elif gap_r == r1:
            assert c0 < gap_c < c1, "gap on bottom side must be non-corner"
            side = "bottom"
        elif gap_c == c0:
            assert r0 < gap_r < r1, "gap on left side must be non-corner"
            side = "left"
        elif gap_c == c1:
            assert r0 < gap_r < r1, "gap on right side must be non-corner"
            side = "right"
        else:
            raise AssertionError("gap must lie on the frame border")

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0, "frame must enclose a nonempty interior"
        assert np.all(interior == BACKGROUND), "frame interior must be empty in the input"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "height": h,
                "width": w,
            }
        )

    assert frames, "expected at least one frame"
    return sorted(frames, key=lambda frame: frame["bbox"])


def _solve_oracle(grid: np.ndarray) -> np.ndarray:
    transposed = np.asarray(grid, dtype=int).T.copy()
    out = transposed.copy()
    for frame in _analyze_frames(transposed):
        r0, c0, r1, c1 = frame["bbox"]
        fill = TRANSPOSED_GAP_TO_FILL[frame["side"]]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
        gap_r, gap_c = frame["gap"]
        out[gap_r, gap_c] = BACKGROUND
    return out


def _wrong_no_transpose(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame in _analyze_frames(out):
        r0, c0, r1, c1 = frame["bbox"]
        fill = TRANSPOSED_GAP_TO_FILL[frame["side"]]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def _wrong_transpose_only(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid, dtype=int).T.copy()


def _wrong_transposed_layout_original_orientation_colors(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.T.copy()
    for frame in _analyze_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        tr0, tc0, tr1, tc1 = c0, r0, c1, r1
        fill = TRANSPOSED_GAP_TO_FILL[frame["side"]]
        out[tr0 + 1 : tr1, tc0 + 1 : tc1] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transpose the square frame grid, then fill each frame interior by the transposed gap orientation.

    Input structure:
    - The input is a square black grid containing 3 or 4 cyan rectangular frames.
    - Each frame is axis-aligned, 4x4 to 6x6, and has exactly one missing non-corner border pixel.
    - Every frame interior is black, and the frames are separated from one another.

    Rule:
    - Reflect the whole grid across the main diagonal by transposing rows and columns.
    - In the transposed grid, determine whether each frame's gap lies on its top, right, bottom, or left side.
    - Fill that frame's interior with blue for top gaps, red for right gaps, green for bottom gaps, and yellow for left gaps.
    - Keep the cyan border pixels unchanged and keep the gap cell itself black.

    Key insight:
    - The missing border pixel must be interpreted after the global transpose, because the transpose changes each gap's side before that side is mapped to a fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "expected a square grid"

    out = g.T.copy()
    for frame in _analyze_frames(out):
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        fill = TRANSPOSED_GAP_TO_FILL[frame["side"]]

        out[gap_r, gap_c] = FRAME_COLOR
        seed = (r0 + 1, c0 + 1)
        assert int(out[seed]) == BACKGROUND, "frame interior must contain a black seed cell"
        _flood_fill(out, seed, fill)
        out[gap_r, gap_c] = BACKGROUND

    return out


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> None:
    r0, c0, r1, c1 = bbox
    border = _frame_border_cells(r0, c0, r1, c1)
    if side == "top":
        gap = (r0, int(_RNG.integers(c0 + 1, c1)))
    elif side == "bottom":
        gap = (r1, int(_RNG.integers(c0 + 1, c1)))
    elif side == "left":
        gap = (int(_RNG.integers(r0 + 1, r1)), c0)
    else:
        assert side == "right"
        gap = (int(_RNG.integers(r0 + 1, r1)), c1)

    for r, c in border:
        if (r, c) != gap:
            grid[r, c] = FRAME_COLOR


def _quadrants(n: int) -> list[tuple[int, int, int, int]]:
    split = n // 2
    top = (0, split - 1)
    bottom = (split + 1, n - 1)
    left = (0, split - 1)
    right = (split + 1, n - 1)
    return [
        (top[0], top[1], left[0], left[1]),
        (top[0], top[1], right[0], right[1]),
        (bottom[0], bottom[1], left[0], left[1]),
        (bottom[0], bottom[1], right[0], right[1]),
    ]


def _sample_frame_size(room_h: int, room_w: int, *, force_non_square: bool) -> tuple[int, int] | None:
    options = [
        (h, w)
        for h in range(FRAME_SIZE_MIN, min(FRAME_SIZE_MAX, room_h) + 1)
        for w in range(FRAME_SIZE_MIN, min(FRAME_SIZE_MAX, room_w) + 1)
        if (not force_non_square) or h != w
    ]
    if not options:
        return None
    idx = int(_RNG.integers(len(options)))
    return options[idx]


def _attempt_generate() -> np.ndarray | None:
    count = int(_RNG.choice(FRAME_COUNT_CHOICES, p=np.array([0.4, 0.6])))
    n = int(_RNG.integers(max(GRID_MIN, 14 if count == 4 else GRID_MIN), GRID_MAX + 1))
    grid = np.zeros((n, n), dtype=int)

    quads = _quadrants(n)
    if any(q[1] - q[0] + 1 < FRAME_SIZE_MIN or q[3] - q[2] + 1 < FRAME_SIZE_MIN for q in quads):
        return None
    quad_ids = _RNG.choice(len(quads), size=count, replace=False)

    if count == 4:
        sides = list(_RNG.permutation(SIDES))
    else:
        chosen = list(_RNG.choice(np.array(SIDES), size=3, replace=False))
        _RNG.shuffle(chosen)
        sides = chosen

    had_non_square = False
    for i, (quad_id, side) in enumerate(zip(quad_ids.tolist(), sides)):
        r_lo, r_hi, c_lo, c_hi = quads[int(quad_id)]
        room_h = r_hi - r_lo + 1
        room_w = c_hi - c_lo + 1
        force_non_square = not had_non_square and i == count - 1
        size = _sample_frame_size(room_h, room_w, force_non_square=force_non_square)
        if size is None:
            return None
        h, w = size
        had_non_square = had_non_square or h != w
        r0 = int(_RNG.integers(r_lo, r_hi - h + 2))
        c0 = int(_RNG.integers(c_lo, c_hi - w + 2))
        _paint_frame(grid, (r0, c0, r0 + h - 1, c0 + w - 1), str(side))

    try:
        frames = _analyze_frames(grid)
    except AssertionError:
        return None

    if len(frames) != count:
        return None
    if len({frame["side"] for frame in frames}) < 3:
        return None
    if len({ORIGINAL_TO_TRANSPOSED_SIDE[frame["side"]] for frame in frames}) < 3:
        return None
    if not any(frame["height"] != frame["width"] for frame in frames):
        return None
    if np.array_equal(grid, grid.T):
        return None

    expected = _solve_oracle(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(expected, _wrong_no_transpose(grid)):
        return None
    if np.array_equal(expected, _wrong_transpose_only(grid)):
        return None
    if np.array_equal(expected, _wrong_transposed_layout_original_orientation_colors(grid)):
        return None

    frame_count_out = len(_analyze_frames(grid.T.copy()))
    if frame_count_out != count:
        return None
    fill_colors = set(map(int, np.unique(expected))) - {BACKGROUND, FRAME_COLOR}
    if len(fill_colors) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
