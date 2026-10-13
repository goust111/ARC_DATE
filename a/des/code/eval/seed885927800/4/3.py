import numpy as np


SIDES = ("top", "bottom", "left", "right")
FRAME_COLORS = (1, 2, 3, 4)
STAMP_COLORS = (5, 6, 7, 8, 9)
GEN_RETRIES = 300
PLACE_RETRIES = 200


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if int(coords.shape[0]) != int(border.sum()):
            continue
        if not np.all(patch[border] == color):
            continue
        if np.any(patch[~border] == color):
            continue
        if not np.all(patch[1:-1, 1:-1] == bg):
            continue

        frames.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    assert frames, "main: expected at least one rectangular frame"
    return frames


def _stamp_info(grid: np.ndarray, bg: int, frame: dict) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    candidates: list[dict] = []

    def add_horizontal(side: str, edge_row: int, step: int) -> None:
        if not (0 <= edge_row < grid.shape[0]):
            return
        vals = grid[edge_row, c0 : c1 + 1]
        idx = np.flatnonzero(vals != bg)
        if idx.size == 0:
            return
        start = int(idx[0])
        end = int(idx[-1])
        size = end - start + 1
        if size not in (2, 3):
            return
        sr0 = edge_row - size + 1 if side == "top" else edge_row
        sc0 = c0 + start
        sr1 = sr0 + size - 1
        sc1 = sc0 + size - 1
        if sr0 < 0 or sr1 >= grid.shape[0] or sc0 < 0 or sc1 >= grid.shape[1]:
            return
        patch = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
        if patch.shape != (size, size) or np.any(patch == bg):
            return
        if sc0 != c0 + (w - size) // 2 or (w - size) % 2 != 0:
            return
        candidates.append(
            {
                "side": side,
                "bbox": (sr0, sc0, sr1, sc1),
                "pattern": patch.copy(),
            }
        )

    def add_vertical(side: str, edge_col: int, step: int) -> None:
        if not (0 <= edge_col < grid.shape[1]):
            return
        vals = grid[r0 : r1 + 1, edge_col]
        idx = np.flatnonzero(vals != bg)
        if idx.size == 0:
            return
        start = int(idx[0])
        end = int(idx[-1])
        size = end - start + 1
        if size not in (2, 3):
            return
        sr0 = r0 + start
        sc0 = edge_col - size + 1 if side == "left" else edge_col
        sr1 = sr0 + size - 1
        sc1 = sc0 + size - 1
        if sr0 < 0 or sr1 >= grid.shape[0] or sc0 < 0 or sc1 >= grid.shape[1]:
            return
        patch = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
        if patch.shape != (size, size) or np.any(patch == bg):
            return
        if sr0 != r0 + (h - size) // 2 or (h - size) % 2 != 0:
            return
        candidates.append(
            {
                "side": side,
                "bbox": (sr0, sc0, sr1, sc1),
                "pattern": patch.copy(),
            }
        )

    add_horizontal("top", r0 - 1, -1)
    add_horizontal("bottom", r1 + 1, 1)
    add_vertical("left", c0 - 1, -1)
    add_vertical("right", c1 + 1, 1)

    assert len(candidates) == 1, "main: each frame must have exactly one attached stamp"
    return candidates[0]


def _tile_inside(out: np.ndarray, frame: dict, stamp: dict, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    sr0, sc0, sr1, sc1 = stamp["bbox"]
    pattern = np.asarray(stamp["pattern"], dtype=int)
    size = int(pattern.shape[0])
    ir0, ic0 = r0 + 1, c0 + 1
    ir1, ic1 = r1 - 1, c1 - 1

    if stamp["side"] == "top":
        row_anchor = ir0
        col_anchor = sc0
    elif stamp["side"] == "bottom":
        row_anchor = ir1 - size + 1
        col_anchor = sc0
    elif stamp["side"] == "left":
        row_anchor = sr0
        col_anchor = ic0
    else:
        row_anchor = sr0
        col_anchor = ic1 - size + 1

    rows = np.arange(ir0, ir1 + 1)[:, None]
    cols = np.arange(ic0, ic1 + 1)[None, :]
    rr = (rows - row_anchor) % size
    cc = (cols - col_anchor) % size
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = pattern[rr, cc]
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each frame's attached stamp inside the frame, with the stamp side
    selecting which interior edge acts as the tiling anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _bg_color(g)
    frames = _frame_candidates(g, bg)

    out = g.copy()
    for frame in frames:
        stamp = _stamp_info(g, bg, frame)
        _tile_inside(out, frame, stamp, bg)
    return out


def _stamp_pattern(size: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        n_colors = 2 if size == 2 else int(rng.integers(2, 4))
        colors = np.asarray(rng.choice(STAMP_COLORS, size=n_colors, replace=False), dtype=int)
        pattern = np.asarray(rng.choice(colors, size=(size, size), replace=True), dtype=int)
        if len(np.unique(pattern)) < 2:
            continue
        if len({tuple(row) for row in pattern}) < 2:
            continue
        if len({tuple(pattern[:, c]) for c in range(size)}) < 2:
            continue
        if any(np.array_equal(pattern, np.roll(pattern, shift, axis=0)) for shift in range(1, size)):
            continue
        if any(np.array_equal(pattern, np.roll(pattern, shift, axis=1)) for shift in range(1, size)):
            continue
        return pattern
    raise RuntimeError("failed to sample a usable stamp pattern")


def _parity_choices(lo: int, hi: int, parity: int) -> list[int]:
    return [value for value in range(lo, hi + 1) if value % 2 == parity]


def _frame_spec(rng: np.random.Generator, frame_color: int) -> dict:
    side = SIDES[int(rng.integers(len(SIDES)))]
    size = int(rng.integers(2, 4))
    pattern = _stamp_pattern(size, rng)

    if side in ("top", "bottom"):
        width = int(rng.choice(_parity_choices(6, 12, size % 2)))
        height = int(rng.integers(6, 12))
    else:
        height = int(rng.choice(_parity_choices(6, 12, size % 2)))
        width = int(rng.integers(6, 12))

    if (height - 2) % size == 0 and (width - 2) % size == 0:
        if side in ("top", "bottom"):
            width += 2
            if width > 12:
                width -= 4
        else:
            height += 2
            if height > 12:
                height -= 4

    if side == "top":
        full_h, full_w = height + size, width
        frame_offset = (size, 0)
        stamp_offset = (0, (width - size) // 2)
    elif side == "bottom":
        full_h, full_w = height + size, width
        frame_offset = (0, 0)
        stamp_offset = (height, (width - size) // 2)
    elif side == "left":
        full_h, full_w = height, width + size
        frame_offset = (0, size)
        stamp_offset = ((height - size) // 2, 0)
    else:
        full_h, full_w = height, width + size
        frame_offset = (0, 0)
        stamp_offset = ((height - size) // 2, width)

    return {
        "frame_color": int(frame_color),
        "side": side,
        "size": size,
        "pattern": pattern,
        "frame_h": height,
        "frame_w": width,
        "full_h": full_h,
        "full_w": full_w,
        "frame_offset": frame_offset,
        "stamp_offset": stamp_offset,
    }


def _place_specs(specs: list[dict], rng: np.random.Generator) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_h = max(spec["full_h"] for spec in specs)
    max_w = max(spec["full_w"] for spec in specs)
    padded_area = sum((spec["full_h"] + 2) * (spec["full_w"] + 2) for spec in specs)
    side = int(np.ceil(np.sqrt(padded_area * 1.6)))
    height = int(max(max_h + 4, side + rng.integers(0, 6), 20))
    width = int(max(max_w + 4, side + rng.integers(0, 6), 20))
    if rng.random() < 0.5:
        height += int(rng.integers(0, 5))
    else:
        width += int(rng.integers(0, 5))

    blocked = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int]] = []
    ordered = sorted(enumerate(specs), key=lambda item: item[1]["full_h"] * item[1]["full_w"], reverse=True)
    placed: dict[int, tuple[int, int]] = {}

    for idx, spec in ordered:
        if height - spec["full_h"] < 3 or width - spec["full_w"] < 3:
            return None
        done = False
        for _ in range(PLACE_RETRIES):
            top = int(rng.integers(1, height - spec["full_h"] - 1))
            left = int(rng.integers(1, width - spec["full_w"] - 1))
            if blocked[top - 1 : top + spec["full_h"] + 1, left - 1 : left + spec["full_w"] + 1].any():
                continue
            blocked[top - 1 : top + spec["full_h"] + 1, left - 1 : left + spec["full_w"] + 1] = True
            placed[idx] = (top, left)
            done = True
            break
        if not done:
            return None

    for idx in range(len(specs)):
        positions.append(placed[idx])
    return height, width, positions


def _draw_spec(grid: np.ndarray, spec: dict, origin: tuple[int, int]) -> None:
    top, left = origin
    frame_r, frame_c = spec["frame_offset"]
    stamp_r, stamp_c = spec["stamp_offset"]
    r0 = top + frame_r
    c0 = left + frame_c
    r1 = r0 + spec["frame_h"] - 1
    c1 = c0 + spec["frame_w"] - 1
    color = spec["frame_color"]

    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    sr0 = top + stamp_r
    sc0 = left + stamp_c
    size = spec["size"]
    grid[sr0 : sr0 + size, sc0 : sc0 + size] = spec["pattern"]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(2, 5))
    frame_colors = [int(c) for c in rng.choice(FRAME_COLORS, size=n_frames, replace=False)]
    specs = [_frame_spec(rng, color) for color in frame_colors]
    placement = _place_specs(specs, rng)
    if placement is None:
        return None

    height, width, positions = placement
    grid = np.zeros((height, width), dtype=int)
    for spec, origin in zip(specs, positions):
        _draw_spec(grid, spec, origin)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
