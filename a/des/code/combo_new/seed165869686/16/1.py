from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _detect_block(comp: dict) -> dict | None:
    if int(comp["area"]) != 4:
        return None
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    if coords != expected:
        return None
    return {
        "color": int(comp["color"]),
        "bbox": (r0, c0, r1, c1),
    }


def _detect_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    size = r1 - r0 + 1
    if size < 5 or c1 - c0 + 1 != size:
        return None

    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))

    if not coords.issubset(perimeter):
        return None

    missing = sorted(perimeter - coords)
    if not (1 <= len(missing) <= 3):
        return None
    if len(coords) != len(perimeter) - len(missing):
        return None

    side: str | None = None
    gap_axis: list[int] = []
    if all(r == r0 for r, _c in missing):
        side = "top"
        gap_axis = sorted(c for _r, c in missing)
        lo, hi = c0 + 1, c1 - 1
    elif all(r == r1 for r, _c in missing):
        side = "bottom"
        gap_axis = sorted(c for _r, c in missing)
        lo, hi = c0 + 1, c1 - 1
    elif all(c == c0 for _r, c in missing):
        side = "left"
        gap_axis = sorted(r for r, _c in missing)
        lo, hi = r0 + 1, r1 - 1
    elif all(c == c1 for _r, c in missing):
        side = "right"
        gap_axis = sorted(r for r, _c in missing)
        lo, hi = r0 + 1, r1 - 1
    else:
        return None

    if gap_axis[0] < lo or gap_axis[-1] > hi:
        return None
    if gap_axis[-1] - gap_axis[0] + 1 != len(gap_axis):
        return None

    return {
        "color": int(comp["color"]),
        "bbox": (r0, c0, r1, c1),
        "gap_side": side,
        "gap_axis": gap_axis,
    }


def _find_opposite_block(frame: dict, blocks: list[dict]) -> dict:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    side = OPPOSITE_SIDE[str(frame["gap_side"])]
    candidates: list[tuple[int, int, dict]] = []
    for block in blocks:
        br0, bc0, br1, bc1 = map(int, block["bbox"])
        if side == "top":
            if br1 < r0 and c0 <= bc0 and bc1 <= c1:
                candidates.append((r0 - br1 - 1, bc0, block))
        elif side == "bottom":
            if br0 > r1 and c0 <= bc0 and bc1 <= c1:
                candidates.append((br0 - r1 - 1, bc0, block))
        elif side == "left":
            if bc1 < c0 and r0 <= br0 and br1 <= r1:
                candidates.append((c0 - bc1 - 1, br0, block))
        else:
            if bc0 > c1 and r0 <= br0 and br1 <= r1:
                candidates.append((bc0 - c1 - 1, br0, block))
    assert candidates, "each frame must have a 2x2 marker on the opposite side"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rays into hollow square frames using the color block opposite each frame gap.

    Input structure:
    - The background is black.
    - The grid contains several hollow square frames with one-pixel-thick borders, and each frame has its own border color.
    - Every frame has exactly one contiguous gap of width 1-3 on one side of the border.
    - Outside each frame there are four solid 2x2 color blocks, one above, one below, one left, and one right.

    Rule:
    - For each frame, detect which side contains the border gap.
    - Find the 2x2 block on the opposite outside side of that frame.
    - Use that block's color to fill the interior strip aligned with the gap: a full interior column set for a top/bottom gap, or a full interior row set for a left/right gap.
    - Leave the frame border, the outside blocks, and all unrelated cells unchanged.

    Key insight:
    - The chosen fill color comes from the block opposite the opening, not from the nearby side of the frame, so the solver must invert the detected gap direction before casting the interior ray.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    components = _same_color_components(g, bg)

    frames = []
    blocks = []
    for comp in components:
        frame = _detect_frame(comp)
        if frame is not None:
            frames.append(frame)
            continue
        block = _detect_block(comp)
        if block is not None:
            blocks.append(block)

    assert frames, "expected at least one framed object"
    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        fill_color = int(_find_opposite_block(frame, blocks)["color"])
        side = str(frame["gap_side"])
        axis = list(map(int, frame["gap_axis"]))
        if side in ("top", "bottom"):
            for col in axis:
                out[r0 + 1 : r1, col] = fill_color
        else:
            for row in axis:
                out[row, c0 + 1 : c1] = fill_color
    return out


BG = 0
BLOCK_SIZE = 2
FRAME_SIZE_RANGE = (5, 7)
FRAME_COUNT_OPTIONS = (2, 3, 4)
BLOCK_DISTANCE_RANGE = (1, 2)
GRID_LIMIT = 30
EDGE_PAD = BLOCK_SIZE + BLOCK_DISTANCE_RANGE[1]
CELL_GAP = 2 * EDGE_PAD
MAX_GENERATION_ATTEMPTS = 300


def _split_total(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _layout_for_count(frame_count: int) -> tuple[int, int]:
    if frame_count == 2:
        return random.choice([(1, 2), (2, 1), (2, 2)])
    return (2, 2)


def _min_grid_size(rows: int, cols: int, size: int) -> tuple[int, int]:
    h = 2 * EDGE_PAD + rows * size + (rows - 1) * CELL_GAP
    w = 2 * EDGE_PAD + cols * size + (cols - 1) * CELL_GAP
    return h, w


def _starts(count: int, size: int, total: int) -> list[int]:
    min_total = 2 * EDGE_PAD + count * size + (count - 1) * CELL_GAP
    extras = _split_total(total - min_total, count + 1)
    pos = EDGE_PAD + extras[0]
    starts = [pos]
    for idx in range(1, count):
        pos += size + CELL_GAP + extras[idx]
        starts.append(pos)
    return starts


def _gap_axis_positions(size: int, width: int) -> list[int]:
    start = random.randint(1, size - width - 1)
    return list(range(start, start + width))


def _place_frame_border(
    grid: np.ndarray,
    *,
    r0: int,
    c0: int,
    size: int,
    color: int,
    gap_side: str,
    gap_axis: list[int],
) -> None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if gap_side == "top":
        for off in gap_axis:
            grid[r0, c0 + off] = BG
    elif gap_side == "bottom":
        for off in gap_axis:
            grid[r1, c0 + off] = BG
    elif gap_side == "left":
        for off in gap_axis:
            grid[r0 + off, c0] = BG
    else:
        for off in gap_axis:
            grid[r0 + off, c1] = BG


def _draw_block(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = color


def _block_origins(r0: int, c0: int, size: int) -> dict[str, tuple[int, int]]:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    dist_top = random.randint(*BLOCK_DISTANCE_RANGE)
    dist_bottom = random.randint(*BLOCK_DISTANCE_RANGE)
    dist_left = random.randint(*BLOCK_DISTANCE_RANGE)
    dist_right = random.randint(*BLOCK_DISTANCE_RANGE)
    return {
        "top": (
            r0 - dist_top - BLOCK_SIZE,
            c0 + random.randint(0, size - BLOCK_SIZE),
        ),
        "bottom": (
            r1 + dist_bottom + 1,
            c0 + random.randint(0, size - BLOCK_SIZE),
        ),
        "left": (
            r0 + random.randint(0, size - BLOCK_SIZE),
            c0 - dist_left - BLOCK_SIZE,
        ),
        "right": (
            r0 + random.randint(0, size - BLOCK_SIZE),
            c1 + dist_right + 1,
        ),
    }


def _paint_expected(
    grid: np.ndarray,
    *,
    r0: int,
    c0: int,
    size: int,
    gap_side: str,
    gap_axis: list[int],
    side_colors: dict[str, int],
) -> None:
    fill_color = int(side_colors[OPPOSITE_SIDE[gap_side]])
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    if gap_side in ("top", "bottom"):
        for off in gap_axis:
            grid[r0 + 1 : r1, c0 + off] = fill_color
    else:
        for off in gap_axis:
            grid[r0 + off, c0 + 1 : c1] = fill_color


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    rows, cols = _layout_for_count(frame_count)
    size = random.randint(*FRAME_SIZE_RANGE)
    min_h, min_w = _min_grid_size(rows, cols, size)
    if min_h > GRID_LIMIT or min_w > GRID_LIMIT:
        raise RuntimeError("layout exceeds ARC size limit")

    height = random.randint(min_h, GRID_LIMIT)
    width = random.randint(min_w, GRID_LIMIT)
    row_starts = _starts(rows, size, height)
    col_starts = _starts(cols, size, width)

    cells = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    active_cells = random.sample(cells, frame_count)

    palette = random.sample(range(1, 10), 4 + frame_count)
    block_palette = palette[:4]
    frame_colors = palette[4:]

    inp = np.full((height, width), BG, dtype=int)
    out = inp.copy()

    for idx, (cell_r, cell_c) in enumerate(active_cells):
        r0 = row_starts[cell_r]
        c0 = col_starts[cell_c]
        frame_color = int(frame_colors[idx])
        gap_side = random.choice(SIDES)
        gap_width = random.randint(1, min(3, size - 2))
        gap_axis = _gap_axis_positions(size, gap_width)
        _place_frame_border(
            inp,
            r0=r0,
            c0=c0,
            size=size,
            color=frame_color,
            gap_side=gap_side,
            gap_axis=gap_axis,
        )
        _place_frame_border(
            out,
            r0=r0,
            c0=c0,
            size=size,
            color=frame_color,
            gap_side=gap_side,
            gap_axis=gap_axis,
        )

        permuted = random.sample(block_palette, len(block_palette))
        side_colors = dict(zip(SIDES, permuted, strict=True))
        for side, (br0, bc0) in _block_origins(r0, c0, size).items():
            _draw_block(inp, br0, bc0, int(side_colors[side]))
            _draw_block(out, br0, bc0, int(side_colors[side]))

        _paint_expected(
            out,
            r0=r0,
            c0=c0,
            size=size,
            gap_side=gap_side,
            gap_axis=gap_axis,
            side_colors=side_colors,
        )

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, expected = _build_sample()
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC input")
