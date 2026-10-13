import random

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
GENERATION_ATTEMPTS = 300
PATTERN_ATTEMPTS = 80


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_info(
    grid: np.ndarray, bg: int, color: int, comp: list[tuple[int, int]]
) -> dict | None:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    border_pixels = 0
    gap: tuple[int, int] | None = None
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            value = int(grid[r, c])
            if on_border:
                if value == int(color):
                    border_pixels += 1
                elif value == int(bg):
                    if gap is not None:
                        return None
                    gap = (r, c)
                else:
                    return None
            elif value == int(color):
                return None

    expected_border = 2 * h + 2 * w - 5
    if border_pixels != expected_border or gap is None:
        return None

    gr, gc = gap
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None

    if gr == r0:
        side = "up"
    elif gr == r1:
        side = "down"
    elif gc == c0:
        side = "left"
    elif gc == c1:
        side = "right"
    else:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.any(interior != int(bg)):
        return None
    if (
        np.any(interior[0, :] != int(bg))
        or np.any(interior[-1, :] != int(bg))
        or np.any(interior[:, 0] != int(bg))
        or np.any(interior[:, -1] != int(bg))
    ):
        return None

    ih, iw = interior.shape
    if side == "up":
        tr0, tc0 = r0 - ih, c0 + 1
    elif side == "down":
        tr0, tc0 = r1 + 1, c0 + 1
    elif side == "left":
        tr0, tc0 = r0 + 1, c0 - iw
    else:
        tr0, tc0 = r0 + 1, c1 + 1

    tr1 = tr0 + ih - 1
    tc1 = tc0 + iw - 1
    h_grid, w_grid = grid.shape
    if tr0 < 0 or tc0 < 0 or tr1 >= h_grid or tc1 >= w_grid:
        return None
    if not np.all(grid[tr0 : tr1 + 1, tc0 : tc1 + 1] == int(bg)):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "side": side,
        "gap": gap,
        "target_bbox": (tr0, tc0, tr1, tc1),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        if int(color) == int(bg):
            continue
        for comp in _color_components(grid, int(color)):
            info = _frame_info(grid, bg, int(color), comp)
            if info is not None:
                frames.append(info)
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each enclosed pattern through the side indicated by its frame gap and stamp it into the adjacent blank region.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular frames appear, each drawn in a single color and containing exactly one missing border pixel.
    - Each frame encloses a smaller multicolor pattern that stays separated from the border.
    - On the outside of the missing-border side, there is a blank rectangle matching the frame interior size.

    Rule:
    - For each frame, locate the unique border gap and determine whether it is on the top, bottom, left, or right side.
    - Copy the full interior crop and mirror it across the axis perpendicular to that side: left/right gaps use a left-right flip, top/bottom gaps use an up-down flip.
    - Paste the mirrored crop into the blank rectangle immediately outside the gap side.
    - Leave the original frame and interior pattern unchanged.

    Key insight:
    - The gap side simultaneously encodes where the copied pattern must go and which reflection axis must be applied before translating it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["bbox"]
        tr0, tc0, tr1, tc1 = frame["target_bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if frame["side"] in ("left", "right"):
            transformed = np.fliplr(interior)
        else:
            transformed = np.flipud(interior)
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = transformed

    return out


def _random_connected_shape(
    rng: random.Random, h: int, w: int, size: int
) -> list[tuple[int, int]]:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = {start}
    while len(cells) < size:
        base = rng.choice(tuple(frontier))
        candidates = [
            (nr, nc)
            for nr, nc in _neighbors4(base[0], base[1], h, w)
            if (nr, nc) not in cells
        ]
        if candidates:
            nxt = rng.choice(candidates)
            cells.add(nxt)
            frontier.add(nxt)
            continue
        frontier.discard(base)
        if not frontier:
            frontier = set(cells)
    return sorted(cells)


def _make_pattern(
    rng: random.Random, ih: int, iw: int, side: str, border_color: int
) -> np.ndarray | None:
    core_h = ih - 2
    core_w = iw - 2
    if core_h < 1 or core_w < 1:
        return None

    colors = [c for c in FRAME_COLORS if c != int(border_color)]
    for _ in range(PATTERN_ATTEMPTS):
        size = rng.randint(2, min(core_h * core_w, 6))
        cells = _random_connected_shape(rng, core_h, core_w, size)
        patch = np.zeros((ih, iw), dtype=int)
        color_count = rng.randint(1, min(3, len(cells), len(colors)))
        palette = rng.sample(colors, color_count)
        for rr, cc in cells:
            patch[rr + 1, cc + 1] = rng.choice(palette)
        if side in ("left", "right"):
            if np.array_equal(patch, np.fliplr(patch)):
                continue
        else:
            if np.array_equal(patch, np.flipud(patch)):
                continue
        return patch
    return None


def _layout_shape(rng: random.Random, n_frames: int) -> tuple[int, int, list[tuple[int, int]]]:
    if n_frames == 1:
        return 1, 1, [(0, 0)]
    if n_frames == 2:
        if rng.random() < 0.5:
            return 1, 2, [(0, 0), (0, 1)]
        return 2, 1, [(0, 0), (1, 0)]
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(slots)
    return 2, 2, slots[:n_frames]


def _block_sizes(rng: random.Random, count: int) -> list[int]:
    if count == 1:
        return [rng.randint(14, 22)]
    return [rng.randint(12, 14) for _ in range(count)]


def _place_frame_in_block(
    rng: random.Random,
    grid: np.ndarray,
    block: tuple[int, int, int, int],
    side: str,
) -> bool:
    br0, bc0, bh, bw = block
    usable_h = bh - 2
    usable_w = bw - 2

    if side in ("left", "right"):
        max_h = min(9, usable_h)
        max_w = min(9, (usable_w + 2) // 2)
    else:
        max_h = min(9, (usable_h + 2) // 2)
        max_w = min(9, usable_w)
    if max_h < 6 or max_w < 6:
        return False

    outer_h = rng.randint(6, max_h)
    outer_w = rng.randint(6, max_w)
    ih = outer_h - 2
    iw = outer_w - 2

    if side in ("left", "right"):
        reserve_h = outer_h
        reserve_w = outer_w + iw
    else:
        reserve_h = outer_h + ih
        reserve_w = outer_w

    rr0 = rng.randint(br0 + 1, br0 + bh - reserve_h - 1)
    cc0 = rng.randint(bc0 + 1, bc0 + bw - reserve_w - 1)

    if side == "right":
        fr0, fc0 = rr0, cc0
    elif side == "left":
        fr0, fc0 = rr0, cc0 + iw
    elif side == "down":
        fr0, fc0 = rr0, cc0
    else:
        fr0, fc0 = rr0 + ih, cc0

    fr1 = fr0 + outer_h - 1
    fc1 = fc0 + outer_w - 1
    border_color = rng.choice(FRAME_COLORS)
    pattern = _make_pattern(rng, ih, iw, side, border_color)
    if pattern is None:
        return False

    grid[fr0, fc0 : fc1 + 1] = border_color
    grid[fr1, fc0 : fc1 + 1] = border_color
    grid[fr0 : fr1 + 1, fc0] = border_color
    grid[fr0 : fr1 + 1, fc1] = border_color

    if side == "up":
        gap = (fr0, rng.randint(fc0 + 1, fc1 - 1))
    elif side == "down":
        gap = (fr1, rng.randint(fc0 + 1, fc1 - 1))
    elif side == "left":
        gap = (rng.randint(fr0 + 1, fr1 - 1), fc0)
    else:
        gap = (rng.randint(fr0 + 1, fr1 - 1), fc1)
    grid[gap] = BG

    interior = grid[fr0 + 1 : fr1, fc0 + 1 : fc1]
    interior[:] = pattern
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(1, 4)
    rows, cols, slots = _layout_shape(rng, n_frames)
    row_heights = _block_sizes(rng, rows)
    col_widths = _block_sizes(rng, cols)
    row_gap = rng.randint(1, 2) if rows == 2 else 0
    col_gap = rng.randint(1, 2) if cols == 2 else 0

    height = sum(row_heights) + row_gap
    width = sum(col_widths) + col_gap
    if height > 30 or width > 30:
        return None

    row_starts = [0]
    if rows == 2:
        row_starts.append(row_heights[0] + row_gap)
    col_starts = [0]
    if cols == 2:
        col_starts.append(col_widths[0] + col_gap)

    blocks: list[tuple[int, int, int, int]] = []
    for rr, cc in slots:
        blocks.append((row_starts[rr], col_starts[cc], row_heights[rr], col_widths[cc]))

    directions = rng.sample(DIRECTIONS, n_frames)
    rng.shuffle(blocks)

    grid = np.zeros((height, width), dtype=int)
    for block, side in zip(blocks, directions):
        if not _place_frame_in_block(rng, grid, block, side):
            return None

    frames = _find_frames(grid, BG)
    if len(frames) != n_frames:
        return None
    if sorted(frame["side"] for frame in frames) != sorted(directions):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
