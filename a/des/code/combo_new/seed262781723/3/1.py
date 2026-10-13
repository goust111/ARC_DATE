import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIR_ORDER = ("up", "down", "left", "right")
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_TO_CROSS_POS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 40
GRID_MIN = 18
GRID_MAX = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _find_legend_mapping(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    assert h >= 3 and w >= 3
    cross_mask = np.zeros((3, 3), dtype=bool)
    for r, c in DIR_TO_CROSS_POS.values():
        cross_mask[r, c] = True

    corners = [
        (0, 0),
        (0, w - 3),
        (h - 3, 0),
        (h - 3, w - 3),
    ]
    mappings = []
    for r0, c0 in corners:
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        if patch.shape != (3, 3):
            continue
        if not np.array_equal(patch != BG, cross_mask):
            continue
        colors = [int(patch[r, c]) for r, c in DIR_TO_CROSS_POS.values()]
        if len(set(colors)) != 4:
            continue
        mapping = {
            int(patch[0, 1]): DIR_TO_DELTA["up"],
            int(patch[2, 1]): DIR_TO_DELTA["down"],
            int(patch[1, 0]): DIR_TO_DELTA["left"],
            int(patch[1, 2]): DIR_TO_DELTA["right"],
        }
        mappings.append(mapping)
    assert len(mappings) == 1, "expected exactly one corner cross legend"
    return mappings[0]


def _parse_frame(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 4 or ww < 4:
        return None

    border = {
        (rr, cc)
        for rr in range(hh)
        for cc in range(ww)
        if rr in (0, hh - 1) or cc in (0, ww - 1)
    }
    actual = {(r - r0, c - c0) for r, c in coords}
    if not actual <= border:
        return None
    missing = border - actual
    if len(missing) != 1:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    block_pts = np.argwhere((inner != BG) & (inner != FRAME))
    if len(block_pts) == 0:
        return None
    colors = {int(inner[r, c]) for r, c in block_pts}
    if len(colors) != 1:
        return None
    color = colors.pop()

    br0, bc0 = block_pts.min(axis=0)
    br1, bc1 = block_pts.max(axis=0)
    bh = int(br1 - br0 + 1)
    bw = int(bc1 - bc0 + 1)
    if (bh, bw) not in {(2, 2), (3, 3)}:
        return None
    if len(block_pts) != bh * bw:
        return None
    if not np.all(inner[br0 : br1 + 1, bc0 : bc1 + 1] == color):
        return None

    return {
        "frame_bbox": (r0, c0, r1, c1),
        "block_color": color,
        "block_bbox": (r0 + 1 + int(br0), c0 + 1 + int(bc0), r0 + int(br1) + 1, c0 + int(bc1) + 1),
        "block_size": bh,
        "gap_local": next(iter(missing)),
    }


def _move_target(frame: dict, direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["frame_bbox"]
    br0, bc0, br1, bc1 = frame["block_bbox"]
    bh = br1 - br0 + 1
    bw = bc1 - bc0 + 1
    dr, dc = direction
    if dr == -1:
        return r0 - bh, bc0
    if dr == 1:
        return r1 + 1, bc0
    if dc == -1:
        return br0, c0 - bw
    return br0, c1 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner color legend, then push each trapped colored block straight out of its grey frame.

    Input structure:
    - The background is black.
    - One corner contains a 3x3 cross legend: four distinct colored single pixels occupy the up, down, left, and right positions around a black center.
    - Elsewhere, several grey one-pixel-thick rectangular frames each have exactly one missing border pixel, and each frame encloses one solid 2x2 or 3x3 colored block.

    Rule:
    - Decode the legend by mapping each color to the direction of its cross position.
    - For every grey frame, identify the enclosed block color and look up its direction in the legend.
    - Ignore the frame gap, erase the block from inside the frame, and translate it straight in the mapped direction until it sits immediately outside the frame.

    Key insight:
    - The border gap is a distractor; only the corner cross legend determines which way each enclosed block moves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    mapping = _find_legend_mapping(g)

    frames = []
    for comp in _components(g == FRAME):
        frame = _parse_frame(g, comp)
        if frame is not None:
            frames.append(frame)
    assert frames, "expected at least one grey frame"

    out = g.copy()
    for frame in frames:
        color = frame["block_color"]
        assert color in mapping, "every trapped block color must appear in the legend"
        br0, bc0, br1, bc1 = frame["block_bbox"]
        out[br0 : br1 + 1, bc0 : bc1 + 1] = BG

        tr0, tc0 = _move_target(frame, mapping[color])
        bh = br1 - br0 + 1
        bw = bc1 - bc0 + 1
        assert 0 <= tr0 and 0 <= tc0
        assert tr0 + bh <= out.shape[0] and tc0 + bw <= out.shape[1]
        target = out[tr0 : tr0 + bh, tc0 : tc0 + bw]
        assert np.all(target == BG), "moved block must land on background"
        target[:, :] = color

    return out


def _legend_patch(color_by_dir: dict[str, int]) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    for direction, (r, c) in DIR_TO_CROSS_POS.items():
        patch[r, c] = int(color_by_dir[direction])
    return patch


def _legend_corner_bbox(h: int, w: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, 2, 2
    if corner == "tr":
        return 0, w - 3, 2, w - 1
    if corner == "bl":
        return h - 3, 0, h - 1, 2
    return h - 3, w - 3, h - 1, w - 1


def _expand_rect(rect: tuple[int, int, int, int], h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _sample_frame_meta(direction: str, color: int, h: int, w: int, rng: random.Random) -> dict | None:
    block_size = rng.choice([2, 3])
    inner_h = rng.randint(block_size, block_size + 3)
    inner_w = rng.randint(block_size, block_size + 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    block_r = rng.randint(0, inner_h - block_size)
    block_c = rng.randint(0, inner_w - block_size)
    gap_side = rng.choice(DIR_ORDER)
    if gap_side in ("up", "down"):
        gap_local = (0 if gap_side == "up" else outer_h - 1, rng.randint(1, outer_w - 2))
    else:
        gap_local = (rng.randint(1, outer_h - 2), 0 if gap_side == "left" else outer_w - 1)

    min_r0 = 1 + (block_size if direction == "up" else 0)
    max_r0 = h - outer_h - 2 - (block_size if direction == "down" else 0)
    min_c0 = 1 + (block_size if direction == "left" else 0)
    max_c0 = w - outer_w - 2 - (block_size if direction == "right" else 0)
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None

    return {
        "direction": direction,
        "color": color,
        "block_size": block_size,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "block_r": block_r,
        "block_c": block_c,
        "gap_local": gap_local,
        "row_range": (min_r0, max_r0),
        "col_range": (min_c0, max_c0),
    }


def _candidate_positions(meta: dict, h: int, w: int, protected: list[tuple[int, int, int, int]]) -> list[dict]:
    outer_h = meta["outer_h"]
    outer_w = meta["outer_w"]
    block_size = meta["block_size"]
    block_r = meta["block_r"]
    block_c = meta["block_c"]
    positions = []
    min_r0, max_r0 = meta["row_range"]
    min_c0, max_c0 = meta["col_range"]
    for frame_r0 in range(min_r0, max_r0 + 1):
        for frame_c0 in range(min_c0, max_c0 + 1):
            frame_r1 = frame_r0 + outer_h - 1
            frame_c1 = frame_c0 + outer_w - 1
            input_r0 = frame_r0 + 1 + block_r
            input_c0 = frame_c0 + 1 + block_c

            if meta["direction"] == "up":
                output_r0, output_c0 = frame_r0 - block_size, input_c0
            elif meta["direction"] == "down":
                output_r0, output_c0 = frame_r1 + 1, input_c0
            elif meta["direction"] == "left":
                output_r0, output_c0 = input_r0, frame_c0 - block_size
            else:
                output_r0, output_c0 = input_r0, frame_c1 + 1

            output_r1 = output_r0 + block_size - 1
            output_c1 = output_c0 + block_size - 1
            if not (0 <= output_r0 and 0 <= output_c0 and output_r1 < h and output_c1 < w):
                continue

            protect = _expand_rect(
                (
                    min(frame_r0, output_r0),
                    min(frame_c0, output_c0),
                    max(frame_r1, output_r1),
                    max(frame_c1, output_c1),
                ),
                h,
                w,
                pad=1,
            )
            if any(_rects_overlap(protect, other) for other in protected):
                continue

            positions.append(
                {
                    **meta,
                    "frame_bbox": (frame_r0, frame_c0, frame_r1, frame_c1),
                    "input_bbox": (input_r0, input_c0, input_r0 + block_size - 1, input_c0 + block_size - 1),
                    "output_bbox": (output_r0, output_c0, output_r1, output_c1),
                    "protect_bbox": protect,
                }
            )
    return positions


def _paint_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], gap_local: tuple[int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    gr, gc = gap_local
    grid[r0 + gr, c0 + gc] = BG


def _paint_block(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _render_grids(h: int, w: int, corner: str, color_by_dir: dict[str, int], frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    lr0, lc0, lr1, lc1 = _legend_corner_bbox(h, w, corner)
    patch = _legend_patch(color_by_dir)
    inp[lr0 : lr1 + 1, lc0 : lc1 + 1] = patch
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = patch

    for frame in frames:
        _paint_frame(inp, frame["frame_bbox"], frame["gap_local"])
        _paint_frame(out, frame["frame_bbox"], frame["gap_local"])
        _paint_block(inp, frame["input_bbox"], frame["color"])
        _paint_block(out, frame["output_bbox"], frame["color"])

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    frame_count = rng.randint(2, 4)
    min_size = min(GRID_MAX, GRID_MIN + 3 * (frame_count - 2))
    h = rng.randint(min_size, GRID_MAX)
    w = rng.randint(min_size, GRID_MAX)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    legend_colors = rng.sample(PALETTE_COLORS, 4)
    color_by_dir = dict(zip(DIR_ORDER, legend_colors))

    protected = [_expand_rect(_legend_corner_bbox(h, w, corner), h, w, pad=1)]
    directions = rng.sample(list(DIR_ORDER), k=frame_count)
    frames = []

    for direction in directions:
        color = color_by_dir[direction]
        placed = None
        for _ in range(PLACEMENT_ATTEMPTS):
            meta = _sample_frame_meta(direction, color, h, w, rng)
            if meta is None:
                continue
            options = _candidate_positions(meta, h, w, protected)
            if not options:
                continue
            placed = rng.choice(options)
            break
        if placed is None:
            return None
        frames.append(placed)
        protected.append(placed["protect_bbox"])

    inp, expected = _render_grids(h, w, corner, color_by_dir, frames)
    if np.count_nonzero(inp == BG) <= np.count_nonzero(inp != BG):
        return None

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
