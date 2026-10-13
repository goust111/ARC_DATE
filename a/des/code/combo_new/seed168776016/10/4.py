import random
from collections import deque

import numpy as np


BG = 0
COLORS = list(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

OBJECT_COUNT_RANGE = (2, 4)
GRID_MIN = 18
GRID_MAX = 30
FRAME_H_RANGE = (6, 9)
FRAME_W_RANGE = (6, 9)
PATTERN_BOX_MAX = 4
GENERATION_RETRIES = 300
OBJECT_RETRIES = 120
POSITION_RETRIES = 80


def _bg_color(grid: np.ndarray) -> int:
    vals = np.bincount(grid.ravel())
    return int(vals.argmax())


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _frame_side_from_gap(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = gap
    r0, c0, r1, c1 = bbox
    sides = []
    if r == r0:
        sides.append("top")
    if r == r1:
        sides.append("bottom")
    if c == c0:
        sides.append("left")
    if c == c1:
        sides.append("right")
    return sides[0] if len(sides) == 1 else None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components4(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            pts = {tuple(map(int, p)) for p in comp.tolist()}
            missing = []
            valid = True
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    on_border = r in (r0, r1) or c in (c0, c1)
                    present = (r, c) in pts
                    if on_border:
                        if not present:
                            missing.append((r, c))
                    elif present:
                        valid = False
                        break
                if not valid:
                    break
            if not valid or len(missing) != 1:
                continue
            side = _frame_side_from_gap(missing[0], (r0, c0, r1, c1))
            if side is None:
                continue

            inner = grid[r0 + 1 : r1, c0 + 1 : c1] != bg
            inner_comps = _components4(inner)
            if len(inner_comps) != 1:
                continue
            pattern = inner_comps[0].copy()
            pattern[:, 0] += r0 + 1
            pattern[:, 1] += c0 + 1
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "pattern": pattern,
                }
            )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _translation_to_outside(
    pattern: np.ndarray, bbox: tuple[int, int, int, int], side: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - 1 - int(pattern[:, 0].min()), 0
    if side == "bottom":
        return r1 + 1 - int(pattern[:, 0].max()), 0
    if side == "left":
        return 0, c0 - 1 - int(pattern[:, 1].min())
    return 0, c1 + 1 - int(pattern[:, 1].max())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each interior pattern straight outside its hollow frame in the direction indicated by the frame gap.

    Input structure:
    - The background is black.
    - The grid contains several colored hollow rectangular frames; each frame has exactly one 1-pixel gap on one side.
    - Each frame encloses one connected non-background pattern that stays strictly inside the frame.

    Rule:
    - Detect every hollow rectangular frame and find which side contains its single missing border pixel.
    - Remove the enclosed pattern from its original location and translate the whole pattern toward that side.
    - The translation stops when the pattern is completely outside the frame and directly adjacent to the outer wall; keep the frame, including its gap, unchanged.

    Key insight:
    - The border gap is a direction marker rather than damage: it tells which side the inside pattern should be poured out through.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    moves = []
    for frame in _find_frames(g, bg):
        pattern = frame["pattern"]
        out[pattern[:, 0], pattern[:, 1]] = bg
        dy, dx = _translation_to_outside(pattern, frame["bbox"], frame["side"])
        moves.append((pattern, dy, dx))

    for pattern, dy, dx in moves:
        rr = pattern[:, 0] + dy
        cc = pattern[:, 1] + dx
        valid = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
        src = pattern[valid]
        out[rr[valid], cc[valid]] = g[src[:, 0], src[:, 1]]

    return out


def _grow_pattern(box_h: int, box_w: int) -> list[tuple[int, int]]:
    target_area = random.randint(1, min(box_h * box_w, box_h + box_w))
    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return norm


def _frame_cells(frame_h: int, frame_w: int, side: str, gap_idx: int) -> list[tuple[int, int]]:
    gap = {
        "top": (0, gap_idx),
        "bottom": (frame_h - 1, gap_idx),
        "left": (gap_idx, 0),
        "right": (gap_idx, frame_w - 1),
    }[side]
    cells = []
    for r in range(frame_h):
        for c in range(frame_w):
            if (r in (0, frame_h - 1) or c in (0, frame_w - 1)) and (r, c) != gap:
                cells.append((r, c))
    return cells


def _mark_occupied(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        mask[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _abs_cells(top: int, left: int, rel_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in rel_cells]


def _build_object(
    frame_h: int,
    frame_w: int,
    side: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    max_box_h = min(PATTERN_BOX_MAX, frame_h - 4)
    max_box_w = min(PATTERN_BOX_MAX, frame_w - 4)
    if max_box_h < 1 or max_box_w < 1:
        return None

    shape = _grow_pattern(random.randint(1, max_box_h), random.randint(1, max_box_w))
    ph = max(r for r, _ in shape) + 1
    pw = max(c for _, c in shape) + 1
    top_min, top_max = 2, frame_h - ph - 2
    left_min, left_max = 2, frame_w - pw - 2
    if top_min > top_max or left_min > left_max:
        return None

    pr = random.randint(top_min, top_max)
    pc = random.randint(left_min, left_max)
    gap_idx = random.randint(1, frame_w - 2) if side in ("top", "bottom") else random.randint(1, frame_h - 2)

    frame_rel = _frame_cells(frame_h, frame_w, side, gap_idx)
    pattern_rel = [(pr + r, pc + c) for r, c in shape]
    pattern_arr = np.array(pattern_rel, dtype=int)
    dy, dx = _translation_to_outside(pattern_arr, (0, 0, frame_h - 1, frame_w - 1), side)
    moved_rel = [(r + dy, c + dx) for r, c in pattern_rel]
    return frame_rel, pattern_rel, moved_rel


def _place_object(
    inp: np.ndarray,
    out: np.ndarray,
    occ_in: np.ndarray,
    occ_out: np.ndarray,
    frame_color: int,
    pattern_color: int,
    side: str,
) -> bool:
    h, w = inp.shape
    for _ in range(OBJECT_RETRIES):
        frame_h = random.randint(*FRAME_H_RANGE)
        frame_w = random.randint(*FRAME_W_RANGE)
        built = _build_object(frame_h, frame_w, side)
        if built is None:
            continue
        frame_rel, pattern_rel, moved_rel = built
        all_rel = frame_rel + pattern_rel + moved_rel
        min_r = min(r for r, _ in all_rel)
        max_r = max(r for r, _ in all_rel)
        min_c = min(c for _, c in all_rel)
        max_c = max(c for _, c in all_rel)
        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            continue

        for _ in range(POSITION_RETRIES):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            in_frame = _abs_cells(top, left, frame_rel)
            in_pattern = _abs_cells(top, left, pattern_rel)
            out_pattern = _abs_cells(top, left, moved_rel)
            in_cells = in_frame + in_pattern
            out_cells = in_frame + out_pattern
            if any(occ_in[r, c] for r, c in in_cells):
                continue
            if any(occ_out[r, c] for r, c in out_cells):
                continue

            for r, c in in_frame:
                inp[r, c] = frame_color
                out[r, c] = frame_color
            for r, c in in_pattern:
                inp[r, c] = pattern_color
            for r, c in out_pattern:
                out[r, c] = pattern_color

            _mark_occupied(occ_in, in_cells)
            _mark_occupied(occ_out, out_cells)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    num_objects = random.randint(*OBJECT_COUNT_RANGE)
    min_side = max(GRID_MIN, 14 + 2 * num_objects)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occ_in = np.zeros((h, w), dtype=bool)
    occ_out = np.zeros((h, w), dtype=bool)

    palette = COLORS[:]
    random.shuffle(palette)
    frame_colors = palette[:num_objects]
    pattern_colors = palette[num_objects : 2 * num_objects]
    sides = random.sample(SIDES, k=4)[:num_objects]

    for frame_color, pattern_color, side in zip(frame_colors, pattern_colors, sides):
        if not _place_object(inp, out, occ_in, occ_out, frame_color, pattern_color, side):
            return None

    if len(_find_frames(inp, BG)) != num_objects:
        return None
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
