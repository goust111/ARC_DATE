from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIRS = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r1, c) for c in range(c0, c1 + 1)]
    coords += [(r, c0) for r in range(r0 + 1, r1)]
    coords += [(r, c1) for r in range(r0 + 1, r1)]
    return coords


def _side_of(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = cell
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            pts = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _run_start(g: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> tuple[int, int]:
    h, w = g.shape
    while 0 <= r - dr < h and 0 <= c - dc < w and int(g[r - dr, c - dc]) == color:
        r -= dr
        c -= dc
    return r, c


def _run_end(g: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> tuple[int, int]:
    h, w = g.shape
    while 0 <= r + dr < h and 0 <= c + dc < w and int(g[r + dr, c + dc]) == color:
        r += dr
        c += dc
    return r, c


def _validate_frame(g: np.ndarray, bbox: tuple[int, int, int, int]) -> dict | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    border = _border_coords(r0, c0, r1, c1)
    vals = [int(g[r, c]) for r, c in border]
    counts = Counter(vals)
    if counts.get(BG, 0) != 1:
        return None
    non_bg = [(col, cnt) for col, cnt in counts.items() if col != BG]
    if len(non_bg) != 2:
        return None

    perim = len(border)
    frame_options = [col for col, cnt in non_bg if cnt == perim - 2]
    marker_options = [col for col, cnt in non_bg if cnt == 1]
    if len(frame_options) != 1 or len(marker_options) != 1:
        return None

    frame_color = int(frame_options[0])
    marker_color = int(marker_options[0])
    gap_cells = [cell for cell in border if int(g[cell]) == BG]
    marker_cells = [cell for cell in border if int(g[cell]) == marker_color]
    if len(gap_cells) != 1 or len(marker_cells) != 1:
        return None

    gap_side = _side_of(gap_cells[0], bbox)
    marker_side = _side_of(marker_cells[0], bbox)
    if gap_side is None or marker_side is None:
        return None
    if marker_side != OPPOSITE[gap_side]:
        return None

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_colors = set(map(int, np.unique(interior))) - {BG}
    if len(inner_colors) != 1:
        return None
    pattern_color = int(next(iter(inner_colors)))
    if pattern_color in (frame_color, marker_color):
        return None

    pattern_mask = interior == pattern_color
    if (
        bool(pattern_mask[0, :].any())
        or bool(pattern_mask[-1, :].any())
        or bool(pattern_mask[:, 0].any())
        or bool(pattern_mask[:, -1].any())
    ):
        return None
    comps = _components(pattern_mask)
    if len(comps) != 1:
        return None

    rel = comps[0]
    abs_coords = np.column_stack((rel[:, 0] + r0 + 1, rel[:, 1] + c0 + 1))
    return {
        "bbox": bbox,
        "gap_side": gap_side,
        "marker_color": marker_color,
        "pattern": abs_coords,
    }


def _find_frames(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    by_bbox: dict[tuple[int, int, int, int], dict] = {}

    for gr, gc in np.argwhere(g == BG):
        gr = int(gr)
        gc = int(gc)

        if 0 < gc < w - 1:
            left = int(g[gr, gc - 1])
            right = int(g[gr, gc + 1])
            if left != BG and left == right:
                _r, c0 = _run_end(g, gr, gc - 1, 0, -1, left)
                _r, c1 = _run_end(g, gr, gc + 1, 0, 1, left)
                if c0 < gc < c1 and c1 - c0 + 1 >= 5:
                    for r1 in range(gr + 4, h):
                        if int(g[r1, c0]) == left and int(g[r1, c1]) == left:
                            frame = _validate_frame(g, (gr, c0, r1, c1))
                            if frame is not None:
                                by_bbox[frame["bbox"]] = frame
                    for r0 in range(0, gr - 3):
                        if int(g[r0, c0]) == left and int(g[r0, c1]) == left:
                            frame = _validate_frame(g, (r0, c0, gr, c1))
                            if frame is not None:
                                by_bbox[frame["bbox"]] = frame

        if 0 < gr < h - 1:
            up = int(g[gr - 1, gc])
            down = int(g[gr + 1, gc])
            if up != BG and up == down:
                r0, _c = _run_end(g, gr - 1, gc, -1, 0, up)
                r1, _c = _run_end(g, gr + 1, gc, 1, 0, up)
                if r0 < gr < r1 and r1 - r0 + 1 >= 5:
                    for c1 in range(gc + 4, w):
                        if int(g[r0, c1]) == up and int(g[r1, c1]) == up:
                            frame = _validate_frame(g, (r0, gc, r1, c1))
                            if frame is not None:
                                by_bbox[frame["bbox"]] = frame
                    for c0 in range(0, gc - 3):
                        if int(g[r0, c0]) == up and int(g[r1, c0]) == up:
                            frame = _validate_frame(g, (r0, c0, r1, gc))
                            if frame is not None:
                                by_bbox[frame["bbox"]] = frame

    frames = [by_bbox[bbox] for bbox in sorted(by_bbox)]
    assert frames, "expected at least one gapped marked frame"
    seen_bboxes = [f["bbox"] for f in frames]
    assert len(seen_bboxes) == len(set(seen_bboxes)), "frame boxes must be unique"
    for i, a in enumerate(frames):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for b in frames[i + 1 :]:
            br0, bc0, br1, bc1 = b["bbox"]
            disjoint = ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0
            assert disjoint, "frames must not overlap"
    return frames


def _target_coords(pattern: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    out = pattern.copy()
    if side == "top":
        out[:, 0] += (r0 - 1) - int(rows.max())
    elif side == "bottom":
        out[:, 0] += (r1 + 1) - int(rows.min())
    elif side == "left":
        out[:, 1] += (c0 - 1) - int(cols.max())
    else:
        out[:, 1] += (c1 + 1) - int(cols.min())
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each framed interior pattern through the frame's gap and recolor it with the opposite-edge marker.

    Input structure:
    - A black background contains one or more one-pixel rectangular hollow frames.
    - Each frame has exactly one black gap pixel on one side and one differently colored marker pixel on the opposite side.
    - Each frame interior contains exactly one connected monochrome pattern that does not touch the frame.

    Rule:
    - For every frame, use the side containing the gap as the ejection direction.
    - Clear the original interior pattern, then translate that pattern outward until its rear edge is one cell beyond the frame border.
    - Paint the translated pattern with the frame's marker color while leaving all frame border, gap, and marker pixels unchanged.

    Key insight:
    The gap gives the movement direction, but the recoloring source is the lone off-color marker on the opposite side of the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    frames = _find_frames(g)
    tasks = []
    for frame in frames:
        target = _target_coords(frame["pattern"], frame["bbox"], frame["gap_side"])
        assert np.all((0 <= target[:, 0]) & (target[:, 0] < h) & (0 <= target[:, 1]) & (target[:, 1] < w))
        for r, c in target:
            assert int(g[int(r), int(c)]) == BG, "ejected pattern must land on background"
        tasks.append((frame["pattern"], target, int(frame["marker_color"])))

    out = g.copy()
    for pattern, _target, _color in tasks:
        for r, c in pattern:
            out[int(r), int(c)] = BG

    occupied_targets: set[tuple[int, int]] = set()
    for _pattern, target, color in tasks:
        for r, c in target:
            cell = (int(r), int(c))
            assert cell not in occupied_targets, "ejected patterns must not overlap"
            assert int(out[cell]) == BG, "ejected pattern must not cover existing structure"
            occupied_targets.add(cell)
            out[cell] = int(color)
    return out


# ---------------- generator ----------------


def _random_polyomino(max_h: int, max_w: int) -> np.ndarray | None:
    area_max = min(7, max_h * max_w)
    if area_max < 2:
        return None
    target = random.randint(2, area_max)
    start = (random.randrange(max_h), random.randrange(max_w))
    cells = {start}
    frontier = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < max_h and 0 <= nc < max_w:
            frontier.add((nr, nc))

    while len(cells) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.discard(cell)
        cells.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    if len(cells) != target:
        return None

    rs = [r for r, _c in cells]
    cs = [c for _r, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _make_object(direction: str) -> dict | None:
    fh = random.randint(6, 10)
    fw = random.randint(6, 10)
    safe_h = fh - 4
    safe_w = fw - 4
    shape = _random_polyomino(safe_h, safe_w)
    if shape is None:
        return None
    sh, sw = shape.shape
    pr = 2 + random.randint(0, safe_h - sh)
    pc = 2 + random.randint(0, safe_w - sw)
    pattern_cells = [(pr + int(r), pc + int(c)) for r, c in np.argwhere(shape)]

    frame_color, pattern_color, marker_color = random.sample(range(1, 10), 3)
    border = {(0, c): frame_color for c in range(fw)}
    border.update({(fh - 1, c): frame_color for c in range(fw)})
    border.update({(r, 0): frame_color for r in range(1, fh - 1)})
    border.update({(r, fw - 1): frame_color for r in range(1, fh - 1)})

    if direction == "top":
        gap = (0, random.choice([c for _r, c in pattern_cells]))
        marker = (fh - 1, random.randint(1, fw - 2))
    elif direction == "bottom":
        gap = (fh - 1, random.choice([c for _r, c in pattern_cells]))
        marker = (0, random.randint(1, fw - 2))
    elif direction == "left":
        gap = (random.choice([r for r, _c in pattern_cells]), 0)
        marker = (random.randint(1, fh - 2), fw - 1)
    else:
        gap = (random.choice([r for r, _c in pattern_cells]), fw - 1)
        marker = (random.randint(1, fh - 2), 0)

    border.pop(gap)
    border[marker] = marker_color

    input_cells = dict(border)
    for cell in pattern_cells:
        input_cells[cell] = pattern_color

    pattern_np = np.array(pattern_cells, dtype=int)
    target_np = _target_coords(pattern_np, (0, 0, fh - 1, fw - 1), direction)
    target_cells = [(int(r), int(c)) for r, c in target_np]
    if set(target_cells) & set(input_cells):
        return None

    return {
        "input": input_cells,
        "pattern": pattern_cells,
        "target": target_cells,
        "target_color": marker_color,
    }


def _expanded(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _place_object(
    grid: np.ndarray,
    expected: np.ndarray,
    obj: dict,
    reserved: set[tuple[int, int]],
    padded: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    local_all = set(obj["input"]) | set(obj["target"])
    min_r = min(r for r, _c in local_all)
    max_r = max(r for r, _c in local_all)
    min_c = min(c for _r, c in local_all)
    max_c = max(c for _r, c in local_all)

    row_lo, row_hi = -min_r, h - 1 - max_r
    col_lo, col_hi = -min_c, w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return False

    positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(positions)
    for r0, c0 in positions[:300]:
        input_abs = {(r0 + r, c0 + c) for r, c in obj["input"]}
        target_abs = {(r0 + r, c0 + c) for r, c in obj["target"]}
        new_all = input_abs | target_abs
        if input_abs & target_abs:
            continue
        if new_all & padded:
            continue
        if new_all & reserved:
            continue

        for (lr, lc), color in obj["input"].items():
            grid[r0 + lr, c0 + lc] = int(color)
            expected[r0 + lr, c0 + lc] = int(color)
        for lr, lc in obj["pattern"]:
            expected[r0 + lr, c0 + lc] = BG
        for lr, lc in obj["target"]:
            expected[r0 + lr, c0 + lc] = int(obj["target_color"])

        reserved.update(new_all)
        padded.update(_expanded(new_all, h, w, pad=1))
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(1, 4)
    lo = 14 if n_objects <= 2 else 18
    h = random.randint(lo, 30)
    w = random.randint(lo, 30)
    grid = np.full((h, w), BG, dtype=int)
    expected = grid.copy()

    dirs = list(DIRS)
    random.shuffle(dirs)
    while len(dirs) < n_objects:
        extra = list(DIRS)
        random.shuffle(extra)
        dirs.extend(extra)
    dirs = dirs[:n_objects]

    reserved: set[tuple[int, int]] = set()
    padded: set[tuple[int, int]] = set()
    for direction in dirs:
        placed = False
        for _ in range(80):
            obj = _make_object(direction)
            if obj is not None and _place_object(grid, expected, obj, reserved, padded):
                placed = True
                break
        if not placed:
            return None

    if max(grid.shape) > 30:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-frame ejection input")
