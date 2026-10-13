import random

import numpy as np


BG = 0
PATTERN_COLOR = 4
FRAME_COLORS = [1, 2, 3, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")

GEN_ATTEMPTS = 400
LAYER_OPTIONS = [2, 3, 4]
MAX_DIM = 30

BASE_PATTERNS = [
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
]


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _gap_side(missing: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> str | None:
    missing = sorted(missing)
    if len(missing) != 2:
        return None
    (r0, c0, r1, c1) = bbox
    (a0, b0), (a1, b1) = missing
    if a0 == a1 == r0 and b1 == b0 + 1:
        return "top"
    if a0 == a1 == r1 and b1 == b0 + 1:
        return "bottom"
    if b0 == b1 == c0 and a1 == a0 + 1:
        return "left"
    if b0 == b1 == c1 and a1 == a0 + 1:
        return "right"
    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    colors = [int(v) for v in np.unique(grid) if int(v) != int(bg)]
    for color in colors:
        for pts in _components(grid == int(color)):
            if len(pts) < 10:
                continue
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            border = _perimeter_cells(r0, c0, r1, c1)
            ptset = set(pts)
            if not ptset.issubset(border):
                continue
            missing = list(border - ptset)
            if len(missing) != 2 or len(ptset) != len(border) - 2:
                continue
            side = _gap_side(missing, (r0, c0, r1, c1))
            if side is None:
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "gap_side": side,
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
    frames.sort(key=lambda frame: frame["area"])
    return frames


def _shift_for_frame(
    frame: dict,
    outer: dict | None,
    shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    h, w = shape
    side = frame["gap_side"]
    if side == "top":
        dist = r0 if outer is None else r0 - outer["bbox"][0] - 1
        return -int(dist), 0
    if side == "bottom":
        dist = (h - 1 - r1) if outer is None else outer["bbox"][2] - r1 - 1
        return int(dist), 0
    if side == "left":
        dist = c0 if outer is None else c0 - outer["bbox"][1] - 1
        return 0, -int(dist)
    dist = (w - 1 - c1) if outer is None else outer["bbox"][3] - c1 - 1
    return 0, int(dist)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour the nested framed contents outward one layer at a time using each frame's two-cell gap as a direction cue.

    Input structure:
    - The background is black.
    - Several one-pixel-thick monochrome rectangular frames are nested inside one another.
    - Each frame is missing exactly two consecutive border cells on one side, creating a gap on the top, bottom, left, or right edge.
    - The innermost frame contains a small solid pattern, and outer frame interiors may also contain inner frames.

    Rule:
    - Detect the nested frames and process them from the smallest frame to the largest.
    - For the current frame, collect every non-background cell strictly inside that frame.
    - Translate that whole interior content in the direction indicated by the frame's gap.
    - The shift amount equals the empty spacing between this frame and the next outer frame on that side, or the spacing from the outermost frame to the grid boundary.
    - Leave the current frame itself unchanged and clear its interior after the content has been moved outward.

    Key insight:
    - The two-cell opening is not content to copy; it encodes a movement vector, and the final position is the cumulative sum of those per-layer translations.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frames = _find_frames(g, bg)
    if not frames:
        return g.copy()

    out = g.copy()
    for i, frame in enumerate(frames):
        outer = frames[i + 1] if i + 1 < len(frames) else None
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        rel = np.argwhere(inner != int(bg))
        if rel.size == 0:
            continue

        moved: list[tuple[int, int, int]] = []
        for rr, cc in rel:
            ar = int(r0 + 1 + rr)
            ac = int(c0 + 1 + cc)
            moved.append((ar, ac, int(out[ar, ac])))

        out[r0 + 1 : r1, c0 + 1 : c1][inner != int(bg)] = int(bg)
        dr, dc = _shift_for_frame(frame, outer, g.shape)
        for ar, ac, color in moved:
            nr = ar + dr
            nc = ac + dc
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                out[nr, nc] = int(color)

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _transform_pattern(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(cells)
    for _ in range(random.randrange(4)):
        pts = {(c, -r) for r, c in pts}
    if random.random() < 0.5:
        pts = {(r, -c) for r, c in pts}
    if random.random() < 0.5:
        pts = {(-r, c) for r, c in pts}
    return _normalize(pts)


def _shift_map(cells: dict[tuple[int, int], int], dr: int, dc: int) -> dict[tuple[int, int], int]:
    return {(r + dr, c + dc): int(color) for (r, c), color in cells.items()}


def _gap_cells(r0: int, c0: int, r1: int, c1: int, side: str) -> set[tuple[int, int]]:
    if side in ("top", "bottom"):
        start = random.randint(c0 + 1, c1 - 2)
        row = r0 if side == "top" else r1
        return {(row, start), (row, start + 1)}
    start = random.randint(r0 + 1, r1 - 2)
    col = c0 if side == "left" else c1
    return {(start, col), (start + 1, col)}


def _random_pattern() -> set[tuple[int, int]]:
    return _transform_pattern(set(random.choice(BASE_PATTERNS)))


def _valid_candidate(inp: np.ndarray, layer_count: int, outer_bbox: tuple[int, int, int, int]) -> bool:
    frames = _find_frames(inp, BG)
    if len(frames) != layer_count:
        return False
    out = main(inp)
    if np.array_equal(out, inp):
        return False
    r0, c0, r1, c1 = outer_bbox
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    return bool(np.any((out != BG) & outside))


def _generate_candidate() -> np.ndarray | None:
    layer_count = random.choice(LAYER_OPTIONS)
    frame_colors = random.sample(FRAME_COLORS, layer_count)

    pattern = _random_pattern()
    cells: dict[tuple[int, int], int] = {coord: PATTERN_COLOR for coord in pattern}
    h = max(r for r, _c in pattern) + 1
    w = max(c for _r, c in pattern) + 1

    for idx in range(layer_count):
        margin_hi = 2 if layer_count == 4 else 3
        mt = random.randint(1, margin_hi)
        mb = random.randint(1, margin_hi)
        ml = random.randint(1, margin_hi)
        mr = random.randint(1, margin_hi)

        cells = _shift_map(cells, mt + 1, ml + 1)
        new_h = h + mt + mb + 2
        new_w = w + ml + mr + 2
        if new_h < 5 or new_w < 5:
            return None

        side = random.choice(SIDES)
        r0, c0, r1, c1 = 0, 0, new_h - 1, new_w - 1
        gap = _gap_cells(r0, c0, r1, c1, side)
        for rr, cc in _perimeter_cells(r0, c0, r1, c1):
            if (rr, cc) not in gap:
                cells[(rr, cc)] = frame_colors[idx]
        h, w = new_h, new_w

    if h + 2 > MAX_DIM or w + 2 > MAX_DIM:
        return None

    extra_h_max = min(8, MAX_DIM - h)
    extra_w_max = min(8, MAX_DIM - w)
    if extra_h_max < 2 or extra_w_max < 2:
        return None

    extra_h = random.randint(2, extra_h_max)
    extra_w = random.randint(2, extra_w_max)
    pad_top = random.randint(1, extra_h - 1)
    pad_bottom = extra_h - pad_top
    pad_left = random.randint(1, extra_w - 1)
    pad_right = extra_w - pad_left

    grid = np.full((h + pad_top + pad_bottom, w + pad_left + pad_right), BG, dtype=int)
    shifted = _shift_map(cells, pad_top, pad_left)
    for (r, c), color in shifted.items():
        grid[r, c] = int(color)

    outer_bbox = (pad_top, pad_left, pad_top + h - 1, pad_left + w - 1)
    if not _valid_candidate(grid, layer_count, outer_bbox):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
