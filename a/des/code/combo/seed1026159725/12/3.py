import random
from collections import deque

import numpy as np


RED = 2
BLUE = 1
GREEN = 3


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    coord_set = set(coords)
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    return True


def _find_panels(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in map(int, np.unique(grid)):
        boxes = []
        for comp in _components(grid == color):
            if _is_rect_frame(comp):
                boxes.append(_bbox(comp))
        if len(boxes) != 9:
            continue
        heights = {r1 - r0 + 1 for r0, c0, r1, c1 in boxes}
        widths = {c1 - c0 + 1 for r0, c0, r1, c1 in boxes}
        row_starts = sorted({r0 for r0, c0, r1, c1 in boxes})
        col_starts = sorted({c0 for r0, c0, r1, c1 in boxes})
        if len(heights) != 1 or len(widths) != 1:
            continue
        if len(row_starts) != 3 or len(col_starts) != 3:
            continue
        expected = {(r, c) for r in row_starts for c in col_starts}
        actual = {(r0, c0) for r0, c0, r1, c1 in boxes}
        if actual != expected:
            continue
        candidates.append((color, sorted(boxes)))
    assert len(candidates) == 1, "expected exactly one 3x3 panel-frame color"
    return candidates[0]


def _has_hole(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any((~mask) & (~seen)))


def _paint_frame(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each panel object as small-solid, large-solid, or hollow, then recolor
    that panel's border red, blue, or green respectively.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    _frame_color, panels = _find_panels(g)

    out = g.copy()
    for box in panels:
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        obj_colors = set(map(int, np.unique(interior))) - {bg}
        assert len(obj_colors) == 1, "each panel must contain exactly one object color"
        obj_color = obj_colors.pop()
        mask = interior == obj_color
        assert len(_components(mask)) == 1, "panel object must be 4-connected"
        area = int(mask.sum())
        border_color = GREEN if _has_hole(mask) else (RED if area <= 5 else BLUE)
        _paint_frame(out, box, border_color)

    return out


GEN_ATTEMPTS = 400
OBJECT_COLOR = 5
RESERVED = {RED, BLUE, GREEN, OBJECT_COLOR}
FREE_COLORS = [0, 4, 6, 7, 8, 9]


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_polyomino(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (int(r + dr), int(c + dc))
                    if nxt not in cells:
                        frontier.add(nxt)
            frontier_list = list(frontier)
            random.shuffle(frontier_list)
            placed = False
            for cand in frontier_list:
                test_cells = set(cells)
                test_cells.add(cand)
                norm = _normalize_cells(test_cells)
                h = max(r for r, _ in norm) + 1
                w = max(c for _, c in norm) + 1
                if h <= max_h and w <= max_w:
                    cells = norm
                    placed = True
                    break
            if not placed:
                break
        if len(cells) == area:
            return _cells_to_mask(cells)
    return None


def _random_small_solid(max_h: int, max_w: int) -> np.ndarray | None:
    area = random.randint(1, 5)
    return _random_polyomino(area, max_h, max_w)


def _random_large_solid(max_h: int, max_w: int) -> np.ndarray | None:
    max_area = min(12, max_h * max_w)
    if max_area < 6:
        return None
    for _ in range(200):
        area = random.randint(6, max_area)
        mask = _random_polyomino(area, max_h, max_w)
        if mask is not None and not _has_hole(mask):
            return mask
    return None


def _random_hollow(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 4 or max_w < 4:
        return None
    for _ in range(200):
        outer_h = random.randint(4, min(max_h, 7))
        outer_w = random.randint(4, min(max_w, 7))
        hole_h = random.randint(1, outer_h - 2)
        hole_w = random.randint(1, outer_w - 2)
        hole_r = random.randint(1, outer_h - hole_h - 1)
        hole_c = random.randint(1, outer_w - hole_w - 1)
        mask = np.ones((outer_h, outer_w), dtype=bool)
        mask[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = False
        if _has_hole(mask):
            return mask
    return None


def _build_object(kind: str, max_h: int, max_w: int) -> np.ndarray | None:
    if kind == "small":
        return _random_small_solid(max_h, max_w)
    if kind == "large":
        return _random_large_solid(max_h, max_w)
    if kind == "hollow":
        return _random_hollow(max_h, max_w)
    raise ValueError(kind)


def _place_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = map(int, mask.shape)
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(FREE_COLORS)
    frame_color = random.choice([c for c in FREE_COLORS if c != bg])

    inner_h = random.randint(7, 10)
    inner_w = random.randint(7, 10)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    gap_h = random.randint(1, 3)
    gap_w = random.randint(1, 3)
    margin_top = random.randint(0, 2)
    margin_left = random.randint(0, 2)
    margin_bottom = random.randint(0, 2)
    margin_right = random.randint(0, 2)

    grid_h = margin_top + 3 * outer_h + 2 * gap_h + margin_bottom
    grid_w = margin_left + 3 * outer_w + 2 * gap_w + margin_right
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    panel_boxes: list[tuple[int, int, int, int]] = []
    for pr in range(3):
        for pc in range(3):
            r0 = margin_top + pr * (outer_h + gap_h)
            c0 = margin_left + pc * (outer_w + gap_w)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            _paint_frame(grid, (r0, c0, r1, c1), frame_color)
            panel_boxes.append((r0, c0, r1, c1))

    kinds = ["small"] * 3 + ["large"] * 3 + ["hollow"] * 3
    random.shuffle(kinds)

    slot_h = inner_h - 2
    slot_w = inner_w - 2
    for box, kind in zip(panel_boxes, kinds):
        mask = _build_object(kind, slot_h, slot_w)
        if mask is None:
            return None
        mh, mw = map(int, mask.shape)
        r0, c0, r1, c1 = box
        top = random.randint(1, inner_h - mh - 1)
        left = random.randint(1, inner_w - mw - 1)
        _place_mask(grid, r0 + 1 + top, c0 + 1 + left, mask, OBJECT_COLOR)

    try:
        out = main(grid)
    except AssertionError:
        return None

    border_colors = {int(x) for x in np.unique(out)} & {RED, BLUE, GREEN}
    if border_colors != {RED, BLUE, GREEN}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
