import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _pattern_kind(pattern: Sequence[int], color: int, bg: int) -> Optional[str]:
    a, b, c = map(int, pattern)
    if (a, b, c) == (color, color, color):
        return "solid"
    if (a, b, c) == (color, bg, color):
        return "dashed"
    return None


def _detect_legend(grid: np.ndarray, bg: int) -> Tuple[Dict[int, str], np.ndarray]:
    h, w = map(int, grid.shape)
    candidates: List[Tuple[int, Dict[int, str], np.ndarray]] = []

    if w >= 4:
        entries: Dict[int, str] = {}
        mask = np.zeros_like(grid, dtype=bool)
        rows = []
        for r in range(h):
            color = int(grid[r, 0])
            if color == bg:
                continue
            kind = _pattern_kind(grid[r, 1:4], color, bg)
            if kind is None:
                continue
            entries[color] = kind
            mask[r, 0:4] = True
            rows.append(r)
        if len(entries) >= 2 and rows:
            candidates.append((len(entries), entries, mask))

        entries = {}
        mask = np.zeros_like(grid, dtype=bool)
        rows = []
        for r in range(h):
            color = int(grid[r, w - 4])
            if color == bg:
                continue
            kind = _pattern_kind(grid[r, w - 3 : w], color, bg)
            if kind is None:
                continue
            entries[color] = kind
            mask[r, w - 4 : w] = True
            rows.append(r)
        if len(entries) >= 2 and rows:
            candidates.append((len(entries), entries, mask))

    if h >= 4:
        entries = {}
        mask = np.zeros_like(grid, dtype=bool)
        cols = []
        for c in range(w):
            color = int(grid[0, c])
            if color == bg:
                continue
            kind = _pattern_kind(grid[1:4, c], color, bg)
            if kind is None:
                continue
            entries[color] = kind
            mask[0:4, c] = True
            cols.append(c)
        if len(entries) >= 2 and cols:
            candidates.append((len(entries), entries, mask))

        entries = {}
        mask = np.zeros_like(grid, dtype=bool)
        cols = []
        for c in range(w):
            color = int(grid[h - 4, c])
            if color == bg:
                continue
            kind = _pattern_kind(grid[h - 3 : h, c], color, bg)
            if kind is None:
                continue
            entries[color] = kind
            mask[h - 4 : h, c] = True
            cols.append(c)
        if len(entries) >= 2 and cols:
            candidates.append((len(entries), entries, mask))

    assert candidates, "legend not found"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _size, mapping, legend_mask = candidates[0]
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    mapping, legend_mask = _detect_legend(g, bg)
    out = g.copy()

    for color, kind in mapping.items():
        coords = np.argwhere((g == int(color)) & ~legend_mask)
        assert coords.shape == (2, 2), f"expected exactly one pair for color {color}"
        (r0, c0), (r1, c1) = coords.tolist()

        if r0 == r1:
            c_start, c_end = sorted((int(c0), int(c1)))
            step = 1 if kind == "solid" else 2
            for c in range(c_start + 1, c_end, step):
                out[r0, c] = int(color)
        else:
            assert c0 == c1, f"pair for color {color} must share a row or column"
            r_start, r_end = sorted((int(r0), int(r1)))
            step = 1 if kind == "solid" else 2
            for r in range(r_start + 1, r_end, step):
                out[r, c0] = int(color)

    return out


def _legend_footprint(side: str, count: int, h: int, w: int) -> Tuple[Tuple[int, int, int, int], List[Tuple[int, int]]]:
    if side == "left":
        start = random.randint(1, h - count - 1)
        return (start, 0, start + count - 1, 3), [(start + i, 0) for i in range(count)]
    if side == "right":
        start = random.randint(1, h - count - 1)
        return (start, w - 4, start + count - 1, w - 1), [(start + i, w - 4) for i in range(count)]
    if side == "top":
        start = random.randint(1, w - count - 1)
        return (0, start, 3, start + count - 1), [(0, start + i) for i in range(count)]
    start = random.randint(1, w - count - 1)
    return (h - 4, start, h - 1, start + count - 1), [(h - 4, start + i) for i in range(count)]


def _paint_legend(grid: np.ndarray, side: str, anchors: List[Tuple[int, int]], colors: Sequence[int], styles: Dict[int, str]) -> None:
    for (r, c), color in zip(anchors, colors):
        if side in ("left", "right"):
            grid[r, c] = color
            grid[r, c + 1] = color
            grid[r, c + 2] = BG if styles[color] == "dashed" else color
            grid[r, c + 3] = color
        else:
            grid[r, c] = color
            grid[r + 1, c] = color
            grid[r + 2, c] = BG if styles[color] == "dashed" else color
            grid[r + 3, c] = color


def _with_moat(mask: np.ndarray, cells: Sequence[Tuple[int, int]]) -> bool:
    h, w = mask.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        if mask[r0:r1, c0:c1].any():
            return False
    return True


def _reserve(mask: np.ndarray, cells: Sequence[Tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _segment_cells(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    cells = [a, b]
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        cells.extend((r0, c) for c in range(lo + 1, hi))
    else:
        lo, hi = sorted((r0, r1))
        cells.extend((r, c0) for r in range(lo + 1, hi))
    return cells


def _sample_pair(mask: np.ndarray, h: int, w: int, orientation: str, min_gap: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if orientation == "h":
        rows = list(range(1, h - 1))
        cols = list(range(1, w - 1))
        random.shuffle(rows)
        for r in rows:
            for _ in range(30):
                c0, c1 = sorted(random.sample(cols, 2))
                if c1 - c0 < min_gap:
                    continue
                cells = _segment_cells((r, c0), (r, c1))
                if _with_moat(mask, cells):
                    return (r, c0), (r, c1)
        return None

    cols = list(range(1, w - 1))
    rows = list(range(1, h - 1))
    random.shuffle(cols)
    for c in cols:
        for _ in range(30):
            r0, r1 = sorted(random.sample(rows, 2))
            if r1 - r0 < min_gap:
                continue
            cells = _segment_cells((r0, c), (r1, c))
            if _with_moat(mask, cells):
                return (r0, c), (r1, c)
    return None


def _style_assignment(colors: Sequence[int]) -> Dict[int, str]:
    shuffled = list(colors)
    random.shuffle(shuffled)
    styles: Dict[int, str] = {}
    if len(shuffled) == 2:
        styles[shuffled[0]] = "solid"
        styles[shuffled[1]] = "dashed"
        return styles

    split = len(shuffled) // 2
    solid_set = set(shuffled[:split])
    if not solid_set:
        solid_set.add(shuffled[0])
    if len(solid_set) == len(shuffled):
        solid_set.remove(shuffled[-1])
    for color in shuffled:
        styles[color] = "solid" if color in solid_set else "dashed"
    return styles


def _orientation_assignment(count: int) -> List[str]:
    if count == 2:
        orientations = ["h", "v"]
    else:
        half = count // 2
        orientations = ["h"] * half + ["v"] * (count - half)
    random.shuffle(orientations)
    return orientations


def _gen_attempt() -> Optional[np.ndarray]:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    count = random.randint(2, min(5, len(PALETTE_COLORS), h - 3, w - 3))
    side = random.choice(["left", "right", "top", "bottom"])

    if side in ("left", "right") and h < count + 3:
        return None
    if side in ("top", "bottom") and w < count + 3:
        return None

    colors = random.sample(PALETTE_COLORS, count)
    styles = _style_assignment(colors)
    orientations = _orientation_assignment(count)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    (r0, c0, r1, c1), anchors = _legend_footprint(side, count, h, w)
    for rr in range(max(0, r0 - 1), min(h, r1 + 2)):
        for cc in range(max(0, c0 - 1), min(w, c1 + 2)):
            occupied[rr, cc] = True
    _paint_legend(grid, side, anchors, colors, styles)

    for color, orientation in zip(colors, orientations):
        min_gap = random.randint(4, 8)
        pair = _sample_pair(occupied, h, w, orientation, min_gap)
        if pair is None:
            return None
        a, b = pair
        segment = _segment_cells(a, b)
        _reserve(occupied, segment)
        grid[a] = color
        grid[b] = color

    if len(np.unique(grid, axis=None)) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
