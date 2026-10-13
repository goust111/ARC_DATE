from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

LINE_COLORS = (RED, BLUE, GREEN, YELLOW)
MIX_TABLE = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, GREEN)): ORANGE,
    frozenset((BLUE, GREEN)): AZURE,
    frozenset((RED, YELLOW)): MAROON,
    frozenset((BLUE, YELLOW)): GREEN,
    frozenset((GREEN, YELLOW)): ORANGE,
}

ORIENT_DELTAS = {
    "H": (0, 1),
    "V": (1, 0),
    "D1": (1, 1),
    "D2": (1, -1),
}
RIGHT_ANGLE_OPTIONS = (("H", "V"), ("D1", "D2"))
SLANTED_OPTIONS = (("H", "D1"), ("H", "D2"), ("V", "D1"), ("V", "D2"))

GEN_RETRIES = 500
PLACEMENT_RETRIES = 250
RESERVE_PAD = 2


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cell_count(height: int, width: int) -> int:
    if height == 1:
        return width
    if width == 1:
        return height
    return 2 * height + 2 * width - 4


def _parse_legend_rows(
    inner: np.ndarray, bg: int, frame_color: int
) -> dict[frozenset[int], int]:
    mapping: dict[frozenset[int], int] = {}
    found = False
    for row in inner:
        vals = [int(v) for v in row if int(v) not in (bg, frame_color)]
        if not vals:
            continue
        if len(vals) != 3:
            return {}
        pair = frozenset((vals[0], vals[1]))
        if len(pair) != 2:
            return {}
        found = True
        if pair in mapping and mapping[pair] != vals[2]:
            return {}
        mapping[pair] = int(vals[2])
    return mapping if found else {}


def _parse_legend(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int, frame_color: int
) -> dict[frozenset[int], int]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mapping = _parse_legend_rows(inner, bg, frame_color)
    if mapping:
        return mapping
    return _parse_legend_rows(inner.T, bg, frame_color)


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[frozenset[int], int], np.ndarray]:
    h, w = grid.shape
    best: tuple[int, dict[frozenset[int], int], tuple[int, int, int, int]] | None = None
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            hh, ww = r1 - r0 + 1, c1 - c0 + 1
            if hh < 4 or ww < 5:
                continue
            near_row_edge = min(r0, h - 1 - r1) <= 1
            near_col_edge = min(c0, w - 1 - c1) <= 1
            if not (near_row_edge and near_col_edge):
                continue
            if len(comp) != _border_cell_count(hh, ww):
                continue
            border_ok = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    on_border = rr in (r0, r1) or cc in (c0, c1)
                    if on_border and int(grid[rr, cc]) != color:
                        border_ok = False
                        break
                if not border_ok:
                    break
            if not border_ok:
                continue
            mapping = _parse_legend(grid, (r0, c0, r1, c1), bg, color)
            if not mapping:
                continue
            score = len(mapping)
            if best is None or score > best[0]:
                best = (score, mapping, (r0, c0, r1, c1))
    if best is None:
        raise AssertionError("legend not found")
    _score, mapping, bbox = best
    r0, c0, r1, c1 = bbox
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mapping, legend_mask


def _evidence_color(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    r: int,
    c: int,
    orient: str,
    allowed: set[int],
) -> int | None:
    dr, dc = ORIENT_DELTAS[orient]
    r1, c1 = r - dr, c - dc
    r2, c2 = r + dr, c + dc
    if not (_in_bounds(grid.shape, r1, c1) and _in_bounds(grid.shape, r2, c2)):
        return None
    if legend_mask[r1, c1] or legend_mask[r2, c2]:
        return None
    v1 = int(grid[r1, c1])
    v2 = int(grid[r2, c2])
    if v1 == v2 and v1 in allowed:
        return v1
    return None


def _intersection_kind(o1: str, o2: str) -> str | None:
    if frozenset((o1, o2)) in (frozenset(("H", "V")), frozenset(("D1", "D2"))):
        return "square"
    axis = {"H", "V"}
    diag = {"D1", "D2"}
    if (o1 in axis and o2 in diag) or (o2 in axis and o1 in diag):
        return "diamond"
    return None


def _find_intersections(
    grid: np.ndarray, legend_mask: np.ndarray, mapping: dict[frozenset[int], int], bg: int
) -> list[tuple[int, int, str, int]]:
    allowed = {int(c) for c in np.unique(grid[~legend_mask]) if int(c) != bg}
    intersections: list[tuple[int, int, str, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if legend_mask[r, c]:
                continue
            hits: list[tuple[str, int]] = []
            for orient in ORIENT_DELTAS:
                color = _evidence_color(grid, legend_mask, r, c, orient, allowed)
                if color is not None:
                    hits.append((orient, color))
            if len(hits) != 2:
                continue
            (o1, c1), (o2, c2) = hits
            if c1 == c2:
                continue
            kind = _intersection_kind(o1, o2)
            if kind is None:
                continue
            mix = mapping.get(frozenset((c1, c2)))
            if mix is None:
                continue
            intersections.append((r, c, kind, int(mix)))
    return intersections


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mapping, legend_mask = _find_legend(g, bg)
    intersections = _find_intersections(g, legend_mask, mapping, bg)

    out = g.copy()
    for r, c, kind, color in intersections:
        if kind == "square":
            targets = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        else:
            targets = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for rr, cc in targets:
            if _in_bounds(out.shape, rr, cc) and not legend_mask[rr, cc] and int(out[rr, cc]) == bg:
                out[rr, cc] = color
    return out


def _expand_rect(
    blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int
) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _legend_bbox(
    h: int, w: int, rows: int, corner: str, row_margin: int, col_margin: int
) -> tuple[int, int, int, int]:
    legend_h = rows + 2
    legend_w = 5
    if corner == "tl":
        r0, c0 = row_margin, col_margin
    elif corner == "tr":
        r0, c0 = row_margin, w - legend_w - col_margin
    elif corner == "bl":
        r0, c0 = h - legend_h - row_margin, col_margin
    else:
        r0, c0 = h - legend_h - row_margin, w - legend_w - col_margin
    return r0, c0, r0 + legend_h - 1, c0 + legend_w - 1


def _draw_legend(
    grid: np.ndarray, rows: list[frozenset[int]], corner: str, row_margin: int, col_margin: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _legend_bbox(grid.shape[0], grid.shape[1], len(rows), corner, row_margin, col_margin)
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    for i, pair in enumerate(rows):
        a, b = random.sample(list(pair), 2)
        out = MIX_TABLE[pair]
        rr = r0 + 1 + i
        grid[rr, c0 + 1] = a
        grid[rr, c0 + 2] = b
        grid[rr, c0 + 3] = out
    return r0, c0, r1, c1


def _segment_relative_cells(orient: str, neg_len: int, pos_len: int) -> list[tuple[int, int]]:
    dr, dc = ORIENT_DELTAS[orient]
    return [(step * dr, step * dc) for step in range(-neg_len, pos_len + 1)]


def _place_crossing(
    grid: np.ndarray,
    blocked: np.ndarray,
    branch: str,
    pair: frozenset[int],
) -> tuple[int, int, str] | None:
    options = RIGHT_ANGLE_OPTIONS if branch == "square" else SLANTED_OPTIONS
    for _ in range(PLACEMENT_RETRIES):
        o1, o2 = random.choice(options)
        rel1 = _segment_relative_cells(o1, random.randint(2, 5), random.randint(2, 5))
        rel2 = _segment_relative_cells(o2, random.randint(2, 5), random.randint(2, 5))
        rel_all = rel1 + rel2
        min_dr = min(dr for dr, _ in rel_all)
        max_dr = max(dr for dr, _ in rel_all)
        min_dc = min(dc for _, dc in rel_all)
        max_dc = max(dc for _, dc in rel_all)
        r_lo = RESERVE_PAD - min_dr
        r_hi = grid.shape[0] - 1 - RESERVE_PAD - max_dr
        c_lo = RESERVE_PAD - min_dc
        c_hi = grid.shape[1] - 1 - RESERVE_PAD - max_dc
        if r_lo > r_hi or c_lo > c_hi:
            continue
        center_r = random.randint(r_lo, r_hi)
        center_c = random.randint(c_lo, c_hi)
        cells1 = [(center_r + dr, center_c + dc) for dr, dc in rel1]
        cells2 = [(center_r + dr, center_c + dc) for dr, dc in rel2]
        all_cells = cells1 + cells2
        r0 = min(r for r, _ in all_cells)
        r1 = max(r for r, _ in all_cells)
        c0 = min(c for _, c in all_cells)
        c1 = max(c for _, c in all_cells)
        rr0 = max(0, r0 - RESERVE_PAD)
        rr1 = min(grid.shape[0] - 1, r1 + RESERVE_PAD)
        cc0 = max(0, c0 - RESERVE_PAD)
        cc1 = min(grid.shape[1] - 1, c1 + RESERVE_PAD)
        if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        color_a, color_b = tuple(pair)
        if random.random() < 0.5:
            color_a, color_b = color_b, color_a
        show_first = random.random() < 0.5
        for rr, cc in cells1:
            if (rr, cc) == (center_r, center_c) and not show_first:
                continue
            grid[rr, cc] = color_a
        for rr, cc in cells2:
            if (rr, cc) == (center_r, center_c) and show_first:
                continue
            grid[rr, cc] = color_b
        _expand_rect(blocked, r0, c0, r1, c1, RESERVE_PAD)
        return center_r, center_c, branch
    return None


def _place_single_line(grid: np.ndarray, blocked: np.ndarray) -> bool:
    for _ in range(PLACEMENT_RETRIES):
        orient = random.choice(tuple(ORIENT_DELTAS))
        rel = _segment_relative_cells(orient, random.randint(2, 5), random.randint(2, 5))
        min_dr = min(dr for dr, _ in rel)
        max_dr = max(dr for dr, _ in rel)
        min_dc = min(dc for _, dc in rel)
        max_dc = max(dc for _, dc in rel)
        r_lo = RESERVE_PAD - min_dr
        r_hi = grid.shape[0] - 1 - RESERVE_PAD - max_dr
        c_lo = RESERVE_PAD - min_dc
        c_hi = grid.shape[1] - 1 - RESERVE_PAD - max_dc
        if r_lo > r_hi or c_lo > c_hi:
            continue
        center_r = random.randint(r_lo, r_hi)
        center_c = random.randint(c_lo, c_hi)
        cells = [(center_r + dr, center_c + dc) for dr, dc in rel]
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        rr0 = max(0, r0 - RESERVE_PAD)
        rr1 = min(grid.shape[0] - 1, r1 + RESERVE_PAD)
        cc0 = max(0, c0 - RESERVE_PAD)
        cc1 = min(grid.shape[1] - 1, c1 + RESERVE_PAD)
        if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        color = random.choice(LINE_COLORS)
        for rr, cc in cells:
            grid[rr, cc] = color
        _expand_rect(blocked, r0, c0, r1, c1, RESERVE_PAD)
        return True
    return False


def _branch_schedule(count: int) -> list[str]:
    extra_square = random.randint(0, 1) if count % 2 else 0
    n_square = count // 2 + extra_square
    n_diamond = count - n_square
    branches = ["square"] * n_square + ["diamond"] * n_diamond
    random.shuffle(branches)
    return branches


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.full((h, w), BLACK, dtype=int)

    legend_rows = random.randint(4, 6)
    legend_pairs = random.sample(list(MIX_TABLE.keys()), legend_rows)
    corner = random.choice(("tl", "tr", "bl", "br"))
    row_margin = random.randint(0, 1)
    col_margin = random.randint(0, 1)
    bbox = _draw_legend(grid, legend_pairs, corner, row_margin, col_margin)

    blocked = np.zeros_like(grid, dtype=bool)
    _expand_rect(blocked, *bbox, 1)

    gadget_count = random.randint(2, 5)
    branches = _branch_schedule(gadget_count)
    pair_order = random.sample(legend_pairs, len(legend_pairs))
    used_centers: list[tuple[int, int, str]] = []
    for i, branch in enumerate(branches):
        pair = pair_order[i % len(pair_order)]
        placed = _place_crossing(grid, blocked, branch, pair)
        if placed is None:
            return None
        used_centers.append(placed)

    for _ in range(random.randint(0, 2)):
        _place_single_line(grid, blocked)

    mapping, legend_mask = _find_legend(grid, BLACK)
    intersections = _find_intersections(grid, legend_mask, mapping, BLACK)
    kinds = Counter(kind for _r, _c, kind, _color in intersections)
    if len(intersections) != len(used_centers):
        return None
    if kinds.get("square", 0) == 0 or kinds.get("diamond", 0) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
