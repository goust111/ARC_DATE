from __future__ import annotations

import random

import numpy as np


CORNER_SIDES = {
    "TL": ("top", "left"),
    "TR": ("top", "right"),
    "BL": ("bottom", "left"),
    "BR": ("bottom", "right"),
}

PAIR_TYPES = (
    ("TL", "BL"),
    ("BL", "TL"),
    ("TR", "BR"),
    ("BR", "TR"),
    ("TL", "TR"),
    ("TR", "TL"),
    ("BL", "BR"),
    ("BR", "BL"),
)

GEN_RETRY_LIMIT = 500
GRID_MIN_SIDE = 26
GRID_MAX_SIDE = 34
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
PIECE_SIDE_MIN = 4
PIECE_SIDE_MAX = 8
PLACEMENT_GAP_MIN = 1
PLACEMENT_GAP_MAX = 4
PLACEMENT_MARGIN = 1
PLACEMENT_ATTEMPTS = 250
COLOR_IDS = tuple(range(10))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(counts.argmax())
    if len(counts) > 1:
        ordered = np.sort(counts)
        assert ordered[-1] > ordered[-2], "expected a unique dominant background color"
    return int(values[best])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            r0, c0, r1, c1 = _bbox_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel = coords - np.array([r0, c0], dtype=int)
            mask[rel[:, 0], rel[:, 1]] = True
            pieces.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return pieces


def _analyze_piece(piece: dict) -> dict:
    mask = np.asarray(piece["mask"], dtype=bool)
    h, w = map(int, mask.shape)
    assert h >= 2 and w >= 2, "piece bbox must have both dimensions >= 2"

    empty = np.argwhere(~mask)
    assert empty.size > 0, "piece must have a missing corner rectangle"
    er0, ec0, er1, ec1 = _bbox_coords(empty)
    cut_h = er1 - er0 + 1
    cut_w = ec1 - ec0 + 1
    assert 1 <= cut_h < h and 1 <= cut_w < w, "cutout must be a proper corner rectangle"

    empty_rect = np.zeros_like(mask, dtype=bool)
    empty_rect[er0 : er1 + 1, ec0 : ec1 + 1] = True
    assert np.array_equal(empty_rect, ~mask), "piece must equal a full rectangle minus one corner block"

    corner: str
    if er0 == 0 and ec0 == 0:
        corner = "TL"
    elif er0 == 0 and ec1 == w - 1:
        corner = "TR"
    elif er1 == h - 1 and ec0 == 0:
        corner = "BL"
    elif er1 == h - 1 and ec1 == w - 1:
        corner = "BR"
    else:
        raise AssertionError("cutout must touch exactly one corner of the bbox")

    row_side, col_side = CORNER_SIDES[corner]
    out = dict(piece)
    out.update(
        {
            "h": h,
            "w": w,
            "cut_h": int(cut_h),
            "cut_w": int(cut_w),
            "corner": corner,
            "row_side": row_side,
            "col_side": col_side,
        }
    )
    return out


def _pair_rule(base_corner: str, fill_corner: str) -> tuple[str, str] | None:
    base_row, base_col = CORNER_SIDES[base_corner]
    fill_row, fill_col = CORNER_SIDES[fill_corner]
    same_row = base_row == fill_row
    same_col = base_col == fill_col
    if same_row == same_col:
        return None
    if same_col:
        return "horizontal", base_col
    return "vertical", base_row


def _is_pair_candidate(base: dict, fill: dict) -> bool:
    if base["color"] == fill["color"]:
        return False
    if (base["h"], base["w"], base["cut_h"], base["cut_w"]) != (
        fill["h"],
        fill["w"],
        fill["cut_h"],
        fill["cut_w"],
    ):
        return False

    rule = _pair_rule(base["corner"], fill["corner"])
    if rule is None:
        return False

    br0, bc0, br1, bc1 = base["bbox"]
    fr0, fc0, fr1, fc1 = fill["bbox"]
    axis, side = rule
    if axis == "horizontal":
        return bool(fc1 < bc0) if side == "left" else bool(fc0 > bc1)
    return bool(fr1 < br0) if side == "top" else bool(fr0 > br1)


def _merge_pair(base: dict, fill: dict, bg: int) -> np.ndarray:
    h = int(base["h"])
    w = int(base["w"])
    cut_h = int(base["cut_h"])
    cut_w = int(base["cut_w"])
    out = np.full((h, w), int(bg), dtype=int)
    covered = np.zeros((h, w), dtype=bool)

    out[base["mask"]] = int(base["color"])
    covered[base["mask"]] = True

    axis, side = _pair_rule(base["corner"], fill["corner"])
    assert axis in {"horizontal", "vertical"}
    if axis == "horizontal":
        dr = 0
        dc = -(w - cut_w) if side == "left" else (w - cut_w)
    else:
        dr = -(h - cut_h) if side == "top" else (h - cut_h)
        dc = 0

    rows, cols = np.where(fill["mask"])
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        tr = int(rr + dr)
        tc = int(cc + dc)
        if 0 <= tr < h and 0 <= tc < w:
            out[tr, tc] = int(fill["color"])
            covered[tr, tc] = True

    assert bool(covered.all()), "paired pieces must merge into a full rectangle"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair equal-bbox corner-cut fragments and merge each pair back into a full rectangle.

    Every foreground component is a rectangle with one rectangular corner removed.
    Two matching fragments have the same bbox size and the same cutout size. Their
    missing corners differ by exactly one side:
    - same left/right side -> the fill piece slides horizontally from that side
    - same top/bottom side -> the fill piece slides vertically from that side

    The completed rectangle is written at the stationary fragment's original bbox.
    Wherever the translated fill piece overlaps the stationary piece, the fill piece's
    color overwrites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    pieces = [_analyze_piece(piece) for piece in _find_components(g, bg)]
    assert pieces and len(pieces) % 2 == 0, "expected an even number of valid fragments"

    candidates: dict[int, list[int]] = {}
    incoming = [0] * len(pieces)
    for i, base in enumerate(pieces):
        js = [j for j, fill in enumerate(pieces) if i != j and _is_pair_candidate(base, fill)]
        candidates[i] = js
        for j in js:
            incoming[j] += 1

    bases = [i for i, js in candidates.items() if len(js) == 1]
    assert bases, "expected at least one stationary fragment"
    assert len(bases) * 2 == len(pieces), "every pair should contribute exactly one stationary fragment"
    assert all(len(candidates[i]) in {0, 1} for i in range(len(pieces))), "pairing must be unambiguous"
    assert all(count in {0, 1} for count in incoming), "each fill fragment must belong to only one pair"

    out = np.full_like(g, int(bg))
    used: set[int] = set()
    for i in sorted(bases, key=lambda idx: pieces[idx]["bbox"]):
        j = candidates[i][0]
        assert i not in used and j not in used, "a fragment may be used in only one pair"
        assert len(candidates[j]) == 0 and incoming[i] == 0 and incoming[j] == 1

        base = pieces[i]
        fill = pieces[j]
        merged = _merge_pair(base, fill, bg)
        r0, c0, r1, c1 = base["bbox"]
        assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == bg), "output rectangles must not overlap"
        out[r0 : r1 + 1, c0 : c1 + 1] = merged
        used.add(i)
        used.add(j)

    assert len(used) == len(pieces), "all fragments must be consumed by pairing"
    return out


def _corner_cut_mask(h: int, w: int, cut_h: int, cut_w: int, corner: str) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if corner == "TL":
        mask[:cut_h, :cut_w] = False
    elif corner == "TR":
        mask[:cut_h, w - cut_w :] = False
    elif corner == "BL":
        mask[h - cut_h :, :cut_w] = False
    else:
        assert corner == "BR"
        mask[h - cut_h :, w - cut_w :] = False
    return mask


def _paint_patch(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _clear_bbox(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occupied.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _randint(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _place_pair_boxes(
    occupied: np.ndarray,
    h: int,
    w: int,
    axis: str,
    side: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    hh, ww = occupied.shape
    m = PLACEMENT_MARGIN
    for _ in range(PLACEMENT_ATTEMPTS):
        gap = random.randint(PLACEMENT_GAP_MIN, PLACEMENT_GAP_MAX)
        if axis == "horizontal" and side == "left":
            base_r = _randint(m, hh - h - m)
            base_c = _randint(m + w + gap, ww - w - m)
            if base_r is None or base_c is None:
                continue
            fill_r = _randint(m, hh - h - m)
            fill_c = _randint(m, base_c - w - gap)
        elif axis == "horizontal" and side == "right":
            base_r = _randint(m, hh - h - m)
            base_c = _randint(m, ww - (2 * w) - gap - m)
            if base_r is None or base_c is None:
                continue
            fill_r = _randint(m, hh - h - m)
            fill_c = _randint(base_c + w + gap, ww - w - m)
        elif axis == "vertical" and side == "top":
            base_r = _randint(m + h + gap, hh - h - m)
            base_c = _randint(m, ww - w - m)
            if base_r is None or base_c is None:
                continue
            fill_r = _randint(m, base_r - h - gap)
            fill_c = _randint(m, ww - w - m)
        else:
            assert axis == "vertical" and side == "bottom"
            base_r = _randint(m, hh - (2 * h) - gap - m)
            base_c = _randint(m, ww - w - m)
            if base_r is None or base_c is None:
                continue
            fill_r = _randint(base_r + h + gap, hh - h - m)
            fill_c = _randint(m, ww - w - m)

        if fill_r is None or fill_c is None:
            continue
        if not _clear_bbox(occupied, base_r, base_c, h, w, pad=1):
            continue
        if not _clear_bbox(occupied, fill_r, fill_c, h, w, pad=1):
            continue

        occupied[base_r : base_r + h, base_c : base_c + w] = True
        occupied[fill_r : fill_r + h, fill_c : fill_c + w] = True
        return (int(base_r), int(base_c)), (int(fill_r), int(fill_c))
    return None


def _all_signatures() -> list[tuple[int, int, int, int]]:
    signatures = []
    for h in range(PIECE_SIDE_MIN, PIECE_SIDE_MAX + 1):
        for w in range(PIECE_SIDE_MIN, PIECE_SIDE_MAX + 1):
            for cut_h in range(1, h):
                for cut_w in range(1, w):
                    signatures.append((h, w, cut_h, cut_w))
    return signatures


ALL_SIGNATURES = _all_signatures()


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    width = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)

    bg = random.choice(COLOR_IDS)
    colors = [c for c in COLOR_IDS if c != bg]
    random.shuffle(colors)
    needed_colors = 2 * pair_count
    if len(colors) < needed_colors:
        return None

    signatures = random.sample(ALL_SIGNATURES, pair_count)
    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for pair_index in range(pair_count):
        base_corner, fill_corner = random.choice(PAIR_TYPES)
        axis, side = _pair_rule(base_corner, fill_corner)
        h, w, cut_h, cut_w = signatures[pair_index]

        placed = _place_pair_boxes(occupied, h, w, axis, side)
        if placed is None:
            return None
        (base_r, base_c), (fill_r, fill_c) = placed

        base_mask = _corner_cut_mask(h, w, cut_h, cut_w, base_corner)
        fill_mask = _corner_cut_mask(h, w, cut_h, cut_w, fill_corner)
        base_color = colors[2 * pair_index]
        fill_color = colors[2 * pair_index + 1]
        _paint_patch(grid, base_r, base_c, base_mask, base_color)
        _paint_patch(grid, fill_r, fill_c, fill_mask, fill_color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape == grid.shape and isinstance(out, np.ndarray):
            return grid
    raise RuntimeError("failed to generate a valid input")
