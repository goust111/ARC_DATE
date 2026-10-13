from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
MATRIX_LINE = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATE_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _component_list(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _corner_region(grid: np.ndarray, corner: str, height: int, width: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = grid.shape
    if corner == "tl":
        return grid[:height, :width], (0, 0)
    if corner == "tr":
        return grid[:height, w - width :], (0, w - width)
    if corner == "bl":
        return grid[h - height :, :width], (h - height, 0)
    if corner == "br":
        return grid[h - height :, w - width :], (h - height, w - width)
    raise ValueError(f"unknown corner {corner}")


def _is_matrix_region(region: np.ndarray, bg: int) -> bool:
    h, w = region.shape
    if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
        return False
    line = int(region[0, 0])
    if line == bg:
        return False
    for r in range(h):
        for c in range(w):
            val = int(region[r, c])
            if r % 2 == 0 or c % 2 == 0:
                if val != line:
                    return False
            elif val == bg or val == line:
                return False
    return True


def _find_lookup_matrix(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for corner in ("tl", "tr", "bl", "br"):
        for rh in range(5, h + 1, 2):
            for cw in range(5, w + 1, 2):
                region, (r0, c0) = _corner_region(grid, corner, rh, cw)
                if _is_matrix_region(region, bg):
                    candidates.append((rh * cw, (r0, c0, rh, cw), region[1::2, 1::2].copy()))
    assert candidates, "expected one corner lookup matrix"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_area = candidates[0][0]
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "lookup matrix must be unique"
    _, (r0, c0, rh, cw), lookup = best[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + rh, c0 : c0 + cw] = True
    return lookup, mask


def _extract_pieces(grid: np.ndarray, bg: int, matrix_mask: np.ndarray) -> list[dict]:
    mask = (grid != bg) & (~matrix_mask)
    pieces = []
    for coords in _component_list(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "pieces must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(coords)
        rel = sorted((r - r0, c - c0) for r, c in coords)
        pieces.append(
            {
                "color": int(color),
                "coords": tuple((int(r), int(c)) for r, c in rel),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
            }
        )
    return pieces


def _piece_mask(piece: dict) -> np.ndarray:
    h = int(piece["height"])
    w = int(piece["width"])
    mask = np.zeros((h, w), dtype=bool)
    for r, c in piece["coords"]:
        mask[int(r), int(c)] = True
    return mask


def _parse_piece(piece: dict, cell_h: int, cell_w: int) -> dict:
    mask = _piece_mask(piece)
    h, w = mask.shape
    assert h in (cell_h, cell_h + 1), "piece height must match base tile height"
    assert w in (cell_w, cell_w + 1), "piece width must match base tile width"

    left = tuple(r for r in range(1, cell_h - 1) if not mask[r, 0])
    top = tuple(c for c in range(1, cell_w - 1) if not mask[0, c])
    right = tuple(r for r in range(1, cell_h - 1) if w == cell_w + 1 and mask[r, cell_w])
    bottom = tuple(c for c in range(1, cell_w - 1) if h == cell_h + 1 and mask[cell_h, c])

    for r in range(cell_h):
        for c in range(cell_w):
            expected = True
            if c == 0 and r in left:
                expected = False
            if r == 0 and c in top:
                expected = False
            assert mask[r, c] == expected, "unexpected interior carving"

    if w == cell_w + 1:
        for r in range(h):
            expected = r in right
            assert mask[r, cell_w] == expected, "unexpected right protrusion"
    if h == cell_h + 1:
        for c in range(w):
            expected = c in bottom
            assert mask[cell_h, c] == expected, "unexpected bottom protrusion"

    return {
        "color": int(piece["color"]),
        "coords": piece["coords"],
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _unique_map(values: list[tuple[tuple[int, ...], int]], name: str) -> dict[tuple[int, ...], int]:
    out: dict[tuple[int, ...], int] = {}
    for key, idx in values:
        if not key:
            continue
        assert key not in out, f"duplicate {name} pattern"
        out[key] = int(idx)
    return out


def _assemble(parsed: list[dict], lookup: np.ndarray, bg: int, cell_h: int, cell_w: int) -> np.ndarray:
    rows, cols = lookup.shape
    assert len(parsed) == rows * cols, "piece count must match lookup matrix"

    by_left_slot = _unique_map([(piece["left"], i) for i, piece in enumerate(parsed)], "left-slot")
    by_top_slot = _unique_map([(piece["top"], i) for i, piece in enumerate(parsed)], "top-slot")

    top_left_candidates = [
        i for i, piece in enumerate(parsed) if not piece["left"] and not piece["top"]
    ]
    assert len(top_left_candidates) == 1, "expected a unique top-left corner piece"

    layout = [[-1] * cols for _ in range(rows)]
    used: set[int] = set()

    layout[0][0] = top_left_candidates[0]
    used.add(layout[0][0])

    for c in range(1, cols):
        prev = parsed[layout[0][c - 1]]
        assert prev["right"], "top-row internal piece must connect rightward"
        nxt = by_left_slot[prev["right"]]
        assert nxt not in used, "piece reused while building first row"
        assert not parsed[nxt]["top"], "top-row piece must have a flat top edge"
        layout[0][c] = nxt
        used.add(nxt)
    assert not parsed[layout[0][-1]]["right"], "top-right piece must have a flat right edge"

    for r in range(1, rows):
        above = parsed[layout[r - 1][0]]
        assert above["bottom"], "left-column internal piece must connect downward"
        nxt = by_top_slot[above["bottom"]]
        assert nxt not in used, "piece reused while building first column"
        assert not parsed[nxt]["left"], "left-column piece must have a flat left edge"
        layout[r][0] = nxt
        used.add(nxt)

        for c in range(1, cols):
            left_piece = parsed[layout[r][c - 1]]
            above_piece = parsed[layout[r - 1][c]]
            assert left_piece["right"], "internal piece must connect rightward"
            assert above_piece["bottom"], "internal piece must connect downward"
            idx_from_left = by_left_slot[left_piece["right"]]
            idx_from_top = by_top_slot[above_piece["bottom"]]
            assert idx_from_left == idx_from_top, "horizontal and vertical matches disagree"
            assert idx_from_left not in used, "piece reused in interior"
            layout[r][c] = idx_from_left
            used.add(idx_from_left)

        assert not parsed[layout[r][-1]]["right"], "right-border piece must have a flat right edge"

    for c in range(cols):
        assert not parsed[layout[-1][c]]["bottom"], "bottom-border piece must have a flat bottom edge"
    assert len(used) == len(parsed), "not all pieces were placed"

    out = np.full((rows * cell_h, cols * cell_w), bg, dtype=int)
    occupied = np.zeros_like(out, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            color = int(lookup[r, c])
            piece = parsed[layout[r][c]]
            base_r = r * cell_h
            base_c = c * cell_w
            for pr, pc in piece["coords"]:
                rr = base_r + int(pr)
                cc = base_c + int(pc)
                assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "assembled piece out of bounds"
                assert not occupied[rr, cc], "interlocking pieces must not overlap"
                occupied[rr, cc] = True
                out[rr, cc] = color
    assert occupied.all(), "assembled rectangle must have no gaps"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble monochrome jigsaw pieces into a rectangle, then recolor each piece
    using the same row/column position in the corner lookup matrix.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    lookup, matrix_mask = _find_lookup_matrix(g, bg)
    pieces = _extract_pieces(g, bg, matrix_mask)
    assert pieces, "expected scattered pieces outside the lookup matrix"

    cell_h = min(piece["height"] for piece in pieces)
    cell_w = min(piece["width"] for piece in pieces)
    parsed = [_parse_piece(piece, cell_h, cell_w) for piece in pieces]
    return _assemble(parsed, lookup, bg, cell_h, cell_w)


def _pattern_pool(side_len: int) -> list[tuple[int, ...]]:
    positions = list(range(2, side_len - 1))
    patterns: list[tuple[int, ...]] = []
    max_size = min(3, len(positions))
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(positions, size):
            patterns.append(tuple(int(v) for v in combo))
    return patterns


def _piece_coords(
    cell_h: int,
    cell_w: int,
    left_slot: tuple[int, ...],
    top_slot: tuple[int, ...],
    right_tab: tuple[int, ...],
    bottom_tab: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    occ = {(r, c) for r in range(cell_h) for c in range(cell_w)}
    for r in left_slot:
        occ.remove((int(r), 0))
    for c in top_slot:
        occ.remove((0, int(c)))
    for r in right_tab:
        occ.add((int(r), cell_w))
    for c in bottom_tab:
        occ.add((cell_h, int(c)))
    return tuple(sorted((int(r), int(c)) for r, c in occ))


def _coords_to_patch(coords: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    patch = np.full((max_r + 1, max_c + 1), bg, dtype=int)
    for r, c in coords:
        patch[r, c] = int(color)
    return patch


def _matrix_patch(lookup: np.ndarray) -> np.ndarray:
    rows, cols = lookup.shape
    patch = np.full((2 * rows + 1, 2 * cols + 1), MATRIX_LINE, dtype=int)
    patch[1::2, 1::2] = lookup
    return patch


def _build_expected(pieces: list[dict], lookup: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    parsed = [
        {
            "color": piece["color"],
            "coords": piece["coords"],
            "left": piece["left"],
            "top": piece["top"],
            "right": piece["right"],
            "bottom": piece["bottom"],
        }
        for piece in pieces
    ]
    return _assemble(parsed, lookup, BG, cell_h, cell_w)


def _inflate(coords: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((int(nr), int(nc)))
    return out


def _place_patch(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    patch: np.ndarray,
    bg: int,
    require_margin: bool = True,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]
    positions = []
    row_range = range(1, h - ph) if require_margin else range(0, h - ph + 1)
    col_range = range(1, w - pw) if require_margin else range(0, w - pw + 1)
    for r0 in row_range:
        for c0 in col_range:
            ok = True
            absolute: set[tuple[int, int]] = set()
            for dr, dc in coords:
                rr = r0 + dr
                cc = c0 + dc
                if (rr, cc) in blocked:
                    ok = False
                    break
                absolute.add((int(rr), int(cc)))
            if ok:
                positions.append((int(r0), int(c0), absolute))
    if not positions:
        return None
    r0, c0, absolute = random.choice(positions)
    for rr, cc in absolute:
        grid[rr, cc] = int(patch[rr - r0, cc - c0])
    blocked.update(_inflate(absolute, h, w))
    return r0, c0


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = random.choice(((2, 2), (2, 3), (3, 2), (2, 4), (4, 2)))
    piece_count = rows * cols
    cell_h = random.randint(6, 8)
    cell_w = random.randint(6, 8)

    vertical_samples = random.sample(_pattern_pool(cell_h), rows * (cols - 1))
    horizontal_samples = random.sample(_pattern_pool(cell_w), (rows - 1) * cols)
    vertical_patterns = [
        [tuple(vertical_samples[r * (cols - 1) + c]) for c in range(cols - 1)]
        for r in range(rows)
    ]
    horizontal_patterns = [
        [tuple(horizontal_samples[r * cols + c]) for c in range(cols)]
        for r in range(rows - 1)
    ]

    lookup_colors = random.sample(PALETTE, piece_count)
    lookup = np.array(lookup_colors, dtype=int).reshape(rows, cols)
    piece_colors = random.sample(PALETTE, piece_count)

    pieces: list[dict] = []
    for r in range(rows):
        for c in range(cols):
            left_slot = vertical_patterns[r][c - 1] if c > 0 else ()
            right_tab = vertical_patterns[r][c] if c < cols - 1 else ()
            top_slot = horizontal_patterns[r - 1][c] if r > 0 else ()
            bottom_tab = horizontal_patterns[r][c] if r < rows - 1 else ()

            coords = _piece_coords(cell_h, cell_w, left_slot, top_slot, right_tab, bottom_tab)
            color = piece_colors[r * cols + c]
            pieces.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "left": tuple(left_slot),
                    "top": tuple(top_slot),
                    "right": tuple(right_tab),
                    "bottom": tuple(bottom_tab),
                }
            )

    out = _build_expected(pieces, lookup, cell_h, cell_w)

    matrix = _matrix_patch(lookup)
    out_h = rows * cell_h
    out_w = cols * cell_w
    grid_h = max(out_h + random.randint(cell_h + 3, cell_h + 9), matrix.shape[0] + cell_h + 4)
    grid_w = max(out_w + random.randint(cell_w + 3, cell_w + 11), matrix.shape[1] + cell_w + 4)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    corner = random.choice(("tl", "tr", "bl", "br"))
    matrix_h, matrix_w = matrix.shape
    if corner == "tl":
        mr, mc = 0, 0
    elif corner == "tr":
        mr, mc = 0, grid_w - matrix_w
    elif corner == "bl":
        mr, mc = grid_h - matrix_h, 0
    else:
        mr, mc = grid_h - matrix_h, grid_w - matrix_w
    grid[mr : mr + matrix_h, mc : mc + matrix_w] = matrix

    blocked = _inflate(
        {(int(r), int(c)) for r, c in np.argwhere(grid != BG)},
        grid_h,
        grid_w,
    )

    patches = [
        (_coords_to_patch(piece["coords"], piece["color"], BG), piece["color"])
        for piece in sorted(pieces, key=lambda item: len(item["coords"]), reverse=True)
    ]
    for patch, _color in patches:
        placed = _place_patch(grid, blocked, patch, BG, require_margin=True)
        if placed is None:
            raise RuntimeError("failed to scatter pieces without overlap")

    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        try:
            inp, expected = _generate_case()
        except RuntimeError:
            continue
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
