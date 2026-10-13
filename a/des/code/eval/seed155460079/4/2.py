from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_MAX_ATTEMPTS = 400


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _find_components(grid: np.ndarray, blocked: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in blocked:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _D4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": _bbox(coord_arr),
                }
            )
    return comps


def _frame_candidate(grid: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    full_rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == color)]
    full_cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == color)]
    if len(full_rows) < 2 or len(full_cols) < 2:
        return None
    if full_rows[0] != r0 or full_rows[-1] != r1 or full_cols[0] != c0 or full_cols[-1] != c1:
        return None

    row_spans = [(full_rows[i] + 1, full_rows[i + 1] - 1) for i in range(len(full_rows) - 1)]
    col_spans = [(full_cols[i] + 1, full_cols[i + 1] - 1) for i in range(len(full_cols) - 1)]
    if not row_spans or not col_spans:
        return None

    row_sizes = [b - a + 1 for a, b in row_spans]
    col_sizes = [b - a + 1 for a, b in col_spans]
    if min(row_sizes) <= 0 or min(col_sizes) <= 0:
        return None
    if len(set(row_sizes)) != 1 or len(set(col_sizes)) != 1:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(sub, dtype=bool)
    expected[[r - r0 for r in full_rows], :] = True
    expected[:, [c - c0 for c in full_cols]] = True
    if not np.array_equal(sub == color, expected):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "row_lines": full_rows,
        "col_lines": full_cols,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "pixel_count": int(coords.shape[0]),
    }


def _parse_frame(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        info = _frame_candidate(grid, color)
        if info is not None:
            candidates.append(info)
    assert candidates, "expected a rectangular frame with full divider lines"
    candidates.sort(key=lambda item: item["pixel_count"], reverse=True)
    return candidates[0]


def _piece_side(bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if r1 < fr0:
        return "top"
    if r0 > fr1:
        return "bottom"
    if c1 < fc0:
        return "left"
    if c0 > fc1:
        return "right"
    raise AssertionError("piece must lie strictly outside the frame")


def _match_span(lo: int, hi: int, spans: list[tuple[int, int]]) -> int:
    matches = [i for i, (a, b) in enumerate(spans) if a <= lo and hi <= b]
    assert len(matches) == 1, "piece must align to exactly one slot"
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each external fragment straight into the slot it faces.

    The frame is a one-pixel-thick rectangular scaffold whose full-width and
    full-height divider lines partition the interior into equal slots. Each
    fragment stays rigid, keeps its perpendicular coordinate fixed, and moves
    along one axis until its leading edge reaches the far interior boundary of
    the target slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frame = _parse_frame(g, bg)
    row_spans = frame["row_spans"]
    col_spans = frame["col_spans"]

    pieces = _find_components(g, blocked={bg, int(frame["color"])})
    out = g.copy()
    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for piece in pieces:
        pr0, pc0, pr1, pc1 = piece["bbox"]
        side = _piece_side(piece["bbox"], frame["bbox"])
        dr = 0
        dc = 0

        if side == "top":
            slot_r = 0
            slot_c = _match_span(pc0, pc1, col_spans)
            dr = row_spans[slot_r][1] - pr1
        elif side == "bottom":
            slot_r = len(row_spans) - 1
            slot_c = _match_span(pc0, pc1, col_spans)
            dr = row_spans[slot_r][0] - pr0
        elif side == "left":
            slot_r = _match_span(pr0, pr1, row_spans)
            slot_c = 0
            dc = col_spans[slot_c][1] - pc1
        else:
            slot_r = _match_span(pr0, pr1, row_spans)
            slot_c = len(col_spans) - 1
            dc = col_spans[slot_c][0] - pc0

        new_coords = piece["coords"] + np.array([dr, dc], dtype=int)
        slot_r0, slot_r1 = row_spans[slot_r]
        slot_c0, slot_c1 = col_spans[slot_c]
        assert np.all((slot_r0 <= new_coords[:, 0]) & (new_coords[:, 0] <= slot_r1))
        assert np.all((slot_c0 <= new_coords[:, 1]) & (new_coords[:, 1] <= slot_c1))
        assert np.all(out[new_coords[:, 0], new_coords[:, 1]] == bg)
        out[new_coords[:, 0], new_coords[:, 1]] = int(piece["color"])

    return out


def _grow_mask(max_h: int, max_w: int) -> np.ndarray:
    h = _randint(1, max_h)
    w = _randint(1, max_w)
    area_lo = max(1, (h * w + 2) // 3)
    area = _randint(area_lo, h * w)

    cells = {(int(_RNG.integers(0, h)), int(_RNG.integers(0, w)))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in _D4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier
        cells.add(frontier[int(_RNG.integers(0, len(frontier)))])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    used_rows = np.any(mask, axis=1)
    used_cols = np.any(mask, axis=0)
    return mask[np.ix_(used_rows, used_cols)]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    assert np.all(patch[mask] != color)
    assert np.all(patch[mask] == patch[mask][0]) if patch[mask].size else True
    patch[mask] = color


def _gen_attempt() -> np.ndarray | None:
    bg = _randint(0, 9)
    remaining = [c for c in range(10) if c != bg]
    frame_color = _choice(remaining)
    remaining = [c for c in remaining if c != frame_color]
    piece_colors = list(_RNG.choice(np.array(remaining, dtype=int), size=4, replace=False))

    slot_h = _randint(2, 6)
    slot_w = _randint(2, 6)

    frame_h = 2 * slot_h + 3
    frame_w = 2 * slot_w + 3
    top_margin = slot_h + _randint(2, 5)
    bottom_margin = slot_h + _randint(2, 5)
    left_margin = slot_w + _randint(2, 5)
    right_margin = slot_w + _randint(2, 5)

    h = frame_h + top_margin + bottom_margin
    w = frame_w + left_margin + right_margin
    grid = np.full((h, w), bg, dtype=int)

    fr0 = top_margin
    fc0 = left_margin
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    row_lines = [fr0, fr0 + slot_h + 1, fr1]
    col_lines = [fc0, fc0 + slot_w + 1, fc1]
    for r in row_lines:
        grid[r, fc0 : fc1 + 1] = frame_color
    for c in col_lines:
        grid[fr0 : fr1 + 1, c] = frame_color

    row_spans = [(row_lines[0] + 1, row_lines[1] - 1), (row_lines[1] + 1, row_lines[2] - 1)]
    col_spans = [(col_lines[0] + 1, col_lines[1] - 1), (col_lines[1] + 1, col_lines[2] - 1)]
    slot_sides = {
        (0, 0): ["top", "left"],
        (0, 1): ["top", "right"],
        (1, 0): ["bottom", "left"],
        (1, 1): ["bottom", "right"],
    }

    slot_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for color, slot in zip(piece_colors, slot_order):
        side = _choice(slot_sides[slot])
        mask = _grow_mask(slot_h, slot_w)
        ph, pw = mask.shape
        sr0, sr1 = row_spans[slot[0]]
        sc0, sc1 = col_spans[slot[1]]

        if side in ("top", "bottom"):
            left = _randint(sc0, sc1 - pw + 1)
            if side == "top":
                top = _randint(0, fr0 - ph - 1)
            else:
                top = _randint(fr1 + 2, h - ph)
        else:
            top = _randint(sr0, sr1 - ph + 1)
            if side == "left":
                left = _randint(0, fc0 - pw - 1)
            else:
                left = _randint(fc1 + 2, w - pw)

        patch = grid[top : top + ph, left : left + pw]
        if np.any(patch[mask] != bg):
            return None
        patch[mask] = int(color)

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-fragments input")
