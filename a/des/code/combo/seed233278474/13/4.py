from __future__ import annotations

import random
from collections import Counter

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SIDES = ("top", "bottom", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or bool(seen[r0, c0]):
                continue

            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for dr, dc in _NEIGHBORS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            rows = coords[:, 0]
            cols = coords[:, 1]
            rr0 = int(rows.min())
            rr1 = int(rows.max())
            cc0 = int(cols.min())
            cc1 = int(cols.max())
            comps.append(
                {
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": (rr0, cc0, rr1, cc1),
                    "h": int(rr1 - rr0 + 1),
                    "w": int(cc1 - cc0 + 1),
                }
            )

    return comps


def _is_straight_line(comp: dict) -> bool:
    return int(comp["area"]) == int(comp["h"]) * int(comp["w"]) and (int(comp["h"]) == 1 or int(comp["w"]) == 1)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords = {(int(r0), int(c)) for c in range(int(c0), int(c1) + 1)}
    coords |= {(int(r1), int(c)) for c in range(int(c0), int(c1) + 1)}
    coords |= {(int(r), int(c0)) for r in range(int(r0), int(r1) + 1)}
    coords |= {(int(r), int(c1)) for r in range(int(r0), int(r1) + 1)}
    return coords


def _gap_from_frame(frame_coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in frame_coords]
    cols = [c for _, c in frame_coords]
    r0 = int(min(rows))
    r1 = int(max(rows))
    c0 = int(min(cols))
    c1 = int(max(cols))

    border = _border_coords(r0, c0, r1, c1)
    assert frame_coords <= border, "frame pixels must lie on a rectangle border"
    missing = sorted(border - frame_coords)
    assert missing, "each frame must have exactly one missing segment"

    row_set = {r for r, _ in missing}
    col_set = {c for _, c in missing}
    assert len(row_set) == 1 or len(col_set) == 1, "gap must be a straight segment"

    if len(row_set) == 1:
        row = next(iter(row_set))
        cols_sorted = sorted(c for _, c in missing)
        assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1)), "gap must be contiguous"
        return [(int(row), int(c)) for c in cols_sorted]

    col = next(iter(col_set))
    rows_sorted = sorted(r for r, _ in missing)
    assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1)), "gap must be contiguous"
    return [(int(r), int(col)) for r in rows_sorted]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each same-color line fragment into the unique missing segment of its matching rectangular frame layer.

    Input structure:
    - Background is black.
    - One side of the grid contains several nested, one-pixel-thick rectangular frame layers in different colors.
    - Each colored layer is missing one contiguous segment on one side.
    - Elsewhere there is exactly one same-color straight line fragment for each layer.

    Rule:
    - Match every fragment to the frame layer with the same color.
    - Detect the layer's missing border segment.
    - Translate the fragment into that gap without rotating it.
    - Clear the fragment's original position to background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        comps = _find_components(g == int(color))
        assert len(comps) >= 2, "each color must contain a frame fragment and a loose piece"

        line_comps = [comp for comp in comps if _is_straight_line(comp)]
        assert line_comps, "each color must include a straight fragment"
        piece = min(line_comps, key=lambda comp: (int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))

        frame_sets = []
        for comp in comps:
            if comp is piece:
                continue
            frame_sets.extend((int(r), int(c)) for r, c in comp["coords"])
        assert frame_sets, "missing frame component"

        gap = _gap_from_frame(set(frame_sets))
        piece_length = int(piece["area"])
        piece_is_horizontal = int(piece["h"]) == 1 and int(piece["w"]) == piece_length
        piece_is_vertical = int(piece["w"]) == 1 and int(piece["h"]) == piece_length
        gap_is_horizontal = len({r for r, _ in gap}) == 1
        gap_is_vertical = len({c for _, c in gap}) == 1

        assert len(gap) == piece_length, "piece length must match its frame gap"
        assert piece_length == 1 or (
            (piece_is_horizontal and gap_is_horizontal) or (piece_is_vertical and gap_is_vertical)
        ), "piece orientation must match the gap"

        out[piece["coords"][:, 0], piece["coords"][:, 1]] = int(bg)
        for r, c in gap:
            out[int(r), int(c)] = int(color)

    return out


# --------------- generator ---------------

_LAYER_COUNT_RANGE = (2, 4)
_OUTER_EXTRA_RANGE = (0, 7)
_FRAME_TOP_MARGIN_RANGE = (0, 4)
_FRAME_BOTTOM_MARGIN_RANGE = (0, 4)
_FRAME_GAP_COL_RANGE = (2, 4)
_RIGHT_MARGIN_RANGE = (1, 3)
_LEFT_EXTRA_WIDTH_RANGE = (2, 7)
_HEIGHT_EXTRA_RANGE = (0, 6)
_PIECE_PADDING = 1
_MAX_ATTEMPTS = 400


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    assert view.shape == patch.shape
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _draw_border(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[int(r0), int(c0) : int(c1) + 1] = int(color)
    grid[int(r1), int(c0) : int(c1) + 1] = int(color)
    grid[int(r0) : int(r1) + 1, int(c0)] = int(color)
    grid[int(r0) : int(r1) + 1, int(c1)] = int(color)


def _sample_gap(r0: int, c0: int, r1: int, c1: int) -> tuple[str, list[tuple[int, int]]]:
    side = random.choice(_SIDES)
    if side in ("top", "bottom"):
        side_len = int(c1 - c0 + 1)
        gap_len = random.randint(1, side_len - 2)
        start = random.randint(1, side_len - gap_len - 1)
        row = int(r0 if side == "top" else r1)
        gap = [(row, int(c0 + start + i)) for i in range(gap_len)]
    else:
        side_len = int(r1 - r0 + 1)
        gap_len = random.randint(1, side_len - 2)
        start = random.randint(1, side_len - gap_len - 1)
        col = int(c0 if side == "left" else c1)
        gap = [(int(r0 + start + i), col) for i in range(gap_len)]
    return side, gap


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, int(r0 - padding))
    rr1 = min(H, int(r0 + h + padding))
    cc0 = max(0, int(c0 - padding))
    cc1 = min(W, int(c0 + w + padding))
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_pieces(H: int, left_w: int, pieces: list[np.ndarray]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((H, left_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(pieces)

    order = list(range(len(pieces)))
    random.shuffle(order)
    order.sort(key=lambda idx: (pieces[idx].shape[0] * pieces[idx].shape[1], max(pieces[idx].shape)), reverse=True)

    for idx in order:
        patch = pieces[idx]
        ph, pw = map(int, patch.shape)
        candidates = [(r, c) for r in range(H - ph + 1) for c in range(left_w - pw + 1)]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            if _can_place(occupied, patch, int(r0), int(c0), _PIECE_PADDING):
                chosen = (int(r0), int(c0))
                break
        if chosen is None:
            return None
        placements[idx] = chosen
        r0, c0 = chosen
        occupied[int(r0) : int(r0) + ph, int(c0) : int(c0) + pw] |= patch != 0

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _make_piece(color: int, gap: list[tuple[int, int]]) -> np.ndarray:
    if len({r for r, _ in gap}) == 1:
        return np.full((1, len(gap)), int(color), dtype=int)
    return np.full((len(gap), 1), int(color), dtype=int)


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.randint(*_LAYER_COUNT_RANGE)
    min_outer = 4 * n_layers - 1
    outer_h = min_outer + random.randint(*_OUTER_EXTRA_RANGE)
    outer_w = min_outer + random.randint(*_OUTER_EXTRA_RANGE)
    colors = random.sample(range(1, 10), n_layers)

    layer_specs = []
    piece_shapes: list[np.ndarray] = []
    piece_heights: list[int] = []
    max_piece_w = 1

    for layer in range(n_layers):
        h = int(outer_h - 4 * layer)
        w = int(outer_w - 4 * layer)
        if h < 3 or w < 3:
            return None
        side, gap = _sample_gap(0, 0, h - 1, w - 1)
        piece = _make_piece(colors[layer], gap)
        layer_specs.append({"color": int(colors[layer]), "h": h, "w": w, "gap": gap, "side": side})
        piece_shapes.append(piece)
        piece_heights.append(int(piece.shape[0]))
        max_piece_w = max(max_piece_w, int(piece.shape[1]))

    left_w = max(6, max_piece_w + random.randint(*_LEFT_EXTRA_WIDTH_RANGE))
    min_piece_stack = sum(piece_heights) + _PIECE_PADDING * max(0, n_layers - 1)
    frame_band = outer_h + random.randint(*_HEIGHT_EXTRA_RANGE)
    H = max(min_piece_stack, frame_band) + 1

    top_margin = random.randint(*_FRAME_TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*_FRAME_BOTTOM_MARGIN_RANGE)
    H = max(H, outer_h + top_margin + bottom_margin)
    frame_r0 = random.randint(0, H - outer_h)

    frame_c0 = left_w + random.randint(*_FRAME_GAP_COL_RANGE)
    W = frame_c0 + outer_w + random.randint(*_RIGHT_MARGIN_RANGE)

    inp = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    for layer, spec in enumerate(layer_specs):
        color = int(spec["color"])
        r0 = int(frame_r0 + 2 * layer)
        c0 = int(frame_c0 + 2 * layer)
        r1 = int(r0 + spec["h"] - 1)
        c1 = int(c0 + spec["w"] - 1)
        _draw_border(inp, color, r0, c0, r1, c1)
        _draw_border(out, color, r0, c0, r1, c1)
        for gr, gc in spec["gap"]:
            inp[int(r0 + gr), int(c0 + gc)] = 0

    placements = _place_pieces(H, left_w, piece_shapes)
    if placements is None:
        return None

    for piece, (r0, c0) in zip(piece_shapes, placements):
        _paste(inp, piece, int(r0), int(c0))

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_ATTEMPTS} attempts")
