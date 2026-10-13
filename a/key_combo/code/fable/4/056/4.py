from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(1, 10))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    return r1 - r0 + 1, c1 - c0 + 1


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = [start]
    while int(mask.sum()) < area:
        r, c = random.choice(frontier)
        neighbors = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            frontier.remove((r, c))
            if not frontier:
                break
            continue
        chosen = random.choice(neighbors)
        mask[chosen] = True
        frontier.append(chosen)
    return mask


def _piece_from_mask(mask: np.ndarray) -> dict:
    cropped, (r_off, c_off) = _crop_mask(mask)
    boundary = []
    h, w = map(int, cropped.shape)
    for r, c in np.argwhere(cropped):
        rr = int(r)
        cc = int(c)
        if sum(
            0 <= rr + dr < h and 0 <= cc + dc < w and bool(cropped[rr + dr, cc + dc])
            for dr, dc in DIRS4
        ) == 0:
            continue
        if any(
            not (0 <= rr + dr < h and 0 <= cc + dc < w) or not bool(cropped[rr + dr, cc + dc])
            for dr, dc in DIRS4
        ):
            boundary.append((rr, cc))
    connector = random.choice(boundary)
    rel = {(int(r - connector[0]), int(c - connector[1])) for r, c in np.argwhere(cropped)}
    return {
        "mask": cropped,
        "shape": tuple(map(int, cropped.shape)),
        "connector": connector,
        "rel_cells": rel,
        "offset": (r_off, c_off),
    }


def _render_piece(piece: dict, body_color: int, connector_color: int) -> np.ndarray:
    arr = np.zeros(piece["shape"], dtype=int)
    arr[piece["mask"]] = body_color
    cr, cc = piece["connector"]
    arr[int(cr), int(cc)] = connector_color
    return arr


def _docking_candidates(piece_a: dict, piece_b: dict) -> list[dict]:
    a_cells = piece_a["rel_cells"]
    b_cells = piece_b["rel_cells"]
    candidates = []
    for dr, dc in DIRS4:
        shifted_b = {(r + dr, c + dc) for r, c in b_cells}
        if a_cells & shifted_b:
            continue
        union = a_cells | shifted_b
        candidates.append(
            {
                "dir": (dr, dc),
                "union": union,
                "bbox": _bbox(union),
                "size": _bbox_size(union),
            }
        )
    return candidates


def _find_marker_window(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        coords = np.argwhere(grid == color)
        if int(coords.shape[0]) != 4:
            continue
        cells = {tuple(map(int, rc)) for rc in coords}
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if cells == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            candidates.append((color, (r0, c0, r1, c1)))
    assert len(candidates) == 1, "expected one four-corner marker window"
    return candidates[0]


def _extract_pieces(grid: np.ndarray, bg: int, marker_color: int) -> tuple[int, int, list[dict]]:
    counts = {int(color): int(np.sum(grid == color)) for color in set(map(int, np.unique(grid))) - {bg, marker_color}}
    assert len(counts) == 2, "expected one body color and one connector color"
    connector_candidates = [color for color, count in counts.items() if count == 2]
    assert len(connector_candidates) == 1, "expected connector color to appear twice"
    connector_color = int(connector_candidates[0])
    body_color = int([color for color in counts if color != connector_color][0])

    movable_mask = (grid == body_color) | (grid == connector_color)
    comps = _components(movable_mask)
    assert len(comps) == 2, "expected exactly two movable sprites"

    pieces = []
    for comp in comps:
        comp_set = set(comp)
        connector_cells = [(r, c) for r, c in comp if int(grid[r, c]) == connector_color]
        assert len(connector_cells) == 1, "expected one connector per sprite"
        connector = connector_cells[0]
        rel_cells = {(r - connector[0], c - connector[1]) for r, c in comp_set}
        pieces.append({"abs_cells": comp_set, "connector": connector, "rel_cells": rel_cells})
    return body_color, connector_color, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """Dock the two connector-marked fragments in the only way whose joint bbox matches the corner window, then move the assembly into that window.

    Input structure
    - The background is a single dominant color.
    - Four same-color singleton markers sit at the corners of a rectangle and
      define a target window.
    - Two same-body-color sprites each contain one connector-color pixel.

    Rule
    - Enumerate every translation that makes the two connector pixels 4-neighbor
      adjacent while keeping the two sprites disjoint.
    - Compute the bbox of each docked union and select the unique one whose size
      exactly equals the interior size of the marker rectangle.
    - Recolor both connector pixels to the body color, erase the original sprite
      locations, and translate the chosen union so its bbox fills the window interior.

    Key insight
    - The connector pixels only specify local adjacency; the distant corner
      markers disambiguate which docking orientation is correct by requiring an
      exact bbox-size match with the window interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    marker_color, frame = _find_marker_window(g, bg)
    body_color, _connector_color, pieces = _extract_pieces(g, bg, marker_color)

    fr0, fc0, fr1, fc1 = frame
    target_size = (fr1 - fr0 - 1, fc1 - fc0 - 1)
    assert target_size[0] >= 1 and target_size[1] >= 1

    candidates = _docking_candidates(pieces[0], pieces[1])
    matches = [cand for cand in candidates if cand["size"] == target_size]
    assert len(matches) == 1, "window should resolve the docking ambiguity uniquely"
    chosen = matches[0]

    br0, bc0, _br1, _bc1 = chosen["bbox"]
    shift_r = (fr0 + 1) - br0
    shift_c = (fc0 + 1) - bc0

    out = g.copy()
    movable = (out == body_color) | ((out != bg) & (out != marker_color) & (out != body_color))
    out[movable] = bg
    for r, c in chosen["union"]:
        rr = int(r + shift_r)
        cc = int(c + shift_c)
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
        assert int(out[rr, cc]) == bg
        out[rr, cc] = body_color

    return out


def _sample_piece() -> dict | None:
    for _ in range(80):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        high = min(height * width, 8)
        low = min(high, max(3, high - 3))
        area = random.randint(low, high)
        mask = _random_connected_mask(height, width, area)
        if int(mask.sum()) < 3:
            continue
        cropped, _ = _crop_mask(mask)
        if int(cropped.sum()) != int(mask.sum()):
            mask = cropped
        piece = _piece_from_mask(mask)
        if piece["shape"][0] >= 2 and piece["shape"][1] >= 2:
            return piece
    return None


def _sample_target_assembly() -> tuple[dict, dict, dict] | None:
    for _ in range(160):
        piece_a = _sample_piece()
        piece_b = _sample_piece()
        if piece_a is None or piece_b is None:
            continue
        candidates = _docking_candidates(piece_a, piece_b)
        if len(candidates) < 2:
            continue
        unique_size_candidates = [
            cand
            for cand in candidates
            if sum(other["size"] == cand["size"] for other in candidates) == 1
        ]
        if not unique_size_candidates:
            continue
        chosen = random.choice(unique_size_candidates)
        target_h, target_w = chosen["size"]
        if not (3 <= target_h <= 8 and 3 <= target_w <= 8):
            continue
        return piece_a, piece_b, chosen
    return None


def _reserve(mask: np.ndarray, top: int, left: int, height: int, width: int, margin: int = 1) -> None:
    h, w = map(int, mask.shape)
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + height + margin)
    c1 = min(w, left + width + margin)
    mask[r0:r1, c0:c1] = True


def _pick_position(
    reserved: np.ndarray, height: int, width: int, tries: int = 120
) -> tuple[int, int] | None:
    h, w = map(int, reserved.shape)
    if height > h or width > w:
        return None
    for _ in range(tries):
        top = random.randint(0, h - height)
        left = random.randint(0, w - width)
        if not np.any(reserved[top : top + height, left : left + width]):
            return top, left
    return None


def _paint_piece(grid: np.ndarray, top: int, left: int, piece_arr: np.ndarray) -> None:
    h, w = map(int, piece_arr.shape)
    patch = grid[top : top + h, left : left + w]
    assert np.all(patch == 0)
    nonzero = piece_arr != 0
    patch[nonzero] = piece_arr[nonzero]


def _paint_window_markers(
    grid: np.ndarray, top: int, left: int, height: int, width: int, marker_color: int
) -> None:
    grid[top, left] = marker_color
    grid[top, left + width - 1] = marker_color
    grid[top + height - 1, left] = marker_color
    grid[top + height - 1, left + width - 1] = marker_color


def _expected_output(
    shape: tuple[int, int],
    frame_top: int,
    frame_left: int,
    frame_h: int,
    frame_w: int,
    marker_color: int,
    body_color: int,
    chosen: dict,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _paint_window_markers(out, frame_top, frame_left, frame_h, frame_w, marker_color)
    br0, bc0, _br1, _bc1 = chosen["bbox"]
    shift_r = (frame_top + 1) - br0
    shift_c = (frame_left + 1) - bc0
    for r, c in chosen["union"]:
        out[int(r + shift_r), int(c + shift_c)] = body_color
    return out


def _gen_attempt() -> np.ndarray | None:
    assembled = _sample_target_assembly()
    if assembled is None:
        return None
    piece_a, piece_b, chosen = assembled

    body_color, connector_color, marker_color = random.sample(PALETTE, 3)
    piece_a_arr = _render_piece(piece_a, body_color, connector_color)
    piece_b_arr = _render_piece(piece_b, body_color, connector_color)

    frame_h = chosen["size"][0] + 2
    frame_w = chosen["size"][1] + 2

    min_h = max(frame_h, piece_a_arr.shape[0], piece_b_arr.shape[0]) + 6
    min_w = max(frame_w, piece_a_arr.shape[1], piece_b_arr.shape[1]) + 6
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        reserved = np.zeros((height, width), dtype=bool)

        frame_pos = _pick_position(reserved, frame_h, frame_w)
        if frame_pos is None:
            continue
        _reserve(reserved, frame_pos[0], frame_pos[1], frame_h, frame_w, margin=1)

        pos_a = _pick_position(reserved, piece_a_arr.shape[0], piece_a_arr.shape[1])
        if pos_a is None:
            continue
        _reserve(reserved, pos_a[0], pos_a[1], piece_a_arr.shape[0], piece_a_arr.shape[1], margin=1)

        pos_b = _pick_position(reserved, piece_b_arr.shape[0], piece_b_arr.shape[1])
        if pos_b is None:
            continue

        inp = np.zeros((height, width), dtype=int)
        _paint_window_markers(inp, frame_pos[0], frame_pos[1], frame_h, frame_w, marker_color)
        _paint_piece(inp, pos_a[0], pos_a[1], piece_a_arr)
        _paint_piece(inp, pos_b[0], pos_b[1], piece_b_arr)

        expected = _expected_output(
            inp.shape,
            frame_pos[0],
            frame_pos[1],
            frame_h,
            frame_w,
            marker_color,
            body_color,
            chosen,
        )
        try:
            if not np.array_equal(main(inp), expected):
                continue
        except AssertionError:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid docking-window ARC input")
