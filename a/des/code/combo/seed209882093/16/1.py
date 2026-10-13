from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_CW_TURNS = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
}
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _cells_to_mask(
    cells: list[tuple[int, int]] | set[tuple[int, int]]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _full_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    return border


def _contiguous(values: list[int]) -> bool:
    if not values:
        return False
    return values == list(range(values[0], values[-1] + 1))


def _marker_side(marker: tuple[int, int], gap_cells: list[tuple[int, int]]) -> str:
    gr0, gc0, gr1, gc1 = _bbox(gap_cells)
    mr, mc = marker
    if mr < gr0:
        return "up"
    if mr > gr1:
        return "down"
    if mc < gc0:
        return "left"
    if mc > gc1:
        return "right"
    raise AssertionError("marker must be adjacent to, not inside, the gap")


def _parse_frame(g: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    cells = set(comp)
    r0, c0, r1, c1 = _bbox(comp)
    assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5

    border = _full_border(r0, c0, r1, c1)
    assert cells <= border, "grey component must be a border-only rectangle"

    missing = sorted(border - cells)
    assert missing, "frame must have a gap"

    candidate_sides = []
    if all(r == r0 for r, _ in missing):
        candidate_sides.append("top")
    if all(r == r1 for r, _ in missing):
        candidate_sides.append("bottom")
    if all(c == c0 for _, c in missing):
        candidate_sides.append("left")
    if all(c == c1 for _, c in missing):
        candidate_sides.append("right")
    assert len(candidate_sides) == 1, "gap must lie on exactly one side"
    side = candidate_sides[0]

    if side in {"top", "bottom"}:
        cols = sorted(c for _, c in missing)
        assert _contiguous(cols), "horizontal gap must be contiguous"
        strip_row = r0 + 1 if side == "top" else r1 - 1
        marker_band = [(strip_row, c) for c in cols]
    else:
        rows = sorted(r for r, _ in missing)
        assert _contiguous(rows), "vertical gap must be contiguous"
        strip_col = c0 + 1 if side == "left" else c1 - 1
        marker_band = [(r, strip_col) for r in rows]

    marker_candidates = [
        (r, c) for r, c in marker_band if int(g[r, c]) not in {BLACK, GREY}
    ]
    assert len(marker_candidates) == 1, "each gap must have exactly one inner marker"
    marker = marker_candidates[0]

    gap_mask, _ = _cells_to_mask(missing)
    direction = _marker_side(marker, missing)
    return {
        "gap_cells": missing,
        "gap_mask": gap_mask,
        "marker": marker,
        "direction": direction,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored fragment to the gap with the same-color marker, rotate by marker-side,
    and restore the grey rectangular frames.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = [_parse_frame(g, comp) for comp in _components(g == GREY)]
    marker_colors = {int(g[frame["marker"]]) for frame in frames}
    all_piece_colors = set(map(int, np.unique(g))) - {BLACK, GREY}
    assert marker_colors == all_piece_colors, "every non-grey, non-black color must be a marker/piece pair"
    assert len(marker_colors) == len(frames), "marker colors must be unique"

    used_colors: set[int] = set()
    for frame in frames:
        marker = frame["marker"]
        color = int(g[marker])
        assert color not in used_colors
        used_colors.add(color)

        piece_mask = g == color
        piece_mask[marker] = False
        piece_comps = _components(piece_mask)
        assert len(piece_comps) == 1, "each color must have exactly one piece"
        piece_cells = piece_comps[0]
        src_mask, _ = _cells_to_mask(piece_cells)

        turns = SIDE_TO_CW_TURNS[frame["direction"]]
        rotated = np.rot90(src_mask, -turns)
        assert np.array_equal(rotated, frame["gap_mask"]), "piece shape must match its gap after rotation"

        for r, c in piece_cells:
            out[r, c] = BLACK
        out[marker] = BLACK
        for r, c in frame["gap_cells"]:
            out[r, c] = GREY

    return out


def _build_frame_patch(
    height: int, width: int, side: str, gap_start: int, gap_len: int, color: int
) -> tuple[np.ndarray, np.ndarray, int]:
    out_patch = np.zeros((height, width), dtype=int)
    out_patch[0, :] = GREY
    out_patch[-1, :] = GREY
    out_patch[:, 0] = GREY
    out_patch[:, -1] = GREY

    inp_patch = out_patch.copy()

    if side == "top":
        cols = list(range(gap_start, gap_start + gap_len))
        inp_patch[0, cols] = BLACK
        marker_index = random.choice(cols)
        inp_patch[1, marker_index] = color
    elif side == "bottom":
        cols = list(range(gap_start, gap_start + gap_len))
        inp_patch[-1, cols] = BLACK
        marker_index = random.choice(cols)
        inp_patch[-2, marker_index] = color
    elif side == "left":
        rows = list(range(gap_start, gap_start + gap_len))
        inp_patch[rows, 0] = BLACK
        marker_index = random.choice(rows)
        inp_patch[marker_index, 1] = color
    elif side == "right":
        rows = list(range(gap_start, gap_start + gap_len))
        inp_patch[rows, -1] = BLACK
        marker_index = random.choice(rows)
        inp_patch[marker_index, -2] = color
    else:
        raise ValueError(side)

    return inp_patch, out_patch, gap_len


def _bbox_is_free(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = reserved.shape
    if top < 1 or left < 1 or top + height > h - 1 or left + width > w - 1:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve_bbox(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    reserved[r0:r1, c0:c1] = True


def _try_generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    max_frames = 4 if height * width < 650 else 5
    n_frames = random.randint(2, max_frames)
    colors = random.sample(COLOR_POOL, n_frames)

    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    piece_specs: list[dict[str, int]] = []

    for color in colors:
        placed = False
        for _ in range(120):
            fh = random.randint(5, 8 if n_frames >= 4 else 10)
            fw = random.randint(5, 8 if n_frames >= 4 else 10)
            if fh >= height - 2 or fw >= width - 2:
                continue

            side = random.choice(["top", "bottom", "left", "right"])
            side_len = fw if side in {"top", "bottom"} else fh
            if side_len < 4:
                continue

            gap_len = random.randint(2, side_len - 2)
            max_start = side_len - gap_len - 1
            if max_start < 1:
                continue
            gap_start = random.randint(1, max_start)
            frame_in, frame_out, piece_len = _build_frame_patch(fh, fw, side, gap_start, gap_len, color)

            top = random.randint(1, height - fh - 1)
            left = random.randint(1, width - fw - 1)
            if not _bbox_is_free(reserved, top, left, fh, fw, pad=1):
                continue

            inp[top : top + fh, left : left + fw] = frame_in
            out[top : top + fh, left : left + fw] = frame_out
            _reserve_bbox(reserved, top, left, fh, fw, pad=1)
            piece_specs.append({"color": color, "length": piece_len})
            placed = True
            break
        if not placed:
            return None

    piece_specs.sort(key=lambda spec: -spec["length"])
    for spec in piece_specs:
        placed = False
        for _ in range(160):
            piece_len = spec["length"]
            top = random.randint(1, height - 2)
            left = random.randint(1, width - piece_len - 1)
            if not _bbox_is_free(reserved, top, left, 1, piece_len, pad=1):
                continue
            inp[top, left : left + piece_len] = spec["color"]
            _reserve_bbox(reserved, top, left, 1, piece_len, pad=1)
            placed = True
            break
        if not placed:
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _try_generate_case()
        if candidate is None:
            continue
        inp, expected = candidate
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
