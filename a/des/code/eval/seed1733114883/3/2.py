from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(arr != int(bg))
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_contiguous(seq: Iterable[int]) -> bool:
    items = [int(x) for x in seq]
    return bool(items) and all(items[i] + 1 == items[i + 1] for i in range(len(items) - 1))


def _all_signatures(length: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for run in (1, 2, 3):
        if run > int(length):
            break
        for start in range(int(length - run + 1)):
            out.append(tuple(range(int(start), int(start + run))))
    return out


def _reverse_signature(signature: tuple[int, ...], length: int) -> tuple[int, ...]:
    return tuple(int(length - 1 - x) for x in reversed(signature))


def _largest_monochrome_component(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    best_color = -1
    best_coords: np.ndarray | None = None
    best_area = -1
    for color in sorted(set(int(x) for x in np.unique(grid)) - {int(bg)}):
        for coords in _components(grid == int(color)):
            area = int(coords.shape[0])
            if area > best_area:
                best_area = area
                best_color = int(color)
                best_coords = coords
    assert best_coords is not None
    return int(best_color), best_coords


def _line_positions(grid: np.ndarray, frame_color: int, bbox4: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = bbox4
    rows = [int(r0)]
    cols = [int(c0)]
    for r in range(int(r0 + 1), int(r1)):
        if bool(np.all(grid[r, c0 : c1 + 1] == int(frame_color))):
            rows.append(int(r))
    for c in range(int(c0 + 1), int(c1)):
        if bool(np.all(grid[r0 : r1 + 1, c] == int(frame_color))):
            cols.append(int(c))
    rows.append(int(r1))
    cols.append(int(c1))
    return rows, cols


def _missing_signature(segment: np.ndarray, frame_color: int, bg: int) -> tuple[int, ...]:
    sig = tuple(int(i) for i, v in enumerate(segment.tolist()) if int(v) != int(frame_color))
    if not sig:
        return ()
    assert all(int(segment[i]) == int(bg) for i in sig)
    assert 1 <= len(sig) <= 3 and _is_contiguous(sig)
    return sig


def _slot_specs(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    row_lines: list[int],
    col_lines: list[int],
) -> tuple[list[dict], int, int]:
    cell_hs = [int(row_lines[i + 1] - row_lines[i] - 1) for i in range(len(row_lines) - 1)]
    cell_ws = [int(col_lines[i + 1] - col_lines[i] - 1) for i in range(len(col_lines) - 1)]
    assert len(set(cell_hs)) == 1 and len(set(cell_ws)) == 1
    cell_h = int(cell_hs[0])
    cell_w = int(cell_ws[0])

    top = int(row_lines[0])
    bottom = int(row_lines[-1])
    left = int(col_lines[0])
    right = int(col_lines[-1])

    slots: list[dict] = []
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            r0 = int(row_lines[i] + 1)
            r1 = int(row_lines[i + 1] - 1)
            c0 = int(col_lines[j] + 1)
            c1 = int(col_lines[j + 1] - 1)
            candidates: list[tuple[str, tuple[int, ...]]] = []
            if i == 0:
                sig = _missing_signature(grid[top, c0 : c1 + 1], frame_color, bg)
                if sig:
                    candidates.append(("top", sig))
            if i == len(row_lines) - 2:
                sig = _missing_signature(grid[bottom, c0 : c1 + 1], frame_color, bg)
                if sig:
                    candidates.append(("bottom", sig))
            if j == 0:
                sig = _missing_signature(grid[r0 : r1 + 1, left], frame_color, bg)
                if sig:
                    candidates.append(("left", sig))
            if j == len(col_lines) - 2:
                sig = _missing_signature(grid[r0 : r1 + 1, right], frame_color, bg)
                if sig:
                    candidates.append(("right", sig))
            assert len(candidates) == 1
            side, sig = candidates[0]
            slots.append(
                {
                    "key": (side, sig),
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                }
            )
    return slots, int(cell_h), int(cell_w)


def _parse_piece_orientation(
    patch: np.ndarray,
    bg: int,
    frame_color: int,
    cell_h: int,
    cell_w: int,
) -> list[tuple[str, tuple[int, ...], int]]:
    colors = sorted(set(int(x) for x in np.unique(patch)) - {int(bg), int(frame_color)})
    if len(colors) != 1:
        return []
    piece_color = int(colors[0])
    out: list[tuple[str, tuple[int, ...], int]] = []
    ph, pw = patch.shape

    def add(side: str, edge: np.ndarray) -> None:
        if not bool(np.all(np.isin(edge, [int(bg), int(frame_color)]))):
            return
        sig = tuple(int(i) for i, v in enumerate(edge.tolist()) if int(v) == int(frame_color))
        if not sig or len(sig) > 3 or not _is_contiguous(sig):
            return
        out.append((side, sig, piece_color))

    if (int(ph), int(pw)) == (int(cell_h + 1), int(cell_w)):
        if bool(np.all(patch[1:, :] == int(piece_color))) and not bool(np.any(patch[0, :] == int(piece_color))):
            add("top", patch[0, :])
        if bool(np.all(patch[:-1, :] == int(piece_color))) and not bool(np.any(patch[-1, :] == int(piece_color))):
            add("bottom", patch[-1, :])

    if (int(ph), int(pw)) == (int(cell_h), int(cell_w + 1)):
        if bool(np.all(patch[:, 1:] == int(piece_color))) and not bool(np.any(patch[:, 0] == int(piece_color))):
            add("left", patch[:, 0])
        if bool(np.all(patch[:, :-1] == int(piece_color))) and not bool(np.any(patch[:, -1] == int(piece_color))):
            add("right", patch[:, -1])

    dedup: list[tuple[str, tuple[int, ...], int]] = []
    seen = set()
    for item in out:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match grey-toothed monochrome pieces to uniquely notched frame cells, rotate them,
    and move them into the framed region on the right.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_coords = _largest_monochrome_component(g, bg)
    frame_bbox = _bbox(frame_coords)
    row_lines, col_lines = _line_positions(g, frame_color, frame_bbox)
    slots, cell_h, cell_w = _slot_specs(g, bg, frame_color, row_lines, col_lines)
    slot_keys = {slot["key"] for slot in slots}

    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[frame_coords[:, 0], frame_coords[:, 1]] = True
    piece_components = _components((g != int(bg)) & (~frame_mask))
    assert len(piece_components) == len(slots)

    piece_by_key: dict[tuple[str, tuple[int, ...]], int] = {}
    for coords in piece_components:
        r0, c0, r1, c1 = _bbox(coords)
        patch = _crop_non_bg(g[r0 : r1 + 1, c0 : c1 + 1], bg)
        candidates: dict[tuple[str, tuple[int, ...]], int] = {}
        for k in range(4):
            rot = _crop_non_bg(np.rot90(patch, k), bg)
            for side, sig, color in _parse_piece_orientation(rot, bg, frame_color, cell_h, cell_w):
                candidates[(side, sig)] = int(color)
        matches = [key for key in slot_keys if key in candidates]
        assert len(matches) == 1
        piece_by_key[matches[0]] = int(candidates[matches[0]])

    assert set(piece_by_key) == slot_keys

    out = np.full_like(g, int(bg))
    r0, c0, r1, c1 = frame_bbox
    for r in row_lines:
        out[r, c0 : c1 + 1] = int(frame_color)
    for c in col_lines:
        out[r0 : r1 + 1, c] = int(frame_color)
    for slot in slots:
        color = int(piece_by_key[slot["key"]])
        out[slot["r0"] : slot["r1"] + 1, slot["c0"] : slot["c1"] + 1] = int(color)
    return out


def _make_piece(
    cell_h: int,
    cell_w: int,
    piece_color: int,
    side: str,
    signature: tuple[int, ...],
) -> np.ndarray:
    if side == "top":
        patch = np.full((cell_h + 1, cell_w), BG, dtype=int)
        patch[1:, :] = int(piece_color)
        patch[0, list(signature)] = FRAME
        return patch
    if side == "bottom":
        patch = np.full((cell_h + 1, cell_w), BG, dtype=int)
        patch[:-1, :] = int(piece_color)
        patch[-1, list(signature)] = FRAME
        return patch
    if side == "left":
        patch = np.full((cell_h, cell_w + 1), BG, dtype=int)
        patch[:, 1:] = int(piece_color)
        patch[list(signature), 0] = FRAME
        return patch
    assert side == "right"
    patch = np.full((cell_h, cell_w + 1), BG, dtype=int)
    patch[:, :-1] = int(piece_color)
    patch[list(signature), -1] = FRAME
    return patch


def _draw_frame(
    canvas: np.ndarray,
    top: int,
    left: int,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
) -> tuple[list[int], list[int]]:
    row_lines = [int(top + i * (cell_h + 1)) for i in range(rows + 1)]
    col_lines = [int(left + j * (cell_w + 1)) for j in range(cols + 1)]
    for r in row_lines:
        canvas[r, left : col_lines[-1] + 1] = FRAME
    for c in col_lines:
        canvas[top : row_lines[-1] + 1, c] = FRAME
    return row_lines, col_lines


def _placeable(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > h0 or c0 + w > w0:
        return False
    rs = max(0, int(r0 - pad))
    re = min(h0, int(r0 + h + pad))
    cs = max(0, int(c0 - pad))
    ce = min(w0, int(c0 + w + pad))
    return not bool(occupied[rs:re, cs:ce].any())


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(BG)
    view[mask] = patch[mask]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    layout = random.choice(
        [
            ("horizontal", 2, 2),
            ("vertical", 2, 2),
            ("vertical", 3, 2),
        ]
    )
    mode, rows, cols = layout
    cell_h = random.choice([4, 5])
    cell_w = random.choice([4, 5])
    n_pieces = int(rows * cols)

    sig_len = int(cell_w if mode == "horizontal" else cell_h)
    colors = random.sample(PIECE_COLORS, n_pieces)

    def choose_split(count_a: int, count_b: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        all_sigs = _all_signatures(sig_len)
        for _ in range(200):
            side_a = random.sample(all_sigs, count_a)
            forbidden = {_reverse_signature(sig, sig_len) for sig in side_a}
            remaining = [sig for sig in all_sigs if sig not in side_a and sig not in forbidden]
            if len(remaining) >= count_b:
                side_b = random.sample(remaining, count_b)
                return side_a, side_b
        raise AssertionError("could not build non-ambiguous signature sets")

    if mode == "horizontal":
        top_sigs, bottom_sigs = choose_split(cols, cols)
        random.shuffle(top_sigs)
        random.shuffle(bottom_sigs)
    else:
        left_sigs, right_sigs = choose_split(rows, rows)
        random.shuffle(left_sigs)
        random.shuffle(right_sigs)

    frame_h = int(rows * cell_h + rows + 1)
    frame_w = int(cols * cell_w + cols + 1)
    piece_box = int(max(cell_h, cell_w) + 1)
    piece_cols = 2
    piece_rows = int((n_pieces + piece_cols - 1) // piece_cols)
    left_w = int(piece_cols * piece_box + (piece_cols - 1))
    left_h = int(piece_rows * piece_box + (piece_rows - 1))

    h_min = max(22, frame_h + 2, left_h + 2)
    w_min = max(22, left_w + frame_w + 2)
    assert h_min <= 28 and w_min <= 28
    h = random.randint(h_min, 28)
    w = random.randint(w_min, 28)

    max_right_margin = min(2, int(w - frame_w - left_w - 1))
    right_margin = random.randint(1, max_right_margin)
    top = random.randint(1, h - frame_h - 1)
    left = int(w - frame_w - right_margin)
    assert left >= left_w

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    row_lines, col_lines = _draw_frame(inp, top, left, rows, cols, cell_h, cell_w)
    _draw_frame(expected, top, left, rows, cols, cell_h, cell_w)

    pieces: list[np.ndarray] = []
    for i in range(rows):
        for j in range(cols):
            if mode == "horizontal":
                side = "top" if i == 0 else "bottom"
                signature = tuple(int(x) for x in (top_sigs.pop() if side == "top" else bottom_sigs.pop()))
            else:
                side = "left" if j == 0 else "right"
                signature = tuple(int(x) for x in (left_sigs.pop() if side == "left" else right_sigs.pop()))
            color = int(colors.pop())

            r0 = int(row_lines[i] + 1)
            r1 = int(row_lines[i + 1] - 1)
            c0 = int(col_lines[j] + 1)
            c1 = int(col_lines[j + 1] - 1)
            expected[r0 : r1 + 1, c0 : c1 + 1] = int(color)

            if side == "top":
                inp[row_lines[0], c0 + np.array(signature, dtype=int)] = BG
            elif side == "bottom":
                inp[row_lines[-1], c0 + np.array(signature, dtype=int)] = BG
            elif side == "left":
                inp[r0 + np.array(signature, dtype=int), col_lines[0]] = BG
            else:
                inp[r0 + np.array(signature, dtype=int), col_lines[-1]] = BG

            piece = _make_piece(cell_h, cell_w, color, side, signature)
            piece = np.rot90(piece, random.randint(0, 3))
            pieces.append(piece)

    random.shuffle(pieces)
    region_top = random.randint(0, h - left_h)
    region_left = random.randint(0, left - left_w - 1)
    boxes = []
    for br in range(piece_rows):
        for bc in range(piece_cols):
            boxes.append(
                (
                    int(region_top + br * (piece_box + 1)),
                    int(region_left + bc * (piece_box + 1)),
                )
            )
    random.shuffle(boxes)

    for patch, (box_r, box_c) in zip(pieces, boxes, strict=False):
        ph, pw = patch.shape
        dr = random.randint(0, piece_box - ph)
        dc = random.randint(0, piece_box - pw)
        _paste_patch(inp, patch, int(box_r + dr), int(box_c + dc))

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, expected = _generate_case()
        if np.array_equal(main(inp), expected):
            return inp
    raise AssertionError("failed to generate a valid puzzle instance")
