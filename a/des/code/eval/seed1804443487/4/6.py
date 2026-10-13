from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_connected_components(mask)) == 1


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_parts(grid: np.ndarray, color: int) -> list[dict]:
    comps = _connected_components(grid == int(color))
    parts = []
    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rel = coords - np.array([r0, c0], dtype=int)
        parts.append(
            {
                "cells": rel.astype(int),
                "shape": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
                "area": int(rel.shape[0]),
            }
        )
    return parts


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            pairs.add((int(h), int(w)))
            pairs.add((int(w), int(h)))
    return sorted(pairs, key=lambda t: (max(t), min(t)))


def _assemble_color(parts: list[dict], color: int) -> tuple[np.ndarray, np.ndarray]:
    assert 2 <= len(parts) <= 3
    total_area = sum(int(part["area"]) for part in parts)
    max_h = max(int(part["shape"][0]) for part in parts)
    max_w = max(int(part["shape"][1]) for part in parts)
    order = sorted(range(len(parts)), key=lambda i: int(parts[i]["area"]), reverse=True)
    solutions: dict[tuple[int, int, bytes], np.ndarray] = {}

    for missing in range(0, 3):
        area = int(total_area + missing)
        for h, w in _factor_pairs(area):
            if h < max_h or w < max_w:
                continue

            placements: list[list[np.ndarray]] = []
            for idx in order:
                cells = parts[idx]["cells"]
                ph, pw = map(int, parts[idx]["shape"])
                opts = []
                for r in range(h - ph + 1):
                    for c in range(w - pw + 1):
                        opts.append(cells + np.array([r, c], dtype=int))
                placements.append(opts)

            board = np.zeros((h, w), dtype=bool)

            def backtrack(depth: int) -> None:
                if len(solutions) > 1:
                    return
                if depth == len(order):
                    holes = np.argwhere(~board)
                    if holes.shape[0] != missing:
                        return
                    if not board[0, :].any() or not board[-1, :].any():
                        return
                    if not board[:, 0].any() or not board[:, -1].any():
                        return
                    key = (int(h), int(w), board.tobytes())
                    if key not in solutions:
                        solutions[key] = holes.copy()
                    return

                for coords in placements[depth]:
                    rr = coords[:, 0]
                    cc = coords[:, 1]
                    if board[rr, cc].any():
                        continue
                    board[rr, cc] = True
                    backtrack(depth + 1)
                    board[rr, cc] = False

            backtrack(0)
            if len(solutions) > 1:
                break
        if len(solutions) > 1:
            break

    assert len(solutions) == 1, "same-color fragments must assemble into one unique rectangle"
    (h, w, board_bytes), holes = next(iter(solutions.items()))
    mask = np.frombuffer(board_bytes, dtype=bool).reshape((int(h), int(w))).copy()
    arr = np.full((int(h), int(w)), BG, dtype=int)
    arr[mask] = int(color)
    return arr, holes


def _stack_rectangles(items: list[dict]) -> np.ndarray:
    items = sorted(items, key=lambda item: int(item["layer"]))
    out_h = sum(int(item["arr"].shape[0]) for item in items)
    out_w = max(int(item["arr"].shape[1]) for item in items)
    out = np.full((out_h, out_w), BG, dtype=int)

    cursor = out_h
    for item in items:
        arr = item["arr"]
        h, w = map(int, arr.shape)
        cursor -= h
        c0 = (out_w - w) // 2
        patch = out[cursor : cursor + h, c0 : c0 + w]
        patch[arr != BG] = arr[arr != BG]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-color fragments into holey rectangles, sort them by the
    leftmost hole column, and stack them into a centered tower.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert colors

    items = []
    for color in colors:
        parts = _component_parts(g, int(color))
        assembled, holes = _assemble_color(parts, int(color))
        assert holes.size > 0, "each rectangle needs at least one window for ordering"
        layer = int(holes[:, 1].min() + 1)
        items.append({"layer": layer, "arr": assembled})

    return _stack_rectangles(items)


def _random_cut_path(length: int, low: int, high: int) -> list[int]:
    vals = [random.randint(int(low), int(high))]
    for _ in range(length - 1):
        nxt = vals[-1] + random.choice((-1, 0, 1))
        vals.append(max(int(low), min(int(high), int(nxt))))
    return vals


def _split_two_vertical(mask: np.ndarray) -> list[np.ndarray] | None:
    h, w = mask.shape
    if w < 2:
        return None
    cut_path = _random_cut_path(h, 1, w - 1)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    for r, cut in enumerate(cut_path):
        left[r, :cut] = mask[r, :cut]
        right[r, cut:] = mask[r, cut:]
    return [left, right]


def _split_three_right(mask: np.ndarray) -> list[np.ndarray] | None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return None
    cut_path = _random_cut_path(h, 2, w - 2)
    row_cut = random.randint(1, h - 1) if h == 4 else random.randint(2, h - 2)
    left = np.zeros_like(mask)
    top_right = np.zeros_like(mask)
    bottom_right = np.zeros_like(mask)
    for r, cut in enumerate(cut_path):
        left[r, :cut] = mask[r, :cut]
        if r < row_cut:
            top_right[r, cut:] = mask[r, cut:]
        else:
            bottom_right[r, cut:] = mask[r, cut:]
    return [left, top_right, bottom_right]


def _split_three_left(mask: np.ndarray) -> list[np.ndarray] | None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return None
    cut_path = _random_cut_path(h, 2, w - 2)
    row_cut = random.randint(1, h - 1) if h == 4 else random.randint(2, h - 2)
    right = np.zeros_like(mask)
    top_left = np.zeros_like(mask)
    bottom_left = np.zeros_like(mask)
    for r, cut in enumerate(cut_path):
        right[r, cut:] = mask[r, cut:]
        if r < row_cut:
            top_left[r, :cut] = mask[r, :cut]
        else:
            bottom_left[r, :cut] = mask[r, :cut]
    return [right, top_left, bottom_left]


def _transpose_masks(masks: list[np.ndarray] | None) -> list[np.ndarray] | None:
    if masks is None:
        return None
    return [mask.T.copy() for mask in masks]


def _split_mask(mask: np.ndarray, piece_count: int) -> list[np.ndarray] | None:
    patterns = ["v2", "h2"] if piece_count == 2 else ["right3", "left3", "down3", "up3"]
    random.shuffle(patterns)

    for pattern in patterns:
        if pattern == "v2":
            pieces = _split_two_vertical(mask)
        elif pattern == "h2":
            pieces = _transpose_masks(_split_two_vertical(mask.T))
        elif pattern == "right3":
            pieces = _split_three_right(mask)
        elif pattern == "left3":
            pieces = _split_three_left(mask)
        elif pattern == "down3":
            pieces = _transpose_masks(_split_three_right(mask.T))
        else:
            pieces = _transpose_masks(_split_three_left(mask.T))
        if pieces is None:
            continue

        counts = np.zeros(mask.shape, dtype=int)
        for piece in pieces:
            counts += piece.astype(int)
        if np.any(counts > 1) or not np.array_equal(counts > 0, mask):
            continue
        if any(piece.sum() == 0 for piece in pieces):
            continue
        if any(not _is_connected(piece) for piece in pieces):
            continue
        return [_crop_bool(piece) for piece in pieces]
    return None


def _sample_holes(h: int, w: int, layer: int) -> list[tuple[int, int]]:
    first_col = int(layer - 1)
    assert first_col < w
    row_choices = list(range(1, h - 1)) if h >= 3 else list(range(h))
    first = (random.choice(row_choices), first_col)
    holes = [first]

    later_cells = [(r, c) for r in row_choices for c in range(first_col + 1, w)]
    if later_cells and random.random() < 0.5:
        holes.append(random.choice(later_cells))
    return holes


def _paint_piece(mask: np.ndarray, color: int) -> np.ndarray:
    arr = np.full(mask.shape, BG, dtype=int)
    arr[mask] = int(color)
    return arr


def _build_rectangle(color: int, layer: int, count: int) -> tuple[dict, list[np.ndarray]] | None:
    for _ in range(80):
        h = random.randint(4, 7)
        w = random.randint(max(4, count), 9)
        mask = np.ones((h, w), dtype=bool)
        for r, c in _sample_holes(h, w, layer):
            mask[r, c] = False

        piece_count = random.choice((2, 3))
        pieces = _split_mask(mask, piece_count)
        if pieces is None:
            continue

        rect_arr = _paint_piece(mask, int(color))
        piece_arrays = [_paint_piece(piece, int(color)) for piece in pieces]
        return {"arr": rect_arr, "layer": int(layer)}, piece_arrays
    return None


def _crop_with_margin(grid: np.ndarray, margin: int = 1) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = max(0, int(r0) - margin)
    c0 = max(0, int(c0) - margin)
    r1 = min(int(grid.shape[0]) - 1, int(r1) + margin)
    c1 = min(int(grid.shape[1]) - 1, int(c1) + margin)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _place_fragments(pieces: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(piece.shape[0]) for piece in pieces)
    max_w = max(int(piece.shape[1]) for piece in pieces)
    bbox_area = sum(int(piece.shape[0] * piece.shape[1]) for piece in pieces)

    base_h = max(max_h + 4, min(26, int(np.ceil((bbox_area * 1.8) ** 0.5)) + random.randint(2, 6)))
    base_w = max(max_w + 4, min(28, int(np.ceil((bbox_area * 2.1) ** 0.5)) + random.randint(2, 8)))

    for _ in range(120):
        h = random.randint(base_h, min(30, base_h + 4))
        w = random.randint(base_w, min(30, base_w + 4))
        grid = np.full((h, w), BG, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        order = list(range(len(pieces)))
        random.shuffle(order)
        ok = True

        for idx in order:
            piece = pieces[idx]
            ph, pw = map(int, piece.shape)
            positions = []
            for r in range(h - ph + 1):
                for c in range(w - pw + 1):
                    rr0 = max(0, r - 1)
                    cc0 = max(0, c - 1)
                    rr1 = min(h, r + ph + 1)
                    cc1 = min(w, c + pw + 1)
                    if blocked[rr0:rr1, cc0:cc1].any():
                        continue
                    positions.append((r, c))
            if not positions:
                ok = False
                break

            r, c = random.choice(positions)
            patch = grid[r : r + ph, c : c + pw]
            patch[piece != BG] = piece[piece != BG]
            blocked[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)] = True

        if ok:
            return _crop_with_margin(grid, margin=1)
    return None


def _generate_attempt() -> np.ndarray | None:
    rect_count = random.randint(2, 4)
    colors = random.sample(PALETTE, rect_count)
    rects: list[dict] = []
    pieces: list[np.ndarray] = []

    for layer, color in enumerate(colors, start=1):
        built = _build_rectangle(int(color), int(layer), int(rect_count))
        if built is None:
            return None
        rect, part_arrays = built
        rects.append(rect)
        pieces.extend(part_arrays)

    grid = _place_fragments(pieces)
    if grid is None:
        return None

    expected = _stack_rectangles(rects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GEN_ATTEMPTS} attempts")
