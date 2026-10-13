from __future__ import annotations

from collections import Counter
import random

import numpy as np


SEPARATOR = 9
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = list(range(9))
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(counts.argmax())
    return int(values[best])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _spans_from_lines(lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:], strict=False):
        assert b - a >= 2, "separator lines must leave nonempty slot interiors"
        spans.append((int(a + 1), int(b - 1)))
    assert spans, "expected at least one span"
    return spans


def _rotate_pos(pos: tuple[int, int], h: int, w: int, k: int) -> tuple[int, int]:
    r, c = map(int, pos)
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return w - 1 - c, r
    if k == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _match_piece_to_slot(
    piece_mask: np.ndarray,
    marker_pos: tuple[int, int],
    slot_shape: tuple[int, int],
) -> list[tuple[int, tuple[int, int]]]:
    matches: list[tuple[int, tuple[int, int]]] = []
    target = np.ones(slot_shape, dtype=bool)
    h, w = piece_mask.shape
    for k in range(4):
        rot = np.rot90(piece_mask, k)
        if rot.shape != target.shape or not np.array_equal(rot, target):
            continue
        matches.append((int(k), _rotate_pos(marker_pos, h, w, k)))
    return matches


def _template_structure(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    sep_mask = grid == SEPARATOR
    assert sep_mask.any(), "expected a separator-colored template"
    r0, c0, r1, c1 = _bbox_from_mask(sep_mask)

    row_lines = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == SEPARATOR)]
    col_lines = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == SEPARATOR)]
    assert row_lines[0] == r0 and row_lines[-1] == r1, "template needs a separator border"
    assert col_lines[0] == c0 and col_lines[-1] == c1, "template needs a separator border"
    return (r0, c0, r1, c1), _spans_from_lines(row_lines), _spans_from_lines(col_lines)


def _extract_piece(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    mask = np.zeros((h, w), dtype=bool)
    counts = Counter(int(grid[r, c]) for r, c in comp)
    ranked = counts.most_common()
    assert len(ranked) >= 2 and ranked[0][1] > ranked[1][1], "piece must have a unique main color"
    main_color = int(ranked[0][0])

    marker_cells: list[tuple[int, int, int]] = []
    for r, c in comp:
        rr = r - r0
        cc = c - c0
        mask[rr, cc] = True
        color = int(grid[r, c])
        if color != main_color:
            marker_cells.append((int(rr), int(cc), color))

    assert len(marker_cells) == 1, "piece must contain exactly one marker pixel"
    marker_r, marker_c, marker_color = marker_cells[0]
    return {
        "cells": comp,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "main_color": main_color,
        "marker_color": int(marker_color),
        "marker_pos": (int(marker_r), int(marker_c)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move scattered colored pieces into matching empty template slots, then propagate each
    piece's marker color along a row or column determined by the marker's midline position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _dominant_color(g)
    assert bg != SEPARATOR, "background must differ from the separator color"

    template_bbox, row_spans, col_spans = _template_structure(g)
    tr0, tc0, tr1, tc1 = template_bbox

    empty_slots: list[dict] = []
    for rs in row_spans:
        for cs in col_spans:
            sr0, sr1 = rs
            sc0, sc1 = cs
            sub = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
            if np.all(sub == bg):
                empty_slots.append(
                    {
                        "bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
                        "shape": sub.shape,
                    }
                )

    outside = np.ones(g.shape, dtype=bool)
    outside[tr0 : tr1 + 1, tc0 : tc1 + 1] = False
    piece_mask = outside & (g != bg) & (g != SEPARATOR)
    comps = _components(piece_mask)
    assert comps, "expected at least one scattered piece"
    pieces = [_extract_piece(g, comp) for comp in comps]

    assignments: list[dict] = []
    used_slots: set[int] = set()
    for piece in pieces:
        dedup: dict[tuple[int, tuple[int, int]], tuple[int, int, tuple[int, int]]] = {}
        for slot_idx, slot in enumerate(empty_slots):
            for rot_k, marker_pos in _match_piece_to_slot(piece["mask"], piece["marker_pos"], slot["shape"]):
                key = (int(slot_idx), marker_pos)
                if key not in dedup or rot_k < dedup[key][1]:
                    dedup[key] = (int(slot_idx), int(rot_k), marker_pos)
        candidates = list(dedup.values())
        assert len(candidates) == 1, "each piece must match exactly one empty slot"
        slot_idx, rot_k, marker_pos = candidates[0]
        assert slot_idx not in used_slots, "two pieces matched the same slot"
        used_slots.add(slot_idx)
        assignments.append(
            {
                "piece": piece,
                "slot": empty_slots[slot_idx],
                "rot_k": int(rot_k),
                "marker_pos": marker_pos,
            }
        )

    out = g.copy()
    for piece in pieces:
        for r, c in piece["cells"]:
            out[r, c] = bg

    propagations: list[tuple[str, int, int]] = []
    for item in assignments:
        piece = item["piece"]
        slot = item["slot"]
        sr0, sc0, sr1, sc1 = slot["bbox"]
        slot_h, slot_w = slot["shape"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = int(piece["main_color"])

        marker_r, marker_c = item["marker_pos"]
        abs_r = sr0 + int(marker_r)
        abs_c = sc0 + int(marker_c)
        out[abs_r, abs_c] = int(piece["marker_color"])

        row_mid = slot_h % 2 == 1 and marker_r == slot_h // 2
        col_mid = slot_w % 2 == 1 and marker_c == slot_w // 2
        if row_mid and col_mid:
            row_bg = sum(out[abs_r, c] == bg for c in range(tc0, tc1 + 1))
            col_bg = sum(out[r, abs_c] == bg for r in range(tr0, tr1 + 1))
            assert (row_bg > 0) ^ (col_bg > 0), "center marker must have a uniquely useful axis"
            row_mid = row_bg > 0
            col_mid = col_bg > 0
        assert row_mid ^ col_mid, "marker must determine exactly one propagation axis"
        if row_mid:
            propagations.append(("row", int(abs_r), int(piece["marker_color"])))
        else:
            propagations.append(("col", int(abs_c), int(piece["marker_color"])))

    for axis, pos, color in propagations:
        if axis == "row":
            for c in range(tc0, tc1 + 1):
                if out[pos, c] == bg:
                    out[pos, c] = color
        else:
            for r in range(tr0, tr1 + 1):
                if out[r, pos] == bg:
                    out[r, pos] = color

    return out


def _choose_palette(piece_count: int) -> tuple[int, list[int], list[int], list[int]]:
    bg = random.choice(COLOR_POOL)
    remaining = [c for c in COLOR_POOL if c != bg]
    chosen = random.sample(remaining, 2 * piece_count + 2)
    main_colors = chosen[:piece_count]
    marker_colors = chosen[piece_count : 2 * piece_count]
    fill_colors = chosen[2 * piece_count :]
    return int(bg), [int(x) for x in main_colors], [int(x) for x in marker_colors], [int(x) for x in fill_colors]


def _partition_with_min(total: int, parts: int, minimum: int) -> list[int]:
    sizes = [minimum] * parts
    for _ in range(total - minimum * parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _layout_canvas(row_sizes: list[int], col_sizes: list[int], bg: int) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]]]:
    h = sum(row_sizes) + len(row_sizes) + 1
    w = sum(col_sizes) + len(col_sizes) + 1
    canvas = np.full((h, w), SEPARATOR, dtype=int)
    bounds: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    r = 1
    for i, rh in enumerate(row_sizes):
        c = 1
        for j, cw in enumerate(col_sizes):
            bounds[(i, j)] = (int(r), int(c), int(r + rh - 1), int(c + cw - 1))
            canvas[r : r + rh, c : c + cw] = bg
            c += cw + 1
        r += rh + 1
    return canvas, bounds


def _paint_slot(canvas: np.ndarray, bounds: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bounds
    canvas[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _row_mode_case() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = random.choice([2, 3])
    total_rows = piece_count + random.choice([1, 2])
    total_cols = 6 if piece_count == 3 else random.randint(4, 6)
    if total_cols < 2 * piece_count:
        return None

    bg, main_colors, marker_colors, fill_colors = _choose_palette(piece_count)
    piece_rows = sorted(random.sample(range(total_rows), piece_count))
    row_sizes = [1 if r in piece_rows else random.randint(2, 4) for r in range(total_rows)]
    col_sizes = random.sample([3, 5, 7, 9, 11, 13], total_cols)

    selected_count = random.randint(2 * piece_count, total_cols)
    selected_cols = random.sample(range(total_cols), selected_count)
    random.shuffle(selected_cols)
    shares = _partition_with_min(selected_count, piece_count, 2)

    role_map = {(r, c): "filled" for r in range(total_rows) for c in range(total_cols)}
    piece_specs: list[dict] = []
    offset = 0
    for idx, row in enumerate(piece_rows):
        owned = selected_cols[offset : offset + shares[idx]]
        offset += shares[idx]
        random.shuffle(owned)
        piece_col = owned[0]
        role_map[(row, piece_col)] = "piece"
        for col in owned[1:]:
            role_map[(row, col)] = "blank"
        piece_specs.append(
            {
                "slot_rc": (int(row), int(piece_col)),
                "main_color": int(main_colors[idx]),
                "marker_color": int(marker_colors[idx]),
            }
        )

    template_in, bounds = _layout_canvas(row_sizes, col_sizes, bg)
    for r in range(total_rows):
        for c in range(total_cols):
            if role_map[(r, c)] != "filled":
                continue
            color = fill_colors[(r + c) % len(fill_colors)]
            _paint_slot(template_in, bounds[(r, c)], color)

    template_out = template_in.copy()
    piece_arrays: list[np.ndarray] = []
    for spec in piece_specs:
        row, col = spec["slot_rc"]
        r0, c0, r1, c1 = bounds[(row, col)]
        width = c1 - c0 + 1
        marker_c = width // 2

        piece = np.full((1, width), int(spec["main_color"]), dtype=int)
        piece[0, marker_c] = int(spec["marker_color"])
        piece_arrays.append(np.rot90(piece, random.randrange(4)).copy())

        template_out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["main_color"])
        template_out[r0, c0 + marker_c] = int(spec["marker_color"])
        for cc in range(template_out.shape[1]):
            if template_out[r0, cc] == bg:
                template_out[r0, cc] = int(spec["marker_color"])

    return _compose_global_case(template_in, template_out, piece_arrays, bg)


def _column_mode_case() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = random.choice([2, 3])
    total_cols = piece_count + random.choice([1, 2])
    total_rows = 6 if piece_count == 3 else random.randint(4, 6)
    if total_rows < 2 * piece_count:
        return None

    bg, main_colors, marker_colors, fill_colors = _choose_palette(piece_count)
    piece_cols = sorted(random.sample(range(total_cols), piece_count))
    col_sizes = [1 if c in piece_cols else random.randint(2, 4) for c in range(total_cols)]
    row_sizes = random.sample([3, 5, 7, 9, 11, 13], total_rows)

    selected_count = random.randint(2 * piece_count, total_rows)
    selected_rows = random.sample(range(total_rows), selected_count)
    random.shuffle(selected_rows)
    shares = _partition_with_min(selected_count, piece_count, 2)

    role_map = {(r, c): "filled" for r in range(total_rows) for c in range(total_cols)}
    piece_specs: list[dict] = []
    offset = 0
    for idx, col in enumerate(piece_cols):
        owned = selected_rows[offset : offset + shares[idx]]
        offset += shares[idx]
        random.shuffle(owned)
        piece_row = owned[0]
        role_map[(piece_row, col)] = "piece"
        for row in owned[1:]:
            role_map[(row, col)] = "blank"
        piece_specs.append(
            {
                "slot_rc": (int(piece_row), int(col)),
                "main_color": int(main_colors[idx]),
                "marker_color": int(marker_colors[idx]),
            }
        )

    template_in, bounds = _layout_canvas(row_sizes, col_sizes, bg)
    for r in range(total_rows):
        for c in range(total_cols):
            if role_map[(r, c)] != "filled":
                continue
            color = fill_colors[(2 * r + c) % len(fill_colors)]
            _paint_slot(template_in, bounds[(r, c)], color)

    template_out = template_in.copy()
    piece_arrays: list[np.ndarray] = []
    for spec in piece_specs:
        row, col = spec["slot_rc"]
        r0, c0, r1, c1 = bounds[(row, col)]
        height = r1 - r0 + 1
        marker_r = height // 2

        piece = np.full((height, 1), int(spec["main_color"]), dtype=int)
        piece[marker_r, 0] = int(spec["marker_color"])
        piece_arrays.append(np.rot90(piece, random.randrange(4)).copy())

        template_out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["main_color"])
        template_out[r0 + marker_r, c0] = int(spec["marker_color"])
        for rr in range(template_out.shape[0]):
            if template_out[rr, c0] == bg:
                template_out[rr, c0] = int(spec["marker_color"])

    return _compose_global_case(template_in, template_out, piece_arrays, bg)


def _can_place_piece(grid: np.ndarray, piece: np.ndarray, r0: int, c0: int, bg: int, min_row: int) -> bool:
    h, w = piece.shape
    gh, gw = grid.shape
    if r0 < min_row or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(gh, r0 + h + 1)
    cc1 = min(gw, c0 + w + 1)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == bg))


def _compose_global_case(template_in: np.ndarray, template_out: np.ndarray, piece_arrays: list[np.ndarray], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    th, tw = template_in.shape
    max_piece_w = max(piece.shape[1] for piece in piece_arrays)
    piece_stack_h = sum(piece.shape[0] for piece in piece_arrays) + len(piece_arrays) + 2

    top_margin = random.randint(0, 2)
    left_margin = random.randint(1, 3)
    gap = random.randint(2, 4)
    lower_h = max(piece_stack_h, th + random.randint(0, 4))
    height = top_margin + th + gap + lower_h + random.randint(1, 3)
    width = max(left_margin + tw + random.randint(1, 4), max_piece_w + 6)

    while top_margin + th - 1 >= height // 2:
        lower_h += 2
        height = top_margin + th + gap + lower_h + random.randint(1, 3)

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    inp[top_margin : top_margin + th, left_margin : left_margin + tw] = template_in
    out[top_margin : top_margin + th, left_margin : left_margin + tw] = template_out

    min_row = top_margin + th + gap
    for piece in piece_arrays:
        placed = False
        for _ in range(200):
            r0 = random.randint(min_row, height - piece.shape[0])
            c0 = random.randint(0, width - piece.shape[1])
            if not _can_place_piece(inp, piece, r0, c0, bg, min_row):
                continue
            inp[r0 : r0 + piece.shape[0], c0 : c0 + piece.shape[1]] = piece
            placed = True
            break
        if not placed:
            return None

    return inp, out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    if random.random() < 0.5:
        return _row_mode_case()
    return _column_mode_case()


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        if _dominant_color(inp) == SEPARATOR:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 2")
