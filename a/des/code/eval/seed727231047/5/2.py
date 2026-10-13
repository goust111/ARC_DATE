from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS = ("N", "S", "W", "E")
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}
STEP = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}

MARKER_COLOR = 9
PIECE_COUNT_RANGE = (3, 5)
BASE_SIZE_OPTIONS = (3, 5)
INPUT_GAP_RANGE = (2, 4)
INPUT_MARGIN_RANGE = (2, 4)
EXTRA_PAD_RANGE = (1, 4)
GRID_MAX = 30
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "expected a unique background color"
    return int(winners[0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for sr, sc in map(tuple, np.argwhere(grid != int(bg))):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        color = int(grid[sr, sc])
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((int(r), int(c)))
            for dr, dc in STEP.values():
                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if seen[nr, nc] or int(grid[nr, nc]) != color:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        comps.append(
            {
                "color": color,
                "coords": coords,
                "area": len(coords),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return comps


def _build_piece_mask(base_h: int, base_w: int, notch_side: str, tab_side: str) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    top_extra = 1 if tab_side == "N" else 0
    bottom_extra = 1 if tab_side == "S" else 0
    left_extra = 1 if tab_side == "W" else 0
    right_extra = 1 if tab_side == "E" else 0

    bbox_h = base_h + top_extra + bottom_extra
    bbox_w = base_w + left_extra + right_extra
    base_r0 = top_extra
    base_c0 = left_extra
    mid_r = base_h // 2
    mid_c = base_w // 2

    mask = np.zeros((bbox_h, bbox_w), dtype=bool)
    mask[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True

    if notch_side == "N":
        notch = (base_r0, base_c0 + mid_c)
    elif notch_side == "S":
        notch = (base_r0 + base_h - 1, base_c0 + mid_c)
    elif notch_side == "W":
        notch = (base_r0 + mid_r, base_c0)
    else:
        notch = (base_r0 + mid_r, base_c0 + base_w - 1)
    mask[notch] = False

    if tab_side == "N":
        tab = (base_r0 - 1, base_c0 + mid_c)
    elif tab_side == "S":
        tab = (base_r0 + base_h, base_c0 + mid_c)
    elif tab_side == "W":
        tab = (base_r0 + mid_r, base_c0 - 1)
    else:
        tab = (base_r0 + mid_r, base_c0 + base_w)
    mask[tab] = True

    return mask, notch, tab


def _mask_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _make_piece(base_h: int, base_w: int, notch_side: str, tab_side: str, color: int) -> dict:
    mask, notch_offset, tab_offset = _build_piece_mask(base_h, base_w, notch_side, tab_side)
    return {
        "color": int(color),
        "mask": mask,
        "cells": _mask_cells(mask),
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "notch_side": notch_side,
        "notch_offset": (int(notch_offset[0]), int(notch_offset[1])),
        "tab_side": tab_side,
        "tab_offset": (int(tab_offset[0]), int(tab_offset[1])),
    }


def _parse_piece(comp: dict) -> dict:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r - r0), int(c - c0)] = True

    counts = {
        "N": int(mask[0, :].sum()),
        "S": int(mask[-1, :].sum()),
        "W": int(mask[:, 0].sum()),
        "E": int(mask[:, -1].sum()),
    }
    tab_candidates = [side for side, count in counts.items() if count == 1]
    assert len(tab_candidates) == 1, "expected exactly one protrusion side"
    tab_side = tab_candidates[0]

    if tab_side == "N":
        tab_col = int(np.flatnonzero(mask[0, :])[0])
        tab_offset = (0, tab_col)
        base_mask = mask[1:, :]
        base_r0 = 1
        base_c0 = 0
    elif tab_side == "S":
        tab_col = int(np.flatnonzero(mask[-1, :])[0])
        tab_offset = (h - 1, tab_col)
        base_mask = mask[:-1, :]
        base_r0 = 0
        base_c0 = 0
    elif tab_side == "W":
        tab_row = int(np.flatnonzero(mask[:, 0])[0])
        tab_offset = (tab_row, 0)
        base_mask = mask[:, 1:]
        base_r0 = 0
        base_c0 = 1
    else:
        tab_row = int(np.flatnonzero(mask[:, -1])[0])
        tab_offset = (tab_row, w - 1)
        base_mask = mask[:, :-1]
        base_r0 = 0
        base_c0 = 0

    boundary = {
        "N": base_mask[0, :],
        "S": base_mask[-1, :],
        "W": base_mask[:, 0],
        "E": base_mask[:, -1],
    }
    notch_candidates = [side for side, line in boundary.items() if int((~line).sum()) == 1]
    assert len(notch_candidates) == 1, "expected exactly one notch side"
    notch_side = notch_candidates[0]
    for side, line in boundary.items():
        missing = int((~line).sum())
        if side == notch_side:
            assert missing == 1
        else:
            assert missing == 0

    if notch_side == "N":
        local = int(np.flatnonzero(~base_mask[0, :])[0])
        notch_offset = (base_r0, base_c0 + local)
    elif notch_side == "S":
        local = int(np.flatnonzero(~base_mask[-1, :])[0])
        notch_offset = (base_r0 + base_mask.shape[0] - 1, base_c0 + local)
    elif notch_side == "W":
        local = int(np.flatnonzero(~base_mask[:, 0])[0])
        notch_offset = (base_r0 + local, base_c0)
    else:
        local = int(np.flatnonzero(~base_mask[:, -1])[0])
        notch_offset = (base_r0 + local, base_c0 + base_mask.shape[1] - 1)

    cells = _mask_cells(mask)
    return {
        "color": int(comp["color"]),
        "mask": mask,
        "cells": cells,
        "h": h,
        "w": w,
        "bbox": (r0, c0, r1, c1),
        "tab_side": tab_side,
        "tab_offset": (int(tab_offset[0]), int(tab_offset[1])),
        "tab_abs": (r0 + int(tab_offset[0]), c0 + int(tab_offset[1])),
        "notch_side": notch_side,
        "notch_offset": (int(notch_offset[0]), int(notch_offset[1])),
        "notch_abs": (r0 + int(notch_offset[0]), c0 + int(notch_offset[1])),
        "coords": [(int(r), int(c)) for r, c in comp["coords"]],
    }


def _start_distance(piece: dict, marker: tuple[int, int]) -> int | None:
    mr, mc = marker
    dist: int | None = None
    for r, c in piece["coords"]:
        if int(r) == int(mr):
            cand = abs(int(c) - int(mc))
            dist = cand if dist is None else min(dist, cand)
        if int(c) == int(mc):
            cand = abs(int(r) - int(mr))
            dist = cand if dist is None else min(dist, cand)
    return dist


def _find_start_index(pieces: list[dict], marker: tuple[int, int]) -> int:
    candidates = []
    for idx, piece in enumerate(pieces):
        dist = _start_distance(piece, marker)
        if dist is not None and dist > 0:
            candidates.append((int(dist), int(idx)))
    assert candidates, "expected at least one piece aligned with the marker"
    best = min(dist for dist, _idx in candidates)
    winners = [idx for dist, idx in candidates if dist == best]
    assert len(winners) == 1, "expected a unique start piece"
    return int(winners[0])


def _is_match(current: dict, candidate: dict) -> bool:
    if candidate["notch_side"] != OPPOSITE[current["tab_side"]]:
        return False
    if current["tab_side"] in ("E", "W"):
        return int(candidate["notch_abs"][0]) == int(current["tab_abs"][0])
    return int(candidate["notch_abs"][1]) == int(current["tab_abs"][1])


def _render_placed_pieces(pieces: list[dict], placements: dict[int, tuple[int, int]], bg: int) -> np.ndarray:
    min_r = min(int(pos[0]) for pos in placements.values())
    min_c = min(int(pos[1]) for pos in placements.values())
    max_r = max(int(placements[idx][0] + pieces[idx]["h"] - 1) for idx in placements)
    max_c = max(int(placements[idx][1] + pieces[idx]["w"] - 1) for idx in placements)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    occupied = np.zeros_like(out, dtype=bool)

    for idx, piece in enumerate(pieces):
        pr = int(placements[idx][0] - min_r)
        pc = int(placements[idx][1] - min_c)
        for r, c in piece["cells"]:
            rr = int(pr + r)
            cc = int(pc + c)
            assert not occupied[rr, cc], "assembled chain must not overlap"
            occupied[rr, cc] = True
            out[rr, cc] = int(piece["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain the puzzle pieces by docking each protrusion into the uniquely aligned opposite notch.

    Input structure:
    - The background is the most common color.
    - Each foreground piece is a monochrome near-rectangle with one 1x1 notch and one 1x1 protrusion.
    - A single marker pixel in the upper-left corner indicates which piece starts the chain.

    Rule:
    - Parse every piece's protrusion side and notch side.
    - Start from the piece nearest to the marker along the marker's row or column.
    - Repeatedly find the only remaining piece whose notch faces the opposite way and whose notch row/column
      is aligned with the current protrusion in the input.
    - Translate pieces so each protrusion lands exactly in the next notch, then crop the finished chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected marker and pieces"

    marker_candidates = [
        comp for comp in comps if int(comp["area"]) == 1 and tuple(map(int, comp["coords"][0])) == (0, 0)
    ]
    assert len(marker_candidates) == 1, "expected a unique marker pixel at (0, 0)"
    marker = (0, 0)

    pieces = [_parse_piece(comp) for comp in comps if comp not in marker_candidates]
    assert pieces, "expected at least one piece"

    start = _find_start_index(pieces, marker)
    placements: dict[int, tuple[int, int]] = {int(start): (0, 0)}
    remaining = set(range(len(pieces))) - {int(start)}
    current = int(start)

    while remaining:
        matches = [idx for idx in remaining if _is_match(pieces[current], pieces[idx])]
        assert len(matches) == 1, "expected a unique next piece"
        nxt = int(matches[0])
        cr, cc = placements[current]
        nr = int(cr + pieces[current]["tab_offset"][0] - pieces[nxt]["notch_offset"][0])
        nc = int(cc + pieces[current]["tab_offset"][1] - pieces[nxt]["notch_offset"][1])
        placements[nxt] = (nr, nc)
        remaining.remove(nxt)
        current = nxt

    return _render_placed_pieces(pieces, placements, bg)


def _chain_order_ok(pieces: list[dict]) -> bool:
    if _find_start_index(pieces, (0, 0)) != 0:
        return False
    remaining = set(range(1, len(pieces)))
    current = 0
    expected = 1
    while remaining:
        matches = [idx for idx in remaining if _is_match(pieces[current], pieces[idx])]
        if matches != [expected]:
            return False
        remaining.remove(expected)
        current = expected
        expected += 1
    return True


def _piece_with_abs(piece: dict, pos: tuple[int, int]) -> dict:
    r0, c0 = map(int, pos)
    return {
        **piece,
        "bbox": (r0, c0, r0 + int(piece["h"]) - 1, c0 + int(piece["w"]) - 1),
        "coords": [(r0 + int(r), c0 + int(c)) for r, c in piece["cells"]],
        "tab_abs": (r0 + int(piece["tab_offset"][0]), c0 + int(piece["tab_offset"][1])),
        "notch_abs": (r0 + int(piece["notch_offset"][0]), c0 + int(piece["notch_offset"][1])),
    }


def _output_positions(pieces: list[dict]) -> list[tuple[int, int]] | None:
    positions = [(0, 0)]
    occupied_by_piece: list[set[tuple[int, int]]] = []
    all_occupied: set[tuple[int, int]] = set()

    for idx, piece in enumerate(pieces):
        if idx > 0:
            prev = pieces[idx - 1]
            prev_pos = positions[idx - 1]
            pos = (
                int(prev_pos[0] + prev["tab_offset"][0] - piece["notch_offset"][0]),
                int(prev_pos[1] + prev["tab_offset"][1] - piece["notch_offset"][1]),
            )
            positions.append(pos)

        pos = positions[idx]
        coords = {(int(pos[0] + r), int(pos[1] + c)) for r, c in piece["cells"]}
        if coords & all_occupied:
            return None
        if idx > 1:
            prior = set().union(*occupied_by_piece[:-1])
            for r, c in coords:
                for dr, dc in STEP.values():
                    if (int(r + dr), int(c + dc)) in prior:
                        return None
        occupied_by_piece.append(coords)
        all_occupied |= coords

    min_r = min(r for coords in occupied_by_piece for r, _ in coords)
    min_c = min(c for coords in occupied_by_piece for _r, c in coords)
    max_r = max(r for coords in occupied_by_piece for r, _ in coords)
    max_c = max(c for coords in occupied_by_piece for _r, c in coords)
    if max_r - min_r + 1 > GRID_MAX or max_c - min_c + 1 > GRID_MAX:
        return None
    return positions


def _input_positions(pieces: list[dict], axis: str) -> list[tuple[int, int]] | None:
    gap = lambda: random.randint(*INPUT_GAP_RANGE)
    margin = random.randint(*INPUT_MARGIN_RANGE)

    if axis == "row":
        positions = [(0, margin)]
        max_row = int(pieces[0]["h"])
        max_col = int(margin + pieces[0]["w"])
        for idx in range(1, len(pieces)):
            prev = pieces[idx - 1]
            prev_pos = positions[idx - 1]
            if prev["tab_side"] in ("E", "W"):
                row = int(prev_pos[0] + prev["tab_offset"][0] - pieces[idx]["notch_offset"][0])
                if row < 0:
                    return None
                col = int(max_col + gap())
            else:
                col = int(prev_pos[1] + prev["tab_offset"][1] - pieces[idx]["notch_offset"][1])
                row = int(max_row + gap())
            positions.append((row, col))
            max_row = max(max_row, int(row + pieces[idx]["h"]))
            max_col = max(max_col, int(col + pieces[idx]["w"]))

        min_col = min(c for _r, c in positions)
        if min_col < 0:
            positions = [(r, c - min_col) for r, c in positions]
    else:
        positions = [(margin, 0)]
        max_row = int(margin + pieces[0]["h"])
        max_col = int(pieces[0]["w"])
        for idx in range(1, len(pieces)):
            prev = pieces[idx - 1]
            prev_pos = positions[idx - 1]
            if prev["tab_side"] in ("E", "W"):
                row = int(prev_pos[0] + prev["tab_offset"][0] - pieces[idx]["notch_offset"][0])
                col = int(max_col + gap())
            else:
                col = int(prev_pos[1] + prev["tab_offset"][1] - pieces[idx]["notch_offset"][1])
                if col < 0:
                    return None
                row = int(max_row + gap())
            positions.append((row, col))
            max_row = max(max_row, int(row + pieces[idx]["h"]))
            max_col = max(max_col, int(col + pieces[idx]["w"]))

        min_row = min(r for r, _c in positions)
        if min_row < 0:
            positions = [(r - min_row, c) for r, c in positions]

    return positions


def _render_input(pieces: list[dict], positions: list[tuple[int, int]], bg: int) -> np.ndarray | None:
    max_r = max(int(pos[0] + pieces[idx]["h"]) for idx, pos in enumerate(positions))
    max_c = max(int(pos[1] + pieces[idx]["w"]) for idx, pos in enumerate(positions))
    height = int(max_r + random.randint(*EXTRA_PAD_RANGE))
    width = int(max_c + random.randint(*EXTRA_PAD_RANGE))
    if height > GRID_MAX or width > GRID_MAX:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    grid[0, 0] = MARKER_COLOR
    occupied[0, 0] = True

    for idx, piece in enumerate(pieces):
        pr, pc = map(int, positions[idx])
        for r, c in piece["cells"]:
            rr = int(pr + r)
            cc = int(pc + c)
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            if occupied[rr, cc]:
                return None
            occupied[rr, cc] = True
            grid[rr, cc] = int(piece["color"])

    bg_count = int(np.count_nonzero(grid == int(bg)))
    non_bg_colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    max_other = max(int(np.count_nonzero(grid == color)) for color in non_bg_colors)
    if bg_count <= max_other or _get_bg(grid) != int(bg):
        return None
    return grid


def _choose_chain_pieces() -> list[dict] | None:
    count = random.randint(*PIECE_COUNT_RANGE)
    bg = random.choice([color for color in range(10) if color != MARKER_COLOR])
    palette = [color for color in range(10) if color not in {bg, MARKER_COLOR}]
    if count > len(palette):
        return None
    colors = random.sample(palette, count)

    tab_sides = []
    notch_sides = []
    base_dims = []

    first_tab = random.choice(DIRS)
    first_notch = random.choice([side for side in DIRS if side != first_tab])
    tab_sides.append(first_tab)
    notch_sides.append(first_notch)
    base_dims.append((random.choice(BASE_SIZE_OPTIONS), random.choice(BASE_SIZE_OPTIONS)))

    for idx in range(1, count):
        notch = OPPOSITE[tab_sides[idx - 1]]
        tab = random.choice([side for side in DIRS if side != notch])
        notch_sides.append(notch)
        tab_sides.append(tab)
        base_dims.append((random.choice(BASE_SIZE_OPTIONS), random.choice(BASE_SIZE_OPTIONS)))

    pieces = [
        _make_piece(base_h, base_w, notch_sides[idx], tab_sides[idx], colors[idx])
        for idx, (base_h, base_w) in enumerate(base_dims)
    ]
    positions = _output_positions(pieces)
    if positions is None:
        return None

    pieces[0]["_bg"] = int(bg)
    pieces[0]["_output_positions"] = positions
    return pieces


def _expected_output(pieces: list[dict], bg: int) -> np.ndarray:
    placements = {idx: pos for idx, pos in enumerate(pieces[0]["_output_positions"])}
    return _render_placed_pieces(pieces, placements, bg)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pieces = _choose_chain_pieces()
        if pieces is None:
            continue
        bg = int(pieces[0]["_bg"])
        axis = random.choice(("row", "col"))
        positions = _input_positions(pieces, axis)
        if positions is None:
            continue

        placed = [_piece_with_abs(piece, pos) for piece, pos in zip(pieces, positions)]
        if not _chain_order_ok(placed):
            continue

        grid = _render_input(pieces, positions, bg)
        if grid is None:
            continue

        expected = _expected_output(pieces, bg)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError(f"failed to generate valid puzzle after {GEN_ATTEMPTS} attempts")
