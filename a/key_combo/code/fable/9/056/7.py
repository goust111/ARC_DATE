from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
NON_BG_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400


def _component_masks(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "mask": mask,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "bottom": r1,
                }
            )

    return components


def _order_key(component: dict) -> tuple[int, int, int, int]:
    return (-int(component["r1"]), -int(component["r0"]), int(component["c0"]), int(component["c1"]))


def _stamp(board: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    locs = np.argwhere(mask)
    for dr, dc in locs:
        board[int(r0 + dr), int(c0 + dc)] = int(color)


def _drop_component(
    board: np.ndarray,
    occupied: np.ndarray,
    component: dict,
) -> tuple[int, str]:
    mask = np.asarray(component["mask"], dtype=bool)
    color = int(component["color"])
    h, w = board.shape
    r0 = int(component["r0"])
    c0 = int(component["c0"])
    locs = np.argwhere(mask)

    shift = 0
    stop_reason = "floor"
    while True:
        blocked = False
        for dr, dc in locs:
            nr = r0 + int(dr) + shift + 1
            nc = c0 + int(dc)
            if nr >= h:
                stop_reason = "floor"
                blocked = True
                break
            if occupied[nr, nc]:
                stop_reason = "piece"
                blocked = True
                break
        if blocked:
            break
        shift += 1

    final_r0 = r0 + shift
    for dr, dc in locs:
        rr = final_r0 + int(dr)
        cc = c0 + int(dc)
        board[rr, cc] = color
        occupied[rr, cc] = True
    return final_r0, stop_reason


def _settle_components(components: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, list[dict]]:
    board = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    settled: list[dict] = []
    for component in sorted(components, key=_order_key):
        final_r0, stop_reason = _drop_component(board, occupied, component)
        settled.append(
            {
                "color": int(component["color"]),
                "mask": np.asarray(component["mask"], dtype=bool).copy(),
                "r0": final_r0,
                "c0": int(component["c0"]),
                "r1": final_r0 + component["mask"].shape[0] - 1,
                "c1": int(component["c0"]) + component["mask"].shape[1] - 1,
                "stop_reason": stop_reason,
            }
        )
    return board, settled


def _settle_components_in_order(components: list[dict], shape: tuple[int, int], key) -> np.ndarray:
    board = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for component in sorted(components, key=key):
        _drop_component(board, occupied, component)
    return board


def _clear_full_rows(board: np.ndarray, bg: int = BACKGROUND) -> tuple[np.ndarray, list[int]]:
    out = np.asarray(board, dtype=int).copy()
    cleared_all: list[int] = []
    while True:
        full_rows = np.flatnonzero(np.all(out != bg, axis=1)).tolist()
        if not full_rows:
            return out, cleared_all
        cleared_all.extend(int(r) for r in full_rows)
        keep = out[~np.all(out != bg, axis=1)]
        pad = np.full((len(full_rows), out.shape[1]), bg, dtype=int)
        out = np.vstack([pad, keep])


def main(grid: np.ndarray) -> np.ndarray:
    """Drop separated plate blocks from lower to higher, then repeatedly clear every fully filled row.

    Input structure:
    - A black background contains several disconnected monochromatic polyomino plate blocks.
    - Blocks float above the floor and may overlap in columns, but they do not touch in the input.
    - The bottom border of the canvas acts as ground.

    Rule:
    - Remove all blocks, then place them back one by one in order of original lower position first.
    - Each block keeps its exact shape and color and falls straight downward until the next step would hit the floor or an already settled block.
    - After all blocks settle, erase every row that is completely filled with foreground cells across the whole canvas.
    - Collapse the remaining foreground downward by the cleared rows and repeat until no full row remains.

    Key insight:
    - The output is determined by two linked stages: rigid-body settling depends on which block started lower, and only after that does full-row erasure remove support and shrink the stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = _component_masks(g, bg=BACKGROUND)
    settled_board, _settled = _settle_components(components, g.shape)
    cleared_board, _rows = _clear_full_rows(settled_board, bg=BACKGROUND)
    return cleared_board


def _hbar(length: int) -> np.ndarray:
    return np.ones((1, int(length)), dtype=bool)


def _vbar(length: int) -> np.ndarray:
    return np.ones((int(length), 1), dtype=bool)


def _square2() -> np.ndarray:
    return np.ones((2, 2), dtype=bool)


def _make_piece(mask: np.ndarray, r0: int, c0: int, level: int) -> dict:
    return {"mask": np.asarray(mask, dtype=bool), "r0": int(r0), "c0": int(c0), "level": int(level)}


def _family_single_column(rng: random.Random) -> tuple[int, int, list[dict]]:
    left_len = rng.randint(2, 4)
    right_len = rng.randint(1, 4)
    center_h = rng.randint(2, 3)
    upper_left = rng.randint(1, left_len - 1)
    include_right = rng.random() < 0.7
    upper_right = rng.randint(1, right_len) if include_right else 0

    width = left_len + 1 + right_len
    local_h = max(center_h, 2)
    floor = local_h - 1
    pieces = [
        _make_piece(_hbar(left_len), floor, 0, 0),
        _make_piece(_hbar(right_len), floor, left_len + 1, 0),
        _make_piece(_vbar(center_h), local_h - center_h, left_len, 0),
        _make_piece(_hbar(upper_left), floor - 1, left_len - upper_left, 1),
    ]
    if include_right:
        pieces.append(_make_piece(_hbar(upper_right), floor - 1, left_len + 1, 1))
    return local_h, width, pieces


def _family_double_row(rng: random.Random) -> tuple[int, int, list[dict]]:
    left_len = rng.randint(1, 3)
    right_len = rng.randint(1, 3)
    include_left_top = rng.random() < 0.75
    include_right_top = rng.random() < 0.75
    left_top_len = rng.randint(1, max(1, left_len)) if include_left_top else 0
    right_top_len = rng.randint(1, max(1, right_len)) if include_right_top else 0
    if include_left_top and left_top_len == left_len and include_right_top and right_top_len == right_len:
        if rng.random() < 0.5:
            left_top_len = max(1, left_len - 1)
        else:
            right_top_len = max(1, right_len - 1)

    width = left_len + 2 + right_len
    floor = 3
    pieces = [
        _make_piece(_hbar(left_len), floor, 0, 0),
        _make_piece(_hbar(right_len), floor, left_len + 2, 0),
        _make_piece(_square2(), floor - 1, left_len, 0),
        _make_piece(_hbar(left_len), floor - 1, 0, 1),
        _make_piece(_hbar(right_len), floor - 1, left_len + 2, 1),
        _make_piece(_hbar(2), floor - 2, left_len, 2),
    ]
    if include_left_top:
        pieces.append(_make_piece(_hbar(left_top_len), floor - 2, left_len - left_top_len, 2))
    if include_right_top:
        pieces.append(_make_piece(_hbar(right_top_len), floor - 2, left_len + 2, 2))
    return floor + 1, width, pieces


def _family_single_square(rng: random.Random) -> tuple[int, int, list[dict]]:
    left_len = rng.randint(1, 3)
    right_len = rng.randint(1, 3)
    upper_left = rng.randint(1, left_len)
    upper_right = rng.randint(1, right_len)
    include_cap = rng.random() < 0.65
    if upper_left == left_len and upper_right == right_len:
        if rng.random() < 0.5 and left_len > 1:
            upper_left -= 1
        elif right_len > 1:
            upper_right -= 1

    width = left_len + 2 + right_len
    local_h = 4 if include_cap else 3
    floor = local_h - 1
    pieces = [
        _make_piece(_hbar(left_len), floor, 0, 0),
        _make_piece(_hbar(right_len), floor, left_len + 2, 0),
        _make_piece(_square2(), floor - 1, left_len, 0),
        _make_piece(_hbar(upper_left), floor - 1, left_len - upper_left, 1),
        _make_piece(_hbar(upper_right), floor - 1, left_len + 2, 1),
    ]
    if include_cap:
        pieces.append(_make_piece(_hbar(2), floor - 2, left_len, 2))
    return local_h, width, pieces


def _build_family(rng: random.Random) -> tuple[int, int, list[dict]]:
    family = rng.choice((_family_single_column, _family_double_row, _family_single_square))
    return family(rng)


def _coords_from_piece(piece: dict, r0: int | None = None) -> set[tuple[int, int]]:
    base_r0 = int(piece["r0"] if r0 is None else r0)
    base_c0 = int(piece["c0"])
    coords = set()
    for dr, dc in np.argwhere(piece["mask"]):
        coords.add((base_r0 + int(dr), base_c0 + int(dc)))
    return coords


def _touch_or_overlap(coords_a: set[tuple[int, int]], coords_b: set[tuple[int, int]]) -> bool:
    for r, c in coords_a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in coords_b:
                    return True
    return False


def _board_from_pieces(shape: tuple[int, int], pieces: list[dict]) -> np.ndarray:
    board = np.zeros(shape, dtype=int)
    for piece in pieces:
        _stamp(board, piece["mask"], int(piece["r0"]), int(piece["c0"]), int(piece["color"]))
    return board


def _input_from_final_layout(
    rng: random.Random,
    local_h: int,
    width: int,
    local_pieces: list[dict],
) -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = len(local_pieces)
    total_h = local_h + rng.randint(2 * piece_count + 1, 2 * piece_count + 5)
    if total_h > 30 or width > 30:
        return None
    top_margin = total_h - local_h

    final_pieces: list[dict] = []
    colors = rng.sample(list(NON_BG_COLORS), piece_count)
    for color, piece in zip(colors, local_pieces):
        final_piece = {
            "mask": piece["mask"].copy(),
            "r0": top_margin + int(piece["r0"]),
            "c0": int(piece["c0"]),
            "level": int(piece["level"]),
            "color": int(color),
        }
        final_pieces.append(final_piece)

    settled_board = _board_from_pieces((total_h, width), final_pieces)
    expected_output, cleared_rows = _clear_full_rows(settled_board, bg=BACKGROUND)
    if not cleared_rows:
        return None
    if not np.any(settled_board[: max(cleared_rows), :] != BACKGROUND):
        return None

    level_order = sorted(range(piece_count), key=lambda idx: (final_pieces[idx]["level"], rng.random()))
    row_candidates = list(range(total_h - 2, -1, -1))

    for _ in range(200):
        if len(row_candidates) < piece_count:
            return None
        assigned_bottoms = sorted(rng.sample(row_candidates, piece_count), reverse=True)
        lifted: list[dict] = [None] * piece_count  # type: ignore[assignment]
        ok = True
        occupied_sets: list[set[tuple[int, int]]] = []

        for target_bottom, idx in zip(assigned_bottoms, level_order):
            piece = final_pieces[idx]
            h = piece["mask"].shape[0]
            final_bottom = int(piece["r0"]) + h - 1
            if target_bottom >= final_bottom or target_bottom < h - 1:
                ok = False
                break
            input_r0 = target_bottom - h + 1
            candidate = {
                "mask": piece["mask"].copy(),
                "r0": input_r0,
                "c0": int(piece["c0"]),
                "color": int(piece["color"]),
                "level": int(piece["level"]),
                "r1": input_r0 + h - 1,
                "c1": int(piece["c0"]) + piece["mask"].shape[1] - 1,
            }
            coords = _coords_from_piece(candidate)
            if any(_touch_or_overlap(coords, other) for other in occupied_sets):
                ok = False
                break
            lifted[idx] = candidate
            occupied_sets.append(coords)

        if not ok:
            continue

        input_grid = _board_from_pieces((total_h, width), lifted)
        if len(_component_masks(input_grid, bg=BACKGROUND)) != piece_count:
            continue

        correct_settled, settled_meta = _settle_components(_component_masks(input_grid, bg=BACKGROUND), input_grid.shape)
        if not np.array_equal(correct_settled, settled_board):
            continue
        if not any(meta["stop_reason"] == "piece" for meta in settled_meta):
            continue

        wrong_components = _component_masks(input_grid, bg=BACKGROUND)
        wrong_board = _settle_components_in_order(
            wrong_components,
            input_grid.shape,
            key=lambda comp: (int(comp["r1"]), int(comp["r0"]), int(comp["c0"]), int(comp["c1"])),
        )
        if np.array_equal(wrong_board, settled_board):
            continue

        if not np.array_equal(main(input_grid), expected_output):
            continue
        if np.array_equal(input_grid, expected_output):
            continue
        if np.array_equal(settled_board, expected_output):
            continue
        return input_grid, expected_output

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        local_h, width, local_pieces = _build_family(rng)
        built = _input_from_final_layout(rng, local_h, width, local_pieces)
        if built is None:
            continue
        input_grid, expected_output = built
        if np.array_equal(main(input_grid), expected_output):
            return input_grid
    raise RuntimeError("failed to generate a valid gravity-clear ARC input")
