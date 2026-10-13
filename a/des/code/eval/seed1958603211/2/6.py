from __future__ import annotations

from collections import Counter
import random

import numpy as np


SIDE_ORDER = ("top", "right", "bottom", "left")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _boundary_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[h - 1, :] = True
    mask[:, 0] = True
    mask[:, w - 1] = True
    return mask


def _tight_crop(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _find_divider_structure(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int], int, int]:
    h, w = grid.shape
    best = None
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    for color in colors:
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(row_lines) < 3 or len(col_lines) < 3:
            continue
        if row_lines[0] != 0 or row_lines[-1] != h - 1:
            continue
        if col_lines[0] != 0 or col_lines[-1] != w - 1:
            continue
        row_steps = np.diff(row_lines)
        col_steps = np.diff(col_lines)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        row_step = int(row_steps[0])
        col_step = int(col_steps[0])
        if row_step < 4 or col_step < 4:
            continue
        score = (len(row_lines) * len(col_lines), len(row_lines) + len(col_lines))
        if best is None or score > best[0]:
            best = (score, color, row_lines, col_lines, row_step - 1, col_step - 1)
    assert best is not None, "expected a regular divider lattice"
    _, divider, row_lines, col_lines, cell_h, cell_w = best
    return int(divider), list(map(int, row_lines)), list(map(int, col_lines)), int(cell_h), int(cell_w)


def _cell_bounds(row_lines: list[int], col_lines: list[int], i: int, j: int) -> tuple[int, int, int, int]:
    r0 = int(row_lines[i] + 1)
    r1 = int(row_lines[i + 1])
    c0 = int(col_lines[j] + 1)
    c1 = int(col_lines[j + 1])
    return r0, r1, c0, c1


def _slot_signature(patch: np.ndarray, bg: int) -> tuple[int | None, ...]:
    h, w = patch.shape
    sides = {
        "top": [(0, c) for c in range(w)],
        "right": [(r, w - 1) for r in range(h)],
        "bottom": [(h - 1, c) for c in range(w)],
        "left": [(r, 0) for r in range(h)],
    }
    signature: list[int | None] = []
    for side in SIDE_ORDER:
        colors = [int(patch[r, c]) for r, c in sides[side] if int(patch[r, c]) != int(bg)]
        unique = sorted(set(colors))
        assert len(colors) <= 1 and len(unique) <= 1, "each slot side may have at most one socket"
        signature.append(unique[0] if unique else None)
    return tuple(signature)


def _piece_signature(patch: np.ndarray, bg: int) -> tuple[int | None, ...]:
    fg = patch != int(bg)
    assert bool(fg.any()), "piece must be nonempty"
    colors, counts = np.unique(patch[fg], return_counts=True)
    order = np.argsort(counts)[::-1]
    fill = int(colors[int(order[0])])
    if len(order) > 1:
        assert int(counts[int(order[0])]) > int(counts[int(order[1])]), "piece needs a unique fill color"

    connector_mask = fg & (patch != int(fill))
    boundary = _boundary_mask(patch.shape)
    assert not bool((connector_mask & ~boundary).any()), "connectors must stay on the piece boundary"

    h, w = patch.shape
    sides = {
        "top": [(0, c) for c in range(w)],
        "right": [(r, w - 1) for r in range(h)],
        "bottom": [(h - 1, c) for c in range(w)],
        "left": [(r, 0) for r in range(h)],
    }
    signature: list[int | None] = []
    for side in SIDE_ORDER:
        colors = [
            int(patch[r, c])
            for r, c in sides[side]
            if int(patch[r, c]) != int(bg) and int(patch[r, c]) != int(fill)
        ]
        unique = sorted(set(colors))
        assert len(colors) <= 1 and len(unique) <= 1, "each piece side may have at most one connector"
        signature.append(unique[0] if unique else None)
    return tuple(signature)


def _match_assignments(
    pieces: list[np.ndarray],
    slots: list[dict],
    bg: int,
) -> dict[int, tuple[int, int]]:
    options: dict[int, list[tuple[int, int]]] = {}
    for pidx, patch in enumerate(pieces):
        piece_options: list[tuple[int, int]] = []
        for rot in range(4):
            rotated = np.rot90(patch, rot)
            signature = _piece_signature(rotated, bg)
            for sidx, slot in enumerate(slots):
                if signature == slot["signature"]:
                    piece_options.append((int(sidx), int(rot)))
        assert piece_options, "every piece must match at least one slot"
        options[pidx] = piece_options

    order = sorted(options, key=lambda idx: len(options[idx]))
    solutions: list[dict[int, tuple[int, int]]] = []

    def dfs(pos: int, used: set[int], current: dict[int, tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(current.copy())
            return
        pidx = order[pos]
        for sidx, rot in options[pidx]:
            if sidx in used:
                continue
            used.add(int(sidx))
            current[int(pidx)] = (int(sidx), int(rot))
            dfs(pos + 1, used, current)
            used.remove(int(sidx))
            current.pop(int(pidx))

    dfs(0, set(), {})
    assert len(solutions) == 1, "piece-slot matching should be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place loose connector-marked pieces into the uniquely matching socket-marked empty slots.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    _divider, row_lines, col_lines, cell_h, cell_w = _find_divider_structure(g, bg)

    row_count = len(row_lines) - 1
    col_count = len(col_lines) - 1
    assert row_count >= 2 and col_count >= 2

    boundary = _boundary_mask((cell_h, cell_w))
    corner_indices = [
        (0, 0),
        (0, col_count - 1),
        (row_count - 1, 0),
        (row_count - 1, col_count - 1),
    ]
    bank_candidates: list[tuple[int, int]] = []
    for i, j in corner_indices:
        r0, r1, c0, c1 = _cell_bounds(row_lines, col_lines, i, j)
        patch = g[r0:r1, c0:c1]
        fg = patch != int(bg)
        if bool((fg & ~boundary).any()) and len(_components(fg)) >= 2:
            bank_candidates.append((int(i), int(j)))
    assert len(bank_candidates) == 1, "expected one corner bank cell"
    bank_i, bank_j = bank_candidates[0]

    pieces: list[np.ndarray] = []
    bank_r0, bank_r1, bank_c0, bank_c1 = _cell_bounds(row_lines, col_lines, bank_i, bank_j)
    bank_patch = g[bank_r0:bank_r1, bank_c0:bank_c1]
    for coords in _components(bank_patch != int(bg)):
        local_mask = np.zeros(bank_patch.shape, dtype=bool)
        local_mask[coords[:, 0], coords[:, 1]] = True
        pieces.append(_tight_crop(bank_patch, local_mask))
    assert len(pieces) >= 2, "bank should contain multiple loose pieces"

    slots: list[dict] = []
    for i in range(row_count):
        for j in range(col_count):
            if (i, j) == (bank_i, bank_j):
                continue
            r0, r1, c0, c1 = _cell_bounds(row_lines, col_lines, i, j)
            patch = g[r0:r1, c0:c1]
            fg = patch != int(bg)
            interior_fg = fg & ~boundary
            if bool(interior_fg.any()):
                continue
            if not bool(fg.any()):
                continue
            signature = _slot_signature(patch, bg)
            if any(color is not None for color in signature):
                slots.append(
                    {
                        "cell": (int(i), int(j)),
                        "signature": signature,
                        "bounds": (int(r0), int(r1), int(c0), int(c1)),
                    }
                )
    assert len(slots) == len(pieces), "piece count must equal target slot count"

    assignment = _match_assignments(pieces, slots, bg)

    out = g.copy()
    out[bank_r0:bank_r1, bank_c0:bank_c1] = int(bg)
    for pidx, patch in enumerate(pieces):
        sidx, rot = assignment[pidx]
        slot = slots[sidx]
        rotated = np.rot90(patch, rot)
        h, w = rotated.shape
        r0, r1, c0, c1 = slot["bounds"]
        assert h <= cell_h - 2 and w <= cell_w - 2, "pieces should fit inside slots with margin"
        pr = int(r0 + (cell_h - h) // 2)
        pc = int(c0 + (cell_w - w) // 2)
        region = out[pr : pr + h, pc : pc + w]
        overlap = (region != int(bg)) & (rotated != int(bg))
        assert not bool(overlap.any()), "piece placement must not overwrite existing marks"
        draw_mask = rotated != int(bg)
        region[draw_mask] = rotated[draw_mask]
    return out


# --------------- generator ---------------

GEN_RETRIES = 400
GEN_CELL_RANGE = (8, 10)
GEN_GRID_ROWS = (3, 4)
GEN_GRID_COLS = (3, 4)
GEN_PIECE_COUNT = (2, 3)
GEN_OCCUPIED_MAX = 2
GEN_SOCKET_COUNT = (1, 4)
GEN_FILL_SHAPES = [(3, 3), (3, 4), (4, 3), (4, 4)]
GEN_CORNERS = ("tl", "tr", "bl", "br")


def _anchor_map(h: int, w: int) -> dict[str, tuple[int, int]]:
    return {
        "top": (0, w // 2),
        "right": (h // 2, w - 1),
        "bottom": (h - 1, w // 2),
        "left": (h // 2, 0),
    }


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _random_mask(min_area: int) -> np.ndarray:
    h, w = random.choice(GEN_FILL_SHAPES)
    mask = np.ones((h, w), dtype=bool)
    anchors = set(_anchor_map(h, w).values())
    removable = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in anchors and (r in (0, h - 1) or c in (0, w - 1))
    ]
    random.shuffle(removable)
    max_remove = min(3, int(mask.sum()) - min_area)
    target_remove = random.randint(0, max(0, max_remove))
    removed = 0
    for r, c in removable:
        if removed >= target_remove:
            break
        trial = mask.copy()
        trial[r, c] = False
        if int(trial.sum()) < min_area:
            continue
        if not _is_connected(trial):
            continue
        mask = trial
        removed += 1
    return mask


def _paint_piece(
    mask: np.ndarray,
    fill: int,
    signature: tuple[int | None, ...],
    bg: int,
) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(fill)
    anchors = _anchor_map(*mask.shape)
    for side, color in zip(SIDE_ORDER, signature, strict=False):
        if color is None:
            continue
        r, c = anchors[side]
        assert bool(mask[r, c])
        patch[r, c] = int(color)
    return patch


def _slot_with_sockets(
    cell_h: int,
    cell_w: int,
    signature: tuple[int | None, ...],
    bg: int,
) -> np.ndarray:
    patch = np.full((cell_h, cell_w), int(bg), dtype=int)
    anchors = _anchor_map(cell_h, cell_w)
    for side, color in zip(SIDE_ORDER, signature, strict=False):
        if color is None:
            continue
        r, c = anchors[side]
        patch[r, c] = int(color)
    return patch


def _bank_positions(canvas: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch.shape
    H, W = canvas.shape
    positions = [
        (r, c)
        for r in range(1, H - h)
        for c in range(1, W - w)
    ]
    random.shuffle(positions)
    return positions


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    if not np.all(canvas[r0 : r0 + h, c0 : c0 + w] == int(bg)):
        return False
    r1 = max(0, r0 - 1)
    r2 = min(canvas.shape[0], r0 + h + 1)
    c1 = max(0, c0 - 1)
    c2 = min(canvas.shape[1], c0 + w + 1)
    return np.all(canvas[r1:r2, c1:c2] == int(bg))


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    draw = patch != int(bg)
    region[draw] = patch[draw]


def _place_in_bank(bank_patch: np.ndarray, pieces: list[np.ndarray], bg: int) -> bool:
    for patch in pieces:
        placed = False
        for r0, c0 in _bank_positions(bank_patch, patch):
            if _can_place(bank_patch, patch, r0, c0, bg):
                _paste_patch(bank_patch, patch, int(r0), int(c0), bg)
                placed = True
                break
        if not placed:
            return False
    return True


def _occupied_patch(cell_h: int, cell_w: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((cell_h, cell_w), int(bg), dtype=int)
    h = random.randint(2, 3)
    w = random.randint(2, 3)
    r0 = (cell_h - h) // 2
    c0 = (cell_w - w) // 2
    patch[r0 : r0 + h, c0 : c0 + w] = int(color)
    return patch


def _corner_to_index(name: str, rows: int, cols: int) -> tuple[int, int]:
    if name == "tl":
        return 0, 0
    if name == "tr":
        return 0, cols - 1
    if name == "bl":
        return rows - 1, 0
    return rows - 1, cols - 1


def _build_expected(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    bg: int,
    divider: int,
    bank_corner: str,
    target_cells: list[tuple[int, int]],
    target_signatures: list[tuple[int | None, ...]],
    solved_pieces: list[np.ndarray],
    occupied: list[tuple[tuple[int, int], np.ndarray]],
) -> np.ndarray:
    H = rows * cell_h + rows + 1
    W = cols * cell_w + cols + 1
    out = np.full((H, W), int(bg), dtype=int)
    out[:: cell_h + 1, :] = int(divider)
    out[:, :: cell_w + 1] = int(divider)

    for (i, j), patch in occupied:
        r0 = 1 + i * (cell_h + 1)
        c0 = 1 + j * (cell_w + 1)
        _paste_patch(out[r0 : r0 + cell_h, c0 : c0 + cell_w], patch, 0, 0, bg)

    for (i, j), signature, piece in zip(target_cells, target_signatures, solved_pieces, strict=False):
        r0 = 1 + i * (cell_h + 1)
        c0 = 1 + j * (cell_w + 1)
        slot_patch = _slot_with_sockets(cell_h, cell_w, signature, bg)
        _paste_patch(out[r0 : r0 + cell_h, c0 : c0 + cell_w], slot_patch, 0, 0, bg)
        ph, pw = piece.shape
        pr = (cell_h - ph) // 2
        pc = (cell_w - pw) // 2
        _paste_patch(out[r0 + pr : r0 + pr + ph, c0 + pc : c0 + pc + pw], piece, 0, 0, bg)

    bank_i, bank_j = _corner_to_index(bank_corner, rows, cols)
    br0 = 1 + bank_i * (cell_h + 1)
    bc0 = 1 + bank_j * (cell_w + 1)
    out[br0 : br0 + cell_h, bc0 : bc0 + cell_w] = int(bg)
    return out


def _random_signature(socket_colors: list[int], used: set[tuple[int | None, ...]]) -> tuple[int | None, ...]:
    while True:
        count = random.randint(*GEN_SOCKET_COUNT)
        sides = random.sample(list(SIDE_ORDER), count)
        colors = random.sample(socket_colors, count)
        sig = []
        mapping = dict(zip(sides, colors, strict=False))
        for side in SIDE_ORDER:
            sig.append(mapping.get(side))
        signature = tuple(sig)
        if signature not in used:
            used.add(signature)
            return signature


def _gen_attempt() -> np.ndarray | None:
    palette = random.sample(list(range(10)), 10)
    bg = palette[0]
    divider = palette[1]
    socket_colors = palette[2:6]
    fill_colors = palette[6:9]
    occupied_color = palette[9]

    rows = random.randint(*GEN_GRID_ROWS)
    cols = random.randint(*GEN_GRID_COLS)
    cell_h = random.randint(*GEN_CELL_RANGE)
    cell_w = random.randint(*GEN_CELL_RANGE)
    bank_corner = random.choice(GEN_CORNERS)
    bank_cell = _corner_to_index(bank_corner, rows, cols)

    free_cells = [(i, j) for i in range(rows) for j in range(cols) if (i, j) != bank_cell]
    piece_count = random.randint(*GEN_PIECE_COUNT)
    if len(free_cells) < piece_count:
        return None
    max_occupied = min(GEN_OCCUPIED_MAX, len(free_cells) - piece_count)
    occupied_count = random.randint(0, max_occupied)
    random.shuffle(free_cells)
    target_cells = free_cells[:piece_count]
    occupied_cells = free_cells[piece_count : piece_count + occupied_count]

    signatures: list[tuple[int | None, ...]] = []
    used_signatures: set[tuple[int | None, ...]] = set()
    solved_pieces: list[np.ndarray] = []
    input_pieces: list[np.ndarray] = []

    for idx in range(piece_count):
        signature = _random_signature(socket_colors, used_signatures)
        connector_count = sum(color is not None for color in signature)
        mask = _random_mask(connector_count + 2)
        fill = fill_colors[idx]
        solved_piece = _paint_piece(mask, fill, signature, bg)
        rotation = random.randint(0, 3)
        input_piece = np.rot90(solved_piece, rotation)
        signatures.append(signature)
        solved_pieces.append(solved_piece)
        input_pieces.append(input_piece)

    occupied: list[tuple[tuple[int, int], np.ndarray]] = []
    for cell in occupied_cells:
        patch = _occupied_patch(cell_h, cell_w, occupied_color, bg)
        occupied.append((cell, patch))

    expected = _build_expected(
        rows,
        cols,
        cell_h,
        cell_w,
        bg,
        divider,
        bank_corner,
        target_cells,
        signatures,
        solved_pieces,
        occupied,
    )

    inp = expected.copy()
    for (i, j), piece in zip(target_cells, solved_pieces, strict=False):
        r0 = 1 + i * (cell_h + 1)
        c0 = 1 + j * (cell_w + 1)
        ph, pw = piece.shape
        pr = (cell_h - ph) // 2
        pc = (cell_w - pw) // 2
        clear = piece != int(bg)
        cell_view = inp[r0 + pr : r0 + pr + ph, c0 + pc : c0 + pc + pw]
        cell_view[clear] = int(bg)

    bank_patch = np.full((cell_h, cell_w), int(bg), dtype=int)
    if not _place_in_bank(bank_patch, input_pieces, bg):
        return None
    bank_i, bank_j = bank_cell
    br0 = 1 + bank_i * (cell_h + 1)
    bc0 = 1 + bank_j * (cell_w + 1)
    inp[br0 : br0 + cell_h, bc0 : bc0 + cell_w] = bank_patch

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector-slot puzzle")
