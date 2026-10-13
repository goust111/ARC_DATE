import numpy as np
from collections import deque


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

MOVE_DELTAS = {
    RED: (0, 1),
    BLUE: (0, -1),
    GREEN: (-1, 0),
    YELLOW: (1, 0),
}
ROTATION_STEPS = {
    ORANGE: -1,   # clockwise
    MAGENTA: 1,   # counterclockwise
}
WORKPIECE_COLORS = (BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE)
DOMINO_SECOND_COLORS = (BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE)

_RNG = np.random.default_rng()
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_MAX_GEN_ATTEMPTS = 400


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0)


def _decode_instruction(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[str, int]:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1 and c1 == c0 + 1:
        left_color = int(grid[r0, c0])
        assert left_color in MOVE_DELTAS, "unknown translation color"
        return ("move", left_color)
    if c0 == c1 and r1 == r0 + 1:
        top_color = int(grid[r0, c0])
        assert top_color in ROTATION_STEPS, "unknown rotation color"
        return ("rot", top_color)
    raise AssertionError("expected every instruction component to be a domino")


def _sort_key(coords: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, _, _ = _bbox(coords)
    return (r0, c0)


def _rotate_patch(
    patch: np.ndarray,
    top_left: tuple[int, int],
    rotation_color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    k = ROTATION_STEPS[rotation_color]
    new_patch = np.rot90(patch, k=k)
    h, w = patch.shape
    new_h, new_w = new_patch.shape
    assert (h - new_h) % 2 == 0 and (w - new_w) % 2 == 0, "rotation center falls between cells"
    new_r0 = top_left[0] + (h - new_h) // 2
    new_c0 = top_left[1] + (w - new_w) // 2
    return new_patch, (new_r0, new_c0)


def _apply_instructions(
    patch: np.ndarray,
    start_top_left: tuple[int, int],
    instructions: list[tuple[str, int]],
) -> tuple[np.ndarray, tuple[int, int]]:
    cur_patch = patch.copy()
    top_left = tuple(start_top_left)
    for kind, code in instructions:
        if kind == "move":
            dr, dc = MOVE_DELTAS[code]
            top_left = (top_left[0] + dr, top_left[1] + dc)
        else:
            cur_patch, top_left = _rotate_patch(cur_patch, top_left, code)
    return cur_patch, top_left


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= grid.shape[0] and c0 + w <= grid.shape[1], "patch out of bounds"
    mask = patch != BLACK
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Execute a row-major domino instruction chain on the top-left workpiece and keep only the final workpiece."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g != BLACK)
    workpieces: list[list[tuple[int, int]]] = []
    dominoes: list[list[tuple[int, int]]] = []

    for coords in components:
        size = len(coords)
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if size == 2 and ((h == 1 and w == 2) or (h == 2 and w == 1)):
            dominoes.append(coords)
        elif size >= 3:
            workpieces.append(coords)

    assert len(workpieces) == 1, "expected exactly one workpiece"
    workpiece = workpieces[0]
    patch, top_left = _extract_patch(g, workpiece)
    instructions = [_decode_instruction(g, coords) for coords in sorted(dominoes, key=_sort_key)]
    final_patch, final_top_left = _apply_instructions(patch, top_left, instructions)

    out = np.full_like(g, BLACK)
    _stamp_patch(out, final_patch, final_top_left)
    return out


def _random_connected_mask(side: int) -> np.ndarray | None:
    for _ in range(200):
        target = int(_RNG.integers(side + 2, side * side + 1))
        cells = {(int(_RNG.integers(side)), int(_RNG.integers(side)))}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = list(frontier)
            cells.add(frontier_list[int(_RNG.integers(len(frontier_list)))])
        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        if rows[0] != 0 or rows[-1] != side - 1 or cols[0] != 0 or cols[-1] != side - 1:
            continue
        return mask
    return None


def _random_workpiece(side: int) -> np.ndarray | None:
    for _ in range(200):
        mask = _random_connected_mask(side)
        if mask is None:
            continue
        patch = np.full((side, side), BLACK, dtype=int)
        color_count = min(int(_RNG.integers(3, 6)), int(mask.sum()))
        chosen_colors = _RNG.choice(np.array(WORKPIECE_COLORS, dtype=int), size=color_count, replace=False)
        filled = np.argwhere(mask)
        assignments = chosen_colors[_RNG.integers(0, len(chosen_colors), size=len(filled))]
        for idx, (r, c) in enumerate(filled):
            patch[int(r), int(c)] = int(assignments[idx])
        if len(set(int(v) for v in patch[mask])) < 2:
            continue
        if any(np.array_equal(patch, np.rot90(patch, k=k)) for k in (1, 2, 3)):
            continue
        return patch
    return None


def _random_sequence() -> list[tuple[str, int]] | None:
    length = int(_RNG.integers(4, 9))
    move_count = length // 2 if length % 2 == 0 else int(_RNG.choice([length // 2, length // 2 + 1]))
    rot_count = length - move_count
    if move_count == 0 or rot_count == 0:
        return None

    move_codes = _RNG.choice(np.array(list(MOVE_DELTAS.keys()), dtype=int), size=move_count, replace=True)
    rot_codes = _RNG.choice(np.array(list(ROTATION_STEPS.keys()), dtype=int), size=rot_count, replace=True)
    if move_count >= 2 and len(set(int(v) for v in move_codes)) < 2:
        return None
    if rot_count >= 2 and len(set(int(v) for v in rot_codes)) < 2:
        return None

    move_positions = set(int(v) for v in _RNG.choice(np.arange(length), size=move_count, replace=False))
    moves_iter = iter(int(v) for v in move_codes)
    rots_iter = iter(int(v) for v in rot_codes)
    instructions: list[tuple[str, int]] = []
    for idx in range(length):
        if idx in move_positions:
            instructions.append(("move", next(moves_iter)))
        else:
            instructions.append(("rot", next(rots_iter)))
    return instructions


def _translation_span(instructions: list[tuple[str, int]]) -> tuple[int, int, int, int]:
    dr = 0
    dc = 0
    min_dr = 0
    max_dr = 0
    min_dc = 0
    max_dc = 0
    for kind, code in instructions:
        if kind != "move":
            continue
        step_r, step_c = MOVE_DELTAS[code]
        dr += step_r
        dc += step_c
        min_dr = min(min_dr, dr)
        max_dr = max(max_dr, dr)
        min_dc = min(min_dc, dc)
        max_dc = max(max_dc, dc)
    return min_dr, max_dr, min_dc, max_dc


def _mark_forbidden(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _candidate_slots(forbidden: np.ndarray) -> list[tuple[int, int]]:
    h, w = forbidden.shape
    slots: list[tuple[int, int]] = []
    for r in range(0, h - 1, 3):
        for c in range(0, w - 1, 3):
            if forbidden[r : r + 2, c : c + 2].any():
                continue
            slots.append((r, c))
    return slots


def _instructions_from_grid(grid: np.ndarray) -> list[tuple[str, int]]:
    components = _find_components(grid != BLACK)
    dominoes = []
    for coords in components:
        if len(coords) != 2:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if (r0 == r1 and c1 == c0 + 1) or (c0 == c1 and r1 == r0 + 1):
            dominoes.append(coords)
    return [_decode_instruction(grid, coords) for coords in sorted(dominoes, key=_sort_key)]


def _validate_generated_case(
    grid: np.ndarray,
    patch: np.ndarray,
    start_top_left: tuple[int, int],
    instructions: list[tuple[str, int]],
) -> bool:
    components = _find_components(grid != BLACK)
    domino_count = 0
    workpiece_count = 0
    maroon_count = int(np.count_nonzero(grid == MAROON))
    for coords in components:
        size = len(coords)
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if size == 2 and ((h == 1 and w == 2) or (h == 2 and w == 1)):
            domino_count += 1
        elif size >= 3:
            workpiece_count += 1
    if maroon_count != 1 or domino_count != len(instructions) or workpiece_count != 1:
        return False

    detected = _instructions_from_grid(grid)
    if detected != instructions:
        return False

    expected_patch, expected_top_left = _apply_instructions(patch, start_top_left, instructions)
    out = main(grid)
    expected = np.full_like(grid, BLACK)
    _stamp_patch(expected, expected_patch, expected_top_left)
    return np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        side = int(_RNG.choice([3, 4]))
        patch = _random_workpiece(side)
        instructions = _random_sequence()
        if patch is None or instructions is None:
            continue

        h = int(_RNG.integers(15, 21))
        w = int(_RNG.integers(15, 21))

        min_dr, max_dr, min_dc, max_dc = _translation_span(instructions)
        row_candidates = [
            r
            for r in range(0, 4)
            if r + min_dr >= 0 and r + max_dr <= h - side
        ]
        col_candidates = [
            c
            for c in range(0, 4)
            if c + min_dc >= 0 and c + max_dc <= w - side
        ]
        if not row_candidates or not col_candidates:
            continue

        start_top_left = (int(_RNG.choice(row_candidates)), int(_RNG.choice(col_candidates)))
        grid = np.full((h, w), BLACK, dtype=int)
        _stamp_patch(grid, patch, start_top_left)

        forbidden = np.zeros((h, w), dtype=bool)
        _mark_forbidden(
            forbidden,
            start_top_left[0],
            start_top_left[1],
            start_top_left[0] + side - 1,
            start_top_left[1] + side - 1,
            pad=1,
        )

        anchor = (h - 1, w - 1)
        if forbidden[anchor]:
            continue
        grid[anchor] = MAROON
        _mark_forbidden(forbidden, anchor[0], anchor[1], anchor[0], anchor[1], pad=1)

        slots = _candidate_slots(forbidden)
        if len(slots) < len(instructions):
            continue
        chosen_indices = sorted(int(v) for v in _RNG.choice(np.arange(len(slots)), size=len(instructions), replace=False))
        chosen_slots = [slots[idx] for idx in chosen_indices]

        signatures: set[tuple[str, int, int]] = set()
        okay = True
        for (kind, code), (r, c) in zip(instructions, chosen_slots):
            if kind == "move":
                orient = "H"
                options = [
                    int(color)
                    for color in DOMINO_SECOND_COLORS
                    if color != code and (orient, code, int(color)) not in signatures
                ]
                if not options:
                    okay = False
                    break
                second = int(_RNG.choice(np.array(options, dtype=int)))
                grid[r, c] = code
                grid[r, c + 1] = second
                signatures.add((orient, code, second))
            else:
                orient = "V"
                options = [
                    int(color)
                    for color in DOMINO_SECOND_COLORS
                    if color != code and (orient, code, int(color)) not in signatures
                ]
                if not options:
                    okay = False
                    break
                second = int(_RNG.choice(np.array(options, dtype=int)))
                grid[r, c] = code
                grid[r + 1, c] = second
                signatures.add((orient, code, second))
        if not okay:
            continue

        if _validate_generated_case(grid, patch, start_top_left, instructions):
            return grid

    raise RuntimeError("failed to generate a valid domino-instruction ARC input")
