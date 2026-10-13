from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 2
INFECT = 3

DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))

FRAME_SIDE_RANGE = (5, 7)
GRID_H_RANGE = (13, 19)
GRID_W_MAX = 25
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BG)
    assert box is not None, "expected at least one non-background pixel"
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _frame_perimeter_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    colors = set(map(int, np.unique(grid)))
    assert colors.issubset({BG, FRAME}), f"unexpected input colors: {sorted(colors)}"

    frame_mask = grid == FRAME
    box = _bbox(frame_mask)
    assert box is not None, "expected one red frame"
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    expected = _frame_perimeter_mask(grid.shape, box)
    assert np.array_equal(frame_mask, expected), "red pixels must form one single-pixel-wide closed rectangle"
    return box


def _infection_step(grid: np.ndarray, source_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = grid.copy()
    new_mask = np.zeros_like(source_mask, dtype=bool)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BG:
                continue

            has_diag_source = any(
                _in_bounds(grid.shape, r + dr, c + dc) and bool(source_mask[r + dr, c + dc])
                for dr, dc in DIAGONALS
            )
            if not has_diag_source:
                continue

            has_orth_nonblack = any(
                _in_bounds(grid.shape, r + dr, c + dc) and int(grid[r + dr, c + dc]) != BG
                for dr, dc in ORTHOGONALS
            )
            if has_orth_nonblack:
                continue

            out[r, c] = INFECT
            new_mask[r, c] = True

    return out, new_mask


def _infect_rounds(grid: np.ndarray, rounds: int = 2) -> tuple[np.ndarray, list[np.ndarray]]:
    assert rounds >= 0
    state = np.asarray(grid, dtype=int).copy()
    source = state == FRAME
    new_masks: list[np.ndarray] = []
    for _ in range(rounds):
        state, source = _infection_step(state, source)
        new_masks.append(source.copy())
    return state, new_masks


def _merge_with_vertical_mirror(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mirrored = np.fliplr(grid)
    merged = grid.copy()
    take_mirror = (merged == BG) & (mirrored != BG)
    merged[take_mirror] = mirrored[take_mirror]
    return merged, mirrored


def _transform(grid: np.ndarray, rounds: int = 2) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    infected, new_masks = _infect_rounds(grid, rounds=rounds)
    merged, _mirrored = _merge_with_vertical_mirror(infected)
    return _crop_nonblack(merged), infected, new_masks, merged


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infect diagonal-only corner cells around a red frame for two rounds, mirror the result, then crop the merged figure.

    Input structure:
    - The grid uses black as background and contains exactly one red single-pixel-wide closed axis-aligned rectangle.
    - The rectangle encloses only black pixels, and the rest of the canvas is also black.
    - No other foreground objects or colors are present in the input.

    Rule:
    - Identify the red rectangular frame.
    - Round 1: recolor every black cell that is diagonally adjacent to the red frame and orthogonally adjacent to no non-black cell as green.
    - Round 2: from the round-1 green cells, repeat the same diagonal-only infection rule, again forbidding orthogonal contact with any non-black cell.
    - Mirror the fully infected grid left-right across the canvas vertical axis, merge mirrored and original non-black pixels with the original grid taking priority on overlaps, then crop to the tight bbox of all non-black cells.

    Key insight:
    - The only cells that can spread are the newly created diagonal corner infections, so the fixed second round extends each corner outward once more before mirroring creates the missing symmetric half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _find_frame_bbox(g)
    out, _infected, _new_masks, _merged = _transform(g, rounds=2)
    return out


def _render_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _sample_row_start(height: int, frame_h: int) -> int:
    choices: list[int] = [2, height - frame_h - 2]
    if height - frame_h - 3 >= 3:
        choices.append(random.randint(3, height - frame_h - 3))
    return random.choice(choices)


def _sample_dimensions() -> tuple[int, int, int, int]:
    frame_h = random.randint(*FRAME_SIDE_RANGE)
    frame_w = random.randint(*FRAME_SIDE_RANGE)
    height = random.randint(max(GRID_H_RANGE[0], frame_h + 6), GRID_H_RANGE[1])
    min_width = max(2 * frame_w + 10, 21)
    width = random.randint(min_width, GRID_W_MAX)
    return height, width, frame_h, frame_w


def _sample_col_start(width: int, frame_w: int) -> int:
    max_col = (width - 2 * frame_w - 4) // 2
    assert max_col >= 3, "width must leave room for a left-biased frame and its mirror"
    return random.randint(3, max_col)


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        r0, c0, r1, c1 = _find_frame_bbox(grid)
        h, w = grid.shape
        assert 5 <= (r1 - r0 + 1) <= 7
        assert 5 <= (c1 - c0 + 1) <= 7
        assert c0 - 2 >= 1 and r0 - 2 >= 0 and r1 + 2 < h and c1 + 2 < w
        assert c0 + c1 < w - 1, "frame center must lie left of the canvas center"

        one_round_out, infected1, masks1, merged1 = _transform(grid, rounds=1)
        final_out, infected2, masks2, merged2 = _transform(grid, rounds=2)
        _ = one_round_out, infected1, merged1

        new1 = masks1[0]
        new1_again, new2 = masks2
        assert np.array_equal(new1, new1_again)
        assert int(new1.sum()) == 4, "round 1 should infect the four outside corner diagonals"
        assert int(new2.sum()) == 12, "round 2 should branch from each round-1 corner infection"

        assert np.any((merged2 != BG) & (infected2 == BG)), "mirror must add visible pixels"
        assert not np.array_equal(final_out, _crop_nonblack(infected2)), "mirror step must change the output"
        assert not np.array_equal(final_out, _transform(grid, rounds=1)[0]), "second infection round must matter"

        merged_box = _bbox(merged2 != BG)
        assert merged_box is not None
        assert merged_box != (0, 0, h - 1, w - 1), "crop step must remove background"

        actual = main(grid)
        assert np.array_equal(actual, final_out)
        assert INFECT in set(map(int, np.unique(actual)))
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height, width, frame_h, frame_w = _sample_dimensions()
        grid = np.zeros((height, width), dtype=int)

        r0 = _sample_row_start(height, frame_h)
        c0 = _sample_col_start(width, frame_w)
        _render_frame(grid, r0, c0, frame_h, frame_w)

        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid diagonal-infection mirror crop input")
