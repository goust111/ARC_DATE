from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
SPRITE_SIDE = 3
CELL_COUNT = SPRITE_SIDE * SPRITE_SIDE
GEN_ATTEMPTS = 400
PROFILE_SEQUENCE = ("left_only", "odd_split", "even_split")

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_GEN_CALL_COUNT = 0


def _is_4_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(np.sum(mask))


def _build_candidate_sprites() -> dict[int, list[np.ndarray]]:
    candidates: dict[int, list[np.ndarray]] = {count: [] for count in range(2, 6)}
    for foreground_count in range(2, 6):
        for cells in itertools.combinations(range(CELL_COUNT), foreground_count):
            mask = np.zeros((SPRITE_SIDE, SPRITE_SIDE), dtype=bool)
            for idx in cells:
                mask[idx // SPRITE_SIDE, idx % SPRITE_SIDE] = True
            if not _is_4_connected(mask):
                continue
            if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
                continue
            candidates[foreground_count].append(mask)
    for foreground_count, masks in candidates.items():
        assert masks, f"expected candidate sprites for F={foreground_count}"
    return candidates


CANDIDATE_SPRITES = _build_candidate_sprites()


def _parse_sprite(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (SPRITE_SIDE, SPRITE_SIDE), "expected a 3x3 sprite"

    colors = set(map(int, np.unique(g)))
    assert BLACK in colors, "expected black background pixels"
    foreground_colors = colors - {BLACK}
    assert len(foreground_colors) == 1, "expected exactly one foreground color"

    foreground_count = int(np.count_nonzero(g != BLACK))
    black_count = CELL_COUNT - foreground_count
    assert 0 < foreground_count <= black_count * black_count, "expected enough slots for all copies"
    return g, foreground_count, black_count


def _quadrant_rotation(slot_r: int, slot_c: int, side: int, middle_to_upper_left: bool = True) -> int:
    split = (side + 1) // 2 if middle_to_upper_left else side // 2
    top = slot_r < split
    left = slot_c < split
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 2
    return 3


def _render_layout(
    sprite: np.ndarray,
    slot_side: int,
    copy_count: int,
    *,
    middle_to_upper_left: bool = True,
    right_turn: int = 1,
    lower_left_turn: int = 2,
    lower_right_turn: int = 3,
) -> np.ndarray:
    out = np.zeros((SPRITE_SIDE * slot_side, SPRITE_SIDE * slot_side), dtype=int)
    for index in range(copy_count):
        slot_r, slot_c = divmod(index, slot_side)
        quadrant = _quadrant_rotation(slot_r, slot_c, slot_side, middle_to_upper_left=middle_to_upper_left)
        if quadrant == 0:
            turn = 0
        elif quadrant == 1:
            turn = right_turn
        elif quadrant == 2:
            turn = lower_left_turn
        else:
            turn = lower_right_turn
        tile = np.rot90(sprite, k=turn)
        r0 = slot_r * SPRITE_SIDE
        c0 = slot_c * SPRITE_SIDE
        out[r0 : r0 + SPRITE_SIDE, c0 : c0 + SPRITE_SIDE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile a 3x3 sprite into a black block grid whose size and per-copy rotation both come from the sprite's pixel counts.

    Input structure:
    - The input is a 3x3 sprite with black background and exactly one non-black foreground color.
    - The foreground pixel count F is between 2 and 5, so the black-pixel count B = 9 - F is between 4 and 7.
    - The sprite itself is the template copied into every filled slot.

    Rule:
    - Count the foreground pixels F and black pixels B.
    - Build a black output canvas split into a B x B array of 3x3 slots.
    - Fill the first F slots in row-major order with copies of the input sprite.
    - Rotate each filled copy according to its slot quadrant in the B x B array: upper-left 0 degrees, upper-right 90 degrees counterclockwise, lower-left 180 degrees, lower-right 270 degrees counterclockwise; when B is odd, the middle row and middle column belong to the upper-left side of the split.
    - Leave every remaining slot black.

    Key insight:
    - The same two pixel counts control both how many copies appear and where the quadrant boundary falls, so changing one foreground pixel changes the layout size and which rotations the early row-major copies receive.
    """
    sprite, foreground_count, black_count = _parse_sprite(grid)
    assert 2 <= foreground_count <= 5, "expected the puzzle's 2..5 foreground-pixel range"
    return _render_layout(sprite, black_count, foreground_count)


def _solve_no_rotation(grid: np.ndarray) -> np.ndarray:
    sprite, foreground_count, black_count = _parse_sprite(grid)
    return _render_layout(sprite, black_count, foreground_count, right_turn=0, lower_left_turn=0, lower_right_turn=0)


def _solve_wrong_middle_split(grid: np.ndarray) -> np.ndarray:
    sprite, foreground_count, black_count = _parse_sprite(grid)
    return _render_layout(sprite, black_count, foreground_count, middle_to_upper_left=False)


def _solve_right_turn_270(grid: np.ndarray) -> np.ndarray:
    sprite, foreground_count, black_count = _parse_sprite(grid)
    return _render_layout(sprite, black_count, foreground_count, right_turn=3)


def _solve_side_from_foreground(grid: np.ndarray) -> np.ndarray:
    sprite, foreground_count, _black_count = _parse_sprite(grid)
    return _render_layout(sprite, foreground_count, foreground_count)


def _parse_generated_sprite(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    sprite, foreground_count, black_count = _parse_sprite(grid)
    mask = sprite != BLACK
    assert 2 <= foreground_count <= 5
    assert 4 <= black_count <= 7
    assert _is_4_connected(mask), "generated sprite must stay 4-connected"
    assert len({np.rot90(mask, k).tobytes() for k in range(4)}) == 4, "generated sprite must expose rotation"
    return sprite, mask, foreground_count, black_count


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    sprite, mask, foreground_count, black_count = _parse_generated_sprite(inp)
    assert out.shape == (SPRITE_SIDE * black_count, SPRITE_SIDE * black_count), "unexpected output size"
    assert out.ndim == 2 and max(out.shape) <= 30, "output must stay within ARC size limits"

    occupied_slots = 0
    for index in range(black_count * black_count):
        slot_r, slot_c = divmod(index, black_count)
        r0 = slot_r * SPRITE_SIDE
        c0 = slot_c * SPRITE_SIDE
        tile = out[r0 : r0 + SPRITE_SIDE, c0 : c0 + SPRITE_SIDE]
        if index < foreground_count:
            occupied_slots += 1
            expected_turn = _quadrant_rotation(slot_r, slot_c, black_count, middle_to_upper_left=True)
            expected = np.rot90(sprite, k=expected_turn)
            assert np.array_equal(tile, expected), "filled slot does not match the specified rotation"
            assert int(np.count_nonzero(tile)) == foreground_count, "each filled slot must copy the full sprite"
        else:
            assert np.all(tile == BLACK), "unused slots must remain black"

    assert occupied_slots == foreground_count, "must fill exactly F row-major slots"
    assert int(np.count_nonzero(out)) == foreground_count * foreground_count, "foreground area must scale by F"

    if foreground_count >= 4:
        right_slots = []
        split = (black_count + 1) // 2
        for index in range(foreground_count):
            slot_r, slot_c = divmod(index, black_count)
            if slot_r < split and slot_c >= split:
                r0 = slot_r * SPRITE_SIDE
                c0 = slot_c * SPRITE_SIDE
                right_slots.append(out[r0 : r0 + SPRITE_SIDE, c0 : c0 + SPRITE_SIDE])
        assert right_slots, "F >= 4 should reach the upper-right quadrant"
        expected_right = np.rot90(sprite, 1)
        assert any(np.array_equal(tile, expected_right) for tile in right_slots), "upper-right slots must rotate 90 degrees"
        assert not np.array_equal(expected_right, sprite), "generated sprite must make rotation visible"

    rotated_masks = {np.rot90(mask, k).tobytes() for k in range(4)}
    assert len(rotated_masks) == 4, "generator must avoid rotational symmetry"


def _quality_checks(inp: np.ndarray, out: np.ndarray, profile: str) -> bool:
    sprite, _mask, foreground_count, black_count = _parse_generated_sprite(inp)

    if np.array_equal(out, _solve_side_from_foreground(inp)):
        return False

    if profile == "left_only":
        if foreground_count not in (2, 3):
            return False
        if black_count not in (6, 7):
            return False
        return True

    if np.array_equal(out, _solve_no_rotation(inp)):
        return False

    if profile == "odd_split":
        if foreground_count != 4 or black_count != 5:
            return False
        if np.array_equal(out, _solve_wrong_middle_split(inp)):
            return False
        split = (black_count + 1) // 2
        middle_col_slot = split - 1
        tile = out[0:SPRITE_SIDE, middle_col_slot * SPRITE_SIDE : (middle_col_slot + 1) * SPRITE_SIDE]
        if not np.array_equal(tile, sprite):
            return False
        return True

    if profile == "even_split":
        if foreground_count != 5 or black_count != 4:
            return False
        if np.array_equal(out, _solve_right_turn_270(inp)):
            return False
        right_tile = out[0:SPRITE_SIDE, 2 * SPRITE_SIDE : 3 * SPRITE_SIDE]
        if not np.array_equal(right_tile, np.rot90(sprite, 1)):
            return False
        return True

    raise ValueError(f"unknown profile: {profile}")


def _choose_profile() -> str:
    global _GEN_CALL_COUNT
    profile = PROFILE_SEQUENCE[_GEN_CALL_COUNT % len(PROFILE_SEQUENCE)]
    _GEN_CALL_COUNT += 1
    return profile


def _make_sprite(foreground_count: int) -> np.ndarray:
    mask = random.choice(CANDIDATE_SPRITES[foreground_count])
    color = random.choice(PALETTE)
    sprite = np.zeros((SPRITE_SIDE, SPRITE_SIDE), dtype=int)
    sprite[mask] = color
    return sprite


def _gen_attempt(profile: str) -> np.ndarray | None:
    if profile == "left_only":
        foreground_count = random.choice((2, 3))
    elif profile == "odd_split":
        foreground_count = 4
    elif profile == "even_split":
        foreground_count = 5
    else:
        raise ValueError(f"unknown profile: {profile}")

    inp = _make_sprite(foreground_count)
    try:
        out = main(inp)
        _validate_output_semantics(inp, out)
    except Exception:
        return None
    return inp if _quality_checks(inp, out, profile) else None


def generate_input() -> np.ndarray:
    profile = _choose_profile()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(profile)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample for profile={profile}")
