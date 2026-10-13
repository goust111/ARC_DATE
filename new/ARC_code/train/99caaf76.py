from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste
import random


_CROSS_MASK = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)


def _marker_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in g[g != int(bg)])
    assert counts, "expected foreground pixels"
    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    assert len(candidates) == 1, "expected a unique repeated marker color"
    marker = int(candidates[0])
    assert counts[marker] % 4 == 0, "marker pixels should form 4-pixel sprites"
    return marker


def _block_kind(block: np.ndarray, bg: int, marker: int) -> str:
    assert block.shape == (3, 3), "expected 3x3 half-block"
    marker_mask = block == int(marker)
    other_mask = (block != int(bg)) & ~marker_mask

    if int(marker_mask.sum()) == 4 and not bool(other_mask.any()):
        return "marker"

    if not bool(marker_mask.any()) and int(other_mask.sum()) == 5:
        assert np.array_equal(other_mask, _CROSS_MASK), "expected a 5-pixel cross"
        return "cross"

    raise AssertionError("expected either a marker block or a colored cross block")


def _paste_checked(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = out[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    assert target.shape == patch.shape, "target patch must fit in the grid"
    assert not bool(np.any((target != int(bg)) & (patch != int(bg)))), "moved objects must not overlap"
    paste(out, patch, int(r0), int(c0), transparent=int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marked cross object away from its marker to the grid edge while rotating the colored cross 180 degrees.

    Input structure:
    - The background is the dominant color.
    - Each 8-connected foreground object has nine pixels: a 4-pixel marker-color
      sprite attached to a 5-pixel non-marker colored cross.
    - Each object bbox is either 3x6 or 6x3, so the marker sprite is on exactly
      one side of the cross.

    Rule:
    - Move the whole object straight away from its marker side until the object
      bbox touches the corresponding grid edge.
    - Keep the marker sprite unchanged within the object.
    - Rotate the 3x3 colored cross block by 180 degrees before placing it.

    Key insight:
    The marker sprite is a directional tail; the cross is the leading payload,
    and only that payload turns around during the slide to the edge.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)
    marker = _marker_color(g, bg)

    out = np.full_like(g, int(bg))
    objects = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert objects, "expected at least one composite object"

    for obj in objects:
        assert int(obj["area"]) == 9, "each composite should contain 9 foreground pixels"
        r0, c0, _r1, _c1 = obj["bbox"]
        patch = crop_bbox(g, obj["bbox"])
        h, w = patch.shape
        moved = patch.copy()

        if (h, w) == (3, 6):
            left = patch[:, :3]
            right = patch[:, 3:]
            left_kind = _block_kind(left, bg, marker)
            right_kind = _block_kind(right, bg, marker)
            assert {left_kind, right_kind} == {"marker", "cross"}, "expected one marker and one cross"

            if left_kind == "marker":
                moved[:, 3:] = np.rot90(right, 2)
                target_r, target_c = int(r0), W - w
            else:
                moved[:, :3] = np.rot90(left, 2)
                target_r, target_c = int(r0), 0

        elif (h, w) == (6, 3):
            top = patch[:3, :]
            bottom = patch[3:, :]
            top_kind = _block_kind(top, bg, marker)
            bottom_kind = _block_kind(bottom, bg, marker)
            assert {top_kind, bottom_kind} == {"marker", "cross"}, "expected one marker and one cross"

            if top_kind == "marker":
                moved[3:, :] = np.rot90(bottom, 2)
                target_r, target_c = H - h, int(c0)
            else:
                moved[:3, :] = np.rot90(top, 2)
                target_r, target_c = 0, int(c0)

        else:
            raise AssertionError("each composite bbox should be 3x6 or 6x3")

        _paste_checked(out, moved, target_r, target_c, bg)

    return out

# --------------- generator (att2) ---------------

# Colors.
_GEN_BG = 8
_GEN_MARKER = 1
_GEN_CENTER = 4
_GEN_ARM_COLORS = [0, 2, 3, 5, 6, 7, 9]

# Object layout.
_GEN_SIDES = ["left", "right", "top", "bottom"]
_GEN_MARKER_COORDS = {
    "left": [(0, 0), (1, 1), (1, 2), (2, 0)],
    "right": [(0, 2), (1, 0), (1, 1), (2, 2)],
    "top": [(0, 0), (0, 2), (1, 1), (2, 1)],
    "bottom": [(0, 1), (1, 1), (2, 0), (2, 2)],
}
_GEN_ARM_POSITIONS = [(0, 1), (1, 0), (1, 2), (2, 1)]

# Canvas and placement bounds.
_GEN_OBJECT_COUNT_RANGE = (1, 4)
_GEN_MAX_SIDE = 18
_GEN_MIN_SIDE_BY_OBJECT_COUNT = {1: 8, 2: 8, 3: 10, 4: 12}
_GEN_PADDING = 1
_GEN_EDGE_GAP = 2
_GEN_HORIZONTAL_SIZE = (3, 6)
_GEN_VERTICAL_SIZE = (6, 3)

# Retry limits.
_GEN_MAX_ATTEMPTS = 300


def _gen_marker_block(side: str) -> np.ndarray:
    block = np.full((3, 3), _GEN_BG, dtype=int)
    for r, c in _GEN_MARKER_COORDS[side]:
        block[r, c] = _GEN_MARKER
    return block


def _gen_cross_block() -> np.ndarray:
    block = np.full((3, 3), _GEN_BG, dtype=int)
    block[1, 1] = _GEN_CENTER
    for (r, c), color in zip(_GEN_ARM_POSITIONS, random.sample(_GEN_ARM_COLORS, 4)):
        block[r, c] = color
    return block


def _gen_composite(side: str, cross: np.ndarray) -> np.ndarray:
    marker = _gen_marker_block(side)
    if side == "left":
        return np.concatenate([marker, cross], axis=1)
    if side == "right":
        return np.concatenate([cross, marker], axis=1)
    if side == "top":
        return np.concatenate([marker, cross], axis=0)
    return np.concatenate([cross, marker], axis=0)


def _gen_input_from_output(side: str, patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    if side == "left":
        out[:, 3:] = np.rot90(out[:, 3:], 2)
    elif side == "right":
        out[:, :3] = np.rot90(out[:, :3], 2)
    elif side == "top":
        out[3:, :] = np.rot90(out[3:, :], 2)
    else:
        out[:3, :] = np.rot90(out[:3, :], 2)
    return out


def _gen_can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != _GEN_BG):
        r = int(r0 + dr)
        c = int(c0 + dc)
        r1 = max(0, r - padding)
        r2 = min(H, r + padding + 1)
        c1 = max(0, c - padding)
        c2 = min(W, c + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != _GEN_BG):
            return False
    return True


def _gen_output_positions(side: str, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        return [(r, W - _GEN_HORIZONTAL_SIZE[1]) for r in range(H - _GEN_EDGE_GAP)]
    if side == "right":
        return [(r, 0) for r in range(H - _GEN_EDGE_GAP)]
    if side == "top":
        return [(H - _GEN_VERTICAL_SIZE[0], c) for c in range(W - _GEN_EDGE_GAP)]
    return [(0, c) for c in range(W - _GEN_EDGE_GAP)]


def _gen_input_positions(side: str, H: int, W: int, target_r: int, target_c: int) -> list[tuple[int, int]]:
    if side == "left":
        return [(target_r, c) for c in range(0, W - _GEN_HORIZONTAL_SIZE[1] - _GEN_PADDING)]
    if side == "right":
        return [(target_r, c) for c in range(_GEN_EDGE_GAP, W - _GEN_HORIZONTAL_SIZE[1] + _GEN_PADDING)]
    if side == "top":
        return [(r, target_c) for r in range(0, H - _GEN_VERTICAL_SIZE[0] - _GEN_PADDING)]
    return [(r, target_c) for r in range(_GEN_EDGE_GAP, H - _GEN_VERTICAL_SIZE[0] + _GEN_PADDING)]


def _gen_canvas_shape(n_obj: int) -> tuple[int, int]:
    min_side = _GEN_MIN_SIDE_BY_OBJECT_COUNT[int(n_obj)]
    return random.randint(min_side, _GEN_MAX_SIDE), random.randint(min_side, _GEN_MAX_SIDE)


def _gen_place_output_objects(expected: np.ndarray, n_obj: int) -> list[tuple[str, np.ndarray, int, int]] | None:
    H, W = expected.shape
    objects = []

    for _ in range(n_obj):
        sides = _GEN_SIDES[:]
        random.shuffle(sides)
        placed = False
        for side in sides:
            cross = _gen_cross_block()
            patch = _gen_composite(side, cross)
            positions = _gen_output_positions(side, H, W)
            random.shuffle(positions)
            for r0, c0 in positions:
                if _gen_can_place(expected, patch, r0, c0, padding=_GEN_PADDING):
                    paste(expected, patch, int(r0), int(c0), transparent=_GEN_BG)
                    objects.append((side, patch, r0, c0))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    return objects


def _gen_place_input_objects(shape: tuple[int, int], objects: list[tuple[str, np.ndarray, int, int]]) -> np.ndarray | None:
    H, W = shape
    inp = np.full(shape, _GEN_BG, dtype=int)
    random.shuffle(objects)
    for side, out_patch, target_r, target_c in objects:
        in_patch = _gen_input_from_output(side, out_patch)
        positions = _gen_input_positions(side, H, W, target_r, target_c)
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if _gen_can_place(inp, in_patch, r0, c0, padding=_GEN_PADDING):
                paste(inp, in_patch, int(r0), int(c0), transparent=_GEN_BG)
                placed = True
                break
        if not placed:
            return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_obj = random.randint(*_GEN_OBJECT_COUNT_RANGE)
    H, W = _gen_canvas_shape(n_obj)

    expected = np.full((H, W), _GEN_BG, dtype=int)
    objects = _gen_place_output_objects(expected, n_obj)
    if objects is None:
        return None

    inp = _gen_place_input_objects((H, W), objects)
    if inp is None:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 99caaf76 input")
