import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components


BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN

_GEN_ATTEMPTS = 400
_GEN_ROW_RANGE = (5, 8)
_GEN_HALF_WIDTHS = (2, 3, 4)
_GEN_CANVAS_MIN = 9
_GEN_CANVAS_MAX = 14
_GEN_AREA_RANGE = (12, 20)
_GEN_EDIT_RANGE = (1, 3)
_GEN_SEEN_INPUTS: set[bytes] = set()
_GEN_SEEN_OUTPUTS: set[bytes] = set()


def _sprite_mask(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, RED}, "expected only black background and one red sprite"

    components = find_components(
        g,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    red_components = [comp for comp in components if int(comp["color"]) == RED]
    assert len(red_components) == 1, "expected exactly one connected red sprite"
    assert len(components) == 1, "expected no extra foreground objects"

    mask = g == RED
    sprite_bbox = bbox(mask)
    assert sprite_bbox is not None, "expected a non-empty sprite"
    local = crop_bbox(mask, sprite_bbox).astype(bool)
    assert local.shape[1] % 2 == 1, "sprite bbox width must be odd"
    return sprite_bbox, local


def _xor_recolor(local: np.ndarray) -> np.ndarray:
    mirror = np.fliplr(local)
    diff = local ^ mirror
    same = local & mirror
    h, w = local.shape
    mid = w // 2

    out = np.zeros((h, w), dtype=int)
    out[same] = RED
    out[:, : mid + 1][diff[:, : mid + 1]] = GREEN
    out[:, mid + 1 :][diff[:, mid + 1 :]] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the red sprite against its own vertical midline, detect the XOR mismatches, and recolor the mismatched sides differently.

    Input structure:
    - The grid has black background and exactly one 4-connected red sprite.
    - The sprite is nearly left-right symmetric within its own bounding box.
    - The sprite bounding box has odd width, so its vertical symmetry axis runs through the center column.

    Rule:
    - Compute the red sprite's bounding box and use its vertical center column as the mirror axis.
    - Reflect the sprite horizontally inside that bounding box to create a virtual comparison copy.
    - Mark every bbox cell where the original sprite mask and its mirror differ.
    - Keep symmetric sprite pixels red, recolor mismatches on the left half green and mismatches on the right half blue, and leave background black.

    Key insight:
    - The mirror image is not drawn directly; it is only a reference used for a pixelwise XOR that exposes which cells break bilateral symmetry and on which side of the axis they lie.
    """
    sprite_bbox, local = _sprite_mask(grid)
    recolored = _xor_recolor(local)

    out = np.zeros_like(np.asarray(grid, dtype=int))
    r0, c0, r1, c1 = sprite_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = recolored
    return out


def _build_symmetric_profile() -> tuple[int, int, list[int]]:
    for _ in range(200):
        height = random.randint(*_GEN_ROW_RANGE)
        half_width = random.choice(_GEN_HALF_WIDTHS)
        current = random.randint(1, half_width)
        extents = [current]
        for _ in range(height - 1):
            current = min(half_width, max(0, current + random.choice([-1, 0, 1])))
            extents.append(current)

        if max(extents) < half_width:
            picks = random.sample(range(height), k=2 if height >= 2 else 1)
            for idx in picks:
                extents[idx] = half_width

        if len(set(extents)) == 1:
            idx = random.randrange(height)
            delta = -1 if extents[idx] == half_width else 1
            extents[idx] += delta

        if sum(1 + 2 * e for e in extents) not in range(_GEN_AREA_RANGE[0], _GEN_AREA_RANGE[1] + 1):
            continue
        if sum(e == half_width for e in extents) < 1:
            continue
        if not any(e < half_width for e in extents):
            continue
        return height, half_width, extents
    raise RuntimeError("failed to build a symmetric profile")


def _mask_from_extents(left: list[int], right: list[int], half_width: int) -> np.ndarray:
    width = 2 * half_width + 1
    mid = half_width
    mask = np.zeros((len(left), width), dtype=bool)
    for r, (ll, rr) in enumerate(zip(left, right)):
        mask[r, mid - ll : mid + rr + 1] = True
    return mask


def _edit_candidates(left: list[int], right: list[int], half_width: int, used_pairs: set[tuple[int, int]]) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    for r, (ll, rr) in enumerate(zip(left, right)):
        if ll > 0 and (r, ll) not in used_pairs:
            candidates.append(("remove_left", r, ll))
        if rr > 0 and (r, rr) not in used_pairs:
            candidates.append(("remove_right", r, rr))
        if ll < half_width and (r, ll + 1) not in used_pairs:
            candidates.append(("add_left", r, ll + 1))
        if rr < half_width and (r, rr + 1) not in used_pairs:
            candidates.append(("add_right", r, rr + 1))
    return candidates


def _apply_edit(left: list[int], right: list[int], edit: tuple[str, int, int]) -> None:
    kind, row, _offset = edit
    if kind == "remove_left":
        left[row] -= 1
    elif kind == "remove_right":
        right[row] -= 1
    elif kind == "add_left":
        left[row] += 1
    else:
        right[row] += 1


def _inject_asymmetry(extents: list[int], half_width: int) -> np.ndarray | None:
    left = extents.copy()
    right = extents.copy()
    edit_count = random.randint(*_GEN_EDIT_RANGE)
    used_pairs: set[tuple[int, int]] = set()
    used_kinds: set[str] = set()

    for step in range(edit_count):
        candidates = _edit_candidates(left, right, half_width, used_pairs)
        if not candidates:
            return None

        if step == 0 and any(c[0].startswith("remove") for c in candidates) and any(c[0].startswith("add") for c in candidates):
            pool = [c for c in candidates if c[0].startswith(random.choice(["remove", "add"]))]
        elif step == edit_count - 1 and len(used_kinds) == 1:
            missing_prefix = "add" if "remove" in used_kinds else "remove"
            pool = [c for c in candidates if c[0].startswith(missing_prefix)] or candidates
        else:
            pool = candidates

        edit = random.choice(pool)
        used_pairs.add((edit[1], edit[2]))
        used_kinds.add("remove" if edit[0].startswith("remove") else "add")
        _apply_edit(left, right, edit)

    return _mask_from_extents(left, right, half_width)


def _place_patch(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    min_rows = max(_GEN_CANVAS_MIN, h + 2)
    min_cols = max(_GEN_CANVAS_MIN, w + 2)
    max_rows = max(min_rows, _GEN_CANVAS_MAX)
    max_cols = max(min_cols, _GEN_CANVAS_MAX)
    height = random.randint(min_rows, max_rows)
    width = random.randint(min_cols, max_cols)

    grid = np.zeros((height, width), dtype=int)
    r0 = random.randint(1, height - h - 1)
    c0 = random.randint(1, width - w - 1)
    grid[r0 : r0 + h, c0 : c0 + w][mask] = RED
    return grid


def _mismatch_pair_count(local: np.ndarray) -> int:
    diff = local ^ np.fliplr(local)
    return int(diff.sum() // 2)


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        sprite_bbox, local = _sprite_mask(grid)
        out = main(grid)
    except AssertionError:
        return False

    area = int(local.sum())
    if not (_GEN_AREA_RANGE[0] <= area <= _GEN_AREA_RANGE[1]):
        return False

    h, w = local.shape
    if not (5 <= h <= 8 and w % 2 == 1 and 5 <= w <= 9):
        return False

    diff = local ^ np.fliplr(local)
    same = local & np.fliplr(local)
    mid = w // 2
    if not diff.any():
        return False
    if diff[:, mid].any():
        return False
    if not diff[:, :mid].any() or not diff[:, mid + 1 :].any():
        return False
    if _mismatch_pair_count(local) not in {1, 2, 3}:
        return False
    if int(same.sum()) < 6:
        return False
    if int(np.count_nonzero(local[:, 0])) == 0 or int(np.count_nonzero(local[:, -1])) == 0:
        return False

    expected_local = _xor_recolor(local)
    r0, c0, r1, c1 = sprite_bbox
    if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], expected_local):
        return False
    if np.any(out[:r0, :] != BLACK) or np.any(out[r1 + 1 :, :] != BLACK):
        return False
    if np.any(out[:, :c0] != BLACK) or np.any(out[:, c1 + 1 :] != BLACK):
        return False

    present = set(map(int, np.unique(out)))
    if not {GREEN, BLUE, RED} <= present:
        return False
    if np.array_equal(grid, out):
        return False
    return True


def _signature(grid: np.ndarray) -> bytes:
    return grid.tobytes() + str(tuple(map(int, grid.shape))).encode()


def _gen_attempt() -> np.ndarray | None:
    _height, half_width, extents = _build_symmetric_profile()
    mask = _inject_asymmetry(extents, half_width)
    if mask is None:
        return None

    grid = _place_patch(mask)
    if not _validate_instance(grid):
        return None

    inp_sig = _signature(grid)
    out_sig = _signature(main(grid))
    if inp_sig in _GEN_SEEN_INPUTS or out_sig in _GEN_SEEN_OUTPUTS:
        return None
    _GEN_SEEN_INPUTS.add(inp_sig)
    _GEN_SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid asymmetric-mirror sprite input")
