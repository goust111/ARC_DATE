from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SPRITE = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500
_GEN_COUNTER = 0

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
TILE = "tile"

HORIZONTAL_MASK = np.array(
    [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ],
    dtype=bool,
)
VERTICAL_MASK = np.array(
    [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ],
    dtype=bool,
)
SCATTER_MASK = np.array(
    [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ],
    dtype=bool,
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comp = np.zeros_like(mask, dtype=bool)
            for rr, cc in pts:
                comp[rr, cc] = True
            comps.append(comp)
    return comps


def _classify_instruction(mask: np.ndarray) -> str | None:
    if mask.shape != (3, 3):
        return None
    if np.array_equal(mask, HORIZONTAL_MASK):
        return HORIZONTAL
    if np.array_equal(mask, VERTICAL_MASK):
        return VERTICAL
    if np.array_equal(mask, SCATTER_MASK):
        return TILE
    return None


def _apply_transform(sprite: np.ndarray, kind: str) -> np.ndarray:
    if kind == HORIZONTAL:
        return np.flipud(sprite)
    if kind == VERTICAL:
        return np.fliplr(sprite)
    if kind == TILE:
        return np.tile(sprite, (2, 2))
    raise ValueError(f"unknown instruction kind: {kind}")


def _render_output(sprite: np.ndarray, instructions: list[dict]) -> np.ndarray:
    patches: list[np.ndarray] = []
    for ins in instructions:
        transformed = _apply_transform(sprite, ins["kind"])
        patch = np.where(transformed != BG, int(ins["color"]), BG).astype(int)
        patches.append(patch)

    out_h = max(p.shape[0] for p in patches)
    out_w = sum(p.shape[1] for p in patches) + len(patches) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    cursor = 0
    for i, patch in enumerate(patches):
        h, w = patch.shape
        out[:h, cursor : cursor + w] = patch
        cursor += w
        if i != len(patches) - 1:
            cursor += 1
    return out


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    sprite_mask = grid == SPRITE
    assert np.any(sprite_mask), "expected a gray reference sprite"
    comps = _component_masks(sprite_mask)
    assert len(comps) == 1, "expected exactly one gray component"
    box = _bbox(sprite_mask)
    assert box is not None
    sprite = _crop_bbox(grid, box)
    mask = sprite == SPRITE
    assert sprite.shape[0] >= 3 and sprite.shape[1] >= 3, "reference sprite must have a >=3x3 bbox"
    assert np.any(mask[1:-1, 1:-1]), "reference sprite must contain an interior pixel"
    assert not np.array_equal(mask, np.flipud(mask)), "reference sprite must not be vertically symmetric"
    assert not np.array_equal(mask, np.fliplr(mask)), "reference sprite must not be horizontally symmetric"
    return sprite


def _parse_instructions(grid: np.ndarray) -> list[dict]:
    inst_mask = (grid != BG) & (grid != SPRITE)
    box = _bbox(inst_mask)
    assert box is not None, "expected instruction pixels"
    H, W = grid.shape
    r_lo = max(0, box[0] - 2)
    r_hi = min(box[0], H - 3)
    candidates: list[list[dict]] = []

    for count in (2, 3):
        span_w = 4 * count - 1
        c_lo = max(0, box[1] - 2)
        c_hi = min(box[1], W - span_w)
        for r0 in range(r_lo, r_hi + 1):
            for c0 in range(c_lo, c_hi + 1):
                covered = np.zeros_like(inst_mask, dtype=bool)
                parsed: list[dict] = []
                ok = True
                for i in range(count):
                    cc = c0 + 4 * i
                    patch = grid[r0 : r0 + 3, cc : cc + 3]
                    if patch.shape != (3, 3) or np.any(patch == SPRITE):
                        ok = False
                        break
                    mask = (patch != BG) & (patch != SPRITE)
                    kind = _classify_instruction(mask)
                    if kind is None:
                        ok = False
                        break
                    colors = sorted({int(v) for v in patch[mask]})
                    if len(colors) != 1:
                        ok = False
                        break
                    covered[r0 : r0 + 3, cc : cc + 3] |= mask
                    parsed.append({"kind": kind, "color": colors[0], "left": cc})
                    if i != count - 1 and np.any(grid[r0 : r0 + 3, cc + 3] != BG):
                        ok = False
                        break
                if ok and not np.any(inst_mask & ~covered):
                    if len({item["kind"] for item in parsed}) >= 2:
                        candidates.append(parsed)

    assert len(candidates) == 1, "instruction panel layout must be uniquely recoverable"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each 3x3 instruction pattern by geometry, transform the gray reference sprite, and recolor the transformed copies by instruction color.

    Input structure:
    - The background is black.
    - On the left there is one gray 4-connected reference sprite with a nontrivial interior and no horizontal or vertical mirror symmetry.
    - On the right there are two or three monochrome 3x3 instruction patterns laid out from left to right with one blank column between them.

    Rule:
    - Crop the gray reference sprite to its bounding box.
    - For each right-side 3x3 instruction, read its geometry: a full middle row means flip the sprite top-to-bottom, a full middle column means flip the sprite left-to-right, and four occupied corners mean tile the sprite into a 2x2 copy.
    - Recolor each transformed copy from gray to that instruction pattern's foreground color.
    - Output only the transformed copies, arranged left to right with one blank column between them, top aligned inside the maximum output height.

    Key insight:
    - The small 3x3 figures are not output objects; their visible geometry implicitly encodes which operation to apply to the separate gray sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    sprite = _extract_sprite(g)
    instructions = _parse_instructions(g)
    return _render_output(sprite, instructions)


def _valid_sprite_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << 16):
        mask = np.zeros((4, 4), dtype=bool)
        for k in range(16):
            if bits & (1 << k):
                mask[k // 4, k % 4] = True
        if not (7 <= int(mask.sum()) <= 13):
            continue
        if len(_component_masks(mask)) != 1:
            continue
        if _bbox(mask) != (0, 0, 3, 3):
            continue
        if not np.any(mask[1:3, 1:3]):
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(np.flipud(mask), np.fliplr(mask)):
            continue
        masks.append(mask)
    assert masks, "expected at least one valid sprite mask"
    return masks


VALID_SPRITE_MASKS = _valid_sprite_masks()


def _instruction_patch(kind: str, color: int) -> np.ndarray:
    mask = {
        HORIZONTAL: HORIZONTAL_MASK,
        VERTICAL: VERTICAL_MASK,
        TILE: SCATTER_MASK,
    }[kind]
    return np.where(mask, int(color), BG).astype(int)


def _sample_instruction_kinds() -> list[str]:
    global _GEN_COUNTER
    mode = _GEN_COUNTER % 6
    _GEN_COUNTER += 1

    if mode == 0:
        kinds = [HORIZONTAL, VERTICAL, TILE]
    elif mode == 1:
        kinds = [HORIZONTAL, VERTICAL]
    elif mode == 2:
        kinds = [HORIZONTAL, TILE]
    elif mode == 3:
        kinds = [VERTICAL, TILE]
    elif mode == 4:
        kinds = [HORIZONTAL, TILE, HORIZONTAL]
    else:
        kinds = [VERTICAL, TILE, VERTICAL]

    random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray | None:
    sprite_mask = random.choice(VALID_SPRITE_MASKS)
    sprite = np.where(sprite_mask, SPRITE, BG).astype(int)

    kinds = _sample_instruction_kinds()
    colors = random.sample(PALETTE, len(kinds))
    right_w = 4 * len(kinds) - 1

    left_col = random.randint(0, 2)
    panel_gap = random.randint(2, 4)
    right_col = left_col + sprite.shape[1] + panel_gap
    sprite_top = random.randint(0, 4)
    inst_top = random.randint(0, 5)

    H = max(sprite_top + sprite.shape[0], inst_top + 3) + random.randint(0, 2)
    W = right_col + right_w + random.randint(0, 2)
    if H > 30 or W > 30:
        return None

    grid = np.full((H, W), BG, dtype=int)
    grid[sprite_top : sprite_top + 4, left_col : left_col + 4] = sprite
    for i, (kind, color) in enumerate(zip(kinds, colors)):
        patch = _instruction_patch(kind, color)
        c0 = right_col + 4 * i
        grid[inst_top : inst_top + 3, c0 : c0 + 3] = patch

    try:
        _validate_generated(grid, expected_kinds=kinds, expected_colors=colors)
    except AssertionError:
        return None
    return grid


def _validate_generated(grid: np.ndarray, *, expected_kinds: list[str], expected_colors: list[int]) -> None:
    sprite = _extract_sprite(grid)
    instructions = _parse_instructions(grid)
    assert [ins["kind"] for ins in instructions] == expected_kinds
    assert [ins["color"] for ins in instructions] == expected_colors
    assert len(instructions) in (2, 3)
    assert len({ins["kind"] for ins in instructions}) >= 2

    base_mask = sprite != BG
    expected_sizes = {
        HORIZONTAL: sprite.shape,
        VERTICAL: sprite.shape,
        TILE: (sprite.shape[0] * 2, sprite.shape[1] * 2),
    }
    for ins in instructions:
        transformed = _apply_transform(base_mask.astype(int), ins["kind"])
        assert transformed.shape == expected_sizes[ins["kind"]]
        if ins["kind"] == HORIZONTAL:
            assert not np.array_equal(transformed, base_mask.astype(int))
        if ins["kind"] == VERTICAL:
            assert not np.array_equal(transformed, base_mask.astype(int))

    rendered = _render_output(sprite, instructions)
    solved = main(grid)
    assert np.array_equal(solved, rendered)
    assert solved.ndim == 2 and max(solved.shape) <= 30

    signatures = set()
    for ins in instructions:
        patch = np.where(_apply_transform(sprite, ins["kind"]) != BG, ins["color"], BG)
        signatures.add((patch.shape, patch.tobytes()))
    assert len(signatures) == len(instructions), "every output block must stay distinguishable"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implicit-geometry instruction grid")
