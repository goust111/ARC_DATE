from __future__ import annotations

from itertools import combinations

import numpy as np


BG = 0
FG = 3
SPRITE_SIZE = 3
ATLAS_SIZE = 7
QUADRANT_ORDER = ("UL", "UR", "LL", "LR")
ATLAS_TOP_LEFT = {
    "UL": (0, 0),
    "UR": (0, 4),
    "LL": (4, 0),
    "LR": (4, 4),
}
CW_TURNS = {
    "UL": 0,
    "UR": 1,
    "LR": 2,
    "LL": 3,
}

_GEN_RNG = np.random.default_rng()
_SHAPE_BANK: list[np.ndarray] | None = None


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _rotate_cw(sprite: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(sprite, k=(-int(turns)) % 4)


def _quadrant_from_bbox(box: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    h, w = shape
    r0, c0, r1, c1 = box
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    half_r = h / 2.0
    half_c = w / 2.0
    assert center_r != half_r and center_c != half_c, "sprite center cannot lie on the center line"
    if center_r < half_r and center_c < half_c:
        return "UL"
    if center_r < half_r and center_c > half_c:
        return "UR"
    if center_r > half_r and center_c < half_c:
        return "LL"
    if center_r > half_r and center_c > half_c:
        return "LR"
    raise AssertionError("every sprite center must lie in exactly one quadrant")


def _extract_quadrant_sprites(grid: np.ndarray) -> dict[str, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({BG, FG}), "expected only black background and green sprites"
    components = _find_components4(g == FG)
    assert len(components) == 4, "expected exactly four 4-connected sprites"

    sprites: dict[str, np.ndarray] = {}
    for coords in components:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE)
        crop = _crop(g, box)
        quad = _quadrant_from_bbox(box, g.shape)
        assert quad not in sprites, "each quadrant must contain exactly one sprite"
        sprites[quad] = _rotate_cw(crop, CW_TURNS[quad])

    assert set(sprites) == set(QUADRANT_ORDER)
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each 3x3 green sprite by a quadrant-specific angle, then pack the rotated sprites into a 2x2 atlas.

    Input structure:
    - The grid is black except for exactly four separated green 4-connected sprites.
    - Each sprite fits in a 3x3 bounding box, and each bounding-box center lies strictly in one of the four quadrants.
    - There is exactly one sprite in each quadrant, using the grid center as the upper/lower and left/right divider.

    Rule:
    - Crop the 3x3 box of each sprite, preserving internal black holes.
    - Rotate the crop clockwise by an angle determined by its quadrant: UL 0, UR 90, LR 180, LL 270.
    - Order sprites by their original quadrant positions as [UL, UR, LL, LR].
    - Paste the rotated crops into a 7x7 black atlas at (0,0), (0,4), (4,0), and (4,4).

    Key insight:
    - The same spatial cue is used twice: quadrant membership chooses the rotation, while the original quadrant order chooses the packing slot.
    """
    sprites = _extract_quadrant_sprites(np.asarray(grid, dtype=int))
    out = np.full((ATLAS_SIZE, ATLAS_SIZE), BG, dtype=int)
    for quadrant in QUADRANT_ORDER:
        _paste(out, sprites[quadrant], *ATLAS_TOP_LEFT[quadrant])
    return out


def _touches_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _canonical_rotation_signature(mask: np.ndarray) -> bytes:
    return min(np.rot90(mask, k).tobytes() for k in range(4))


def _build_shape_bank() -> list[np.ndarray]:
    bank: dict[bytes, np.ndarray] = {}
    for bits in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.array([(bits >> i) & 1 for i in range(SPRITE_SIZE * SPRITE_SIZE)], dtype=bool).reshape(
            SPRITE_SIZE, SPRITE_SIZE
        )
        area = int(mask.sum())
        if not (4 <= area <= 8):
            continue
        if not _touches_full_bbox(mask):
            continue
        if len(_find_components4(mask)) != 1:
            continue
        if len({_rotate.tobytes() for _rotate in (np.rot90(mask, k) for k in range(4))}) != 4:
            continue
        bank.setdefault(_canonical_rotation_signature(mask), mask.copy())
    sprites = [np.where(mask, FG, BG).astype(int) for mask in bank.values()]
    assert len(sprites) >= 4, "need enough asymmetric 3x3 sprites"
    return sprites


def _shape_bank() -> list[np.ndarray]:
    global _SHAPE_BANK
    if _SHAPE_BANK is None:
        _SHAPE_BANK = _build_shape_bank()
    return _SHAPE_BANK


def _sample_sprite_set() -> dict[str, np.ndarray]:
    bank = _shape_bank()
    hole_indices = [i for i, sprite in enumerate(bank) if int(sprite[1, 1]) == BG]
    use_hole = bool(hole_indices) and float(_GEN_RNG.random()) < 0.55
    chosen: list[int] = []
    if use_hole:
        chosen.append(int(_GEN_RNG.choice(hole_indices)))
    remaining = [i for i in range(len(bank)) if i not in chosen]
    need = 4 - len(chosen)
    chosen.extend(int(i) for i in _GEN_RNG.choice(remaining, size=need, replace=False))
    _GEN_RNG.shuffle(chosen)

    sprites: dict[str, np.ndarray] = {}
    for quadrant, idx in zip(QUADRANT_ORDER, chosen):
        turns = int(_GEN_RNG.integers(0, 4))
        sprites[quadrant] = np.rot90(bank[idx], k=turns)
    return sprites


def _candidate_starts(limit: int, *, upper_or_left: bool) -> list[int]:
    starts: list[int] = []
    half = limit / 2.0
    for start in range(limit - SPRITE_SIZE + 1):
        center = start + 1
        if upper_or_left and center < half:
            starts.append(start)
        if not upper_or_left and center > half:
            starts.append(start)
    return starts


def _gap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return -1


def _boxes_have_clearance(
    placements: dict[str, tuple[int, int]],
    *,
    min_empty_gap: int = 2,
) -> bool:
    boxes = {
        quad: (r, c, r + SPRITE_SIZE - 1, c + SPRITE_SIZE - 1)
        for quad, (r, c) in placements.items()
    }
    for qa, qb in combinations(QUADRANT_ORDER, 2):
        ar0, ac0, ar1, ac1 = boxes[qa]
        br0, bc0, br1, bc1 = boxes[qb]
        row_gap = _gap_1d(ar0, ar1, br0, br1)
        col_gap = _gap_1d(ac0, ac1, bc0, bc1)
        if max(row_gap, col_gap) < min_empty_gap:
            return False
    return True


def _sample_placements(height: int, width: int) -> dict[str, tuple[int, int]] | None:
    upper_rows = _candidate_starts(height, upper_or_left=True)
    lower_rows = _candidate_starts(height, upper_or_left=False)
    left_cols = _candidate_starts(width, upper_or_left=True)
    right_cols = _candidate_starts(width, upper_or_left=False)
    if not upper_rows or not lower_rows or not left_cols or not right_cols:
        return None

    for _ in range(200):
        placements = {
            "UL": (int(_GEN_RNG.choice(upper_rows)), int(_GEN_RNG.choice(left_cols))),
            "UR": (int(_GEN_RNG.choice(upper_rows)), int(_GEN_RNG.choice(right_cols))),
            "LL": (int(_GEN_RNG.choice(lower_rows)), int(_GEN_RNG.choice(left_cols))),
            "LR": (int(_GEN_RNG.choice(lower_rows)), int(_GEN_RNG.choice(right_cols))),
        }
        if _boxes_have_clearance(placements):
            return placements
    return None


def _render_input(
    height: int,
    width: int,
    sprites: dict[str, np.ndarray],
    placements: dict[str, tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for quadrant in QUADRANT_ORDER:
        _paste(grid, sprites[quadrant], *placements[quadrant])
    return grid


def _expected_output_from_generation(sprites: dict[str, np.ndarray]) -> np.ndarray:
    out = np.full((ATLAS_SIZE, ATLAS_SIZE), BG, dtype=int)
    for quadrant in QUADRANT_ORDER:
        rotated = _rotate_cw(sprites[quadrant], CW_TURNS[quadrant])
        _paste(out, rotated, *ATLAS_TOP_LEFT[quadrant])
    return out


def _packed_without_rotation(sprites: dict[str, np.ndarray]) -> np.ndarray:
    out = np.full((ATLAS_SIZE, ATLAS_SIZE), BG, dtype=int)
    for quadrant in QUADRANT_ORDER:
        _paste(out, sprites[quadrant], *ATLAS_TOP_LEFT[quadrant])
    return out


def _validate_generated_case(inp: np.ndarray, sprites: dict[str, np.ndarray], expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    components = _find_components4(inp == FG)
    if len(components) != 4:
        return False
    if np.array_equal(expected, _packed_without_rotation(sprites)):
        return False

    seen_quadrants: set[str] = set()
    for coords in components:
        box = _bbox(coords)
        if box[2] - box[0] != 2 or box[3] - box[1] != 2:
            return False
        quadrant = _quadrant_from_bbox(box, inp.shape)
        if quadrant in seen_quadrants:
            return False
        seen_quadrants.add(quadrant)
    return seen_quadrants == set(QUADRANT_ORDER)


def _gen_attempt() -> np.ndarray | None:
    height = int(_GEN_RNG.integers(15, 24))
    width = int(_GEN_RNG.integers(15, 24))
    sprites = _sample_sprite_set()
    placements = _sample_placements(height, width)
    if placements is None:
        return None
    inp = _render_input(height, width, sprites, placements)
    expected = _expected_output_from_generation(sprites)
    if not _validate_generated_case(inp, sprites, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-rotation atlas input")
