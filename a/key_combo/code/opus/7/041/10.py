from __future__ import annotations

import random

import numpy as np


BG = 0
TILE_SIDE = 4
MOTIF_SIDE = TILE_SIDE * 2
POSITIONS = ("UL", "UR", "LR", "LL")
ROTATION_BY_POSITION = {"UL": 0, "UR": 1, "LR": 2, "LL": 3}
POSITION_SLICES = {
    "UL": (slice(0, TILE_SIDE), slice(0, TILE_SIDE)),
    "UR": (slice(0, TILE_SIDE), slice(TILE_SIDE, MOTIF_SIDE)),
    "LR": (slice(TILE_SIDE, MOTIF_SIDE), slice(TILE_SIDE, MOTIF_SIDE)),
    "LL": (slice(TILE_SIDE, MOTIF_SIDE), slice(0, TILE_SIDE)),
}
GEN_ATTEMPTS = 300


def _rotate_cw(grid: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(grid, -int(quarter_turns) % 4).copy()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, side: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < side and 0 <= nc < side:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    side = int(mask.shape[0])
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, side):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _rotations_are_distinct(mask: np.ndarray) -> bool:
    return len({_rotate_cw(mask, k).tobytes() for k in range(4)}) == 4


def _enumerate_source_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (TILE_SIDE * TILE_SIDE)):
        pop = int(bits).bit_count()
        if not 5 <= pop <= 8:
            continue
        mask = np.array(
            [(bits >> idx) & 1 for idx in range(TILE_SIDE * TILE_SIDE)],
            dtype=int,
        ).reshape(TILE_SIDE, TILE_SIDE).astype(bool)
        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if not _rotations_are_distinct(mask):
            continue
        masks.append(mask.astype(int))
    assert masks, "expected a non-empty source-mask bank"
    return masks


SOURCE_MASKS = _enumerate_source_masks()


def _extract_motif(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BG)
    assert box is not None, "expected at least one non-background quadrant"
    r0, c0, r1, c1 = box
    motif = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = motif.shape
    assert h == w and h % 2 == 0, "expected a tight even square motif bbox"
    return motif


def _split_quadrants(motif: np.ndarray) -> dict[str, np.ndarray]:
    half = motif.shape[0] // 2
    slices = {
        "UL": (slice(0, half), slice(0, half)),
        "UR": (slice(0, half), slice(half, 2 * half)),
        "LR": (slice(half, 2 * half), slice(half, 2 * half)),
        "LL": (slice(half, 2 * half), slice(0, half)),
    }
    quadrants: dict[str, np.ndarray] = {}
    for pos, (rs, cs) in slices.items():
        quadrants[pos] = motif[rs, cs].copy()
    return quadrants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a four-quadrant rotational collage by inferring the missing tile's rotation.

    Input structure:
    - The grid is black except for one central even-sided square collage split into four equal quadrants.
    - Three quadrants contain the same monochrome sprite on black, each shown under a different 90-degree rotation.
    - The remaining quadrant is entirely black, and the unrotated source sprite may belong to any quadrant position.

    Rule:
    - Crop the tight foreground bbox to isolate the 8x8 collage and split it into the four quadrants.
    - Identify the blank quadrant, then use each visible quadrant as an anchor in the clockwise rotation cycle.
    - Verify that the other visible quadrants match the anchor after the position-implied quarter-turns.
    - Generate the unique missing rotation that is consistent with all successful anchors and place it into the blank quadrant.

    Key insight:
    - The absolute source corner is not observable, but the relative quarter-turn offsets between quadrant positions are; those offsets are enough to reconstruct the absent tile uniquely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    motif = _extract_motif(g)
    half = motif.shape[0] // 2
    slices = {
        "UL": (slice(0, half), slice(0, half)),
        "UR": (slice(0, half), slice(half, 2 * half)),
        "LR": (slice(half, 2 * half), slice(half, 2 * half)),
        "LL": (slice(half, 2 * half), slice(0, half)),
    }
    quadrants = _split_quadrants(motif)
    empty_positions = [pos for pos, tile in quadrants.items() if not np.any(tile != BG)]
    assert len(empty_positions) == 1, "expected exactly one blank quadrant"
    missing_pos = empty_positions[0]
    present_positions = [pos for pos in POSITIONS if pos != missing_pos]

    candidates: list[np.ndarray] = []
    for anchor_pos in present_positions:
        if all(
            np.array_equal(
                quadrants[pos],
                _rotate_cw(
                    quadrants[anchor_pos],
                    (ROTATION_BY_POSITION[pos] - ROTATION_BY_POSITION[anchor_pos]) % 4,
                ),
            )
            for pos in present_positions
        ):
            candidates.append(
                _rotate_cw(
                    quadrants[anchor_pos],
                    (ROTATION_BY_POSITION[missing_pos] - ROTATION_BY_POSITION[anchor_pos]) % 4,
                )
            )

    unique_tiles: dict[bytes, np.ndarray] = {}
    for tile in candidates:
        unique_tiles.setdefault(tile.tobytes(), tile)
    assert len(unique_tiles) == 1, "expected a unique missing quadrant tile"
    missing_tile = next(iter(unique_tiles.values()))

    out = motif.copy()
    rs, cs = slices[missing_pos]
    out[rs, cs] = missing_tile
    return out


def _build_complete_motif(mask: np.ndarray, color: int, source_pos: str) -> np.ndarray:
    source_tile = (mask.astype(int) * int(color)).astype(int)
    motif = np.zeros((MOTIF_SIDE, MOTIF_SIDE), dtype=int)
    for pos, (rs, cs) in POSITION_SLICES.items():
        turns = (ROTATION_BY_POSITION[pos] - ROTATION_BY_POSITION[source_pos]) % 4
        motif[rs, cs] = _rotate_cw(source_tile, turns)
    return motif


def _render_input(mask: np.ndarray, color: int, source_pos: str, missing_pos: str) -> tuple[np.ndarray, np.ndarray]:
    motif = _build_complete_motif(mask, color, source_pos)
    inp_motif = motif.copy()
    rs, cs = POSITION_SLICES[missing_pos]
    inp_motif[rs, cs] = BG

    margin_r = random.randint(1, 6)
    margin_c = random.randint(1, 6)
    grid = np.zeros((MOTIF_SIDE + 2 * margin_r, MOTIF_SIDE + 2 * margin_c), dtype=int)
    grid[margin_r : margin_r + MOTIF_SIDE, margin_c : margin_c + MOTIF_SIDE] = inp_motif
    return grid, motif


def _gen_attempt() -> np.ndarray | None:
    mask = random.choice(SOURCE_MASKS)
    color = random.randint(1, 9)
    source_pos = random.choice(POSITIONS)
    missing_pos = random.choice(POSITIONS)
    inp, expected = _render_input(mask, color, source_pos, missing_pos)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-missing-quadrant input")
