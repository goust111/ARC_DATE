from __future__ import annotations

import random

import numpy as np


BLACK = 0
ALL_COLORS = tuple(range(10))
QUAD_ORDER = ("tl", "tr", "br", "bl")  # clockwise around the separator cross
REFLECTION_KINDS = ("vertical", "horizontal", "diag", "anti")

GRID_SIDE_RANGE = (11, 25)  # odd side lengths only
MOTIF_COUNT_SMALL = (3, 4)
MOTIF_COUNT_LARGE = (4, 5)
MAX_GENERATION_ATTEMPTS = 400
PATTERN_ATTEMPTS = 200


MOTIFS = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Local copy of arc_utils.bbox semantics: (r0, c0, r1, c1), inclusive."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rot(tile: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(tile, quarter_turns % 4).copy()


def _reflect(tile: np.ndarray, kind: str) -> np.ndarray:
    if kind == "vertical":
        return np.fliplr(tile).copy()
    if kind == "horizontal":
        return np.flipud(tile).copy()
    if kind == "diag":
        return tile.T.copy()
    if kind == "anti":
        return np.flipud(np.fliplr(tile)).T.copy()
    raise ValueError(f"unknown reflection kind: {kind}")


def _all_rotations(tile: np.ndarray) -> list[np.ndarray]:
    return [_rot(tile, k) for k in range(4)]


def _all_reflections(tile: np.ndarray) -> list[np.ndarray]:
    return [_reflect(tile, kind) for kind in REFLECTION_KINDS]


def _dihedral_orbit_size(tile: np.ndarray) -> int:
    seen = set()
    for variant in _all_rotations(tile) + _all_reflections(tile):
        seen.add(variant.tobytes())
    return len(seen)


def _is_mirrored_impostor(tile: np.ndarray, expected: np.ndarray) -> bool:
    if any(np.array_equal(tile, rotated) for rotated in _all_rotations(expected)):
        return False
    return any(np.array_equal(tile, mirrored) for mirrored in _all_reflections(expected))


def _split_quadrants(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    side = int(g.shape[0])
    assert side % 2 == 1, "expected an odd side length"

    mid = side // 2
    quads = [
        g[:mid, :mid].copy(),
        g[:mid, mid + 1 :].copy(),
        g[mid + 1 :, mid + 1 :].copy(),
        g[mid + 1 :, :mid].copy(),
    ]
    for quad in quads[1:]:
        assert quad.shape == quads[0].shape, "all quadrants must have the same shape"
    return quads, mid


def _assemble_quadrants(source_grid: np.ndarray, quads: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(source_grid, dtype=int).copy()
    _, mid = _split_quadrants(out)
    out[:mid, :mid] = quads[0]
    out[:mid, mid + 1 :] = quads[1]
    out[mid + 1 :, mid + 1 :] = quads[2]
    out[mid + 1 :, :mid] = quads[3]
    return out


def _repair_candidates(quads: list[np.ndarray]) -> list[tuple[list[np.ndarray], int]]:
    candidates: list[tuple[list[np.ndarray], int]] = []
    for step in (-1, 1):
        for ref_idx, ref_tile in enumerate(quads):
            expected: list[np.ndarray] = []
            matches: list[bool] = []
            for pos in range(4):
                expected_tile = _rot(ref_tile, step * (pos - ref_idx))
                expected.append(expected_tile)
                matches.append(np.array_equal(quads[pos], expected_tile))
            if sum(matches) != 3:
                continue
            wrong_idx = matches.index(False)
            if not _is_mirrored_impostor(quads[wrong_idx], expected[wrong_idx]):
                continue
            candidates.append((expected, wrong_idx))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the unique mirrored impostor quadrant so the four quadrants form one 90-degree pinwheel family.

    Input structure:
    - The grid is a square odd-sized array whose middle row and middle column form a separator cross.
    - The four corner quadrants are equal-sized square tiles.
    - Three quadrants are correct quarter-turn copies of one asymmetric pattern, while one quadrant is a mirror of the tile that should be there.

    Rule:
    - Compare the four quadrants under both clockwise and counterclockwise 90-degree pinwheel relations.
    - Keep only the candidate families where exactly three quadrants already agree and the remaining quadrant is a reflection rather than a rotation.
    - Replace that impostor with the correctly rotated tile and leave the separator cross unchanged.

    Key insight:
    - The fake quadrant keeps the same local pixels but flips chirality, so only the rotation relation among quadrant positions can isolate it from the three true quarter-turn copies.
    """
    g = np.asarray(grid, dtype=int)
    quads, _mid = _split_quadrants(g)

    repaired_grids: dict[bytes, np.ndarray] = {}
    for repaired_quads, _wrong_idx in _repair_candidates(quads):
        out = _assemble_quadrants(g, repaired_quads)
        repaired_grids[out.tobytes()] = out

    assert repaired_grids, "expected one mirrored impostor quadrant"
    assert len(repaired_grids) == 1, "repair must be uniquely determined"
    return next(iter(repaired_grids.values()))


def _place_motif(
    pattern: np.ndarray, reserved: np.ndarray, shape: np.ndarray, color: int, gap: int
) -> bool:
    size = int(pattern.shape[0])
    h, w = map(int, shape.shape)
    positions = [(r, c) for r in range(size - h + 1) for c in range(size - w + 1)]
    random.shuffle(positions)

    for r, c in positions:
        if reserved[r : r + h, c : c + w].any():
            continue
        view = pattern[r : r + h, c : c + w]
        view[shape] = int(color)
        r0 = max(0, r - gap)
        c0 = max(0, c - gap)
        r1 = min(size, r + h + gap)
        c1 = min(size, c + w + gap)
        reserved[r0:r1, c0:c1] = True
        return True

    return False


def _foreground_span_ok(pattern: np.ndarray, bg: int) -> bool:
    box = _bbox(pattern != bg)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    need = max(3, pattern.shape[0] // 2 + 1)
    return h >= need and w >= need


def _gen_base_pattern(size: int, bg: int) -> np.ndarray:
    gap = 1 if size >= 7 else 0
    max_fg = (size * size) // 2

    for _ in range(PATTERN_ATTEMPTS):
        pattern = np.full((size, size), bg, dtype=int)
        reserved = np.zeros((size, size), dtype=bool)

        usable_colors = [color for color in ALL_COLORS if color != bg]
        palette_size = random.randint(2, 4)
        palette = random.sample(usable_colors, palette_size)
        motif_count = random.randint(*(MOTIF_COUNT_LARGE if size >= 8 else MOTIF_COUNT_SMALL))

        motif_colors = palette.copy()
        while len(motif_colors) < motif_count:
            motif_colors.append(random.choice(palette))
        random.shuffle(motif_colors)

        success = True
        for idx in range(motif_count):
            shape = _rot(random.choice(MOTIFS).astype(int), random.randrange(4)).astype(bool)
            if not _place_motif(pattern, reserved, shape, motif_colors[idx], gap):
                success = False
                break
        if not success:
            continue

        fg_count = int(np.count_nonzero(pattern != bg))
        if fg_count < size or fg_count > max_fg:
            continue
        if len(set(map(int, pattern[pattern != bg].flatten()))) < 2:
            continue
        if not _foreground_span_ok(pattern, bg):
            continue
        if _dihedral_orbit_size(pattern) != 8:
            continue
        return pattern

    raise RuntimeError("failed to generate an asymmetric base pattern")


def _build_expected_quadrants(base: np.ndarray, step: int, zero_rotation_idx: int) -> list[np.ndarray]:
    return [_rot(base, step * (pos - zero_rotation_idx)) for pos in range(4)]


def _gen_case() -> tuple[np.ndarray, np.ndarray, dict]:
    side = random.randrange(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1, 2)
    quad_size = side // 2
    bg = BLACK

    base = _gen_base_pattern(quad_size, bg)
    used_colors = set(map(int, np.unique(base)))
    used_colors.discard(bg)
    separator_color = random.choice([color for color in ALL_COLORS if color not in used_colors and color != bg])

    chirality = random.choice((-1, 1))
    zero_rotation_idx = random.randrange(4)
    expected_quads = _build_expected_quadrants(base, chirality, zero_rotation_idx)

    wrong_idx = random.randrange(4)
    reflection_kind = random.choice(REFLECTION_KINDS)
    input_quads = [quad.copy() for quad in expected_quads]
    input_quads[wrong_idx] = _reflect(expected_quads[wrong_idx], reflection_kind)
    assert _is_mirrored_impostor(input_quads[wrong_idx], expected_quads[wrong_idx])

    grid = np.full((side, side), bg, dtype=int)
    mid = side // 2
    grid[mid, :] = separator_color
    grid[:, mid] = separator_color

    inp = _assemble_quadrants(grid, input_quads)
    out = _assemble_quadrants(grid, expected_quads)
    meta = {
        "wrong_idx": wrong_idx,
        "reflection_kind": reflection_kind,
        "chirality": chirality,
        "zero_rotation_idx": zero_rotation_idx,
        "separator_color": separator_color,
    }
    return inp, out, meta


def _semantic_check(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    in_quads, mid = _split_quadrants(inp)
    out_quads, _ = _split_quadrants(out)

    if not np.array_equal(inp[mid, :], out[mid, :]):
        return False
    if not np.array_equal(inp[:, mid], out[:, mid]):
        return False

    wrong_idx = int(meta["wrong_idx"])
    changed = [not np.array_equal(a, b) for a, b in zip(in_quads, out_quads)]
    if changed != [idx == wrong_idx for idx in range(4)]:
        return False

    for idx in range(4):
        if idx == wrong_idx:
            if not _is_mirrored_impostor(in_quads[idx], out_quads[idx]):
                return False
        else:
            if not np.array_equal(in_quads[idx], out_quads[idx]):
                return False

    candidates = _repair_candidates(in_quads)
    if not candidates:
        return False
    repaired = {_assemble_quadrants(inp, quads).tobytes() for quads, _ in candidates}
    if repaired != {out.tobytes()}:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, out, meta = _gen_case()
        if not _semantic_check(inp, out, meta):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    raise RuntimeError("failed to generate a valid windmill-impostor input")
