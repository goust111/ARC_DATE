from __future__ import annotations

import random

import numpy as np


UNKNOWN = 4
HALO = 7
TILE_COLOR_POOL = (1, 2, 3)
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
TILE_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))
DEFECT_COUNT_RANGE = (2, 5)
DEFECT_CLEARANCE = 3
GEN_ATTEMPTS = 400
TILE_ATTEMPTS = 150
DEFECT_ATTEMPTS = 120


def _tile_grid(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr = np.arange(h) % template.shape[0]
    cc = np.arange(w) % template.shape[1]
    return template[np.ix_(rr, cc)]


def _all_minimal_candidates(grid: np.ndarray, unknown: int | None = None) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if unknown is None:
        observed = np.argwhere(np.ones_like(g, dtype=bool))
    else:
        observed = np.argwhere(g != unknown)
    if observed.size == 0:
        raise ValueError("cannot infer a template from an all-unknown grid")

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            template = np.full((ph, pw), -1, dtype=int)
            ok = True
            for r, c in observed:
                rr = int(r % ph)
                cc = int(c % pw)
                value = int(g[r, c])
                if template[rr, cc] == -1:
                    template[rr, cc] = value
                elif template[rr, cc] != value:
                    ok = False
                    break
            if ok and np.all(template != -1):
                candidates.append((ph * pw, ph, pw, template.copy()))

    if not candidates:
        raise ValueError("no repeating template matches the observed cells")

    candidates.sort(key=lambda item: (item[0], item[1], item[2], tuple(item[3].ravel().tolist())))
    best_area = candidates[0][0]
    return [template for area, _ph, _pw, template in candidates if area == best_area]


def _infer_minimal_tile(grid: np.ndarray, unknown: int = UNKNOWN) -> np.ndarray:
    candidates = _all_minimal_candidates(grid, unknown=unknown)
    if len(candidates) != 1:
        raise ValueError("smallest repeating tile is not unique")
    return candidates[0]


def _minimal_period_shape(grid: np.ndarray) -> tuple[int, int]:
    candidates = _all_minimal_candidates(grid, unknown=None)
    best = candidates[0]
    return int(best.shape[0]), int(best.shape[1])


def _halo_mask(shape: tuple[int, int], defect_positions: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c in defect_positions:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the repeated tile, repair the off-pattern pixels, and surround each repaired defect with an orange halo.

    Input structure:
    - A 2D grid is mostly filled by a small rectangular color tile repeated across the canvas.
    - The repeated tile uses only non-yellow, non-orange colors, while a few interior yellow pixels mark suspicious locations.
    - Grid edges remain part of the repeated pattern; suspicious pixels never need halo clipping to define the rule.

    Rule:
    - Treat yellow (4) cells as unknown and infer the unique smallest rectangular repeating tile consistent with the other cells.
    - Tile that pattern over the whole grid to obtain the expected color at every position.
    - Every position whose actual color differs from the expected tiled color is a defect pixel.
    - Replace each defect with its expected tile color, then paint orange (7) on every other cell in the defect's 8-neighborhood.

    Key insight:
    - The yellow cells are not identified by local shape; they become defects only after a global tiling reference is reconstructed and compared against the observed grid.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    template = _infer_minimal_tile(g, unknown=UNKNOWN)
    expected = _tile_grid(template, g.shape)
    defect_mask = g != expected
    defect_positions = [(int(r), int(c)) for r, c in np.argwhere(defect_mask)]

    out = expected.copy()
    halo = _halo_mask(g.shape, defect_positions)
    out[halo & ~defect_mask] = HALO
    return out


def _sample_tile(tile_h: int, tile_w: int, palette: tuple[int, ...]) -> np.ndarray:
    for _ in range(TILE_ATTEMPTS):
        tile = np.random.choice(np.array(palette, dtype=int), size=(tile_h, tile_w))
        if len(set(map(int, tile.ravel()))) != len(palette):
            continue
        if len({tuple(map(int, row)) for row in tile}) < 2:
            continue
        if len({tuple(map(int, col)) for col in tile.T}) < 2:
            continue
        if _minimal_period_shape(tile) != (tile_h, tile_w):
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a primitive tile")


def _visible_residue_classes(input_grid: np.ndarray, tile_shape: tuple[int, int]) -> bool:
    th, tw = tile_shape
    for rr in range(th):
        for cc in range(tw):
            seen = False
            for r in range(rr, input_grid.shape[0], th):
                for c in range(cc, input_grid.shape[1], tw):
                    if int(input_grid[r, c]) != UNKNOWN:
                        seen = True
                        break
                if seen:
                    break
            if not seen:
                return False
    return True


def _far_enough(candidate: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(
        max(abs(candidate[0] - r), abs(candidate[1] - c)) >= DEFECT_CLEARANCE
        for r, c in chosen
    )


def _choose_defects(expected: np.ndarray, count: int, tile_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = expected.shape
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)

    by_color: dict[int, list[tuple[int, int]]] = {}
    for pos in candidates:
        by_color.setdefault(int(expected[pos]), []).append(pos)

    chosen: list[tuple[int, int]] = []
    used_residues: set[tuple[int, int]] = set()
    colors = list(by_color)
    random.shuffle(colors)

    for color in colors:
        for pos in by_color[color]:
            residue = (pos[0] % tile_shape[0], pos[1] % tile_shape[1])
            if residue in used_residues:
                continue
            if _far_enough(pos, chosen):
                chosen.append(pos)
                used_residues.add(residue)
                break
        if len(chosen) >= min(count, max(2, len(colors))):
            break

    for pos in candidates:
        if len(chosen) >= count:
            break
        residue = (pos[0] % tile_shape[0], pos[1] % tile_shape[1])
        if residue in used_residues and len(used_residues) < tile_shape[0] * tile_shape[1]:
            continue
        if _far_enough(pos, chosen):
            chosen.append(pos)
            used_residues.add(residue)

    if len(chosen) != count:
        raise RuntimeError("failed to place enough well-separated defects")
    return chosen


def _expected_output(base: np.ndarray, defects: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    halo = _halo_mask(base.shape, defects)
    defect_mask = np.zeros(base.shape, dtype=bool)
    for r, c in defects:
        defect_mask[r, c] = True
    out[halo & ~defect_mask] = HALO
    return out


def _semantic_checks(
    tile: np.ndarray,
    base: np.ndarray,
    inp: np.ndarray,
    out: np.ndarray,
    defects: list[tuple[int, int]],
) -> None:
    tile_shape = tile.shape
    if _minimal_period_shape(tile) != tile_shape:
        raise AssertionError("sampled tile is not primitive")
    if not _visible_residue_classes(inp, tile_shape):
        raise AssertionError("some tile residues are fully hidden")
    if len(_all_minimal_candidates(inp, unknown=UNKNOWN)) != 1:
        raise AssertionError("smallest tile should remain uniquely inferable")

    expected_colors = {int(base[r, c]) for r, c in defects}
    if len(expected_colors) < min(2, len(set(map(int, tile.ravel())))):
        raise AssertionError("defects should cover more than one expected color when possible")

    for r, c in defects:
        if not (1 <= r < inp.shape[0] - 1 and 1 <= c < inp.shape[1] - 1):
            raise AssertionError("defects must stay off the outer border")
        if int(inp[r, c]) != UNKNOWN:
            raise AssertionError("all defects must be yellow in the input")

    for i, a in enumerate(defects):
        for b in defects[i + 1 :]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < DEFECT_CLEARANCE:
                raise AssertionError("defect halos must not overlap")

    if np.any(inp == HALO):
        raise AssertionError("orange must appear only in the output")
    if np.any(out == UNKNOWN):
        raise AssertionError("yellow should be erased in the output")

    solver_out = main(inp)
    if not np.array_equal(solver_out, out):
        raise AssertionError("solver output does not match the constructed target")

    mismatch = inp != base
    if {(int(r), int(c)) for r, c in np.argwhere(mismatch)} != set(defects):
        raise AssertionError("input should differ from the clean tiling only at defect cells")


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    tile_h, tile_w = random.choice(TILE_SHAPES)
    palette_size = random.choice((2, 3))
    palette = tuple(random.sample(TILE_COLOR_POOL, palette_size))

    max_defects = min(
        DEFECT_COUNT_RANGE[1],
        max(DEFECT_COUNT_RANGE[0], ((h - 2) // DEFECT_CLEARANCE + 1) * ((w - 2) // DEFECT_CLEARANCE + 1)),
    )
    defect_count = random.randint(DEFECT_COUNT_RANGE[0], max_defects)

    try:
        tile = _sample_tile(tile_h, tile_w, palette)
        base = _tile_grid(tile, (h, w))
        defects = _choose_defects(base, defect_count, (tile_h, tile_w))
    except Exception:
        return None

    inp = base.copy()
    for r, c in defects:
        inp[r, c] = UNKNOWN
    out = _expected_output(base, defects)

    try:
        _semantic_checks(tile, base, inp, out, defects)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
