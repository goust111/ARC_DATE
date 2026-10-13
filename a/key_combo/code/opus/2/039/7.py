from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BACK_TO_FRONT = ("tl", "br", "bl", "tr")
QUADRANT_COLORS = {
    "tl": 2,
    "tr": 1,
    "bl": 3,
    "br": 4,
}
GEN_RETRY_LIMIT = 400
RNG = np.random.default_rng()


def _split_quadrants(grid: np.ndarray) -> dict[str, np.ndarray]:
    h, w = map(int, grid.shape)
    assert h % 2 == 0 and w % 2 == 0, "grid must split into four equal quadrants"
    hh, ww = h // 2, w // 2
    return {
        "tl": grid[:hh, :ww],
        "tr": grid[:hh, ww:],
        "bl": grid[hh:, :ww],
        "br": grid[hh:, ww:],
    }


def _recover_quadrant(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mirrored = np.fliplr(region)
    recovered = (region == BG) & (mirrored != BG)
    restored = region.copy()
    restored[recovered] = mirrored[recovered]
    assert np.array_equal(restored, np.fliplr(restored)), "quadrant must recover to vertical symmetry"
    return restored, recovered


def _overlay_quadrants(quadrants: dict[str, np.ndarray], *, transparent_recovered: bool) -> np.ndarray:
    sample = next(iter(quadrants.values()))
    out = np.full(sample.shape, BG, dtype=int)
    for name in BACK_TO_FRONT:
        restored, recovered = _recover_quadrant(quadrants[name])
        opaque = restored != BG
        if transparent_recovered:
            opaque &= ~recovered
        out[opaque] = restored[opaque]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the four quadrants onto one canvas after symmetric recovery, while treating recovered pixels as transparent.

    Input structure:
    - A black background grid has even height and width and splits into four equal quadrants.
    - Each quadrant contains exactly one non-black color forming a left-right symmetric motif with a few missing pixels erased to black.
    - The four quadrant colors are distinct, and every missing pixel is uniquely recoverable from its horizontal mirror mate.

    Rule:
    - For each quadrant, restore every black cell whose horizontal mirror cell is non-black, and mark those restored cells as recovered.
    - Create an output canvas the size of one quadrant.
    - Fold the quadrants onto that shared canvas in this back-to-front order: top-left, bottom-right, bottom-left, then top-right.
    - During overlay, black cells are transparent, and recovered cells are also transparent even though they have restored colors.
    - Only original non-black pixels are opaque and may overwrite lower layers.

    Key insight:
    - Symmetry recovery produces not just colors but also a hidden original-versus-recovered label; that label controls transparency during the quadrant overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    quadrants = _split_quadrants(g)
    seen_colors: set[int] = set()
    for name, region in quadrants.items():
        colors = {int(v) for v in np.unique(region) if int(v) != BG}
        assert len(colors) == 1, f"{name} must contain exactly one foreground color"
        color = next(iter(colors))
        assert color not in seen_colors, "quadrant foreground colors must be distinct"
        seen_colors.add(color)
        _recover_quadrant(region)

    assert len(seen_colors) == 4, "expected four colored quadrants"
    return _overlay_quadrants(quadrants, transparent_recovered=True)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _mirror_mask_left_right(half_mask: np.ndarray, width: int) -> np.ndarray:
    h = half_mask.shape[0]
    full = np.zeros((h, width), dtype=bool)
    for r, c in np.argwhere(half_mask):
        rr = int(r)
        cc = int(c)
        full[rr, cc] = True
        full[rr, width - 1 - cc] = True
    return full


def _generate_symmetric_mask(size: int) -> np.ndarray:
    half_w = (size + 1) // 2
    min_area = size + 4
    max_area = size * size * 3 // 5
    target_half_lo = max(5, size // 2 + 2)
    target_half_hi = min(size * half_w - 1, size + 4)

    for _ in range(200):
        target_half = int(RNG.integers(target_half_lo, target_half_hi + 1))
        half = np.zeros((size, half_w), dtype=bool)
        seed = (int(RNG.integers(size)), half_w - 1)
        half[seed] = True
        cells = [seed]

        while int(half.sum()) < target_half:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for rr, cc in _neighbors4(r, c, size, half_w):
                    if not half[rr, cc]:
                        frontier.add((rr, cc))
            if not frontier:
                break
            rr, cc = list(frontier)[int(RNG.integers(len(frontier)))]
            half[rr, cc] = True
            cells.append((rr, cc))

        full = _mirror_mask_left_right(half, size)
        area = int(full.sum())
        left_pairs = int(np.sum(full[:, : size // 2]))
        rows_used = np.any(full, axis=1).sum()
        if area < min_area or area > max_area:
            continue
        if left_pairs < 5:
            continue
        if rows_used < max(4, size - 2):
            continue
        if not _is_connected(full):
            continue
        return full

    raise RuntimeError("failed to generate a symmetric quadrant mask")


def _side_coords(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        mc = w - 1 - cc
        if side == "left" and cc < mc:
            coords.append((rr, cc))
        if side == "right" and cc > mc:
            coords.append((rr, cc))
    return coords


def _sample_coords(coords: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    idx = RNG.permutation(len(coords))[:n]
    return [coords[int(i)] for i in idx]


def _choose_missing(mask: np.ndarray, above: np.ndarray, below: np.ndarray) -> list[tuple[int, int]] | None:
    best_visible: list[tuple[int, int]] | None = None
    best_reveal: list[tuple[int, int]] | None = None
    best_score: tuple[int, int, float] | None = None
    visible_mask = mask & (~above)

    for side in ("left", "right"):
        visible = [(r, c) for r, c in _side_coords(visible_mask, side)]
        if len(visible) < 2:
            continue
        reveal = [(r, c) for r, c in visible if below[r, c]]
        score = (len(reveal), len(visible), float(RNG.random()))
        if best_score is None or score > best_score:
            best_score = score
            best_visible = visible
            best_reveal = reveal

    if best_visible is None or best_reveal is None:
        return None

    max_missing = min(4, len(best_visible))
    n_missing = int(RNG.integers(2, max_missing + 1))
    chosen: list[tuple[int, int]] = []

    if best_reveal:
        must_take = min(len(best_reveal), max(1, min(2, n_missing - 1)))
        chosen.extend(_sample_coords(best_reveal, must_take))

    remaining_pool = [coord for coord in best_visible if coord not in chosen]
    need = n_missing - len(chosen)
    if need > 0:
        if len(remaining_pool) < need:
            return None
        chosen.extend(_sample_coords(remaining_pool, need))

    return chosen


def _render_input(size: int, quadrant_masks: dict[str, np.ndarray]) -> np.ndarray:
    grid = np.zeros((size * 2, size * 2), dtype=int)
    for name, mask in quadrant_masks.items():
        region = np.zeros((size, size), dtype=int)
        region[mask] = QUADRANT_COLORS[name]
        if name == "tl":
            grid[:size, :size] = region
        elif name == "tr":
            grid[:size, size:] = region
        elif name == "bl":
            grid[size:, :size] = region
        else:
            grid[size:, size:] = region
    return grid


def _naive_opaque_completion(inp: np.ndarray) -> np.ndarray:
    return _overlay_quadrants(_split_quadrants(inp), transparent_recovered=False)


def _gen_attempt() -> np.ndarray | None:
    size = int(RNG.choice([7, 8]))
    full_masks = {name: _generate_symmetric_mask(size) for name in QUADRANT_COLORS}

    missing_coords: dict[str, list[tuple[int, int]]] = {}
    for idx, name in enumerate(BACK_TO_FRONT):
        above = np.zeros((size, size), dtype=bool)
        below = np.zeros((size, size), dtype=bool)
        for other in BACK_TO_FRONT[idx + 1 :]:
            above |= full_masks[other]
        for other in BACK_TO_FRONT[:idx]:
            below |= full_masks[other]
        chosen = _choose_missing(full_masks[name], above, below)
        if chosen is None:
            return None
        missing_coords[name] = chosen

    input_masks = {name: mask.copy() for name, mask in full_masks.items()}
    for name, coords in missing_coords.items():
        for r, c in coords:
            input_masks[name][r, c] = False

    inp = _render_input(size, input_masks)
    out = main(inp)
    naive = _naive_opaque_completion(inp)

    if np.count_nonzero(out != naive) < 8:
        return None
    if len({int(v) for v in np.unique(out) if int(v) != BG}) < 3:
        return None
    if int(np.count_nonzero(out)) < size * 2:
        return None
    if int(np.count_nonzero(out)) > size * size * 3 // 4:
        return None

    for name, region in _split_quadrants(inp).items():
        color_set = {int(v) for v in np.unique(region) if int(v) != BG}
        if color_set != {QUADRANT_COLORS[name]}:
            return None
        restored, recovered = _recover_quadrant(region)
        if int(recovered.sum()) < 2 or int(recovered.sum()) > 4:
            return None
        if np.array_equal(restored, region):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid transparent-symmetry overlay input")
