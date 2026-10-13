from __future__ import annotations

import math
from typing import Iterable

import numpy as np


HOLE = 9
BG = 0
PALETTE = (1, 2, 3)
TILE_HEIGHTS = (3, 4)
TILE_WIDTHS = (4, 6)
ROW_MULTIPLIERS = (3, 4)
COL_MULTIPLIERS = (2, 3)
MAX_GENERATE_ATTEMPTS = 400


def _valid_period(grid: np.ndarray, tile_h: int, tile_w: int, hole: int = HOLE) -> bool:
    seen: dict[tuple[int, int], int] = {}
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            value = int(grid[r, c])
            if value == hole:
                continue
            key = (r % tile_h, c % tile_w)
            prev = seen.get(key)
            if prev is None:
                seen[key] = value
            elif prev != value:
                return False
    return True


def _period_candidates(grid: np.ndarray, hole: int = HOLE) -> list[tuple[int, int]]:
    height, width = grid.shape
    best_area: int | None = None
    candidates: list[tuple[int, int]] = []
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            area = tile_h * tile_w
            if best_area is not None and area > best_area:
                continue
            if _valid_period(grid, tile_h, tile_w, hole):
                if best_area is None:
                    best_area = area
                candidates.append((tile_h, tile_w))
    return sorted(candidates, key=lambda item: (item[0] * item[1], item[0], item[1]))


def _infer_period(grid: np.ndarray, hole: int = HOLE) -> tuple[int, int]:
    candidates = _period_candidates(grid, hole)
    if not candidates:
        raise ValueError("no compatible tile period")
    return candidates[0]


def _axis_candidates(grid: np.ndarray, hole: int = HOLE) -> list[int]:
    height, width = grid.shape
    scored: list[tuple[int, int]] = []
    center = width - 1
    for axis_sum in range(2 * width - 1):
        evidence = 0
        conflicts = 0
        for r in range(height):
            for c in range(width):
                mirror_c = axis_sum - c
                if not (0 <= mirror_c < width) or c >= mirror_c:
                    continue
                left = int(grid[r, c])
                right = int(grid[r, mirror_c])
                if left == hole or right == hole:
                    continue
                evidence += 1
                if left != right:
                    conflicts += 1
        if evidence > 0 and conflicts == 0:
            scored.append((evidence, axis_sum))
    if not scored:
        return []
    best_evidence = max(score for score, _axis_sum in scored)
    return sorted(
        [axis_sum for score, axis_sum in scored if score == best_evidence],
        key=lambda axis_sum: (abs(axis_sum - center), axis_sum),
    )


def _infer_axis(grid: np.ndarray, hole: int = HOLE) -> int:
    candidates = _axis_candidates(grid, hole)
    if not candidates:
        raise ValueError("no compatible vertical mirror axis")
    return candidates[0]


def _same_class_coords(
    shape: tuple[int, int], r: int, c: int, tile_h: int, tile_w: int
) -> Iterable[tuple[int, int]]:
    height, width = shape
    row0 = r % tile_h
    col0 = c % tile_w
    for rr in range(row0, height, tile_h):
        for cc in range(col0, width, tile_w):
            yield rr, cc


def _partial_tile_cell(shape: tuple[int, int], r: int, c: int, tile_h: int, tile_w: int) -> bool:
    height, width = shape
    r0 = (r // tile_h) * tile_h
    c0 = (c // tile_w) * tile_w
    return r0 + tile_h > height or c0 + tile_w > width


def _propagate(
    grid: np.ndarray,
    tile_h: int,
    tile_w: int,
    axis_sum: int,
    *,
    use_symmetry: bool,
    use_tiling: bool,
    max_rounds: int | None = None,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    height, width = out.shape
    rounds = 0
    while True:
        changed = False

        if use_symmetry:
            updates: list[tuple[int, int, int]] = []
            for rr, cc in np.argwhere(out == HOLE):
                r = int(rr)
                c = int(cc)
                mirror_c = axis_sum - c
                if 0 <= mirror_c < width and out[r, mirror_c] != HOLE:
                    updates.append((r, c, int(out[r, mirror_c])))
            for r, c, value in updates:
                if out[r, c] == HOLE:
                    out[r, c] = value
                    changed = True

        if use_tiling:
            updates = []
            for rr, cc in np.argwhere(out == HOLE):
                r = int(rr)
                c = int(cc)
                values = {
                    int(out[r2, c2])
                    for r2, c2 in _same_class_coords(out.shape, r, c, tile_h, tile_w)
                    if out[r2, c2] != HOLE
                }
                if len(values) == 1:
                    updates.append((r, c, next(iter(values))))
            for r, c, value in updates:
                if out[r, c] == HOLE:
                    out[r, c] = value
                    changed = True

        rounds += 1
        if not changed:
            break
        if max_rounds is not None and rounds >= max_rounds:
            break

    return out


def _solve_grid(
    grid: np.ndarray,
    *,
    use_symmetry: bool = True,
    use_tiling: bool = True,
    max_rounds: int | None = None,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    tile_h, tile_w = _infer_period(g)
    axis_sum = _infer_axis(g)
    out = _propagate(
        g,
        tile_h,
        tile_w,
        axis_sum,
        use_symmetry=use_symmetry,
        use_tiling=use_tiling,
        max_rounds=max_rounds,
    )
    for rr, cc in np.argwhere(out == HOLE):
        r = int(rr)
        c = int(cc)
        if _partial_tile_cell(out.shape, r, c, tile_h, tile_w):
            out[r, c] = BG
        else:
            out[r, c] = BG
    return out


def _min_row_period(tile: np.ndarray) -> int:
    height = tile.shape[0]
    for period in range(1, height + 1):
        if height % period != 0:
            continue
        if all(np.array_equal(tile[r], tile[r % period]) for r in range(height)):
            return period
    return height


def _min_col_period(tile: np.ndarray) -> int:
    width = tile.shape[1]
    for period in range(1, width + 1):
        if width % period != 0:
            continue
        if all(np.array_equal(tile[:, c], tile[:, c % period]) for c in range(width)):
            return period
    return width


def _column_orbits(tile_w: int, axis_mod: int) -> list[tuple[int, int]] | None:
    partner = [(axis_mod - c) % tile_w for c in range(tile_w)]
    seen: set[int] = set()
    orbits: list[tuple[int, int]] = []
    for c in range(tile_w):
        if c in seen:
            continue
        d = partner[c]
        if d == c or partner[d] != c:
            return None
        seen.add(c)
        seen.add(d)
        orbits.append((c, d))
    return orbits


def _build_tile(
    rng: np.random.Generator, tile_h: int, tile_w: int, axis_mod: int
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    orbits = _column_orbits(tile_w, axis_mod)
    if orbits is None:
        return None

    tile = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        nz_count = 1 if tile_w == 4 else int(rng.choice([1, 2]))
        chosen = set(map(int, np.atleast_1d(rng.choice(len(orbits), size=nz_count, replace=False))))
        for idx, (c0, c1) in enumerate(orbits):
            color = int(rng.choice(PALETTE)) if idx in chosen else BG
            tile[r, c0] = color
            tile[r, c1] = color

    if set(map(int, np.unique(tile))) != {0, 1, 2, 3}:
        return None
    density = float(np.count_nonzero(tile)) / float(tile.size)
    if not (0.33 <= density <= 0.67):
        return None
    if _min_row_period(tile) != tile_h or _min_col_period(tile) != tile_w:
        return None
    return tile, orbits


def _render_periodic(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = math.ceil(height / tile.shape[0])
    reps_c = math.ceil(width / tile.shape[1])
    return np.tile(tile, (reps_r, reps_c))[:height, :width].copy()


def _class_candidates(
    tile: np.ndarray, height: int, width: int, orbits: list[tuple[int, int]]
) -> list[list[tuple[int, int, int]]]:
    tile_h, tile_w = tile.shape
    candidates: list[list[tuple[int, int, int]]] = []
    for row_mod in range(tile_h):
        for col_a, _col_b in orbits:
            if int(tile[row_mod, col_a]) == BG:
                continue
            pairs: list[tuple[int, int, int]] = []
            for r in range(row_mod, height, tile_h):
                for c in range(col_a, width, tile_w):
                    pairs.append((r, c, width - 1 - c))
            candidates.append(pairs)
    return candidates


def _classify_holes(
    grid: np.ndarray, tile_h: int, tile_w: int, axis_sum: int
) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0, "both": 0}
    height, width = grid.shape
    for rr, cc in np.argwhere(grid == HOLE):
        r = int(rr)
        c = int(cc)
        mirror_c = axis_sum - c
        mirror_known = 0 <= mirror_c < width and int(grid[r, mirror_c]) != HOLE
        tile_known = any(
            int(grid[r2, c2]) != HOLE
            for r2, c2 in _same_class_coords(grid.shape, r, c, tile_h, tile_w)
            if (r2, c2) != (r, c)
        )
        if mirror_known and not tile_known:
            counts["A"] += 1
        elif tile_known and not mirror_known:
            counts["B"] += 1
        elif not mirror_known and not tile_known:
            counts["C"] += 1
        else:
            counts["both"] += 1
    return counts


def _balanced_counts(counts: dict[str, int]) -> bool:
    abc = [counts["A"], counts["B"], counts["C"]]
    return min(abc) >= 3 and max(abc) <= 3 * min(abc)


def _validated_case(
    inp: np.ndarray, out: np.ndarray, tile_h: int, tile_w: int, axis_sum: int
) -> bool:
    nonzero = int(np.count_nonzero(out))
    hole_count = int(np.count_nonzero(inp == HOLE))
    if not (math.ceil(nonzero * 0.20) <= hole_count <= math.floor(nonzero * 0.30)):
        return False

    period_candidates = _period_candidates(inp)
    if period_candidates != [(tile_h, tile_w)]:
        return False

    axis_candidates = _axis_candidates(inp)
    if axis_candidates != [axis_sum]:
        return False

    counts = _classify_holes(inp, tile_h, tile_w, axis_sum)
    if counts["both"] != 0 or not _balanced_counts(counts):
        return False

    combined = _propagate(
        inp,
        tile_h,
        tile_w,
        axis_sum,
        use_symmetry=True,
        use_tiling=True,
    )
    if np.any(combined == HOLE) or not np.array_equal(combined, out):
        return False

    symmetry_only = _propagate(
        inp,
        tile_h,
        tile_w,
        axis_sum,
        use_symmetry=True,
        use_tiling=False,
    )
    if not np.any(symmetry_only == HOLE):
        return False

    tiling_only = _propagate(
        inp,
        tile_h,
        tile_w,
        axis_sum,
        use_symmetry=False,
        use_tiling=True,
    )
    if not np.any(tiling_only == HOLE):
        return False

    return True


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _attempt in range(MAX_GENERATE_ATTEMPTS):
        tile_h = int(rng.choice(TILE_HEIGHTS))
        tile_w = int(rng.choice(TILE_WIDTHS))

        base_rows = int(rng.choice(ROW_MULTIPLIERS)) * tile_h
        height = int(rng.integers(base_rows, min(16, base_rows + tile_h - 1) + 1))

        base_cols = int(rng.choice(COL_MULTIPLIERS)) * tile_w
        width_options = [
            w
            for w in range(base_cols, min(18, base_cols + tile_w - 1) + 1)
            if ((w - 1) % tile_w) % 2 == 1
        ]
        if not width_options:
            continue
        width = int(rng.choice(width_options))
        axis_sum = width - 1
        axis_mod = axis_sum % tile_w

        built = _build_tile(rng, tile_h, tile_w, axis_mod)
        if built is None:
            continue
        tile, orbits = built

        out = _render_periodic(tile, height, width)
        if not np.array_equal(out, out[:, ::-1]):
            continue
        if _axis_candidates(out) != [axis_sum]:
            continue

        nonzero = int(np.count_nonzero(out))
        target_lo = math.ceil(nonzero * 0.20)
        target_hi = math.floor(nonzero * 0.30)
        candidates = _class_candidates(tile, height, width, orbits)
        rng.shuffle(candidates)

        for pairs in candidates:
            n_pairs = len(pairs)
            valid_ks = [k for k in range(1, n_pairs) if target_lo <= n_pairs + k <= target_hi]
            if not valid_ks:
                continue
            k = min(valid_ks, key=lambda value: (abs(value - n_pairs / 2.0), -value))

            first_hide_primary = bool(rng.integers(2))
            for hide_primary in (first_hide_primary, not first_hide_primary):
                inp = out.copy()
                hidden_secondary = set(
                    map(int, np.atleast_1d(rng.choice(n_pairs, size=k, replace=False)))
                )
                for idx, (r, primary_c, mirror_c) in enumerate(pairs):
                    if hide_primary:
                        inp[r, primary_c] = HOLE
                        if idx in hidden_secondary:
                            inp[r, mirror_c] = HOLE
                    else:
                        inp[r, mirror_c] = HOLE
                        if idx in hidden_secondary:
                            inp[r, primary_c] = HOLE

                if _validated_case(inp, out, tile_h, tile_w, axis_sum):
                    return inp, out

    raise RuntimeError("failed to generate a valid puzzle instance")


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the maroon-occluded periodic picture by jointly using vertical symmetry and tile repetition.

    Input structure:
    - The grid has black background plus a periodically tiled foreground made from colors 1, 2, and 3.
    - Some foreground pixels are replaced by maroon 9, while the remaining visible pixels still agree with one global tile period.
    - The complete hidden picture is vertically mirror-symmetric, but many mirror pairs and many tile-mate sets are both partially occluded.

    Rule:
    - Treat color 9 as an occlusion hole.
    - Infer the smallest rectangular tile period consistent with every non-9 pixel, and infer the vertical mirror axis that has zero visible conflicts and the strongest evidence.
    - Repeatedly do two passes: first copy colors from visible mirror partners into 9-cells, then copy colors from visible same-phase tile replicas into any still-hidden 9-cells, until no more cells can be resolved.
    - If any 9-cells somehow remain after propagation, erase them to black.

    Key insight:
    - Neither symmetry nor periodic tiling is sufficient on its own: some holes can be solved only after the other constraint first resolves their mirror or tile counterpart, so the repair must alternate the two constraints until they stabilize.
    """
    return _solve_grid(np.asarray(grid, dtype=int), use_symmetry=True, use_tiling=True)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    inp, _out = _generate_case(rng)
    return inp
