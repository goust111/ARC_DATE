import numpy as np


BG_COLOR = 0
DIRECT_COLOR = 2
MIRROR_COLOR = 8
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 1000
MASK_ATTEMPTS = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a two-color asymmetric sprite into a monochrome mirrored fractal.

    Input structure:
    - The input is a square black-background grid.
    - Foreground cells use exactly two colors; in this instantiation RED requests the original orientation and TEAL requests a left-right mirror.
    - The combined foreground silhouette is left-right asymmetric.

    Rule:
    - Let n be the input side length and partition the output into n x n blocks, each of size n x n.
    - Every RED host cell emits a RED monochrome copy of the full input foreground silhouette into its block.
    - Every TEAL host cell emits a TEAL monochrome copy of the silhouette after a left-right mirror; black host cells emit all-black blocks.

    Key insight:
    - The input pixels are control cells rather than final content: each occupied cell projects the whole sprite, and its visible color simultaneously chooses the copy color and orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "expected a square control grid"
    assert h * w <= 25, "the fractal expansion must stay within ARC size bounds"

    fg_colors = {int(color) for color in np.unique(g) if int(color) != BG_COLOR}
    assert fg_colors == {DIRECT_COLOR, MIRROR_COLOR}, "expected the generated red/teal color pair"

    mask = g != BG_COLOR
    assert bool(mask.any()), "expected foreground cells"
    assert not np.array_equal(mask, np.fliplr(mask)), "the foreground silhouette must be left-right asymmetric"

    n = h
    direct_mask = mask
    mirror_mask = np.fliplr(mask)
    out = np.zeros((n * n, n * n), dtype=int)

    for r, c in np.argwhere(g == DIRECT_COLOR):
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        block[direct_mask] = DIRECT_COLOR

    for r, c in np.argwhere(g == MIRROR_COLOR):
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        block[mirror_mask] = MIRROR_COLOR

    return out


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(int(v) for v in coords[0])
    stack = [start]
    seen = {start}
    h, w = mask.shape

    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if not mask[rr, cc]:
                continue
            nxt = (rr, cc)
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)

    return len(seen) == int(mask.sum())


def _covers_all_rows_and_cols(mask: np.ndarray) -> bool:
    return bool(np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)))


def _sample_mask(n: int, rng: np.random.Generator) -> np.ndarray:
    min_area = max(4, 2 * n - 2)
    max_area = n * n - n

    for _ in range(MASK_ATTEMPTS):
        target_area = int(rng.integers(min_area, max_area + 1))
        mask = np.zeros((n, n), dtype=bool)
        start = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        mask[start] = True

        while int(mask.sum()) < max_area and (int(mask.sum()) < target_area or not _covers_all_rows_and_cols(mask)):
            rows_covered = mask.any(axis=1)
            cols_covered = mask.any(axis=0)
            candidate_scores: dict[tuple[int, int], int] = {}

            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    rr, cc = int(r + dr), int(c + dc)
                    if not (0 <= rr < n and 0 <= cc < n):
                        continue
                    if mask[rr, cc]:
                        continue

                    score = 0
                    if not rows_covered[rr]:
                        score += 3
                    if not cols_covered[cc]:
                        score += 3
                    if rr in (0, n - 1):
                        score += 1
                    if cc in (0, n - 1):
                        score += 1
                    candidate_scores[(rr, cc)] = max(candidate_scores.get((rr, cc), -1), score)

            if not candidate_scores:
                break

            best_score = max(candidate_scores.values())
            best_cells = [cell for cell, score in candidate_scores.items() if score == best_score]
            rr, cc = best_cells[int(rng.integers(0, len(best_cells)))]
            mask[rr, cc] = True

        if int(mask.sum()) < min_area:
            continue
        if not _covers_all_rows_and_cols(mask):
            continue
        if not _is_connected4(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask

    raise RuntimeError("failed to sample an asymmetric connected sprite")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_ATTEMPTS):
        n = int(rng.integers(3, 6))
        mask = _sample_mask(n, rng)
        coords = np.argwhere(mask)
        area = len(coords)
        min_color_count = max(2, area // 4)
        direct_count = int(rng.integers(min_color_count, area - min_color_count + 1))

        grid = np.zeros((n, n), dtype=int)
        rng.shuffle(coords)
        direct_coords = coords[:direct_count]
        mirror_coords = coords[direct_count:]

        if len(mirror_coords) < 2:
            continue

        grid[direct_coords[:, 0], direct_coords[:, 1]] = DIRECT_COLOR
        grid[mirror_coords[:, 0], mirror_coords[:, 1]] = MIRROR_COLOR

        out = main(grid)
        if max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid

    raise RuntimeError("failed to generate a valid input")
