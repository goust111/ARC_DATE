import random
from typing import Iterable

import numpy as np


BLACK = 0
MASK = 6
PATTERN_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)
GEN_ATTEMPTS = 400


# Small helpers keep arc_utils' canonical (r, c) / inclusive-bbox conventions.
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


def _visible_pattern_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != MASK)


def _candidate_ps(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    vis = _visible_pattern_mask(g)
    candidates: list[int] = []

    for p in range(1, w):
        coarse_shift = 2 * p
        if coarse_shift >= w:
            continue

        coarse_pairs = 0
        ok = True
        for c in range(w - coarse_shift):
            left = vis[:, c]
            right = vis[:, c + coarse_shift]
            both = left & right
            if np.any(g[:, c][both] != g[:, c + coarse_shift][both]):
                ok = False
                break
            coarse_pairs += int(both.sum())
        if not ok or coarse_pairs == 0:
            continue

        glide_pairs = 0
        for c in range(w - p):
            both = vis[:, c] & vis[::-1, c + p]
            if np.any(g[:, c][both] != g[::-1, c + p][both]):
                ok = False
                break
            glide_pairs += int(both.sum())
        if ok and glide_pairs > 0:
            candidates.append(p)

    return candidates


def _infer_glide_p(grid: np.ndarray) -> int:
    candidates = _candidate_ps(grid)
    if len(candidates) != 1:
        raise ValueError(f"expected a unique glide half-period, got {candidates}")
    return int(candidates[0])


def _orbit_coords(shape: tuple[int, int], r: int, c: int, p: int) -> list[tuple[int, int]]:
    h, w = shape
    period = 2 * p
    residue = int(c) % period
    glide_residue = (residue + p) % period
    mate_row = h - 1 - int(r)
    coords: list[tuple[int, int]] = []

    for cc in range(w):
        if cc % period == residue:
            coords.append((int(r), cc))
        elif cc % period == glide_residue:
            coords.append((mate_row, cc))
    return coords


def _orbit_visible_colors(grid: np.ndarray, r: int, c: int, p: int) -> set[int]:
    return {
        int(grid[rr, cc])
        for rr, cc in _orbit_coords(grid.shape, r, c, p)
        if grid[rr, cc] not in (BLACK, MASK)
    }


def _repair_damage(grid: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    mask = g == MASK
    if not mask.any():
        raise ValueError("input must contain masked cells")

    repaired = g.copy()
    damage_only = np.zeros_like(g)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        colors = _orbit_visible_colors(g, rr, cc, p)
        if len(colors) != 1:
            raise ValueError(f"masked orbit {(rr, cc)} is not uniquely determined: {colors}")
        color = colors.pop()
        repaired[rr, cc] = color
        damage_only[rr, cc] = color
    return repaired, damage_only


def _glide_only_mask(mask: np.ndarray, shape: tuple[int, int], p: int, r: int, c: int) -> bool:
    h, _ = shape
    same_row_has_visible = False
    glide_has_visible = False
    target_row = int(r)
    target_col = int(c)
    mate_row = h - 1 - target_row

    for rr, cc in _orbit_coords(shape, target_row, target_col, p):
        if rr == target_row and cc == target_col:
            continue
        if rr == target_row:
            if not mask[rr, cc]:
                same_row_has_visible = True
        elif rr == mate_row:
            if not mask[rr, cc]:
                glide_has_visible = True

    return (not same_row_has_visible) and glide_has_visible


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the masked cells from a horizontally repeating glide-symmetric frieze and return only the damaged area.

    Input structure:
    - The canvas is filled by a non-black decorative band pattern; pink is a mask color and black is not part of the pattern.
    - The completed pattern is invariant under a horizontal shift by p together with a top-bottom flip, and therefore under a pure horizontal shift by 2p.
    - Some cells are masked, and some same-row 2p companions of a masked cell may be masked as well.

    Rule:
    - Infer the unique half-period p whose visible cells satisfy both the coarse 2p translation and the finer glide relation.
    - For each masked cell, collect its glide orbit: same-row copies spaced by 2p and flipped-row copies shifted by p.
    - Recolor every masked position with the unique visible orbit color, then crop to the bounding box of all masked cells and keep only the repaired masked pixels there, with every other output cell black.

    Key insight:
    - The apparent 2p translation is a decoy: some damages hide an entire same-row translation class, so the missing color becomes visible only in the shifted-and-flipped half-cycle.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    p = _infer_glide_p(g)
    _, damage_only = _repair_damage(g, p)
    box = _bbox(g == MASK)
    if box is None:
        raise ValueError("input must contain masked cells")
    return _crop_bbox(damage_only, box)


def _paint_rect(arr: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    arr[r0 : r0 + h, c0 : c0 + w] = int(color)


def _make_left_half(height: int, p: int) -> np.ndarray:
    for _ in range(120):
        palette = random.sample(PATTERN_COLORS, random.randint(3, min(5, len(PATTERN_COLORS))))
        left = np.full((height, p), random.choice(palette), dtype=int)

        feature_count = random.randint(4, max(5, height + p))
        for _ in range(feature_count):
            color = random.choice(palette)
            mode = random.choice(("rect", "hline", "vline", "cell"))
            if mode == "rect":
                fh = random.randint(1, height)
                fw = random.randint(1, p)
                r0 = random.randint(0, height - fh)
                c0 = random.randint(0, p - fw)
                _paint_rect(left, r0, c0, fh, fw, color)
            elif mode == "hline":
                r0 = random.randrange(height)
                c0 = random.randrange(p)
                fw = random.randint(1, p - c0)
                left[r0, c0 : c0 + fw] = color
            elif mode == "vline":
                c0 = random.randrange(p)
                r0 = random.randrange(height)
                fh = random.randint(1, height - r0)
                left[r0 : r0 + fh, c0] = color
            else:
                left[random.randrange(height), random.randrange(p)] = color

        if len(set(map(int, left.flatten()))) < 3:
            continue
        if np.array_equal(left, left[::-1, :]):
            continue
        return left
    raise RuntimeError("failed to build a non-degenerate half motif")


def _build_full_pattern(height: int, p: int, repeats: int) -> np.ndarray:
    for _ in range(120):
        left = _make_left_half(height, p)
        tile = np.concatenate([left, left[::-1, :]], axis=1)
        full = np.tile(tile, (1, repeats))
        if np.all(full[:, :-p] == full[:, p:]):
            continue
        try:
            inferred = _infer_glide_p(full)
        except ValueError:
            continue
        if inferred == p:
            return full
    raise RuntimeError("failed to build a unique-period glide pattern")


def _place_glide_only_damage(mask: np.ndarray, p: int, repeats: int) -> None:
    h, _ = mask.shape
    for _ in range(120):
        dh = random.randint(1, min(3, h))
        dw = random.randint(1, max(1, min(3, p)))
        r0 = random.randint(0, h - dh)
        c0 = random.randint(0, p - dw)

        trial = mask.copy()
        for k in range(repeats):
            cc = c0 + 2 * p * k
            trial[r0 : r0 + dh, cc : cc + dw] = True

        found = False
        for rr in range(r0, r0 + dh):
            for cc in range(c0, c0 + dw):
                if _glide_only_mask(trial, trial.shape, p, rr, cc):
                    found = True
                    break
            if found:
                break
        if found:
            mask[:] = trial
            return
    raise RuntimeError("failed to place glide-only damage")


def _add_regular_damage(mask: np.ndarray, p: int) -> None:
    h, w = mask.shape
    extra_count = random.randint(0, 3)
    for _ in range(extra_count):
        for _ in range(80):
            dh = random.randint(1, min(3, h))
            dw = random.randint(1, min(3, w))
            r0 = random.randint(0, h - dh)
            c0 = random.randint(0, w - dw)
            trial = mask.copy()
            trial[r0 : r0 + dh, c0 : c0 + dw] = True
            if trial.mean() <= 0.32:
                mask[:] = trial
                break

    if not np.any(
        [
            any(
                (rr == r) and (cc != c) and (not mask[rr, cc])
                for rr, cc in _orbit_coords(mask.shape, r, c, p)
            )
            for r, c in np.argwhere(mask)
        ]
    ):
        for _ in range(120):
            r = random.randrange(h)
            c = random.randrange(w)
            if mask[r, c]:
                continue
            same_row_mates = [
                (rr, cc)
                for rr, cc in _orbit_coords(mask.shape, r, c, p)
                if rr == r and cc != c
            ]
            if same_row_mates and any(not mask[rr, cc] for rr, cc in same_row_mates):
                mask[r, c] = True
                return


def _damage_output_from_solution(solution: np.ndarray, mask: np.ndarray) -> np.ndarray:
    damage_only = np.zeros_like(solution)
    damage_only[mask] = solution[mask]
    box = _bbox(mask)
    if box is None:
        raise ValueError("mask cannot be empty")
    return _crop_bbox(damage_only, box)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, int] | None:
    repeats = random.randint(2, 3)
    max_p = 30 // (2 * repeats)
    if max_p < 2:
        return None
    p = random.randint(2, max_p)
    width = 2 * p * repeats
    height = random.randint(5, min(12, 30))
    if width > 30 or height > 30:
        return None

    solution = _build_full_pattern(height, p, repeats)
    mask = np.zeros_like(solution, dtype=bool)
    _place_glide_only_damage(mask, p, repeats)
    _add_regular_damage(mask, p)
    if not mask.any():
        return None
    if mask.mean() > 0.35:
        return None

    inp = solution.copy()
    inp[mask] = MASK

    try:
        inferred = _infer_glide_p(inp)
        repaired, _ = _repair_damage(inp, inferred)
    except ValueError:
        return None
    if inferred != p or not np.array_equal(repaired, solution):
        return None

    if not any(_glide_only_mask(mask, inp.shape, p, int(r), int(c)) for r, c in np.argwhere(mask)):
        return None
    if not any(
        any((rr == r) and (cc != c) and (not mask[rr, cc]) for rr, cc in _orbit_coords(mask.shape, int(r), int(c), p))
        for r, c in np.argwhere(mask)
    ):
        return None

    expected = _damage_output_from_solution(solution, mask)
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected, p


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid glide-symmetry damage input")


if __name__ == "__main__":
    for _ in range(3):
        grid = generate_input()
        print(grid.shape, main(grid).shape)
