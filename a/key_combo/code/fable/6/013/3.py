from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SEPARATOR = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 600
SELF_CHECK_TRIALS = 12


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [r for r in range(grid.shape[0]) if int(grid[r, 0]) != int(bg) and np.all(grid[r, :] == grid[r, 0])]
    assert len(rows) == 1, "expected a unique separator row"
    return int(rows[0])


def _valid_period(grid: np.ndarray, visible: np.ndarray, pr: int, pc: int) -> bool:
    seen: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(visible):
        key = (int(r) % int(pr), int(c) % int(pc))
        val = int(grid[r, c])
        old = seen.get(key)
        if old is None:
            seen[key] = val
        elif old != val:
            return False
    return True


def _infer_period(grid: np.ndarray, bg: int) -> tuple[int, int]:
    visible = grid != int(bg)
    h, w = grid.shape
    best: tuple[tuple[int, int, int, int], tuple[int, int]] | None = None
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            if not _valid_period(grid, visible, pr, pc):
                continue
            score = (int(pr * pc), int(pr + pc), int(pr), int(pc))
            if best is None or score < best[0]:
                best = (score, (int(pr), int(pc)))
    assert best is not None, "expected at least one valid period"
    return best[1]


def _known_hole_cells(
    top: np.ndarray,
    hole_box: tuple[int, int, int, int],
    bg: int,
    period: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = top.shape
    pr, pc = map(int, period)
    r0, c0, r1, c1 = hole_box
    hole_h = int(r1 - r0 + 1)
    hole_w = int(c1 - c0 + 1)
    known = np.zeros((hole_h, hole_w), dtype=bool)
    partial = np.full((hole_h, hole_w), -1, dtype=int)

    for rr in range(hole_h):
        for cc in range(hole_w):
            src_r = int(r0 + rr)
            src_c = int(c0 + cc)
            colors = set()
            for r in range(src_r % pr, h, pr):
                for c in range(src_c % pc, w, pc):
                    if r0 <= r <= r1 and c0 <= c <= c1:
                        continue
                    colors.add(int(top[r, c]))
            colors.discard(int(bg))
            if colors:
                assert len(colors) == 1, "period-implied cells must agree"
                partial[rr, cc] = next(iter(colors))
                known[rr, cc] = True
    return known, partial


def _candidate_patches(bottom: np.ndarray, bg: int) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for coords in _components(bottom != int(bg)):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = bottom[r0 : r1 + 1, c0 : c1 + 1].copy()
        if int(coords.shape[0]) != int(patch.shape[0] * patch.shape[1]):
            continue
        if np.any(patch == int(bg)):
            continue
        patches.append(patch)
    return patches


def _matching_rotations(patch: np.ndarray, known: np.ndarray, partial: np.ndarray) -> list[np.ndarray]:
    matches: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(patch, k)
        if rot.shape != partial.shape:
            continue
        if np.array_equal(rot[known], partial[known]):
            matches.append(np.array(rot, dtype=int))
    return matches


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Repair a periodic top pattern by selecting the unique rotated candidate patch.

    Input structure:
    - The upper region is a periodic non-background tiling with one rectangular hole cut to background.
    - A full separator row splits off a lower candidate area containing several solid rectangular patches.

    Rule:
    - Infer the smallest 2D period from visible top cells.
    - Use that period to determine every hole cell whose color is still visible elsewhere in the same residue class.
    - Find the unique candidate patch whose 0/90/180/270 rotation agrees with all determined cells.
    - Fill the hole with that rotated patch and return only the repaired top region.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg = _get_bg(grid)
    sep_r = _find_separator_row(grid, bg)
    top = grid[:sep_r, :].copy()
    bottom = grid[sep_r + 1 :, :]

    hole_box = _bbox(top == int(bg))
    assert hole_box is not None, "expected a rectangular hole in the top region"
    period = _infer_period(top, bg)
    known, partial = _known_hole_cells(top, hole_box, bg, period)
    assert bool(known.any()), "expected at least one period-determined hole cell"

    matches: list[np.ndarray] = []
    for patch in _candidate_patches(bottom, bg):
        matches.extend(_matching_rotations(patch, known, partial))
    assert len(matches) == 1, "expected a unique candidate patch and rotation"

    r0, c0, r1, c1 = hole_box
    top[r0 : r1 + 1, c0 : c1 + 1] = matches[0]
    return top


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    tr, tc = tile.shape
    reps_r = (int(h) + int(tr) - 1) // int(tr)
    reps_c = (int(w) + int(tc) - 1) // int(tc)
    return np.tile(tile, (reps_r, reps_c))[:h, :w].copy()


def _sample_tile(pr: int, pc: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(300):
        n_colors = rng.randint(3, min(6, len(PALETTE)))
        colors = rng.sample(PALETTE, n_colors)
        tile = np.array([[rng.choice(colors) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if len(set(map(int, tile.flat))) < 3:
            continue
        if _infer_period(tile, -1) != (int(pr), int(pc)):
            continue
        return tile
    return None


def _constant_non_bg_rows(grid: np.ndarray, bg: int) -> list[int]:
    return [r for r in range(grid.shape[0]) if int(grid[r, 0]) != int(bg) and np.all(grid[r, :] == grid[r, 0])]


def _random_distractor(
    shape: tuple[int, int],
    known: np.ndarray,
    partial: np.ndarray,
    truth: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    h, w = shape
    for _ in range(200):
        n_colors = rng.randint(2, min(5, len(PALETTE)))
        colors = rng.sample(PALETTE, n_colors)
        patch = np.array([[rng.choice(colors) for _ in range(w)] for _ in range(h)], dtype=int)
        if np.array_equal(patch, truth):
            continue
        if _matching_rotations(patch, known, partial):
            continue
        return patch
    return None


def _build_case(
    top: np.ndarray,
    hole_box: tuple[int, int, int, int],
    shown_patch: np.ndarray,
    distractors: list[np.ndarray],
    rng: random.Random,
) -> np.ndarray:
    h, w = top.shape
    hh, ww = shown_patch.shape
    bottom_patches = list(distractors) + [shown_patch]
    rng.shuffle(bottom_patches)

    broken = top.copy()
    r0, c0, r1, c1 = hole_box
    broken[r0 : r1 + 1, c0 : c1 + 1] = BG

    bottom_h = len(bottom_patches) * hh + len(bottom_patches) + 1
    bottom = np.full((bottom_h, w), BG, dtype=int)
    for idx, patch in enumerate(bottom_patches):
        rr = 1 + idx * (hh + 1)
        cc = rng.randint(0, w - ww)
        bottom[rr : rr + hh, cc : cc + ww] = patch

    return np.vstack([broken, np.full((1, w), SEPARATOR, dtype=int), bottom])


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    pr = rng.randint(3, 6)
    pc = rng.randint(3, 6)
    top_h = rng.randint(pr + 1, min(2 * pr - 1, 12))
    top_w = rng.randint(pc + 1, min(2 * pc - 1, 12))

    tile = _sample_tile(pr, pc, rng)
    if tile is None:
        return None
    top = _repeat_tile(tile, (top_h, top_w))

    for _ in range(300):
        hole_h = rng.randint(2, min(4, top_h - 1))
        hole_w = rng.randint(2, min(4, top_w - 1))
        r0 = rng.randint(0, top_h - hole_h)
        c0 = rng.randint(0, top_w - hole_w)
        if hole_h == top_h or hole_w == top_w:
            continue

        broken = top.copy()
        broken[r0 : r0 + hole_h, c0 : c0 + hole_w] = BG
        if _constant_non_bg_rows(broken, BG):
            continue
        if _bbox(broken == BG) != (r0, c0, r0 + hole_h - 1, c0 + hole_w - 1):
            continue
        if _infer_period(broken, BG) != (pr, pc):
            continue

        hole_box = (r0, c0, r0 + hole_h - 1, c0 + hole_w - 1)
        known, partial = _known_hole_cells(broken, hole_box, BG, (pr, pc))
        known_count = int(known.sum())
        if not (0 < known_count < hole_h * hole_w):
            continue

        truth = top[r0 : r0 + hole_h, c0 : c0 + hole_w].copy()
        allowed_show_rots = [0, 1, 2, 3] if hole_h == hole_w else [0, 2]
        shown_options = []
        for show_k in allowed_show_rots:
            shown = np.rot90(truth, show_k)
            if shown.shape != truth.shape:
                continue
            if len(_matching_rotations(shown, known, partial)) == 1:
                shown_options.append(np.array(shown, dtype=int))
        if not shown_options:
            continue
        shown_patch = rng.choice(shown_options)

        distractors: list[np.ndarray] = []
        while len(distractors) < 2:
            distractor = _random_distractor((hole_h, hole_w), known, partial, truth, rng)
            if distractor is None:
                break
            distractors.append(distractor)
        if len(distractors) != 2:
            continue

        grid = _build_case(top, hole_box, shown_patch, distractors, rng)
        if grid.shape[0] > 30 or grid.shape[1] > 30:
            continue
        if _constant_non_bg_rows(grid, BG) != [top_h]:
            continue

        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, top):
            continue
        return grid, top
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")


def _self_check(trials: int = SELF_CHECK_TRIALS) -> None:
    rng = random.Random(0)
    for _ in range(int(trials)):
        case = None
        for _attempt in range(MAX_GENERATION_ATTEMPTS):
            case = _generate_case(rng)
            if case is not None:
                break
        assert case is not None, "generator failed during self-check"
        inp, expected = case
        out = main(inp)
        assert isinstance(out, np.ndarray), "main must return an ndarray"
        assert out.ndim == 2, "output must be 2D"
        assert out.shape == expected.shape, "output shape must match repaired top"
        assert out.shape[0] <= 30 and out.shape[1] <= 30, "output must stay within ARC limits"
        assert np.array_equal(out, expected), "solver output must equal the repaired top area"


if __name__ == "__main__":
    _self_check()
