from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLUE = 1
RED = 2
GEN_BG_CHOICES = (3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components_of_color(grid, RED):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4, "red components must be hollow rectangular frames"
        assert r0 > 0 and c0 > 0 and r1 < grid.shape[0] - 1 and c1 < grid.shape[1] - 1, "frames may not touch the grid edge"

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        frame_mask = sub == RED
        border = np.zeros_like(frame_mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(frame_mask, border), "red components must be one-cell-thick hollow rectangles"
        frames.append((r0, c0, r1, c1))

    frames.sort()
    assert 1 <= len(frames) <= 2, "expected 1 to 2 red frames"
    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            ar0, ac0, ar1, ac1 = a
            br0, bc0, br1, bc1 = b
            assert ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0, "frames may not overlap"
    return frames


def _erase_frame_borders(grid: np.ndarray, frames: list[tuple[int, int, int, int]], bg: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r0, c0, r1, c1 in frames:
        out[r0, c0 : c1 + 1] = int(bg)
        out[r1, c0 : c1 + 1] = int(bg)
        out[r0 : r1 + 1, c0] = int(bg)
        out[r0 : r1 + 1, c1] = int(bg)
    return out


def _translation_scores(mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = mask.shape
    scores: list[tuple[int, int, int]] = []
    for dr in range(2, h):
        for dc in range(2, w):
            score = int(np.count_nonzero(mask[:-dr, :-dc] & mask[dr:, dc:]))
            if score > 0:
                scores.append((score, dr, dc))
    scores.sort(key=lambda item: (-item[0], item[1] + item[2], max(item[1], item[2]), item[1], item[2]))
    return scores


def _detect_translation_vector(grid: np.ndarray, bg: int) -> tuple[int, int]:
    blue_mask = grid == BLUE
    scores = _translation_scores(blue_mask)
    assert scores, "expected a detectable diagonal translation vector"
    _best_score, dr, dc = scores[0]
    return int(dr), int(dc)


def _complete_pattern(grid: np.ndarray, bg: int, dr: int, dc: int) -> np.ndarray:
    out = np.full_like(grid, int(bg))
    blue_cells = np.argwhere(grid == BLUE)
    for r, c in blue_cells:
        rr = int(r)
        cc = int(c)
        while rr - dr >= 0 and cc - dc >= 0:
            rr -= dr
            cc -= dc
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = BLUE
            rr += dr
            cc += dc
    return out


def _apply_phase_shift(full_pattern: np.ndarray, frames: list[tuple[int, int, int, int]], bg: int, shift: int) -> np.ndarray:
    out = np.array(full_pattern, copy=True)
    for r0, c0, r1, c1 in frames:
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.roll(inner, shift=shift, axis=1)
        out[r0, c0 : c1 + 1] = int(bg)
        out[r1, c0 : c1 + 1] = int(bg)
        out[r0 : r1 + 1, c0] = int(bg)
        out[r0 : r1 + 1, c1] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a partial diagonal translation pattern, phase-shift it inside red frames, then erase the frames.

    Input structure:
    - The border reveals a non-black background color.
    - Blue pixels are visible samples of a diagonal translational pattern repeated by a positive vector `(pr, pc)` with `pr >= 2` and `pc >= 2`.
    - One or two red one-cell-thick hollow rectangles mark interior windows; the pattern underneath them should still be treated as present.

    Rule:
    - Infer the background from the border.
    - Detect the blue pattern's diagonal translation vector `(pr, pc)`.
    - Ignore the red borders, extend every visible blue orbit under that translation until the full grid is tiled.
    - For each frame, cyclically shift the strict interior region right by `pc` columns, then erase the red border to background.

    Key insight:
    - The red rectangles are not containers to fill; they are control windows that tell you where the completed translational pattern must jump to a different horizontal phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_background(g)
    frames = _detect_frames(g, bg)
    stripped = _erase_frame_borders(g, frames, bg)
    pr, pc = _detect_translation_vector(stripped, bg)
    full_pattern = _complete_pattern(stripped, bg, pr, pc)
    return _apply_phase_shift(full_pattern, frames, bg, pc)


def _fundamental_cells(h: int, w: int, dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w) if r < dr or c < dc]


def _paint_full_pattern(h: int, w: int, dr: int, dc: int, seeds: set[tuple[int, int]], bg: int) -> np.ndarray:
    full = np.full((h, w), int(bg), dtype=int)
    for r, c in seeds:
        rr = int(r)
        cc = int(c)
        while 0 <= rr < h and 0 <= cc < w:
            full[rr, cc] = BLUE
            rr += dr
            cc += dc
    return full


def _generate_full_pattern(h: int, w: int, dr: int, dc: int, bg: int) -> np.ndarray | None:
    fundamental = _fundamental_cells(h, w, dr, dc)
    for _ in range(120):
        seed_prob = random.uniform(0.22, 0.42)
        seeds = {cell for cell in fundamental if random.random() < seed_prob}
        if len(seeds) < 8:
            continue
        if not any(r < dr and c >= dc for r, c in seeds):
            continue
        if not any(r >= dr and c < dc for r, c in seeds):
            continue

        full = _paint_full_pattern(h, w, dr, dc, seeds, bg)
        density = float(np.mean(full == BLUE))
        if not (0.16 <= density <= 0.45):
            continue

        if _detect_translation_vector(full, bg) != (dr, dc):
            continue
        scores = _translation_scores(full == BLUE)
        if len(scores) > 1 and scores[0][0] <= scores[1][0]:
            continue
        return full
    return None


def _choose_hidden_rect(h: int, w: int) -> tuple[int, int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for hh in range(max(4, h // 3), h - 2):
        for ww in range(max(4, w // 3), w - 2):
            hidden_ratio = (hh * ww) / float(h * w)
            if 0.20 <= hidden_ratio <= 0.40:
                candidates.append((h - hh, w - ww, h - 1, w - 1))
    if candidates:
        return random.choice(candidates)
    return None


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _pick_frames(
    full: np.ndarray,
    dr: int,
    dc: int,
    hidden_rect: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]] | None:
    h, w = full.shape
    hr0, hc0, hr1, hc1 = hidden_rect
    frame_count = random.choice((1, 2))
    frames: list[tuple[int, int, int, int]] = []

    for _ in range(200):
        if len(frames) == frame_count:
            break
        fh = random.randint(4, min(8, h - dr - 2))
        fw = random.randint(max(4, dc + 2), min(8, w - dc - 2))
        r0_min = dr + 1
        c0_min = dc + 1
        r0_max = h - fh - 1
        c0_max = w - fw - 1
        if r0_min > r0_max or c0_min > c0_max:
            return None
        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if any(not _bbox_separated(bbox, other, gap=1) for other in frames):
            continue

        visible_mask = np.ones((fh, fw), dtype=bool)
        rr0 = max(r0, hr0)
        cc0 = max(c0, hc0)
        rr1 = min(r0 + fh - 1, hr1)
        cc1 = min(c0 + fw - 1, hc1)
        if rr0 <= rr1 and cc0 <= cc1:
            visible_mask[rr0 - r0 : rr1 - r0 + 1, cc0 - c0 : cc1 - c0 + 1] = False
        if float(np.mean(visible_mask)) < 0.35:
            continue

        inner = full[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1]
        if np.array_equal(inner, np.roll(inner, shift=dc, axis=1)):
            continue
        frames.append(bbox)

    if len(frames) != frame_count:
        return None
    frames.sort()
    return frames


def _overlay_frames(grid: np.ndarray, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r0, c0, r1, c1 in frames:
        out[r0, c0 : c1 + 1] = RED
        out[r1, c0 : c1 + 1] = RED
        out[r0 : r1 + 1, c0] = RED
        out[r0 : r1 + 1, c1] = RED
    return out


def _semantic_checks(
    inp: np.ndarray,
    full: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    bg: int,
    dr: int,
    dc: int,
    hidden_rect: tuple[int, int, int, int],
) -> bool:
    h, w = inp.shape
    hr0, hc0, hr1, hc1 = hidden_rect
    visible_ratio = 1.0 - ((hr1 - hr0 + 1) * (hc1 - hc0 + 1)) / float(h * w)
    if not (0.60 <= visible_ratio <= 0.80):
        return False
    if _border_background(inp) != bg:
        return False
    if _detect_frames(inp, bg) != frames:
        return False

    stripped = _erase_frame_borders(inp, frames, bg)
    if _detect_translation_vector(stripped, bg) != (dr, dc):
        return False
    scores = _translation_scores(stripped == BLUE)
    if len(scores) > 1 and scores[0][0] <= scores[1][0]:
        return False

    expected = _apply_phase_shift(full, frames, bg, dc)
    if not np.array_equal(main(inp), expected):
        return False

    erased_only = _erase_frame_borders(full, frames, bg)
    if np.array_equal(expected, erased_only):
        return False

    hidden_removed = full[hr0 : hr1 + 1, hc0 : hc1 + 1]
    if not np.any(hidden_removed == BLUE):
        return False

    for r0, c0, r1, c1 in frames:
        inner = full[r0 + 1 : r1, c0 + 1 : c1]
        if np.array_equal(inner, np.roll(inner, shift=dc, axis=1)):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(16, 20)
    w = random.randint(16, 20)
    bg = random.choice(GEN_BG_CHOICES)

    period_choices = [(dr, dc) for dr in range(2, 5) for dc in range(2, 5) if np.gcd(dr, dc) == 1]
    dr, dc = random.choice(period_choices)

    full = _generate_full_pattern(h, w, dr, dc, bg)
    if full is None:
        return None

    hidden_rect = _choose_hidden_rect(h, w)
    if hidden_rect is None:
        return None
    hr0, hc0, hr1, hc1 = hidden_rect
    partial = np.array(full, copy=True)
    partial[hr0 : hr1 + 1, hc0 : hc1 + 1] = int(bg)
    if np.array_equal(partial, full):
        return None

    frames = _pick_frames(full, dr, dc, hidden_rect)
    if frames is None:
        return None

    inp = _overlay_frames(partial, frames)
    if not _semantic_checks(inp, full, frames, bg, dr, dc, hidden_rect):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
