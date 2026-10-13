from __future__ import annotations

import random

import numpy as np


BG_COLOR = 7
PALETTE = [c for c in range(10) if c != BG_COLOR]
GEN_RETRIES = 300
AXIS_GAP_MIN = 2


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = set() if exclude is None else set(exclude)
    return [int(c) for c in np.unique(grid) if int(c) not in exclude]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _best_vertical_axis(mask: np.ndarray) -> int:
    h, w = mask.shape
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert pts

    best_score = -1
    best_k = 0
    for k in range(2 * w - 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < w and mask[r, cc]:
                score += 1
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each non-background color layer by enforcing its own best vertical mirror symmetry.

    For each color independently, find the vertical axis that maximizes same-color mirrored
    matches, then remove pixels that do not have a same-color mirror partner across that axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    out = g.copy()

    for color in _colors_present(g, exclude={bg}):
        mask = g == color
        if not mask.any():
            continue

        k = _best_vertical_axis(mask)
        for r_raw, c_raw in np.argwhere(mask):
            r = int(r_raw)
            c = int(c_raw)
            cc = k - c
            if not (0 <= cc < g.shape[1] and g[r, cc] == color):
                out[r, c] = bg

    return out


def _axis_scores(mask: np.ndarray) -> dict[int, int]:
    h, w = mask.shape
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    scores: dict[int, int] = {}
    for k in range(2 * w - 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < w and mask[r, cc]:
                score += 1
        scores[k] = score
    return scores


def _has_unique_best_axis(mask: np.ndarray, gap_min: int) -> bool:
    scores = _axis_scores(mask)
    best_k = max(scores, key=lambda k: scores[k])
    best = scores[best_k]
    second = max((score for k, score in scores.items() if k != best_k), default=-1)
    return best >= 0 and best - second >= gap_min


def _mirror_col(c: int, axis_k: int) -> int:
    return axis_k - c


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    for _ in range(120):
        h_hi = min(8, max_h)
        half_hi = min(4, max_w // 2)
        if h_hi < 5 or half_hi < 2:
            break
        h = random.randint(5, h_hi)
        odd_width = random.choice([False, True])
        if odd_width:
            half_w = random.randint(2, half_hi)
            w = 2 * half_w + 1
            if w > max_w:
                continue
            axis_k = w - 1
            side_cols = list(range(half_w))
            center_col = half_w
            center_rows = random.sample(range(h), random.randint(0, max(1, h // 3)))
        else:
            half_w = random.randint(2, half_hi)
            w = 2 * half_w
            axis_k = w - 1
            side_cols = list(range(half_w))
            center_col = None
            center_rows = []

        left_cells = set()
        for r in range(h):
            left_cells.add((r, random.choice(side_cols)))
            if random.random() < 0.45:
                extra_choices = [c for c in side_cols if (r, c) not in left_cells]
                if extra_choices:
                    left_cells.add((r, random.choice(extra_choices)))

        sprite = np.full((h, w), BG_COLOR, dtype=int)
        for r, c in left_cells:
            sprite[r, c] = color
            sprite[r, _mirror_col(c, axis_k)] = color
        for r in center_rows:
            if center_col is not None:
                sprite[r, center_col] = color

        mask = sprite == color
        if mask.sum() < 10:
            continue
        rows = np.any(mask, axis=1)
        if rows.sum() < 3:
            continue
        used_cols = {c for _, c in left_cells}
        if len(used_cols) < 2:
            continue
        if not _has_unique_best_axis(mask, max(AXIS_GAP_MIN, 3)):
            continue
        return sprite, axis_k
    raise RuntimeError("failed to build clean sprite")


def _noise_candidates(mask: np.ndarray, axis_k: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            cc = _mirror_col(c, axis_k)
            if cc == c or not (0 <= cc < w) or mask[r, cc]:
                continue
            if not any(0 <= rr < h and 0 <= cc2 < w and mask[rr, cc2] for rr, cc2 in _neighbors4(r, c)):
                continue
            candidates.append((r, c))
    return candidates


def _add_noise(sprite: np.ndarray, color: int, axis_k: int) -> np.ndarray | None:
    clean_mask = sprite == color
    candidates = _noise_candidates(clean_mask, axis_k)
    if not candidates:
        return None

    random.shuffle(candidates)
    sample_pool = candidates[: min(len(candidates), 10)]
    combos: list[list[tuple[int, int]]] = [[cell] for cell in sample_pool]
    combos += [[sample_pool[i], sample_pool[j]] for i in range(len(sample_pool)) for j in range(i + 1, len(sample_pool))]
    random.shuffle(combos)
    for chosen in combos[:50]:
        trial = sprite.copy()
        for r, c in chosen:
            trial[r, c] = color
        if np.array_equal(main(trial), sprite):
            return trial
    return None


def _sample_slots(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    mid_r = h // 2
    mid_c = w // 2
    gutter = 1
    slots = [
        (1, 1, mid_r - gutter, mid_c - gutter),
        (1, mid_c + gutter, mid_r - gutter, w - 1),
        (mid_r + gutter, 1, h - 1, mid_c - gutter),
        (mid_r + gutter, mid_c + gutter, h - 1, w - 1),
    ]
    slots = [slot for slot in slots if slot[2] - slot[0] >= 5 and slot[3] - slot[1] >= 6]
    if len(slots) < count:
        return None
    random.shuffle(slots)
    return slots[:count]


def _sample_placement(sprite: np.ndarray, slot: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = slot
    sh, sw = sprite.shape
    if sh > r1 - r0 or sw > c1 - c0:
        return None
    top = random.randint(r0, r1 - sh)
    left = random.randint(c0, c1 - sw)
    return top, left


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> bool:
    sh, sw = sprite.shape
    patch = grid[top : top + sh, left : left + sw]
    if np.any(patch != BG_COLOR):
        return False
    patch[:] = sprite
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    color_count = random.randint(2, 4)
    slots = _sample_slots(h, w, color_count)
    if slots is None:
        return None

    colors = random.sample(PALETTE, color_count)
    inp = np.full((h, w), BG_COLOR, dtype=int)
    clean = np.full_like(inp, BG_COLOR)

    for color, slot in zip(colors, slots):
        slot_h = slot[2] - slot[0]
        slot_w = slot[3] - slot[1]
        placed = False
        for _ in range(20):
            sprite, axis_k = _build_clean_sprite(color, slot_h, slot_w)
            noisy = _add_noise(sprite, color, axis_k)
            if noisy is None:
                continue
            pos = _sample_placement(sprite, slot)
            if pos is None:
                continue
            top, left = pos
            if not _paste_sprite(clean, sprite, top, left):
                continue
            if not _paste_sprite(inp, noisy, top, left):
                clean[top : top + sprite.shape[0], left : left + sprite.shape[1]] = BG_COLOR
                continue
            placed = True
            break
        if not placed:
            return None

    if not np.array_equal(main(inp), clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
