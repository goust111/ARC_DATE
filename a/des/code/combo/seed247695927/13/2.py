from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(10))


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=int)
    cur = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or lab[r, c] != 0:
                continue
            cur += 1
            q = deque([(r, c)])
            lab[r, c] = cur
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and lab[nr, nc] == 0:
                        lab[nr, nc] = cur
                        q.append((nr, nc))
    return lab, cur


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    lab, n = _label_mask(mask)
    comps = []
    for i in range(1, n + 1):
        pts = np.argwhere(lab == i)
        comps.append([(int(r), int(c)) for r, c in pts])
    return comps


def _largest_solid_square(mask: np.ndarray) -> tuple[int, int, int]:
    h, w = mask.shape
    ps = mask.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[r1, c1])
        if r0 > 0:
            s -= int(ps[r0 - 1, c1])
        if c0 > 0:
            s -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            s += int(ps[r0 - 1, c0 - 1])
        return s

    best = None
    for k in range(2, min(h, w) + 1):
        found = None
        for r0 in range(h - k + 1):
            r1 = r0 + k - 1
            for c0 in range(w - k + 1):
                c1 = c0 + k - 1
                if rect_sum(r0, c0, r1, c1) == k * k:
                    found = (k, r0, c0)
        if found is not None:
            best = found
    if best is None:
        raise AssertionError("no solid square found")
    return best


def _network_score(mask: np.ndarray) -> float:
    comps = _components(mask)
    if not comps:
        return -10**9
    pts = max(comps, key=len)
    h, w = mask.shape
    deg2 = 0
    deg4 = 0
    for r, c in pts:
        d = 0
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                d += 1
        if d == 2:
            deg2 += 1
        elif d >= 3:
            deg4 += 1
    return len(pts) + 2.0 * deg2 - 3.0 * deg4


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors
    path_color = max(colors, key=lambda c: _network_score(g == c))

    hub_size, hub_r0, hub_c0 = _largest_solid_square(g == path_color)
    hub_mask = np.zeros((h, w), dtype=bool)
    hub_mask[hub_r0 : hub_r0 + hub_size, hub_c0 : hub_c0 + hub_size] = True

    seed_mask = (g != bg) & (g != path_color)
    seed_lab, seed_n = _label_mask(seed_mask)
    assert seed_n > 0
    seed_major: dict[int, int] = {}
    for sid in range(1, seed_n + 1):
        vals = [int(g[r, c]) for r, c in np.argwhere(seed_lab == sid)]
        seed_major[sid] = int(Counter(vals).most_common(1)[0][0])

    branch_mask = (g == path_color) & (~hub_mask)
    branch_lab, branch_n = _label_mask(branch_mask)
    assert branch_n > 0
    out = g.copy()
    branch_to_seed_color: dict[int, int] = {}

    for bid in range(1, branch_n + 1):
        touches = Counter()
        pts = np.argwhere(branch_lab == bid)
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    sid = int(seed_lab[nr, nc])
                    if sid:
                        touches[sid] += 1
        assert touches
        sid = int(touches.most_common(1)[0][0])
        color = seed_major[sid]
        branch_to_seed_color[bid] = color
        for r, c in pts:
            out[int(r), int(c)] = color

    hub_touch = Counter()
    for r in range(h):
        for c in range(w):
            if not hub_mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    bid = int(branch_lab[nr, nc])
                    if bid:
                        hub_touch[branch_to_seed_color[bid]] += 1
    assert hub_touch
    center_r = hub_r0 + hub_size // 2
    center_c = hub_c0 + hub_size // 2
    out[center_r, center_c] = int(hub_touch.most_common(1)[0][0])
    return out


def _paint_rect(grid: np.ndarray, r0: int, c0: int, hh: int, ww: int, color: int) -> None:
    grid[r0 : r0 + hh, c0 : c0 + ww] = color


def _choose_colors(rng: random.Random, n: int) -> list[int]:
    return rng.sample(PALETTE, n)


def _branch_cells(side: str, hub_r0: int, hub_c0: int, hub_size: int, attach_w: int, merge_shift: int) -> list[tuple[int, int]]:
    if side == "N":
        c0 = hub_c0 + merge_shift
        start = hub_c0 + (hub_size - attach_w) // 2
        return [(hub_r0 - 1, c) for c in range(start, start + attach_w)], c0
    if side == "S":
        c0 = hub_c0 + merge_shift
        start = hub_c0 + (hub_size - attach_w) // 2
        return [(hub_r0 + hub_size, c) for c in range(start, start + attach_w)], c0
    if side == "W":
        r0 = hub_r0 + merge_shift
        start = hub_r0 + (hub_size - attach_w) // 2
        return [(r, hub_c0 - 1) for r in range(start, start + attach_w)], r0
    r0 = hub_r0 + merge_shift
    start = hub_r0 + (hub_size - attach_w) // 2
    return [(r, hub_c0 + hub_size) for r in range(start, start + attach_w)], r0


def _add_optional_secondary(
    grid: np.ndarray,
    rng: random.Random,
    side: str,
    terminal: list[tuple[int, int]],
    used_colors: set[int],
    bg: int,
    path_color: int,
) -> None:
    if len(terminal) < 2 or rng.random() < 0.45:
        return
    color = rng.choice([c for c in PALETTE if c not in used_colors and c not in {bg, path_color}])
    used_colors.add(color)
    span = rng.randint(1, len(terminal) - 1)
    start = rng.randint(0, len(terminal) - span)
    chosen = terminal[start : start + span]
    if side == "N":
        rows = [r + 1 for r, _ in chosen]
        rr = min(rows)
        cc0 = min(c for _, c in chosen)
        _paint_rect(grid, rr, cc0, rng.randint(1, 2), span, color)
    elif side == "S":
        rr = min(r for r, _ in chosen) - rng.randint(1, 2)
        cc0 = min(c for _, c in chosen)
        _paint_rect(grid, rr, cc0, rng.randint(1, 2), span, color)
    elif side == "W":
        rr0 = min(r for r, _ in chosen)
        cc = min(c for _, c in chosen) + 1
        _paint_rect(grid, rr0, cc, span, rng.randint(1, 2), color)
    else:
        rr0 = min(r for r, _ in chosen)
        cc = min(c for _, c in chosen) - rng.randint(1, 2)
        _paint_rect(grid, rr0, cc, span, rng.randint(1, 2), color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_branches = rng.choice([2, 3, 4])
    hub_size = 5 if num_branches == 4 or rng.random() < 0.6 else 3
    if num_branches > hub_size:
        hub_size = 5

    h = rng.randint(18, 29)
    w = rng.randint(18, 29)
    colors = _choose_colors(rng, min(10, num_branches * 2 + 3))
    bg, path_color = colors[0], colors[1]
    used_colors = {bg, path_color}

    margin = 6
    if h < hub_size + 2 * margin or w < hub_size + 2 * margin:
        return None
    hub_r0 = rng.randint(margin, h - hub_size - margin)
    hub_c0 = rng.randint(margin, w - hub_size - margin)

    grid = np.full((h, w), bg, dtype=int)
    _paint_rect(grid, hub_r0, hub_c0, hub_size, hub_size, path_color)

    sides = rng.sample(["N", "S", "W", "E"], num_branches)
    widths = rng.sample(list(range(1, hub_size + 1)), num_branches)
    rng.shuffle(widths)

    for side, attach_w in zip(sides, widths):
        seed_color = rng.choice([c for c in PALETTE if c not in used_colors])
        used_colors.add(seed_color)
        contact_len = rng.randint(max(2, attach_w), max(4, attach_w))

        if side in ("N", "S"):
            seed_h = rng.randint(2, 4)
            term_row_gap = rng.randint(3, 6)
            merge_shift = rng.randint(0, hub_size - 1)
            attach, merge_col = _branch_cells(side, hub_r0, hub_c0, hub_size, attach_w, merge_shift)
            for r, c in attach:
                grid[r, c] = path_color
            stem_col = int(merge_col)
            if side == "N":
                term_row = hub_r0 - term_row_gap
                if term_row - seed_h < 1:
                    return None
                left = max(1, stem_col - contact_len + 1)
                right = min(w - contact_len - 1, stem_col)
                if left > right:
                    return None
                seed_c0 = rng.randint(left, right)
                for r in range(term_row, hub_r0 - 1):
                    grid[r, stem_col] = path_color
                c1 = seed_c0 + contact_len - 1
                for c in range(min(stem_col, c1), max(stem_col, c1) + 1):
                    grid[term_row, c] = path_color
                _paint_rect(grid, term_row - seed_h, seed_c0, seed_h, contact_len, seed_color)
                terminal = [(term_row, c) for c in range(seed_c0, seed_c0 + contact_len)]
            else:
                term_row = hub_r0 + hub_size - 1 + term_row_gap
                if term_row + seed_h >= h - 1:
                    return None
                left = max(1, stem_col - contact_len + 1)
                right = min(w - contact_len - 1, stem_col)
                if left > right:
                    return None
                seed_c0 = rng.randint(left, right)
                for r in range(hub_r0 + hub_size, term_row + 1):
                    grid[r, stem_col] = path_color
                c1 = seed_c0 + contact_len - 1
                for c in range(min(stem_col, c1), max(stem_col, c1) + 1):
                    grid[term_row, c] = path_color
                _paint_rect(grid, term_row + 1, seed_c0, seed_h, contact_len, seed_color)
                terminal = [(term_row, c) for c in range(seed_c0, seed_c0 + contact_len)]
        else:
            seed_w = rng.randint(2, 4)
            term_col_gap = rng.randint(3, 6)
            merge_shift = rng.randint(0, hub_size - 1)
            attach, merge_row = _branch_cells(side, hub_r0, hub_c0, hub_size, attach_w, merge_shift)
            for r, c in attach:
                grid[r, c] = path_color
            stem_row = int(merge_row)
            if side == "W":
                term_col = hub_c0 - term_col_gap
                if term_col - seed_w < 1:
                    return None
                top = max(1, stem_row - contact_len + 1)
                bot = min(h - contact_len - 1, stem_row)
                if top > bot:
                    return None
                seed_r0 = rng.randint(top, bot)
                for c in range(term_col, hub_c0 - 1):
                    grid[stem_row, c] = path_color
                r1 = seed_r0 + contact_len - 1
                for r in range(min(stem_row, r1), max(stem_row, r1) + 1):
                    grid[r, term_col] = path_color
                _paint_rect(grid, seed_r0, term_col - seed_w, contact_len, seed_w, seed_color)
                terminal = [(r, term_col) for r in range(seed_r0, seed_r0 + contact_len)]
            else:
                term_col = hub_c0 + hub_size - 1 + term_col_gap
                if term_col + seed_w >= w - 1:
                    return None
                top = max(1, stem_row - contact_len + 1)
                bot = min(h - contact_len - 1, stem_row)
                if top > bot:
                    return None
                seed_r0 = rng.randint(top, bot)
                for c in range(hub_c0 + hub_size, term_col + 1):
                    grid[stem_row, c] = path_color
                r1 = seed_r0 + contact_len - 1
                for r in range(min(stem_row, r1), max(stem_row, r1) + 1):
                    grid[r, term_col] = path_color
                _paint_rect(grid, seed_r0, term_col + 1, contact_len, seed_w, seed_color)
                terminal = [(r, term_col) for r in range(seed_r0, seed_r0 + contact_len)]

        _add_optional_secondary(grid, rng, side, terminal, used_colors, bg, path_color)

    if np.any(grid[0] != bg) or np.any(grid[-1] != bg) or np.any(grid[:, 0] != bg) or np.any(grid[:, -1] != bg):
        return None

    if _get_bg(grid) != bg:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
