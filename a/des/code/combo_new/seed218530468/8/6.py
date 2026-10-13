from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_border_color(sub: np.ndarray) -> int:
    border = np.concatenate((sub[0, :], sub[-1, :], sub[:, 0], sub[:, -1]))
    border = border[border != BG]
    assert border.size > 0, "envelope border must contain shell color"
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[np.argmax(counts)])


def _gap_side(sub: np.ndarray) -> str:
    counts = {
        "up": int(np.count_nonzero(sub[0, :] == BG)),
        "down": int(np.count_nonzero(sub[-1, :] == BG)),
        "left": int(np.count_nonzero(sub[:, 0] == BG)),
        "right": int(np.count_nonzero(sub[:, -1] == BG)),
    }
    positive = [side for side, count in counts.items() if count > 0]
    assert positive, "envelope must have one border gap"
    return max(positive, key=lambda side: counts[side])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each embedded letter out through its envelope opening and repair the shell behind it.

    Input structure:
    - The background is black.
    - Each foreground object is a dense monochrome rectangle whose border has one contiguous black gap on exactly one side.
    - Inside each rectangle sits one connected letter-like pattern of a second color; the letter bbox is strictly inside the shell bbox.

    Rule:
    - For every envelope, detect which border side contains the gap; that side determines the slide direction.
    - Move the inner letter rigidly straight through that gap until the entire letter bbox is just outside the shell, leaving exactly the former border thickness as the separation.
    - Repaint the letter's original cells with the shell color, keep the gap black, and place the moved letter outside the shell.

    Key insight:
    - The missing border segment is both the directional cue and the exact exit slot for the inner letter, so solving the task reduces to bbox-based gap detection plus rigid translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    for comp in _components4(g != BG):
        r0, c0, r1, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]

        shell_color = _dominant_border_color(sub)
        letter_mask = (sub != BG) & (sub != int(shell_color))
        assert bool(letter_mask.any()), "envelope must contain an inner letter"

        letter_colors = np.unique(sub[letter_mask])
        assert letter_colors.size == 1, "expected a single-color inner letter"
        letter_color = int(letter_colors[0])

        letter_coords = np.argwhere(letter_mask)
        lr0, lc0 = letter_coords.min(axis=0)
        lr1, lc1 = letter_coords.max(axis=0)
        lr0, lc0, lr1, lc1 = map(int, (lr0, lc0, lr1, lc1))

        side = _gap_side(sub)
        dr = dc = 0
        if side == "up":
            dr = int(-lr1 - 1)
        elif side == "down":
            dr = int(sub.shape[0] - lr0)
        elif side == "left":
            dc = int(-lc1 - 1)
        else:
            dc = int(sub.shape[1] - lc0)

        for rr, cc in letter_coords:
            out[r0 + int(rr), c0 + int(cc)] = int(shell_color)

        for rr, cc in letter_coords:
            nr = int(r0 + rr + dr)
            nc = int(c0 + cc + dc)
            assert 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1], "generated puzzles keep moved letters in bounds"
            out[nr, nc] = int(letter_color)

    return out


def _random_connected_letter(h: int, w: int) -> np.ndarray:
    area_min = max(3, h + w - 2, int(np.ceil(0.45 * h * w)))
    area_max = max(area_min, int(np.floor(0.8 * h * w)))

    for _ in range(80):
        target = random.randint(area_min, area_max)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            rr, cc = cell
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))

        add_frontier(start)
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            add_frontier(cell)

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        if not mask.any(axis=0).all():
            continue
        if not mask.any(axis=1).all():
            continue
        if int(mask.sum()) == h * w and random.random() < 0.85:
            continue
        return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, -1] = True
    if random.random() < 0.5:
        mask = np.rot90(mask)
    return mask


def _build_envelope(shell_color: int, letter_color: int, side: str) -> dict[str, object]:
    shell_h = random.randint(5, 9)
    shell_w = random.randint(5, 9)
    letter_h = random.randint(2, min(5, shell_h - 2))
    letter_w = random.randint(2, min(5, shell_w - 2))
    letter_r0 = random.randint(1, shell_h - letter_h - 1)
    letter_c0 = random.randint(1, shell_w - letter_w - 1)
    letter_mask = _random_connected_letter(letter_h, letter_w)

    patch = np.full((shell_h, shell_w), int(shell_color), dtype=int)
    letter_view = patch[letter_r0 : letter_r0 + letter_h, letter_c0 : letter_c0 + letter_w]
    letter_view[letter_mask] = int(letter_color)

    if side == "up":
        patch[0, letter_c0 : letter_c0 + letter_w] = BG
        safe_rect = (-letter_h, 0, shell_h - 1, shell_w - 1)
    elif side == "down":
        patch[-1, letter_c0 : letter_c0 + letter_w] = BG
        safe_rect = (0, 0, shell_h + letter_h - 1, shell_w - 1)
    elif side == "left":
        patch[letter_r0 : letter_r0 + letter_h, 0] = BG
        safe_rect = (0, -letter_w, shell_h - 1, shell_w - 1)
    else:
        patch[letter_r0 : letter_r0 + letter_h, -1] = BG
        safe_rect = (0, 0, shell_h - 1, shell_w + letter_w - 1)

    return {
        "patch": patch,
        "side": side,
        "safe_rect": tuple(map(int, safe_rect)),
        "shape": (int(shell_h), int(shell_w)),
    }


def _place_objects(grid: np.ndarray, objects: list[dict[str, object]]) -> bool:
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)

    ordered = sorted(
        objects,
        key=lambda obj: (
            -(
                (int(obj["safe_rect"][2]) - int(obj["safe_rect"][0]) + 1)
                * (int(obj["safe_rect"][3]) - int(obj["safe_rect"][1]) + 1)
            )
        ),
    )

    for obj in ordered:
        patch = np.asarray(obj["patch"], dtype=int)
        shell_h, shell_w = patch.shape
        sr0, sc0, sr1, sc1 = map(int, obj["safe_rect"])
        row_lo = int(-sr0)
        row_hi = int(h - 1 - sr1)
        col_lo = int(-sc0)
        col_hi = int(w - 1 - sc1)
        if row_lo > row_hi or col_lo > col_hi:
            return False

        candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            rr0 = int(r + sr0)
            rr1 = int(r + sr1)
            cc0 = int(c + sc0)
            cc1 = int(c + sc1)
            pad_r0 = max(0, rr0 - 1)
            pad_r1 = min(h - 1, rr1 + 1)
            pad_c0 = max(0, cc0 - 1)
            pad_c1 = min(w - 1, cc1 + 1)
            if reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                continue
            grid[r : r + shell_h, c : c + shell_w] = patch
            reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_env = random.choices([1, 2, 3, 4], weights=[3, 4, 2, 1], k=1)[0]
    directions = random.sample(DIRS, k=n_env)
    palette = random.sample(list(range(1, 10)), 2 * n_env)

    objects = []
    for idx, side in enumerate(directions):
        shell_color = int(palette[idx])
        letter_color = int(palette[n_env + idx])
        objects.append(_build_envelope(shell_color, letter_color, side))

    min_h = max(12, max(int(obj["safe_rect"][2]) - int(obj["safe_rect"][0]) + 1 for obj in objects) + 2)
    min_w = max(12, max(int(obj["safe_rect"][3]) - int(obj["safe_rect"][1]) + 1 for obj in objects) + 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        grid = np.zeros((h, w), dtype=int)
        if not _place_objects(grid, objects):
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid envelope puzzle")
