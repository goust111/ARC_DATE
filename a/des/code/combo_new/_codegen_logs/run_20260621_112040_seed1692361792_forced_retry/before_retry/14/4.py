import random
from collections import deque

import numpy as np

BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _normalize(coords):
    r0, _, c0, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _gap_groups(cells):
    cell_set = set(cells)
    groups = []
    seen = set()
    for cell in cell_set:
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (rr + dr, cc + dc)
                if nxt in cell_set and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        groups.append(comp)
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the matching missing border segment of a grey frame.

    Input structure:
    - Black background with several grey hollow rectangular frames.
    - Each frame border is missing one or two contiguous runs of border pixels.
    - Outside the frames, monochrome connected fragments match those missing runs.

    Rule:
    - For every frame, identify each missing border run and the border side it belongs to.
    - Match an outside fragment to a gap by shape, allowing horizontal or vertical reflection but no
      90-degree rotation.
    - Recolor the black gap cells with the fragment's color and erase the original fragment location.
    - Keep the frame interiors black and leave the rest of each grey border unchanged.

    Key insight:
    - The fragment shape identifies the destination gap, while the gap's side determines which reflection
      aligns the fragment to the frame.
    """
    out = grid.copy()
    frames = []
    for comp in _components(grid == GREY):
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        border = []
        for c in range(c0, c1 + 1):
            border.append((r0, c))
            border.append((r1, c))
        for r in range(r0 + 1, r1):
            border.append((r, c0))
            border.append((r, c1))
        missing = [(r, c) for r, c in border if grid[r, c] == BLACK]
        if not missing:
            continue
        frames.append((r0, r1, c0, c1, _gap_groups(missing)))

    fragments = []
    for color in np.unique(grid):
        if color in (BLACK, GREY):
            continue
        for comp in _components(grid == color):
            fragments.append((int(color), comp))

    unmatched = fragments[:]
    for _, _, _, _, gap_groups in frames:
        for gap in gap_groups:
            gap_norm = _normalize(gap)
            chosen_idx = None
            for idx, (_, comp) in enumerate(unmatched):
                comp_norm = _normalize(comp)
                rows = [r for r, _ in comp_norm]
                cols = [c for _, c in comp_norm]
                h = max(rows) + 1
                w = max(cols) + 1
                variants = [
                    comp_norm,
                    {(r, w - 1 - c) for r, c in comp_norm},
                    {(h - 1 - r, c) for r, c in comp_norm},
                ]
                if gap_norm in variants:
                    chosen_idx = idx
                    break
            if chosen_idx is None:
                continue
            color, comp = unmatched.pop(chosen_idx)
            for r, c in comp:
                out[r, c] = BLACK
            for r, c in gap:
                out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        grid = np.zeros((h, w), dtype=int)
        used = np.zeros((h, w), dtype=bool)

        num_frames = rng.randint(2, 4)
        colors = rng.sample(PALETTE, num_frames * 2)
        color_idx = 0
        gap_lengths = set()
        fragment_specs = []
        ok = True

        for _frame_id in range(num_frames):
            placed = False
            for _ in range(300):
                fh = rng.randint(6, 10)
                fw = rng.randint(6, 10)
                top = rng.randint(1, h - fh - 1)
                left = rng.randint(1, w - fw - 1)
                bottom = top + fh - 1
                right = left + fw - 1
                rr0 = max(0, top - 1)
                rr1 = min(h, bottom + 2)
                cc0 = max(0, left - 1)
                cc1 = min(w, right + 2)
                if used[rr0:rr1, cc0:cc1].any():
                    continue
                gaps = []
                sides = rng.sample(["top", "bottom", "left", "right"], rng.randint(1, 2))
                local_lengths = set()
                for side in sides:
                    max_len = (fw - 2) if side in ("top", "bottom") else (fh - 2)
                    if max_len < 2:
                        break
                    length = rng.randint(2, min(4, max_len))
                    if (side in ("top", "bottom") and length in local_lengths) or length in gap_lengths:
                        break
                    local_lengths.add(length)
                    gap_lengths.add(length)
                    if side == "top":
                        start = rng.randint(left + 1, right - length)
                        cells = [(top, c) for c in range(start, start + length)]
                    elif side == "bottom":
                        start = rng.randint(left + 1, right - length)
                        cells = [(bottom, c) for c in range(start, start + length)]
                    elif side == "left":
                        start = rng.randint(top + 1, bottom - length)
                        cells = [(r, left) for r in range(start, start + length)]
                    else:
                        start = rng.randint(top + 1, bottom - length)
                        cells = [(r, right) for r in range(start, start + length)]
                    gaps.append(cells)
                else:
                    grid[top, left : right + 1] = GREY
                    grid[bottom, left : right + 1] = GREY
                    grid[top : bottom + 1, left] = GREY
                    grid[top : bottom + 1, right] = GREY
                    for cells in gaps:
                        for r, c in cells:
                            grid[r, c] = BLACK
                    used[rr0:rr1, cc0:cc1] = True
                    for cells in gaps:
                        color = colors[color_idx]
                        color_idx += 1
                        fragment_specs.append((color, cells))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        for color, cells in fragment_specs:
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            ph = max(rows) - min(rows) + 1
            pw = max(cols) - min(cols) + 1
            placed = False
            for _ in range(300):
                top = rng.randint(0, h - ph)
                left = rng.randint(0, w - pw)
                rr0 = max(0, top - 1)
                rr1 = min(h, top + ph + 1)
                cc0 = max(0, left - 1)
                cc1 = min(w, left + pw + 1)
                if used[rr0:rr1, cc0:cc1].any():
                    continue
                nr0, _, nc0, _ = _bbox(cells)
                for r, c in cells:
                    grid[top + (r - nr0), left + (c - nc0)] = color
                used[rr0:rr1, cc0:cc1] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
