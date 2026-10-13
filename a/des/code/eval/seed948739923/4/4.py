from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict[str, object]] = []

    for comp in _components(g != BG):
        rows = [r for r, _c in comp]
        top_r = min(rows)
        top_cells = [(r, c) for r, c in comp if r == top_r]
        assert len(top_cells) == 1

        body = [(r, c) for r, c in comp if r > top_r]
        assert body

        marker_r, marker_c = top_cells[0]
        body_rows = [r for r, _c in body]
        body_cols = [c for _r, c in body]
        r0 = int(top_r + 1)
        r1 = int(max(body_rows))
        c0 = int(min(body_cols))
        c1 = int(max(body_cols))
        assert int(min(body_rows)) == r0
        assert c0 <= marker_c <= c1

        body_set = set(body)
        for rr in range(r0, r1 + 1):
            assert any((rr, cc) in body_set for cc in range(c0, c1 + 1))
        for cc in range(c0, c1 + 1):
            assert any((rr, cc) in body_set for rr in range(r0, r1 + 1))

        objects.append(
            {
                "marker_r": int(marker_r),
                "marker_c": int(marker_c),
                "marker_color": int(g[marker_r, marker_c]),
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )

    assert 3 <= len(objects) <= 5
    shapes = {tuple(np.asarray(obj["patch"]).shape) for obj in objects}
    assert len(shapes) == 1
    return sorted(objects, key=lambda obj: (int(obj["r0"]), int(obj["c0"])))


def _restore_pattern(objects: list[dict[str, object]]) -> np.ndarray:
    patches = np.stack([np.asarray(obj["patch"], dtype=int) for obj in objects], axis=0)
    _n, h, w = patches.shape
    restored = np.zeros((h, w), dtype=int)

    for r in range(h):
        for c in range(w):
            votes = patches[:, r, c]
            votes = votes[votes != BG]
            assert votes.size > 0
            colors, counts = np.unique(votes, return_counts=True)
            restored[r, c] = int(colors[np.argmax(counts)])

    return restored


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    assert r0 >= 1 and c0 >= 1 and r1 + 1 < grid.shape[0] and c1 + 1 < grid.shape[1]
    grid[r0 - 1, c0 - 1 : c1 + 2] = int(color)
    grid[r1 + 1, c0 - 1 : c1 + 2] = int(color)
    grid[r0 - 1 : r1 + 2, c0 - 1] = int(color)
    grid[r0 - 1 : r1 + 2, c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the common undamaged pattern by majority vote, then box each damaged copy
    with its marker color.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    restored = _restore_pattern(objects)

    out = g.copy()
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        r1 = int(obj["r1"])
        c1 = int(obj["c1"])
        patch = np.asarray(obj["patch"], dtype=int)
        out[r0 : r1 + 1, c0 : c1 + 1] = restored

        missing = (patch == BG) & (restored != BG)
        if bool(missing.any()):
            _draw_frame(out, r0, c0, r1, c1, int(obj["marker_color"]))

    return out


def _segment_lengths(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    lengths: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        lengths.append(int(cut - prev))
        prev = int(cut)
    return lengths


def _build_pattern(h: int, w: int, colors: list[int]) -> np.ndarray:
    pattern = np.empty((h, w), dtype=int)
    mode = random.choice(["vertical", "horizontal", "checker", "blocks"])

    if mode == "vertical":
        parts = random.randint(2, min(4, w))
        widths = _segment_lengths(w, parts)
        c = 0
        for i, width in enumerate(widths):
            pattern[:, c : c + width] = int(colors[i % len(colors)])
            c += width
    elif mode == "horizontal":
        parts = random.randint(2, min(4, h))
        heights = _segment_lengths(h, parts)
        r = 0
        for i, height in enumerate(heights):
            pattern[r : r + height, :] = int(colors[i % len(colors)])
            r += height
    elif mode == "checker":
        step = random.choice([1, 2])
        for r in range(h):
            for c in range(w):
                pattern[r, c] = int(colors[(r + step * c) % len(colors)])
    else:
        row_parts = random.randint(2, min(3, h))
        col_parts = random.randint(2, min(3, w))
        heights = _segment_lengths(h, row_parts)
        widths = _segment_lengths(w, col_parts)
        r = 0
        for i, height in enumerate(heights):
            c = 0
            for j, width in enumerate(widths):
                pattern[r : r + height, c : c + width] = int(colors[(2 * i + j) % len(colors)])
                c += width
            r += height

    if len(colors) >= 3 and random.random() < 0.7:
        accent = int(random.choice(colors))
        if random.random() < 0.5:
            pattern[random.randint(1, h - 2), :] = accent
        else:
            pattern[:, random.randint(1, w - 2)] = accent

    if len(colors) >= 4 and h >= 6 and w >= 6 and random.random() < 0.45:
        r0 = random.randint(1, h - 3)
        r1 = random.randint(r0 + 1, h - 2)
        c0 = random.randint(1, w - 3)
        c1 = random.randint(c0 + 1, w - 2)
        pattern[r0 : r1 + 1, c0 : c1 + 1] = int(random.choice(colors))

    if np.unique(pattern).size < 2:
        pattern[h // 2, :] = int(colors[1])

    return pattern


def _choose_layout(n_copies: int, h: int, w: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    padded_h = h + 2
    padded_w = w + 2
    options: list[tuple[int, int, int, int, int, int]] = []

    for rows in range(1, 4):
        for cols in range(1, 4):
            if rows * cols < n_copies:
                continue
            max_cell_h = (30 - 2) // rows
            max_cell_w = (30 - 2) // cols
            if padded_h > max_cell_h or padded_w > max_cell_w:
                continue
            for cell_h in range(padded_h, min(padded_h + 3, max_cell_h) + 1):
                for cell_w in range(padded_w, min(padded_w + 3, max_cell_w) + 1):
                    grid_h = rows * cell_h + 2
                    grid_w = cols * cell_w + 2
                    if 12 <= grid_h <= 30 and 12 <= grid_w <= 30:
                        options.append((rows, cols, cell_h, cell_w, grid_h, grid_w))

    if not options:
        return None

    rows, cols, cell_h, cell_w, grid_h, grid_w = random.choice(options)
    chosen_cells = random.sample([(r, c) for r in range(rows) for c in range(cols)], n_copies)
    positions: list[tuple[int, int]] = []
    for cell_r, cell_c in chosen_cells:
        off_r = random.randint(0, cell_h - padded_h)
        off_c = random.randint(0, cell_w - padded_w)
        padded_top = 1 + cell_r * cell_h + off_r
        padded_left = 1 + cell_c * cell_w + off_c
        positions.append((int(padded_top + 1), int(padded_left + 1)))

    return int(grid_h), int(grid_w), positions


def _damage_masks(h: int, w: int, n_copies: int, clean_ids: set[int]) -> list[set[tuple[int, int]]] | None:
    spine_r = h // 2
    spine_c = w // 2
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if r != spine_r and c != spine_c
    ]
    if not candidates:
        return None

    masks = [set() for _ in range(n_copies)]
    damaged_ids = [idx for idx in range(n_copies) if idx not in clean_ids]
    assert damaged_ids
    counts = {idx: 0 for idx in damaged_ids}
    has_clean = bool(clean_ids)

    shuffled = candidates[:]
    random.shuffle(shuffled)
    for cell in shuffled:
        max_k = min(2, len(damaged_ids))
        if not has_clean:
            max_k = min(max_k, len(damaged_ids) - 1)
        choices = [0, 1]
        weights = [2, 3]
        if max_k >= 2:
            choices.append(2)
            weights.append(2)
        k = int(random.choices(choices, weights=weights, k=1)[0])
        if k == 0:
            continue
        ranked = damaged_ids[:]
        random.shuffle(ranked)
        ranked.sort(key=lambda idx: counts[idx])
        for idx in ranked[:k]:
            masks[idx].add(cell)
            counts[idx] += 1

    for idx in damaged_ids:
        if counts[idx] > 0:
            continue
        fallback = candidates[:]
        random.shuffle(fallback)
        placed = False
        for cell in fallback:
            if cell in masks[idx]:
                continue
            if not has_clean:
                already = sum(cell in masks[other] for other in damaged_ids)
                if already >= len(damaged_ids) - 1:
                    continue
            masks[idx].add(cell)
            counts[idx] += 1
            placed = True
            break
        if not placed:
            return None

    if max(counts.values()) - min(counts.values()) > max(3, len(candidates) // 3):
        return None
    return masks


def _gen_attempt() -> np.ndarray | None:
    n_copies = random.randint(3, 5)
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    layout = _choose_layout(n_copies, h, w)
    if layout is None:
        return None
    grid_h, grid_w, positions = layout

    max_pattern_colors = min(4, len(COLORS) - n_copies)
    n_pattern_colors = random.randint(2, max_pattern_colors)
    palette = list(COLORS)
    random.shuffle(palette)
    marker_colors = palette[:n_copies]
    pattern_colors = palette[n_copies : n_copies + n_pattern_colors]
    pattern = _build_pattern(h, w, pattern_colors)

    clean_ids: set[int] = set()
    if random.random() < 0.5:
        clean_ids.add(random.randrange(n_copies))

    masks = _damage_masks(h, w, n_copies, clean_ids)
    if masks is None or not any(masks):
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros_like(inp)
    for idx, (r0, c0) in enumerate(positions):
        patch = pattern.copy()
        for rr, cc in masks[idx]:
            patch[rr, cc] = BG
        if len(_components(patch != BG)) != 1:
            return None

        inp[r0 : r0 + h, c0 : c0 + w] = patch
        expected[r0 : r0 + h, c0 : c0 + w] = pattern

        marker_c = c0 + random.randint(0, w - 1)
        marker_color = int(marker_colors[idx])
        inp[r0 - 1, marker_c] = marker_color
        expected[r0 - 1, marker_c] = marker_color

        if masks[idx]:
            _draw_frame(expected, r0, c0, r0 + h - 1, c0 + w - 1, marker_color)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
