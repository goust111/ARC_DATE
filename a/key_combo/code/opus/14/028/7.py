from __future__ import annotations

import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _read_legend(grid: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        src = int(grid[r, 0])
        fill = int(grid[r, 1])
        if src == 0 or fill == 0:
            break
        pairs.append((src, fill))
    if not pairs:
        raise ValueError("expected a non-empty legend in columns 0-1")
    return pairs


def _compose_nested(
    motif: np.ndarray,
    pairs: list[tuple[int, int]],
    out_shape: tuple[int, int],
    *,
    stride: int | None = None,
) -> np.ndarray:
    n = int(motif.shape[0])
    stride = n - 1 if stride is None else int(stride)
    fill_of = {src: fill for src, fill in pairs}
    priority = {src: idx for idx, (src, _fill) in enumerate(pairs)}

    out = np.zeros(out_shape, dtype=int)
    best = np.full(out_shape, len(pairs) + 1, dtype=int)

    active = np.argwhere(motif != 0)
    colored = np.argwhere(motif != 0)
    for br, bc in active:
        base_r = int(br) * stride
        base_c = int(bc) * stride
        for mr, mc in colored:
            src = int(motif[mr, mc])
            tr = base_r + int(mr)
            tc = base_c + int(mc)
            if not (0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]):
                continue
            pr = priority[src]
            if pr < best[tr, tc]:
                best[tr, tc] = pr
                out[tr, tc] = fill_of[src]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a multicolor motif into a self-similar nested stamp and recolor it through the legend, resolving shared-border conflicts by legend order.

    Input structure:
    - The background is black.
    - A two-column legend starts at the upper-left corner; each row maps one source color to one fill color, and earlier rows have higher priority.
    - Away from the legend there is one compact square-bbox motif of size 3x3 or 4x4 using exactly the legend source colors plus black.

    Rule:
    - Read the legend rows, then ignore the legend cells when extracting the motif.
    - Crop the remaining foreground to its square bounding box M.
    - For every non-black pixel position of M, stamp the whole motif again on the output lattice so neighboring activated blocks share their touching border pixels.
    - Every non-black stamped pixel is recolored from its source color to the paired legend fill color.
    - If several stamps assign different source colors to the same output pixel, keep the color whose source appears earlier in the legend.

    Key insight:
    - The motif is both the activation mask and the stamp, but the legend has two jobs at once: it remaps colors and also encodes the per-pixel layer priority when shared motif borders collide.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    pairs = _read_legend(g)
    work = g.copy()
    work[: len(pairs), :2] = 0

    box = _bbox(work != 0)
    if box is None:
        raise ValueError("expected a motif outside the legend")
    motif = _crop_bbox(work, box)
    if motif.shape[0] != motif.shape[1]:
        raise ValueError("expected a square motif bbox")

    n = int(motif.shape[0])
    if g.shape != (n * n, n * n):
        raise ValueError("expected grid size equal to motif side length squared")

    src_colors = {src for src, _fill in pairs}
    motif_colors = set(map(int, motif[motif != 0]))
    if motif_colors != src_colors:
        raise ValueError("motif colors must match the legend source colors")

    return _compose_nested(motif, pairs, g.shape)


def _random_pairs(k: int) -> list[tuple[int, int]]:
    colors = list(range(1, 10))
    sources = random.sample(colors, k)
    remaining = [c for c in colors if c not in sources]
    fills = random.sample(remaining, k)
    rows = list(zip(sources, fills))
    random.shuffle(rows)
    return rows


def _assign_colors(mask: np.ndarray, sources: list[int]) -> np.ndarray:
    motif = np.zeros(mask.shape, dtype=int)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    random.shuffle(cells)

    forced = sources[:]
    random.shuffle(forced)
    for (r, c), src in zip(cells, forced):
        motif[r, c] = src
    for r, c in cells[len(forced) :]:
        motif[r, c] = random.choice(sources)
    return motif


def _bbox_is_full(mask: np.ndarray) -> bool:
    return (
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _has_neighbor_activation(mask: np.ndarray) -> bool:
    n = mask.shape[0]
    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < n and 0 <= cc < n and (rr, cc) in coords:
                    return True
    return False


def _count_conflicts(motif: np.ndarray, pairs: list[tuple[int, int]], stride: int) -> int:
    n = motif.shape[0]
    priority = {src: idx for idx, (src, _fill) in enumerate(pairs)}
    hits: dict[tuple[int, int], set[int]] = {}
    for br, bc in np.argwhere(motif != 0):
        base_r = int(br) * stride
        base_c = int(bc) * stride
        for mr, mc in np.argwhere(motif != 0):
            src = int(motif[mr, mc])
            pos = (base_r + int(mr), base_c + int(mc))
            hits.setdefault(pos, set()).add(priority[src])
    return sum(len(v) >= 2 for v in hits.values())


def _motif_is_valid(motif: np.ndarray, pairs: list[tuple[int, int]]) -> bool:
    n = motif.shape[0]
    area = n * n
    fg_count = int(np.count_nonzero(motif))
    if not (int(np.ceil(0.4 * area)) <= fg_count <= int(np.floor(0.7 * area))):
        return False
    if not _bbox_is_full(motif != 0):
        return False
    if set(map(int, motif[motif != 0])) != {src for src, _fill in pairs}:
        return False
    if not _has_neighbor_activation(motif != 0):
        return False

    shared = _compose_nested(motif, pairs, (n * n, n * n), stride=n - 1)
    separated = _compose_nested(motif, pairs, (n * n, n * n), stride=n)
    reversed_pairs = list(reversed(pairs))
    reversed_priority = _compose_nested(motif, reversed_pairs, (n * n, n * n), stride=n - 1)

    if np.array_equal(shared, separated):
        return False
    if np.array_equal(shared, reversed_priority):
        return False
    if _count_conflicts(motif, pairs, stride=n - 1) == 0:
        return False

    out_colors = set(map(int, shared[shared != 0]))
    if len(out_colors) < 2:
        return False
    return True


def _random_motif(n: int, pairs: list[tuple[int, int]]) -> np.ndarray | None:
    area = n * n
    min_fg = int(np.ceil(0.4 * area))
    max_fg = int(np.floor(0.7 * area))
    sources = [src for src, _fill in pairs]

    for _ in range(200):
        fg_count = random.randint(min_fg, max_fg)
        mask = np.zeros((n, n), dtype=bool)
        indices = random.sample(range(area), fg_count)
        for idx in indices:
            mask[idx // n, idx % n] = True
        if not _bbox_is_full(mask):
            continue
        motif = _assign_colors(mask, sources)
        if _motif_is_valid(motif, pairs):
            return motif
    return None


def _place_motif(grid: np.ndarray, motif: np.ndarray, legend_rows: int) -> None:
    n = motif.shape[0]
    top = grid.shape[0] - n
    left = grid.shape[1] - n
    if top < legend_rows or left < 2:
        raise ValueError("motif placement would overlap the legend")
    grid[top : top + n, left : left + n] = motif


def _build_input(n: int, pairs: list[tuple[int, int]], motif: np.ndarray) -> np.ndarray:
    grid = np.zeros((n * n, n * n), dtype=int)
    grid[: len(pairs), :2] = np.asarray(pairs, dtype=int)
    _place_motif(grid, motif, len(pairs))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        n = random.choice([3, 4])
        k = random.choice([2, 3])
        pairs = _random_pairs(k)
        motif = _random_motif(n, pairs)
        if motif is None:
            continue
        grid = _build_input(n, pairs, motif)
        out = main(grid)
        if not np.array_equal(out, _compose_nested(motif, pairs, grid.shape)):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
