from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GEN_ATTEMPTS = 300

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _corner_cards(grid: np.ndarray) -> tuple[dict[frozenset[int], int], set[tuple[int, int]]]:
    h, w = grid.shape
    cards = [
        grid[0:2, 0:2],
        grid[0:2, w - 2 : w],
        grid[h - 2 : h, 0:2],
        grid[h - 2 : h, w - 2 : w],
    ]
    used = {(r, c) for r in (0, 1, h - 2, h - 1) for c in (0, 1, w - 2, w - 1)}
    mixes: dict[frozenset[int], int] = {}
    for card in cards:
        a = int(card[0, 0])
        b = int(card[0, 1])
        out = int(card[1, 1])
        if a != BG and b != BG and out != BG:
            mixes[frozenset({a, b})] = out
    return mixes, used


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit cardinal rays from each source object and recolor ray crossings with the corner-card mix lookup.

    Input structure:
    - The background is black.
    - Four 2x2 corner cards encode unordered color-pair mixes by placing two input colors on the top row and the mixed output on the bottom-right.
    - Away from the corners, a few disconnected monochrome source objects occupy the main field.

    Rule:
    - Read the corner cards into an unordered pair-to-output-color mapping.
    - From every source pixel, cast rays in the four cardinal directions until the ray reaches another source object or leaves the grid.
    - Keep the source objects unchanged.
    - Paint empty traversed cells with the ray color when only one color reaches them; when two different colors reach the same cell, repaint that cell with the mixed color from the corner-card mapping.

    Key insight:
    - The corner blocks are not ordinary objects: they form a visible mix dictionary that must be consulted only at ray intersections, while the sources themselves still keep their own colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    mixes, card_cells = _corner_cards(g)
    source_mask = (g != BG).copy()
    for r, c in card_cells:
        source_mask[r, c] = False
    comps = _components(source_mask)

    source_cells = {cell for comp in comps for cell in comp}
    contributions: dict[tuple[int, int], set[int]] = defaultdict(set)
    for comp in comps:
        color = int(g[comp[0]])
        for r, c in comp:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w and (nr, nc) not in source_cells and (nr, nc) not in card_cells:
                    contributions[(nr, nc)].add(color)
                    nr += dr
                    nc += dc

    out = g.copy()
    for (r, c), colors in contributions.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        elif len(colors) >= 2:
            pair = frozenset(sorted(colors)[:2])
            out[r, c] = mixes[pair]
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.full((h, w), BG, dtype=int)
        source_colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 3))
        mix_colors = [c for c in range(1, 10) if c not in source_colors]
        pairs = []
        for i in range(len(source_colors)):
            for j in range(i + 1, len(source_colors)):
                pairs.append((source_colors[i], source_colors[j]))
        random.shuffle(mix_colors)
        mappings = {frozenset(pair): mix_colors[idx] for idx, pair in enumerate(pairs)}
        cards = list(mappings.items())
        while len(cards) < 4:
            cards.append(cards[-1])

        corners = [
            (0, 0),
            (0, w - 2),
            (h - 2, 0),
            (h - 2, w - 2),
        ]
        for (pair, out_color), (r0, c0) in zip(cards, corners):
            a, b = tuple(pair)
            grid[r0, c0] = a
            grid[r0, c0 + 1] = b
            grid[r0 + 1, c0 + 1] = out_color

        ok = True
        for color in source_colors:
            placed = False
            for _ in range(80):
                sh = random.randint(1, 2)
                sw = random.randint(1, 2)
                r0 = random.randint(3, h - sh - 4)
                c0 = random.randint(3, w - sw - 4)
                if np.any(grid[r0 : r0 + sh, c0 : c0 + sw] != BG):
                    continue
                grid[r0 : r0 + sh, c0 : c0 + sw] = color
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
