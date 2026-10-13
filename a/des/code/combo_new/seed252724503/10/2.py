from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GEN_ATTEMPTS = 300


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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _holes_for_template(grid: np.ndarray, comp: list[tuple[int, int]], color: int) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(comp)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    seen = np.zeros_like(sub, dtype=bool)
    holes = []
    for r in range(sub.shape[0]):
        for c in range(sub.shape[1]):
            if sub[r, c] != BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr + r0, cc + c0))
                if rr in (0, sub.shape[0] - 1) or cc in (0, sub.shape[1] - 1):
                    touches_edge = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and sub[nr, nc] == BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes.append(cells)
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return each colored fragment to the matching same-colored template hole, allowing rotation.

    Input structure:
    - The background is black.
    - For each active color, one large component is a rectangular template with black internal holes.
    - Smaller components of the same color are loose fragments whose shapes match those holes up to rotation.

    Rule:
    - For each color, identify the largest same-colored component as the template and treat the smaller components as fragments.
    - Extract the enclosed black holes inside that template.
    - Match each fragment to the unique hole whose binary shape equals one of the fragment's four rotations.
    - Fill the matched hole with the color and erase the fragment from its original location.

    Key insight:
    - Color only narrows the search space; the actual placement is a local shape-fitting problem solved by rotating each fragment against the template's missing regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    by_color: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components(g == color):
            by_color[color].append(comp)

    for color, comps in by_color.items():
        comps.sort(key=len, reverse=True)
        template = comps[0]
        holes = _holes_for_template(g, template, color)
        hole_masks = [_crop_mask(hole) for hole in holes]
        used = [False] * len(holes)
        for frag in comps[1:]:
            frag_mask = _crop_mask(frag)
            match_idx = None
            for idx, hole_mask in enumerate(hole_masks):
                if used[idx]:
                    continue
                if any(np.array_equal(np.rot90(frag_mask, k), hole_mask) for k in range(4)):
                    match_idx = idx
                    break
            assert match_idx is not None
            used[match_idx] = True
            for r, c in frag:
                out[r, c] = BG
            for r, c in holes[match_idx]:
                out[r, c] = color
    return out


def _random_hole_mask() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    mask = np.zeros((h, w), dtype=bool)
    r, c = random.randrange(h), random.randrange(w)
    mask[r, c] = True
    target = random.randint(2, min(h * w, 5))
    while int(mask.sum()) < target:
        frontier = []
        for rr, cc in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(20, 30)
        w = random.randint(20, 30)
        grid = np.full((h, w), BG, dtype=int)
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 3))
        ok = True

        for color in colors:
            placed = False
            for _ in range(120):
                th = random.randint(7, 10)
                tw = random.randint(7, 10)
                r0 = random.randint(1, h - th - 1)
                c0 = random.randint(1, w - tw - 1)
                if np.any(grid[max(0, r0 - 1) : min(h, r0 + th + 1), max(0, c0 - 1) : min(w, c0 + tw + 1)] != BG):
                    continue
                hole_count = 1
                holes = []
                temp = np.full((th, tw), color, dtype=int)
                for _hole in range(hole_count):
                    hole_mask = _random_hole_mask()
                    hh, hw = hole_mask.shape
                    tries = 0
                    while tries < 50:
                        hr = random.randint(1, th - hh - 1)
                        hc = random.randint(1, tw - hw - 1)
                        region = temp[hr : hr + hh, hc : hc + hw]
                        if np.all(region == color):
                            temp[hr : hr + hh, hc : hc + hw][hole_mask] = BG
                            holes.append(np.argwhere(hole_mask))
                            break
                        tries += 1
                    else:
                        break
                if len(holes) != hole_count:
                    continue
                grid[r0 : r0 + th, c0 : c0 + tw] = temp

                for hole in holes:
                    frag_mask = np.zeros((hole[:, 0].max() + 1, hole[:, 1].max() + 1), dtype=bool)
                    for rr, cc in hole:
                        frag_mask[int(rr), int(cc)] = True
                    frag_mask = np.rot90(frag_mask, random.randrange(4))
                    fh, fw = frag_mask.shape
                    for _frag_try in range(80):
                        fr = random.randint(1, h - fh - 1)
                        fc = random.randint(1, w - fw - 1)
                        if np.any(grid[max(0, fr - 1) : min(h, fr + fh + 1), max(0, fc - 1) : min(w, fc + fw + 1)] != BG):
                            continue
                        grid[fr : fr + fh, fc : fc + fw][frag_mask] = color
                        break
                    else:
                        break
                else:
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
