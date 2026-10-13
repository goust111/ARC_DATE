import random
from collections import Counter, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


BACKGROUND = 8
YELLOW = 4


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each complementary fragment pair into a solid rectangle at the marker piece position.
Input structure:
- Background is a single azure color and all fragments are connected non-background components.
- Fragments come in pairs with equal bounding-box size and disjoint cell sets that exactly complement each other into a solid rectangle.
- Exactly one fragment in each pair has one corner pixel marked yellow, while all other cells in the pair are monochromatic.
Rule:
- Find all non-background connected components and match complementary pairs by equal bounding-box dimensions where their normalized masks are disjoint and union to a full rectangle.
- For each pair, align the two fragments by their bounding box top-left corner.
- The component carrying the yellow marker is fixed in place: its normal color fills the merged rectangle where it is present, and its yellow marker pixel reverts to its color.
- The other fragment is translated to the aligned position to fill the remaining area; its original cells become background.
- Regions not participating in a complementary pair remain unchanged.
Key insight:
- The puzzle is solved by shape-level pairing using complementarity, then by marker-driven painting priority during overlay and position cleanup.
"""

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]

    def neighbors4(r: int, c: int):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc

    def component_data(mask: np.ndarray, cells: List[Tuple[int, int]]) -> Dict:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h0, w0 = r1 - r0 + 1, c1 - c0 + 1
        rel = {(r - r0, c - c0) for r, c in cells}
        colors = [int(grid[r, c]) for r, c in cells]
        colors_no_bg = [x for x in colors if x != BACKGROUND]
        cnt = Counter(colors_no_bg)
        normal_color = cnt.most_common(1)[0][0]
        marker_cells = [(r, c) for r, c in cells if grid[r, c] == YELLOW]
        return {
            "cells": cells,
            "rows": set(rows),
            "cols": set(cols),
            "bbox": (r0, r1, c0, c1),
            "h": h0,
            "w": w0,
            "shape": rel,
            "normal_color": normal_color,
            "has_marker": len(marker_cells) > 0,
            "marker_cells": marker_cells,
            "pixels": len(cells),
        }

    def connected_components(g: np.ndarray):
        seen = np.zeros_like(g, dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if g[r, c] == BACKGROUND or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                cells: List[Tuple[int, int]] = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for nr, nc in neighbors4(cr, cc):
                        if not seen[nr, nc] and g[nr, nc] != BACKGROUND:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                comps.append(component_data(g, cells))
        return comps

    h, w = grid.shape
    fg_mask = grid != BACKGROUND
    comps = connected_components(fg_mask)

    n = len(comps)
    used = [False] * n
    pairs = []

    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if used[j]:
                continue
            a, b = comps[i], comps[j]
            if a["h"] != b["h"] or a["w"] != b["w"]:
                continue
            if a["has_marker"] == b["has_marker"]:
                continue
            if len(a["shape"]) + len(b["shape"]) != a["h"] * a["w"]:
                continue
            if a["shape"] & b["shape"]:
                continue
            # Full coverage test by direct set arithmetic.
            if a["shape"] | b["shape"] != {(r, c) for r in range(a["h"]) for c in range(a["w"]) }:
                continue
            pairs.append((i, j))
            used[i] = used[j] = True
            break

    out = np.array(grid, copy=True)

    for i, j in pairs:
        a, b = comps[i], comps[j]
        pri, sec = (a, b) if a["has_marker"] else (b, a)
        r0, _, c0, _ = pri["bbox"]
        H, W = pri["h"], pri["w"]

        # Translate the secondary piece into primary bbox coordinates.
        shift_r = sec["bbox"][0] - r0
        shift_c = sec["bbox"][2] - c0

        secondary_rel_shifted = {(r + shift_r, c + shift_c) for r, c in sec["shape"]}

        # Paint merged rectangle by priority/secondary priority.
        for rr in range(H):
            gr = r0 + rr
            for cc in range(W):
                gc = c0 + cc
                if (rr, cc) in pri["shape"]:
                    out[gr, gc] = pri["normal_color"]
                elif (rr, cc) in secondary_rel_shifted:
                    out[gr, gc] = sec["normal_color"]

        # Clear secondary original location to background.
        for sr, sc in sec["cells"]:
            out[sr, sc] = BACKGROUND

        # Ensure marker pixel is restored to normal color if it exists in priority piece.
        for mr, mc in pri["marker_cells"]:
            out[mr, mc] = pri["normal_color"]

    return out


def _is_connected(mask: np.ndarray) -> bool:
    h, w = mask.shape
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    q = [tuple(cells[0])]
    seen = np.zeros_like(mask, dtype=bool)
    seen[q[0]] = True
    got = 1

    while q:
        r, c = q.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                got += 1
                q.append((nr, nc))

    return got == mask.sum()


def _random_connected_shape(h: int, w: int, target: int, rng: random.Random) -> Optional[np.ndarray]:
    # Random growing polyomino via frontier expansion.
    if target <= 0 or target >= h * w:
        return None

    start = (rng.randrange(h), rng.randrange(w))
    mask = np.zeros((h, w), dtype=bool)
    mask[start] = True
    cells = [start]
    frontier: Set[Tuple[int, int]] = set()

    def push_frontier(r: int, c: int):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    push_frontier(*start)

    while cells.__len__() < target:
        if not frontier:
            return None
        idx = rng.randrange(len(frontier))
        nr, nc = tuple(frontier)[idx]
        frontier.remove((nr, nc))
        mask[nr, nc] = True
        cells.append((nr, nc))
        push_frontier(nr, nc)

    return mask


def _corner_candidates(mask: np.ndarray) -> List[Tuple[int, int]]:
    h, w = mask.shape
    cand = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            missing = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    missing += 1
            if missing >= 2:
                cand.append((r, c))
    return cand


def _generate_complementary_masks(h: int, w: int, rng: random.Random) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    # Returns mask_a, mask_b, marker in A coord, marker in B coord (one will be invalid marker only).
    area = h * w
    tries = 0
    while True:
        tries += 1
        if tries > 400:
            # Relax constraints by changing geometry if no mask found.
            return None
        a_size = rng.randint(2, area - 2)
        mask_a = _random_connected_shape(h, w, a_size, rng)
        if mask_a is None:
            continue
        mask_b = np.logical_not(mask_a)
        if not _is_connected(mask_b):
            continue
        # Both pieces must be connected with boundary-like marker positions.
        if not _corner_candidates(mask_a) or not _corner_candidates(mask_b):
            continue
        return mask_a, mask_b, tuple(rng.choice(_corner_candidates(mask_a))), tuple(rng.choice(_corner_candidates(mask_b)))


def generate_input() -> np.ndarray:
    rng = random.Random()
    palette = [0, 1, 2, 3, 5, 6, 7, 9]

    while True:
        H = rng.randint(12, 30)
        W = rng.randint(12, 30)
        canvas = np.full((H, W), BACKGROUND, dtype=int)
        used = np.zeros((H, W), dtype=bool)
        n_pairs = rng.randint(1, 3)
        pair_shapes = []

        # Generate pair templates (shape + colors + marker orientation).
        ok = True
        for _ in range(n_pairs):
            tries = 0
            pair = None
            while tries < 120:
                tries += 1
                h = rng.randint(2, 8)
                w = rng.randint(2, 8)
                if h * w < 4:
                    continue
                res = _generate_complementary_masks(h, w, rng)
                if res is None or res is False:
                    continue
                mask_a, mask_b, ma, mb = res
                pair = (h, w, mask_a, mask_b, ma, mb)
                break
            if pair is None:
                ok = False
                break

            c1, c2 = rng.sample(palette, 2)
            marker_on_a = rng.choice((True, False))
            pair_shapes.append((pair, (c1, c2, marker_on_a)))

        if not ok:
            continue

        # Place pairs with no overlap in the current canvas.
        placed = []
        for (h, w, mask_a, mask_b, marker_a, marker_b), (c1, c2, marker_on_a) in pair_shapes:
            attempt = 0
            placed_ok = False
            while attempt < 300:
                attempt += 1
                r0 = rng.randint(0, H - h)
                c0 = rng.randint(0, W - w)
                if np.any(used[r0:r0 + h, c0:c0 + w]):
                    continue

                placed_ok = True
                placed.append((r0, c0, h, w, mask_a, mask_b, marker_a, marker_b, c1, c2, marker_on_a))
                used[r0:r0 + h, c0:c0 + w] = True
                break
            if not placed_ok:
                ok = False
                break

        if not ok:
            continue

        for r0, c0, h, w, mask_a, mask_b, marker_a, marker_b, c1, c2, marker_on_a in placed:
            color_a, color_b = (c1, c2)
            # Draw piece A.
            for r in range(h):
                gr = r0 + r
                for c in range(w):
                    if not mask_a[r, c]:
                        continue
                    color = color_a
                    if (r, c) == marker_a and marker_on_a:
                        color = YELLOW
                    canvas[gr, c0 + c] = color

            # Draw piece B.
            for r in range(h):
                gr = r0 + r
                for c in range(w):
                    if not mask_b[r, c]:
                        continue
                    color = color_b
                    if (r, c) == marker_b and not marker_on_a:
                        color = YELLOW
                    canvas[gr, c0 + c] = color

        # Branch diversification: optionally add spare tiny colored noise? currently false.
        # Keep samples clean and unambiguous.
        return canvas


if __name__ == "__main__":
    # Quick smoke check for local execution.
    g = generate_input()
    out = main(g)
    print(g.shape, out.shape)
