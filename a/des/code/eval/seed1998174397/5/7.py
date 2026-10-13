import random
from collections import Counter

import numpy as np


BLACK = 0
ORANGE = 7
COLORS = [1, 2, 3, 4, 5, 6, 8, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _marker_direction(markers: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    matches = []
    for mr, mc in markers:
        if mr == r0 - 1 and c0 <= mc <= c1:
            matches.append(("up", (mr, mc)))
        elif mr == r1 + 1 and c0 <= mc <= c1:
            matches.append(("down", (mr, mc)))
        elif mc == c0 - 1 and r0 <= mr <= r1:
            matches.append(("left", (mr, mc)))
        elif mc == c1 + 1 and r0 <= mr <= r1:
            matches.append(("right", (mr, mc)))
    assert len(matches) == 1, "each region needs exactly one side marker"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marked multicolor region by its directional majority-color projection.
    """
    g = np.asarray(grid, dtype=int)
    markers = [(int(r), int(c)) for r, c in np.argwhere(g == ORANGE)]
    comps = _components((g != BLACK) & (g != ORANGE))
    out = g.copy()

    projections: list[tuple[int, int, int]] = []
    erase_markers = []
    for comp in comps:
        bbox = _bbox(comp)
        direction, marker = _marker_direction(markers, bbox)
        erase_markers.append(marker)
        counts = Counter(int(g[r, c]) for r, c in comp)
        majority, top_count = counts.most_common(1)[0]
        assert list(counts.values()).count(top_count) == 1, "majority color must be unique"

        for r, c in comp:
            out[r, c] = BLACK
        if direction in ("up", "down"):
            by_col: dict[int, list[int]] = {}
            for r, c in comp:
                by_col.setdefault(c, []).append(r)
            for c, rows in by_col.items():
                rr = min(rows) if direction == "up" else max(rows)
                projections.append((rr, c, majority))
        else:
            by_row: dict[int, list[int]] = {}
            for r, c in comp:
                by_row.setdefault(r, []).append(c)
            for r, cols in by_row.items():
                cc = min(cols) if direction == "left" else max(cols)
                projections.append((r, cc, majority))

    for r, c in erase_markers:
        out[r, c] = BLACK
    for r, c, color in projections:
        out[r, c] = int(color)
    return out


def _random_mask() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    target = random.randint(max(6, h + w - 2), min(h * w, 15))
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _place_shape(blocked: np.ndarray, mask: np.ndarray, direction: str) -> tuple[int, int, tuple[int, int]] | None:
    h, w = blocked.shape
    mh, mw = mask.shape
    positions = [(r, c) for r in range(2, h - mh - 2) for c in range(2, w - mw - 2)]
    random.shuffle(positions)
    for r0, c0 in positions:
        marker = {
            "up": (r0 - 1, c0 + mw // 2),
            "down": (r0 + mh, c0 + mw // 2),
            "left": (r0 + mh // 2, c0 - 1),
            "right": (r0 + mh // 2, c0 + mw),
        }[direction]
        region = blocked[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1]
        if np.any(region):
            continue
        mr, mc = marker
        if blocked[mr, mc]:
            continue
        blocked[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1] = True
        return r0, c0, marker
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        height = random.randint(18, 26)
        width = random.randint(18, 28)
        grid = np.full((height, width), BLACK, dtype=int)
        blocked = np.zeros_like(grid, dtype=bool)
        count = random.randint(3, 5)
        directions = ["up", "down", "left", "right"]
        random.shuffle(directions)

        ok = True
        for i in range(count):
            direction = directions[i % 4]
            mask = _random_mask()
            placed = _place_shape(blocked, mask, direction)
            if placed is None:
                ok = False
                break
            r0, c0, marker = placed
            cells = list(map(tuple, np.argwhere(mask)))
            palette = random.sample(COLORS, random.randint(2, 3))
            majority = palette[0]
            majority_count = len(cells) // 2 + 1
            random.shuffle(cells)
            for j, (rr, cc) in enumerate(cells):
                color = majority if j < majority_count else random.choice(palette[1:])
                grid[r0 + rr, c0 + cc] = color
            grid[marker] = ORANGE
        if not ok:
            continue
        result = main(grid)
        assert isinstance(result, np.ndarray)
        return grid
    raise RuntimeError("failed to generate projection input")
