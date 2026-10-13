import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

ARROW_COLORS = [RED, BLUE, GREEN, YELLOW, MAGENTA, ORANGE, AZURE]
TRANSFORMS = ["hflip", "vflip", "rot_cw", "rot_ccw"]
CODE_TO_TRANSFORM = {RED: "hflip", BLUE: "vflip", GREEN: "rot_cw", YELLOW: "rot_ccw"}

RIGHT_ARROW = [(0, 0), (0, 1), (0, 2), (-1, 3), (1, 3)]
LEFT_ARROW = [(0, 0), (0, 1), (0, 2), (-1, -1), (1, -1)]
DOWN_ARROW = [(0, 0), (1, 0), (2, 0), (3, -1), (3, 1)]
UP_ARROW = [(0, 0), (1, 0), (2, 0), (-1, -1), (-1, 1)]
ARROW_PATTERNS = {
    (0, 1): RIGHT_ARROW,
    (0, -1): LEFT_ARROW,
    (1, 0): DOWN_ARROW,
    (-1, 0): UP_ARROW,
}

BASE_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
]


def _components(grid, color, mask=None, diagonal=True):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in zip(*np.where(grid == color)):
        if seen[r, c] or (mask is not None and not mask[r, c]):
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        cells = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            steps = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
            if not diagonal:
                steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
            for dr, dc in steps:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    if grid[nr, nc] == color and (mask is None or mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
        comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _legend_bbox(grid):
    cells = list(zip(*np.where(grid == MAROON)))
    if not cells:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 >= 2 and c1 - c0 >= 3:
        return r0, c0, r1, c1
    return None


def _read_legend(grid, box):
    if box is None:
        return {}
    r0, c0, r1, c1 = box
    mapping = {}
    for r in range(r0 + 1, r1):
        color = int(grid[r, c0 + 1]) if c0 + 1 < c1 else BLACK
        code = int(grid[r, c0 + 3]) if c0 + 3 < c1 else BLACK
        if color in ARROW_COLORS and code in CODE_TO_TRANSFORM:
            mapping[color] = CODE_TO_TRANSFORM[code]
    return mapping


def _in_box(pos, box):
    if box is None:
        return False
    r, c = pos
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _normalize(cells):
    r0, c0, _, _ = _bbox(cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _arrow_direction(cells):
    norm = _normalize(cells)
    for direction, pattern in ARROW_PATTERNS.items():
        pr = min(r for r, _ in pattern)
        pc = min(c for _, c in pattern)
        expected = sorted((r - pr, c - pc) for r, c in pattern)
        if norm == expected:
            return direction
    return None


def _mask_from_cells(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _apply_transform(mask, transform):
    if transform == "hflip":
        return np.fliplr(mask)
    if transform == "vflip":
        return np.flipud(mask)
    if transform == "rot_cw":
        return np.rot90(mask, -1)
    if transform == "rot_ccw":
        return np.rot90(mask, 1)
    return mask.copy()


def _ray_target(arrow_cells, direction, targets):
    dr, dc = direction
    ar0, ac0, ar1, ac1 = _bbox(arrow_cells)
    anchor_r = (ar0 + ar1) // 2
    anchor_c = (ac0 + ac1) // 2
    best = None
    for idx, cells in enumerate(targets):
        r0, c0, r1, c1 = _bbox(cells)
        if dr == 0 and r0 <= anchor_r <= r1:
            if dc == 1 and c0 > ac1:
                dist = c0 - ac1
            elif dc == -1 and c1 < ac0:
                dist = ac0 - c1
            else:
                continue
        elif dc == 0 and c0 <= anchor_c <= c1:
            if dr == 1 and r0 > ar1:
                dist = r0 - ar1
            elif dr == -1 and r1 < ar0:
                dist = ar0 - r1
            else:
                continue
        else:
            continue
        if best is None or dist < best[0]:
            best = (dist, idx)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    legend = _legend_bbox(grid)
    mapping = _read_legend(grid, legend)

    targets = _components(grid, GREY, diagonal=False)
    arrows = []
    valid_mask = np.ones(grid.shape, dtype=bool)
    if legend is not None:
        r0, c0, r1, c1 = legend
        valid_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    for color in ARROW_COLORS:
        for cells in _components(grid, color, valid_mask):
            if len(cells) != 5:
                continue
            direction = _arrow_direction(cells)
            if direction is not None:
                hit = _ray_target(cells, direction, targets)
                if hit is not None:
                    dist, target_idx = hit
                    arrows.append((target_idx, dist, color, cells))

    for _, _, _, cells in arrows:
        for r, c in cells:
            out[r, c] = BLACK
    if legend is not None:
        r0, c0, r1, c1 = legend
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    by_target = defaultdict(list)
    for target_idx, dist, color, _cells in arrows:
        by_target[target_idx].append((dist, mapping.get(color, "hflip")))

    for idx, cells in enumerate(targets):
        mask = _mask_from_cells(cells)
        for _dist, transform in sorted(by_target.get(idx, [])):
            mask = _apply_transform(mask, transform)
        r0, c0, r1, c1 = _bbox(cells)
        out[r0 : r1 + 1, c0 : c1 + 1][grid[r0 : r1 + 1, c0 : c1 + 1] == GREY] = BLACK
        mh, mw = mask.shape
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        rr = r0 + max(0, (bh - mh) // 2)
        cc = c0 + max(0, (bw - mw) // 2)
        out[rr : rr + mh, cc : cc + mw][mask] = GREY
    return out


def _draw_legend(grid, mapping):
    h = len(mapping) + 2
    w = 5
    grid[:h, :w] = BLACK
    grid[0, :w] = MAROON
    grid[h - 1, :w] = MAROON
    grid[:h, 0] = MAROON
    grid[:h, w - 1] = MAROON
    trans_to_code = {v: k for k, v in CODE_TO_TRANSFORM.items()}
    for i, (color, transform) in enumerate(mapping.items(), start=1):
        grid[i, 1] = color
        grid[i, 3] = trans_to_code[transform]


def _draw_arrow(grid, r, c, direction, color):
    for dr, dc in ARROW_PATTERNS[direction]:
        grid[r + dr, c + dc] = color


def generate_input() -> np.ndarray:
    n = random.randint(2, 4)
    h = max(16, 12 + n * 5)
    w = random.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(ARROW_COLORS, n)
    transforms = random.choices(TRANSFORMS, k=n)
    mapping = dict(zip(colors, transforms))
    _draw_legend(grid, mapping)

    rows = [8 + i * 5 + random.randint(0, 1) for i in range(n)]
    for i, (row, color) in enumerate(zip(rows, colors)):
        shape = random.choice(BASE_SHAPES)
        tr = row - 1
        tc = random.randint(15, w - 5)
        grid[tr : tr + 3, tc : tc + 3][shape] = GREY
        ac = random.randint(6, max(6, tc - 7))
        _draw_arrow(grid, row, ac, (0, 1), color)
    return grid


if __name__ == "__main__":
    for _ in range(100):
        x = generate_input()
        y = main(x)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
