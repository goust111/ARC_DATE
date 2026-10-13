from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

BLACK = 0
GREY = 5
VISIBLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
LEGEND_CORNERS = ("TL", "TR", "BL", "BR")
GEN_RETRIES = 500


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _find_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], tuple[int, int, int, int]]:
    h, w = grid.shape
    mask = (grid != BLACK) & (grid != GREY)
    best: tuple[int, tuple[int, int, int, int]] | None = None

    for comp in _components(mask, connectivity=4):
        if not _touches_border(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue
        if c1 <= c0:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(comp):
            continue
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))

    assert best is not None, "expected one solid corner legend rectangle"
    r0, c0, r1, c1 = best[1]
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]
    mapping: dict[int, np.ndarray] = {}
    for row in legend:
        label = int(row[0])
        pattern = np.asarray(row[1:], dtype=int)
        assert pattern.size >= 1, "legend rows need at least one pattern cell"
        assert label not in mapping, "legend labels must be unique"
        mapping[label] = pattern
    return mapping, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-enclosed region with the repeating legend pattern keyed by its singleton marker color.

    Input structure:
    - The background is black and one corner contains a solid legend rectangle whose rows start with a color label followed by a horizontal pattern.
    - Grey pixels form separate closed borders around irregular regions elsewhere in the grid.
    - Each enclosed region contains black interior cells and exactly one non-black, non-grey marker pixel.

    Rule:
    - Keep the grey borders and the corner legend unchanged.
    - For each connected component of non-grey cells that does not touch the grid border, read its unique marker color and use it to select the matching legend row.
    - Overwrite every cell in that region row by row with the chosen pattern, repeating horizontally within a row and restarting from the first pattern cell on each new row.

    Key insight:
    - The singleton marker is not part of the final picture; it is a lookup key that chooses which legend pattern should be masked into that closed region.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, _legend_bbox = _find_legend(g)
    out = g.copy()

    for comp in _components(g != GREY, connectivity=4):
        if _touches_border(comp, h, w):
            continue
        markers = [(r, c) for r, c in comp if int(g[r, c]) != BLACK]
        if not markers:
            continue
        assert len(markers) == 1, "each enclosed region must contain exactly one marker"
        marker_color = int(g[markers[0]])
        assert marker_color in mapping, "every marker color must appear in the legend"
        pattern = mapping[marker_color]

        by_row: dict[int, list[int]] = defaultdict(list)
        for r, c in comp:
            by_row[int(r)].append(int(c))
        for r in sorted(by_row):
            cols = sorted(by_row[r])
            for idx, c in enumerate(cols):
                out[r, c] = int(pattern[idx % len(pattern)])

    return out


def _make_pattern(label: int, width: int) -> np.ndarray:
    pool = [c for c in VISIBLE_COLORS if c != int(label)]
    while True:
        if width >= 3 and random.random() < 0.5:
            a, b, c = random.sample(pool, 3)
            motif = random.choice(
                [
                    [a, b, c],
                    [a, c, b],
                    [a, b, a, c],
                    [a, c, a, b],
                ]
            )
        else:
            a, b = random.sample(pool, 2)
            motif = random.choice(
                [
                    [a, b],
                    [a, a, b],
                    [a, b, b],
                    [a, b, a],
                    [b, a, b],
                ]
            )
        pattern = np.array([motif[i % len(motif)] for i in range(width)], dtype=int)
        if len(set(map(int, pattern))) >= 2:
            return pattern


def _make_legend(labels: list[int], width: int) -> np.ndarray:
    rows = []
    for label in random.sample(labels, len(labels)):
        pattern = _make_pattern(int(label), int(width))
        rows.append(np.concatenate(([int(label)], pattern)))
    return np.array(rows, dtype=int)


def _make_region_patch(marker_color: int) -> np.ndarray:
    for _ in range(100):
        inner_h = random.randint(3, 6)
        inner_w = random.randint(4, 8)
        spans: list[tuple[int, int]] = []
        width0 = random.randint(max(2, inner_w // 2), inner_w)
        left0 = random.randint(0, inner_w - width0)
        spans.append((left0, left0 + width0 - 1))

        for _row in range(1, inner_h):
            prev_l, prev_r = spans[-1]
            l = max(0, min(inner_w - 2, prev_l + random.choice([-1, 0, 1])))
            r = max(l + 1, min(inner_w - 1, prev_r + random.choice([-1, 0, 1])))
            if random.random() < 0.35 and l > 0:
                l -= 1
            if random.random() < 0.35 and r < inner_w - 1:
                r += 1
            if l > prev_r:
                l = prev_r
            if r < prev_l:
                r = prev_l
            if r <= l:
                r = min(inner_w - 1, l + 1)
                l = max(0, r - 1)
            spans.append((l, r))

        body = np.zeros((inner_h, inner_w), dtype=bool)
        for r, (left, right) in enumerate(spans):
            body[r, left : right + 1] = True

        if bool(np.all(body)):
            continue
        if len({span for span in spans}) == 1:
            continue
        if random.random() < 0.5:
            body = np.fliplr(body)
        if random.random() < 0.5:
            body = np.flipud(body)

        interior = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
        interior[1:-1, 1:-1] = body
        boundary = np.zeros_like(interior)
        for r, c in np.argwhere(interior):
            rr = int(r)
            cc = int(c)
            for dr, dc in NEIGHBORS8:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1] and not bool(interior[nr, nc]):
                    boundary[nr, nc] = True

        patch = np.full(interior.shape, BLACK, dtype=int)
        patch[boundary] = GREY
        cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        marker_r, marker_c = random.choice(cells)
        patch[marker_r, marker_c] = int(marker_color)
        return patch

    raise RuntimeError("failed to build a non-rectangular region patch")


def _legend_bbox(height: int, width: int, legend_shape: tuple[int, int], corner: str) -> tuple[int, int, int, int]:
    lh, lw = map(int, legend_shape)
    if corner == "TL":
        r0, c0 = 0, 0
    elif corner == "TR":
        r0, c0 = 0, width - lw
    elif corner == "BL":
        r0, c0 = height - lh, 0
    else:
        r0, c0 = height - lh, width - lw
    return r0, c0, r0 + lh - 1, c0 + lw - 1


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded(rect: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, rect[0] - pad),
        max(0, rect[1] - pad),
        min(h - 1, rect[2] + pad),
        min(w - 1, rect[3] + pad),
    )


def _place_patches(
    height: int,
    width: int,
    patches: list[np.ndarray],
    legend_rect: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    occupied = [legend_rect]
    placements: list[tuple[int, int]] = [(-1, -1)] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        positions: list[tuple[int, int]] = []
        for r0 in range(1, height - ph):
            for c0 in range(1, width - pw):
                rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_intersects(rect, _expanded(other, 1, height, width)) for other in occupied):
                    continue
                positions.append((r0, c0))
        if not positions:
            return None
        placements[idx] = random.choice(positions)
        r0, c0 = placements[idx]
        occupied.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

    return placements


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    window = grid[top : top + ph, left : left + pw]
    non_black = patch != BLACK
    window[non_black] = patch[non_black]


def _build_candidate() -> np.ndarray | None:
    label_count = random.randint(2, 4)
    labels = random.sample(VISIBLE_COLORS, label_count)
    region_count = random.randint(label_count, min(4, label_count + 1))
    region_labels = labels + [random.choice(labels) for _ in range(region_count - label_count)]
    random.shuffle(region_labels)

    pattern_width = random.randint(2, 5)
    legend = _make_legend(labels, pattern_width)
    patches = [_make_region_patch(color) for color in region_labels]

    min_h = max(12, legend.shape[0] + 4, max(p.shape[0] for p in patches) + 3)
    min_w = max(14, legend.shape[1] + 4, max(p.shape[1] for p in patches) + 3)
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    corner = random.choice(LEGEND_CORNERS)
    legend_rect = _legend_bbox(height, width, legend.shape, corner)
    placements = _place_patches(height, width, patches, legend_rect)
    if placements is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    lr0, lc0, lr1, lc1 = legend_rect
    grid[lr0 : lr1 + 1, lc0 : lc1 + 1] = legend
    for patch, (r0, c0) in zip(patches, placements, strict=False):
        _paint_patch(grid, patch, int(r0), int(c0))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
