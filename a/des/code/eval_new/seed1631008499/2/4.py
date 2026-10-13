import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PATTERN_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ROTATION_STEPS = {"up": 0, "right": 1, "down": 2, "left": 3}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _side_mid_colors(sub: np.ndarray):
    if sub.shape[0] != sub.shape[1] or sub.shape[0] < 4:
        return None
    sides = {
        "up": sub[0, 1:-1],
        "down": sub[-1, 1:-1],
        "left": sub[1:-1, 0],
        "right": sub[1:-1, -1],
    }
    colors = {}
    for name, arr in sides.items():
        vals = np.unique(arr)
        if len(vals) != 1:
            return None
        colors[name] = int(vals[0])
    return colors


def _extract_canvases(grid: np.ndarray):
    candidates = []
    for cells in _components(grid != BG):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 != c1 - c0:
            continue
        side = r1 - r0 + 1
        if side < 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        ring = np.zeros_like(sub, dtype=bool)
        ring[0, :] = True
        ring[-1, :] = True
        ring[:, 0] = True
        ring[:, -1] = True
        if np.any(sub[ring] == BG):
            continue
        if not np.any(sub[1:-1, 1:-1] == BG):
            continue
        side_colors = _side_mid_colors(sub)
        if side_colors is None:
            continue
        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "side_colors": side_colors,
                "sub": sub,
            }
        )

    if not candidates:
        return []

    grey = Counter(
        color
        for cand in candidates
        for color in cand["side_colors"].values()
    ).most_common(1)[0][0]

    canvases = []
    for cand in candidates:
        exit_sides = [
            name
            for name, color in cand["side_colors"].items()
            if color != grey
        ]
        if len(exit_sides) != 1:
            continue
        direction = exit_sides[0]
        r0, c0, r1, c1 = cand["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        canvases.append(
            {
                "bbox": cand["bbox"],
                "direction": direction,
                "exit_color": int(cand["side_colors"][direction]),
                "interior": interior,
            }
        )

    if not canvases:
        return []

    inner_sizes = [canvas["interior"].shape[0] for canvas in canvases]
    common_inner = Counter(inner_sizes).most_common(1)[0][0]
    canvases = [
        canvas
        for canvas in canvases
        if canvas["interior"].shape == (common_inner, common_inner)
    ]
    return canvases


def _find_neighbor(canvases, idx: int):
    src = canvases[idx]
    sr0, sc0, sr1, sc1 = src["bbox"]
    srow = sr0 + sr1
    scol = sc0 + sc1
    direction = src["direction"]
    best = None
    best_gap = None

    for j, other in enumerate(canvases):
        if j == idx:
            continue
        or0, oc0, or1, oc1 = other["bbox"]
        orow = or0 + or1
        ocol = oc0 + oc1
        if direction == "up" and ocol == scol and or1 < sr0:
            gap = sr0 - or1
        elif direction == "down" and ocol == scol and or0 > sr1:
            gap = or0 - sr1
        elif direction == "left" and orow == srow and oc1 < sc0:
            gap = sc0 - oc1
        elif direction == "right" and orow == srow and oc0 > sc1:
            gap = oc0 - sc1
        else:
            continue
        if best_gap is None or gap < best_gap:
            best = j
            best_gap = gap
    return best


def _rotate_for_direction(arr: np.ndarray, direction: str):
    return np.rot90(arr, k=(-ROTATION_STEPS[direction]) % 4)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    canvases = _extract_canvases(g)
    out = g.copy()

    for idx, canvas in enumerate(canvases):
        if not np.any(canvas["interior"] != BG):
            continue
        target_idx = _find_neighbor(canvases, idx)
        if target_idx is None:
            continue
        rotated = _rotate_for_direction(canvas["interior"], canvas["direction"])
        tr0, tc0, tr1, tc1 = canvases[target_idx]["bbox"]
        target = out[tr0 + 1 : tr1, tc0 + 1 : tc1]
        incoming = rotated != BG
        occupied = target != BG
        target[incoming & occupied] = MAROON
        target[incoming & ~occupied] = rotated[incoming & ~occupied]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy interior motifs from framed canvases into the adjacent canvas indicated by each colored border side.

    Input structure:
    - bg = black background.
    - The grid contains several equal-sized square canvases with 1-pixel borders; most border cells are grey.
    - In each canvas, exactly one border side is recolored to a non-grey exit color, and some canvas interiors contain a same-colored motif while others are empty.

    Rule:
    - Detect each square canvas and read the colored border side as a direction: top=up, right=right, bottom=down, left=left.
    - For every canvas whose interior is non-empty, copy its entire interior pattern once into the nearest canvas in that direction.
    - Rotate the copied pattern by the exit-side rule before placing it: up=0 degrees, right=90 degrees clockwise, down=180 degrees, left=270 degrees clockwise.
    - Keep source canvases unchanged; when copied pixels land on already occupied interior pixels, recolor those overlap cells to maroon.

    Key insight:
    - The colored edge is both a routing arrow and a rotation code, while overlap inside target canvases is revealed by the special maroon collision color.
    """
    return _transform(grid)


def _valid_dirs(r: int, c: int, rows: int, cols: int):
    dirs = []
    if r > 0:
        dirs.append("up")
    if c < cols - 1:
        dirs.append("right")
    if r < rows - 1:
        dirs.append("down")
    if c > 0:
        dirs.append("left")
    return dirs


def _dir_from_to(src, dst):
    sr, sc = src
    tr, tc = dst
    if tr == sr - 1 and tc == sc:
        return "up"
    if tr == sr + 1 and tc == sc:
        return "down"
    if tr == sr and tc == sc + 1:
        return "right"
    if tr == sr and tc == sc - 1:
        return "left"
    raise ValueError("cells are not adjacent")


def _random_pattern_mask(inner: int, rng: random.Random):
    area = inner - 2
    max_cells = min(area * area, 8)
    min_cells = min(3, max_cells)
    for _ in range(40):
        target = rng.randint(min_cells, max_cells)
        start = (rng.randrange(area), rng.randrange(area))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < area and 0 <= nc < area and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) < min_cells:
            continue
        mask = np.zeros((inner, inner), dtype=bool)
        for r, c in cells:
            mask[r + 1, c + 1] = True
        return mask
    mask = np.zeros((inner, inner), dtype=bool)
    mask[1:3, 1:3] = np.array([[1, 1], [1, 0]], dtype=bool)
    return mask


def _pattern_with_overlap(inner: int, rng: random.Random, existing: np.ndarray):
    for _ in range(60):
        cand = _random_pattern_mask(inner, rng)
        if np.any(cand & existing):
            return cand
    rr, cc = np.argwhere(existing)[0]
    cand = _random_pattern_mask(inner, rng)
    cand[rr, cc] = True
    return cand


def _inverse_rotate(mask: np.ndarray, direction: str):
    return np.rot90(mask, k=ROTATION_STEPS[direction])


def _render_instance(
    rows: int,
    cols: int,
    inner: int,
    gap: int,
    margins,
    directions,
    colors,
    patterns,
):
    top, left, bottom, right = margins
    frame = inner + 2
    h = top + rows * frame + (rows - 1) * gap + bottom
    w = left + cols * frame + (cols - 1) * gap + right
    grid = np.full((h, w), BG, dtype=int)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            r0 = top + r * (frame + gap)
            c0 = left + c * (frame + gap)
            r1 = r0 + frame - 1
            c1 = c0 + frame - 1
            grid[r0 : r1 + 1, c0] = GREY
            grid[r0 : r1 + 1, c1] = GREY
            grid[r0, c0 : c1 + 1] = GREY
            grid[r1, c0 : c1 + 1] = GREY

            color = colors[idx]
            direction = directions[idx]
            if direction == "up":
                grid[r0, c0 + 1 : c1] = color
            elif direction == "down":
                grid[r1, c0 + 1 : c1] = color
            elif direction == "left":
                grid[r0 + 1 : r1, c0] = color
            else:
                grid[r0 + 1 : r1, c1] = color

            grid[r0, c0] = GREY
            grid[r0, c1] = GREY
            grid[r1, c0] = GREY
            grid[r1, c1] = GREY

            mask = patterns.get(idx)
            if mask is not None:
                interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                interior[mask] = color

    return grid


def _choose_layout(rng: random.Random):
    for _ in range(80):
        rows = rng.choice([2, 2, 3])
        cols = rng.choice([2, 3, 3])
        inner = rng.randint(4, 6)
        gap = rng.randint(1, 2)
        margins = (
            rng.randint(1, 3),
            rng.randint(1, 3),
            rng.randint(1, 3),
            rng.randint(1, 3),
        )
        frame = inner + 2
        h = margins[0] + rows * frame + (rows - 1) * gap + margins[2]
        w = margins[1] + cols * frame + (cols - 1) * gap + margins[3]
        if h <= 30 and w <= 30:
            return rows, cols, inner, gap, margins
    return None


def _initial_directions(rows: int, cols: int, rng: random.Random):
    directions = {}
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            directions[idx] = rng.choice(_valid_dirs(r, c, rows, cols))
    return directions


def _greedy_non_overlap_sources(target_of, count: int, n: int, rng: random.Random):
    order = list(range(n))
    rng.shuffle(order)
    chosen = []
    used_targets = set()
    for idx in order:
        tgt = target_of[idx]
        if tgt in used_targets or tgt in chosen or idx in used_targets:
            continue
        chosen.append(idx)
        used_targets.add(tgt)
        if len(chosen) >= count:
            break
    return chosen


def _gen_attempt(rng: random.Random):
    layout = _choose_layout(rng)
    if layout is None:
        return None
    rows, cols, inner, gap, margins = layout
    n = rows * cols

    directions = _initial_directions(rows, cols, rng)
    positions = {idx: divmod(idx, cols) for idx in range(n)}
    want_overlap = rng.random() < 0.5

    overlap_target = None
    overlap_sources = []
    if want_overlap:
        target_candidates = list(range(n))
        rng.shuffle(target_candidates)
        for tgt in target_candidates:
            tr, tc = positions[tgt]
            neighbors = []
            for nr, nc in ((tr - 1, tc), (tr + 1, tc), (tr, tc - 1), (tr, tc + 1)):
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(nr * cols + nc)
            if not neighbors:
                continue
            overlap_target = tgt
            rng.shuffle(neighbors)
            overlap_sources = [neighbors[0]]
            if len(neighbors) > 1 and rng.random() < 0.5:
                overlap_sources.append(neighbors[1])
            for src in overlap_sources:
                directions[src] = _dir_from_to(positions[src], positions[tgt])
            break
        if overlap_target is None:
            return None

    if len(set(directions.values())) < 2:
        return None

    target_of = {}
    for idx in range(n):
        r, c = positions[idx]
        dr, dc = DIRS[directions[idx]]
        nr, nc = r + dr, c + dc
        target_of[idx] = nr * cols + nc

    colors = {idx: rng.choice(PATTERN_COLORS) for idx in range(n)}
    if overlap_target is not None:
        special = [overlap_target] + overlap_sources
        special_colors = rng.sample(PATTERN_COLORS, k=len(set(special)))
        for idx, color in zip(dict.fromkeys(special), special_colors):
            colors[idx] = color

    patterns = {}
    if want_overlap:
        base = _random_pattern_mask(inner, rng)
        patterns[overlap_target] = base
        occupied = base.copy()
        for src in overlap_sources:
            transferred = _pattern_with_overlap(inner, rng, occupied)
            patterns[src] = _inverse_rotate(transferred, directions[src])
            occupied |= transferred
        extra_candidates = [
            idx
            for idx in range(n)
            if idx not in patterns and target_of[idx] not in patterns and target_of[idx] != overlap_target
        ]
        rng.shuffle(extra_candidates)
        for idx in extra_candidates[: rng.randint(0, max(0, min(2, len(extra_candidates))))]:
            patterns[idx] = _random_pattern_mask(inner, rng)
    else:
        src_count = rng.randint(1, max(1, min(3, n - 1)))
        sources = _greedy_non_overlap_sources(target_of, src_count, n, rng)
        if not sources:
            return None
        for idx in sources:
            patterns[idx] = _random_pattern_mask(inner, rng)

    if not patterns:
        return None
    if len(patterns) == n:
        return None

    grid = _render_instance(rows, cols, inner, gap, margins, directions, colors, patterns)
    canvases = _extract_canvases(grid)
    if len(canvases) != n:
        return None

    out = _transform(grid)
    has_maroon = bool(np.any(out == MAROON))
    if want_overlap != has_maroon:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
