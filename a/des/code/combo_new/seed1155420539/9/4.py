from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

ARC_MAX = 30
BG = 0
GEN_RETRIES = 400


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    dirs = DIR4 if connectivity == 4 else DIR8
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _mask_from_coords(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _adjacency_score(coords: list[tuple[int, int]], boundary_mask: np.ndarray) -> int:
    h, w = boundary_mask.shape
    score = 0
    for r, c in coords:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and boundary_mask[nr, nc]:
                score += 1
    return score


def _layer_distances(
    shape: tuple[int, int],
    cavity_coords: list[tuple[int, int]],
    boundary_mask: np.ndarray,
) -> np.ndarray:
    h, w = shape
    cavity_mask = _mask_from_coords(shape, cavity_coords)
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in cavity_coords:
        if any(
            0 <= r + dr < h and 0 <= c + dc < w and boundary_mask[r + dr, c + dc]
            for dr, dc in DIR4
        ):
            dist[r, c] = 1
            q.append((r, c))
    while q:
        r, c = q.popleft()
        next_d = dist[r, c] + 1
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and cavity_mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = next_d
                q.append((nr, nc))
    return dist


def _visible_mode(items: list[tuple[int, int]]) -> int:
    counts = Counter(color for _col, color in items)
    best = max(counts.values())
    winners = {color for color, count in counts.items() if count == best}
    for _col, color in items:
        if color in winners:
            return color
    raise AssertionError("visible mode requires at least one item")


def _read_palette_levels(grid: np.ndarray, pillar_components: list[list[tuple[int, int]]]) -> list[int]:
    pillars: list[tuple[int, list[int]]] = []
    for comp in pillar_components:
        cols = {c for _r, c in comp}
        if len(cols) != 1:
            continue
        col = next(iter(cols))
        ordered = sorted(comp, key=lambda rc: rc[0], reverse=True)
        colors = [int(grid[r, c]) for r, c in ordered]
        pillars.append((col, colors))
    assert pillars, "expected at least one palette pillar"
    pillars.sort(key=lambda item: item[0])
    max_height = max(len(colors) for _col, colors in pillars)
    levels: list[int] = []
    for level in range(max_height):
        items = [(col, colors[level]) for col, colors in pillars if level < len(colors)]
        levels.append(_visible_mode(items))
    return levels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a hollow left-side region with inward contour bands determined by right-side color columns.

    Input structure:
    - Background is black.
    - The left side contains one large irregular monochrome closed boundary enclosing background cells.
    - The right side contains several 1-pixel-wide vertical columns whose stacked colors are read from bottom to top.

    Rule:
    - Treat the largest non-background component as the closed boundary and find its enclosed background cavity.
    - Keep the boundary in its original color, then fill cavity distance-1 cells with the mode of the columns' bottom colors, distance-2 cells with the mode of the next colors, and so on.
    - If a cavity layer lies deeper than every available stack level, leave that remaining inner area as background.

    Key insight:
    - The right-side columns form a stacked palette, and the left hollow shape is colored by mapping those palette levels onto successive inward contour layers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h <= ARC_MAX and w <= ARC_MAX

    non_bg_components = _components_from_mask(g != BG, connectivity=8)
    assert len(non_bg_components) >= 2, "expected one closed region and at least one pillar"

    boundary_coords = max(non_bg_components, key=len)
    boundary_mask = _mask_from_coords((h, w), boundary_coords)
    pillar_components = [comp for comp in non_bg_components if comp is not boundary_coords]
    palette_levels = _read_palette_levels(g, pillar_components)

    best_cavity = None
    best_key = None
    for comp in _components_from_mask(g == BG, connectivity=4):
        if _touches_edge(comp, h, w):
            continue
        key = (_adjacency_score(comp, boundary_mask), len(comp))
        if best_key is None or key > best_key:
            best_key = key
            best_cavity = comp
    assert best_cavity is not None, "expected an enclosed cavity inside the boundary"

    dist = _layer_distances((h, w), best_cavity, boundary_mask)
    out = g.copy()
    for r, c in best_cavity:
        d = int(dist[r, c])
        if 1 <= d <= len(palette_levels):
            out[r, c] = palette_levels[d - 1]
        else:
            out[r, c] = BG
    return out


def _build_local_cavity(core_h: int, core_w: int, allowed_sides: set[str]) -> np.ndarray:
    top_room = random.randint(0, 3 if "top" in allowed_sides else 1)
    bottom_room = random.randint(0, 3 if "bottom" in allowed_sides else 1)
    left_room = random.randint(0, 3 if "left" in allowed_sides else 1)
    right_room = random.randint(0, 3 if "right" in allowed_sides else 1)

    h = core_h + top_room + bottom_room
    w = core_w + left_room + right_room
    mask = np.zeros((h, w), dtype=bool)
    r0, c0 = top_room, left_room
    mask[r0 : r0 + core_h, c0 : c0 + core_w] = True

    def add_side(side: str) -> bool:
        if side == "top" and top_room > 0:
            ph = random.randint(1, top_room)
            pw = random.randint(max(2, core_w // 3), core_w)
            start = random.randint(c0, c0 + core_w - pw)
            mask[r0 - ph : r0, start : start + pw] = True
            return True
        if side == "bottom" and bottom_room > 0:
            ph = random.randint(1, bottom_room)
            pw = random.randint(max(2, core_w // 3), core_w)
            start = random.randint(c0, c0 + core_w - pw)
            mask[r0 + core_h : r0 + core_h + ph, start : start + pw] = True
            return True
        if side == "left" and left_room > 0:
            pw = random.randint(1, left_room)
            ph = random.randint(max(2, core_h // 3), core_h)
            start = random.randint(r0, r0 + core_h - ph)
            mask[start : start + ph, c0 - pw : c0] = True
            return True
        if side == "right" and right_room > 0:
            pw = random.randint(1, right_room)
            ph = random.randint(max(2, core_h // 3), core_h)
            start = random.randint(r0, r0 + core_h - ph)
            mask[start : start + ph, c0 + core_w : c0 + core_w + pw] = True
            return True
        return False

    candidates = [side for side in ("top", "bottom", "left", "right") if side in allowed_sides]
    random.shuffle(candidates)
    added = False
    for side in candidates:
        if random.random() < 0.75 and add_side(side):
            added = True
    if not added and candidates:
        random.shuffle(candidates)
        for side in candidates:
            if add_side(side):
                break
    return mask


def _choose_core_dims(h: int, split: int, max_height: int, want_deep_center: bool) -> tuple[int, int, set[str]] | None:
    max_core_h = h - 3
    max_core_w = split - 2
    if max_core_h < 3 or max_core_w < 3:
        return None

    if want_deep_center:
        need = 2 * max_height + 1
        if max_core_h < need or max_core_w < need:
            return None
        core_h = random.randint(need, max_core_h)
        core_w = random.randint(need, max_core_w)
        return core_h, core_w, {"top", "bottom", "left", "right"}

    if random.random() < 0.5:
        limit_w = min(max_core_w, 2 * max_height)
        if limit_w < 3:
            return None
        core_h = random.randint(max(5, max_height + 1), max_core_h)
        core_w = random.randint(3, limit_w)
        return core_h, core_w, {"top", "bottom"}

    limit_h = min(max_core_h, 2 * max_height)
    if limit_h < 3:
        return None
    core_h = random.randint(3, limit_h)
    core_w = random.randint(max(5, max_height + 1), max_core_w)
    return core_h, core_w, {"left", "right"}


def _coords_to_shell(cavity_coords: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    cavity_set = set(cavity_coords)
    shell: set[tuple[int, int]] = set()
    for r, c in cavity_coords:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cavity_set:
                shell.add((nr, nc))
    return list(shell)


def _place_cavity(
    h: int,
    w: int,
    split: int,
    max_height: int,
    want_deep_center: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray] | None:
    chosen = _choose_core_dims(h, split, max_height, want_deep_center)
    if chosen is None:
        return None
    core_h, core_w, allowed_sides = chosen
    local = _build_local_cavity(core_h, core_w, allowed_sides)
    lh, lw = local.shape
    if h - lh - 2 < 1 or split - lw - 1 < 1:
        return None

    top = random.randint(1, h - lh - 2)
    left = random.randint(1, split - lw - 1)
    local_coords = [(int(r), int(c)) for r, c in np.argwhere(local)]
    cavity = [(top + r, left + c) for r, c in local_coords]
    shell = _coords_to_shell(cavity, h, w)
    if not shell:
        return None

    shell_cols = [c for _r, c in shell]
    if min(shell_cols) < 0 or max(shell_cols) >= split:
        return None

    shell_mask = _mask_from_coords((h, w), shell)
    shell_components = _components_from_mask(shell_mask, connectivity=8)
    if len(shell_components) != 1:
        return None

    cavity_components = [comp for comp in _components_from_mask(_mask_from_coords((h, w), cavity), connectivity=4)]
    if len(cavity_components) != 1:
        return None

    return cavity, shell, shell_mask


def _build_pillars(
    n_pillars: int,
    max_height: int,
    colors: list[int],
) -> tuple[list[list[int]], list[int], list[int]] | None:
    target_levels = random.sample(colors, max_height)
    heights = [random.randint(1, max_height) for _ in range(n_pillars - 1)] + [max_height]
    random.shuffle(heights)
    pillar_colors = [[] for _ in range(n_pillars)]

    for level in range(max_height):
        active = [i for i, height in enumerate(heights) if height > level]
        if not active:
            return None
        target = target_levels[level]
        majority_count = len(active) // 2 + 1
        majority = set(random.sample(active, majority_count))
        for idx in active:
            used = set(pillar_colors[idx])
            if idx in majority:
                color = target
            else:
                options = [c for c in colors if c != target and c not in used]
                if not options:
                    return None
                color = random.choice(options)
            if color in used:
                return None
            pillar_colors[idx].append(color)
    return pillar_colors, heights, target_levels


def _pillar_columns(split: int, w: int, n_pillars: int) -> list[int] | None:
    start = split + 1
    end = w - 2
    if end - start + 1 < 2 * n_pillars - 1:
        return None
    cols: list[int] = []
    cur_min = start
    remaining = n_pillars
    for _ in range(n_pillars):
        max_col = end - 2 * (remaining - 1)
        if cur_min > max_col:
            return None
        col = random.randint(cur_min, max_col)
        cols.append(col)
        cur_min = col + 2
        remaining -= 1
    return cols


def _expected_output(
    grid: np.ndarray,
    cavity: list[tuple[int, int]],
    boundary_mask: np.ndarray,
    level_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    dist = _layer_distances(grid.shape, cavity, boundary_mask)
    for r, c in cavity:
        d = int(dist[r, c])
        if 1 <= d <= len(level_colors):
            out[r, c] = level_colors[d - 1]
        else:
            out[r, c] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    want_deep_center = random.random() < 0.5
    max_height = random.randint(2, 5)

    if want_deep_center:
        h_min = max(16, 2 * max_height + 4)
        w_min = max(22, 4 * max_height + 6)
    else:
        h_min = 16
        w_min = 20
    h = random.randint(h_min, ARC_MAX)
    w = random.randint(w_min, ARC_MAX)
    split = w // 2

    max_pillars = min(6, (w - split - 1) // 2)
    if max_pillars < 3:
        return None
    n_pillars = random.randint(3, max_pillars)

    region_color = random.randint(1, 9)
    available_colors = [c for c in range(1, 10) if c != region_color]
    built = _build_pillars(n_pillars, max_height, available_colors)
    if built is None:
        return None
    pillar_colors, heights, target_levels = built

    placed = _place_cavity(h, w, split, max_height, want_deep_center)
    if placed is None:
        return None
    cavity, shell, shell_mask = placed

    cols = _pillar_columns(split, w, n_pillars)
    if cols is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in shell:
        grid[r, c] = region_color

    for idx, col in enumerate(cols):
        height = heights[idx]
        top = random.randint(1, h - height - 1)
        for level, color in enumerate(pillar_colors[idx]):
            row = top + height - 1 - level
            grid[row, col] = color

    if int((grid != BG).sum()) == 0 or int((grid == BG).sum()) <= int((grid != BG).sum()):
        return None

    dist = _layer_distances((h, w), cavity, shell_mask)
    max_dist = int(dist.max())
    if want_deep_center != (max_dist > max_height):
        return None

    expected = _expected_output(grid, cavity, shell_mask, target_levels)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input after repeated attempts")
