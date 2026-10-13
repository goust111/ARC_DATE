import random
from collections import Counter, deque

import numpy as np


BG = 0
CTRL = 2
SPRITE = 5
MARKER_POOL = (1, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _component_areas(mask: np.ndarray) -> list[int]:
    return [int(coords.shape[0]) for coords in _components_from_mask(mask)]


def _border_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    pts = {(0, c) for c in range(w)}
    pts |= {(h - 1, c) for c in range(w)}
    pts |= {(r, 0) for r in range(h)}
    pts |= {(r, w - 1) for r in range(h)}
    return sorted(pts)


def _pick_sprite_color(grid: np.ndarray, bg: int, ctrl: int) -> int:
    colors = sorted(set(map(int, grid.ravel())) - {bg, ctrl})
    best = None
    best_score = None
    for color in colors:
        comps = _components_from_mask(grid == color)
        if not comps:
            continue
        largest = max(int(comp.shape[0]) for comp in comps)
        total = int(np.count_nonzero(grid == color))
        score = (largest, total)
        if best_score is None or score > best_score:
            best = int(color)
            best_score = score
    if best is None:
        raise AssertionError("could not identify the sprite color")
    return best


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)

    ctrl_pts = [(r, c) for (r, c) in _border_positions(g.shape) if int(g[r, c]) == CTRL]
    assert len(ctrl_pts) == 2, "expected exactly two border control pixels"

    horizontal = [pt for pt in ctrl_pts if pt[1] in (0, w - 1)]
    vertical = [pt for pt in ctrl_pts if pt[0] in (0, h - 1)]
    assert len(horizontal) == 1 and len(vertical) == 1, "expected one row control and one column control"

    h_row, h_col = horizontal[0]
    v_row, v_col = vertical[0]
    h_side = "L" if h_col == 0 else "R"
    v_side = "T" if v_row == 0 else "B"

    sprite_color = _pick_sprite_color(g, bg, CTRL)
    sprite_mask = g == sprite_color
    sprite_components = _components_from_mask(sprite_mask)
    assert len(sprite_components) == 1, "sprite must be one 4-connected component"

    marker_colors = sorted(set(map(int, g.ravel())) - {bg, CTRL, sprite_color})
    assert len(marker_colors) == 3, "expected exactly three marker colors"
    marker_positions = [(r, c) for r in range(h) for c in range(w) if int(g[r, c]) in marker_colors]

    return {
        "grid": g,
        "shape": g.shape,
        "bg": int(bg),
        "sprite_color": int(sprite_color),
        "sprite_mask": sprite_mask,
        "marker_colors": marker_colors,
        "marker_positions": marker_positions,
        "h_row": int(h_row),
        "v_col": int(v_col),
        "h_side": h_side,
        "v_side": v_side,
        "ctrl_pts": ctrl_pts,
    }


def _quadrant_label(r: int, c: int, h_row: int, v_col: int, h_side: str, v_side: str) -> str:
    vert = "T" if (r < h_row or (r == h_row and v_side == "T")) else "B"
    horiz = "L" if (c < v_col or (c == v_col and h_side == "L")) else "R"
    return vert + horiz


def _fragment_components(sprite_mask: np.ndarray, h_row: int, v_col: int, h_side: str, v_side: str) -> list[dict]:
    label_masks = {
        "TL": np.zeros_like(sprite_mask, dtype=bool),
        "TR": np.zeros_like(sprite_mask, dtype=bool),
        "BL": np.zeros_like(sprite_mask, dtype=bool),
        "BR": np.zeros_like(sprite_mask, dtype=bool),
    }
    for r, c in np.argwhere(sprite_mask):
        label = _quadrant_label(int(r), int(c), h_row, v_col, h_side, v_side)
        label_masks[label][int(r), int(c)] = True

    fragments = []
    for label, mask in label_masks.items():
        for coords in _components_from_mask(mask):
            fragments.append({"label": label, "coords": coords, "area": int(coords.shape[0])})
    return fragments


def _draw_rays(out: np.ndarray, sprite_mask: np.ndarray, h_row: int, v_col: int, h_side: str, v_side: str) -> None:
    h, w = out.shape
    row_hits = [int(c) for c in np.where(sprite_mask[h_row])[0]]
    col_hits = [int(r) for r in np.where(sprite_mask[:, v_col])[0]]
    assert row_hits and col_hits, "control rays must intersect the sprite"

    if h_side == "L":
        stop = max(row_hits)
        for c in range(0, stop + 1):
            if not sprite_mask[h_row, c]:
                out[h_row, c] = CTRL
    else:
        stop = min(row_hits)
        for c in range(w - 1, stop - 1, -1):
            if not sprite_mask[h_row, c]:
                out[h_row, c] = CTRL

    if v_side == "T":
        stop = max(col_hits)
        for r in range(0, stop + 1):
            if not sprite_mask[r, v_col]:
                out[r, v_col] = CTRL
    else:
        stop = min(col_hits)
        for r in range(h - 1, stop - 1, -1):
            if not sprite_mask[r, v_col]:
                out[r, v_col] = CTRL


def _solve(grid: np.ndarray, strict: bool = True) -> np.ndarray:
    info = _parse_input(grid)
    g = info["grid"]
    bg = info["bg"]
    sprite_mask = info["sprite_mask"]
    marker_colors = info["marker_colors"]
    h_row = info["h_row"]
    v_col = info["v_col"]
    h_side = info["h_side"]
    v_side = info["v_side"]

    fragments = _fragment_components(sprite_mask, h_row, v_col, h_side, v_side)
    if strict:
        assert len(fragments) == 3, "the two rays must produce exactly three fragments"
        areas = [frag["area"] for frag in fragments]
        assert len(set(areas)) == 3, "fragment areas must be pairwise distinct"
    else:
        if len(fragments) != 3:
            raise AssertionError("expected exactly three fragments")

    color_counts = {color: int(np.count_nonzero(g == color)) for color in marker_colors}
    assert len(set(color_counts.values())) == 3, "marker frequencies must be pairwise distinct"

    ordered_frags = sorted(fragments, key=lambda frag: (-frag["area"], frag["label"]))
    ordered_colors = sorted(marker_colors, key=lambda color: (-color_counts[color], color))

    out = np.full_like(g, bg)
    _draw_rays(out, sprite_mask, h_row, v_col, h_side, v_side)
    for r, c in info["ctrl_pts"]:
        out[r, c] = CTRL
    for frag, color in zip(ordered_frags, ordered_colors):
        coords = frag["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the three ray-cut sprite fragments by matching fragment area rank to scattered marker-frequency rank.

    Input structure:
    - The background is black, with a single irregular foreground sprite and two red control pixels on different grid edges.
    - One control pixel lies on the left or right edge and defines a horizontal cut row; the other lies on the top or bottom edge and defines a vertical cut column.
    - Three other colors appear only as scattered single-pixel markers away from the sprite, and their frequencies are pairwise distinct.

    Rule:
    - Project each red control pixel inward along its row or column, painting red only on background cells until the ray reaches the farthest sprite pixel hit on that line.
    - Use the cut row and cut column to split the sprite into three 4-connected fragments; cut-line pixels are assigned to the control-facing side, so the crossing pixel belongs to the common control corner.
    - Rank the three fragments by area from largest to smallest, rank the three marker colors by input frequency from most common to least common, and recolor each fragment by the correspondingly ranked marker color.
    - Keep the red cut rays, erase all marker pixels to background, and leave the remaining background unchanged.

    Key insight:
    - The output colors are not derived from the sprite itself: the hidden mapping comes from comparing fragment-size rank against the frequency rank of remote marker colors.
    """
    return _solve(np.asarray(grid, dtype=int), strict=True)


def _random_path_between(a: tuple[int, int], b: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = [(r, c)]
    axes = ["r", "c"]
    while (r, c) != (rr, cc):
        rng.shuffle(axes)
        moved = False
        for axis in axes:
            if axis == "r" and r != rr:
                r += 1 if rr > r else -1
                cells.append((r, c))
                moved = True
                break
            if axis == "c" and c != cc:
                c += 1 if cc > c else -1
                cells.append((r, c))
                moved = True
                break
        if not moved:
            break
    return cells


def _connected_subset(
    allowed: set[tuple[int, int]],
    area: int,
    required: set[tuple[int, int]],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    if not required or not required.issubset(allowed):
        return None
    req = list(required)
    current = {req[0]}
    for target in req[1:]:
        path = _random_path_between(next(iter(current)), target, rng)
        if any(cell not in allowed for cell in path):
            return None
        current.update(path)
    if len(current) > area:
        return None

    while len(current) < area:
        frontier = []
        for r, c in current:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in current:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        current.add(rng.choice(frontier))
    return current


def _canonical_shape(rng: random.Random) -> tuple[set[tuple[int, int]], int, int, int, int] | None:
    for _ in range(300):
        h = rng.randint(5, 7)
        w = rng.randint(5, 7)
        cut_r = rng.randint(1, h - 2)
        cut_c = rng.randint(1, w - 2)

        tl = {(r, c) for r in range(0, cut_r + 1) for c in range(0, cut_c + 1)}
        tr = {(r, c) for r in range(0, cut_r + 1) for c in range(cut_c + 1, w)}
        bl = {(r, c) for r in range(cut_r + 1, h) for c in range(0, cut_c + 1)}
        if min(len(tl), len(tr), len(bl)) < 2:
            continue

        row_bridge = rng.randint(max(0, cut_r - 1), cut_r)
        col_bridge = rng.randint(max(0, cut_c - 1), cut_c)

        req_tl = {(cut_r, cut_c), (row_bridge, cut_c), (cut_r, col_bridge)}
        req_tr = {(row_bridge, cut_c + 1)}
        req_bl = {(cut_r + 1, col_bridge)}

        candidates = []
        for a in range(len(req_tl), min(len(tl), 6) + 1):
            for b in range(2, min(len(tr), 5) + 1):
                for c in range(2, min(len(bl), 5) + 1):
                    if len({a, b, c}) != 3:
                        continue
                    total = a + b + c
                    if 8 <= total <= 15:
                        candidates.append((a, b, c))
        rng.shuffle(candidates)
        if not candidates:
            continue

        for area_tl, area_tr, area_bl in candidates[:25]:
            cells_tl = _connected_subset(tl, area_tl, req_tl, rng)
            cells_tr = _connected_subset(tr, area_tr, req_tr, rng)
            cells_bl = _connected_subset(bl, area_bl, req_bl, rng)
            if cells_tl is None or cells_tr is None or cells_bl is None:
                continue
            sprite = cells_tl | cells_tr | cells_bl
            local = np.zeros((h, w), dtype=int)
            for r, c in sprite:
                local[r, c] = SPRITE
            if len(_components_from_mask(local == SPRITE)) != 1:
                continue
            if not (2 <= int(np.count_nonzero(local[cut_r] == SPRITE)) <= 4):
                continue
            if not (2 <= int(np.count_nonzero(local[:, cut_c] == SPRITE)) <= 4):
                continue
            frags = _fragment_components(local == SPRITE, cut_r, cut_c, "L", "T")
            if len(frags) != 3:
                continue
            areas = sorted((frag["area"] for frag in frags), reverse=True)
            if len(set(areas)) != 3:
                continue
            return sprite, h, w, cut_r, cut_c
    return None


def _transform_shape(
    coords: set[tuple[int, int]],
    h: int,
    w: int,
    cut_r: int,
    cut_c: int,
    h_side: str,
    v_side: str,
) -> tuple[set[tuple[int, int]], int, int]:
    out = set()
    for r, c in coords:
        rr = h - 1 - r if v_side == "B" else r
        cc = w - 1 - c if h_side == "R" else c
        out.add((rr, cc))
    rr_cut = h - 1 - cut_r if v_side == "B" else cut_r
    cc_cut = w - 1 - cut_c if h_side == "R" else cut_c
    return out, rr_cut, cc_cut


def _place_markers(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    colors: list[int],
    counts: list[int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    placed: list[tuple[int, int]] = []
    for color, count in zip(colors, counts):
        for _ in range(count):
            candidates = []
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if grid[r, c] != BG or (r, c) in forbidden:
                        continue
                    if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in placed):
                        continue
                    candidates.append((r, c))
            if not candidates:
                return False
            r, c = rng.choice(candidates)
            grid[r, c] = int(color)
            placed.append((r, c))
    return True


def _sprite_bbox(sprite_cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in sprite_cells]
    cols = [c for _, c in sprite_cells]
    return min(rows), min(cols), max(rows), max(cols)


def _validate_generated(grid: np.ndarray) -> None:
    info = _parse_input(grid)
    sprite_mask = info["sprite_mask"]
    h_row = info["h_row"]
    v_col = info["v_col"]
    assert 8 <= int(sprite_mask.sum()) <= 15
    assert len(_components_from_mask(sprite_mask)) == 1
    assert 2 <= int(np.count_nonzero(sprite_mask[h_row])) <= 4
    assert 2 <= int(np.count_nonzero(sprite_mask[:, v_col])) <= 4

    fragments = _fragment_components(sprite_mask, h_row, v_col, info["h_side"], info["v_side"])
    assert len(fragments) == 3
    areas = [frag["area"] for frag in fragments]
    assert len(set(areas)) == 3

    for color in info["marker_colors"]:
        comps = _components_from_mask(grid == color)
        assert comps and all(int(comp.shape[0]) == 1 for comp in comps)
    counts = [int(np.count_nonzero(grid == color)) for color in info["marker_colors"]]
    assert len(set(counts)) == 3

    out = main(grid)
    for r, c in info["marker_positions"]:
        assert int(out[r, c]) == info["bg"]
    for r, c in info["ctrl_pts"]:
        assert int(out[r, c]) == CTRL

    ordered_frags = sorted(fragments, key=lambda frag: (-frag["area"], frag["label"]))
    ordered_colors = sorted(info["marker_colors"], key=lambda color: (-int(np.count_nonzero(grid == color)), color))
    for frag, color in zip(ordered_frags, ordered_colors):
        coords = frag["coords"]
        vals = {int(out[r, c]) for r, c in coords}
        assert vals == {int(color)}


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    base = _canonical_shape(rng)
    if base is None:
        return None
    sprite_cells, local_h, local_w, cut_r, cut_c = base

    grid_h = rng.randint(13, 16)
    grid_w = rng.randint(13, 16)
    h_side = rng.choice(["L", "R"])
    v_side = rng.choice(["T", "B"])
    sprite_cells, cut_r, cut_c = _transform_shape(sprite_cells, local_h, local_w, cut_r, cut_c, h_side, v_side)

    top_min = 3
    top_max = grid_h - local_h - 3
    left_min = 2
    left_max = min(grid_w - local_w - 3, max(2, grid_w // 2 - local_w // 2))
    if top_max < top_min or left_max < left_min:
        return None
    top = rng.randint(top_min, top_max)
    left = rng.randint(left_min, left_max)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    global_sprite = {(top + r, left + c) for r, c in sprite_cells}
    for r, c in global_sprite:
        grid[r, c] = SPRITE

    h_row = top + cut_r
    v_col = left + cut_c
    grid[h_row, 0 if h_side == "L" else grid_w - 1] = CTRL
    grid[0 if v_side == "T" else grid_h - 1, v_col] = CTRL

    r0, c0, r1, c1 = _sprite_bbox(global_sprite)
    forbidden = set()
    for r in range(grid_h):
        for c in range(grid_w):
            if r0 - 2 <= r <= r1 + 2 and c0 - 2 <= c <= c1 + 2:
                forbidden.add((r, c))
            if r == h_row or c == v_col:
                forbidden.add((r, c))
    forbidden.update(_border_positions(grid.shape))

    colors = rng.sample(list(MARKER_POOL), 3)
    count_triplets = [(1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4), (2, 4, 5)]
    counts = list(rng.choice(count_triplets))
    rng.shuffle(counts)
    if not _place_markers(grid, forbidden, colors, counts, rng):
        return None

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
