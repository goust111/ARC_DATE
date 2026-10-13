import random
from collections import deque

import numpy as np


BG = 0
HOLE_RECOLOR = 3
INPUT_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GENERATION_ATTEMPTS = 500


def _normalize(coords):
    arr = np.asarray(list(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _rotate(coords):
    arr = np.asarray(coords, dtype=int)
    return np.column_stack((arr[:, 1], -arr[:, 0]))


def _random_transform(coords):
    arr = np.asarray(list(coords), dtype=int).copy()
    for _ in range(random.randrange(4)):
        arr = _rotate(arr)
    if random.random() < 0.5:
        arr[:, 1] *= -1
    if random.random() < 0.5:
        arr[:, 0] *= -1
    return _normalize(arr)


def _shape_dims(coords):
    arr = np.asarray(coords, dtype=int)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def _coords_to_set(coords):
    return {tuple(map(int, rc)) for rc in np.asarray(coords, dtype=int)}


def _is_connected(coords):
    pts = _coords_to_set(coords)
    if not pts:
        return False
    q = deque([next(iter(pts))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _has_hole(coords):
    arr = _normalize(coords)
    h, w = _shape_dims(arr)
    mask = np.zeros((h + 2, w + 2), dtype=bool)
    mask[arr[:, 0] + 1, arr[:, 1] + 1] = True

    exterior = np.zeros_like(mask, dtype=bool)
    q = deque([(0, 0)])
    exterior[0, 0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                if not mask[rr, cc] and not exterior[rr, cc]:
                    exterior[rr, cc] = True
                    q.append((rr, cc))

    enclosed_bg = (~mask) & (~exterior)
    return bool(enclosed_bg.any())


def _find_components(grid):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": np.asarray(coords, dtype=int),
                    "area": len(coords),
                }
            )
    return comps


def _paint_coords(grid, coords, color):
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each object vertically by its area, using hole topology to choose both direction and recolor.

    Input structure:
    - The input is a black background grid with several separated 4-connected monochrome objects.
    - Each object has area 3-8 and a distinct non-black color.
    - Some objects enclose at least one black cell as a true hole, while others are solid or open.

    Rule:
    - Detect every 4-connected foreground object and compute its pixel area.
    - If an object has a hole, move it upward by its area; otherwise move it downward by its area.
    - Clip pixels that move above the top edge or below the bottom edge.
    - After moving, recolor hole-objects to green and leave solid objects in their original colors.

    Key insight:
    - The same topological property controls two independent outputs at once: hole-objects move upward and turn green, while non-hole objects move downward and keep their colors; area supplies the displacement magnitude.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out = np.full((H, W), BG, dtype=int)

    for comp in _find_components(g):
        coords = comp["coords"]
        area = int(comp["area"])
        hollow = _has_hole(coords)
        dr = -area if hollow else area
        color = HOLE_RECOLOR if hollow else int(comp["color"])
        for r, c in coords:
            rr = int(r) + dr
            cc = int(c)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = color

    return out


def _polyomino(area):
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < area:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr not in cells:
                frontier.add(nbr)
    return _random_transform(cells)


def _open_frame():
    ring = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    boundary = list(ring)
    random.shuffle(boundary)
    remove_count = random.randint(1, 4)
    cells = ring - set(boundary[:remove_count])
    arr = _normalize(cells)
    if len(cells) < 3 or not _is_connected(arr) or _has_hole(arr):
        return _open_frame()
    return _random_transform(arr)


def _solid_shape():
    for _ in range(100):
        if random.random() < 0.45:
            coords = _open_frame()
        else:
            coords = _polyomino(random.randint(3, 8))
        area = int(len(coords))
        if 3 <= area <= 8 and _is_connected(coords) and not _has_hole(coords):
            return _normalize(coords)
    raise RuntimeError("failed to build a solid shape")


def _hollow_shape():
    return np.asarray(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
        dtype=int,
    )


def _expanded_mask(shape, coords, pad=1):
    H, W = shape
    out = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr, nc] = True
    return out


def _placeable(blocked, coords):
    rr = coords[:, 0]
    cc = coords[:, 1]
    return not bool(blocked[rr, cc].any())


def _solid_rows(H, h, area, want_clip):
    max_row = min(H - 1, H - h + area)
    rows = list(range(area, max_row + 1))
    if want_clip:
        rows = [r for r in rows if r > H - h]
    else:
        rows = [r for r in rows if r <= H - h]
    return rows


def _hollow_rows(H, h, area, want_clip):
    min_row = 1 - h
    max_row = H - h - area
    rows = list(range(min_row, max_row + 1))
    if want_clip:
        rows = [r for r in rows if r < 0]
    else:
        rows = [r for r in rows if r >= 0]
    return rows


def _build_specs(H, W):
    count = random.randint(4, 6)
    hollow_count = random.randint(1, min(2, count - 1))
    colors = random.sample(INPUT_COLORS, count)

    specs = []
    for idx in range(count):
        hollow = idx < hollow_count
        coords = _hollow_shape() if hollow else _solid_shape()
        coords = _normalize(coords)
        h, w = _shape_dims(coords)
        specs.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "color": colors[idx],
                "target_color": HOLE_RECOLOR if hollow else colors[idx],
                "hollow": hollow,
                "h": h,
                "w": w,
                "want_clip": False,
            }
        )

    hollow_indices = [i for i, spec in enumerate(specs) if spec["hollow"]]
    solid_indices = [i for i, spec in enumerate(specs) if not spec["hollow"] and spec["h"] >= 2]
    if hollow_indices and random.random() < 0.6:
        specs[random.choice(hollow_indices)]["want_clip"] = True
    if solid_indices and random.random() < 0.6:
        specs[random.choice(solid_indices)]["want_clip"] = True

    random.shuffle(specs)
    return specs


def _candidate_positions(spec, H, W):
    coords = spec["coords"]
    area = spec["area"]
    h = spec["h"]
    w = spec["w"]
    want_clip = spec["want_clip"]

    if spec["hollow"]:
        row_choices = _hollow_rows(H, h, area, want_clip)
    else:
        row_choices = _solid_rows(H, h, area, want_clip)
    if not row_choices:
        return []

    random.shuffle(row_choices)
    col_choices = list(range(0, W - w + 1))
    random.shuffle(col_choices)

    candidates = []
    for out_r in row_choices:
        in_r = out_r + area if spec["hollow"] else out_r - area
        for out_c in col_choices:
            input_coords = coords + np.array([in_r, out_c])
            if input_coords[:, 0].min() < 0 or input_coords[:, 0].max() >= H:
                continue
            visible = coords + np.array([out_r, out_c])
            keep = (0 <= visible[:, 0]) & (visible[:, 0] < H)
            visible = visible[keep]
            if visible.size == 0:
                continue
            candidates.append((input_coords, visible))
    return candidates


def _semantic_ok(inp, expected, specs):
    comps = _find_components(inp)
    if len(comps) != len(specs):
        return False
    colors = [comp["color"] for comp in comps]
    if len(set(colors)) != len(colors) or any(color == HOLE_RECOLOR for color in colors):
        return False

    hollow_seen = 0
    solid_seen = 0
    for comp in comps:
        area = comp["area"]
        if area < 3 or area > 8:
            return False
        if _has_hole(comp["coords"]):
            hollow_seen += 1
        else:
            solid_seen += 1
    if hollow_seen == 0 or solid_seen == 0:
        return False

    occ = np.argwhere(inp != BG)
    for i in range(len(occ)):
        for j in range(i + 1, len(occ)):
            if max(abs(int(occ[i, 0]) - int(occ[j, 0])), abs(int(occ[i, 1]) - int(occ[j, 1]))) <= 1:
                if int(inp[occ[i, 0], occ[i, 1]]) != int(inp[occ[j, 0], occ[j, 1]]):
                    return False

    out = main(inp)
    if not np.array_equal(out, expected):
        return False

    if np.count_nonzero(expected == HOLE_RECOLOR) == 0:
        return False
    return True


def _gen_attempt():
    H = random.randint(14, 18)
    W = random.randint(14, 18)
    specs = _build_specs(H, W)

    order = sorted(
        range(len(specs)),
        key=lambda i: (0 if specs[i]["hollow"] else 1, -specs[i]["area"], -specs[i]["h"], -specs[i]["w"]),
    )

    blocked_input = np.zeros((H, W), dtype=bool)
    blocked_output = np.zeros((H, W), dtype=bool)
    inp = np.full((H, W), BG, dtype=int)
    expected = np.full((H, W), BG, dtype=int)

    for idx in order:
        spec = specs[idx]
        candidates = _candidate_positions(spec, H, W)
        if not candidates:
            return None
        random.shuffle(candidates)

        placed = False
        for input_coords, visible_coords in candidates:
            if not _placeable(blocked_input, input_coords):
                continue
            if not _placeable(blocked_output, visible_coords):
                continue

            _paint_coords(inp, input_coords, spec["color"])
            _paint_coords(expected, visible_coords, spec["target_color"])
            blocked_input |= _expanded_mask((H, W), input_coords, pad=1)
            blocked_output |= _expanded_mask((H, W), visible_coords, pad=1)
            placed = True
            break

        if not placed:
            return None

    if not _semantic_ok(inp, expected, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
