import random
from collections import Counter, deque

import numpy as np


BG = 0
GRAY = 5
ORANGE = 7
WALL = 8
BODY_PALETTE = (1, 2, 3, 4, 6, 9)
MAX_GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components(mask)) == 1


def _find_wall(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    row_hits = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color):
            row_hits.append(("h", r, color))

    col_hits = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            col_hits.append(("v", c, color))

    hits = row_hits + col_hits
    if len(hits) != 1:
        raise AssertionError("expected exactly one full-row or full-column wall")
    return hits[0]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    wall_orientation, wall_index, wall_color = _find_wall(g, bg)

    sprite_mask = (g != bg) & (g != wall_color)
    comps = _components(sprite_mask)
    assert len(comps) == 1, "expected exactly one sprite component"
    coords = comps[0]

    gray_pts = np.argwhere(g == GRAY)
    orange_pts = np.argwhere(g == ORANGE)
    assert gray_pts.shape == (1, 2), "expected exactly one gray instruction pixel"
    assert orange_pts.shape == (1, 2), "expected exactly one orange instruction pixel"
    gray = gray_pts[0].astype(int)
    orange = orange_pts[0].astype(int)

    coord_set = {tuple(rc) for rc in coords.tolist()}
    assert tuple(gray.tolist()) in coord_set and tuple(orange.tolist()) in coord_set

    sprite_colors = {int(g[r, c]) for r, c in coord_set}
    body_colors = sprite_colors - {GRAY, ORANGE}
    assert len(body_colors) == 1, "sprite body must have one color besides the instruction pair"
    body_color = int(next(iter(body_colors)))

    vector = orange - gray
    assert int(vector[0]) != 0 or int(vector[1]) != 0

    axis = 1 if wall_orientation == "v" else 0
    axis_vals = coords[:, axis]
    if int(axis_vals.max()) < wall_index:
        side = -1
    elif int(axis_vals.min()) > wall_index:
        side = 1
    else:
        raise AssertionError("sprite must lie entirely on one side of the wall")

    step_along_wall_axis = int(vector[axis])
    assert (side == -1 and step_along_wall_axis > 0) or (side == 1 and step_along_wall_axis < 0), (
        "instruction vector must point toward the wall"
    )

    return {
        "grid": g,
        "bg": bg,
        "wall_orientation": wall_orientation,
        "wall_index": int(wall_index),
        "wall_color": int(wall_color),
        "coords": coords.astype(int),
        "gray": gray,
        "orange": orange,
        "vector": vector.astype(int),
        "body_color": body_color,
        "axis": axis,
        "side": side,
    }


def _paint_coords(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    if coords.size == 0:
        return
    out[coords[:, 0], coords[:, 1]] = int(color)


def _shifted_sets(coords: np.ndarray, vector: np.ndarray, max_step: int) -> list[set[tuple[int, int]]]:
    return [{tuple(rc) for rc in (coords + step * vector).tolist()} for step in range(max_step + 1)]


def _solve_variant(grid: np.ndarray, *, drop_clipped: bool = False, ignore_wall: bool = False) -> np.ndarray:
    parsed = _parse_scene(grid)
    out = parsed["grid"].copy()
    out[out == GRAY] = parsed["body_color"]
    out[out == ORANGE] = parsed["body_color"]

    coords = parsed["coords"]
    vector = parsed["vector"]
    axis = parsed["axis"]
    wall_index = parsed["wall_index"]
    side = parsed["side"]
    body_color = parsed["body_color"]
    h, w = out.shape

    for step in range(1, 64):
        shifted = coords + step * vector
        if ignore_wall:
            if np.any(shifted[:, 0] < 0) or np.any(shifted[:, 0] >= h) or np.any(shifted[:, 1] < 0) or np.any(shifted[:, 1] >= w):
                break
            _paint_coords(out, shifted, body_color)
            continue

        vals = shifted[:, axis]
        if side == -1:
            keep = vals < wall_index
            touched = np.any(vals >= wall_index)
        else:
            keep = vals > wall_index
            touched = np.any(vals <= wall_index)

        if touched:
            if not drop_clipped:
                kept = shifted[keep]
                assert kept.size > 0, "generator should avoid zero-area clipped copies"
                _paint_coords(out, kept, body_color)
            break

        if np.any(shifted[:, 0] < 0) or np.any(shifted[:, 0] >= h) or np.any(shifted[:, 1] < 0) or np.any(shifted[:, 1] >= w):
            raise AssertionError("copies must stop because of the wall, not because of the grid boundary")
        _paint_coords(out, shifted, body_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the sprite by its internal gray-to-orange vector, clipping the first wall-hitting copy and erasing the instruction colors.

    Input structure:
    - A black background contains one connected sprite whose body is a single color except for one gray pixel and one orange pixel embedded inside it.
    - A cyan wall appears elsewhere as one full row or one full column.
    - The gray and orange pixels belong to the sprite itself; they are not separate objects.

    Rule:
    - Read the displacement vector from the gray pixel to the orange pixel.
    - Copy the entire sprite at offsets +v, +2v, +3v, ... from the original position.
    - When the first copied sprite touches or crosses the wall, keep only the pixels that remain on the original side of the wall, discard the wall cell and everything beyond it, then stop copying.
    - Recolor the gray and orange instruction pixels in both the original and copied sprites back to the sprite body color; keep the wall unchanged.

    Key insight:
    - The instruction is encoded inside the sprite: the gray-orange displacement gives the translation step, and the stopping condition is the wall-clipped final copy rather than the grid boundary.
    """
    return _solve_variant(np.asarray(grid, dtype=int), drop_clipped=False, ignore_wall=False)


def _random_shape() -> np.ndarray:
    for _ in range(200):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        target_area = random.randint(8, 12)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {start}

        while len(cells) < target_area and frontier:
            cr, cc = random.choice(tuple(frontier))
            neighbors = []
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier.discard((cr, cc))
                continue
            nxt = random.choice(neighbors)
            cells.add(nxt)
            frontier.add(nxt)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if min(mask.shape) < 4 or int(mask.sum()) < 8:
            continue
        degrees = _cell_degrees(mask)
        if sum(deg >= 2 for deg in degrees.values()) < 3:
            continue
        return mask
    raise RuntimeError("failed to generate a connected sprite shape")


def _cell_degrees(mask: np.ndarray) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        deg = 0
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                deg += 1
        degrees[(int(r), int(c))] = deg
    return degrees


def _vector_category(dr: int, dc: int) -> str | None:
    if dr == 0 and abs(dc) >= 2:
        return "horizontal"
    if dc == 0 and abs(dr) >= 2:
        return "vertical"
    if dr != 0 and dc != 0 and abs(dr) == abs(dc) and abs(dr) >= 2:
        return "diag45"
    if dr != 0 and dc != 0 and abs(dr) != abs(dc) and max(abs(dr), abs(dc)) >= 2:
        return "skew"
    return None


def _marker_pairs(mask: np.ndarray) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    degrees = _cell_degrees(mask)
    preferred = [rc for rc in coords if degrees[rc] >= 2]
    pool = preferred if len(preferred) >= 3 else coords
    by_cat = {"horizontal": [], "vertical": [], "diag45": [], "skew": []}
    for gray in pool:
        for orange in pool:
            if gray == orange:
                continue
            dr = orange[0] - gray[0]
            dc = orange[1] - gray[1]
            cat = _vector_category(dr, dc)
            if cat is not None:
                by_cat[cat].append((np.array(gray, dtype=int), np.array(orange, dtype=int)))
    return by_cat


def _copies_disjoint(coords: np.ndarray, vector: np.ndarray, steps: int) -> bool:
    shifted_sets = _shifted_sets(coords, vector, steps)
    for i in range(len(shifted_sets)):
        for j in range(i + 1, len(shifted_sets)):
            overlap = shifted_sets[i] & shifted_sets[j]
            if j - i == 1:
                if not (1 <= len(overlap) <= 2):
                    return False
            elif overlap:
                return False
    return True


def _wall_thresholds(values: np.ndarray, delta: int, full_copies: int) -> list[int]:
    sign = 1 if delta > 0 else -1
    scaled = sign * values
    jump = abs(delta)
    clip_step = full_copies + 1
    prev_max = int(np.max(scaled + full_copies * jump))
    clip_min = int(np.min(scaled + clip_step * jump))
    clip_max = int(np.max(scaled + clip_step * jump))
    lo = max(prev_max, clip_min) + 1
    hi = clip_max
    return [sign * threshold for threshold in range(lo, hi + 1)]


def _build_grid(
    mask: np.ndarray,
    gray: np.ndarray,
    orange: np.ndarray,
    wall_orientation: str,
    wall_local_index: int,
    body_color: int,
    full_copies: int,
) -> np.ndarray | None:
    coords = np.argwhere(mask).astype(int)
    vector = orange - gray
    clip_step = full_copies + 1
    all_coords = np.vstack([coords + step * vector for step in range(clip_step + 1)])
    min_r, min_c = all_coords.min(axis=0)
    max_r, max_c = all_coords.max(axis=0)

    top = random.randint(1, 4)
    bottom = random.randint(1, 4)
    left = random.randint(1, 4)
    right = random.randint(1, 4)
    row_offset = top - int(min_r)
    col_offset = left - int(min_c)
    height = int(max_r - min_r + 1 + top + bottom)
    width = int(max_c - min_c + 1 + left + right)
    if height > 30 or width > 30:
        return None

    wall_index = row_offset + wall_local_index if wall_orientation == "h" else col_offset + wall_local_index
    wall_limit = height if wall_orientation == "h" else width
    if wall_index <= 0 or wall_index >= wall_limit - 1:
        return None

    grid = np.full((height, width), BG, dtype=int)
    if wall_orientation == "h":
        grid[wall_index, :] = WALL
    else:
        grid[:, wall_index] = WALL

    abs_coords = coords + np.array([row_offset, col_offset], dtype=int)
    grid[abs_coords[:, 0], abs_coords[:, 1]] = body_color
    grid[row_offset + int(gray[0]), col_offset + int(gray[1])] = GRAY
    grid[row_offset + int(orange[0]), col_offset + int(orange[1])] = ORANGE
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    parsed = _parse_scene(grid)
    coords = parsed["coords"]
    vector = parsed["vector"]
    axis = parsed["axis"]
    wall_index = parsed["wall_index"]
    side = parsed["side"]
    h, w = grid.shape

    assert 0 < wall_index < (w - 1 if axis == 1 else h - 1)

    saw_clip = False
    saw_full = 0
    shifted_sets = _shifted_sets(coords, vector, 4)
    for i in range(len(shifted_sets)):
        for j in range(i + 1, len(shifted_sets)):
            overlap = shifted_sets[i] & shifted_sets[j]
            if j - i == 1:
                if not (1 <= len(overlap) <= 2):
                    return False
            elif overlap:
                return False

    for step in range(1, 32):
        shifted = coords + step * vector
        if np.any(shifted[:, 0] < 0) or np.any(shifted[:, 0] >= h) or np.any(shifted[:, 1] < 0) or np.any(shifted[:, 1] >= w):
            return False
        vals = shifted[:, axis]
        if side == -1:
            keep = vals < wall_index
            touched = np.any(vals >= wall_index)
        else:
            keep = vals > wall_index
            touched = np.any(vals <= wall_index)
        if touched:
            if keep.sum() < 2 or keep.sum() >= len(coords):
                return False
            saw_clip = True
            break
        saw_full += 1

    if not saw_clip or saw_full < 1:
        return False

    true_out = main(grid)
    if GRAY in set(map(int, true_out.ravel())) or ORANGE in set(map(int, true_out.ravel())):
        return False
    if np.array_equal(true_out, _solve_variant(grid, drop_clipped=True, ignore_wall=False)):
        return False
    if np.array_equal(true_out, _solve_variant(grid, drop_clipped=False, ignore_wall=True)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mask = _random_shape()
    pairs_by_cat = _marker_pairs(mask)
    category_pool = ["horizontal", "vertical", "diag45", "skew"]
    category_order = [random.choices(category_pool, weights=[4, 4, 3, 5], k=1)[0]]
    coords = np.argwhere(mask).astype(int)
    body_color = random.choice(BODY_PALETTE)

    for category in category_order:
        candidates = pairs_by_cat.get(category, [])
        if not candidates:
            continue
        random.shuffle(candidates)
        for gray, orange in candidates[:24]:
            vector = orange - gray
            orientation_choices = []
            if int(vector[0]) != 0:
                orientation_choices.append("h")
            if int(vector[1]) != 0:
                orientation_choices.append("v")
            random.shuffle(orientation_choices)
            for wall_orientation in orientation_choices:
                axis = 0 if wall_orientation == "h" else 1
                delta = int(vector[axis])
                if delta == 0:
                    continue
                for full_copies in random.sample([1, 2, 3], k=3):
                    if not _copies_disjoint(coords, vector, full_copies + 1):
                        continue
                    wall_options = _wall_thresholds(coords[:, axis], delta, full_copies)
                    if not wall_options:
                        continue
                    random.shuffle(wall_options)
                    for wall_local_index in wall_options:
                        clip_vals = coords[:, axis] + (full_copies + 1) * delta
                        if delta > 0:
                            kept = int(np.sum(clip_vals < wall_local_index))
                        else:
                            kept = int(np.sum(clip_vals > wall_local_index))
                        if kept < 2 or kept >= len(coords):
                            continue
                        grid = _build_grid(
                            mask,
                            gray,
                            orange,
                            wall_orientation,
                            wall_local_index,
                            body_color,
                            full_copies,
                        )
                        if grid is None:
                            continue
                        if _semantic_checks(grid):
                            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
