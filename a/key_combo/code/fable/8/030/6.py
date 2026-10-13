import itertools
import random
from collections import deque

import numpy as np


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _base_to_quad_coord(r: int, c: int, h: int, w: int, q: int) -> tuple[int, int]:
    if q == 0:
        return int(r), int(c)
    if q == 1:
        return int(r), int(w - 1 - c)
    if q == 2:
        return int(h - 1 - r), int(c)
    if q == 3:
        return int(h - 1 - r), int(w - 1 - c)
    raise ValueError("invalid quadrant index")


def _quad_to_base_coord(r: int, c: int, h: int, w: int, q: int) -> tuple[int, int]:
    return _base_to_quad_coord(r, c, h, w, q)


def _global_coord(local_r: int, local_c: int, sep_r: int, sep_c: int, q: int) -> tuple[int, int]:
    if q == 0:
        return int(local_r), int(local_c)
    if q == 1:
        return int(local_r), int(sep_c + 1 + local_c)
    if q == 2:
        return int(sep_r + 1 + local_r), int(local_c)
    if q == 3:
        return int(sep_r + 1 + local_r), int(sep_c + 1 + local_c)
    raise ValueError("invalid quadrant index")


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _canonical_cycle(cycle: tuple[int, ...]) -> tuple[int, ...]:
    if not cycle:
        return cycle
    start = min(range(len(cycle)), key=lambda i: cycle[i])
    return tuple(cycle[(start + i) % len(cycle)] for i in range(len(cycle)))


def _find_separator(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 1 and w % 2 == 1, "grid must have odd height and width"
    sep_r = h // 2
    sep_c = w // 2
    bg = int(g[sep_r, 0])
    assert np.all(g[sep_r, :] == bg), "middle row must be the separator row"
    assert np.all(g[:, sep_c] == bg), "middle column must be the separator column"
    return sep_r, sep_c, bg


def _extract_observations(grid: np.ndarray) -> tuple[int, int, int, dict[tuple[int, int], list[tuple[int, int]]], np.ndarray]:
    sep_r, sep_c, bg = _find_separator(grid)
    quad_h, quad_w = sep_r, sep_c
    observations: dict[tuple[int, int], list[tuple[int, int]]] = {}
    shape = np.zeros((quad_h, quad_w), dtype=bool)

    quadrants = [
        grid[:sep_r, :sep_c],
        grid[:sep_r, sep_c + 1 :],
        grid[sep_r + 1 :, :sep_c],
        grid[sep_r + 1 :, sep_c + 1 :],
    ]

    for q, quad in enumerate(quadrants):
        assert quad.shape == (quad_h, quad_w), "quadrants must have equal shape"
        for r, c in np.argwhere(quad != bg):
            br, bc = _quad_to_base_coord(int(r), int(c), quad_h, quad_w, q)
            shape[br, bc] = True
            observations.setdefault((br, bc), []).append((q, int(quad[r, c])))

    assert observations, "expected at least one visible motif pixel"
    return sep_r, sep_c, bg, observations, shape


def _infer_cycle(observations: dict[tuple[int, int], list[tuple[int, int]]]) -> tuple[int, ...]:
    colors = sorted({color for obs in observations.values() for _q, color in obs})
    assert len(colors) >= 2, "expected at least two motif colors"
    assert any(len(obs) >= 2 for obs in observations.values()), "expected cross-quadrant color evidence"

    anchor = min(colors)
    others = [c for c in colors if c != anchor]
    valid = []

    for tail in itertools.permutations(others):
        cycle = (anchor,) + tail
        index = {color: i for i, color in enumerate(cycle)}
        period = len(cycle)
        ok = True
        for obs in observations.values():
            for i in range(len(obs)):
                q1, c1 = obs[i]
                for q2, c2 in obs[i + 1 :]:
                    if cycle[(index[c1] + (q2 - q1)) % period] != c2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            valid.append(cycle)

    assert valid, "could not infer a valid color cycle"
    assert len(valid) == 1, "color cycle is ambiguous"
    return valid[0]


def _filled_output(grid: np.ndarray, cycle: tuple[int, ...], observations: dict[tuple[int, int], list[tuple[int, int]]], shape: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    sep_r, sep_c, bg = _find_separator(out)
    quad_h, quad_w = shape.shape
    index = {color: i for i, color in enumerate(cycle)}

    for br, bc in np.argwhere(shape):
        obs = observations[(int(br), int(bc))]
        ref_q, ref_color = obs[0]
        ref_idx = index[ref_color]

        for q in range(4):
            lr, lc = _base_to_quad_coord(int(br), int(bc), quad_h, quad_w, q)
            gr, gc = _global_coord(lr, lc, sep_r, sep_c, q)
            expected = cycle[(ref_idx + q - ref_q) % len(cycle)]
            if out[gr, gc] == bg:
                out[gr, gc] = expected
            else:
                assert int(out[gr, gc]) == expected, "visible motif pixels must match the inferred cycle"
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Restore the missing pixels in four mirrored motif copies, converting source colors back through the quadrant phase cycle.

    Input structure:
    - One solid separator row and one solid separator column of the background color
      split the canvas into four equal quadrants.
    - The four quadrants contain one multicolor motif each: upper-right is the
      upper-left mirrored left-to-right, lower-left is mirrored top-to-bottom,
      and lower-right is rotated 180 degrees.
    - Each quadrant uses the same ordered color cycle with phase shifts 0, 1, 2,
      and 3 across the quadrants, and some motif pixels are missing as background.

    Rule:
    - Use the mirror and rotation correspondence to align every visible quadrant
      pixel back to one base motif position.
    - Infer the unique color cycle that makes all cross-quadrant observations
      consistent with the fixed phase offsets.
    - Fill each missing motif pixel by taking any visible corresponding pixel and
      shifting its color by the phase difference into the target quadrant.

    Key insight:
    - Shape recovery comes from geometric symmetry across quadrants, while color
      recovery is a separate cyclic phase conversion instead of direct copying.
    """
    g = np.asarray(input_grid, dtype=int)
    sep_r, sep_c, bg, observations, shape = _extract_observations(g)
    assert sep_r > 0 and sep_c > 0, "quadrants must be non-empty"
    assert np.all(g[sep_r, :] == bg) and np.all(g[:, sep_c] == bg)
    cycle = _infer_cycle(observations)
    return _filled_output(g, cycle, observations, shape)


def _random_connected_mask(h: int, w: int, target_area: int) -> np.ndarray:
    for _ in range(200):
        cells: set[tuple[int, int]] = set()
        c = random.randrange(w)
        path_cols = []
        for r in range(h):
            if r > 0:
                c = max(0, min(w - 1, c + random.choice([-1, 0, 1])))
            cells.add((r, c))
            path_cols.append(c)

        anchor_r = random.randrange(h)
        anchor_c = path_cols[anchor_r]

        r = anchor_r
        for cc in range(anchor_c - 1, -1, -1):
            r = max(0, min(h - 1, r + random.choice([-1, 0, 1])))
            cells.add((r, cc))

        r = anchor_r
        for cc in range(anchor_c + 1, w):
            r = max(0, min(h - 1, r + random.choice([-1, 0, 1])))
            cells.add((r, cc))

        while len(cells) < target_area:
            frontier = []
            for r0, c0 in cells:
                for nr, nc in _neighbors4(r0, c0):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r0, c0 in cells:
            mask[r0, c0] = True

        if not _is_connected(mask):
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        density = mask.mean()
        if 0.22 <= density <= 0.62:
            return mask
    raise RuntimeError("failed to sample a valid motif mask")


def _assign_base_indices(mask: np.ndarray, color_count: int) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    reps = (len(coords) + color_count - 1) // color_count
    pool = list(range(color_count)) * reps
    random.shuffle(pool)
    grid = np.full(mask.shape, -1, dtype=int)
    for (r, c), idx in zip(coords, pool[: len(coords)]):
        grid[r, c] = int(idx)
    return grid


def _random_visibility(base_indices: np.ndarray) -> np.ndarray:
    h, w = base_indices.shape
    visible = np.zeros((4, h, w), dtype=bool)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(base_indices >= 0)]
    miss_prob = random.uniform(0.20, 0.45)

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in coords:
        idx = int(base_indices[r, c])
        by_color.setdefault(idx, []).append((r, c))
        flags = [random.random() > miss_prob for _ in range(4)]
        if not any(flags):
            flags[random.randrange(4)] = True
        visible[:, r, c] = flags

    for cells in by_color.values():
        r, c = random.choice(cells)
        visible[0, r, c] = True
        visible[1, r, c] = True

    for q in range(4):
        if np.all(visible[q, base_indices >= 0]):
            r, c = random.choice(coords)
            visible[q, r, c] = False
        if not np.any(visible[q, base_indices >= 0]):
            r, c = random.choice(coords)
            visible[q, r, c] = True

    if not any(int(visible[:, r, c].sum()) == 1 for r, c in coords):
        r, c = random.choice(coords)
        keep = random.randrange(4)
        visible[:, r, c] = False
        visible[keep, r, c] = True

    if not any(int(visible[:, r, c].sum()) == 4 for r, c in coords):
        r, c = random.choice(coords)
        visible[:, r, c] = True

    return visible


def _build_quadrant(base_indices: np.ndarray, cycle: tuple[int, ...], q: int, bg: int) -> np.ndarray:
    h, w = base_indices.shape
    quad = np.full((h, w), bg, dtype=int)
    for br, bc in np.argwhere(base_indices >= 0):
        lr, lc = _base_to_quad_coord(int(br), int(bc), h, w, q)
        quad[lr, lc] = int(cycle[(int(base_indices[br, bc]) + q) % len(cycle)])
    return quad


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(200):
        quad_h = random.randint(6, 13)
        quad_w = random.randint(6, 13)
        bg = random.randrange(10)
        palette = [c for c in range(10) if c != bg]

        min_colors = 4
        max_colors = min(6, len(palette))
        color_count = random.randint(min_colors, max_colors)
        cycle = tuple(random.sample(palette, color_count))

        min_area = max(quad_h + quad_w + 2, color_count + 6)
        max_area = min(int(quad_h * quad_w * 0.55), quad_h * quad_w - 2)
        if max_area < min_area:
            continue

        try:
            target_area = random.randint(min_area, max_area)
            mask = _random_connected_mask(quad_h, quad_w, target_area)
        except RuntimeError:
            continue

        base_indices = _assign_base_indices(mask, color_count)
        visible = _random_visibility(base_indices)

        full_quads = [_build_quadrant(base_indices, cycle, q, bg) for q in range(4)]
        input_quads = [quad.copy() for quad in full_quads]

        for q in range(4):
            for br, bc in np.argwhere(base_indices >= 0):
                if not visible[q, br, bc]:
                    lr, lc = _base_to_quad_coord(int(br), int(bc), quad_h, quad_w, q)
                    input_quads[q][lr, lc] = bg

        grid_h = quad_h * 2 + 1
        grid_w = quad_w * 2 + 1
        inp = np.full((grid_h, grid_w), bg, dtype=int)
        out = np.full((grid_h, grid_w), bg, dtype=int)
        sep_r, sep_c = quad_h, quad_w

        inp[:sep_r, :sep_c] = input_quads[0]
        inp[:sep_r, sep_c + 1 :] = input_quads[1]
        inp[sep_r + 1 :, :sep_c] = input_quads[2]
        inp[sep_r + 1 :, sep_c + 1 :] = input_quads[3]

        out[:sep_r, :sep_c] = full_quads[0]
        out[:sep_r, sep_c + 1 :] = full_quads[1]
        out[sep_r + 1 :, :sep_c] = full_quads[2]
        out[sep_r + 1 :, sep_c + 1 :] = full_quads[3]

        meta = {
            "bg": bg,
            "cycle": cycle,
            "base_indices": base_indices,
            "visible": visible,
            "sep_r": sep_r,
            "sep_c": sep_c,
        }
        return inp, out, meta

    raise RuntimeError("failed to build a valid puzzle instance")


def _naive_copy_fill(grid: np.ndarray) -> np.ndarray:
    sep_r, sep_c, bg, observations, shape = _extract_observations(grid)
    out = np.asarray(grid, dtype=int).copy()
    quad_h, quad_w = shape.shape
    for br, bc in np.argwhere(shape):
        ref_q, ref_color = observations[(int(br), int(bc))][0]
        _ = ref_q
        for q in range(4):
            lr, lc = _base_to_quad_coord(int(br), int(bc), quad_h, quad_w, q)
            gr, gc = _global_coord(lr, lc, sep_r, sep_c, q)
            if out[gr, gc] == bg:
                out[gr, gc] = ref_color
    return out


def _case_is_valid(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    sep_r, sep_c, bg, observations, shape = _extract_observations(inp)
    if (sep_r, sep_c) != (meta["sep_r"], meta["sep_c"]):
        return False
    if bg != meta["bg"]:
        return False
    if np.any(np.all(inp == bg, axis=1) & (np.arange(inp.shape[0]) != sep_r)):
        return False
    if np.any(np.all(inp == bg, axis=0) & (np.arange(inp.shape[1]) != sep_c)):
        return False

    cycle = _infer_cycle(observations)
    if _canonical_cycle(cycle) != _canonical_cycle(tuple(meta["cycle"])):
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    if np.array_equal(_naive_copy_fill(inp), expected):
        return False

    visible_counts = [len(obs) for obs in observations.values()]
    if min(visible_counts) < 1:
        return False
    if max(visible_counts) < 2:
        return False

    missing_by_quadrant = []
    base_indices = meta["base_indices"]
    for q in range(4):
        missing_by_quadrant.append(bool(np.any((base_indices >= 0) & (~meta["visible"][q]))))
    if not all(missing_by_quadrant):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp, expected, meta = _build_case()
        if _case_is_valid(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid input")
