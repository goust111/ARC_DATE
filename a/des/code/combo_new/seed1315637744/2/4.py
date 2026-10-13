from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PATH_COLOR = 4
RELAY_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 6, 7, 8, 9)
OBJECT_SHAPES = ((3, 2), (2, 3))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

PAIR_COUNT_OPTIONS = (1, 2, 3)
HORIZONTAL_WIDTH_RANGE = (16, 27)
VERTICAL_HEIGHT_RANGE = (16, 27)
PAIR_BAND_RANGE = (6, 9)
OUTER_PADDING_MAX = 3
MAX_GENERATION_ATTEMPTS = 400
LOCAL_PATCH_ATTEMPTS = 120
RELAY_EVEN_COUNTS = (0, 2, 4)
RELAY_ODD_COUNTS = (1, 3)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in NEIGHBORS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _extract_objects(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != PATH_COLOR) & (grid != RELAY_COLOR)
    objects = []
    for coords in _components(mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        colors = np.unique(grid[rows, cols])
        assert colors.size == 1, "objects must be monochrome"
        color = int(colors[0])
        r0, c0, r1, c1 = _bbox(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert (height, width) in OBJECT_SHAPES, "unexpected object shape"
        assert int(coords.shape[0]) == height * width == 6, "objects must be 3x2 or 2x3 solid rectangles"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "object rectangle must be solid"
        objects.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
            }
        )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Swap each path-connected object pair when the path uses an odd number of relay stations.

    Input structure:
    - The background is black.
    - Several solid 3x2 or 2x3 monochrome rectangles act as the paired objects.
    - Yellow one-cell-wide orthogonal paths connect the objects in disjoint pairs.
    - Grey single-pixel relay stations belong to those path networks and stay 4-connected to their path.

    Rule:
    - Find every connected path network, including its adjacent grey relay stations.
    - Identify the two rectangular objects touched by that network and count the grey relay pixels inside it.
    - If the relay count is even, keep both object colors unchanged; if it is odd, swap the two object colors.
    - Leave all yellow path pixels and grey relay pixels unchanged.

    Key insight:
    - The yellow connectivity tells you which two objects form a pair, but the output color action is controlled by the parity of the grey relay count on that same connection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    assert objects and len(objects) % 2 == 0, "expected an even number of paired objects"

    object_index = np.full(g.shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        object_index[coords[:, 0], coords[:, 1]] = idx

    network_mask = (g == PATH_COLOR) | (g == RELAY_COLOR)
    networks = _components(network_mask)
    assert networks, "expected at least one path network"

    out = g.copy()
    touch_counts = [0] * len(objects)

    for coords in networks:
        rows = coords[:, 0]
        cols = coords[:, 1]
        colors = g[rows, cols]
        assert np.any(colors == PATH_COLOR), "every network must contain yellow path pixels"
        relay_count = int(np.sum(colors == RELAY_COLOR))

        adjacent_objects: set[int] = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c), *g.shape):
                idx = int(object_index[nr, nc])
                if idx >= 0:
                    adjacent_objects.add(idx)

        assert len(adjacent_objects) == 2, "each path network must connect exactly two objects"
        a, b = sorted(adjacent_objects)
        touch_counts[a] += 1
        touch_counts[b] += 1

        if relay_count % 2 == 1:
            coords_a = objects[a]["coords"]
            coords_b = objects[b]["coords"]
            color_a = int(objects[a]["color"])
            color_b = int(objects[b]["color"])
            out[coords_a[:, 0], coords_a[:, 1]] = color_b
            out[coords_b[:, 0], coords_b[:, 1]] = color_a

    assert len(networks) == len(objects) // 2, "path count must match object pairs"
    assert all(count == 1 for count in touch_counts), "every object must belong to exactly one path"
    return out


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            segment = [(r0, c) for c in range(c0, c1 + step, step)]
        else:
            assert c0 == c1
            step = 1 if r1 >= r0 else -1
            segment = [(r, c0) for r in range(r0, r1 + step, step)]
        if cells and segment and segment[0] == cells[-1]:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _choose_station_cells(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    for _ in range(80):
        order = candidates[:]
        random.shuffle(order)
        picked: list[tuple[int, int]] = []
        for cell in order:
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) != 1 for rr, cc in picked):
                picked.append(cell)
                if len(picked) == count:
                    return picked
    return None


def _station_candidates(
    height: int,
    width: int,
    path_cells: list[tuple[int, int]],
    object_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    path_set = set(path_cells)
    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in path_cells:
        for nr, nc in _neighbors4(r, c, height, width):
            cell = (nr, nc)
            if cell in seen or cell in path_set or cell in object_cells:
                continue
            if any(nei in object_cells for nei in _neighbors4(nr, nc, height, width)):
                continue
            seen.add(cell)
            candidates.append(cell)
    return candidates


def _make_horizontal_pair_patch(
    height: int,
    width: int,
    color_a: int,
    color_b: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(LOCAL_PATCH_ATTEMPTS):
        h_a, w_a = random.choice(OBJECT_SHAPES)
        h_b, w_b = random.choice(OBJECT_SHAPES)

        left_max = min(4, width - w_a - w_b - 8)
        if left_max < 1:
            continue
        left = random.randint(1, left_max)

        right_hi = width - w_b - 2
        right_low = max(width * 2 // 3, left + w_a + 6)
        if right_low > right_hi:
            continue
        right = random.randint(right_low, right_hi)

        top_a = random.randint(1, height - h_a - 1)
        top_b = random.randint(1, height - h_b - 1)
        rows_a = list(range(top_a, top_a + h_a))
        rows_b = list(range(top_b, top_b + h_b))

        shared_rows = [r for r in rows_a if r in rows_b]
        if shared_rows and random.random() < 0.4:
            anchor_a = anchor_b = random.choice(shared_rows)
            style = "straight" if random.random() < 0.6 else "dogleg"
        else:
            anchor_a = random.choice(rows_a)
            anchor_b = random.choice(rows_b)
            if anchor_a == anchor_b and random.random() < 0.4:
                style = "straight"
            else:
                style = "dogleg" if width >= 18 and random.random() < 0.4 else "bend"

        start = (anchor_a, left + w_a)
        end = (anchor_b, right - 1)
        assert start[1] < end[1]

        if style == "straight":
            if anchor_a != anchor_b:
                continue
            points = [start, end]
        elif style == "bend":
            if start[1] + 1 > end[1] - 1:
                continue
            pivot = random.randint(start[1] + 1, end[1] - 1)
            points = [start, (anchor_a, pivot), (anchor_b, pivot), end]
        else:
            if end[1] - start[1] < 5:
                continue
            detour_rows = [r for r in range(height) if r != anchor_a and r != anchor_b]
            if not detour_rows:
                continue
            mid1_hi = end[1] - 3
            if start[1] + 1 > mid1_hi:
                continue
            mid1 = random.randint(start[1] + 1, mid1_hi)
            mid2 = random.randint(mid1 + 2, end[1] - 1)
            detour = random.choice(detour_rows)
            points = [
                start,
                (anchor_a, mid1),
                (detour, mid1),
                (detour, mid2),
                (anchor_b, mid2),
                end,
            ]

        path_cells = _polyline(points)
        path_set = set(path_cells)
        if len(path_set) != len(path_cells):
            continue

        object_a_cells = {
            (r, c)
            for r in range(top_a, top_a + h_a)
            for c in range(left, left + w_a)
        }
        object_b_cells = {
            (r, c)
            for r in range(top_b, top_b + h_b)
            for c in range(right, right + w_b)
        }
        object_cells = object_a_cells | object_b_cells

        if path_set & object_cells:
            continue

        candidates = _station_candidates(height, width, path_cells, object_cells)
        want_odd = random.random() < 0.5
        count_choices = RELAY_ODD_COUNTS if want_odd else RELAY_EVEN_COUNTS
        feasible_counts = [count for count in count_choices if count <= len(candidates)]
        if not feasible_counts:
            continue
        relay_count = random.choice(feasible_counts)
        relay_cells = _choose_station_cells(candidates, relay_count)
        if relay_cells is None:
            continue

        inp = np.full((height, width), BG, dtype=int)
        out = np.full((height, width), BG, dtype=int)
        inp[top_a : top_a + h_a, left : left + w_a] = color_a
        inp[top_b : top_b + h_b, right : right + w_b] = color_b
        out[:] = inp

        if relay_count % 2 == 1:
            out[top_a : top_a + h_a, left : left + w_a] = color_b
            out[top_b : top_b + h_b, right : right + w_b] = color_a

        for r, c in path_cells:
            inp[r, c] = PATH_COLOR
            out[r, c] = PATH_COLOR
        for r, c in relay_cells:
            inp[r, c] = RELAY_COLOR
            out[r, c] = RELAY_COLOR
        return inp, out
    return None


def _stack_rows(patches: list[np.ndarray]) -> np.ndarray:
    height = sum(int(p.shape[0]) for p in patches) + max(0, len(patches) - 1)
    width = int(patches[0].shape[1])
    canvas = np.full((height, width), BG, dtype=int)
    r = 0
    for idx, patch in enumerate(patches):
        ph = int(patch.shape[0])
        canvas[r : r + ph, :] = patch
        r += ph
        if idx != len(patches) - 1:
            r += 1
    return canvas


def _stack_cols(patches: list[np.ndarray]) -> np.ndarray:
    height = int(patches[0].shape[0])
    width = sum(int(p.shape[1]) for p in patches) + max(0, len(patches) - 1)
    canvas = np.full((height, width), BG, dtype=int)
    c = 0
    for idx, patch in enumerate(patches):
        pw = int(patch.shape[1])
        canvas[:, c : c + pw] = patch
        c += pw
        if idx != len(patches) - 1:
            c += 1
    return canvas


def _pad_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spare_h = 30 - int(inp.shape[0])
    spare_w = 30 - int(inp.shape[1])
    top = random.randint(0, min(OUTER_PADDING_MAX, spare_h))
    bottom = random.randint(0, min(OUTER_PADDING_MAX, spare_h - top))
    left = random.randint(0, min(OUTER_PADDING_MAX, spare_w))
    right = random.randint(0, min(OUTER_PADDING_MAX, spare_w - left))

    shape = (inp.shape[0] + top + bottom, inp.shape[1] + left + right)
    padded_inp = np.full(shape, BG, dtype=int)
    padded_out = np.full(shape, BG, dtype=int)
    padded_inp[top : top + inp.shape[0], left : left + inp.shape[1]] = inp
    padded_out[top : top + out.shape[0], left : left + out.shape[1]] = out
    return padded_inp, padded_out


def _build_rows_layout(pair_count: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    width = random.randint(*HORIZONTAL_WIDTH_RANGE)
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    for i in range(pair_count):
        height = random.randint(*PAIR_BAND_RANGE)
        patch = _make_horizontal_pair_patch(height, width, colors[2 * i], colors[2 * i + 1])
        if patch is None:
            return None
        inp_patch, out_patch = patch
        inputs.append(inp_patch)
        outputs.append(out_patch)

    inp = _stack_rows(inputs)
    out = _stack_rows(outputs)
    return _pad_pair(inp, out)


def _build_cols_layout(pair_count: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*VERTICAL_HEIGHT_RANGE)
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    for i in range(pair_count):
        width = random.randint(*PAIR_BAND_RANGE)
        patch = _make_horizontal_pair_patch(width, height, colors[2 * i], colors[2 * i + 1])
        if patch is None:
            return None
        inp_patch, out_patch = patch
        inputs.append(inp_patch.T.copy())
        outputs.append(out_patch.T.copy())

    inp = _stack_cols(inputs)
    out = _stack_cols(outputs)
    return _pad_pair(inp, out)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    colors = random.sample(list(OBJECT_COLORS), 2 * pair_count)
    layout = random.choice(("rows", "cols"))

    built = _build_rows_layout(pair_count, colors) if layout == "rows" else _build_cols_layout(pair_count, colors)
    if built is None:
        return None
    inp, expected = built

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-parity ARC input")
