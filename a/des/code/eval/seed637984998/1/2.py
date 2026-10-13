from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# Generator configuration.
PAIR_COUNT_BOUNDS = (2, 4)
RECT_DIMS = [
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    (5, 4),
    (5, 5),
    (5, 6),
    (5, 7),
    (6, 4),
    (6, 5),
    (6, 6),
    (7, 4),
    (7, 5),
]
GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 30
BOX_PADDING = 1
PLACEMENT_TRIES = 250
PAIR_TRIES = 100
GENERATION_TRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mask_connected(mask: np.ndarray) -> bool:
    h, w = mask.shape
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(int(v) for v in coords[0])
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _touches_full_bbox(mask: np.ndarray) -> bool:
    b = _mask_bbox(mask)
    if b is None:
        return False
    r0, c0, r1, c1 = b
    h, w = mask.shape
    return r0 == 0 and c0 == 0 and r1 == h - 1 and c1 == w - 1


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or bool(seen[r, c]):
                continue

            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "shape": mask.shape,
                    "mask": mask,
                    "area": len(coords),
                }
            )

    return components


def _pair_compatible(a: dict, b: dict) -> bool:
    if a["color"] == b["color"] or a["shape"] != b["shape"]:
        return False
    return bool(np.all(a["mask"] | b["mask"]))


def _unique_pairing(components: list[dict]) -> list[tuple[int, int]]:
    n = len(components)
    assert n > 0 and n % 2 == 0, "expected an even number of components"

    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            compat[i][j] = compat[j][i] = _pair_compatible(components[i], components[j])

    matchings: list[list[tuple[int, int]]] = []

    def search(remaining: tuple[int, ...], pairs: list[tuple[int, int]]) -> None:
        if len(matchings) > 1:
            return
        if not remaining:
            matchings.append(pairs.copy())
            return

        i = remaining[0]
        rest = remaining[1:]
        for pos, j in enumerate(rest):
            if compat[i][j]:
                nxt = rest[:pos] + rest[pos + 1 :]
                pairs.append((i, j))
                search(nxt, pairs)
                pairs.pop()

    search(tuple(range(n)), [])
    assert len(matchings) == 1, "component pairing must be unique"
    return matchings[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary colored fragments, merge each pair into a solid rectangle,
    place the rectangle at the larger fragment, and color it with the smaller
    fragment's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    pairs = _unique_pairing(components)

    out = g.copy()
    for i, j in pairs:
        a = components[i]
        b = components[j]
        assert a["area"] != b["area"], "each pair must have a strictly larger fragment"
        large, small = (a, b) if a["area"] > b["area"] else (b, a)

        for r, c in small["coords"]:
            out[r, c] = BG

        r0, c0, r1, c1 = large["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(small["color"])

    return out


def _make_scaffold_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mode = random.choice(["top_bottom", "left_right"])

    if mode == "top_bottom":
        connector = random.randint(1, w - 2)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, connector] = True
        if random.random() < 0.8:
            row = random.randint(1, h - 2)
            if random.random() < 0.5:
                mask[row, : connector + 1] = True
            else:
                mask[row, connector:] = True
    else:
        connector = random.randint(1, h - 2)
        mask[:, 0] = True
        mask[:, -1] = True
        mask[connector, :] = True
        if random.random() < 0.8:
            col = random.randint(1, w - 2)
            if random.random() < 0.5:
                mask[: connector + 1, col] = True
            else:
                mask[connector:, col] = True

    assert _mask_connected(mask)
    assert _touches_full_bbox(mask)
    assert int(mask.sum()) < h * w
    return mask


def _make_dense_mask(scaffold: np.ndarray) -> np.ndarray | None:
    h, w = scaffold.shape
    dense = np.ones((h, w), dtype=bool)
    removable = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if bool(scaffold[r, c])]
    if not removable:
        return None

    random.shuffle(removable)
    max_holes = min(len(removable), max(2, (h * w) // 5))
    n_holes = random.randint(1, max_holes)
    for r, c in removable[:n_holes]:
        dense[r, c] = False

    assert _mask_connected(dense)
    assert _touches_full_bbox(dense)
    return dense


def _build_pair_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PAIR_TRIES):
        scaffold = _make_scaffold_mask(h, w)
        dense = _make_dense_mask(scaffold)
        if dense is None:
            continue
        if not np.all(scaffold | dense):
            continue
        if int(scaffold.sum()) == int(dense.sum()):
            continue
        if int(scaffold.sum()) >= int(dense.sum()):
            continue
        return scaffold, dense
    return None


def _place_boxes(grid_h: int, grid_w: int, dims: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(dims)), key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)
    padded = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(dims)

    for idx in order:
        h, w = dims[idx]
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)

            rr0 = max(0, r0 - BOX_PADDING)
            cc0 = max(0, c0 - BOX_PADDING)
            rr1 = min(grid_h, r0 + h + BOX_PADDING)
            cc1 = min(grid_w, c0 + w + BOX_PADDING)
            if np.any(padded[rr0:rr1, cc0:cc1]):
                continue

            placements[idx] = (r0, c0)
            padded[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _render_component(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                rr = top + r
                cc = left + c
                grid[rr, cc] = color
                coords.append((rr, cc))
    return coords


def _build_expected(grid: np.ndarray, pair_specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in pair_specs:
        a_area = int(spec["mask_a"].sum())
        b_area = int(spec["mask_b"].sum())
        assert a_area != b_area

        if a_area > b_area:
            large_bbox = spec["bbox_a"]
            small_color = int(spec["color_b"])
            small_coords = spec["coords_b"]
        else:
            large_bbox = spec["bbox_b"]
            small_color = int(spec["color_a"])
            small_coords = spec["coords_a"]

        for r, c in small_coords:
            out[r, c] = BG

        r0, c0, r1, c1 = large_bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = small_color

    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_BOUNDS)
    dims = random.sample(RECT_DIMS, pair_count)
    masks: list[tuple[np.ndarray, np.ndarray]] = []
    for h, w in dims:
        pair = _build_pair_masks(h, w)
        if pair is None:
            return None
        masks.append(pair)

    component_dims: list[tuple[int, int]] = []
    for h, w in dims:
        component_dims.extend([(h, w), (h, w)])

    min_h = max(GRID_SIDE_MIN, max(h for h, _ in component_dims) + 2)
    min_w = max(GRID_SIDE_MIN, max(w for _, w in component_dims) + 2)
    grid_h = random.randint(min_h, GRID_SIDE_MAX)
    grid_w = random.randint(min_w, GRID_SIDE_MAX)
    placements = _place_boxes(grid_h, grid_w, component_dims)
    if placements is None:
        return None

    colors = random.sample(COLORS, 2 * pair_count)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    pair_specs: list[dict] = []
    place_iter = iter(placements)
    color_iter = iter(colors)

    for (h, w), (scaffold, dense) in zip(dims, masks):
        if random.random() < 0.5:
            mask_a, mask_b = scaffold, dense
        else:
            mask_a, mask_b = dense, scaffold

        top_a, left_a = next(place_iter)
        top_b, left_b = next(place_iter)
        color_a = int(next(color_iter))
        color_b = int(next(color_iter))

        coords_a = _render_component(grid, mask_a, top_a, left_a, color_a)
        coords_b = _render_component(grid, mask_b, top_b, left_b, color_b)
        pair_specs.append(
            {
                "mask_a": mask_a,
                "mask_b": mask_b,
                "coords_a": coords_a,
                "coords_b": coords_b,
                "bbox_a": (top_a, left_a, top_a + h - 1, left_a + w - 1),
                "bbox_b": (top_b, left_b, top_b + h - 1, left_b + w - 1),
                "color_a": color_a,
                "color_b": color_b,
            }
        )

    expected = _build_expected(grid, pair_specs)

    try:
        actual = main(grid)
        comps = _find_components(grid)
        pairs = _unique_pairing(comps)
    except Exception:
        return None

    if len(comps) != 2 * pair_count or len(pairs) != pair_count:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_TRIES} attempts")
