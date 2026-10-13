import random

import numpy as np


QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    full_rows = [r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    full_cols = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]

    matches = []
    for r in full_rows:
        for c in full_cols:
            if int(grid[r, 0]) == int(grid[0, c]):
                matches.append((int(r), int(c), int(grid[r, 0])))

    assert len(matches) == 1, "expected exactly one full-row/full-col divider"
    return matches[0]


def _quadrant_bounds(mid_r: int, mid_c: int, key: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = 0 if key[0] == 0 else mid_r + 1
    r1 = mid_r if key[0] == 0 else h
    c0 = 0 if key[1] == 0 else mid_c + 1
    c1 = mid_c if key[1] == 0 else w
    return r0, r1, c0, c1


def _lexicographic_first(coords: np.ndarray) -> tuple[int, int]:
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    first = coords[order[0]]
    return int(first[0]), int(first[1])


def _is_connected(coords: np.ndarray) -> bool:
    pts = {tuple(map(int, cell)) for cell in coords.tolist()}
    if not pts:
        return False

    stack = [next(iter(pts))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(pts)


def _transform_kind(src: tuple[int, int], dst: tuple[int, int]) -> str:
    if src[0] == dst[0] and src[1] != dst[1]:
        return "horizontal"
    if src[1] == dst[1] and src[0] != dst[0]:
        return "vertical"
    return "diagonal"


def _transform_global(r: int, c: int, mid_r: int, mid_c: int, kind: str) -> tuple[int, int]:
    if kind == "horizontal":
        return r, 2 * mid_c - c
    if kind == "vertical":
        return 2 * mid_r - r, c
    return 2 * mid_r - r, 2 * mid_c - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the empty quadrant by projecting each source object with the symmetry implied by
    its position relative to the empty quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1

    bg = _get_bg(g)
    mid_r, mid_c, divider = _find_divider(g, bg)

    objects = []
    occupied_quadrants = set()
    for key in QUADRANTS:
        r0, r1, c0, c1 = _quadrant_bounds(mid_r, mid_c, key, g.shape)
        patch = g[r0:r1, c0:c1]
        mask = (patch != bg) & (patch != divider)
        if not mask.any():
            continue

        colors = np.unique(patch[mask])
        assert len(colors) == 1, "each non-empty quadrant must contain exactly one color"

        local = np.argwhere(mask)
        assert 3 <= len(local) <= 6, "objects must remain small"
        assert _is_connected(local), "quadrant object must be 4-connected"

        coords = local + np.array([[r0, c0]], dtype=int)
        objects.append(
            {
                "quadrant": key,
                "color": int(colors[0]),
                "coords": coords,
                "first": _lexicographic_first(coords),
            }
        )
        occupied_quadrants.add(key)

    assert len(objects) == 3, "expected exactly three non-empty quadrants"
    target = next(key for key in QUADRANTS if key not in occupied_quadrants)

    out = g.copy()
    for obj in sorted(objects, key=lambda item: item["first"]):
        kind = _transform_kind(obj["quadrant"], target)
        for r_raw, c_raw in obj["coords"]:
            r = int(r_raw)
            c = int(c_raw)
            rr, cc = _transform_global(r, c, mid_r, mid_c, kind)
            assert 0 <= rr < h and 0 <= cc < w
            if int(out[rr, cc]) == bg:
                out[rr, cc] = int(obj["color"])

    return out


# --------------- generator ---------------

GRID_SIDES = (11, 13, 15, 17)
AREA_RANGE = (3, 6)
MAX_SHAPE_ATTEMPTS = 250
MAX_LAYOUT_ATTEMPTS = 250
MAX_GENERATION_ATTEMPTS = 400


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _transform_local(
    cells: tuple[tuple[int, int], ...], qh: int, qw: int, kind: str
) -> tuple[tuple[int, int], ...]:
    if kind == "horizontal":
        out = {(r, qw - 1 - c) for r, c in cells}
    elif kind == "vertical":
        out = {(qh - 1 - r, c) for r, c in cells}
    else:
        out = {(qh - 1 - r, qw - 1 - c) for r, c in cells}
    return tuple(sorted(out))


def _transform_in_bbox(cells: tuple[tuple[int, int], ...], kind: str) -> tuple[tuple[int, int], ...]:
    h, w = _shape_dims(cells)
    if kind == "horizontal":
        out = {(r, w - 1 - c) for r, c in cells}
    elif kind == "vertical":
        out = {(h - 1 - r, c) for r, c in cells}
    else:
        out = {(h - 1 - r, w - 1 - c) for r, c in cells}
    return tuple(sorted(out))


def _is_asymmetric(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(cells != _transform_in_bbox(cells, kind) for kind in ("horizontal", "vertical", "diagonal"))


def _grow_random_shape(area: int, max_h: int, max_w: int) -> tuple[tuple[int, int], ...] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(2, min(max_h, area))
        w = random.randint(2, min(max_w, area))
        if h * w < area:
            continue

        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = set()
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
            if not frontier:
                break
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            cells.add(nxt)

        if len(cells) != area:
            continue

        shape = _normalize(cells)
        sh, sw = _shape_dims(shape)
        if sh > max_h or sw > max_w:
            continue
        if not _is_asymmetric(shape):
            continue
        return shape
    return None


def _translate(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr, c + dc) for r, c in cells))


def _place_target_shapes(qh: int, qw: int) -> list[tuple[tuple[int, int], ...]] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        bases = []
        for _shape_idx in range(3):
            area = random.randint(*AREA_RANGE)
            base = _grow_random_shape(area, qh, qw)
            if base is None:
                break
            bases.append(base)
        if len(bases) != 3:
            continue

        bases.sort(key=len, reverse=True)
        occupied: set[tuple[int, int]] = set()
        placed = []
        ok = True
        for base in bases:
            sh, sw = _shape_dims(base)
            candidates = []
            for r0 in range(qh - sh + 1):
                for c0 in range(qw - sw + 1):
                    moved = _translate(base, r0, c0)
                    if occupied.isdisjoint(moved):
                        candidates.append(moved)
            if not candidates:
                ok = False
                break
            chosen = random.choice(candidates)
            placed.append(chosen)
            occupied.update(chosen)

        if ok:
            random.shuffle(placed)
            return placed
    return None


def _origin(mid_r: int, mid_c: int, key: tuple[int, int]) -> tuple[int, int]:
    return (0 if key[0] == 0 else mid_r + 1, 0 if key[1] == 0 else mid_c + 1)


def _to_global(cells: tuple[tuple[int, int], ...], origin: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((origin[0] + r, origin[1] + c) for r, c in cells))


def _draw_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _expected_output(
    inp: np.ndarray,
    target: tuple[int, int],
    objects: list[dict[str, object]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for obj in sorted(objects, key=lambda item: item["first"]):
        for r, c in obj["dest_global"]:
            if int(out[r, c]) == bg:
                out[r, c] = int(obj["color"])
    return out


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.choice(GRID_SIDES)
    w = random.choice(GRID_SIDES)
    mid_r = h // 2
    mid_c = w // 2
    qh = mid_r
    qw = mid_c

    roles = random.sample(range(10), 5)
    bg, divider, color_a, color_b, color_c = roles
    colors = [color_a, color_b, color_c]

    target = random.choice(QUADRANTS)
    target_shapes = _place_target_shapes(qh, qw)
    if target_shapes is None:
        return None

    source_quadrants = [key for key in QUADRANTS if key != target]
    random.shuffle(source_quadrants)
    random.shuffle(colors)
    random.shuffle(target_shapes)

    inp = np.full((h, w), bg, dtype=int)
    inp[mid_r, :] = divider
    inp[:, mid_c] = divider

    objects = []
    for source_q, color, dest_local in zip(source_quadrants, colors, target_shapes, strict=False):
        kind = _transform_kind(source_q, target)
        source_local = _transform_local(dest_local, qh, qw, kind)
        source_global = _to_global(source_local, _origin(mid_r, mid_c, source_q))
        dest_global = _to_global(dest_local, _origin(mid_r, mid_c, target))

        _draw_cells(inp, source_global, color)
        source_coords = np.array(source_global, dtype=int)
        objects.append(
            {
                "color": color,
                "source_global": source_global,
                "dest_global": dest_global,
                "first": _lexicographic_first(source_coords),
            }
        )

    expected = _expected_output(inp, target, objects, bg)
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_candidate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
