from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotation_from_marker(marker: tuple[int, int], shape: tuple[int, int]) -> int:
    mr, mc = marker
    h, w = shape
    upper = 2 * mr < h - 1
    lower = 2 * mr > h - 1
    left = 2 * mc < w - 1
    right = 2 * mc > w - 1
    assert upper != lower and left != right, "marker must lie in a proper quadrant"
    if upper and left:
        return 0
    if upper and right:
        return 1
    if lower and right:
        return 2
    return 3


def _relative_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cr, cc = h // 2, w // 2
    return [(int(r - cr), int(c - cc)) for r, c in np.argwhere(mask)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each grey anchor with the adjacent color label, then stamp the matching shape after rotating it by the maroon-marker quadrant code.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    non_anchor_fg = (g != BG) & (g != GREY)
    comps = _components(non_anchor_fg)

    stamps: dict[int, tuple[np.ndarray, int]] = {}
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        if MAROON not in colors:
            continue
        base_colors = colors - {MAROON}
        assert len(base_colors) == 1, "each stamp should have one visible color plus maroon"
        color = next(iter(base_colors))
        assert color not in stamps, "stamp colors should be unique"
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch != BG
        marker_pos = np.argwhere(patch == MAROON)
        assert marker_pos.shape == (1, 2), "each stamp should have one maroon marker"
        marker = (int(marker_pos[0, 0]), int(marker_pos[0, 1]))
        rot_cw = _rotation_from_marker(marker, mask.shape)
        stamps[color] = (mask, rot_cw)

    out = np.full_like(g, BG)
    anchors = np.argwhere(g == GREY)
    assert len(anchors) > 0
    for ar, ac in anchors:
        label_colors = []
        for dr, dc in DIRS4:
            nr, nc = int(ar + dr), int(ac + dc)
            if 0 <= nr < h and 0 <= nc < w:
                color = int(g[nr, nc])
                if color in stamps:
                    label_colors.append(color)
        assert len(label_colors) == 1, "each anchor should have one adjacent color label"
        color = label_colors[0]
        mask, rot_cw = stamps[color]
        rotated = np.rot90(mask, -rot_cw)
        for dr, dc in _relative_cells(rotated):
            rr, cc = int(ar + dr), int(ac + dc)
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = color
        out[int(ar), int(ac)] = GREY

    return out


GEN_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))
GEN_GRID_MIN = 18
GEN_GRID_MAX = 26
GEN_OBJECT_RANGE = (2, 4)
GEN_MAX_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 80
GEN_PADDING = 1


def _quadrant_cells(h: int, w: int, rot_cw: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            upper = 2 * r < h - 1
            lower = 2 * r > h - 1
            left = 2 * c < w - 1
            right = 2 * c > w - 1
            if rot_cw == 0 and upper and left:
                cells.append((r, c))
            elif rot_cw == 1 and upper and right:
                cells.append((r, c))
            elif rot_cw == 2 and lower and right:
                cells.append((r, c))
            elif rot_cw == 3 and lower and left:
                cells.append((r, c))
    return cells


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        options = []
        if r < er:
            options.append((r + 1, c))
        elif r > er:
            options.append((r - 1, c))
        if c < ec:
            options.append((r, c + 1))
        elif c > ec:
            options.append((r, c - 1))
        r, c = random.choice(options)
        path.append((r, c))
    return path


def _grow_mask(mask: np.ndarray, target: int) -> None:
    h, w = mask.shape
    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in DIRS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return
        random.shuffle(frontier := list(frontier))
        mask[frontier[0]] = True


def _make_stamp_mask(rot_cw: int) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        h, w = random.choice(GEN_SHAPES)
        center = (h // 2, w // 2)
        marker = random.choice(_quadrant_cells(h, w, rot_cw))
        mask = np.zeros((h, w), dtype=bool)

        endpoints = {
            marker,
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        }
        for endpoint in endpoints:
            for r, c in _random_path(center, endpoint):
                mask[r, c] = True

        base_area = int(mask.sum())
        lo = max(6, base_area)
        hi = min(h * w - 1, base_area + 5)
        if lo > hi:
            continue
        target = random.randint(lo, hi)
        _grow_mask(mask, target)

        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if int(mask.sum()) < 6:
            continue
        return mask, marker
    return None


def _patch_from_mask(mask: np.ndarray, marker: tuple[int, int], color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    patch[marker] = MAROON
    return patch


def _cells_clear(occupied: np.ndarray, cells: list[tuple[int, int]], padding: int) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - padding)
        r1 = min(h, r + padding + 1)
        c0 = max(0, c - padding)
        c1 = min(w, c + padding + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _mark_cells(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _patch_cells(patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(patch != BG)]


def _rotated_cells(mask: np.ndarray, anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(ar + dr, ac + dc) for dr, dc in _relative_cells(mask)]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _try_place_anchor(
    input_occ: np.ndarray,
    output_occ: np.ndarray,
    rotated_mask: np.ndarray,
) -> tuple[int, int, int] | None:
    h, w = input_occ.shape
    rel = _relative_cells(rotated_mask)
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if all(0 <= r + dr < h and 0 <= c + dc < w for dr, dc in rel)
    ]
    random.shuffle(candidates)
    directions = list(range(len(DIRS4)))
    for ar, ac in candidates:
        out_cells = _rotated_cells(rotated_mask, (ar, ac))
        if not _cells_clear(output_occ, out_cells, GEN_PADDING):
            continue
        random.shuffle(directions)
        for idx in directions:
            dr, dc = DIRS4[idx]
            label = (ar + dr, ac + dc)
            pair = [(ar, ac), label]
            if _cells_clear(input_occ, pair, GEN_PADDING):
                return ar, ac, idx
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    n_obj = random.randint(*GEN_OBJECT_RANGE)

    colors = random.sample(STAMP_COLORS, n_obj)
    rotations = random.sample([0, 1, 2, 3], n_obj)
    specs = []
    for color, rot_cw in zip(colors, rotations):
        made = _make_stamp_mask(rot_cw)
        if made is None:
            return None
        mask, marker = made
        patch = _patch_from_mask(mask, marker, color)
        rotated = np.rot90(mask, -rot_cw)
        specs.append(
            {
                "color": color,
                "rot_cw": rot_cw,
                "patch": patch,
                "rotated": rotated,
                "area": int(rotated.sum()),
            }
        )

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    input_occ = np.zeros((h, w), dtype=bool)
    output_occ = np.zeros((h, w), dtype=bool)

    specs.sort(key=lambda item: (-item["area"], random.random()))
    for spec in specs:
        placed = _try_place_anchor(input_occ, output_occ, spec["rotated"])
        if placed is None:
            return None
        ar, ac, dir_idx = placed
        dr, dc = DIRS4[dir_idx]
        lr, lc = ar + dr, ac + dc
        grid[ar, ac] = GREY
        grid[lr, lc] = spec["color"]
        _mark_cells(input_occ, [(ar, ac), (lr, lc)])

        out_cells = _rotated_cells(spec["rotated"], (ar, ac))
        _paint_cells(expected, out_cells, spec["color"])
        expected[ar, ac] = GREY
        _mark_cells(output_occ, out_cells)

    source_specs = specs[:]
    random.shuffle(source_specs)
    for spec in source_specs:
        patch = spec["patch"]
        ph, pw = patch.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            cells = _patch_cells(patch, r0, c0)
            if _cells_clear(input_occ, cells, GEN_PADDING):
                for rr in range(ph):
                    for cc in range(pw):
                        if patch[rr, cc] != BG:
                            grid[r0 + rr, c0 + cc] = patch[rr, cc]
                _mark_cells(input_occ, cells)
                placed = True
                break
        if not placed:
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
