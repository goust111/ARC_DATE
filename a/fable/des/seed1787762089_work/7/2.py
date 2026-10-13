from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
MAGENTA = 6
AZURE = 8

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SYMMETRY_KINDS = ("lr", "ud", "diag")
MAX_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "mask must be nonempty"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _component_mask(points: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in points:
        out[int(r - r0), int(c - c0)] = True
    return out, (int(r0), int(c0))


def _reflect_anti_diag(mask: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(mask)).T


def _classify_symmetry(mask: np.ndarray) -> str:
    m = np.asarray(mask, dtype=bool)
    lr = np.array_equal(m, np.fliplr(m))
    ud = np.array_equal(m, np.flipud(m))
    diag_main = m.shape[0] == m.shape[1] and np.array_equal(m, m.T)
    diag_anti = m.shape[0] == m.shape[1] and np.array_equal(m, _reflect_anti_diag(m))

    if lr and not ud and not diag_main and not diag_anti:
        return "lr"
    if ud and not lr and not diag_main and not diag_anti:
        return "ud"
    if not lr and not ud and (diag_main ^ diag_anti):
        return "diag"
    raise AssertionError("mask must have exactly one allowed reflection symmetry")


def _touches(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = mask.shape
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _orbit_lr(r: int, c: int, width: int) -> list[tuple[int, int]]:
    mc = width - 1 - c
    if mc == c:
        return [(r, c)]
    return [(r, c), (r, mc)]


def _make_lr_mask(height_choices: tuple[int, ...], width_choices: tuple[int, ...], area_min: int) -> np.ndarray:
    for _ in range(300):
        h = random.choice(height_choices)
        w = random.choice(width_choices)
        if w % 2 == 0:
            continue

        mask = np.zeros((h, w), dtype=bool)
        center = w // 2
        spine_len = random.randint(2, h)
        spine_r0 = random.randint(0, h - spine_len)
        mask[spine_r0 : spine_r0 + spine_len, center] = True

        target_area = random.randint(area_min, max(area_min, min(h * w - 1, area_min + h + w)))
        off_axis_pairs = 0

        for _step in range(h * w * 6):
            if int(mask.sum()) >= target_area and off_axis_pairs > 0 and random.random() < 0.55:
                break

            candidates: list[tuple[int, int]] = []
            preferred: list[tuple[int, int]] = []
            for rr in range(h):
                for cc in range(center + 1):
                    cells = _orbit_lr(rr, cc, w)
                    if any(mask[r, c] for r, c in cells):
                        continue
                    if not _touches(mask, cells):
                        continue
                    candidates.append((rr, cc))
                    if cc != center:
                        preferred.append((rr, cc))
            if not candidates:
                break

            if preferred and (off_axis_pairs == 0 or random.random() < 0.75):
                rr, cc = random.choice(preferred)
            else:
                rr, cc = random.choice(candidates)

            for r, c in _orbit_lr(rr, cc, w):
                mask[r, c] = True
            if cc != center:
                off_axis_pairs += 1

        if int(mask.sum()) < area_min or off_axis_pairs == 0:
            continue

        tight = _tight_mask(mask)
        if tight.shape[0] < 3 or tight.shape[1] < 3:
            continue
        if len(_components(tight)) != 1:
            continue
        try:
            if _classify_symmetry(tight) == "lr":
                return tight
        except AssertionError:
            continue
    raise RuntimeError("failed to generate lr mask")


def _make_ud_mask(height_choices: tuple[int, ...], width_choices: tuple[int, ...], area_min: int) -> np.ndarray:
    for _ in range(200):
        mask = _make_lr_mask(width_choices, height_choices, area_min).T
        if mask.shape[0] not in height_choices or mask.shape[1] not in width_choices:
            continue
        try:
            if _classify_symmetry(mask) == "ud":
                return mask
        except AssertionError:
            continue
    raise RuntimeError("failed to generate ud mask")


def _make_diag_mask(side_choices: tuple[int, ...], area_min: int) -> np.ndarray:
    for _ in range(300):
        n = random.choice(side_choices)
        mask = np.zeros((n, n), dtype=bool)

        for i in range(n):
            mask[i, i] = True
        for i in range(n - 1):
            mask[i, i + 1] = True
            mask[i + 1, i] = True

        candidates = [(r, c) for r in range(n) for c in range(r + 2, n)]
        random.shuffle(candidates)
        extra_target = random.randint(1, max(1, len(candidates)))
        extras = 0

        for r, c in candidates:
            cells = [(r, c), (c, r)]
            if not _touches(mask, cells):
                continue
            for rr, cc in cells:
                mask[rr, cc] = True
            extras += 1
            if extras >= extra_target and random.random() < 0.6:
                break

        if extras == 0 or int(mask.sum()) < area_min:
            continue

        if random.random() < 0.5:
            mask = np.fliplr(mask)

        try:
            if _classify_symmetry(mask) == "diag":
                return mask
        except AssertionError:
            continue
    raise RuntimeError("failed to generate diagonal mask")


def _sample_shape(kind: str, legend: bool) -> np.ndarray:
    if legend:
        if kind == "lr":
            return _make_lr_mask((3, 4), (3,), area_min=5)
        if kind == "ud":
            return _make_ud_mask((3,), (3, 4), area_min=5)
        return _make_diag_mask((4,), area_min=6)

    if kind == "lr":
        return _make_lr_mask((4, 5, 6, 7), (3, 5), area_min=5)
    if kind == "ud":
        return _make_ud_mask((3, 5), (4, 5, 6, 7), area_min=5)
    return _make_diag_mask((4, 5, 6), area_min=7)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = np.where(np.all(g == GREY, axis=1))[0]
    assert len(sep_rows) == 1, "expected exactly one solid grey separator row"
    sep = int(sep_rows[0])

    legend = g[:sep]
    main_area = g[sep + 1 :]
    assert legend.size and main_area.size, "legend and main area must both exist"

    legend_comps = _components(legend != BLACK)
    assert len(legend_comps) == 3, "expected three legend samples"
    sym_to_color: dict[str, int] = {}
    for pts in legend_comps:
        colors = {int(legend[r, c]) for r, c in pts}
        assert len(colors) == 1, "each legend sample must be monochrome"
        mask, _ = _component_mask(pts)
        kind = _classify_symmetry(mask)
        assert kind not in sym_to_color, "legend symmetries must be unique"
        sym_to_color[kind] = int(next(iter(colors)))
    assert set(sym_to_color) == set(SYMMETRY_KINDS)

    out = np.full(main_area.shape, BLACK, dtype=int)
    object_comps = _components(main_area == AZURE)
    assert object_comps, "expected at least one azure object"

    for pts in object_comps:
        mask, _ = _component_mask(pts)
        color = sym_to_color[_classify_symmetry(mask)]
        for r, c in pts:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < main_area.shape[0] and 0 <= nc < main_area.shape[1] and main_area[nr, nc] == BLACK:
                    assert out[nr, nc] in (BLACK, color), "outline colors must not conflict"
                    out[nr, nc] = color

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _place_legend(grid: np.ndarray, shapes: list[np.ndarray], colors: list[int]) -> bool:
    legend_h, width = grid.shape
    widths = [int(shape.shape[1]) for shape in shapes]
    required = sum(widths) + (len(shapes) - 1)
    extra = width - required
    if extra < 0:
        return False

    offsets = _random_partition(extra, len(shapes) + 1)
    cursor = offsets[0]
    for idx, (shape, color) in enumerate(zip(shapes, colors)):
        h, w = map(int, shape.shape)
        if h > legend_h:
            return False
        r0 = random.randint(0, legend_h - h)
        grid[r0 : r0 + h, cursor : cursor + w][shape] = color
        cursor += w + 1 + offsets[idx + 1]
    return True


def _place_objects(shapes: list[tuple[str, np.ndarray]], height: int, width: int) -> tuple[np.ndarray, list[tuple[str, np.ndarray, int, int]]] | None:
    if height < 3 or width < 3:
        return None

    board = np.full((height, width), BLACK, dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)
    placements: list[tuple[str, np.ndarray, int, int]] = []

    for kind, shape in sorted(shapes, key=lambda item: int(item[1].sum()), reverse=True):
        h, w = map(int, shape.shape)
        if h > height - 2 or w > width - 2:
            return None

        placed = False
        cells = np.argwhere(shape)
        for _ in range(250):
            r0 = random.randint(1, height - h - 1)
            c0 = random.randint(1, width - w - 1)

            zone: set[tuple[int, int]] = set()
            ok = True
            for r, c in cells:
                br, bc = r0 + int(r), c0 + int(c)
                if forbidden[br, bc]:
                    ok = False
                    break
                zone.add((br, bc))
                for dr, dc in DIR4:
                    nr, nc = br + dr, bc + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if forbidden[nr, nc]:
                            ok = False
                            break
                        zone.add((nr, nc))
                if not ok:
                    break
            if not ok:
                continue

            for r, c in cells:
                board[r0 + int(r), c0 + int(c)] = AZURE
            for zr, zc in zone:
                forbidden[zr, zc] = True
            placements.append((kind, shape, r0, c0))
            placed = True
            break

        if not placed:
            return None

    return board, placements


def _expected_output(main_shape: tuple[int, int], placements: list[tuple[str, np.ndarray, int, int]], sym_to_color: dict[str, int]) -> np.ndarray | None:
    height, width = main_shape
    object_mask = np.zeros((height, width), dtype=bool)
    out = np.full((height, width), BLACK, dtype=int)

    for _kind, shape, r0, c0 in placements:
        for r, c in np.argwhere(shape):
            object_mask[r0 + int(r), c0 + int(c)] = True

    for kind, shape, r0, c0 in placements:
        color = sym_to_color[kind]
        for r, c in np.argwhere(shape):
            br, bc = r0 + int(r), c0 + int(c)
            for dr, dc in DIR4:
                nr, nc = br + dr, bc + dc
                if 0 <= nr < height and 0 <= nc < width and not object_mask[nr, nc]:
                    if out[nr, nc] not in (BLACK, color):
                        return None
                    out[nr, nc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(15, 20)
    legend_h = random.choice((4, 5))
    main_h = height - legend_h - 1
    if main_h < 11:
        return None

    legend_kinds = random.sample(list(SYMMETRY_KINDS), k=3)
    legend_colors = [RED, BLUE, MAGENTA]
    sym_to_color = {kind: color for kind, color in zip(legend_kinds, legend_colors)}

    try:
        legend_shapes = [_sample_shape(kind, legend=True) for kind in legend_kinds]
    except RuntimeError:
        return None

    object_count = random.randint(2, 4)
    object_kinds = [random.choice(SYMMETRY_KINDS) for _ in range(object_count)]
    try:
        object_shapes = [(kind, _sample_shape(kind, legend=False)) for kind in object_kinds]
    except RuntimeError:
        return None

    placed = _place_objects(object_shapes, main_h, width)
    if placed is None:
        return None
    main_board, placements = placed

    expected = _expected_output((main_h, width), placements, sym_to_color)
    if expected is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    if not _place_legend(grid[:legend_h], legend_shapes, legend_colors):
        return None
    grid[legend_h, :] = GREY
    grid[legend_h + 1 :, :] = main_board

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
