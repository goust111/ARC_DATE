import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _transform(coords, variant: int) -> tuple[tuple[int, int], ...]:
    transformed = []
    for r, c in coords:
        if variant == 0:
            rr, cc = r, c
        elif variant == 1:
            rr, cc = r, -c
        elif variant == 2:
            rr, cc = -r, c
        elif variant == 3:
            rr, cc = -r, -c
        elif variant == 4:
            rr, cc = c, r
        elif variant == 5:
            rr, cc = c, -r
        elif variant == 6:
            rr, cc = -c, r
        else:
            rr, cc = -c, -r
        transformed.append((int(rr), int(cc)))
    return _normalize(transformed)


def _all_orientations(coords) -> list[tuple[tuple[int, int], ...]]:
    seen = []
    for variant in range(8):
        oriented = _transform(coords, variant)
        if oriented not in seen:
            seen.append(oriented)
    return seen


def _canonical_key(coords) -> tuple[tuple[int, int], ...]:
    return min(_all_orientations(coords))


def _shape_dims(coords) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _separator_columns(grid: np.ndarray, bg: int) -> list[int]:
    height, width = grid.shape
    cols = []
    for c in range(width):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color):
            cols.append(int(c))
    return cols


def _panel_spans(width: int, sep_cols: list[int]) -> list[tuple[int, int]]:
    spans = []
    left = 0
    for c in sorted(sep_cols):
        if left < c:
            spans.append((int(left), int(c)))
        left = c + 1
    if left < width:
        spans.append((int(left), int(width)))
    return spans


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            rel = _normalize(coords)
            rel_rows = [r0 for r0, _ in rel]
            rel_cols = [c0 for _, c0 in rel]
            r0 = min(r for r, _ in coords)
            c0 = min(c for _, c in coords)
            r1 = max(r for r, _ in coords)
            c1 = max(c for _, c in coords)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for rr, cc in coords:
                patch[rr - r0, cc - c0] = color
            components.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "rel_coords": rel,
                    "patch": patch,
                    "area": len(coords),
                    "color": color,
                    "height": max(rel_rows) + 1,
                    "width": max(rel_cols) + 1,
                }
            )
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return components


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    dst[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep either the unique sprites or one repeated sprite per panel, then center the survivors.

    Input structure:
    - Azure background.
    - One-cell maroon separator columns split the grid into equal-width full-height panels.
    - Each panel contains several small 4-connected monochrome sprites.

    Rule:
    - Group panel sprites by canonical shape, invariant to rotation and reflection.
    - Let unique sprites be those whose canonical shape appears once in the panel.
    - Let repeated sprites be those whose canonical shape appears at least twice.
    - If unique_count < repeated_count, keep all unique sprites.
    - Otherwise keep the largest repeated sprite; scan order breaks ties.
    - Translate every kept sprite so its bbox is centered inside the panel and inside the full grid height.
    - Separator columns stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = _get_bg(g)
    sep_cols = _separator_columns(g, bg)

    out = np.full_like(g, bg)
    for c in sep_cols:
        out[:, c] = g[:, c]

    for left, right in _panel_spans(width, sep_cols):
        panel = g[:, left:right]
        components = _find_components(panel, bg)
        if not components:
            continue

        groups: dict[tuple[tuple[int, int], ...], list[dict]] = {}
        for comp in components:
            key = _canonical_key(comp["rel_coords"])
            groups.setdefault(key, []).append(comp)

        unique = []
        repeated = []
        for group in groups.values():
            if len(group) == 1:
                unique.extend(group)
            else:
                repeated.extend(group)

        if len(unique) < len(repeated):
            keepers = unique
        elif repeated:
            keepers = [sorted(repeated, key=lambda comp: (-comp["area"], comp["bbox"][0], comp["bbox"][1]))[0]]
        else:
            keepers = []

        for comp in keepers:
            patch = comp["patch"]
            patch_h, patch_w = patch.shape
            target_r = (height - patch_h) // 2
            target_c = left + (panel.shape[1] - patch_w) // 2
            _paste(out, patch, int(target_r), int(target_c), bg)

    return out


_RNG = np.random.default_rng()
_BG = 8
_SEP = 9
_SPRITE_COLORS = (1, 2, 3, 4, 5, 6, 7)
_MAX_TOTAL_W = 30
_GRID_H_RANGE = (12, 19)
_PANEL_COUNT_CHOICES = (2, 3, 4)
_MIN_PANEL_W = 6
_MAX_GEN_ATTEMPTS = 400
_PANEL_ATTEMPTS = 80
_RAW_SHAPES = (
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
)


def _build_shape_library() -> list[tuple[tuple[int, int], ...]]:
    library = []
    seen = set()
    for raw in _RAW_SHAPES:
        norm = _normalize(raw)
        key = _canonical_key(norm)
        if key in seen:
            continue
        seen.add(key)
        library.append(norm)
    return library


_SHAPES = _build_shape_library()


def _rand_color() -> int:
    return int(_RNG.choice(_SPRITE_COLORS))


def _sample_distinct_shapes(count: int, exclude=()) -> list[tuple[tuple[int, int], ...]]:
    banned = {_canonical_key(shape) for shape in exclude}
    pool = [shape for shape in _SHAPES if _canonical_key(shape) not in banned]
    chosen_idx = _RNG.choice(len(pool), size=count, replace=False)
    return [pool[int(i)] for i in np.atleast_1d(chosen_idx)]


def _random_orientation(shape) -> tuple[tuple[int, int], ...]:
    options = _all_orientations(shape)
    return options[int(_RNG.integers(len(options)))]


def _shape_cells_at(coords, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in coords]


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    blocked[nr, nc] = True


def _place_panel(panel_h: int, panel_w: int, sprites: list[dict]) -> np.ndarray | None:
    panel = np.full((panel_h, panel_w), _BG, dtype=int)
    blocked = np.zeros((panel_h, panel_w), dtype=bool)
    ordered = sorted(
        sprites,
        key=lambda sprite: (-len(sprite["coords"]), -_shape_dims(sprite["coords"])[0], -_shape_dims(sprite["coords"])[1]),
    )

    for sprite in ordered:
        shape = sprite["coords"]
        color = sprite["color"]
        shape_h, shape_w = _shape_dims(shape)
        positions = [(r, c) for r in range(panel_h - shape_h + 1) for c in range(panel_w - shape_w + 1)]
        _RNG.shuffle(positions)

        placed = False
        for top, left in positions:
            cells = _shape_cells_at(shape, int(top), int(left))
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                panel[r, c] = color
            _mark_blocked(blocked, cells)
            placed = True
            break

        if not placed:
            return None

    return panel


def _count_groups(panel: np.ndarray) -> tuple[int, int]:
    comps = _find_components(panel, _BG)
    groups = {}
    for comp in comps:
        key = _canonical_key(comp["rel_coords"])
        groups.setdefault(key, []).append(comp)
    unique = sum(len(group) for group in groups.values() if len(group) == 1)
    repeated = sum(len(group) for group in groups.values() if len(group) >= 2)
    return int(unique), int(repeated)


def _build_panel(panel_h: int, panel_w: int, branch_kind: int) -> np.ndarray | None:
    for _ in range(_PANEL_ATTEMPTS):
        repeated_shape = _SHAPES[int(_RNG.integers(len(_SHAPES)))]
        repeated_copies = 2 if branch_kind == 1 else int(_RNG.choice([2, 3]))
        unique_count = int(_RNG.choice([2, 3])) if branch_kind == 1 else 1
        unique_shapes = _sample_distinct_shapes(unique_count, exclude=[repeated_shape])

        sprites = []
        if branch_kind == 1:
            repeated_orientation = _random_orientation(repeated_shape)
            repeated_color = _rand_color()
            for _copy in range(repeated_copies):
                sprites.append({"coords": repeated_orientation, "color": repeated_color})
        else:
            for _copy in range(repeated_copies):
                coords = _random_orientation(repeated_shape) if _RNG.random() < 0.7 else repeated_shape
                sprites.append({"coords": coords, "color": _rand_color()})

        for shape in unique_shapes:
            sprites.append({"coords": _random_orientation(shape), "color": _rand_color()})

        panel = _place_panel(panel_h, panel_w, sprites)
        if panel is None:
            continue

        unique, repeated = _count_groups(panel)
        if branch_kind == 0 and not (unique == 1 and repeated in (2, 3) and unique < repeated):
            continue
        if branch_kind == 1 and not (repeated == 2 and unique in (2, 3) and unique >= repeated):
            continue
        return panel

    return None


def _panel_center_bbox(panel_w: int, comp: dict, grid_h: int) -> tuple[int, int]:
    patch_h, patch_w = comp["patch"].shape
    return (int((grid_h - patch_h) // 2), int((panel_w - patch_w) // 2))


def _valid_generated_grid(grid: np.ndarray, panel_count: int, panel_w: int) -> bool:
    try:
        if _get_bg(grid) != _BG:
            return False
        expected_seps = [panel_w * (i + 1) + i for i in range(panel_count - 1)]
        if _separator_columns(grid, _BG) != expected_seps:
            return False

        branches_seen = set()
        for left, right in _panel_spans(grid.shape[1], expected_seps):
            panel = grid[:, left:right]
            comps = _find_components(panel, _BG)
            if len(comps) < 3:
                return False
            unique, repeated = _count_groups(panel)
            if unique < repeated:
                branches_seen.add(0)
            elif repeated:
                branches_seen.add(1)
            else:
                return False

        if branches_seen != {0, 1}:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if not np.array_equal(out[:, expected_seps], grid[:, expected_seps]):
            return False

        for left, right in _panel_spans(out.shape[1], expected_seps):
            panel = out[:, left:right]
            comps = _find_components(panel, _BG)
            if len(comps) != 1:
                return False
            comp = comps[0]
            expected_r, expected_c = _panel_center_bbox(panel_w, comp, grid.shape[0])
            if comp["bbox"][0] != expected_r or comp["bbox"][1] != expected_c:
                return False

        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    panel_count = int(_RNG.choice(_PANEL_COUNT_CHOICES))
    panel_w_max = min(10, (_MAX_TOTAL_W - (panel_count - 1)) // panel_count)
    if panel_w_max < _MIN_PANEL_W:
        return None
    panel_w = int(_RNG.integers(_MIN_PANEL_W, panel_w_max + 1))
    grid_h = int(_RNG.integers(_GRID_H_RANGE[0], _GRID_H_RANGE[1]))
    grid_w = panel_count * panel_w + (panel_count - 1)

    branches = [0, 1]
    while len(branches) < panel_count:
        branches.append(int(_RNG.integers(0, 2)))
    _RNG.shuffle(branches)

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    for idx in range(panel_count - 1):
        sep_col = panel_w * (idx + 1) + idx
        grid[:, sep_col] = _SEP

    for idx, branch_kind in enumerate(branches):
        panel = _build_panel(grid_h, panel_w, branch_kind)
        if panel is None:
            return None
        left = idx * (panel_w + 1)
        grid[:, left : left + panel_w] = panel

    if not _valid_generated_grid(grid, panel_count, panel_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
