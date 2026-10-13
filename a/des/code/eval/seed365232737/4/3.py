from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
SEPARATOR = 8
COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
ROTATION = "rotation"
NONE = "none"
TYPES = (HORIZONTAL, VERTICAL, ROTATION, NONE)

ROW_COUNT = 3
COL_CHOICES = (3, 4)
ROW_HEIGHT_RANGE = (7, 9)
COL_WIDTH_CHOICES = (6, 7)
MAX_GRID_SIDE = 30
GENERATION_ATTEMPTS = 300
MASK_ATTEMPTS = 200


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(separators) + [length]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            spans.append((left + 1, right))
    return spans


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = np.argwhere(mask)
    assert len(points) > 0, "pattern must contain at least one pixel"
    (r0, c0), (r1, c1) = points.min(axis=0), points.max(axis=0)
    return int(r0), int(c0), int(r1) + 1, int(c1) + 1


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _tight_bbox(mask)
    return mask[r0:r1, c0:c1].copy()


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(map(int, points[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _classify_mask(mask: np.ndarray) -> str:
    tight = _tight_mask(mask)
    if np.array_equal(tight, tight[::-1, :]):
        return HORIZONTAL
    if np.array_equal(tight, tight[:, ::-1]):
        return VERTICAL
    if np.array_equal(tight, tight[::-1, ::-1]):
        return ROTATION
    return NONE


def _fixed_points(mask: np.ndarray, kind: str) -> np.ndarray:
    fixed = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    if kind == HORIZONTAL and h % 2 == 1:
        fixed[h // 2, :] = mask[h // 2, :]
    elif kind == VERTICAL and w % 2 == 1:
        fixed[:, w // 2] = mask[:, w // 2]
    elif kind == ROTATION and h % 2 == 1 and w % 2 == 1 and mask[h // 2, w // 2]:
        fixed[h // 2, w // 2] = True
    return fixed


def _palette_pairs_from_region(region: np.ndarray) -> list[tuple[int, int]] | None:
    points = np.argwhere(region != BLACK)
    if len(points) != 8 or np.any(region == GREY):
        return None
    rows = sorted({int(r) for r, _ in points})
    cols = sorted({int(c) for _, c in points})
    if len(rows) != 4 or len(cols) != 2:
        return None
    if any(int(np.count_nonzero(region[r, :])) != 2 for r in rows):
        return None
    if int(np.count_nonzero(region)) != 8:
        return None
    pairs: list[tuple[int, int]] = []
    left, right = cols
    for r in rows:
        c1 = int(region[r, left])
        c2 = int(region[r, right])
        if c1 in {BLACK, GREY, SEPARATOR} or c2 in {BLACK, GREY, SEPARATOR} or c1 == c2:
            return None
        pairs.append((c1, c2))
    return pairs


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    rots = [np.rot90(mask, k) for k in range(4)]
    return rots + [np.fliplr(x) for x in rots]


def _pick_transformed(mask: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind in {HORIZONTAL, VERTICAL}:
        variants = [mask, np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    else:
        variants = _transforms(mask)
    return variants[int(rng.integers(len(variants)))].copy()


def _render_example(mask: np.ndarray, kind: str, pair: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    first, second = map(int, pair)
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = second
    fixed = _fixed_points(mask, kind)
    if fixed.any():
        out[fixed] = first
    else:
        coords = np.argwhere(mask)
        rr, cc = map(int, coords[int(rng.integers(len(coords)))])
        out[rr, cc] = first
    assert np.any(out == first) and np.any(out == second)
    return out


def _render_test(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = GREY
    return out


def _place_sprite(region: np.ndarray, sprite: np.ndarray, rng: np.random.Generator) -> None:
    sh, sw = sprite.shape
    rh, rw = region.shape
    assert sh <= rh and sw <= rw
    top = int(rng.integers(rh - sh + 1))
    left = int(rng.integers(rw - sw + 1))
    patch = region[top : top + sh, left : left + sw]
    mask = sprite != BLACK
    patch[mask] = sprite[mask]


def _orbit(kind: str, r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    if kind == HORIZONTAL:
        points = {(r, c), (h - 1 - r, c)}
    elif kind == VERTICAL:
        points = {(r, c), (r, w - 1 - c)}
    elif kind == ROTATION:
        points = {(r, c), (h - 1 - r, w - 1 - c)}
    else:
        raise ValueError(kind)
    return sorted(points)


def _frontier_orbits(mask: np.ndarray, kind: str) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[tuple[int, int], ...]] = set()
    frontier: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                orbit = tuple(_orbit(kind, nr, nc, h, w))
                if orbit not in seen:
                    seen.add(orbit)
                    frontier.append(list(orbit))
    return frontier


def _odd_target(rng: np.random.Generator, lo: int, hi: int) -> int:
    choices = [n for n in range(lo, hi + 1) if n % 2 == 1]
    return int(choices[int(rng.integers(len(choices)))])


def _build_symmetric_mask(rng: np.random.Generator, kind: str, h: int, w: int) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        target = _odd_target(rng, 5, min(11, h * w))
        mask = np.zeros((h, w), dtype=bool)
        if kind == HORIZONTAL:
            seed = _orbit(kind, h // 2, int(rng.integers(w)), h, w)
        elif kind == VERTICAL:
            seed = _orbit(kind, int(rng.integers(h)), w // 2, h, w)
        elif kind == ROTATION:
            seed = [(h // 2, w // 2)]
        else:
            raise ValueError(kind)
        for rr, cc in seed:
            mask[rr, cc] = True

        for _step in range(100):
            if int(mask.sum()) >= target:
                break
            frontier = _frontier_orbits(mask, kind)
            if not frontier:
                break
            order = rng.permutation(len(frontier))
            added = False
            for idx in order:
                orbit = frontier[int(idx)]
                new_cells = [(rr, cc) for rr, cc in orbit if not mask[rr, cc]]
                if not new_cells:
                    continue
                if int(mask.sum()) + len(new_cells) > target + 1:
                    continue
                for rr, cc in new_cells:
                    mask[rr, cc] = True
                added = True
                break
            if not added:
                orbit = frontier[int(order[0])]
                for rr, cc in orbit:
                    mask[rr, cc] = True

        tight = _tight_mask(mask)
        fixed = _fixed_points(tight, kind)
        if not fixed.any():
            continue
        if np.count_nonzero(tight & ~fixed) == 0:
            continue
        if _connected(tight) and _classify_mask(tight) == kind:
            return tight
    raise RuntimeError(f"failed to generate {kind} mask")


def _build_none_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        target = int(rng.integers(5, min(10, h * w) + 1))
        mask = np.zeros((h, w), dtype=bool)
        sr = int(rng.integers(h))
        sc = int(rng.integers(w))
        mask[sr, sc] = True
        for _step in range(100):
            if int(mask.sum()) >= target:
                break
            frontier: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                r = int(r)
                c = int(c)
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True
        tight = _tight_mask(mask)
        if _connected(tight) and _classify_mask(tight) == NONE:
            return tight
    raise RuntimeError("failed to generate asymmetric mask")


def _make_example_masks(rng: np.random.Generator) -> dict[str, np.ndarray]:
    dims = {
        HORIZONTAL: (5, int(rng.choice([4, 5]))),
        VERTICAL: (int(rng.choice([4, 5])), 5),
        ROTATION: (int(rng.choice([3, 5])), int(rng.choice([3, 5]))),
        NONE: (int(rng.choice([4, 5])), int(rng.choice([4, 5]))),
    }
    masks: dict[str, np.ndarray] = {}
    for kind, (h, w) in dims.items():
        if kind == NONE:
            masks[kind] = _build_none_mask(rng, h, w)
        else:
            masks[kind] = _build_symmetric_mask(rng, kind, h, w)
    return masks


def _sample_palette_pairs(rng: np.random.Generator) -> dict[str, tuple[int, int]]:
    unordered = [(a, b) for i, a in enumerate(COLOR_POOL) for b in COLOR_POOL[i + 1 :]]
    chosen = rng.choice(len(unordered), size=4, replace=False)
    pairs: dict[str, tuple[int, int]] = {}
    for kind, idx in zip(TYPES, chosen):
        a, b = unordered[int(idx)]
        if int(rng.integers(2)) == 1:
            a, b = b, a
        pairs[kind] = (int(a), int(b))
    return pairs


def _render_palette_region(shape: tuple[int, int], pairs: dict[str, tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    region = np.zeros((h, w), dtype=int)
    start_r = int(rng.integers(h - 6))
    rows = [start_r + 2 * i for i in range(4)]
    gap = 2 if w >= 4 else 1
    start_c = int(rng.integers(w - gap))
    cols = [start_c, start_c + gap]
    for row, kind in zip(rows, TYPES):
        c1, c2 = pairs[kind]
        region[row, cols[0]] = c1
        region[row, cols[1]] = c2
    return region


def _cell_origins(sizes: list[int]) -> list[int]:
    origins: list[int] = []
    cursor = 0
    for i, size in enumerate(sizes):
        origins.append(cursor)
        cursor += size
        if i != len(sizes) - 1:
            cursor += 1
    return origins


def _build_region_layout(rng: np.random.Generator) -> tuple[list[int], list[int]] | None:
    col_count = int(rng.choice(COL_CHOICES))
    row_heights = [int(rng.integers(ROW_HEIGHT_RANGE[0], ROW_HEIGHT_RANGE[1] + 1)) for _ in range(ROW_COUNT)]
    col_widths = [int(rng.choice(COL_WIDTH_CHOICES)) for _ in range(col_count)]
    total_h = sum(row_heights) + ROW_COUNT - 1
    total_w = sum(col_widths) + col_count - 1
    if total_h > MAX_GRID_SIDE or total_w > MAX_GRID_SIDE:
        return None
    return row_heights, col_widths


def _roles_for_grid(cell_count: int, rng: np.random.Generator) -> list[tuple[str, str | None]]:
    roles: list[tuple[str, str | None]] = [("palette", None)]
    roles.extend(("example", kind) for kind in TYPES)
    tests_needed = cell_count - len(roles)
    cycle: list[str] = []
    while len(cycle) < tests_needed:
        batch = list(TYPES)
        rng.shuffle(batch)
        cycle.extend(batch)
    roles.extend(("test", kind) for kind in cycle[:tests_needed])
    return roles


def _make_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layout = _build_region_layout(rng)
    if layout is None:
        return None
    row_heights, col_widths = layout
    row_origins = _cell_origins(row_heights)
    col_origins = _cell_origins(col_widths)
    total_h = sum(row_heights) + len(row_heights) - 1
    total_w = sum(col_widths) + len(col_widths) - 1
    grid = np.zeros((total_h, total_w), dtype=int)

    for r in range(1, len(row_heights)):
        sep_row = row_origins[r] - 1
        grid[sep_row, :] = SEPARATOR
    for c in range(1, len(col_widths)):
        sep_col = col_origins[c] - 1
        grid[:, sep_col] = SEPARATOR

    pairs = _sample_palette_pairs(rng)
    example_masks = _make_example_masks(rng)
    placed_example_masks = {kind: _pick_transformed(mask, kind, rng) for kind, mask in example_masks.items()}

    cells = [(ri, ci) for ri in range(len(row_heights)) for ci in range(len(col_widths))]
    rng.shuffle(cells)
    roles = _roles_for_grid(len(cells), rng)
    for (ri, ci), (role, kind) in zip(cells, roles):
        r0 = row_origins[ri]
        c0 = col_origins[ci]
        region = grid[r0 : r0 + row_heights[ri], c0 : c0 + col_widths[ci]]
        if role == "palette":
            region[:, :] = _render_palette_region(region.shape, pairs, rng)
            continue
        assert kind is not None
        if role == "example":
            mask = placed_example_masks[kind]
            sprite = _render_example(mask, kind, pairs[kind], rng)
        else:
            mask = _pick_transformed(placed_example_masks[kind], kind, rng)
            sprite = _render_test(mask)
        _place_sprite(region, sprite, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.any(out == GREY) or np.array_equal(out, grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_spans = _spans(h, [r for r in range(h) if np.all(g[r, :] == SEPARATOR)])
    col_spans = _spans(w, [c for c in range(w) if np.all(g[:, c] == SEPARATOR)])
    assert len(row_spans) * len(col_spans) >= 5, "expected at least five regions"

    palette_pairs: dict[str, tuple[int, int]] | None = None
    palette_region: tuple[int, int, int, int] | None = None
    examples: list[tuple[np.ndarray, str]] = []
    tests: list[tuple[tuple[int, int, int, int], np.ndarray, str]] = []

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0:r1, c0:c1]
            palette = _palette_pairs_from_region(region)
            if palette is not None:
                assert palette_pairs is None, "palette region must be unique"
                palette_pairs = {kind: pair for kind, pair in zip(TYPES, palette)}
                palette_region = (r0, r1, c0, c1)
                continue

            mask = region != BLACK
            assert mask.any(), "every non-palette region must contain a pattern"
            tight = _tight_mask(mask)
            kind = _classify_mask(tight)
            if np.any(region == GREY):
                assert set(map(int, np.unique(region))).issubset({BLACK, GREY})
                tests.append(((r0, r1, c0, c1), tight, kind))
            else:
                colors = {int(x) for x in np.unique(region) if int(x) != BLACK}
                assert GREY not in colors and len(colors) == 2
                examples.append((region, kind))

    assert palette_pairs is not None and palette_region is not None, "missing palette region"
    assert tests, "expected at least one grey test region"
    for region, kind in examples:
        colors = {int(x) for x in np.unique(region) if int(x) != BLACK}
        assert colors == set(palette_pairs[kind]), "example colors must match palette pair"

    out = g.copy()
    for (r0, r1, c0, c1), tight, kind in tests:
        first, second = palette_pairs[kind]
        colored = np.zeros((r1 - r0, c1 - c0), dtype=int)
        rr0, cc0, rr1, cc1 = _tight_bbox(g[r0:r1, c0:c1] != BLACK)
        fixed = _fixed_points(tight, kind)
        colored_block = np.zeros_like(tight, dtype=int)
        colored_block[tight] = second
        colored_block[fixed] = first
        colored[rr0:rr1, cc0:cc1] = colored_block
        out[r0:r1, c0:c1] = np.where(colored != BLACK, colored, out[r0:r1, c0:c1])
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-palette task")
