from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Palette order: left-right mirror only, top-bottom mirror only, both.
LR_ONLY_INDEX = 0
UD_ONLY_INDEX = 1
BOTH_INDEX = 2

# Generator bounds.
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PLACE_ATTEMPTS = 120
GRID_MIN = 14
GRID_MAX = 30
OBJECT_COUNT_CHOICES = (4, 5, 6)
SHAPE_SIZE_RANGE = (3, 7)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _has_lr_symmetry(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _has_ud_symmetry(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)))


def _find_palette(components: list[dict[str, object]]) -> tuple[list[int], set[tuple[int, int]]]:
    by_col: dict[int, list[tuple[int, int]]] = {}
    for comp in components:
        if int(comp["area"]) != 1:
            continue
        coord = np.asarray(comp["coords"], dtype=int)[0]
        r = int(coord[0])
        c = int(coord[1])
        by_col.setdefault(c, []).append((r, int(comp["color"])))

    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for c, items in by_col.items():
        items = sorted(items)
        if len(items) != 3:
            continue
        if [items[1][0] - items[0][0], items[2][0] - items[1][0]] != [2, 2]:
            continue
        if len({color for _r, color in items}) != 3:
            continue
        candidates.append((c, items))

    assert candidates, "expected a right-side palette made of three isolated pixels"
    palette_col = max(candidates, key=lambda item: item[0])[0]
    matches = [items for c, items in candidates if c == palette_col]
    assert len(matches) == 1, "palette column must be unique"
    ordered = matches[0]
    colors = [color for _r, color in ordered]
    cells = {(r, palette_col) for r, _color in ordered}
    return colors, cells


def _paint_border(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    bg: int,
    protected: set[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    h, w = out.shape
    assert r0 > 0 and c0 > 0 and r1 + 1 < h and c1 + 1 < w

    for r in range(r0 - 1, r1 + 2):
        for c in (c0 - 1, c1 + 1):
            assert (r, c) not in protected
            current = int(out[r, c])
            assert current in {bg, color}
            out[r, c] = color

    for c in range(c0 - 1, c1 + 2):
        for r in (r0 - 1, r1 + 1):
            assert (r, c) not in protected
            current = int(out[r, c])
            assert current in {bg, color}
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a color-coded outer bounding box around each object according to its reflection symmetries.

    Input structure:
    - The background is the dominant color, usually black.
    - Several scattered foreground objects appear as monochrome 4-connected shapes,
      and different objects use different colors.
    - On the right side there is one vertical palette column made of three single
      pixels separated by one background row.

    Rule:
    - Read the palette from top to bottom as the border colors for shapes that are
      left-right symmetric only, top-bottom symmetric only, and symmetric in both
      directions.
    - For each non-palette object, crop its bounding box and test whether its mask
      matches its left-right mirror and/or its top-bottom mirror.
    - If exactly one symmetry type matches, draw a one-pixel-wide border just
      outside the object's bounding box using the corresponding palette color.
    - If both symmetries match, use the third palette color; if neither matches,
      leave the object unchanged.
    - Keep the palette itself unchanged.

    Key insight:
    - The right-side palette is an ordered legend for symmetry classes, so the task
      is to classify each object's mirror behavior first and only then map that
      class to the correct border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    all_components = _find_components(g, bg)
    palette_colors, palette_cells = _find_palette(all_components)

    working = g.copy()
    for r, c in palette_cells:
        working[r, c] = bg

    objects = _find_components(working, bg)
    assert objects, "expected at least one non-palette object"

    out = g.copy()
    for comp in objects:
        mask = _mask_from_coords(np.asarray(comp["coords"], dtype=int))
        lr = _has_lr_symmetry(mask)
        ud = _has_ud_symmetry(mask)

        if lr and ud:
            border_color = int(palette_colors[BOTH_INDEX])
        elif lr:
            border_color = int(palette_colors[LR_ONLY_INDEX])
        elif ud:
            border_color = int(palette_colors[UD_ONLY_INDEX])
        else:
            continue

        _paint_border(out, tuple(map(int, comp["bbox"])), border_color, bg, palette_cells)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _classify_mask(mask: np.ndarray) -> str:
    lr = _has_lr_symmetry(mask)
    ud = _has_ud_symmetry(mask)
    if lr and ud:
        return "both"
    if lr:
        return "lr"
    if ud:
        return "ud"
    return "none"


def _symmetry_orbits(height: int, width: int, kind: str) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            if kind == "lr":
                cells = {(r, c), (r, width - 1 - c)}
            elif kind == "ud":
                cells = {(r, c), (height - 1 - r, c)}
            else:
                cells = {
                    (r, c),
                    (r, width - 1 - c),
                    (height - 1 - r, c),
                    (height - 1 - r, width - 1 - c),
                }
            orbit = tuple(sorted(cells))
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _orbit_adjacency(orbits: list[tuple[tuple[int, int], ...]]) -> list[set[int]]:
    owner: dict[tuple[int, int], int] = {}
    for idx, orbit in enumerate(orbits):
        for cell in orbit:
            owner[cell] = idx

    adjacency = [set() for _ in orbits]
    for idx, orbit in enumerate(orbits):
        for r, c in orbit:
            for dr, dc in DIR4:
                other = owner.get((r + dr, c + dc))
                if other is not None and other != idx:
                    adjacency[idx].add(other)
                    adjacency[other].add(idx)
    return adjacency


def _center_rows(size: int) -> set[int]:
    return {size // 2} if size % 2 else {size // 2 - 1, size // 2}


def _seed_orbits(orbits: list[tuple[tuple[int, int], ...]], kind: str, height: int, width: int) -> list[int]:
    center_r = _center_rows(height)
    center_c = _center_rows(width)
    seeds: list[int] = []
    for idx, orbit in enumerate(orbits):
        rows = {r for r, _c in orbit}
        cols = {c for _r, c in orbit}
        if kind == "lr" and cols & center_c:
            seeds.append(idx)
        elif kind == "ud" and rows & center_r:
            seeds.append(idx)
        elif kind == "both" and rows & center_r and cols & center_c:
            seeds.append(idx)
    return seeds


def _orbit_mask(height: int, width: int, chosen: set[int], orbits: list[tuple[tuple[int, int], ...]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for idx in chosen:
        for r, c in orbits[idx]:
            mask[r, c] = True
    return _crop_mask(mask)


def _generate_symmetric_mask(kind: str) -> np.ndarray | None:
    assert kind in {"lr", "ud", "both"}
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(*SHAPE_SIZE_RANGE)
        width = random.randint(*SHAPE_SIZE_RANGE)
        orbits = _symmetry_orbits(height, width, kind)
        adjacency = _orbit_adjacency(orbits)
        seeds = _seed_orbits(orbits, kind, height, width)
        if not seeds:
            continue

        min_orbits = 1 if kind == "both" else 2
        max_orbits = min(len(orbits), max(min_orbits, 1 + (height * width) // 4))
        target = random.randint(min_orbits, max_orbits)
        chosen = {random.choice(seeds)}
        frontier = set(adjacency[next(iter(chosen))])

        while frontier and len(chosen) < target:
            idx = random.choice(tuple(frontier))
            frontier.remove(idx)
            chosen.add(idx)
            frontier |= adjacency[idx] - chosen

        mask = _orbit_mask(height, width, chosen, orbits)
        if int(mask.sum()) < 3 or not _mask_connected(mask):
            continue
        if _classify_mask(mask) == kind:
            return mask
    return None


def _generate_asymmetric_mask() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(*SHAPE_SIZE_RANGE)
        width = random.randint(*SHAPE_SIZE_RANGE)
        start = (random.randrange(height), random.randrange(width))
        chosen = {start}
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            r, c = cell
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    frontier.add((nr, nc))

        add_frontier(start)
        target = random.randint(3, min(height * width, 9))
        while frontier and len(chosen) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            chosen.add(cell)
            add_frontier(cell)

        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if int(mask.sum()) < 3 or not _mask_connected(mask):
            continue
        if _classify_mask(mask) == "none":
            return mask
    return None


def _fallback_mask(kind: str) -> np.ndarray:
    if kind == "lr":
        return np.array(
            [
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )
    if kind == "ud":
        return np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 1, 0],
            ],
            dtype=bool,
        )
    if kind == "both":
        return np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )
    return np.array(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=bool,
    )


def _make_mask(kind: str) -> np.ndarray:
    if kind == "none":
        mask = _generate_asymmetric_mask()
    else:
        mask = _generate_symmetric_mask(kind)
    if mask is None:
        mask = _fallback_mask(kind)
    assert _classify_mask(mask) == kind
    return mask


def _place_mask(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    palette_col: int,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    rows = list(range(1, h - mh))
    cols = list(range(1, w - mw - 1))
    random.shuffle(rows)
    random.shuffle(cols)

    for r0 in rows:
        for c0 in cols:
            if c0 + mw >= palette_col:
                continue
            footprint = occupied[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1]
            if footprint.any():
                continue
            coords = np.argwhere(mask)
            grid[r0 + coords[:, 0], c0 + coords[:, 1]] = color
            occupied[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1] = True
            return True
    return False


def _build_input() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    if height < 5 or width < 6:
        return None

    grid = np.zeros((height, width), dtype=int)
    palette_col = width - 1
    palette_top = random.randint(0, height - 5)
    object_count = random.choice(OBJECT_COUNT_CHOICES)

    available_colors = list(range(1, 10))
    random.shuffle(available_colors)
    palette_colors = available_colors[:3]
    object_colors = available_colors[3 : 3 + object_count]
    if len(object_colors) != object_count:
        return None

    palette_rows = [palette_top + 2 * i for i in range(3)]
    for r, color in zip(palette_rows, palette_colors):
        grid[r, palette_col] = color

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:, palette_col] = True
    kinds = ["lr", "ud", "both", "none"]
    while len(kinds) < object_count:
        kinds.append(random.choice(["lr", "ud", "both", "none"]))
    random.shuffle(kinds)

    masks = [_make_mask(kind) for kind in kinds]
    order = list(range(object_count))
    order.sort(key=lambda idx: masks[idx].shape[0] * masks[idx].shape[1], reverse=True)
    for idx in order:
        if not _place_mask(grid, occupied, masks[idx], object_colors[idx], palette_col):
            return None

    if len(_find_components(grid, 0)) != object_count + 3:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
