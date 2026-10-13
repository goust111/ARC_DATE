from __future__ import annotations

from collections import Counter

import numpy as np


GREY = 5
_BG = 0
_STAMP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_PAIR_COUNTS = [2, 3, 4]
_SHAPE_AREA_RANGE = (3, 7)  # high exclusive
_GRID_MIN_BY_PAIRS = {2: 24, 3: 27, 4: 30}
_GRID_MAX_BY_PAIRS = {2: 30, 3: 34, 4: 38}
_STAMP_SIDE_MIN = 5
_STAMP_EXTRA_MARGIN_MAX = 3
_PLACEMENT_PADDING = 1
_GEN_ATTEMPTS = 400
_SHAPE_ATTEMPTS = 200

_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid).ravel())
    return int(max(counts, key=counts.get))


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(rel)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _extract_stamp_holes(
    grid: np.ndarray,
    coords: np.ndarray,
    bg: int,
) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    hole_mask = crop == int(bg)
    holes = []
    for hole in _components_mask(hole_mask):
        if not _touches_edge(hole, hole_mask.shape):
            holes.append(hole)
    return holes


def _scene_records(grid: np.ndarray) -> tuple[dict[tuple[tuple[int, int], ...], np.ndarray], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    bases: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for comp in _components_mask(g == GREY):
        key = _shape_key(comp)
        assert key not in bases, "base shapes must be unique"
        bases[key] = comp.copy()

    stamps: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {bg, GREY}):
        for comp in _components_mask(g == int(color)):
            holes = _extract_stamp_holes(g, comp, bg)
            if not holes:
                continue
            assert len(holes) == 1, "each stamp should contain exactly one enclosed hole"
            stamps.append({"color": int(color), "hole_key": _shape_key(holes[0])})

    assert bases, "expected at least one grey base"
    assert stamps, "expected at least one stamp"
    assert len(bases) == len(stamps), "stamp/base counts must match"
    assert {stamp["hole_key"] for stamp in stamps} == set(bases), "hole shapes must match bases"
    return bases, stamps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each stamp's internal hole to the grey base with the same shape, then recolor that base.

    Input structure
    - Black background.
    - Several small grey connected shapes ("bases").
    - Several larger monochrome rectangular stamps whose enclosed background hole
      matches one base shape exactly, up to translation only.

    Rule
    - Extract each stamp's enclosed hole shape.
    - Find the grey base with the same normalized relative coordinates.
    - Recolor that base to the stamp's color.
    - Leave the stamps unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bases, stamps = _scene_records(g)

    for stamp in stamps:
        coords = bases[stamp["hole_key"]]
        out[coords[:, 0], coords[:, 1]] = int(stamp["color"])

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    rows = coords[:, 0]
    cols = coords[:, 1]
    coords = coords - np.array([int(rows.min()), int(cols.min())], dtype=int)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    out = np.zeros((h, w), dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _random_shape_mask(area: int) -> np.ndarray | None:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < int(area):
        if not frontier:
            return None
        pick = tuple(frontier)[int(_RNG.integers(len(frontier)))]
        frontier.remove(pick)
        cells.add((int(pick[0]), int(pick[1])))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (int(pick[0] + dr), int(pick[1] + dc))
            if nxt not in cells:
                frontier.add(nxt)

    coords = np.asarray(sorted(cells), dtype=int)
    coords = coords - np.array([int(coords[:, 0].min()), int(coords[:, 1].min())], dtype=int)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    mask = np.rot90(mask, int(_RNG.integers(4)))
    if bool(_RNG.integers(2)):
        mask = np.fliplr(mask)
    if bool(_RNG.integers(2)):
        mask = np.flipud(mask)
    return _normalize_mask(mask)


def _sample_unique_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    keys: set[tuple[tuple[int, int], ...]] = set()

    for _ in range(_SHAPE_ATTEMPTS):
        if len(shapes) == int(count):
            return shapes
        area = int(_RNG.integers(*_SHAPE_AREA_RANGE))
        mask = _random_shape_mask(area)
        if mask is None:
            continue
        key = _shape_key(np.argwhere(mask))
        if key in keys:
            continue
        shapes.append(mask)
        keys.add(key)

    return None


def _make_base_patch(shape_mask: np.ndarray) -> np.ndarray:
    patch = np.full(shape_mask.shape, _BG, dtype=int)
    patch[shape_mask] = GREY
    return patch


def _make_stamp_patch(shape_mask: np.ndarray, color: int) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    hole_h, hole_w = map(int, shape_mask.shape)
    stamp_h = int(max(_STAMP_SIDE_MIN, hole_h + 2 + int(_RNG.integers(_STAMP_EXTRA_MARGIN_MAX + 1))))
    stamp_w = int(max(_STAMP_SIDE_MIN, hole_w + 2 + int(_RNG.integers(_STAMP_EXTRA_MARGIN_MAX + 1))))
    row_offset = int(_RNG.integers(1, stamp_h - hole_h))
    col_offset = int(_RNG.integers(1, stamp_w - hole_w))

    patch = np.full((stamp_h, stamp_w), int(color), dtype=int)
    hole_view = patch[row_offset : row_offset + hole_h, col_offset : col_offset + hole_w]
    hole_view[shape_mask] = _BG
    patch[row_offset : row_offset + hole_h, col_offset : col_offset + hole_w] = hole_view

    hole_coords = np.argwhere(shape_mask) + np.array([row_offset, col_offset], dtype=int)
    return patch, _shape_key(hole_coords)


def _find_placement(blocked: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = blocked.shape
    positions = [
        (int(r0), int(c0))
        for r0 in range(H - h + 1)
        for c0 in range(W - w + 1)
        if not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())
    ]
    if not positions:
        return None
    return positions[int(_RNG.integers(len(positions)))]


def _reserve(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    H, W = blocked.shape
    r0 = max(0, int(top) - _PLACEMENT_PADDING)
    c0 = max(0, int(left) - _PLACEMENT_PADDING)
    r1 = min(H, int(top) + int(h) + _PLACEMENT_PADDING)
    c1 = min(W, int(left) + int(w) + _PLACEMENT_PADDING)
    blocked[r0:r1, c0:c1] = True


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != _BG
    target[mask] = patch[mask]
    grid[top : top + h, left : left + w] = target


def _build_expected(
    grid: np.ndarray,
    base_records: list[dict[str, object]],
    stamp_records: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    stamp_by_key = {record["hole_key"]: int(record["color"]) for record in stamp_records}
    for base in base_records:
        coords = np.asarray(base["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = stamp_by_key[base["shape_key"]]
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.choice(_PAIR_COUNTS))
    shape_masks = _sample_unique_shapes(pair_count)
    if shape_masks is None:
        return None

    height = int(_RNG.integers(_GRID_MIN_BY_PAIRS[pair_count], _GRID_MAX_BY_PAIRS[pair_count] + 1))
    width = int(_RNG.integers(_GRID_MIN_BY_PAIRS[pair_count], _GRID_MAX_BY_PAIRS[pair_count] + 1))
    grid = np.full((height, width), _BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = list(_RNG.choice(_STAMP_COLORS, size=pair_count, replace=False))
    order = list(_RNG.permutation(pair_count))

    stamp_records: list[dict[str, object]] = []
    for idx in order:
        patch, hole_key = _make_stamp_patch(shape_masks[idx], int(colors[idx]))
        placement = _find_placement(blocked, *patch.shape)
        if placement is None:
            return None
        top, left = placement
        _paint_patch(grid, patch, top, left)
        _reserve(blocked, top, left, *patch.shape)
        stamp_records.append(
            {
                "color": int(colors[idx]),
                "hole_key": hole_key,
            }
        )

    base_records: list[dict[str, object]] = []
    for idx in _RNG.permutation(pair_count):
        patch = _make_base_patch(shape_masks[int(idx)])
        placement = _find_placement(blocked, *patch.shape)
        if placement is None:
            return None
        top, left = placement
        _paint_patch(grid, patch, top, left)
        _reserve(blocked, top, left, *patch.shape)
        coords = np.argwhere(shape_masks[int(idx)]) + np.array([top, left], dtype=int)
        base_records.append(
            {
                "shape_key": _shape_key(coords),
                "coords": coords,
            }
        )

    expected = _build_expected(grid, base_records, stamp_records)
    if not np.array_equal(main(grid), expected):
        return None

    bases, stamps = _scene_records(grid)
    if len(bases) != pair_count or len(stamps) != pair_count:
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp/base puzzle")
