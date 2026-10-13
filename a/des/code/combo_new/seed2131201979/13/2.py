from __future__ import annotations

import random

import numpy as np


BG = 0
AZURE = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MIN_GROUPS = 2
MAX_GROUPS = 4
MIN_AREA = 4
MAX_AREA = 6
MAX_DIM = 4
MAX_SHAPE_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            comps.append(np.array(coords, dtype=int))

    return comps


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    assert coords.size > 0
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    mask = np.zeros((int(r1) - int(r0) + 1, int(c1) - int(c0) + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - int(r0), int(c) - int(c0)] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.ascontiguousarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    trimmed = np.ascontiguousarray(mask, dtype=bool)
    return [np.ascontiguousarray(np.rot90(trimmed, k), dtype=bool) for k in range(4)]


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    return min(_rotations(mask), key=_mask_key)


def _rotation_index(mask: np.ndarray, canonical: np.ndarray) -> int:
    matches = [k for k, rot in enumerate(_rotations(canonical)) if np.array_equal(rot, mask)]
    assert len(matches) == 1, "shape must match exactly one orthogonal rotation"
    return int(matches[0])


def _infer_missing_rotation(observed_masks: list[np.ndarray]) -> np.ndarray:
    assert len(observed_masks) == 3
    canonical = _canonical_mask(observed_masks[0])
    for mask in observed_masks[1:]:
        assert _mask_key(_canonical_mask(mask)) == _mask_key(canonical), "same-color objects must share one base shape"

    seen = {_rotation_index(mask, canonical) for mask in observed_masks}
    assert len(seen) == 3, "expected exactly three distinct rotations"
    missing = ({0, 1, 2, 3} - seen).pop()
    return np.ascontiguousarray(np.rot90(canonical, int(missing)), dtype=bool)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    assert 0 <= int(r0) and 0 <= int(c0)
    assert int(r0) + h <= grid.shape[0] and int(c0) + w <= grid.shape[1]

    for dr, dc in np.argwhere(mask):
        rr = int(r0) + int(dr)
        cc = int(c0) + int(dc)
        current = int(grid[rr, cc])
        assert current in {BG, int(color)}, "generated shape placement must not overwrite unrelated content"
        grid[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color group by adding the missing orthogonal rotation of its monochrome fragment at the marked anchor.

    Input structure:
    - The background is black, with several small monochrome 4-connected objects of size 4-6 scattered in the upper area.
    - For each non-azure foreground color, there are exactly three such objects, and they are three different 90-degree rotations of one base shape.
    - A single full-width azure reference line sits near the bottom, and each color also has one isolated single-pixel marker above that line.

    Rule:
    - Group objects by color, identify the shared shape up to rotation, and determine which of the four orthogonal rotations is missing.
    - Remove the single-pixel marker, then place the missing rotation with the marker position as the missing shape's bounding-box top-left anchor.
    - Keep the reference line and every pre-existing source object unchanged.

    Key insight:
    - Color identifies which fragments belong together, but solving requires rotation-invariant shape matching: the missing answer is the one rotation not already present among the three same-color pieces.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    azure_rows = [r for r in range(h) if bool(np.all(g[r, :] == AZURE))]
    assert len(azure_rows) == 1, "expected one full-width azure reference line"
    line_r = int(azure_rows[0])
    assert line_r >= 1
    azure_coords = np.argwhere(g == AZURE)
    assert azure_coords.size > 0 and np.all(azure_coords[:, 0] == line_r), "azure should appear only on the reference line"

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in {BG, AZURE}]
    assert colors

    for color in colors:
        comps = _connected_components(g == color)
        assert comps, "each foreground color should appear at least once"

        marker_coords: list[tuple[int, int]] = []
        object_masks: list[np.ndarray] = []

        for comp in comps:
            area = int(len(comp))
            rows = comp[:, 0]
            assert int(rows.max()) < line_r, "all non-reference objects must lie above the line"
            if area == 1:
                marker_coords.append((int(comp[0, 0]), int(comp[0, 1])))
            else:
                assert MIN_AREA <= area <= MAX_AREA, "source objects should have size 4-6"
                object_masks.append(_coords_to_mask(comp))

        assert len(marker_coords) == 1, "each color should have exactly one anchor marker"
        assert len(object_masks) == 3, "each color should contribute exactly three source objects"

        marker_r, marker_c = marker_coords[0]
        missing_mask = _infer_missing_rotation(object_masks)
        assert marker_r + missing_mask.shape[0] <= line_r, "completed shape must fit above the line"

        out[marker_r, marker_c] = BG
        _paint_mask(out, missing_mask, marker_r, marker_c, color)

    return out


def _random_connected_shape() -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        target = random.randint(MIN_AREA, MAX_AREA)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (int(r + dr), int(c + dc))
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(random.choice(tuple(frontier)))

        coords = np.array(sorted(cells), dtype=int)
        coords[:, 0] -= int(coords[:, 0].min())
        coords[:, 1] -= int(coords[:, 1].min())
        mask = _coords_to_mask(coords)
        if max(mask.shape) > MAX_DIM:
            continue

        rotations = _rotations(mask)
        if len({_mask_key(rot) for rot in rotations}) != 4:
            continue
        return _canonical_mask(mask)

    raise RuntimeError("failed to sample a valid asymmetric 4-connected shape")


def _can_place_with_gap(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, gap: int = 1) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if int(r0) < 0 or int(c0) < 0 or int(r0) + mh > h or int(c0) + mw > w:
        return False

    for dr, dc in np.argwhere(mask):
        rr = int(r0) + int(dr)
        cc = int(c0) + int(dc)
        r1 = max(0, rr - gap)
        r2 = min(h, rr + gap + 1)
        c1 = max(0, cc - gap)
        c2 = min(w, cc + gap + 1)
        if bool(np.any(occupied[r1:r2, c1:c2])):
            return False
    return True


def _mark_occupied(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    for dr, dc in np.argwhere(mask):
        occupied[int(r0) + int(dr), int(c0) + int(dc)] = True


def _place_source_objects(inp: np.ndarray, items: list[dict], max_bottom_row: int) -> bool:
    occupied = inp != BG
    order = list(range(len(items)))
    random.shuffle(order)
    order.sort(
        key=lambda i: (items[i]["mask"].shape[0] * items[i]["mask"].shape[1], int(items[i]["mask"].sum())),
        reverse=True,
    )

    for idx in order:
        color = int(items[idx]["color"])
        mask = items[idx]["mask"]
        mh, mw = mask.shape
        max_r0 = int(max_bottom_row) - mh + 1
        if max_r0 < 0:
            return False

        positions = [(r, c) for r in range(max_r0 + 1) for c in range(inp.shape[1] - mw + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place_with_gap(occupied, mask, int(r0), int(c0), gap=1):
                continue
            _paint_mask(inp, mask, int(r0), int(c0), color)
            _mark_occupied(occupied, mask, int(r0), int(c0))
            placed = True
            break

        if not placed:
            return False

    return True


def _marker_layout(widths: list[int], width: int) -> list[int] | None:
    n = len(widths)
    order = list(range(n))
    random.shuffle(order)
    gaps = [random.randint(1, 3) for _ in range(max(0, n - 1))]
    span = sum(widths[i] for i in order) + sum(gaps)
    if span > width:
        return None

    left_margin = random.randint(0, width - span)
    cols = [0] * n
    cursor = left_margin
    for i, idx in enumerate(order):
        cols[idx] = int(cursor)
        cursor += widths[idx]
        if i < n - 1:
            cursor += gaps[i]
    return cols


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.randint(MIN_GROUPS, MAX_GROUPS)
    colors = random.sample(PALETTE, n_groups)
    groups: list[dict] = []

    for color in colors:
        base = _random_connected_shape()
        missing_rot = random.randrange(4)
        observed_rots = [rot for rot in range(4) if rot != missing_rot]
        random.shuffle(observed_rots)
        groups.append(
            {
                "color": int(color),
                "missing_mask": np.ascontiguousarray(np.rot90(base, missing_rot), dtype=bool),
                "observed_masks": [np.ascontiguousarray(np.rot90(base, rot), dtype=bool) for rot in observed_rots],
            }
        )

    max_missing_h = max(int(group["missing_mask"].shape[0]) for group in groups)
    widths = [int(group["missing_mask"].shape[1]) for group in groups]
    marker_span_min = sum(widths) + max(0, n_groups - 1)

    min_h = max(18, max_missing_h + 5 + 3 * n_groups)
    min_w = max(18, marker_span_min + 2, 7 + 4 * n_groups)
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    marker_cols = _marker_layout(widths, width)
    if marker_cols is None:
        return None

    line_r = height - 1
    marker_r = line_r - max_missing_h
    source_bottom = marker_r - 2
    if source_bottom < 4:
        return None

    inp = np.full((height, width), BG, dtype=int)
    inp[line_r, :] = AZURE

    for group, marker_c in zip(groups, marker_cols):
        group["anchor"] = (int(marker_r), int(marker_c))
        inp[marker_r, marker_c] = int(group["color"])

    items: list[dict] = []
    for group in groups:
        for mask in group["observed_masks"]:
            items.append({"color": int(group["color"]), "mask": mask})

    if not _place_source_objects(inp, items, source_bottom):
        return None

    expected = inp.copy()
    for group in groups:
        anchor_r, anchor_c = group["anchor"]
        expected[anchor_r, anchor_c] = BG
        _paint_mask(expected, group["missing_mask"], anchor_r, anchor_c, int(group["color"]))

    try:
        actual = main(inp)
    except AssertionError:
        return None

    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-completion input")
