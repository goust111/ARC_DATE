import math
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8

TRANSFORMS = ("hflip", "vflip", "rot_cw", "rot_ccw")
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 9)
ICON_MASKS = {
    "hflip": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    "vflip": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    "rot_cw": np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
    "rot_ccw": np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
}

GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_is_perimeter(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    rows, cols = np.nonzero(mask)
    return bool(np.all((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    box = _bbox(coords)
    return _coords_to_mask(coords, box)


def _apply_transform(patch: np.ndarray, transform: str) -> np.ndarray:
    if transform == "hflip":
        return np.fliplr(patch)
    if transform == "vflip":
        return np.flipud(patch)
    if transform == "rot_cw":
        return np.rot90(patch, -1)
    if transform == "rot_ccw":
        return np.rot90(patch, 1)
    raise ValueError(f"unknown transform: {transform}")


def _decode_icon(mask: np.ndarray) -> str:
    for name, icon in ICON_MASKS.items():
        if mask.shape == icon.shape and np.array_equal(mask, icon):
            return name
    raise ValueError("unrecognized azure icon")


def _legend_layout(grid: np.ndarray) -> tuple[int, list[int]]:
    grey_rows = [r for r in range(grid.shape[0]) if np.any(grid[r] == GREY)]
    assert grey_rows, "expected grey legend separators"
    legend_w = max(int(np.max(np.where(grid[r] == GREY)[0])) for r in grey_rows) + 1
    sep_rows: list[int] = []
    expected_cols = np.arange(legend_w)
    for r in grey_rows:
        cols = np.where(grid[r] == GREY)[0]
        if np.array_equal(cols, expected_cols) and np.all(grid[r, legend_w:] == BLACK):
            sep_rows.append(r)
    assert len(sep_rows) >= 4, "expected 3 to 5 legend entries"
    return legend_w, sep_rows


def _parse_legends(grid: np.ndarray) -> tuple[int, list[dict]]:
    legend_w, sep_rows = _legend_layout(grid)
    entries: list[dict] = []
    for top, bottom in zip(sep_rows, sep_rows[1:]):
        if bottom <= top + 1:
            continue
        r0 = top + 1
        r1 = bottom - 1
        entry = grid[r0 : r1 + 1, :legend_w]
        border_colors = [int(color) for color in np.unique(entry) if int(color) not in {BLACK, GREY, AZURE}]
        assert len(border_colors) == 1, "each legend entry must have one fingerprint color"
        border_color = border_colors[0]

        border_coords = np.argwhere(entry == border_color)
        border_box = _bbox(border_coords)
        fingerprint = _coords_to_mask(border_coords, border_box)
        assert _mask_is_perimeter(fingerprint), "legend fingerprint must lie on a rectangle perimeter"

        icon_coords = np.argwhere(entry == AZURE)
        assert icon_coords.size > 0, "legend entry needs an azure icon"
        icon_box = _bbox(icon_coords)
        icon_mask = _coords_to_mask(icon_coords, icon_box)
        transform = _decode_icon(icon_mask)
        entries.append({"color": border_color, "mask": fingerprint, "transform": transform})

    assert 3 <= len(entries) <= 5
    return legend_w, entries


def _parse_regions(grid: np.ndarray, legend_w: int, legends: list[dict]) -> list[dict]:
    by_color: dict[int, list[dict]] = {}
    for entry in legends:
        by_color.setdefault(int(entry["color"]), []).append(entry)

    regions: list[dict] = []
    for color, entries in by_color.items():
        for coords in _components(grid == color):
            box = _bbox(coords)
            if box[3] < legend_w:
                continue
            local = _coords_to_mask(coords, box)
            if not _mask_is_perimeter(local):
                continue
            matched = [
                entry
                for entry in entries
                if entry["mask"].shape == local.shape and np.array_equal(entry["mask"], local)
            ]
            if not matched:
                continue
            assert len(matched) == 1, "color and fingerprint should determine a unique legend entry"
            r0, c0, r1, c1 = box
            assert r1 - r0 >= 2 and c1 - c0 >= 2
            regions.append({"bbox": box, "transform": matched[0]["transform"]})

    assert regions, "expected at least one transformable region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed region's border fingerprint to the legend, then transform the interior accordingly.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_w, legends = _parse_legends(g)
    regions = _parse_regions(g, legend_w, legends)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        transformed = _apply_transform(inner, str(region["transform"]))
        assert transformed.shape == inner.shape, "generator always uses square interiors for rotation"
        out[r0 + 1 : r1, c0 + 1 : c1] = transformed
    return out


def _fingerprint_mask(size: int, side: int, start: int, length: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if side == 0:
        mask[0, start : start + length] = False
    elif side == 1:
        mask[start : start + length, -1] = False
    elif side == 2:
        mask[-1, start : start + length] = False
    else:
        mask[start : start + length, 0] = False
    assert _mask_is_perimeter(mask)
    assert len(_components(mask)) == 1
    return mask


def _draw_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _make_inner_patch(
    rng: np.random.Generator,
    size: int,
    palette: tuple[int, ...],
    transform: str,
) -> np.ndarray:
    cell_count = size * size
    min_fill = max(3, size)
    max_fill = max(min_fill, cell_count - 1)
    for _ in range(200):
        patch = np.zeros((size, size), dtype=int)
        fill = int(rng.integers(min_fill, max_fill + 1))
        positions = rng.choice(cell_count, size=fill, replace=False)
        color_count = 1 if len(palette) == 1 else int(rng.integers(1, min(3, len(palette)) + 1))
        colors = tuple(int(x) for x in rng.choice(np.array(palette), size=color_count, replace=False))
        for pos in positions:
            r = int(pos // size)
            c = int(pos % size)
            patch[r, c] = int(colors[rng.integers(0, len(colors))])
        if np.array_equal(_apply_transform(patch, transform), patch):
            continue
        if np.count_nonzero(patch) < min_fill:
            continue
        return patch
    raise RuntimeError("failed to build a non-degenerate interior patch")


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0 = spec["top"], spec["left"]
        size = spec["size"]
        inner = grid[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
        out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = _apply_transform(
            inner, str(spec["transform"])
        )
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    legend_count = int(rng.choice([4, 5]))
    box_size = int(rng.choice([5, 6]))
    entry_h = box_size
    legend_w = box_size + 5
    legend_h = legend_count * entry_h + legend_count + 1

    border_colors = tuple(int(x) for x in rng.choice(np.array(NON_SPECIAL_COLORS), size=legend_count, replace=False))
    transforms = list(TRANSFORMS)
    if legend_count == 5:
        transforms.append(str(rng.choice(np.array(TRANSFORMS))))
    rng.shuffle(transforms)

    pattern_keys: set[tuple[int, int, int]] = set()
    fingerprints: list[np.ndarray] = []
    for _ in range(legend_count):
        for _try in range(100):
            side = int(rng.integers(0, 4))
            start = int(rng.integers(1, box_size - 1))
            max_len = box_size - start
            length = int(rng.integers(1, max_len + 1))
            key = (side, start, length)
            if key in pattern_keys:
                continue
            pattern_keys.add(key)
            fingerprints.append(_fingerprint_mask(box_size, side, start, length))
            break
        else:
            return None

    inner_palette = tuple(color for color in NON_SPECIAL_COLORS if color not in border_colors)
    if not inner_palette:
        return None

    extra_regions = int(rng.integers(0, 3))
    region_entries = list(range(legend_count))
    if extra_regions:
        region_entries.extend(int(x) for x in rng.choice(np.arange(legend_count), size=extra_regions, replace=True))
    rng.shuffle(region_entries)
    region_count = len(region_entries)

    cols = int(rng.integers(2, min(4, region_count) + 1))
    rows = int(math.ceil(region_count / cols))
    h_gap = int(rng.integers(1, 4))
    v_gap = int(rng.integers(1, 4))
    region_w = cols * box_size + (cols - 1) * h_gap
    region_h = rows * box_size + (rows - 1) * v_gap

    base_c = legend_w + int(rng.integers(2, 5))
    grid_w = base_c + region_w + int(rng.integers(1, 5))
    grid_h = max(legend_h, region_h + 2 + int(rng.integers(0, 5)))
    max_r_shift = max(1, grid_h - region_h - 1)
    base_r = int(rng.integers(1, max_r_shift + 1))

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    sep_rows = [0]
    for idx in range(legend_count):
        sep_rows.append((idx + 1) * entry_h + idx + 1)
    for row in sep_rows:
        grid[row, :legend_w] = GREY

    legend_entries: list[dict] = []
    for idx in range(legend_count):
        r0 = sep_rows[idx] + 1
        color = int(border_colors[idx])
        transform = str(transforms[idx])
        fingerprint = fingerprints[idx]
        _draw_mask(grid, r0, 0, fingerprint, color)

        icon = ICON_MASKS[transform]
        icon_top = r0 + (entry_h - icon.shape[0]) // 2
        icon_left = box_size + 2
        _draw_mask(grid, icon_top, icon_left, icon, AZURE)
        legend_entries.append({"color": color, "mask": fingerprint, "transform": transform})

    specs: list[dict] = []
    slot_positions = []
    for row in range(rows):
        for col in range(cols):
            slot_positions.append((base_r + row * (box_size + v_gap), base_c + col * (box_size + h_gap)))
    rng.shuffle(slot_positions)

    used_positions = slot_positions[:region_count]
    for (top, left), entry_idx in zip(used_positions, region_entries):
        color = int(border_colors[entry_idx])
        transform = str(transforms[entry_idx])
        fingerprint = fingerprints[entry_idx]
        inner = _make_inner_patch(rng, box_size - 2, inner_palette, transform)
        _draw_mask(grid, top, left, fingerprint, color)
        grid[top + 1 : top + box_size - 1, left + 1 : left + box_size - 1] = inner
        specs.append(
            {
                "top": top,
                "left": left,
                "size": box_size,
                "color": color,
                "transform": transform,
            }
        )

    if {spec["transform"] for spec in specs} != set(TRANSFORMS):
        return None

    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
