from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
USABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"coords": coords, "touches_edge": touches_edge})
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(mask)
            r_src0 = max(0, -dr)
            r_src1 = min(h, h - dr)
            c_src0 = max(0, -dc)
            c_src1 = min(w, w - dc)
            r_dst0 = max(0, dr)
            r_dst1 = min(h, h + dr)
            c_dst0 = max(0, dc)
            c_dst1 = min(w, w + dc)
            shifted[r_dst0:r_dst1, c_dst0:c_dst1] = mask[r_src0:r_src1, c_src0:c_src1]
            out |= shifted
    return out


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _find_palette_band(grid: np.ndarray, bg: int) -> tuple[int, int, int, list[int]]:
    h, w = grid.shape
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        full_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        if len(full_rows) < 2:
            continue
        for i, top in enumerate(full_rows):
            for bottom in full_rows[i + 1 :]:
                if bottom - top < 2:
                    continue
                band = grid[top : bottom + 1, :]
                sep_cols = [c for c in range(w) if np.all(band[:, c] == color)]
                if len(sep_cols) < 3 or sep_cols[0] != 0 or sep_cols[-1] != w - 1:
                    continue
                widths = [sep_cols[j + 1] - sep_cols[j] - 1 for j in range(len(sep_cols) - 1)]
                if min(widths) <= 0 or len(set(widths)) != 1:
                    continue
                candidates.append((int(bottom), int(top), int(bottom), int(color), sep_cols))
    assert candidates, "failed to find the palette strip"
    _, top, bottom, sep, sep_cols = max(candidates)
    return int(top), int(bottom), int(sep), list(map(int, sep_cols))


def _parse_palette(grid: np.ndarray, bg: int) -> tuple[int, int, dict[int, np.ndarray]]:
    top, bottom, sep, sep_cols = _find_palette_band(grid, bg)
    band = grid[top : bottom + 1, :]
    mapping: dict[int, np.ndarray] = {}
    for left, right in zip(sep_cols, sep_cols[1:]):
        if right - left <= 1:
            continue
        cell = band[1:-1, left + 1 : right].copy()
        assert cell.size > 0, "palette cells must have positive area"
        seed = int(cell[0, 0])
        assert seed not in (bg, sep), "seed colors must differ from background and border"
        other = set(map(int, np.unique(cell))) - {bg, seed}
        assert len(other) == 1, "each palette cell must encode exactly one pattern color"
        tile = cell.copy()
        tile[0, 0] = bg
        assert seed not in mapping, "seed colors must be unique"
        mapping[seed] = tile
    assert mapping, "palette must contain at least one mapping"
    return top, sep, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the periodic pattern indexed by its seed color,
    then crop away the bottom palette strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    palette_top, sep, palette = _parse_palette(g, bg)
    upper = g[:palette_top, :].copy()
    region_mask = upper != sep
    comps = [comp for comp in _connected_components(region_mask) if not comp["touches_edge"]]
    assert comps, "expected at least one enclosed region"

    out = upper.copy()
    for comp in comps:
        coords = comp["coords"]
        colors = Counter(int(upper[r, c]) for r, c in coords)
        if bg in colors:
            del colors[bg]
        assert len(colors) == 1, "each enclosed region must contain exactly one seed color"
        seed_color = int(next(iter(colors)))
        assert seed_color in palette, "missing seed color in palette"
        tile = palette[seed_color]
        tile_h, tile_w = map(int, tile.shape)
        r0, c0, _, _ = _bbox(coords)
        for r, c in coords:
            out[r, c] = int(tile[(r - r0) % tile_h, (c - c0) % tile_w])
    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_connected_components(mask)) == 1


def _make_region_shape(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(100):
        h = random.randint(4, max_h)
        w = random.randint(4, max_w)
        mask = np.ones((h, w), dtype=bool)
        corners = ["tl", "tr", "bl", "br"]
        notch_count = random.choices([0, 1, 2], weights=[1, 4, 4], k=1)[0]
        for corner in random.sample(corners, notch_count):
            notch_h = random.randint(1, h - 2)
            notch_w = random.randint(1, w - 2)
            if corner == "tl":
                mask[:notch_h, :notch_w] = False
            elif corner == "tr":
                mask[:notch_h, w - notch_w :] = False
            elif corner == "bl":
                mask[h - notch_h :, :notch_w] = False
            else:
                mask[h - notch_h :, w - notch_w :] = False

        if not mask.any():
            continue
        mask = _tight_crop(mask)
        area = int(mask.sum())
        if area < 8:
            continue
        if not _mask_is_connected(mask):
            continue
        if area == mask.size and random.random() < 0.8:
            continue
        if random.choice([False, True]):
            mask = np.flipud(mask)
        if random.choice([False, True]):
            mask = np.fliplr(mask)
        return np.asarray(mask, dtype=bool)
    return None


def _make_region_layers(shape: np.ndarray) -> dict:
    inner = np.pad(shape, 1, constant_values=False)
    boundary = _dilate8(inner) & ~inner
    full = inner | boundary
    buffer_mask = _dilate8(np.pad(full, 1, constant_values=False))
    return {"inner": inner, "boundary": boundary, "full": full, "buffer": buffer_mask}


def _choose_pattern_mask(h: int, w: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    candidates = []
    phase = random.randint(0, 1)
    candidates.append((rr + cc + phase) % 2 == 0)
    candidates.append((cc + phase) % 2 == 0)
    candidates.append((rr + phase) % 2 == 0)
    candidates.append(rr == (cc + random.randrange(h)) % h)
    candidates.append(rr == (random.randrange(h) - cc) % h)
    candidates.append((rr == h // 2) | (cc == w // 2))
    candidates.append((rr == h - 1) | (cc == w - 1))
    cut_r = random.randint(1, h - 1)
    cut_c = random.randint(1, w - 1)
    candidates.append(((rr == cut_r - 1) & (cc < cut_c)) | ((cc == cut_c - 1) & (rr < cut_r)))
    random.shuffle(candidates)

    for mask in candidates:
        mask = np.asarray(mask, dtype=bool).copy()
        if random.choice([False, True]):
            mask = np.flipud(mask)
        if random.choice([False, True]):
            mask = np.fliplr(mask)
        mask[0, 0] = False
        count = int(mask.sum())
        used_rows = int(np.sum(mask.any(axis=1)))
        used_cols = int(np.sum(mask.any(axis=0)))
        if 2 <= count <= h * w - 2 and used_rows >= 2 and used_cols >= 2:
            return mask

    while True:
        mask = np.zeros((h, w), dtype=bool)
        coords = [(r, c) for r in range(h) for c in range(w) if (r, c) != (0, 0)]
        for r, c in random.sample(coords, random.randint(2, max(2, len(coords) - 1))):
            mask[r, c] = True
        if mask.any(axis=1).sum() >= 2 and mask.any(axis=0).sum() >= 2 and int(mask.sum()) <= h * w - 2:
            return mask


def _make_tile(seed_color: int, pattern_color: int, tile_h: int, tile_w: int) -> np.ndarray:
    tile = np.full((tile_h, tile_w), BG, dtype=int)
    tile[0, 0] = int(seed_color)
    tile[_choose_pattern_mask(tile_h, tile_w)] = int(pattern_color)
    return tile


def _place_regions(upper_h: int, width: int, regions: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((upper_h, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(regions)
    order = sorted(range(len(regions)), key=lambda idx: int(regions[idx]["full"].sum()), reverse=True)
    for idx in order:
        full = regions[idx]["full"]
        buffer_mask = regions[idx]["buffer"]
        fh, fw = map(int, full.shape)
        candidates = []
        for r0 in range(1, upper_h - fh):
            for c0 in range(1, width - fw):
                view = occupied[r0 - 1 : r0 + fh + 1, c0 - 1 : c0 + fw + 1]
                if np.any(view & buffer_mask):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placements[idx] = pos
        r0, c0 = pos
        occupied[r0 - 1 : r0 + fh + 1, c0 - 1 : c0 + fw + 1] |= buffer_mask
    return [(int(r), int(c)) for r, c in placements]


def _render_case(
    upper_h: int,
    width: int,
    regions: list[dict],
    placements: list[tuple[int, int]],
    palette_order: list[int],
    tile_h: int,
    tile_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    upper = np.full((upper_h, width), BG, dtype=int)

    for region, (r0, c0) in zip(regions, placements, strict=True):
        inner = region["inner"]
        boundary = region["boundary"]
        fh, fw = inner.shape
        patch = upper[r0 : r0 + fh, c0 : c0 + fw]
        patch[boundary] = SEP
        patch[inner] = BG
        seed_local = region["seed_local"]
        patch[int(seed_local[0]), int(seed_local[1])] = int(region["seed_color"])
        upper[r0 : r0 + fh, c0 : c0 + fw] = patch

    palette = np.full((tile_h + 2, width), SEP, dtype=int)
    for cell_idx, region_idx in enumerate(palette_order):
        col0 = 1 + cell_idx * (tile_w + 1)
        palette[1 : tile_h + 1, col0 : col0 + tile_w] = regions[region_idx]["tile"]

    expected = upper.copy()
    for region, (r0, c0) in zip(regions, placements, strict=True):
        inner = region["inner"]
        tile = region["tile_for_fill"]
        anchor_r = r0 + int(np.where(inner.any(axis=1))[0][0])
        anchor_c = c0 + int(np.where(inner.any(axis=0))[0][0])
        tile_h_local, tile_w_local = map(int, tile.shape)
        for rr, cc in np.argwhere(inner):
            gr = int(r0 + rr)
            gc = int(c0 + cc)
            expected[gr, gc] = int(tile[(gr - anchor_r) % tile_h_local, (gc - anchor_c) % tile_w_local])

    inp = np.vstack([upper, palette])
    return inp, expected


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_regions = random.randint(2, 4)
    if n_regions == 2:
        tile_w = random.randint(6, 8)
    elif n_regions == 3:
        tile_w = random.randint(4, 6)
    else:
        tile_w = random.randint(4, 5)
    tile_h = random.randint(3, 5)
    width = 1 + n_regions * (tile_w + 1)
    upper_h = random.randint(max(12, tile_h + 8), 20)

    colors = random.sample(USABLE_COLORS, 2 * n_regions)
    seed_colors = colors[:n_regions]
    pattern_colors = colors[n_regions:]

    regions = []
    saw_large_region = False
    for idx in range(n_regions):
        shape = _make_region_shape(min(7, upper_h - 4), min(7, width - 4))
        if shape is None:
            return None
        layers = _make_region_layers(shape)
        seed_coords = np.argwhere(layers["inner"])
        seed_local = tuple(map(int, seed_coords[random.randrange(len(seed_coords))]))
        tile = _make_tile(seed_colors[idx], pattern_colors[idx], tile_h, tile_w)
        regions.append(
            {
                **layers,
                "seed_color": int(seed_colors[idx]),
                "seed_local": seed_local,
                "tile": tile,
                "tile_for_fill": tile.copy(),
            }
        )
        regions[-1]["tile_for_fill"][0, 0] = BG
        if shape.shape[0] > tile_h or shape.shape[1] > tile_w:
            saw_large_region = True

    if not saw_large_region:
        return None

    placements = _place_regions(upper_h, width, regions)
    if placements is None:
        return None

    palette_order = random.sample(list(range(n_regions)), n_regions)
    inp, expected = _render_case(upper_h, width, regions, placements, palette_order, tile_h, tile_w)

    values, counts = np.unique(inp, return_counts=True)
    if int(values[int(np.argmax(counts))]) != BG:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
