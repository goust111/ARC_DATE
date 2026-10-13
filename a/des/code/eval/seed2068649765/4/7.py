from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
AZURE = 8
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 9], dtype=int)
_RNG = np.random.default_rng()


def _intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *separators, int(length)]
    spans = []
    for i in range(len(bounds) - 1):
        start = bounds[i] + 1
        end = bounds[i + 1]
        if end > start:
            spans.append((int(start), int(end)))
    return spans


def _single_component(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    assert pts.size > 0

    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    assert len(seen) == int(mask.sum())
    return sorted(seen)


def _extract_seed_tile(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed_mask = (region != BG) & (region != AZURE) & (region != GREY)
    coords = _single_component(seed_mask)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    tile = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        tile[r - r0, c - c0] = int(region[r, c])
    return tile, tile != BG


def _classify_mask(mask: np.ndarray) -> tuple[str, bool, bool]:
    has_rot4 = mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.rot90(mask, 1))
    has_h = np.array_equal(mask, np.flipud(mask))
    has_v = np.array_equal(mask, np.fliplr(mask))
    if has_rot4:
        return "R", has_h, has_v
    if has_h or has_v:
        return "M", has_h, has_v
    return "N", has_h, has_v


def _growth_tile(tile: np.ndarray, has_h: bool, has_v: bool) -> np.ndarray:
    kind, _, _ = _classify_mask(tile != BG)
    if kind == "R":
        return tile.copy()
    if kind == "N":
        return np.rot90(tile, 2)
    # Mirror along the direction perpendicular to the detected symmetry axis.
    if has_v and not has_h:
        return np.flipud(tile)
    if has_h and not has_v:
        return np.fliplr(tile)
    if has_v:
        return np.flipud(tile)
    return np.fliplr(tile)


def _paint_centered(region: np.ndarray, tile: np.ndarray, center_r: int, center_c: int) -> None:
    h, w = tile.shape
    assert h % 2 == 1 and w % 2 == 1
    top = center_r - h // 2
    left = center_c - w // 2
    for tr, tc in np.argwhere(tile != BG):
        rr = int(top + tr)
        cc = int(left + tc)
        if 0 <= rr < region.shape[0] and 0 <= cc < region.shape[1]:
            region[rr, cc] = int(tile[tr, tc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grid by azure lines, classify each seed by symmetry, and grow transformed copies at grey sites.

    R: seed mask has 90-degree rotational symmetry -> copy as-is.
    M: seed mask lacks 90-degree rotational symmetry but has a horizontal or vertical mirror axis ->
       mirror the colored seed along the direction perpendicular to that axis.
    N: seed mask has neither symmetry -> rotate the colored seed 180 degrees.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_separators = [r for r in range(g.shape[0]) if np.all(g[r, :] == AZURE)]
    col_separators = [c for c in range(g.shape[1]) if np.all(g[:, c] == AZURE)]
    row_regions = _intervals(row_separators, g.shape[0])
    col_regions = _intervals(col_separators, g.shape[1])
    assert row_regions and col_regions

    out = g.copy()
    for r0, r1 in row_regions:
        for c0, c1 in col_regions:
            region = g[r0:r1, c0:c1]
            tile, mask = _extract_seed_tile(region)
            kind, has_h, has_v = _classify_mask(mask)
            growth = _growth_tile(tile, has_h, has_v)
            sites = np.argwhere(region == GREY)
            assert len(sites) > 0
            out_region = out[r0:r1, c0:c1]
            for sr, sc in sites:
                _paint_centered(out_region, growth, int(sr), int(sc))
            assert _classify_mask(mask)[0] == kind
    return out


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


R_TEMPLATES = [
    _mask_from_rows(("010", "111", "010")),
    _mask_from_rows(("101", "010", "101")),
    _mask_from_rows(("00100", "01110", "11111", "01110", "00100")),
]

M_TEMPLATES = [
    _mask_from_rows(("111", "010", "010")),
    _mask_from_rows(("11111", "00100", "01110", "00100", "00100")),
    _mask_from_rows(("01110", "00100", "01110", "00100", "00100")),
]

N_TEMPLATES = [
    _mask_from_rows(("110", "010", "011")),
    _mask_from_rows(("111", "010", "011")),
    _mask_from_rows(("01100", "00100", "11110", "00110", "00010")),
]

LAYOUTS = [(1, 3), (3, 1), (2, 2), (2, 3), (3, 2)]
REGION_SIZE_RANGE = (9, 15)
REGION_RETRIES = 200
GRID_RETRIES = 300


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if int(_RNG.integers(2)):
        out = np.fliplr(out)
    return np.rot90(out, int(_RNG.integers(4)))


def _choose_mask(kind: str) -> np.ndarray:
    bank = {"R": R_TEMPLATES, "M": M_TEMPLATES, "N": N_TEMPLATES}[kind]
    for _ in range(40):
        mask = _random_dihedral(bank[int(_RNG.integers(len(bank)))])
        if _classify_mask(mask)[0] == kind and bool(mask[mask.shape[0] // 2, mask.shape[1] // 2]):
            return mask
    raise RuntimeError(f"failed to sample a {kind} template")


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    tile = np.zeros(mask.shape, dtype=int)
    color_count = int(_RNG.integers(1, min(4, len(coords) + 1)))
    colors = _RNG.choice(PALETTE, size=color_count, replace=False)
    mode = int(_RNG.integers(4))
    cr = mask.shape[0] // 2
    cc = mask.shape[1] // 2

    def score(rc: tuple[int, int]) -> tuple[int, int, int]:
        r, c = rc
        if mode == 0:
            return (r, c, 0)
        if mode == 1:
            return (c, r, 0)
        if mode == 2:
            return (r + c, r, c)
        return (abs(r - cr) + abs(c - cc), r, c)

    coords.sort(key=score)
    for idx, (r, c) in enumerate(coords):
        color = int(colors[min(color_count - 1, idx * color_count // len(coords))])
        tile[r, c] = color
    return tile


def _footprint(mask: np.ndarray, center_r: int, center_c: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    top = center_r - h // 2
    left = center_c - w // 2
    cells = set()
    for tr, tc in np.argwhere(mask):
        rr = int(top + tr)
        cc = int(left + tc)
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            cells.add((rr, cc))
    return cells


def _place_seed(region: np.ndarray, tile: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells = set()
    for tr, tc in np.argwhere(tile != BG):
        rr = int(top + tr)
        cc = int(left + tc)
        region[rr, cc] = int(tile[tr, tc])
        cells.add((rr, cc))
    return cells


def _build_region(height: int, width: int, kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(REGION_RETRIES):
        mask = _choose_mask(kind)
        tile = _colorize_mask(mask)
        th, tw = tile.shape
        if th > height or tw > width:
            continue

        top = int(_RNG.integers(0, height - th + 1))
        left = int(_RNG.integers(0, width - tw + 1))
        inp = np.full((height, width), BG, dtype=int)
        out = np.full((height, width), BG, dtype=int)
        seed_cells = _place_seed(inp, tile, top, left)
        _place_seed(out, tile, top, left)

        _, has_h, has_v = _classify_mask(mask)
        grown = _growth_tile(tile, has_h, has_v)
        grown_mask = grown != BG
        full_size = int(grown_mask.sum())

        candidates = []
        edge_candidates = []
        for r in range(height):
            for c in range(width):
                if (r, c) in seed_cells:
                    continue
                fp = _footprint(grown_mask, r, c, inp.shape)
                if not fp or (r, c) not in fp or (fp & seed_cells):
                    continue
                item = ((r, c), fp)
                candidates.append(item)
                if len(fp) < full_size:
                    edge_candidates.append(item)

        if not candidates:
            continue

        target_sites = int(_RNG.integers(1, 4))
        chosen_sites: list[tuple[int, int]] = []
        blocked = set(seed_cells)

        if edge_candidates and bool(int(_RNG.integers(2))):
            edge_order = list(_RNG.permutation(len(edge_candidates)))
            for idx in edge_order:
                site, fp = edge_candidates[int(idx)]
                if not (blocked & fp):
                    chosen_sites.append(site)
                    blocked |= fp
                    break

        order = list(_RNG.permutation(len(candidates)))
        for idx in order:
            if len(chosen_sites) >= target_sites:
                break
            site, fp = candidates[int(idx)]
            if site in chosen_sites or (blocked & fp):
                continue
            chosen_sites.append(site)
            blocked |= fp

        if not chosen_sites:
            continue

        for r, c in chosen_sites:
            inp[r, c] = GREY
            _paint_centered(out, grown, r, c)

        try:
            if np.array_equal(main(_embed_single_region(inp)), _embed_single_region(out)):
                return inp, out
        except Exception:
            continue
    return None


def _embed_single_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    grid = np.full((h + 2, w + 2), AZURE, dtype=int)
    grid[1:-1, 1:-1] = region
    return grid


def _separator_positions(sizes: list[int]) -> tuple[list[int], list[int]]:
    seps = [0]
    starts = []
    cur = 1
    for size in sizes:
        starts.append(cur)
        cur += int(size)
        seps.append(cur)
        cur += 1
    return seps, starts


def _build_grid() -> tuple[np.ndarray, np.ndarray] | None:
    nr, nc = LAYOUTS[int(_RNG.integers(len(LAYOUTS)))]
    region_count = nr * nc
    classes = ["R", "M", "N"] + [str(x) for x in _RNG.choice(np.array(["R", "M", "N"]), size=region_count - 3, replace=True)]
    classes = [str(x) for x in _RNG.permutation(classes)]

    row_sizes = [int(_RNG.integers(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1] + 1)) for _ in range(nr)]
    col_sizes = [int(_RNG.integers(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1] + 1)) for _ in range(nc)]
    row_seps, row_starts = _separator_positions(row_sizes)
    col_seps, col_starts = _separator_positions(col_sizes)
    H = sum(row_sizes) + nr + 1
    W = sum(col_sizes) + nc + 1

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    inp[row_seps, :] = AZURE
    inp[:, col_seps] = AZURE
    out[row_seps, :] = AZURE
    out[:, col_seps] = AZURE

    idx = 0
    for r_idx, (r0, rh) in enumerate(zip(row_starts, row_sizes)):
        for c_idx, (c0, cw) in enumerate(zip(col_starts, col_sizes)):
            built = _build_region(rh, cw, classes[idx])
            if built is None:
                return None
            region_in, region_out = built
            inp[r0 : r0 + rh, c0 : c0 + cw] = region_in
            out[r0 : r0 + rh, c0 : c0 + cw] = region_out
            idx += 1

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GRID_RETRIES):
        built = _build_grid()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid grid")
