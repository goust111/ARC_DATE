from __future__ import annotations

import numpy as np


BG = 0
SEP = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BRANCHES = ("horizontal", "vertical", "rotational", "none")
REGION_SIZE_CHOICES = (4, 6, 8)
GEN_ATTEMPTS = 300
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool]:
    return (
        bool(np.array_equal(mask, np.flipud(mask))),
        bool(np.array_equal(mask, np.fliplr(mask))),
        bool(np.array_equal(mask, np.rot90(mask, 2))),
    )


def _paint_region(branch: str, color: int, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    if branch == "horizontal":
        out[: h // 2, :] = color
    elif branch == "vertical":
        out[:, : w // 2] = color
    elif branch == "rotational":
        rr, cc = np.indices((h, w))
        out[(rr - cc) % 2 == 0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redraw each enclosed region according to the symmetry encoded by its markers.

    Input structure:
    - Black background.
    - Grey single-pixel separator walls divide the canvas into closed rectangular regions.
    - Each region contains a sparse single-color marker pattern.

    Rule:
    - Keep the grey walls unchanged.
    - If the marker pattern is horizontally symmetric, fill the top half of the region.
    - If it is vertically symmetric, fill the left half of the region.
    - If it is only 180-degree rotationally symmetric, draw NW-SE diagonal stripes.
    - Otherwise clear the region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for comp in _components(g != SEP):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        h, w = r1 - r0 + 1, c1 - c0 + 1

        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert comp.shape[0] == h * w

        colors = [int(v) for v in np.unique(region) if int(v) not in (BG, SEP)]
        if not colors:
            out[r0 : r1 + 1, c0 : c1 + 1] = BG
            continue
        assert len(colors) == 1
        color = colors[0]
        mask = region == color

        h_sym, v_sym, r_sym = _symmetry_flags(mask)
        if h_sym:
            branch = "horizontal"
        elif v_sym:
            branch = "vertical"
        elif r_sym:
            branch = "rotational"
        else:
            branch = "none"

        out[r0 : r1 + 1, c0 : c1 + 1] = _paint_region(branch, color, h, w)
    return out


def _random_mask(h: int, w: int, count: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    idx = RNG.choice(h * w, size=count, replace=False)
    mask.flat[idx] = True
    return mask


def _make_horizontal_mask(h: int, w: int) -> np.ndarray:
    top_h = h // 2
    max_seed = min(4, top_h * w)
    for _ in range(200):
        seed = np.zeros((top_h, w), dtype=bool)
        k = int(RNG.integers(1, max_seed + 1))
        idx = RNG.choice(top_h * w, size=k, replace=False)
        seed.flat[idx] = True
        mask = np.vstack([seed, np.flipud(seed)])
        h_sym, v_sym, _ = _symmetry_flags(mask)
        if h_sym and not v_sym:
            return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    mask[h - 1, 0] = True
    return mask


def _make_vertical_mask(h: int, w: int) -> np.ndarray:
    left_w = w // 2
    max_seed = min(4, h * left_w)
    for _ in range(200):
        seed = np.zeros((h, left_w), dtype=bool)
        k = int(RNG.integers(1, max_seed + 1))
        idx = RNG.choice(h * left_w, size=k, replace=False)
        seed.flat[idx] = True
        mask = np.hstack([seed, np.fliplr(seed)])
        h_sym, v_sym, _ = _symmetry_flags(mask)
        if v_sym and not h_sym:
            return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    mask[0, w - 1] = True
    return mask


def _make_rotational_mask(h: int, w: int) -> np.ndarray:
    canon = [(r, c) for r in range(h) for c in range(w) if (r < h - 1 - r) or (r == h - 1 - r and c < w - 1 - c)]
    max_orbits = min(4, len(canon))
    for _ in range(200):
        k = int(RNG.integers(1, max_orbits + 1))
        picks = RNG.choice(len(canon), size=k, replace=False)
        mask = np.zeros((h, w), dtype=bool)
        for idx in np.atleast_1d(picks):
            r, c = canon[int(idx)]
            mask[r, c] = True
            mask[h - 1 - r, w - 1 - c] = True
        h_sym, v_sym, r_sym = _symmetry_flags(mask)
        if r_sym and not h_sym and not v_sym:
            return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    mask[h - 1, w - 1] = True
    return mask


def _make_none_mask(h: int, w: int) -> np.ndarray:
    max_cells = min(6, h * w - 1)
    for _ in range(200):
        k = int(RNG.integers(3, max_cells + 1))
        mask = _random_mask(h, w, k)
        if not any(_symmetry_flags(mask)):
            return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[1, 0] = True
    return mask


def _make_mask(branch: str, h: int, w: int) -> np.ndarray:
    if branch == "horizontal":
        return _make_horizontal_mask(h, w)
    if branch == "vertical":
        return _make_vertical_mask(h, w)
    if branch == "rotational":
        return _make_rotational_mask(h, w)
    return _make_none_mask(h, w)


def _balanced_branches(count: int) -> list[str]:
    branches = list(BRANCHES) * (count // len(BRANCHES))
    extra = RNG.permutation(np.array(BRANCHES, dtype=object))[: count % len(BRANCHES)]
    branches.extend(str(v) for v in extra.tolist())
    RNG.shuffle(branches)
    return branches


def _region_starts(sizes: list[int]) -> list[int]:
    starts: list[int] = []
    pos = 1
    for size in sizes:
        starts.append(pos)
        pos += size + 1
    return starts


def _gen_attempt() -> np.ndarray | None:
    n_rows = int(RNG.integers(2, 4))
    n_cols = int(RNG.integers(2, 4))
    row_sizes = [int(RNG.choice(REGION_SIZE_CHOICES)) for _ in range(n_rows)]
    col_sizes = [int(RNG.choice(REGION_SIZE_CHOICES)) for _ in range(n_cols)]

    h = sum(row_sizes) + n_rows + 1
    w = sum(col_sizes) + n_cols + 1
    inp = np.full((h, w), SEP, dtype=int)
    expected = np.full((h, w), SEP, dtype=int)

    branches = _balanced_branches(n_rows * n_cols)
    row_starts = _region_starts(row_sizes)
    col_starts = _region_starts(col_sizes)

    idx = 0
    for r0, rh in zip(row_starts, row_sizes):
        for c0, cw in zip(col_starts, col_sizes):
            branch = branches[idx]
            idx += 1
            color = int(RNG.choice(MARKER_COLORS))
            mask = _make_mask(branch, rh, cw)

            region = np.zeros((rh, cw), dtype=int)
            region[mask] = color
            inp[r0 : r0 + rh, c0 : c0 + cw] = region
            expected[r0 : r0 + rh, c0 : c0 + cw] = _paint_region(branch, color, rh, cw)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
