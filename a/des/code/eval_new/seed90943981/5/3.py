import numpy as np


BLACK = 0
RED = 2
PALETTE = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)
RNG = np.random.default_rng()

GEN_ATTEMPTS = 300
PATCH_ATTEMPTS = 200
SEED_ATTEMPTS = 200

_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    else:
        dirs = _DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + H, dc : dc + W]
    return out


def _symmetry_kind(seed: np.ndarray) -> str:
    has_h = np.array_equal(seed, seed[::-1, :])
    has_v = np.array_equal(seed, seed[:, ::-1])
    if has_h and has_v:
        return "both"
    if has_h:
        return "h"
    if has_v:
        return "v"
    return "diag"


def _stamp(out: np.ndarray, region_mask: np.ndarray, seed: np.ndarray, top: int, left: int) -> None:
    H, W = out.shape
    h, w = seed.shape
    for r in range(h):
        rr = top + r
        if rr >= H:
            break
        for c in range(w):
            cc = left + c
            if cc >= W:
                break
            if region_mask[rr, cc]:
                out[rr, cc] = int(seed[r, c])


def _fill_region(out: np.ndarray, region_mask: np.ndarray, seed: np.ndarray, origin: tuple[int, int], kind: str) -> None:
    r0, c0 = origin
    h, w = seed.shape
    coords = np.argwhere(region_mask)
    max_r, max_c = map(int, coords.max(axis=0))

    if kind == "h":
        c = c0
        while c <= max_c:
            _stamp(out, region_mask, seed, r0, c)
            c += w
        return

    if kind == "v":
        r = r0
        while r <= max_r:
            _stamp(out, region_mask, seed, r, c0)
            r += h
        return

    if kind == "both":
        r = r0
        while r <= max_r:
            c = c0
            while c <= max_c:
                _stamp(out, region_mask, seed, r, c)
                c += w
            r += h
        return

    r, c = r0, c0
    while r <= max_r and c <= max_c:
        _stamp(out, region_mask, seed, r, c)
        r += h
        c += w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed into a region-wide periodic fill chosen by the seed's symmetry type.

    Input structure:
    - Background is black.
    - One or more red one-pixel boundaries enclose separate interior regions.
    - Near the top-left of each enclosed region sits a small multicolor rectangular seed made of non-black, non-red colors.

    Rule:
    - Detect each enclosed non-red region and extract the seed rectangle inside it.
    - Classify the seed by reflection symmetry: horizontal only, vertical only, both axes, or none.
    - Horizontal symmetry -> repeat the seed only to the right; vertical symmetry -> repeat only downward.
    - Both symmetries -> tile in both directions; no symmetry -> place repeated copies along the down-right diagonal, always aligned to the original seed position.
    - Keep the red boundary unchanged and paint only cells inside the enclosed region.

    Key insight:
    The seed does not merely provide colors; its symmetry class selects which translation directions are allowed for the fill.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    for comp in _components(g != RED, connectivity=8):
        if any(r == 0 or c == 0 or r == H - 1 or c == W - 1 for r, c in comp):
            continue
        seed_cells = [(r, c) for r, c in comp if g[r, c] not in (BLACK, RED)]
        if not seed_cells:
            continue

        rows = [r for r, _ in seed_cells]
        cols = [c for _, c in seed_cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        seed = g[r0 : r1 + 1, c0 : c1 + 1].copy()

        region_mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            region_mask[r, c] = True

        _fill_region(out, region_mask, seed, (r0, c0), _symmetry_kind(seed))

    return out


def _random_palette() -> np.ndarray:
    n = int(RNG.integers(2, 5))
    return np.array(RNG.choice(PALETTE, size=n, replace=False), dtype=int)


def _random_fill(shape: tuple[int, int], palette: np.ndarray) -> np.ndarray:
    idx = RNG.integers(0, len(palette), size=shape)
    return palette[idx]


def _make_seed(kind: str) -> np.ndarray:
    for _ in range(SEED_ATTEMPTS):
        h = int(RNG.integers(2, 5))
        w = int(RNG.integers(2, 5))
        palette = _random_palette()

        if kind == "both":
            seed = np.zeros((h, w), dtype=int)
            for r in range((h + 1) // 2):
                for c in range((w + 1) // 2):
                    color = int(palette[int(RNG.integers(0, len(palette)))])
                    seed[r, c] = color
                    seed[h - 1 - r, c] = color
                    seed[r, w - 1 - c] = color
                    seed[h - 1 - r, w - 1 - c] = color
        elif kind == "h":
            top = _random_fill(((h + 1) // 2, w), palette)
            seed = np.zeros((h, w), dtype=int)
            for r in range(h):
                src = r if r < len(top) else h - 1 - r
                seed[r, :] = top[src, :]
        elif kind == "v":
            left = _random_fill((h, (w + 1) // 2), palette)
            seed = np.zeros((h, w), dtype=int)
            for c in range(w):
                src = c if c < left.shape[1] else w - 1 - c
                seed[:, c] = left[:, src]
        else:
            seed = _random_fill((h, w), palette)

        colors = set(map(int, seed.ravel()))
        if not (2 <= len(colors) <= 4):
            continue
        if _symmetry_kind(seed) == kind:
            return seed.astype(int)

    raise RuntimeError(f"failed to build a seed for branch {kind}")


def _interval_with_overlap(start: tuple[int, int], length: int) -> tuple[int, int]:
    prev_lo, prev_hi = start
    candidates = []
    for lo in range(length):
        for hi in range(lo + 1, length + 1):
            if max(lo, prev_lo) < min(hi, prev_hi):
                candidates.append((lo, hi))
    idx = int(RNG.integers(0, len(candidates)))
    return candidates[idx]


def _make_h_mask(seed: np.ndarray, max_h: int, max_w: int) -> np.ndarray | None:
    h, w = seed.shape
    if h > max_h:
        return None
    max_blocks = min(3, max_w // w)
    if max_blocks < 2:
        return None
    n_blocks = int(RNG.integers(2, max_blocks + 1))
    intervals = [(0, h)]
    for _ in range(1, n_blocks):
        intervals.append(_interval_with_overlap(intervals[-1], h))
    mask = np.zeros((h, n_blocks * w), dtype=bool)
    for i, (lo, hi) in enumerate(intervals):
        mask[lo:hi, i * w : (i + 1) * w] = True
    return mask


def _make_v_mask(seed: np.ndarray, max_h: int, max_w: int) -> np.ndarray | None:
    h, w = seed.shape
    if w > max_w:
        return None
    max_blocks = min(3, max_h // h)
    if max_blocks < 2:
        return None
    n_blocks = int(RNG.integers(2, max_blocks + 1))
    intervals = [(0, w)]
    for _ in range(1, n_blocks):
        intervals.append(_interval_with_overlap(intervals[-1], w))
    mask = np.zeros((n_blocks * h, w), dtype=bool)
    for i, (lo, hi) in enumerate(intervals):
        mask[i * h : (i + 1) * h, lo:hi] = True
    return mask


def _make_both_mask(seed: np.ndarray, max_h: int, max_w: int) -> np.ndarray | None:
    h, w = seed.shape
    max_rows = min(3, max_h // h)
    max_cols = min(3, max_w // w)
    if max_rows < 2 or max_cols < 2:
        return None
    n_rows = int(RNG.integers(2, max_rows + 1))
    n_cols = int(RNG.integers(2, max_cols + 1))
    mask = np.ones((n_rows * h, n_cols * w), dtype=bool)
    if RNG.random() < 0.8:
        r_cut = int(RNG.integers(h, n_rows * h))
        c_cut = int(RNG.integers(w, n_cols * w))
        mask[r_cut:, c_cut:] = False
    return mask


def _make_diag_mask(seed: np.ndarray, max_h: int, max_w: int) -> np.ndarray | None:
    h, w = seed.shape
    max_steps = min(3, max_h // h, max_w // w)
    if max_steps < 2:
        return None
    steps = int(RNG.integers(2, max_steps + 1))
    mask = np.zeros((steps * h, steps * w), dtype=bool)
    for k in range(steps):
        mask[k * h : (k + 1) * h, k * w : (k + 1) * w] = True
    return mask


def _compose_patch(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    inner_h, inner_w = mask.shape
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    interior = np.zeros_like(patch, dtype=bool)
    interior[1 : inner_h + 1, 1 : inner_w + 1] = mask
    boundary = _dilate8(interior) & ~interior
    patch[boundary] = RED
    h, w = seed.shape
    patch[1 : h + 1, 1 : w + 1] = seed
    return patch


def _enclosed_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    inner = []
    for comp in _components(grid != RED, connectivity=8):
        if any(r == 0 or c == 0 or r == H - 1 or c == W - 1 for r, c in comp):
            continue
        inner.append(comp)
    return inner


def _build_patch(kind: str, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        seed = _make_seed(kind)
        inner_h, inner_w = max_h - 2, max_w - 2
        if inner_h < seed.shape[0] or inner_w < seed.shape[1]:
            continue

        if kind == "h":
            mask = _make_h_mask(seed, inner_h, inner_w)
        elif kind == "v":
            mask = _make_v_mask(seed, inner_h, inner_w)
        elif kind == "both":
            mask = _make_both_mask(seed, inner_h, inner_w)
        else:
            mask = _make_diag_mask(seed, inner_h, inner_w)
        if mask is None:
            continue

        patch = _compose_patch(mask, seed)
        if patch.shape[0] > max_h or patch.shape[1] > max_w:
            continue
        inner = _enclosed_components(patch)
        if len(inner) != 1:
            continue
        filled = main(patch)
        if any(filled[r, c] == BLACK for r, c in inner[0]):
            continue
        return patch
    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    nz = patch != BLACK
    sub[nz] = patch[nz]


def _slot_ranges(H: int, W: int) -> list[tuple[int, int, int, int]]:
    split_r = H // 2
    split_c = W // 2
    return [
        (1, split_r - 1, 1, split_c - 1),
        (1, split_r - 1, split_c, W - 2),
        (split_r, H - 2, 1, split_c - 1),
        (split_r, H - 2, split_c, W - 2),
    ]


def _gen_attempt() -> np.ndarray | None:
    H = int(RNG.integers(27, 31))
    W = int(RNG.integers(27, 31))
    grid = np.zeros((H, W), dtype=int)
    branches = list(RNG.permutation(["h", "v", "both", "diag"]))
    slots = _slot_ranges(H, W)

    for branch, (r0, r1, c0, c1) in zip(branches, slots):
        max_h = r1 - r0 + 1
        max_w = c1 - c0 + 1
        patch = _build_patch(branch, max_h, max_w)
        if patch is None:
            return None
        ph, pw = patch.shape
        top = int(RNG.integers(r0, r1 - ph + 2))
        left = int(RNG.integers(c0, c1 - pw + 2))
        if np.any(grid[top : top + ph, left : left + pw] != BLACK):
            return None
        _paste_patch(grid, patch, top, left)

    if len(_enclosed_components(grid)) != 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
