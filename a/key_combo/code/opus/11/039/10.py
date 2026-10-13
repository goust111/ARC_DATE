from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

ALLOWED_WIDTHS = (2, 4, 6, 8)
HEIGHT_CHOICES = tuple(range(6, 21, 2))
MAX_ATTEMPTS = 400


def _full_red_columns(grid: np.ndarray) -> np.ndarray:
    return np.all(grid == RED, axis=0)


def _band_spans(grid: np.ndarray) -> list[tuple[int, int]]:
    is_stripe = _full_red_columns(grid)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for c, stripe in enumerate(is_stripe):
        if stripe:
            if start is not None:
                spans.append((start, c))
                start = None
        elif start is None:
            start = c
    if start is not None:
        spans.append((start, grid.shape[1]))
    return spans


def _band_center(c0: int, width: int) -> float:
    return c0 + width / 2.0


def _seed_for_band(grid: np.ndarray, c0: int) -> np.ndarray:
    return grid[:2, c0 : c0 + 2].copy()


def _expand_seed(seed: np.ndarray, height: int, width: int) -> np.ndarray:
    sh = height // 2
    sw = width // 2
    return np.kron(seed.astype(int), np.ones((sh, sw), dtype=int))


def _render_with_policy(grid: np.ndarray, mirror_mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, total_width = g.shape
    out = np.zeros_like(g)
    stripe_cols = _full_red_columns(g)
    out[:, stripe_cols] = RED
    midline = (total_width - 1) / 2.0

    for c0, c1 in _band_spans(g):
        width = c1 - c0
        seed = _seed_for_band(g, c0)
        block = _expand_seed(seed, height, width)
        center = _band_center(c0, width)
        mirror = False
        if mirror_mode == "position":
            mirror = center > midline
        elif mirror_mode == "all":
            mirror = True
        elif mirror_mode == "none":
            mirror = False
        else:
            raise ValueError(f"unknown mirror mode: {mirror_mode}")
        if mirror:
            block = block[:, ::-1]
        out[:, c0:c1] = block

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Independently scale each stripe-defined band from its 2x2 seed, then mirror only the bands on the right half.

    Input structure:
    - Black background with 3-4 vertical bands separated by full-height red single-column stripes.
    - Each non-red band has even width 2/4/6/8, and the whole grid has even height.
    - The top-left 2x2 cells of each band form a blue/black seed pattern; the rest of the band is black.

    Rule:
    - Detect the red divider columns and treat the non-red runs between them as independent vertical bands.
    - For each band of width W and grid height H, enlarge its 2x2 seed so each seed cell becomes a (H/2) x (W/2) rectangle, producing an H x W pattern.
    - Compute the band's center column as left_edge + W/2 and compare it to the grid midline (total_cols - 1) / 2.
    - Bands with center <= midline keep the enlarged pattern; bands with center > midline are horizontally mirrored. Red divider columns remain unchanged.

    Key insight:
    - The red stripes encode both pieces of hidden structure at once: their spacing determines each band's private scaling factor, and their placement relative to the grid midline determines whether that enlarged band must be mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _width = g.shape
    assert height % 2 == 0, "grid height must be even"

    for c0, c1 in _band_spans(g):
        band_width = c1 - c0
        assert band_width in ALLOWED_WIDTHS, f"unexpected band width {band_width}"
        assert c0 + 2 <= c1, "each band must contain a 2x2 seed"

    return _render_with_policy(g, "position")


def _seed_pool() -> list[np.ndarray]:
    seeds: list[np.ndarray] = []
    for bits in range(16):
        seed = np.array([(bits >> shift) & 1 for shift in (3, 2, 1, 0)], dtype=int).reshape(2, 2)
        if int(seed.sum()) in (0, 4):
            continue
        if np.array_equal(seed, np.fliplr(seed)):
            continue
        seeds.append(seed)
    return seeds


SEED_POOL = _seed_pool()


def _seed_key(seed: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in seed.ravel())


def _parse_instance(grid: np.ndarray) -> list[dict[str, object]]:
    bands: list[dict[str, object]] = []
    midline = (grid.shape[1] - 1) / 2.0
    for c0, c1 in _band_spans(grid):
        width = c1 - c0
        seed = _seed_for_band(grid, c0)
        center = _band_center(c0, width)
        bands.append(
            {
                "c0": c0,
                "c1": c1,
                "width": width,
                "seed": seed,
                "seed_key": _seed_key(seed),
                "right": center > midline,
            }
        )
    return bands


def _has_scale_witness(bands: list[dict[str, object]]) -> bool:
    for i, a in enumerate(bands):
        for b in bands[i + 1 :]:
            if a["right"] != b["right"]:
                continue
            if a["seed_key"] != b["seed_key"]:
                continue
            if a["width"] != b["width"]:
                return True
    return False


def _has_mirror_witness(bands: list[dict[str, object]]) -> bool:
    for i, a in enumerate(bands):
        for b in bands[i + 1 :]:
            if a["right"] == b["right"]:
                continue
            if a["seed_key"] != b["seed_key"]:
                continue
            if a["width"] != b["width"]:
                continue
            return True
    return False


def _instance_ok(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    height, width = map(int, grid.shape)
    if max(height, width) > 30 or height % 2 != 0:
        return False
    if not np.all(np.isin(grid, [BLACK, BLUE, RED])):
        return False

    stripe_cols = _full_red_columns(grid)
    if int(stripe_cols.sum()) not in (2, 3):
        return False

    bands = _parse_instance(grid)
    if len(bands) not in (3, 4):
        return False
    if not any(not bool(b["right"]) for b in bands):
        return False
    if not any(bool(b["right"]) for b in bands):
        return False

    for band in bands:
        c0 = int(band["c0"])
        c1 = int(band["c1"])
        band_width = int(band["width"])
        seed = np.asarray(band["seed"], dtype=int)
        if band_width not in ALLOWED_WIDTHS:
            return False
        if not np.all(np.isin(seed, [BLACK, BLUE])):
            return False
        if np.array_equal(seed, np.fliplr(seed)):
            return False
        if np.any(grid[2:, c0:c1] != BLACK):
            return False

    if not _has_scale_witness(bands):
        return False
    if not _has_mirror_witness(bands):
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False
    if not np.all(out[:, stripe_cols] == RED):
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _render_with_policy(grid, "none")):
        return False
    if np.array_equal(out, _render_with_policy(grid, "all")):
        return False

    return True


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    n_bands = int(rng.choice([3, 4]))
    seed_main = SEED_POOL[int(rng.integers(len(SEED_POOL)))]
    seed_alt = SEED_POOL[int(rng.integers(len(SEED_POOL)))]
    height = int(rng.choice(HEIGHT_CHOICES))

    if n_bands == 3:
        outer = int(rng.choice(ALLOWED_WIDTHS))
        middle_choices = [w for w in ALLOWED_WIDTHS if w != outer]
        middle = int(rng.choice(middle_choices))
        widths = [outer, middle, outer]
        seeds = [seed_main, seed_main, seed_main]
    else:
        valid_pairs = [(a, b) for a in ALLOWED_WIDTHS for b in ALLOWED_WIDTHS if a != b and 2 * (a + b) + 3 <= 30]
        outer, inner = valid_pairs[int(rng.integers(len(valid_pairs)))]
        if rng.random() < 0.5:
            widths = [outer, inner, inner, outer]
        else:
            widths = [inner, outer, outer, inner]
        seeds = [seed_main, seed_main, seed_main, seed_alt if rng.random() < 0.7 else seed_main]

    total_width = sum(widths) + (len(widths) - 1)
    grid = np.zeros((height, total_width), dtype=int)

    c = 0
    for idx, (band_width, seed) in enumerate(zip(widths, seeds)):
        grid[:2, c : c + 2] = seed
        c += band_width
        if idx != len(widths) - 1:
            grid[:, c] = RED
            c += 1

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate(rng)
        if _instance_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid striped-scaling ARC input")
