import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("TL", "TR", "BR", "BL")
ROTATION_K = {"TL": 0, "TR": 3, "BR": 2, "BL": 1}
GEN_RETRIES = 400
PATTERN_DIMS = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (4, 4)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _rotation_count(mask: np.ndarray) -> int:
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        seen.add((rot.shape, rot.tobytes()))
    return len(seen)


def _quadrant_of(seed_r: int, seed_c: int, inner_h: int, inner_w: int) -> str:
    row_cmp = 2 * seed_r - (inner_h - 1)
    col_cmp = 2 * seed_c - (inner_w - 1)
    assert row_cmp != 0 and col_cmp != 0, "seed must not lie on a center line"
    if row_cmp < 0 and col_cmp < 0:
        return "TL"
    if row_cmp < 0 and col_cmp > 0:
        return "TR"
    if row_cmp > 0 and col_cmp > 0:
        return "BR"
    return "BL"


def _stamp_clipped(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    top: int,
    left: int,
    clip_rect: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = clip_rect
    mh, mw = mask.shape
    for r in range(mh):
        gr = top + r
        if gr < r0 or gr > r1:
            continue
        for c in range(mw):
            if not mask[r, c]:
                continue
            gc = left + c
            if c0 <= gc <= c1:
                out[gr, gc] = color


def _is_rectangular_fence(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return False
    pts = {tuple(map(int, rc)) for rc in coords}
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced interior with the matching outside pattern, rotated by the seed's quadrant.

    Input structure:
    - Background is black.
    - Several one-cell-thick grey rectangular fences enclose black interiors of varying sizes.
    - Each fence interior contains exactly one non-black, non-grey seed pixel.
    - Outside the fences, each seed color appears once as a monochrome pattern component.

    Rule:
    - For each fence, use the seed color to select the outside pattern of the same color.
    - Determine which interior quadrant contains the seed, using the interior center as the divider.
    - Rotate the chosen pattern by 0 degrees for top-left, 90 degrees clockwise for top-right,
      180 degrees for bottom-right, and 270 degrees clockwise for bottom-left.
    - Draw the rotated pattern with its bounding box top-left anchored at the seed position,
      clipping any cells that extend beyond the fence interior.

    Key insight:
    - The seed pixel encodes two independent facts at once: its color selects the pattern and
      its quadrant selects the rotation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    fences: list[dict] = []
    seed_positions: set[tuple[int, int]] = set()
    for coords in _components(g == GRAY):
        if not _is_rectangular_fence(coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((inner != BLACK) & (inner != GRAY))
        assert seeds.shape[0] == 1, "each fence must contain exactly one seed"
        sr, sc = map(int, seeds[0])
        color = int(inner[sr, sc])
        quadrant = _quadrant_of(sr, sc, int(inner.shape[0]), int(inner.shape[1]))
        seed_abs = (r0 + 1 + sr, c0 + 1 + sc)
        seed_positions.add(seed_abs)
        fences.append(
            {
                "color": color,
                "quadrant": quadrant,
                "seed_abs": seed_abs,
                "clip_rect": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )

    patterns: dict[int, np.ndarray] = {}
    for coords in _components((g != BLACK) & (g != GRAY)):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "pattern components must be monochrome"
        color = colors.pop()
        if coords.shape[0] == 1 and tuple(map(int, coords[0])) in seed_positions:
            continue
        patterns[color] = _crop_mask(coords)

    for fence in fences:
        color = int(fence["color"])
        assert color in patterns, "every seed color must have a matching outside pattern"
        rotated = np.rot90(patterns[color], ROTATION_K[str(fence["quadrant"])])
        seed_r, seed_c = fence["seed_abs"]
        _stamp_clipped(out, rotated, color, int(seed_r), int(seed_c), fence["clip_rect"])

    return out


def _quadrant_cells(inner_h: int, inner_w: int, quadrant: str) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            try:
                q = _quadrant_of(r, c, inner_h, inner_w)
            except AssertionError:
                continue
            if q == quadrant:
                cells.append((r, c))
    return cells


def _visible_cells(mask: np.ndarray, seed_r: int, seed_c: int, inner_h: int, inner_w: int) -> int:
    vis_h = max(0, min(int(mask.shape[0]), inner_h - seed_r))
    vis_w = max(0, min(int(mask.shape[1]), inner_w - seed_c))
    if vis_h == 0 or vis_w == 0:
        return 0
    return int(mask[:vis_h, :vis_w].sum())


def _make_pattern_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        ph, pw = rng.choice(PATTERN_DIMS)
        mask = np.ones((ph, pw), dtype=bool)
        removable = [
            (r, c)
            for r in range(ph)
            for c in range(pw)
            if (r, c) not in {(0, 0), (0, pw - 1), (ph - 1, 0), (ph - 1, pw - 1)}
        ]
        rng.shuffle(removable)
        min_area = max(4, (ph * pw + 1) // 2)
        target = rng.randint(1, max(1, len(removable)))
        removed = 0
        for r, c in removable:
            if removed >= target:
                break
            if rng.random() < 0.75:
                trial = mask.copy()
                trial[r, c] = False
                if int(trial.sum()) < min_area or not _is_connected(trial):
                    continue
                mask = trial
                removed += 1
        if _rotation_count(mask) != 4:
            continue
        return mask
    raise RuntimeError("failed to build a non-symmetric pattern")


def _make_fence_spec(quadrant: str, color: int, rng: random.Random) -> dict:
    for _ in range(200):
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(4, 7)
        pattern = _make_pattern_mask(rng)
        rotated = np.rot90(pattern, ROTATION_K[quadrant])
        candidates = _quadrant_cells(inner_h, inner_w, quadrant)
        valid: list[tuple[int, int, bool]] = []
        for seed_r, seed_c in candidates:
            visible = _visible_cells(rotated, seed_r, seed_c, inner_h, inner_w)
            if visible < min(3, int(rotated.sum())):
                continue
            clipped = visible < int(rotated.sum())
            valid.append((seed_r, seed_c, clipped))
        if not valid:
            continue
        clipped_cells = [(r, c) for r, c, clipped in valid if clipped]
        unclipped_cells = [(r, c) for r, c, clipped in valid if not clipped]
        if clipped_cells and unclipped_cells:
            pool = clipped_cells if rng.random() < 0.5 else unclipped_cells
        else:
            pool = clipped_cells or unclipped_cells
        seed_r, seed_c = rng.choice(pool)
        return {
            "color": color,
            "quadrant": quadrant,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": inner_h + 2,
            "outer_w": inner_w + 2,
            "seed_local": (seed_r, seed_c),
            "pattern": pattern,
        }
    raise RuntimeError("failed to build a valid fence spec")


def _can_place(used: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = used.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(used[r0:r1, c0:c1].any())


def _pick_location(
    used: np.ndarray, height: int, width: int, rng: random.Random, pad: int = 1
) -> tuple[int, int] | None:
    h, w = used.shape
    candidates: list[tuple[int, int]] = []
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            if _can_place(used, top, left, height, width, pad=pad):
                candidates.append((top, left))
    if not candidates:
        return None
    return rng.choice(candidates)


def _draw_fence(grid: np.ndarray, spec: dict) -> None:
    top = int(spec["top"])
    left = int(spec["left"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    color = int(spec["color"])
    seed_r, seed_c = spec["seed_local"]
    grid[top, left : left + outer_w] = GRAY
    grid[top + outer_h - 1, left : left + outer_w] = GRAY
    grid[top : top + outer_h, left] = GRAY
    grid[top : top + outer_h, left + outer_w - 1] = GRAY
    grid[top + 1 + seed_r, left + 1 + seed_c] = color


def _draw_pattern(grid: np.ndarray, spec: dict) -> None:
    top = int(spec["pattern_top"])
    left = int(spec["pattern_left"])
    mask = spec["pattern"]
    color = int(spec["color"])
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = color


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        rotated = np.rot90(spec["pattern"], ROTATION_K[str(spec["quadrant"])])
        seed_r = int(spec["top"] + 1 + spec["seed_local"][0])
        seed_c = int(spec["left"] + 1 + spec["seed_local"][1])
        clip_rect = (
            int(spec["top"] + 1),
            int(spec["left"] + 1),
            int(spec["top"] + spec["outer_h"] - 2),
            int(spec["left"] + spec["outer_w"] - 2),
        )
        _stamp_clipped(out, rotated, int(spec["color"]), seed_r, seed_c, clip_rect)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = 4 + (1 if rng.random() < 0.35 else 0)
    colors = rng.sample(PALETTE, count)
    quadrants = list(QUADRANTS)
    rng.shuffle(quadrants)
    if count == 5:
        quadrants.append(rng.choice(QUADRANTS))

    specs = [_make_fence_spec(quadrants[i], colors[i], rng) for i in range(count)]
    total_area = sum(int(s["outer_h"] * s["outer_w"] + s["pattern"].size) for s in specs)
    min_side = max(20, int(np.ceil((total_area * 1.65) ** 0.5)))
    if min_side > 30:
        return None
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)
    grid = np.full((height, width), BLACK, dtype=int)
    used = np.zeros((height, width), dtype=bool)

    for spec in sorted(specs, key=lambda s: (-int(s["outer_h"] * s["outer_w"]), rng.random())):
        loc = _pick_location(used, int(spec["outer_h"]), int(spec["outer_w"]), rng, pad=1)
        if loc is None:
            return None
        top, left = loc
        spec["top"] = top
        spec["left"] = left
        used[top : top + int(spec["outer_h"]), left : left + int(spec["outer_w"])] = True
        _draw_fence(grid, spec)

    for spec in sorted(specs, key=lambda s: (-int(s["pattern"].size), rng.random())):
        ph, pw = spec["pattern"].shape
        loc = _pick_location(used, int(ph), int(pw), rng, pad=1)
        if loc is None:
            return None
        top, left = loc
        spec["pattern_top"] = top
        spec["pattern_left"] = left
        used[top : top + int(ph), left : left + int(pw)] = True
        _draw_pattern(grid, spec)

    expected = _expected_output(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
