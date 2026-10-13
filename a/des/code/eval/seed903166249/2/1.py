import random
from collections import Counter

import numpy as np


BG = 0
AXIS = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
BRANCHES = ("h_left", "h_right", "v_up", "v_down")
MAX_RETRIES = 300


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _quadrant_slices(h: int, w: int, mid_r: int, mid_c: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(0, mid_r), slice(0, mid_c)),
        "tr": (slice(0, mid_r), slice(mid_c + 1, w)),
        "bl": (slice(mid_r + 1, h), slice(0, mid_c)),
        "br": (slice(mid_r + 1, h), slice(mid_c + 1, w)),
    }


def _dominant_color(patch: np.ndarray, bg: int) -> int:
    vals = patch[patch != bg]
    counts = Counter(map(int, vals.tolist()))
    [(color, best_count)] = counts.most_common(1)
    assert sum(count == best_count for count in counts.values()) == 1
    return int(color)


def _stamp_transparent(
    out: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    bg: int,
    occupied: np.ndarray | None = None,
) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    assert 0 <= top and 0 <= left and top + ph <= h and left + pw <= w
    for rr in range(ph):
        for cc in range(pw):
            val = int(patch[rr, cc])
            if val == bg:
                continue
            tr = top + rr
            tc = left + cc
            if occupied is not None:
                assert not occupied[tr, tc]
                occupied[tr, tc] = True
            out[tr, tc] = val


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror matched source templates from the seed-guided cross into target quadrants."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _mode_color(g)
    full_rows = np.flatnonzero(np.all(g != bg, axis=1))
    full_cols = np.flatnonzero(np.all(g != bg, axis=0))
    assert full_rows.size == 1 and full_cols.size == 1
    mid_r = int(full_rows[0])
    mid_c = int(full_cols[0])

    axis_color = _mode_color(np.concatenate((g[mid_r, :], g[:, mid_c])))
    quadrants = _quadrant_slices(h, w, mid_r, mid_c)

    source_name = None
    source_count = -1
    for name, (rs, cs) in quadrants.items():
        count = int(np.count_nonzero(g[rs, cs] != bg))
        if count > source_count:
            source_count = count
            source_name = name
    assert source_name is not None and source_count > 0

    source = g[quadrants[source_name]].copy()
    comps = _components(source != bg)
    assert comps

    templates: dict[int, np.ndarray] = {}
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = source[r0 : r1 + 1, c0 : c1 + 1].copy()
        color = _dominant_color(patch, bg)
        assert color not in templates
        templates[color] = patch

    seeds: list[tuple[str, int, int, int]] = []
    for c in range(w):
        if c == mid_c:
            continue
        val = int(g[mid_r, c])
        if val != axis_color:
            branch = "h_left" if c < mid_c else "h_right"
            seeds.append((branch, mid_r, c, val))
    for r in range(h):
        if r == mid_r:
            continue
        val = int(g[r, mid_c])
        if val != axis_color:
            branch = "v_up" if r < mid_r else "v_down"
            seeds.append((branch, r, mid_c, val))

    assert seeds
    assert {color for _, _, _, color in seeds} <= set(templates)

    out = np.full_like(g, bg)
    out[mid_r, :] = axis_color
    out[:, mid_c] = axis_color
    occupied = np.zeros_like(g, dtype=bool)
    occupied[mid_r, :] = True
    occupied[:, mid_c] = True

    for branch, r, c, color in seeds:
        patch = templates[color]
        if branch.startswith("h_"):
            patch = np.flipud(patch)
            top = mid_r + 1
            left = c
            if branch == "h_left":
                assert left + patch.shape[1] <= mid_c
            else:
                assert left >= mid_c + 1
        else:
            patch = np.fliplr(patch)
            top = r
            left = mid_c + 1
            if branch == "v_up":
                assert top + patch.shape[0] <= mid_r
            else:
                assert top >= mid_r + 1
        _stamp_transparent(out, patch, top, left, bg, occupied)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _generate_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    max_removed = min(4, height * width - 5)
    remove_target = random.randint(0, max_removed)
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)

    removed = 0
    for r, c in cells:
        if removed >= remove_target:
            break
        trial = mask.copy()
        trial[r, c] = False
        if not _is_connected(trial):
            continue
        rows = np.flatnonzero(trial.any(axis=1))
        cols = np.flatnonzero(trial.any(axis=0))
        if rows[0] != 0 or rows[-1] != height - 1 or cols[0] != 0 or cols[-1] != width - 1:
            continue
        if trial.sum() < 5:
            continue
        mask = trial
        removed += 1
    return mask


def _generate_template(main_color: int, accent_pool: list[int]) -> np.ndarray:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    mask = _generate_mask(height, width)
    patch = np.full((height, width), BG, dtype=int)
    patch[mask] = main_color

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    max_accents = min(3, max(1, len(coords) // 3))
    accent_count = random.randint(1, max_accents)
    accent_cells = random.sample(coords, accent_count)
    accent_choices = accent_pool.copy()
    random.shuffle(accent_choices)
    accent_colors = accent_choices[: max(1, min(2, len(accent_choices)))]
    for idx, (r, c) in enumerate(accent_cells):
        patch[r, c] = accent_colors[idx % len(accent_colors)]

    assert _dominant_color(patch, BG) == main_color
    return patch


def _clear_with_padding(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _choose_box(
    region_h: int,
    region_w: int,
    height: int,
    width: int,
    occupied: np.ndarray,
    pad: int,
    fixed_row: int | None = None,
    fixed_col: int | None = None,
) -> tuple[int, int] | None:
    rows = [fixed_row] if fixed_row is not None else list(range(region_h - height + 1))
    cols = [fixed_col] if fixed_col is not None else list(range(region_w - width + 1))
    candidates: list[tuple[int, int]] = []
    for r in rows:
        for c in cols:
            if _clear_with_padding(occupied, r, c, height, width, pad):
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _expected_from_specs(
    shape: tuple[int, int],
    mid_r: int,
    mid_c: int,
    specs: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[mid_r, :] = AXIS
    out[:, mid_c] = AXIS
    occupied = np.zeros(shape, dtype=bool)
    occupied[mid_r, :] = True
    occupied[:, mid_c] = True

    quad_origin = {
        "tr": (0, mid_c + 1),
        "bl": (mid_r + 1, 0),
        "br": (mid_r + 1, mid_c + 1),
    }
    branch_quad = {"h_left": "bl", "h_right": "br", "v_up": "tr", "v_down": "br"}

    for spec in specs:
        patch = spec["patch"]
        if spec["branch"].startswith("h_"):
            patch = np.flipud(patch)
        else:
            patch = np.fliplr(patch)
        quad = branch_quad[spec["branch"]]
        qr, qc = quad_origin[quad]
        rr, cc = spec["target_local"]
        _stamp_transparent(out, patch, qr + rr, qc + cc, BG, occupied)
    return out


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(11, 16)
    qw = random.randint(11, 16)
    height = 2 * qh + 1
    width = 2 * qw + 1
    mid_r = qh
    mid_c = qw

    branch_count = random.randint(2, 4)
    chosen_branches = random.sample(BRANCHES, branch_count)
    main_colors = random.sample(COLOR_POOL, branch_count)
    accent_pool = [color for color in COLOR_POOL if color not in main_colors]

    specs: list[dict] = []
    for branch, color in zip(chosen_branches, main_colors, strict=True):
        patch = _generate_template(color, accent_pool)
        specs.append({"branch": branch, "color": color, "patch": patch})

    source_occ = np.zeros((qh, qw), dtype=bool)
    for spec in sorted(specs, key=lambda item: int(np.count_nonzero(item["patch"])), reverse=True):
        ph, pw = spec["patch"].shape
        pos = _choose_box(qh, qw, ph, pw, source_occ, pad=1)
        if pos is None:
            return None
        spec["source_local"] = pos
        r0, c0 = pos
        source_occ[r0 : r0 + ph, c0 : c0 + pw] = True

    target_occ = {
        "tr": np.zeros((qh, qw), dtype=bool),
        "bl": np.zeros((qh, qw), dtype=bool),
        "br": np.zeros((qh, qw), dtype=bool),
    }
    branch_quad = {"h_left": "bl", "h_right": "br", "v_up": "tr", "v_down": "br"}

    for spec in random.sample(specs, len(specs)):
        branch = spec["branch"]
        quad = branch_quad[branch]
        ph, pw = spec["patch"].shape
        pos = _choose_box(
            qh,
            qw,
            ph,
            pw,
            target_occ[quad],
            pad=1,
            fixed_row=0 if branch.startswith("h_") else None,
            fixed_col=0 if branch.startswith("v_") else None,
        )
        if pos is None:
            return None
        spec["target_local"] = pos
        r0, c0 = pos
        target_occ[quad][r0 : r0 + ph, c0 : c0 + pw] = True

    grid = np.full((height, width), BG, dtype=int)
    grid[mid_r, :] = AXIS
    grid[:, mid_c] = AXIS

    for spec in specs:
        sr, sc = spec["source_local"]
        _stamp_transparent(grid, spec["patch"], sr, sc, BG)

    for spec in specs:
        rr, cc = spec["target_local"]
        if spec["branch"] == "h_left":
            grid[mid_r, cc] = spec["color"]
        elif spec["branch"] == "h_right":
            grid[mid_r, mid_c + 1 + cc] = spec["color"]
        elif spec["branch"] == "v_up":
            grid[rr, mid_c] = spec["color"]
        else:
            grid[mid_r + 1 + rr, mid_c] = spec["color"]

    expected = _expected_from_specs(grid.shape, mid_r, mid_c, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
