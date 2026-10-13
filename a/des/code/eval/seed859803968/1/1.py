import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
PANEL_SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400

STAMP_CATALOG = [
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 1], [0, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size, "expected a non-empty mask"
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _is_panel(region: np.ndarray) -> bool:
    return bool(region.size and not np.any(region == GREY) and np.any((region != BLACK) & (region != GREY)))


def _is_main_area(region: np.ndarray) -> bool:
    return bool(region.size and np.any(region == GREY) and np.any((region != BLACK) & (region != GREY)))


def _split_panel(grid: np.ndarray) -> tuple[str, np.ndarray, tuple[int, int], np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, np.ndarray, tuple[int, int], np.ndarray, tuple[int, int]]] = []

    for r in range(h):
        if not np.all(g[r, :] == GREY):
            continue
        top = g[:r, :]
        bottom = g[r + 1 :, :]
        if _is_panel(top) and _is_main_area(bottom):
            candidates.append(("top", top, (0, 0), bottom, (r + 1, 0)))
        if _is_panel(bottom) and _is_main_area(top):
            candidates.append(("bottom", bottom, (r + 1, 0), top, (0, 0)))

    for c in range(w):
        if not np.all(g[:, c] == GREY):
            continue
        left = g[:, :c]
        right = g[:, c + 1 :]
        if _is_panel(left) and _is_main_area(right):
            candidates.append(("left", left, (0, 0), right, (0, c + 1)))
        if _is_panel(right) and _is_main_area(left):
            candidates.append(("right", right, (0, c + 1), left, (0, 0)))

    assert len(candidates) == 1, "expected one legend divider"
    return candidates[0]


def _extract_stamps(panel: np.ndarray) -> dict[int, np.ndarray]:
    comps = _components(panel != BLACK)
    stamps: dict[int, np.ndarray] = {}
    for coords in comps:
        vals = panel[coords[:, 0], coords[:, 1]]
        colors = {int(v) for v in np.unique(vals)}
        assert len(colors) == 1, "each legend stamp must be monochrome"
        color = int(next(iter(colors)))
        assert color not in stamps, "legend colors must be unique"
        r0, c0, r1, c1 = _bbox(coords)
        stamp = panel[r0 : r1 + 1, c0 : c1 + 1] == color
        stamps[color] = stamp
    assert stamps, "legend panel must contain at least one stamp"
    return stamps


def _extract_regions(main_area: np.ndarray) -> list[dict]:
    regions = []
    for coords in _components(main_area != GREY):
        if _touches_border(coords, main_area.shape):
            continue
        vals = main_area[coords[:, 0], coords[:, 1]]
        colors = {int(v) for v in np.unique(vals)}
        colors.discard(BLACK)
        assert len(colors) == 1, "each enclosed region must contain exactly one seed color"
        color = int(next(iter(colors)))
        assert int(np.sum(vals == color)) == 1, "each enclosed region must contain exactly one seed pixel"
        black_coords = coords[vals == BLACK]
        assert black_coords.size, "each enclosed region must contain at least one black anchor cell"
        order = np.lexsort((black_coords[:, 1], black_coords[:, 0]))
        anchor = tuple(map(int, black_coords[order[0]]))
        regions.append({"color": color, "coords": coords, "anchor": anchor})
    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed color to the same-colored legend stamp and tile that stamp
    inside the corresponding enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    side, panel, _panel_offset, main_area, main_offset = _split_panel(g)
    assert side in PANEL_SIDES
    stamps = _extract_stamps(panel)
    regions = _extract_regions(main_area)

    out = g.copy()
    mr0, mc0 = main_offset
    for region in regions:
        color = int(region["color"])
        assert color in stamps, "every seed color must appear in the legend"
        stamp = stamps[color]
        sh, sw = map(int, stamp.shape)
        ar, ac = map(int, region["anchor"])
        for rr, cc in region["coords"]:
            rr = int(rr)
            cc = int(cc)
            out[mr0 + rr, mc0 + cc] = color if stamp[(rr - ar) % sh, (cc - ac) % sw] else BLACK
    return out


def _random_stamp() -> np.ndarray:
    stamp = STAMP_CATALOG[random.randrange(len(STAMP_CATALOG))].copy()
    stamp = np.rot90(stamp, random.randrange(4))
    if random.random() < 0.5:
        stamp = np.fliplr(stamp)
    if random.random() < 0.5:
        stamp = np.flipud(stamp)
    return _tight_mask(stamp)


def _choose_stamps(count: int) -> list[np.ndarray]:
    chosen: list[np.ndarray] = []
    seen = set()
    attempts = 0
    while len(chosen) < count and attempts < 200:
        attempts += 1
        stamp = _random_stamp()
        key = _mask_key(stamp)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(stamp)
    assert len(chosen) == count, "failed to sample enough distinct stamps"
    return chosen


def _draw_border(patch: np.ndarray) -> None:
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY


def _apply_notch(patch: np.ndarray, side: str, pos: int, depth: int) -> bool:
    trial = patch.copy()
    if side == "top":
        trial[1 : 1 + depth, pos] = GREY
    elif side == "bottom":
        trial[-1 - depth : -1, pos] = GREY
    elif side == "left":
        trial[pos, 1 : 1 + depth] = GREY
    else:
        trial[pos, -1 - depth : -1] = GREY
    if np.array_equal(trial, patch):
        return False
    interiors = _components(trial == BLACK)
    if len(interiors) != 1 or interiors[0].shape[0] < 10:
        return False
    patch[:, :] = trial
    return True


def _make_region_patch(color: int) -> np.ndarray:
    for _ in range(200):
        h = random.randint(7, 10)
        w = random.randint(7, 10)
        patch = np.full((h, w), BLACK, dtype=int)
        _draw_border(patch)

        target_notches = random.randint(1, 4)
        placed = 0
        for _ in range(24):
            side = random.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                pos = random.randint(2, w - 3)
                depth = random.randint(2, h - 4)
            else:
                pos = random.randint(2, h - 3)
                depth = random.randint(2, w - 4)
            placed += int(_apply_notch(patch, side, pos, depth))
            if placed >= target_notches:
                break

        interiors = _components(patch == BLACK)
        if placed == 0 or len(interiors) != 1:
            continue
        interior = interiors[0]
        if interior.shape[0] < 10:
            continue

        sr, sc = map(int, interior[random.randrange(len(interior))])
        patch[sr, sc] = color
        return patch
    raise RuntimeError("failed to generate a valid enclosed region")


def _try_place_regions(main_h: int, main_w: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.full((main_h, main_w), BLACK, dtype=int)
    occupied = np.zeros((main_h, main_w), dtype=bool)
    for patch in sorted(patches, key=lambda x: x.shape[0] * x.shape[1], reverse=True):
        ph, pw = map(int, patch.shape)
        if ph + 2 >= main_h or pw + 2 >= main_w:
            return None
        placed = False
        for _ in range(200):
            r0 = random.randint(1, main_h - ph - 1)
            c0 = random.randint(1, main_w - pw - 1)
            rr0 = max(0, r0 - 2)
            cc0 = max(0, c0 - 2)
            rr1 = min(main_h, r0 + ph + 2)
            cc1 = min(main_w, c0 + pw + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _legend_requirements(stamps: list[np.ndarray], side: str) -> tuple[int, int]:
    max_gap_total = 3 * max(0, len(stamps) - 1)
    if side in ("top", "bottom"):
        need_len = 2 + sum(stamp.shape[1] for stamp in stamps) + max_gap_total
        thickness = max(stamp.shape[0] for stamp in stamps) + random.randint(2, 4)
    else:
        need_len = 2 + sum(stamp.shape[0] for stamp in stamps) + max_gap_total
        thickness = max(stamp.shape[1] for stamp in stamps) + random.randint(2, 4)
    return int(need_len), int(thickness)


def _render_panel(height: int, width: int, pairs: list[tuple[int, np.ndarray]], side: str) -> np.ndarray:
    panel = np.full((height, width), BLACK, dtype=int)
    if side in ("top", "bottom"):
        gaps = [random.randint(1, 3) for _ in range(max(0, len(pairs) - 1))]
        need = 2 + sum(stamp.shape[1] for _, stamp in pairs) + sum(gaps)
        cursor = 1 + random.randint(0, width - need)
        for idx, (color, stamp) in enumerate(pairs):
            sh, sw = map(int, stamp.shape)
            r0 = random.randint(0, height - sh)
            region = panel[r0 : r0 + sh, cursor : cursor + sw]
            region[stamp] = color
            cursor += sw
            if idx < len(gaps):
                cursor += gaps[idx]
    else:
        gaps = [random.randint(1, 3) for _ in range(max(0, len(pairs) - 1))]
        need = 2 + sum(stamp.shape[0] for _, stamp in pairs) + sum(gaps)
        cursor = 1 + random.randint(0, height - need)
        for idx, (color, stamp) in enumerate(pairs):
            sh, sw = map(int, stamp.shape)
            c0 = random.randint(0, width - sw)
            region = panel[cursor : cursor + sh, c0 : c0 + sw]
            region[stamp] = color
            cursor += sh
            if idx < len(gaps):
                cursor += gaps[idx]
    return panel


def _build_input() -> np.ndarray | None:
    region_count = random.randint(2, 4)
    side = random.choice(PANEL_SIDES)
    colors = random.sample(COLOR_POOL, region_count)
    stamps = _choose_stamps(region_count)
    pairs = list(zip(colors, stamps, strict=False))
    random.shuffle(pairs)

    patches = [_make_region_patch(color) for color, _stamp in pairs]
    max_patch_h = max(patch.shape[0] for patch in patches)
    max_patch_w = max(patch.shape[1] for patch in patches)
    sum_patch_h = sum(patch.shape[0] for patch in patches)
    sum_patch_w = sum(patch.shape[1] for patch in patches)
    legend_need, panel_thickness = _legend_requirements([stamp for _color, stamp in pairs], side)

    min_main_h = max(18, max_patch_h + 3, (sum_patch_h + 1) // 2 + 4)
    min_main_w = max(18, max_patch_w + 3, (sum_patch_w + 1) // 2 + 4)
    if side in ("top", "bottom"):
        min_main_w = max(min_main_w, legend_need)
    else:
        min_main_h = max(min_main_h, legend_need)
    if min_main_h > 24 or min_main_w > 24:
        return None

    main_h = random.randint(min_main_h, 24)
    main_w = random.randint(min_main_w, 24)
    main_area = _try_place_regions(main_h, main_w, patches)
    if main_area is None:
        return None

    if side in ("top", "bottom"):
        panel = _render_panel(panel_thickness, main_w, [(color, stamp) for color, stamp in pairs], side)
        divider = np.full((1, main_w), GREY, dtype=int)
        grid = np.vstack([panel, divider, main_area]) if side == "top" else np.vstack([main_area, divider, panel])
    else:
        panel = _render_panel(main_h, panel_thickness, [(color, stamp) for color, stamp in pairs], side)
        divider = np.full((main_h, 1), GREY, dtype=int)
        grid = np.hstack([panel, divider, main_area]) if side == "left" else np.hstack([main_area, divider, panel])

    if np.any(grid == GREY) and np.sum(np.all(grid == GREY, axis=0)) + np.sum(np.all(grid == GREY, axis=1)) != 1:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.sum(out != grid)) < region_count * 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
