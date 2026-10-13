import numpy as np


BLACK = 0
GRAY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500
TEXTURE_FAMILIES = ("hstripe", "vstripe", "checker", "tile")
RECT_SIDES = ("top", "bottom", "left", "right")
RECT_SIZES = (5, 7, 9, 11)


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
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _tile_color(tile: np.ndarray, r: int, c: int) -> int:
    return int(tile[r % tile.shape[0], c % tile.shape[1]])


def _rect_border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _marker_position(rect: dict) -> tuple[int, int]:
    r0 = int(rect["r0"])
    c0 = int(rect["c0"])
    r1 = int(rect["r1"])
    c1 = int(rect["c1"])
    side = rect["side"]
    if side == "top":
        return r0 + 1, (c0 + c1) // 2
    if side == "bottom":
        return r1 - 1, (c0 + c1) // 2
    if side == "left":
        return (r0 + r1) // 2, c0 + 1
    return (r0 + r1) // 2, c1 - 1


def _detect_rectangles(g: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in _components(g == GRAY):
        r0, c0 = np.min(comp, axis=0)
        r1, c1 = np.max(comp, axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 5 or width < 5:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = _rect_border_mask(height, width)
        if not np.array_equal(sub == GRAY, border):
            continue

        interior = sub[1:-1, 1:-1]
        marker_locs = np.argwhere((interior != BLACK) & (interior != GRAY))
        assert marker_locs.shape[0] == 1, "each rectangle must contain exactly one marker"
        ir, ic = map(int, marker_locs[0])
        on_sides = [
            ir == 0,
            ir == interior.shape[0] - 1,
            ic == 0,
            ic == interior.shape[1] - 1,
        ]
        assert sum(on_sides) == 1, "marker must lie on exactly one inner edge"

        interior_without_marker = interior.copy()
        interior_without_marker[ir, ic] = BLACK
        assert np.all(interior_without_marker == BLACK), "rectangle interior must otherwise be black"

        if ir == 0:
            side = "top"
        elif ir == interior.shape[0] - 1:
            side = "bottom"
        elif ic == 0:
            side = "left"
        else:
            side = "right"

        rects.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "side": side})

    rects.sort(key=lambda rect: (int(rect["r0"]), int(rect["c0"])))
    assert rects, "expected at least one grey frame"
    return rects


def _occluded_mask(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    occ = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        patch = occ[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(patch), "frames must not overlap"
        patch[:, :] = True
    return occ


def _infer_texture(g: np.ndarray, occluded: np.ndarray) -> np.ndarray:
    h, w = g.shape
    observed = np.argwhere(~occluded)
    assert observed.size > 0, "need visible texture outside the frames"

    best_score = None
    best_tile = None
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            tile = np.full((pr, pc), -1, dtype=int)
            seen = np.zeros((pr, pc), dtype=bool)
            ok = True
            for r, c in observed:
                rr = int(r) % pr
                cc = int(c) % pc
                value = int(g[int(r), int(c)])
                if seen[rr, cc] and int(tile[rr, cc]) != value:
                    ok = False
                    break
                seen[rr, cc] = True
                tile[rr, cc] = value
            if ok and np.all(seen):
                score = (pr * pc, pr + pc, pr, pc)
                if best_score is None or score < best_score:
                    best_score = score
                    best_tile = tile.copy()

    assert best_tile is not None, "failed to infer a fully observed periodic texture"
    return best_tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each marked grey frame so the hidden repeating wallpaper shows through it.

    Input structure:
    - The whole grid is covered by a repeating colored background texture rather than a solid background.
    - One or more grey 1-pixel rectangular frames sit on top of that texture; each frame encloses a black interior.
    - Inside every frame there is exactly one non-grey, non-black marker pixel on the inner side of one edge.

    Rule:
    - Read the marker position to determine which frame edge opens: top marker -> remove the top edge, bottom -> bottom edge, left -> left edge, right -> right edge.
    - Infer the global repeating background texture from the still-visible cells outside all frame bounding boxes.
    - Restore every black interior cell, the marker cell, and the removed grey edge to their original texture colors at the same coordinates.
    - Keep the other three grey frame edges unchanged.

    Key insight:
    The colored marker and the wallpaper repetition encode independent clues: first infer the periodic texture globally, then use the marker to decide which border segment should become transparent.
    """
    g = np.asarray(grid, dtype=int)
    rects = _detect_rectangles(g)
    occluded = _occluded_mask(g.shape, rects)
    tile = _infer_texture(g, occluded)

    out = g.copy()
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        side = rect["side"]

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = _tile_color(tile, r, c)

        if side == "top":
            edge_cells = [(r0, c) for c in range(c0, c1 + 1)]
        elif side == "bottom":
            edge_cells = [(r1, c) for c in range(c0, c1 + 1)]
        elif side == "left":
            edge_cells = [(r, c0) for r in range(r0, r1 + 1)]
        else:
            edge_cells = [(r, c1) for r in range(r0, r1 + 1)]

        for r, c in edge_cells:
            out[r, c] = _tile_color(tile, r, c)

    return out


def _tile_background(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    reps_r = (shape[0] + tile.shape[0] - 1) // tile.shape[0]
    reps_c = (shape[1] + tile.shape[1] - 1) // tile.shape[1]
    return np.tile(tile, (reps_r, reps_c))[: shape[0], : shape[1]].copy()


def _build_texture_tile(rng: np.random.Generator, colors: list[int]) -> np.ndarray:
    family = TEXTURE_FAMILIES[int(rng.integers(0, len(TEXTURE_FAMILIES)))]
    if family == "hstripe":
        height = int(rng.integers(2, 5))
        rows = rng.choice(colors, size=height, replace=True)
        while len(set(map(int, rows.tolist()))) < 2:
            rows = rng.choice(colors, size=height, replace=True)
        return np.array(rows, dtype=int).reshape(height, 1)

    if family == "vstripe":
        width = int(rng.integers(2, 5))
        cols = rng.choice(colors, size=width, replace=True)
        while len(set(map(int, cols.tolist()))) < 2:
            cols = rng.choice(colors, size=width, replace=True)
        return np.array(cols, dtype=int).reshape(1, width)

    if family == "checker":
        height = int(rng.choice(np.array([2, 3])))
        width = int(rng.choice(np.array([2, 3])))
        use = colors[: int(rng.integers(2, min(4, len(colors)) + 1))]
        tile = np.zeros((height, width), dtype=int)
        for r in range(height):
            for c in range(width):
                tile[r, c] = int(use[(r + c) % len(use)])
        return tile

    height = int(rng.integers(2, 5))
    width = int(rng.integers(2, 5))
    tile = rng.choice(colors, size=(height, width), replace=True).astype(int)
    while len(set(map(int, tile.ravel().tolist()))) < 2:
        tile = rng.choice(colors, size=(height, width), replace=True).astype(int)
    return tile


def _pick_rect_count(rng: np.random.Generator) -> int:
    return int(rng.choice(np.array([1, 2, 3, 4])))


def _place_rectangles(rng: np.random.Generator, shape: tuple[int, int], count: int) -> list[dict] | None:
    h, w = shape
    occ = np.zeros((h, w), dtype=bool)
    rects: list[dict] = []
    for _ in range(count):
        placed = False
        for _attempt in range(120):
            heights = [size for size in RECT_SIZES if size <= h - 2]
            widths = [size for size in RECT_SIZES if size <= w - 2]
            if not heights or not widths:
                return None
            height = int(rng.choice(np.array(heights)))
            width = int(rng.choice(np.array(widths)))
            max_r0 = h - height - 1
            max_c0 = w - width - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            r0 = int(rng.integers(1, max_r0 + 1))
            c0 = int(rng.integers(1, max_c0 + 1))
            r1 = r0 + height - 1
            c1 = c0 + width - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r1 + 2)
            cc1 = min(w, c1 + 2)
            if np.any(occ[rr0:rr1, cc0:cc1]):
                continue

            side = RECT_SIDES[int(rng.integers(0, len(RECT_SIDES)))]
            rect = {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "side": side}
            occ[rr0:rr1, cc0:cc1] = True
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None
    rects.sort(key=lambda rect: (int(rect["r0"]), int(rect["c0"])))
    return rects


def _render_input(background: np.ndarray, rects: list[dict], marker_colors: list[int]) -> np.ndarray:
    inp = background.copy()
    for rect, marker_color in zip(rects, marker_colors, strict=False):
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        inp[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        inp[r0, c0 : c1 + 1] = GRAY
        inp[r1, c0 : c1 + 1] = GRAY
        inp[r0 : r1 + 1, c0] = GRAY
        inp[r0 : r1 + 1, c1] = GRAY
        mr, mc = _marker_position(rect)
        inp[mr, mc] = int(marker_color)
    return inp


def _render_expected(background: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = background.copy()
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        side = rect["side"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r not in (r0, r1) and c not in (c0, c1):
                    continue
                removed = (
                    (side == "top" and r == r0)
                    or (side == "bottom" and r == r1)
                    or (side == "left" and c == c0)
                    or (side == "right" and c == c1)
                )
                if not removed:
                    out[r, c] = GRAY
    return out


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(15, 31)),
        int(rng.integers(15, 31)),
    )
    rect_count = _pick_rect_count(rng)

    color_order = list(COLOR_POOL)
    rng.shuffle(color_order)
    max_texture_colors = len(COLOR_POOL) - rect_count
    texture_color_count = int(rng.integers(2, min(4, max_texture_colors) + 1))
    texture_colors = color_order[:texture_color_count]
    marker_colors = color_order[texture_color_count : texture_color_count + rect_count]
    if len(marker_colors) < rect_count:
        return None

    tile = _build_texture_tile(rng, texture_colors)
    background = _tile_background(tile, shape)

    rects = _place_rectangles(rng, shape, rect_count)
    if rects is None:
        return None

    inp = _render_input(background, rects, marker_colors)
    expected = _render_expected(background, rects)

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        inp = _attempt_generate(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid texture-window ARC input")
