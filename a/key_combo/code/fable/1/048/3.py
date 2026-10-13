import numpy as np


BG = 0
TERRAIN = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
GEN_HEIGHT_RANGE = (22, 29)
GEN_WIDTH_RANGE = (12, 19)
GEN_PIECE_COUNT_VALUES = (2, 3, 4)
GEN_PIECE_COUNT_WEIGHTS = (0.2, 0.55, 0.25)
GEN_SURFACE_TOP_MARGIN = (4, 7)
GEN_SURFACE_HEIGHT_SPAN = (7, 11)
GEN_SEGMENT_LEN = (1, 4)
GEN_PIECE_WIDTH = (2, 6)
GEN_PIECE_HEIGHT_MAX = 5


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_color(grid: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(grid == int(color))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)), (int(r0), int(c0), int(r1), int(c1))


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, TERRAIN)):
        mask, bbox = _crop_color(grid, color)
        assert _is_connected(mask), "each colored piece must be 4-connected"
        assert bool(mask.any(axis=0).all()), "piece span must occupy every column in its bbox"
        for c in range(mask.shape[1]):
            rows = np.flatnonzero(mask[:, c])
            assert rows.size > 0 and int(rows[-1] - rows[0] + 1) == int(rows.size), (
                "piece columns must be vertically solid"
            )
        pieces.append({"color": color, "mask": mask, "bbox": bbox, "bottom": int(bbox[2])})
    bottoms = [piece["bottom"] for piece in pieces]
    assert len(bottoms) == len(set(bottoms)), "initial piece bottom rows must be distinct"
    pieces.sort(key=lambda piece: piece["bottom"], reverse=True)
    return pieces


def _surface_rows(settled: np.ndarray) -> np.ndarray:
    h, w = settled.shape
    surface = np.full(w, h, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(settled[:, c] != BG)
        if rows.size:
            surface[c] = int(rows[0])
    return surface


def _bottom_profile(mask: np.ndarray) -> np.ndarray:
    return np.array([int(np.flatnonzero(mask[:, c])[-1]) for c in range(mask.shape[1])], dtype=int)


def _candidate_landings(settled: np.ndarray, piece_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = piece_mask.shape
    H, W = settled.shape
    occ = settled != BG
    surface = _surface_rows(settled)
    bottom = _bottom_profile(piece_mask)
    candidates = []

    for x in range(W - w + 1):
        cols = surface[x : x + w]
        if np.any(cols == H):
            continue
        row_candidates = cols - 1 - bottom
        if not np.all(row_candidates == row_candidates[0]):
            continue
        r = int(row_candidates[0])
        if r < 0 or r + h > H:
            continue
        if np.any(occ[r : r + h, x : x + w] & piece_mask):
            continue
        if not np.all(cols == r + bottom + 1):
            continue
        if r + 1 <= H - h and not np.any(occ[r + 1 : r + 1 + h, x : x + w] & piece_mask):
            continue
        candidates.append((r, x))
    return candidates


def _paint_piece(grid: np.ndarray, piece_mask: np.ndarray, color: int, r: int, c: int) -> None:
    h, w = piece_mask.shape
    view = grid[r : r + h, c : c + w]
    view[piece_mask] = int(color)


def _solve(grid: np.ndarray, *, ascending: bool = False) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    settled = np.where(g == TERRAIN, TERRAIN, BG)
    pieces = _extract_pieces(g)
    if ascending:
        pieces = list(reversed(pieces))

    trace = []
    for piece in pieces:
        candidates = _candidate_landings(settled, piece["mask"])
        assert len(candidates) == 1, "each piece must have exactly one flush-fit landing offset"
        r, c = candidates[0]
        _paint_piece(settled, piece["mask"], piece["color"], r, c)
        trace.append((piece["color"], r, c))
    return settled, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the floating colored pieces onto the unique complementary terrain slots from low to high.

    Input structure:
    - The background is black.
    - A grey stepped terrain occupies the lower part of the grid.
    - Several differently colored solid pieces float above it, with distinct initial lowest rows.

    Rule:
    - Process the floating pieces from the initially lowest one to the highest one.
    - For each piece, ignore its starting column and find the unique horizontal offset whose bottom profile is exactly complementary to the current terrain top profile, with no gaps and no overlaps.
    - Translate the piece to that offset, let it fall vertically into the flush-fit position, and merge it into the terrain for the next piece.

    Key insight:
    - Placement is determined by profile matching against the evolving terrain, not by dropping straight down from the decoy starting positions.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _make_piece_mask(profile: tuple[int, ...]) -> np.ndarray:
    width = len(profile)
    height = max(profile) + 1
    mask = np.zeros((height, width), dtype=bool)
    for c, depth in enumerate(profile):
        mask[: depth + 1, c] = True
    return mask


def _random_surface(rng: np.random.Generator, H: int, W: int) -> np.ndarray:
    upper = H - int(rng.integers(*GEN_SURFACE_TOP_MARGIN))
    lower = max(3, upper - int(rng.integers(*GEN_SURFACE_HEIGHT_SPAN)))
    value = int(rng.integers(lower, upper + 1))
    surface = np.empty(W, dtype=int)
    c = 0
    while c < W:
        seg_len = int(rng.integers(*GEN_SEGMENT_LEN))
        delta = int(rng.choice([-2, -1, 0, 1, 2], p=[0.08, 0.22, 0.4, 0.22, 0.08]))
        if c > 0:
            value = int(np.clip(value + delta, lower, upper))
        end = min(W, c + seg_len)
        surface[c:end] = value
        c = end
    if np.all(surface == surface[0]):
        surface[W // 2 :] = np.clip(surface[W // 2 :] - 1, lower, upper)
    return surface


def _relative_profile(window: np.ndarray) -> tuple[int, ...]:
    m = int(window.min())
    return tuple(int(v - m) for v in window)


def _unique_profile_options(surface: np.ndarray, used_profiles: set[tuple[int, tuple[int, ...]]]) -> list[dict]:
    W = len(surface)
    options = []
    max_width = min(GEN_PIECE_WIDTH[1], W)
    for width in range(GEN_PIECE_WIDTH[0], max_width + 1):
        buckets: dict[tuple[int, ...], list[int]] = {}
        for x in range(W - width + 1):
            window = surface[x : x + width]
            profile = _relative_profile(window)
            landing_row = int(window.min()) - 1
            if landing_row < 0 or max(profile) + 1 > GEN_PIECE_HEIGHT_MAX:
                continue
            buckets.setdefault(profile, []).append(x)
        for profile, xs in buckets.items():
            key = (width, profile)
            if len(xs) == 1 and key not in used_profiles:
                x = xs[0]
                window = surface[x : x + width]
                options.append(
                    {
                        "x": int(x),
                        "width": int(width),
                        "profile": profile,
                        "landing_row": int(window.min()) - 1,
                        "nonflat": int(max(profile)) > 0,
                    }
                )
    return options


def _plan_scene(rng: np.random.Generator, H: int, W: int) -> tuple[np.ndarray, list[dict]] | None:
    initial_surface = _random_surface(rng, H, W)
    current_surface = initial_surface.copy()
    piece_count = int(rng.choice(GEN_PIECE_COUNT_VALUES, p=GEN_PIECE_COUNT_WEIGHTS))
    need_dependency = piece_count >= 3 and bool(rng.random() < 0.85)
    dependency_done = not need_dependency
    dependency_index = int(rng.integers(1, piece_count)) if need_dependency else -1
    require_flat = piece_count >= 3 and bool(rng.random() < 0.45)
    flat_done = False
    nonflat_done = False
    modified = np.zeros(W, dtype=bool)
    used_profiles: set[tuple[int, tuple[int, ...]]] = set()
    pieces = []

    for i in range(piece_count):
        options = _unique_profile_options(current_surface, used_profiles)
        if not options:
            return None

        if i == dependency_index or (need_dependency and not dependency_done and piece_count - i <= 2):
            dependent = [opt for opt in options if bool(modified[opt["x"] : opt["x"] + opt["width"]].any())]
            if not dependent:
                return None
            options = dependent

        if require_flat and not flat_done and piece_count - i <= 2:
            flat_options = [opt for opt in options if not opt["nonflat"]]
            if not flat_options:
                return None
            options = flat_options

        if not nonflat_done and piece_count - i == 1:
            nonflat_options = [opt for opt in options if opt["nonflat"]]
            if not nonflat_options:
                return None
            options = nonflat_options

        choice = options[int(rng.integers(len(options)))]
        profile = tuple(int(v) for v in choice["profile"])
        width = int(choice["width"])
        x = int(choice["x"])
        landing_row = int(choice["landing_row"])
        mask = _make_piece_mask(profile)
        uses_modified = bool(modified[x : x + width].any())
        pieces.append(
            {
                "profile": profile,
                "mask": mask,
                "final_row": landing_row,
                "final_col": x,
                "uses_modified": uses_modified,
            }
        )
        used_profiles.add((width, profile))
        current_surface[x : x + width] = landing_row
        modified[x : x + width] = True
        dependency_done = dependency_done or uses_modified
        flat_done = flat_done or (max(profile) == 0)
        nonflat_done = nonflat_done or (max(profile) > 0)

    if need_dependency and not dependency_done:
        return None
    if require_flat and not flat_done:
        return None
    if not nonflat_done:
        return None
    if len({(piece["mask"].shape[1], piece["profile"]) for piece in pieces}) != len(pieces):
        return None
    return initial_surface, pieces


def _terrain_from_surface(surface: np.ndarray, H: int) -> np.ndarray:
    W = len(surface)
    grid = np.zeros((H, W), dtype=int)
    for c, top in enumerate(surface):
        grid[int(top) :, c] = TERRAIN
    return grid


def _assign_floating_positions(
    rng: np.random.Generator, surface: np.ndarray, pieces: list[dict]
) -> list[tuple[int, int]] | None:
    highest_terrain = int(surface.min())
    upper_bound = highest_terrain - 2
    heights = [int(piece["mask"].shape[0]) for piece in pieces]
    min_rows_needed = sum(heights) + max(0, len(pieces) - 1)
    if min_rows_needed > upper_bound + 1:
        return None

    extra = int(upper_bound + 1 - min_rows_needed)
    extras = rng.multinomial(extra, np.full(len(pieces), 1.0 / len(pieces)))

    tops = [0] * len(pieces)
    current = int(extras[0])
    for rev_idx, piece_idx in enumerate(range(len(pieces) - 1, -1, -1)):
        height = heights[piece_idx]
        tops[piece_idx] = current
        current += height
        if rev_idx < len(pieces) - 1:
            current += 1 + int(extras[rev_idx + 1])

    if tops[0] + heights[0] - 1 > upper_bound:
        return None

    positions = []
    moved = False
    W = len(surface)
    for piece, top in zip(pieces, tops):
        width = int(piece["mask"].shape[1])
        candidates = list(range(W - width + 1))
        rng.shuffle(candidates)
        if len(candidates) > 1:
            candidates = [x for x in candidates if x != piece["final_col"]] + [piece["final_col"]]
        init_col = int(candidates[0])
        moved = moved or (init_col != int(piece["final_col"]))
        positions.append((int(top), init_col))

    if not moved:
        return None
    return positions


def _build_scene(
    surface: np.ndarray, pieces: list[dict], colors: np.ndarray, floating_positions: list[tuple[int, int]], H: int
) -> tuple[np.ndarray, np.ndarray]:
    inp = _terrain_from_surface(surface, H)
    out = inp.copy()
    for piece, color in zip(pieces, colors):
        piece["color"] = int(color)
    for idx, (piece, color) in enumerate(zip(pieces, colors)):
        top, left = floating_positions[idx]
        _paint_piece(inp, piece["mask"], int(color), int(top), int(left))
        _paint_piece(out, piece["mask"], int(color), int(piece["final_row"]), int(piece["final_col"]))
    return inp, out


def _wrong_order_differs(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        wrong, _ = _solve(inp, ascending=True)
    except AssertionError:
        return True
    return not np.array_equal(wrong, out)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*GEN_HEIGHT_RANGE))
    W = int(rng.integers(*GEN_WIDTH_RANGE))
    planned = _plan_scene(rng, H, W)
    if planned is None:
        return None
    surface, pieces = planned
    floating_positions = _assign_floating_positions(rng, surface, pieces)
    if floating_positions is None:
        return None

    colors = rng.choice(np.array(PIECE_COLORS, dtype=int), size=len(pieces), replace=False)
    inp, expected = _build_scene(surface, pieces, colors, floating_positions, H)

    try:
        solved, trace = _solve(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    for piece, (color, row, col) in zip(pieces, trace):
        if int(color) != int(piece["color"]):
            return None
        if int(row) != int(piece["final_row"]) or int(col) != int(piece["final_col"]):
            return None

    if len(pieces) >= 3 and any(piece["uses_modified"] for piece in pieces[1:]):
        if not _wrong_order_differs(inp, expected):
            return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-profile-complement input")
