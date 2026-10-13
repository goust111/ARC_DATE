import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 300
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_funnels(grid: np.ndarray) -> list[dict[str, int]]:
    funnels: list[dict[str, int]] = []
    for comp in _components_of_color(grid, WALL):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert width == 5, "each funnel must have outer width 5"
        assert height >= 2, "each funnel must have positive depth"
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]

        wall_mask = np.zeros_like(sub, dtype=bool)
        wall_mask[:-1, 0] = True
        wall_mask[:-1, -1] = True
        wall_mask[-1, :] = True
        assert np.array_equal(sub == WALL, wall_mask), "grey component must be a U-shaped container"
        assert np.all(sub[:-1, 1:4] == BG), "funnel interior must be empty in the input"

        marker_r = r0 - 1
        marker_c = c0 + 2
        assert marker_r >= 0, "marker must sit above the funnel opening"
        color = int(grid[marker_r, marker_c])
        assert color not in (BG, WALL), "marker must be a non-grey color"

        funnels.append(
            {
                "color": color,
                "top_r": r0,
                "bottom_r": r1,
                "left_c": c0,
                "marker_r": marker_r,
                "marker_c": marker_c,
            }
        )

    assert funnels, "expected at least one funnel"
    colors = [f["color"] for f in funnels]
    assert len(colors) == len(set(colors)), "funnel marker colors must be unique"
    funnels.sort(key=lambda f: (f["left_c"], f["top_r"]))
    return funnels


def _extract_pieces(grid: np.ndarray, funnels: list[dict[str, int]]) -> dict[int, list[dict[str, int]]]:
    working = grid.copy()
    for funnel in funnels:
        working[funnel["marker_r"], funnel["marker_c"]] = BG

    pieces_by_color = {f["color"]: [] for f in funnels}
    for color in np.unique(working):
        color = int(color)
        if color in (BG, WALL):
            continue
        assert color in pieces_by_color, "every colored fragment must match some funnel color"
        for comp in _components_of_color(working, color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            sub = working[r0 : r1 + 1, c0 : c1 + 1]
            assert width == 3, "each fragment must be three pixels wide"
            assert height in (1, 2), "each fragment must be one or two pixels tall"
            assert len(comp) == height * width and np.all(sub == color), "fragment must be a solid rectangle"
            pieces_by_color[color].append(
                {
                    "color": color,
                    "top_r": r0,
                    "bottom_r": r1,
                    "left_c": c0,
                    "height": height,
                }
            )
    return pieces_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the funnel marked with the same color.

    Fragments are sorted by their original vertical distance to the matching
    funnel opening: the closest fragment ends up at the bottom of the stack.
    Funnel markers are cleared in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    funnels = _find_funnels(g)
    pieces_by_color = _extract_pieces(g, funnels)

    out = g.copy()
    for funnel in funnels:
        out[funnel["marker_r"], funnel["marker_c"]] = BG
    for pieces in pieces_by_color.values():
        for piece in pieces:
            out[piece["top_r"] : piece["bottom_r"] + 1, piece["left_c"] : piece["left_c"] + 3] = BG

    funnel_by_color = {f["color"]: f for f in funnels}
    for color, pieces in pieces_by_color.items():
        funnel = funnel_by_color[color]
        distances = [funnel["top_r"] - piece["bottom_r"] for piece in pieces]
        assert all(distance > 0 for distance in distances), "every fragment must start above its funnel"
        assert len(distances) == len(set(distances)), "distances must be unique within a color group"

        ordered = sorted(
            pieces,
            key=lambda piece: (funnel["top_r"] - piece["bottom_r"], piece["top_r"], piece["left_c"]),
        )
        cursor = funnel["bottom_r"] - 1
        fill_c0 = funnel["left_c"] + 1
        for piece in ordered:
            dest_r0 = cursor - piece["height"] + 1
            assert dest_r0 >= funnel["top_r"], "funnel must be deep enough to hold all assigned fragments"
            out[dest_r0 : cursor + 1, fill_c0 : fill_c0 + 3] = color
            cursor = dest_r0 - 1

    return out


def _can_place_piece(occupied: np.ndarray, top: int, left: int, height: int) -> bool:
    h, w = map(int, occupied.shape)
    if top < 0 or left < 0 or top + height > h or left + 3 > w:
        return False
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + 4)
    return not occupied[r0:r1, c0:c1].any()


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        marker_r = int(spec["marker_r"])
        marker_c = int(spec["marker_c"])
        out[marker_r, marker_c] = BG
        for piece in spec["pieces"]:
            top = int(piece["top"])
            height = int(piece["height"])
            left = int(piece["left"])
            out[top : top + height, left : left + 3] = BG

        cursor = int(spec["bottom_r"]) - 1
        fill_c0 = int(spec["left_c"]) + 1
        ordered = sorted(spec["pieces"], key=lambda piece: int(spec["top_r"]) - (int(piece["top"]) + int(piece["height"]) - 1))
        for piece in ordered:
            height = int(piece["height"])
            dest_r0 = cursor - height + 1
            out[dest_r0 : cursor + 1, fill_c0 : fill_c0 + 3] = int(spec["color"])
            cursor = dest_r0 - 1
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    n_funnels = int(rng.integers(2, 5))
    colors = list(map(int, rng.choice(PALETTE, size=n_funnels, replace=False)))

    piece_counts = [int(rng.integers(1, 4)) for _ in range(n_funnels)]
    if max(piece_counts) == 1:
        piece_counts[int(rng.integers(0, n_funnels))] = int(rng.integers(2, 4))

    piece_heights: list[list[int]] = []
    depths: list[int] = []
    for count in piece_counts:
        heights = [1 + int(rng.random() < 0.45) for _ in range(count)]
        piece_heights.append(heights)
        slack = int(rng.integers(0, 4))
        depths.append(sum(heights) + slack)

    left_margin = int(rng.integers(1, 3))
    right_margin = int(rng.integers(1, 3))
    gaps = [int(rng.integers(2, 4)) for _ in range(n_funnels - 1)]
    funnel_cols = [left_margin]
    for gap in gaps:
        funnel_cols.append(funnel_cols[-1] + 5 + gap)
    width = funnel_cols[-1] + 5 + right_margin

    piece_zone_h = int(rng.integers(8, 15))
    vertical_gap = int(rng.integers(2, 5))
    max_depth = max(depths)
    bottom_r = piece_zone_h + vertical_gap + max_depth
    bottom_margin = int(rng.integers(0, 2))
    height = bottom_r + bottom_margin + 1

    grid = np.full((height, width), BG, dtype=int)
    specs: list[dict[str, object]] = []

    for color, c0, depth, heights in zip(colors, funnel_cols, depths, piece_heights):
        top_r = bottom_r - depth
        marker_r = top_r - 1
        if marker_r < 0:
            return None
        grid[top_r:bottom_r, c0] = WALL
        grid[top_r:bottom_r, c0 + 4] = WALL
        grid[bottom_r, c0 : c0 + 5] = WALL
        grid[marker_r, c0 + 2] = color
        specs.append(
            {
                "color": color,
                "left_c": c0,
                "top_r": top_r,
                "bottom_r": bottom_r,
                "marker_r": marker_r,
                "marker_c": c0 + 2,
                "pieces": [],
                "heights": heights,
            }
        )

    occupied = grid != BG
    for spec in specs:
        heights = list(spec["heights"])
        if len(heights) > piece_zone_h - 1:
            return None
        bottom_rows = list(map(int, rng.choice(np.arange(1, piece_zone_h), size=len(heights), replace=False)))
        rng.shuffle(bottom_rows)
        for piece_height, piece_bottom in zip(heights, bottom_rows):
            top = piece_bottom - piece_height + 1
            placed = False
            for _ in range(200):
                left = int(rng.integers(0, width - 2))
                if not _can_place_piece(occupied, top, left, piece_height):
                    continue
                grid[top : top + piece_height, left : left + 3] = int(spec["color"])
                occupied[top : top + piece_height, left : left + 3] = True
                spec["pieces"].append({"top": top, "left": left, "height": piece_height})
                placed = True
                break
            if not placed:
                return None

    # Reject overly uniform layouts so the color-matching and stacking remain visible.
    if sum(len(spec["pieces"]) for spec in specs) < n_funnels + 1:
        return None

    expected = _expected_output(grid, specs)
    try:
        observed = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid funnel-stacking input")
