import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    dirs = DIRS8 if connectivity == 8 else DIRS4
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _dominant_non_bg(grid: np.ndarray, coords: np.ndarray, bg: int) -> int:
    colors = np.array([int(grid[r, c]) for r, c in coords if int(grid[r, c]) != bg], dtype=int)
    values, counts = np.unique(colors, return_counts=True)
    return int(values[np.argmax(counts)])


def _aligned_patch(grid: np.ndarray, coords: np.ndarray, bg: int, shape: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full(shape, bg, dtype=int)
    raw = grid[r0 : r1 + 1, c0 : c1 + 1]
    patch[: raw.shape[0], : raw.shape[1]] = raw
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the pixelwise differences between the two horizontally separated objects
    as the output code. Difference cells become yellow, and each difference also
    places a red or blue marker one cell up-left depending on whether the left
    object's pixel is its dominant color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = _find_components(g != bg, connectivity=8)
    assert len(components) == 2, "expected exactly two foreground objects"

    parts = []
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        parts.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )
    parts.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    left, right = parts

    out_h = max(int(left["height"]), int(right["height"]))
    out_w = max(int(left["width"]), int(right["width"]))
    shape = (out_h, out_w)

    left_patch = _aligned_patch(g, left["coords"], bg, shape)
    right_patch = _aligned_patch(g, right["coords"], bg, shape)
    diff_positions = np.argwhere(left_patch != right_patch)
    assert len(diff_positions) > 0, "expected at least one pixelwise difference"

    dominant = _dominant_non_bg(g, left["coords"], bg)
    out = np.full(shape, bg, dtype=int)

    for r, c in diff_positions:
        if r > 0 and c > 0:
            out[r - 1, c - 1] = RED if int(left_patch[r, c]) == dominant else BLUE

    for r, c in diff_positions:
        out[r, c] = YELLOW

    return out


_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400
_GEN_PALETTE = np.array([3, 5, 6, 7, 8, 9], dtype=int)
_GEN_H_RANGE = (4, 8)
_GEN_W_RANGE = (4, 8)
_GEN_ROW_SHIFT = (-2, 2)
_GEN_MARGIN_RANGE = (1, 4)
_GEN_NOTCH_ATTEMPTS = 40


def _is_connected(mask: np.ndarray) -> bool:
    return len(_find_components(mask, connectivity=4)) == 1


def _sample_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(_GEN_NOTCH_ATTEMPTS):
        mask = np.ones((h, w), dtype=bool)
        notch_count = int(_RNG.integers(0, 4))
        for _ in range(notch_count):
            side = str(_RNG.choice(["top", "bottom", "left", "right"]))
            if side in ("top", "bottom"):
                max_len = min(3, w - 2)
                if max_len < 1:
                    continue
                length = int(_RNG.integers(1, max_len + 1))
                start_lo = 1
                start_hi = w - length - 1
                if start_hi < start_lo:
                    continue
                start = int(_RNG.integers(start_lo, start_hi + 1))
                depth = int(_RNG.integers(1, min(2, h - 1) + 1))
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                mask[rows, start : start + length] = False
            else:
                max_len = min(3, h - 2)
                if max_len < 1:
                    continue
                length = int(_RNG.integers(1, max_len + 1))
                start_lo = 1
                start_hi = h - length - 1
                if start_hi < start_lo:
                    continue
                start = int(_RNG.integers(start_lo, start_hi + 1))
                depth = int(_RNG.integers(1, min(2, w - 1) + 1))
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                mask[start : start + length, cols] = False

        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.count_nonzero(mask[1:, 1:]) < 6:
            continue
        if not _is_connected(mask):
            continue
        return mask
    return None


def _coords_to_tuples(coords: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in coords]


def _well_spaced(pos: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(max(abs(pos[0] - r), abs(pos[1] - c)) > 1 for r, c in chosen)


def _pick_spaced(
    candidates: list[tuple[int, int]],
    count: int,
    blocked: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    blocked = [] if blocked is None else list(blocked)
    if count == 0:
        return []
    if len(candidates) < count:
        return None

    order = list(candidates)
    _RNG.shuffle(order)
    chosen: list[tuple[int, int]] = []
    for pos in order:
        if _well_spaced(pos, blocked) and _well_spaced(pos, chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return chosen
    return None


def _color_patch(mask: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]]] | None:
    h, w = mask.shape
    patch = np.full((h, w), BG, dtype=int)
    colors = list(map(int, _RNG.choice(_GEN_PALETTE, size=3, replace=False)))
    main_color = colors[0]
    accent_colors = colors[1:]
    patch[mask] = main_color

    safe_cells = [pos for pos in _coords_to_tuples(np.argwhere(mask)) if pos[0] > 0 and pos[1] > 0]
    accent_count = int(_RNG.integers(2, min(4, len(safe_cells)) + 1))
    accent_positions = _pick_spaced(safe_cells, accent_count)
    if accent_positions is None:
        return None

    for pos in accent_positions:
        patch[pos] = int(_RNG.choice(accent_colors))
    values, counts = np.unique(patch[mask], return_counts=True)
    if int(values[np.argmax(counts)]) != main_color:
        return None
    if np.count_nonzero(counts == counts.max()) != 1:
        return None
    return patch, main_color, accent_positions


def _recolor_choices(left_color: int, main_color: int, used_colors: list[int]) -> list[int]:
    if left_color == main_color:
        return [color for color in used_colors if color != left_color]
    choices = [color for color in used_colors if color != left_color]
    if main_color not in choices:
        choices.append(main_color)
    return choices


def _make_pair(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], int] | None:
    colored = _color_patch(mask)
    if colored is None:
        return None
    left_patch, main_color, accent_positions = colored
    right_patch = left_patch.copy()

    all_safe = [pos for pos in _coords_to_tuples(np.argwhere(mask)) if pos[0] > 0 and pos[1] > 0]
    main_candidates = [pos for pos in all_safe if int(left_patch[pos]) == main_color]
    blue_positions = _pick_spaced(accent_positions, 1)
    if blue_positions is None:
        return None
    red_positions = _pick_spaced(main_candidates, 1, blocked=blue_positions)
    if red_positions is None:
        return None

    diff_positions = blue_positions + red_positions
    extra_count = int(_RNG.integers(0, 3))
    for _ in range(extra_count):
        prefer_main = bool(_RNG.integers(0, 2))
        candidates = main_candidates if prefer_main else accent_positions
        available = [pos for pos in candidates if pos not in diff_positions]
        extra = _pick_spaced(available, 1, blocked=diff_positions)
        if extra is not None:
            diff_positions.extend(extra)

    used_colors = sorted({int(color) for color in np.unique(left_patch) if int(color) != BG})
    for pos in diff_positions:
        choices = _recolor_choices(int(left_patch[pos]), main_color, used_colors)
        if not choices:
            return None
        right_patch[pos] = int(_RNG.choice(choices))

    if len({int(color) for color in np.unique(right_patch) if int(color) != BG}) < 2:
        return None
    return left_patch, right_patch, diff_positions, main_color


def _expected_output(
    left_patch: np.ndarray,
    diff_positions: list[tuple[int, int]],
    main_color: int,
) -> np.ndarray:
    out = np.full(left_patch.shape, BG, dtype=int)
    for r, c in diff_positions:
        out[r - 1, c - 1] = RED if int(left_patch[r, c]) == main_color else BLUE
    for r, c in diff_positions:
        out[r, c] = YELLOW
    return out


def _place_scene(left_patch: np.ndarray, right_patch: np.ndarray) -> np.ndarray:
    h, w = left_patch.shape
    row_shift = int(_RNG.integers(_GEN_ROW_SHIFT[0], _GEN_ROW_SHIFT[1] + 1))
    top_margin = int(_RNG.integers(_GEN_MARGIN_RANGE[0], _GEN_MARGIN_RANGE[1] + 1))
    bottom_margin = int(_RNG.integers(_GEN_MARGIN_RANGE[0], _GEN_MARGIN_RANGE[1] + 1))
    left_margin = int(_RNG.integers(_GEN_MARGIN_RANGE[0], _GEN_MARGIN_RANGE[1] + 1))
    right_margin = int(_RNG.integers(_GEN_MARGIN_RANGE[0], _GEN_MARGIN_RANGE[1] + 1))
    gap = int(_RNG.integers(w + 1, w + 5))

    if row_shift >= 0:
        left_r = top_margin
        right_r = top_margin + row_shift
    else:
        left_r = top_margin - row_shift
        right_r = top_margin

    left_c = left_margin
    right_c = left_c + w + gap
    grid_h = max(left_r + h, right_r + h) + bottom_margin
    grid_w = right_c + w + right_margin

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[left_r : left_r + h, left_c : left_c + w] = left_patch
    grid[right_r : right_r + h, right_c : right_c + w] = right_patch
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(_GEN_H_RANGE[0], _GEN_H_RANGE[1] + 1))
    w = int(_RNG.integers(_GEN_W_RANGE[0], _GEN_W_RANGE[1] + 1))
    mask = _sample_mask(h, w)
    if mask is None:
        return None

    pair = _make_pair(mask)
    if pair is None:
        return None
    left_patch, right_patch, diff_positions, main_color = pair
    grid = _place_scene(left_patch, right_patch)

    if len(_find_components(grid != BG, connectivity=8)) != 2:
        return None

    expected = _expected_output(left_patch, diff_positions, main_color)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if not np.any(expected == RED) or not np.any(expected == BLUE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
