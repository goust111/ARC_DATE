import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MARKER_COLOR = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    dirs = DIR4 if connectivity == 4 else DIR8
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _mono_components(
    grid: np.ndarray,
    colors: set[int] | None = None,
    connectivity: int = 4,
    bg: int = 0,
) -> list[dict]:
    h, w = grid.shape
    dirs = DIR4 if connectivity == 4 else DIR8
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    allowed = None if colors is None else set(colors)
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg or (allowed is not None and color not in allowed):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": _bbox_from_coords(arr),
                }
            )
    return comps


def _largest_component_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    comps = _components_from_mask(grid != bg, connectivity=8)
    assert comps, "expected a foreground board component"
    largest = max(comps, key=len)
    return _bbox_from_coords(largest)


def _find_marker_color(board: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in sorted(int(x) for x in np.unique(board) if int(x) != bg):
        coords = np.argwhere(board == color)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1):
            candidates.append(color)
    assert len(candidates) == 1, f"expected one solid-rectangle marker, found {candidates}"
    return int(candidates[0])


def _recover_board(board: np.ndarray, marker_color: int) -> np.ndarray:
    out = board.copy()
    h, w = board.shape
    for r, c in np.argwhere(board == marker_color):
        rr, cc = h - 1 - int(r), w - 1 - int(c)
        vals = []
        for nr, nc in ((int(r), cc), (rr, int(c)), (rr, cc)):
            value = int(board[nr, nc])
            if value != marker_color:
                vals.append(value)
        uniq = set(vals)
        assert uniq, f"marker cell {(int(r), int(c))} has no visible mirror source"
        assert len(uniq) == 1, f"ambiguous mirror values for {(int(r), int(c))}: {uniq}"
        out[int(r), int(c)] = uniq.pop()
    return out


def _cue_sizes(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    board_colors: list[int],
    bg: int,
) -> dict[int, int]:
    outside = grid.copy()
    r0, c0, r1, c1 = board_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    sizes: dict[int, int] = {}
    for color in board_colors:
        comps = _mono_components(outside, colors={color}, connectivity=4, bg=bg)
        assert len(comps) <= 1, f"expected at most one cue for color {color}"
        if comps:
            sizes[int(color)] = int(comps[0]["area"])
    return sizes


def _rotate_layer(crop: np.ndarray, color: int, rot_k: int) -> np.ndarray:
    layer = np.where(crop == color, color, 0)
    rot_k %= 4
    if rot_k:
        layer = np.rot90(layer, -rot_k)
    return layer


def _compose_output(crop: np.ndarray, cue_sizes: dict[int, int], bg: int) -> np.ndarray:
    colors = [int(c) for c in np.unique(crop) if int(c) != bg]
    out = np.full(crop.shape, bg, dtype=int)
    for color in colors:
        if color not in cue_sizes:
            out[crop == color] = color
    for color in sorted(cue_sizes):
        if color not in colors:
            continue
        rot_k = int(cue_sizes[color]) % 4
        rotated = _rotate_layer(crop, color, rot_k)
        out[rotated == color] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the occluded symmetric board and then rotate same-color layers using outside cue sprites.

    Input structure:
    - Black background with one central multi-color board as the largest foreground component.
    - Inside that board, one unique color forms a solid filled rectangle that occludes part of the board.
    - Outside the board bbox, there are 1-2 small monochrome cue sprites whose colors also appear in the board.

    Rule:
    - Treat the unique solid rectangle as a marker, not as part of the true board pattern.
    - Recover every marker cell from the board's horizontal and vertical reflection symmetry; if both direct mirrors are still inside the marker, use the diagonal mirror.
    - Crop the restored board to its non-background bbox.
    - For each outside cue sprite, rotate the matching same-color layer in the crop clockwise by (sprite_area mod 4) quarter turns; colors without a cue stay fixed.
    - Recompose the layers on black, with rotated layers overwriting unchanged ones where they overlap.

    Key insight:
    The marker must be removed by bilateral symmetry before the cue-controlled layer rotations are applied, because hidden pixels can belong to the very layers that need rotating.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    board_bbox = _largest_component_bbox(g, bg)
    board = _crop(g, board_bbox).copy()

    marker_color = _find_marker_color(board, bg)
    restored = _recover_board(board, marker_color)

    crop_bbox = _bbox_from_mask(restored != bg)
    assert crop_bbox is not None, "expected a non-empty restored board"
    crop = _crop(restored, crop_bbox)

    board_colors = [int(c) for c in np.unique(crop) if int(c) != bg]
    cue_sizes = _cue_sizes(g, board_bbox, board_colors, bg)
    return _compose_output(crop, cue_sizes, bg)


def _orbit_cells(n: int, r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r, n - 1 - c), (n - 1 - r, c), (n - 1 - r, n - 1 - c)]


def _seed_set_breaks_90_symmetry(seeds: list[tuple[int, int]]) -> bool:
    seed_set = set(seeds)
    return any(r != c and (c, r) not in seed_set for r, c in seed_set)


def _paint_board(
    n: int,
    colors: list[int],
    layer_seeds: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    board = np.full((n, n), colors[0], dtype=int)
    for color in colors[1:]:
        for r, c in layer_seeds[color]:
            for rr, cc in _orbit_cells(n, r, c):
                board[rr, cc] = color
    return board


def _all_non_marker_rectangles(board: np.ndarray, bg: int) -> list[int]:
    bad: list[int] = []
    for color in sorted(int(x) for x in np.unique(board) if int(x) not in {bg, MARKER_COLOR}):
        coords = np.argwhere(board == color)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1):
            bad.append(color)
    return bad


def _solid_rectangle_colors(board: np.ndarray, bg: int) -> list[int]:
    out: list[int] = []
    for color in sorted(int(x) for x in np.unique(board) if int(x) != bg):
        coords = np.argwhere(board == color)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1):
            out.append(color)
    return out


def _cue_rotations(cue_colors: list[int]) -> dict[int, int]:
    return {color: random.choice([1, 3]) for color in cue_colors}


def _cue_shape(size: int) -> np.ndarray:
    if size == 3:
        shapes = [
            np.array([[1, 1, 1]], dtype=int),
            np.array([[1, 0], [1, 1]], dtype=int),
        ]
    else:
        shapes = [
            np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
            np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
            np.array([[1, 1], [1, 1], [1, 0]], dtype=int),
        ]
    shape = random.choice(shapes)
    k = random.randrange(4)
    return np.rot90(shape, k)


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(row_gap, col_gap)


def _place_shape(
    grid: np.ndarray,
    shape: np.ndarray,
    color: int,
    forbidden_bboxes: list[tuple[int, int, int, int]],
    min_dist: int,
) -> bool:
    h, w = grid.shape
    sh, sw = shape.shape
    positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if np.any((shape == 1) & (grid[r0 : r0 + sh, c0 : c0 + sw] != 0)):
            continue
        bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        if any(_bbox_distance(bbox, other) < min_dist for other in forbidden_bboxes):
            continue
        grid[r0 : r0 + sh, c0 : c0 + sw][shape == 1] = color
        forbidden_bboxes.append(bbox)
        return True
    return False


def _marker_sources_ok(n: int, r0: int, c0: int, h: int, w: int) -> bool:
    def inside(rr: int, cc: int) -> bool:
        return r0 <= rr < r0 + h and c0 <= cc < c0 + w

    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            rr, cc = n - 1 - r, n - 1 - c
            if inside(r, cc) and inside(rr, c) and inside(rr, cc):
                return False
    return True


def _marker_needs_diagonal(n: int, r0: int, c0: int, h: int, w: int) -> bool:
    def inside(rr: int, cc: int) -> bool:
        return r0 <= rr < r0 + h and c0 <= cc < c0 + w

    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            rr, cc = n - 1 - r, n - 1 - c
            if inside(r, cc) and inside(rr, c) and not inside(rr, cc):
                return True
    return False


def _marker_candidates(
    board: np.ndarray,
    cue_colors: list[int],
    require_diagonal: bool,
) -> list[tuple[int, int, int, int]]:
    n = board.shape[0]
    colors = [int(c) for c in np.unique(board)]
    counts = Counter(map(int, board.flat))
    candidates: list[tuple[int, int, int, int]] = []
    for h, w in ((3, 4), (4, 3)):
        for r0 in range(1, n - h):
            for c0 in range(1, n - w):
                if not _marker_sources_ok(n, r0, c0, h, w):
                    continue
                if _marker_needs_diagonal(n, r0, c0, h, w) != require_diagonal:
                    continue
                patch = board[r0 : r0 + h, c0 : c0 + w]
                if not all(np.any(patch == color) for color in cue_colors):
                    continue
                if not all(counts[color] > int(np.count_nonzero(patch == color)) for color in colors):
                    continue
                candidates.append((r0, c0, h, w))
    return candidates


def _visible_shortcut_output(board_with_marker: np.ndarray, cue_sizes: dict[int, int]) -> np.ndarray:
    visible = board_with_marker.copy()
    visible[visible == MARKER_COLOR] = 0
    bbox = _bbox_from_mask(visible != 0)
    assert bbox is not None
    crop = _crop(visible, bbox)
    return _compose_output(crop, cue_sizes, 0)


def _layer_mask(board: np.ndarray, color: int) -> np.ndarray:
    return board == color


def _sample_layer_seeds(n: int, colors: list[int], cue_colors: list[int]) -> dict[int, list[tuple[int, int]]] | None:
    q = n // 2
    all_seeds = [(r, c) for r in range(q) for c in range(q)]
    central = [
        (r, c)
        for r, c in all_seeds
        if r != c and r >= max(0, q // 2 - 1) and c >= max(0, q // 2 - 1)
    ]
    if len(central) < len(colors) - 1:
        return None

    used: set[tuple[int, int]] = set()
    layer_seeds: dict[int, list[tuple[int, int]]] = {color: [] for color in colors[1:]}
    random.shuffle(central)
    for color in cue_colors:
        seed = central.pop()
        used.add(seed)
        layer_seeds[color].append(seed)

    others = [color for color in colors[1:] if color not in cue_colors]
    pool = [s for s in all_seeds if s not in used]
    random.shuffle(pool)
    for color in others:
        if not pool:
            return None
        seed = pool.pop()
        used.add(seed)
        layer_seeds[color].append(seed)

    for color in colors[1:]:
        extra_count = random.randint(1, 2)
        for _ in range(extra_count):
            choices = [s for s in all_seeds if s not in used]
            if not choices:
                break
            seed = random.choice(choices)
            used.add(seed)
            layer_seeds[color].append(seed)

    for color in cue_colors:
        if not _seed_set_breaks_90_symmetry(layer_seeds[color]):
            choices = [s for s in all_seeds if s not in used and s[::-1] not in layer_seeds[color]]
            if not choices:
                return None
            seed = random.choice(choices)
            used.add(seed)
            layer_seeds[color].append(seed)
        if not _seed_set_breaks_90_symmetry(layer_seeds[color]):
            return None
    return layer_seeds


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.choice([8, 10, 12])
    num_colors = random.choice([2, 3])
    colors = random.sample(PALETTE, num_colors)
    cue_count = 1 if num_colors == 2 else random.choice([1, 2])
    cue_colors = random.sample(colors[1:], cue_count)
    cue_rot = _cue_rotations(cue_colors)
    cue_sizes = {color: (5 if rot == 1 else 3) for color, rot in cue_rot.items()}

    layer_seeds = _sample_layer_seeds(n, colors, cue_colors)
    if layer_seeds is None:
        return None
    board = _paint_board(n, colors, layer_seeds)
    if _all_non_marker_rectangles(board, bg=0):
        return None
    if any(np.array_equal(_layer_mask(board, color), np.rot90(_layer_mask(board, color), 1)) for color in cue_colors):
        return None

    require_diagonal = bool(random.getrandbits(1))
    marker_options = _marker_candidates(board, cue_colors, require_diagonal)
    if not marker_options:
        return None
    mr, mc, mh, mw = random.choice(marker_options)

    board_with_marker = board.copy()
    board_with_marker[mr : mr + mh, mc : mc + mw] = MARKER_COLOR
    if _solid_rectangle_colors(board_with_marker, 0) != [MARKER_COLOR]:
        return None

    h = n + random.randint(6, 10)
    w = n + random.randint(6, 10)
    if h > 30 or w > 30:
        return None
    grid = np.zeros((h, w), dtype=int)
    board_r0 = random.randint(2, h - n - 2)
    board_c0 = random.randint(2, w - n - 2)
    grid[board_r0 : board_r0 + n, board_c0 : board_c0 + n] = board_with_marker
    board_bbox = (board_r0, board_c0, board_r0 + n - 1, board_c0 + n - 1)

    occupied_bboxes = [board_bbox]
    for color in cue_colors:
        shape = _cue_shape(cue_sizes[color])
        if not _place_shape(grid, shape, color, occupied_bboxes, min_dist=2):
            return None

    expected = _compose_output(board, cue_sizes, 0)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(expected, board):
        return None
    if np.array_equal(expected, _crop(board, _bbox_from_mask(board != 0))):
        return None
    if np.array_equal(expected, _visible_shortcut_output(board_with_marker, cue_sizes)):
        return None

    largest_bbox = _largest_component_bbox(grid, 0)
    if largest_bbox != board_bbox:
        return None

    outside = grid.copy()
    r0, c0, r1, c1 = board_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = 0
    for color in cue_colors:
        comps = _mono_components(outside, colors={color}, connectivity=4, bg=0)
        if len(comps) != 1 or comps[0]["area"] != cue_sizes[color]:
            return None

    for color in cue_colors:
        masked = board[mr : mr + mh, mc : mc + mw]
        if not np.any(masked == color):
            return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _generate_case()
        if result is not None:
            grid, _ = result
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
