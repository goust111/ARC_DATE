"""ARC task 6: fit scaled, oriented grey keys into color-marked holes."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
_RETRIES = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    """Make the tight boolean sprite for a coordinate component."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    sprite = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return sprite


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    """Return distinct rotations and reflections, each in its tight bbox."""
    variants = []
    seen = set()
    for reflected in (False, True):
        seed = np.fliplr(mask) if reflected else mask
        for turns in range(4):
            candidate = np.rot90(seed, turns)
            rows, cols = np.where(candidate)
            candidate = candidate[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
            key = (candidate.shape, candidate.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
    return variants


def _scaled_match(piece: np.ndarray, hole: np.ndarray) -> list[tuple[int, int]]:
    """List the distinct (orientation index, integer scale) exact fits."""
    matches = []
    for index, oriented in enumerate(_orientations(piece)):
        oh, ow = oriented.shape
        hh, hw = hole.shape
        if hh % oh or hw % ow:
            continue
        scale_r, scale_c = hh // oh, hw // ow
        if scale_r != scale_c or scale_r < 2:
            continue
        enlarged = np.kron(oriented, np.ones((scale_r, scale_r), dtype=bool))
        if np.array_equal(enlarged, hole):
            matches.append((index, scale_r))
    return matches


def _parse(grid: np.ndarray):
    """Extract the rectangular board, its holes, and outside grey pieces."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    board_candidates = []
    for color in set(map(int, g.ravel())) - {BLACK, GREY}:
        for coords in _components(g == color):
            board_candidates.append((len(coords), color, coords))
    assert board_candidates, "expected a colored board"
    board_candidates.sort(key=lambda item: item[0], reverse=True)
    assert len(board_candidates) == 1 or board_candidates[0][0] > board_candidates[1][0]
    _area, board_color, board_coords = board_candidates[0]
    br0, bc0 = board_coords.min(axis=0)
    br1, bc1 = board_coords.max(axis=0)
    br0, bc0, br1, bc1 = map(int, (br0, bc0, br1, bc1))
    board_region = g[br0 : br1 + 1, bc0 : bc1 + 1]
    assert np.all(board_region[0, :] == board_color)
    assert np.all(board_region[-1, :] == board_color)
    assert np.all(board_region[:, 0] == board_color)
    assert np.all(board_region[:, -1] == board_color)

    holes = []
    for local_coords in _components(board_region != board_color):
        global_coords = local_coords + np.array([br0, bc0])
        values = g[global_coords[:, 0], global_coords[:, 1]]
        markers = [(int(r), int(c), int(g[r, c])) for r, c in global_coords if int(g[r, c]) != BLACK]
        assert len(markers) == 1, "each hole must contain exactly one color dot"
        assert markers[0][2] not in {BLACK, GREY, board_color}
        holes.append({"coords": global_coords, "mask": _crop_mask(global_coords), "color": markers[0][2]})
    assert len(holes) >= 2, "expected several holes"

    pieces = []
    for coords in _components(g == GREY):
        inside = ((coords[:, 0] >= br0) & (coords[:, 0] <= br1)
                  & (coords[:, 1] >= bc0) & (coords[:, 1] <= bc1))
        assert not np.any(inside), "grey pieces must lie outside the board"
        pieces.append({"coords": coords, "mask": _crop_mask(coords)})
    assert len(pieces) == len(holes) + 1, "expected one extra grey piece"
    return holes, pieces


def _match_table(holes, pieces):
    table = []
    for piece in pieces:
        candidates = []
        for hole_index, hole in enumerate(holes):
            for orientation, scale in _scaled_match(piece["mask"], hole["mask"]):
                candidates.append((hole_index, orientation, scale))
        table.append(candidates)
    return table


def main(grid: np.ndarray) -> np.ndarray:
    """Fit enlarged and oriented grey keys into their uniquely matching color-marked holes.

    Input structure:
    - A black background contains one large rectangular monochrome board with several
      enclosed black holes, each containing exactly one non-grey colored control pixel.
    - Outside the board are grey 4-connected key pieces, exactly one more than the holes.

    Rule:
    - Compare each grey key's tight mask under all rotations and reflections, enlarged by
      one uniform integer pixel scale, with every whole hole region (black cells plus its dot).
    - Every usable key has exactly one fit. Erase it, fill that entire hole with the dot's
      color, and leave the sole key having no fit unchanged in its original position.

    Key insight:
    - Hole size, orientation, and silhouette jointly identify the key; the embedded dot is
      a recoloring instruction that is covered rather than a pixel to preserve separately.
    """
    g = np.asarray(grid, dtype=int)
    holes, pieces = _parse(g)
    matches = _match_table(holes, pieces)

    matched = [(piece_index, candidates[0]) for piece_index, candidates in enumerate(matches) if candidates]
    assert sum(not candidates for candidates in matches) == 1, "expected exactly one distractor"
    assert all(len(candidates) == 1 for candidates in matches if candidates), "every usable key must have one unique fit"
    assert len(matched) == len(holes)
    assert {candidate[0] for _piece_index, candidate in matched} == set(range(len(holes)))

    out = g.copy()
    for piece_index, (hole_index, _orientation, _scale) in matched:
        piece_coords = pieces[piece_index]["coords"]
        out[piece_coords[:, 0], piece_coords[:, 1]] = BLACK
        hole = holes[hole_index]
        hole_coords = hole["coords"]
        out[hole_coords[:, 0], hole_coords[:, 1]] = hole["color"]
    return out


def _random_key(max_side: int = 4) -> np.ndarray:
    """Grow an asymmetric 4-connected key with eight distinct orientations."""
    for _ in range(300):
        canvas = np.zeros((max_side, max_side), dtype=bool)
        start = (random.randrange(max_side), random.randrange(max_side))
        canvas[start] = True
        target_area = random.randint(5, min(8, max_side * max_side - 2))
        while int(canvas.sum()) < target_area:
            boundary = []
            for r, c in np.argwhere(canvas):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < max_side and 0 <= nc < max_side and not canvas[nr, nc]:
                        boundary.append((nr, nc))
            canvas[random.choice(boundary)] = True
        coords = np.argwhere(canvas)
        sprite = _crop_mask(coords)
        if min(sprite.shape) >= 2 and len(_orientations(sprite)) == 8:
            return sprite
    raise RuntimeError("failed to grow an asymmetric key")


def _free_bbox(occupied: np.ndarray, r: int, c: int, h: int, w: int, padding: int = 1) -> bool:
    r0, c0 = max(0, r - padding), max(0, c - padding)
    r1, c1 = min(occupied.shape[0], r + h + padding), min(occupied.shape[1], c + w + padding)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _place_holes(grid, board_box, targets, marker_colors):
    br0, bc0, br1, bc1 = board_box
    occupied = np.zeros(grid.shape, dtype=bool)
    records = []
    for target, marker_color in zip(targets, marker_colors):
        th, tw = target.shape
        choices = []
        for r in range(br0 + 2, br1 - th):
            for c in range(bc0 + 2, bc1 - tw):
                if _free_bbox(occupied, r, c, th, tw, padding=1):
                    choices.append((r, c))
        if not choices:
            return None
        r, c = random.choice(choices)
        cells = np.argwhere(target) + np.array([r, c])
        grid[cells[:, 0], cells[:, 1]] = BLACK
        mr, mc = map(int, random.choice(cells.tolist()))
        grid[mr, mc] = marker_color
        occupied[r : r + th, c : c + tw] = True
        records.append(cells)
    return records


def _place_pieces(grid, board_box, masks):
    br0, bc0, br1, bc1 = board_box
    height, width = grid.shape
    occupied = np.zeros(grid.shape, dtype=bool)
    occupied[br0 : br1 + 1, bc0 : bc1 + 1] = True
    for mask in masks:
        mh, mw = mask.shape
        choices = []
        for r in range(1, height - mh):
            for c in range(1, width - mw):
                if not (r + mh <= br0 - 1 or r >= br1 + 2 or c + mw <= bc0 - 1 or c >= bc1 + 2):
                    continue
                if _free_bbox(occupied, r, c, mh, mw, padding=1):
                    choices.append((r, c))
        if not choices:
            return False
        r, c = random.choice(choices)
        cells = np.argwhere(mask) + np.array([r, c])
        grid[cells[:, 0], cells[:, 1]] = GREY
        occupied[r : r + mh, c : c + mw] = True
    return True


def _generation_attempt() -> np.ndarray | None:
    height = random.randint(25, 30)
    width = random.randint(27, 30)
    n_holes = random.choice((2, 2, 3))
    board_height = height - random.randint(2, 4)
    board_width = random.randint(16, min(19, width - 8))
    br0 = random.randint(1, height - board_height - 1)
    if random.choice((False, True)):
        bc0 = random.randint(1, 2)
    else:
        bc0 = width - board_width - random.randint(1, 2)
    br1, bc1 = br0 + board_height - 1, bc0 + board_width - 1

    palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n_holes + 1)
    board_color, marker_colors = palette[0], palette[1:]
    grid = np.zeros((height, width), dtype=int)
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = board_color

    keys = []
    while len(keys) < n_holes:
        # Scale-3 and optional third keys stay compact enough to permit several
        # well-separated holes; the scale-2 key may exercise a 4-cell side.
        key = _random_key(4 if len(keys) == 0 else 3)
        if all(not any(np.array_equal(key, v) for v in _orientations(old)) for old in keys):
            keys.append(key)

    # Every example visibly requires both a proper rotation and a reflection.
    oriented = [np.rot90(keys[0], random.choice((1, 2, 3)))]
    oriented.append(np.rot90(np.fliplr(keys[1]), random.randrange(4)))
    for key in keys[2:]:
        oriented.append(random.choice(_orientations(key)[1:]))
    scales = [2, 3] + [2] * (n_holes - 2)
    targets = [np.kron(mask, np.ones((scale, scale), dtype=bool)) for mask, scale in zip(oriented, scales)]
    if _place_holes(grid, (br0, bc0, br1, bc1), targets, marker_colors) is None:
        return None

    distractor = _random_key(4)
    piece_masks = keys + [distractor]
    random.shuffle(piece_masks)
    if not _place_pieces(grid, (br0, bc0, br1, bc1), piece_masks):
        return None

    try:
        holes, pieces = _parse(grid)
        table = _match_table(holes, pieces)
        if sorted(map(len, table)) != [0] + [1] * n_holes:
            return None
        usable = [candidates[0] for candidates in table if candidates]
        if {item[0] for item in usable} != set(range(n_holes)):
            return None
        if {item[2] for item in usable} != {2, 3}:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
        # The one unmatched grey component must be preserved exactly.
        unmatched = next(i for i, candidates in enumerate(table) if not candidates)
        coords = pieces[unmatched]["coords"]
        if not np.all(out[coords[:, 0], coords[:, 1]] == GREY):
            return None
    except (AssertionError, RuntimeError, ValueError):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate instance satisfying all matching invariants."""
    for _ in range(_RETRIES):
        result = _generation_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid hole-key instance")
