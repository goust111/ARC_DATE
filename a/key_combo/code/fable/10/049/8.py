from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


Coord = tuple[int, int]
ORTHO_DIRS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
MARKER_COLORS = tuple(c for c in range(10) if c not in (Color.BLACK, Color.GREY, Color.TEAL))
MAX_GENERATION_ATTEMPTS = 500

BASE_GREY_SHAPES: tuple[tuple[Coord, ...], ...] = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
)


def _normalize(coords: set[Coord]) -> set[Coord]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}


def _transform_shape(coords: tuple[Coord, ...]) -> set[Coord]:
    arr = np.array(coords, dtype=int)
    for _ in range(random.randrange(4)):
        arr = np.column_stack((-arr[:, 1], arr[:, 0]))
    if random.random() < 0.5:
        arr[:, 1] *= -1
    if random.random() < 0.5:
        arr[:, 0] *= -1
    return _normalize({(int(r), int(c)) for r, c in arr})


def _shape_dims(coords: set[Coord]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _paint(grid: np.ndarray, coords: set[Coord], color: int, *, origin: Coord = (0, 0)) -> None:
    r0, c0 = origin
    for r, c in coords:
        grid[r0 + r, c0 + c] = int(color)


def _distance_map(shape: tuple[int, int], wall_mask: np.ndarray, sources: list[Coord]) -> np.ndarray:
    dist = np.full(shape, -1, dtype=int)
    q: deque[Coord] = deque()
    for r, c in sources:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        step = int(dist[r, c]) + 1
        for dr, dc in ORTHO_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < shape[0] and 0 <= nc < shape[1] and not wall_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = step
                q.append((nr, nc))
    return dist


def _collect_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for color in sorted(set(map(int, grid.ravel())) - {bg, Color.GREY, Color.TEAL}):
        comps = components_mask(grid == color, connectivity=4)
        assert len(comps) == 1 and int(comps[0].shape[0]) == 1, "markers must be isolated single pixels"
        assert color not in seen_colors
        seen_colors.add(color)
        r, c = map(int, comps[0][0])
        markers.append((r, c, int(color)))
    assert len(markers) >= 2, "expected multiple markers"
    return markers


def _winner_info(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    grey_mask = g == Color.GREY
    grey_components = components_mask(grey_mask, connectivity=4)
    assert len(grey_components) == 1, "expected exactly one grey object"

    wall_mask = g == Color.TEAL
    assert bool(wall_mask.any()), "expected teal wall cells"

    markers = _collect_markers(g, bg)
    grey_sources = [tuple(map(int, pt)) for pt in np.argwhere(grey_mask)]
    dist = _distance_map(g.shape, wall_mask, grey_sources)

    scores: list[tuple[int, int, int, int]] = []
    for r, c, color in markers:
        d = int(dist[r, c])
        assert d >= 0, "every marker must reach the grey object"
        scores.append((d, r, c, color))

    best = min(score[0] for score in scores)
    assert sum(score[0] == best for score in scores) == 1, "winner must be unique"
    return grey_mask, scores, wall_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey object with the color of the marker that reaches it in the fewest wall-avoiding steps.

    Input structure
    - A black background contains one 4-connected grey object, several teal wall segments, and several isolated one-pixel markers of distinct colors.
    - Markers are outside the grey object, and teal cells act as impassable walls for path measurement.

    Rule
    - For each marker, compute the 4-neighbor shortest path length from that marker to any grey-object cell, forbidding paths that pass through teal walls.
    - Select the unique marker with the smallest such geodesic distance.
    - Erase all markers, keep the teal walls, and repaint the entire grey object with the winning marker color.

    Key insight
    - Straight-line closeness is a decoy: walls redefine which marker is truly nearest, so the answer comes from obstacle-aware path distance rather than Euclidean distance.
    """
    g = np.asarray(grid, dtype=int)
    grey_mask, scores, wall_mask = _winner_info(g)
    winner_color = next(color for d, _r, _c, color in scores if d == min(score[0] for score in scores))

    out = np.full_like(g, Color.BLACK)
    out[wall_mask] = Color.TEAL
    out[grey_mask] = int(winner_color)
    return out


def _rect_outline_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _gate_position(side: str, fr0: int, fc0: int, fr1: int, fc1: int, anchor: int) -> Coord:
    if side == "top":
        return fr0, max(fc0 + 1, min(fc1 - 1, anchor))
    if side == "bottom":
        return fr1, max(fc0 + 1, min(fc1 - 1, anchor))
    if side == "left":
        return max(fr0 + 1, min(fr1 - 1, anchor)), fc0
    if side == "right":
        return max(fr0 + 1, min(fr1 - 1, anchor)), fc1
    raise ValueError(side)


def _adjacent_outside(side: str, gate: Coord) -> Coord:
    r, c = gate
    if side == "top":
        return r - 1, c
    if side == "bottom":
        return r + 1, c
    if side == "left":
        return r, c - 1
    if side == "right":
        return r, c + 1
    raise ValueError(side)


def _outside_side(fr0: int, fc0: int, fr1: int, fc1: int, cell: Coord) -> str | None:
    r, c = cell
    if r < fr0:
        return "top"
    if r > fr1:
        return "bottom"
    if c < fc0:
        return "left"
    if c > fc1:
        return "right"
    return None


def _sqeuclid_to_grey(cell: Coord, grey_cells: np.ndarray) -> int:
    rr, cc = cell
    diffs = grey_cells - np.array([[rr, cc]], dtype=int)
    return int(np.min(np.sum(diffs * diffs, axis=1)))


def _open_winner(scores: list[tuple[int, int, int, int]], grey_mask: np.ndarray) -> int | None:
    grey_cells = np.argwhere(grey_mask)
    euclid_scores = [(_sqeuclid_to_grey((r, c), grey_cells), color) for _d, r, c, color in scores]
    best = min(score[0] for score in euclid_scores)
    winners = [color for dist, color in euclid_scores if dist == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _validate_semantics(grid: np.ndarray, fence_bbox: tuple[int, int, int, int], primary_side: str) -> bool:
    try:
        grey_mask, scores, wall_mask = _winner_info(grid)
    except Exception:
        return False

    fr0, fc0, fr1, fc1 = fence_bbox
    grey_cells = np.argwhere(grey_mask)
    best_dist = min(score[0] for score in scores)
    winner = next(score for score in scores if score[0] == best_dist)
    winner_color = int(winner[3])
    winner_sq = _sqeuclid_to_grey((winner[1], winner[2]), grey_cells)

    if sum(_outside_side(fr0, fc0, fr1, fc1, (r, c)) == primary_side for _d, r, c, _color in scores) == 0:
        return False

    if not any(_sqeuclid_to_grey((r, c), grey_cells) < winner_sq and d > best_dist for d, r, c, _color in scores):
        return False

    no_wall_dist = _distance_map(grid.shape, np.zeros_like(wall_mask, dtype=bool), [tuple(map(int, pt)) for pt in grey_cells])
    open_scores = [(int(no_wall_dist[r, c]), color) for _d, r, c, color in scores]
    open_best = min(score[0] for score in open_scores)
    open_winners = [color for dist, color in open_scores if dist == open_best]
    if len(open_winners) != 1 or int(open_winners[0]) == winner_color:
        return False

    if _open_winner(scores, grey_mask) == winner_color:
        return False

    expected = np.full_like(grid, Color.BLACK)
    expected[wall_mask] = Color.TEAL
    expected[grey_mask] = winner_color
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _sample_extra_markers(
    wall_mask: np.ndarray,
    grey_mask: np.ndarray,
    base_markers: list[Coord],
    fence_bbox: tuple[int, int, int, int],
    target_count: int,
) -> list[Coord] | None:
    fr0, fc0, fr1, fc1 = fence_bbox
    grey_sources = [tuple(map(int, pt)) for pt in np.argwhere(grey_mask)]
    dist = _distance_map(grey_mask.shape, wall_mask, grey_sources)
    winner_dist = min(int(dist[r, c]) for r, c in base_markers)

    candidates = []
    for r in range(grey_mask.shape[0]):
        for c in range(grey_mask.shape[1]):
            if grey_mask[r, c] or wall_mask[r, c]:
                continue
            if _outside_side(fr0, fc0, fr1, fc1, (r, c)) is None:
                continue
            if int(dist[r, c]) <= winner_dist:
                continue
            if any(abs(r - pr) + abs(c - pc) <= 1 for pr, pc in base_markers):
                continue
            candidates.append((int(dist[r, c]), random.random(), (r, c)))

    if len(candidates) < target_count:
        return None

    extras: list[Coord] = []
    used_sides = {_outside_side(fr0, fc0, fr1, fc1, cell) for cell in base_markers}
    for _dist, _rand, cell in sorted(candidates):
        side = _outside_side(fr0, fc0, fr1, fc1, cell)
        if side not in used_sides:
            extras.append(cell)
            used_sides.add(side)
        if len(extras) == target_count:
            return extras

    for _dist, _rand, cell in sorted(candidates):
        if cell in extras:
            continue
        if any(abs(cell[0] - er) + abs(cell[1] - ec) <= 1 for er, ec in base_markers + extras):
            continue
        extras.append(cell)
        if len(extras) == target_count:
            return extras
    return None


def _build_attempt() -> tuple[np.ndarray, tuple[int, int, int, int], str] | None:
    primary_side = random.choice(("top", "bottom", "left", "right"))
    opposite_side = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[primary_side]

    grey_shape = _transform_shape(random.choice(BASE_GREY_SHAPES))
    grey_h, grey_w = _shape_dims(grey_shape)

    far_pad = random.randint(2, 4)
    near_pad = random.randint(1, 2)
    side_pad0 = random.randint(1, 3)
    side_pad1 = random.randint(1, 3)
    if primary_side == "top":
        pad_t, pad_b, pad_l, pad_r = far_pad, near_pad, side_pad0, side_pad1
    elif primary_side == "bottom":
        pad_t, pad_b, pad_l, pad_r = near_pad, far_pad, side_pad0, side_pad1
    elif primary_side == "left":
        pad_t, pad_b, pad_l, pad_r = side_pad0, side_pad1, far_pad, near_pad
    else:
        pad_t, pad_b, pad_l, pad_r = side_pad0, side_pad1, near_pad, far_pad

    ext_t = random.randint(2, 5)
    ext_b = random.randint(2, 5)
    ext_l = random.randint(2, 5)
    ext_r = random.randint(2, 5)

    fence_h = grey_h + pad_t + pad_b + 2
    fence_w = grey_w + pad_l + pad_r + 2
    height = fence_h + ext_t + ext_b
    width = fence_w + ext_l + ext_r
    if max(height, width) > 30:
        return None

    fr0, fc0 = ext_t, ext_l
    fr1, fc1 = fr0 + fence_h - 1, fc0 + fence_w - 1
    fence_bbox = (fr0, fc0, fr1, fc1)
    grey_origin = (fr0 + 1 + pad_t, fc0 + 1 + pad_l)

    grid = np.full((height, width), Color.BLACK, dtype=int)
    _paint(grid, grey_shape, Color.GREY, origin=grey_origin)

    grey_center_r = grey_origin[0] + grey_h // 2
    grey_center_c = grey_origin[1] + grey_w // 2
    primary_anchor = grey_center_c if primary_side in ("top", "bottom") else grey_center_r
    primary_gate = _gate_position(primary_side, fr0, fc0, fr1, fc1, primary_anchor + random.randint(-1, 1))

    gates = {primary_gate}
    if random.random() < 0.5:
        adj_sides = {
            "top": ("left", "right"),
            "bottom": ("left", "right"),
            "left": ("top", "bottom"),
            "right": ("top", "bottom"),
        }[primary_side]
        second_side = random.choice(adj_sides)
        second_anchor = grey_center_c if second_side in ("top", "bottom") else grey_center_r
        gates.add(_gate_position(second_side, fr0, fc0, fr1, fc1, second_anchor + random.randint(-2, 2)))

    local_outline = _rect_outline_mask(fence_h, fence_w)
    wall_mask = np.zeros_like(grid, dtype=bool)
    wall_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = local_outline
    for gr, gc in gates:
        wall_mask[gr, gc] = False
    grid[wall_mask] = Color.TEAL

    winner_cell = _adjacent_outside(primary_side, primary_gate)
    if not (0 <= winner_cell[0] < height and 0 <= winner_cell[1] < width):
        return None

    if opposite_side == "top":
        decoy_cell = (fr0 - 1, grey_center_c)
    elif opposite_side == "bottom":
        decoy_cell = (fr1 + 1, grey_center_c)
    elif opposite_side == "left":
        decoy_cell = (grey_center_r, fc0 - 1)
    else:
        decoy_cell = (grey_center_r, fc1 + 1)
    if not (0 <= decoy_cell[0] < height and 0 <= decoy_cell[1] < width):
        return None

    if abs(winner_cell[0] - decoy_cell[0]) + abs(winner_cell[1] - decoy_cell[1]) <= 1:
        return None

    grey_mask = grid == Color.GREY
    if any(grid[r, c] != Color.BLACK for r, c in (winner_cell, decoy_cell)):
        return None

    marker_cells = [winner_cell, decoy_cell]
    target_marker_count = random.randint(3, 5)
    extras = _sample_extra_markers(wall_mask, grey_mask, marker_cells, fence_bbox, target_marker_count - 2)
    if extras is None:
        return None
    marker_cells.extend(extras)

    if len(marker_cells) > len(MARKER_COLORS):
        return None
    marker_colors = random.sample(list(MARKER_COLORS), len(marker_cells))
    winner_color = marker_colors[0]
    if winner_color == Color.TEAL:
        return None
    for (r, c), color in zip(marker_cells, marker_colors):
        grid[r, c] = int(color)

    if not _validate_semantics(grid, fence_bbox, primary_side):
        return None
    return grid, fence_bbox, primary_side


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, _fence_bbox, _primary_side = built
        return grid
    raise RuntimeError("failed to generate a valid obstacle-aware nearest-marker instance")
