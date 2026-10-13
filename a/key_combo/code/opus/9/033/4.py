import random
from collections import Counter
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_shared_for_opus_9_033_4",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
crop_bbox = _ARC_UTILS.crop_bbox


BLACK = 0
PINK = 6
ORANGE = 7
BASE_COLORS = (1, 2, 3, 4, 5, 8, 9)

TILE_H = 6
TILE_W = 6
FRAME_OUTER_H = 5
FRAME_OUTER_W = 5
FRAME_INNER_H = FRAME_OUTER_H - 2
FRAME_INNER_W = FRAME_OUTER_W - 2
MAX_ATTEMPTS = 400


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return mask.astype(int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(ps[r1, c1])
    if r0 > 0:
        total -= int(ps[r0 - 1, c1])
    if c0 > 0:
        total -= int(ps[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(ps[r0 - 1, c0 - 1])
    return total


def _frame_border_count(r0: int, c0: int, r1: int, c1: int) -> int:
    return 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4


def _frame_border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _is_complete_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    border = _frame_border_cells(r0, c0, r1, c1)
    return all(int(grid[r, c]) == ORANGE for r, c in border)


def _rectangular_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = grid == ORANGE
    ps = _prefix_sum(mask)
    H, W = grid.shape
    frames: list[tuple[int, int, int, int]] = []
    for r0 in range(H - 2):
        for r1 in range(r0 + 2, H):
            for c0 in range(W - 2):
                for c1 in range(c0 + 2, W):
                    border_count = _frame_border_count(r0, c0, r1, c1)
                    outer = _rect_sum(ps, r0, c0, r1, c1)
                    inner = _rect_sum(ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    if outer - inner == border_count:
                        frames.append((r0, c0, r1, c1))
    return frames


def _infer_template(grid: np.ndarray, ignored: tuple[int, ...] = (BLACK, PINK)) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    unknown = np.isin(g, ignored)
    coords = np.argwhere(~unknown)
    best_shape = None
    best_template = None

    for h in range(1, H + 1):
        for w in range(1, W + 1):
            area = h * w
            if best_shape is not None and area > best_shape[0] * best_shape[1]:
                continue

            template = np.full((h, w), -1, dtype=int)
            seen = np.zeros((h, w), dtype=bool)
            ok = True
            for r, c in coords:
                rr = int(r) % h
                cc = int(c) % w
                val = int(g[r, c])
                if seen[rr, cc]:
                    if int(template[rr, cc]) != val:
                        ok = False
                        break
                else:
                    seen[rr, cc] = True
                    template[rr, cc] = val
            if ok and seen.all():
                if best_shape is None or area < best_shape[0] * best_shape[1] or (area == best_shape[0] * best_shape[1] and (h, w) < best_shape):
                    best_shape = (h, w)
                    best_template = template
        if best_shape is not None and h * 1 > best_shape[0] * best_shape[1]:
            break

    if best_template is None:
        raise AssertionError("expected a uniquely recoverable repeating template")
    return best_template


def _tile_over_shape(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = map(int, shape)
    rr = np.arange(H) % template.shape[0]
    cc = np.arange(W) % template.shape[1]
    return template[np.ix_(rr, cc)]


def _restore_grid(grid: np.ndarray) -> np.ndarray:
    template = _infer_template(grid)
    tiled = _tile_over_shape(template, grid.shape)
    out = np.asarray(grid, dtype=int).copy()
    out[out == BLACK] = tiled[out == BLACK]
    return out


def _select_frame(restored: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    frames = _rectangular_frames(restored)
    assert frames, "expected at least one complete orange frame after restoration"

    scored: list[tuple[int, tuple[int, int, int, int]]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame
        interior = restored[r0 + 1 : r1, c0 + 1 : c1]
        scored.append((int(np.count_nonzero(interior == PINK)), frame))

    min_markers = min(score for score, _ in scored)
    winners = [frame for score, frame in scored if score == min_markers]
    assert len(winners) == 1, "expected a unique sparsest frame interior"
    winner = winners[0]
    r0, c0, r1, c1 = winner
    return winner, restored[r0 + 1 : r1, c0 + 1 : c1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the periodic canvas, then extract the interior of the complete orange frame with the fewest pink markers.

    Input structure:
    - The grid is a small crop from a two-dimensional repeating pattern whose smallest tile is fully recoverable from non-black cells.
    - Several one-pixel-thick orange rectangular borders are embedded in that periodic pattern, and all complete borders share the same interior size.
    - Each border interior contains zero or more pink marker pixels, while one or two black rectangles occlude part of the canvas and may break a border.

    Rule:
    - Treat black pixels as occlusions and infer the smallest repeating tile that matches every visible non-black, non-pink cell.
    - Tile that template across the full canvas and replace only the black pixels to restore the hidden cells.
    - In the restored grid, find every complete one-pixel-thick orange rectangular border.
    - Count the pink pixels inside each border, choose the unique border with the fewest such markers, and output its strict interior.

    Key insight:
    - At least one candidate border is incomplete before restoration, so the correct sparsest frame can only be identified after recovering the periodic background and the hidden orange border segment.
    """
    restored = _restore_grid(np.asarray(grid, dtype=int))
    _winner, interior = _select_frame(restored)
    return interior


def _minimal_period_exact(grid: np.ndarray) -> tuple[int, int]:
    H, W = grid.shape
    best = None
    for h in range(1, H + 1):
        for w in range(1, W + 1):
            area = h * w
            if best is not None and area > best[0] * best[1]:
                continue
            tile = grid[:h, :w]
            if np.array_equal(_tile_over_shape(tile, grid.shape), grid):
                if best is None or area < best[0] * best[1] or (area == best[0] * best[1] and (h, w) < best):
                    best = (h, w)
        if best is not None and h > best[0]:
            break
    if best is None:
        raise AssertionError("full grid should be periodic")
    return best


def _sample_base_tile() -> np.ndarray:
    for _ in range(200):
        palette = random.sample(BASE_COLORS, random.randint(4, len(BASE_COLORS)))
        tile = np.array(
            [[random.choice(palette) for _ in range(TILE_W)] for _ in range(TILE_H)],
            dtype=int,
        )
        if _minimal_period_exact(tile) == (TILE_H, TILE_W):
            return tile
    raise RuntimeError("failed to sample a primitive base tile")


def _copy_positions(start: int, period: int, span: int, limit: int) -> list[int]:
    coords = []
    pos = start
    while pos + span <= limit:
        if pos >= 0:
            coords.append(pos)
        pos += period
    return coords


def _sample_layout() -> tuple[int, int, int, int, list[tuple[int, int, int, int]]]:
    branch = random.choice(("2x2", "1x3", "3x1"))
    r0 = random.randint(2, 5)
    c0 = random.randint(2, 5)

    if branch == "2x2":
        H = random.randint(r0 + FRAME_OUTER_H + TILE_H, min(20, r0 + FRAME_OUTER_H + 2 * TILE_H - 1))
        W = random.randint(c0 + FRAME_OUTER_W + TILE_W, min(20, c0 + FRAME_OUTER_W + 2 * TILE_W - 1))
    elif branch == "1x3":
        H = random.randint(max(10, r0 + FRAME_OUTER_H), min(14, r0 + FRAME_OUTER_H + TILE_H - 1))
        W = random.randint(c0 + FRAME_OUTER_W + 2 * TILE_W, min(22, c0 + FRAME_OUTER_W + 3 * TILE_W - 1))
    else:
        H = random.randint(r0 + FRAME_OUTER_H + 2 * TILE_H, min(22, r0 + FRAME_OUTER_H + 3 * TILE_H - 1))
        W = random.randint(max(10, c0 + FRAME_OUTER_W), min(14, c0 + FRAME_OUTER_W + TILE_W - 1))

    row_positions = _copy_positions(r0, TILE_H, FRAME_OUTER_H, H)
    col_positions = _copy_positions(c0, TILE_W, FRAME_OUTER_W, W)
    frames = [(rr, cc, rr + FRAME_OUTER_H - 1, cc + FRAME_OUTER_W - 1) for rr in row_positions for cc in col_positions]
    assert len(frames) in (3, 4)
    return H, W, r0, c0, frames


def _marker_counts(n_frames: int) -> list[int]:
    winner = random.randint(0, 1)
    others = []
    for _ in range(n_frames - 1):
        others.append(random.randint(winner + 1, 4))
    counts = [winner] + others
    random.shuffle(counts)
    if counts.count(min(counts)) != 1:
        counts[0] = winner
        for i in range(1, n_frames):
            counts[i] = max(counts[i], winner + 1)
    return counts


def _place_markers(grid: np.ndarray, frames: list[tuple[int, int, int, int]]) -> tuple[list[int], tuple[int, int, int, int], dict[tuple[int, int, int, int], set[tuple[int, int]]]]:
    counts = _marker_counts(len(frames))
    winner_idx = counts.index(min(counts))
    marker_cells: dict[tuple[int, int, int, int], set[tuple[int, int]]] = {}

    for idx, (frame, count) in enumerate(zip(frames, counts)):
        r0, c0, r1, c1 = frame
        interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        if idx == winner_idx and count > 0:
            corners = [(r0 + 1, c0 + 1), (r0 + 1, c1 - 1), (r1 - 1, c0 + 1), (r1 - 1, c1 - 1)]
            chosen = {random.choice(corners)}
        else:
            chosen = set(random.sample(interior_cells, count))
        marker_cells[frame] = chosen
        for r, c in chosen:
            grid[r, c] = PINK

    return counts, frames[winner_idx], marker_cells


def _occluder_for_frame(frame: tuple[int, int, int, int], forbidden: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    H, W = shape
    r0, c0, r1, c1 = frame
    side_order = random.sample(("top", "bottom", "left", "right"), 4)
    for side in side_order:
        if side in ("top", "bottom"):
            seg_w = random.randint(3, 3)
            start_c = random.randint(c0 + 1, c1 - seg_w)
            rr0 = r0 if side == "top" else r1 - 2
            rr1 = rr0 + 2
            cc0 = start_c
            cc1 = start_c + seg_w - 1
            border_cells = [(r0 if side == "top" else r1, c) for c in range(start_c, start_c + seg_w)]
        else:
            seg_h = random.randint(3, 3)
            start_r = random.randint(r0 + 1, r1 - seg_h)
            rr0 = start_r
            rr1 = start_r + seg_h - 1
            cc0 = c0 if side == "left" else c1 - 2
            cc1 = cc0 + 2
            border_cells = [(r, c0 if side == "left" else c1) for r in range(start_r, start_r + seg_h)]

        if rr0 < 0 or cc0 < 0 or rr1 >= H or cc1 >= W:
            continue
        rect_cells = {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}
        if rect_cells & forbidden:
            continue
        return (rr0, cc0, rr1, cc1), border_cells
    raise RuntimeError("failed to place a border-breaking occluder")


def _random_background_occluder(shape: tuple[int, int], forbidden: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    H, W = shape
    for _ in range(120):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        if h > H or w > W:
            continue
        r0 = random.randint(0, H - h)
        c0 = random.randint(0, W - w)
        cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
        if cells & forbidden:
            continue
        return (r0, c0, r0 + h - 1, c0 + w - 1)
    return None


def _apply_black_rect(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK


def _generation_attempt() -> np.ndarray | None:
    H, W, proto_r0, proto_c0, frames = _sample_layout()
    base_tile = _sample_base_tile()
    border_residues = set()
    for r, c in _frame_border_cells(proto_r0, proto_c0, proto_r0 + FRAME_OUTER_H - 1, proto_c0 + FRAME_OUTER_W - 1):
        border_residues.add((r % TILE_H, c % TILE_W))

    tile = base_tile.copy()
    for rr, cc in border_residues:
        tile[rr, cc] = ORANGE

    periodic = _tile_over_shape(tile, (H, W))
    if _minimal_period_exact(periodic) != (TILE_H, TILE_W):
        return None

    expected_frames = sorted(frames)
    detected_periodic = sorted(_rectangular_frames(periodic))
    if detected_periodic != expected_frames:
        return None

    restored = periodic.copy()
    counts, winner_frame, marker_cells = _place_markers(restored, frames)
    winner_markers = min(counts)
    if counts.count(winner_markers) != 1:
        return None

    forbidden = set().union(*marker_cells.values())
    primary_rect, broken_cells = _occluder_for_frame(winner_frame, forbidden, restored.shape)

    input_grid = restored.copy()
    _apply_black_rect(input_grid, primary_rect)

    if random.random() < 0.55:
        background_forbidden = forbidden | set(_frame_border_cells(*winner_frame))
        extra = _random_background_occluder(restored.shape, background_forbidden)
        if extra is not None:
            _apply_black_rect(input_grid, extra)

    input_frames = _rectangular_frames(input_grid)
    if winner_frame in input_frames:
        return None
    if len(input_frames) >= len(expected_frames):
        return None

    visible_mask = ~np.isin(input_grid, (BLACK, PINK))
    for rr in range(TILE_H):
        for cc in range(TILE_W):
            if not np.any(visible_mask[rr::TILE_H, cc::TILE_W]):
                return None

    try:
        restored_by_solver = _restore_grid(input_grid)
    except Exception:
        return None
    if not np.array_equal(restored_by_solver, restored):
        return None

    detected_restored = sorted(_rectangular_frames(restored_by_solver))
    if detected_restored != expected_frames:
        return None

    try:
        winner, output = _select_frame(restored_by_solver)
    except Exception:
        return None
    if winner != winner_frame:
        return None

    if input_frames:
        pre_scores = []
        for frame in input_frames:
            r0, c0, r1, c1 = frame
            interior = input_grid[r0 + 1 : r1, c0 + 1 : c1]
            pre_scores.append((int(np.count_nonzero(interior == PINK)), frame))
        pre_min = min(score for score, _ in pre_scores)
        pre_winners = [frame for score, frame in pre_scores if score == pre_min]
        if len(pre_winners) == 1 and pre_winners[0] == winner_frame:
            return None

    expected_output = crop_bbox(restored, (winner_frame[0] + 1, winner_frame[1] + 1, winner_frame[2] - 1, winner_frame[3] - 1))
    if not np.array_equal(output, expected_output):
        return None
    if np.array_equal(input_grid, restored):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
