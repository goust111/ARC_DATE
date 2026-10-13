import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

ROTATION_FROM_BORDER = {
    RED: 0,
    BLUE: 1,
    GREEN: 2,
    YELLOW: 3,
}
BOARD_COLOR_CHOICES = [MAGENTA, ORANGE, CYAN]

GEN_RETRIES = 400
MAX_SHAPE_RETRIES = 240
LOCAL_PAD = 5


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_true_mask(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    assert len(points) > 0, "mask must contain at least one true cell"
    r0 = int(points[:, 0].min())
    r1 = int(points[:, 0].max())
    c0 = int(points[:, 1].min())
    c1 = int(points[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_mask(mask: np.ndarray, turns_cw: int) -> np.ndarray:
    return np.rot90(mask, -turns_cw).copy()


def _sprite_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BLACK, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _mask_is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    if total == 0:
        return False
    return len(_components(mask.astype(bool))) == 1


def _ring_bbox_if_valid(grid: np.ndarray, color: int, points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox_from_points(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] == color):
        return None
    if not np.all(sub[-1, :] == color):
        return None
    if not np.all(sub[:, 0] == color):
        return None
    if not np.all(sub[:, -1] == color):
        return None
    if np.any(sub[1:-1, 1:-1] == color):
        return None
    perimeter = 2 * sub.shape[0] + 2 * sub.shape[1] - 4
    if len(points) != perimeter:
        return None
    return r0, c0, r1, c1


def _find_collection_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(grid == MAROON)
    candidates = []
    for comp in comps:
        bbox = _ring_bbox_if_valid(grid, MAROON, comp)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if np.all(interior == BLACK):
            candidates.append(bbox)
    assert candidates, "expected one maroon collection frame"
    return max(candidates, key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))


def _find_molds(grid: np.ndarray) -> list[dict[str, object]]:
    molds: list[dict[str, object]] = []
    for color in ROTATION_FROM_BORDER:
        for comp in _components(grid == color):
            bbox = _ring_bbox_if_valid(grid, color, comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if not np.all((inner == BLACK) | (inner == GREY)):
                continue
            mask = inner == GREY
            if not mask.any():
                continue
            molds.append({"bbox": bbox, "color": int(color), "mask": mask.copy()})
    assert molds, "expected at least one mold"
    molds.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))  # type: ignore[index]
    return molds


def _find_board(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    reserved = {BLACK, GREY, MAROON, *ROTATION_FROM_BORDER.keys()}
    best: tuple[int, tuple[int, int, int, int], int] | None = None
    for color in sorted(set(int(v) for v in grid.reshape(-1))):
        if color in reserved:
            continue
        for comp in _components(grid == color):
            bbox = _bbox_from_points(comp)
            r0, c0, r1, c1 = bbox
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(sub == color):
                continue
            area = sub.shape[0] * sub.shape[1]
            if best is None or area > best[2]:
                best = (color, bbox, area)
    assert best is not None, "expected a solid material board"
    return best[0], best[1]


def _all_template_matches(board_mask: np.ndarray, template: np.ndarray) -> list[tuple[int, int]]:
    h, w = board_mask.shape
    th, tw = template.shape
    hits: list[tuple[int, int]] = []
    for r in range(h - th + 1):
        for c in range(w - tw + 1):
            if np.array_equal(board_mask[r : r + th, c : c + tw], template):
                hits.append((r, c))
    return hits


def _find_unique_template_match(board_mask: np.ndarray, template: np.ndarray) -> tuple[int, int]:
    hits = _all_template_matches(board_mask, template)
    assert len(hits) == 1, f"template must match exactly once, got {len(hits)}"
    return hits[0]


def _pack_sprites(
    sprites: list[np.ndarray], inner_h: int, inner_w: int
) -> tuple[list[tuple[int, int]], int, int]:
    placements: list[tuple[int, int]] = []
    row = 0
    col = 0
    row_h = 0
    rows_used = 0
    for sprite in sprites:
        sh, sw = sprite.shape
        if col > 0 and col + sw > inner_w:
            row += row_h
            col = 0
            row_h = 0
        if sw > inner_w or row + sh > inner_h:
            raise ValueError("sprites do not fit in collection area")
        placements.append((row, col))
        col += sw
        row_h = max(row_h, sh)
        rows_used = max(rows_used, row + row_h)
    row_count = 0
    prev_r = None
    for r, _ in placements:
        if prev_r != r:
            row_count += 1
            prev_r = r
    return placements, rows_used, row_count


def _paint_collection(
    out: np.ndarray,
    collector_bbox: tuple[int, int, int, int],
    sprites: list[np.ndarray],
) -> None:
    r0, c0, r1, c1 = collector_bbox
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    inner[:, :] = BLACK
    placements, _, _ = _pack_sprites(sprites, inner.shape[0], inner.shape[1])
    for sprite, (sr, sc) in zip(sprites, placements):
        sh, sw = sprite.shape
        window = inner[sr : sr + sh, sc : sc + sw]
        window[sprite != BLACK] = sprite[sprite != BLACK]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Punch template-shaped holes out of the material board, then rotate and pack
    the cut fragments into the maroon collection area.
    """
    g = np.asarray(grid, dtype=int)
    collector_bbox = _find_collection_bbox(g)
    molds = _find_molds(g)
    board_color, _board_bbox = _find_board(g)

    out = g.copy()
    board_mask = out == board_color
    collected: list[np.ndarray] = []

    for mold in molds:
        template = np.asarray(mold["mask"], dtype=bool)
        top, left = _find_unique_template_match(board_mask, template)
        th, tw = template.shape
        patch = board_mask[top : top + th, left : left + tw]
        assert np.array_equal(patch, template)

        piece_mask = _crop_true_mask(template)
        turns_cw = ROTATION_FROM_BORDER[int(mold["color"])]
        piece_mask = _rotate_mask(piece_mask, turns_cw)
        collected.append(_sprite_from_mask(piece_mask, board_color))

        out[top : top + th, left : left + tw][template] = BLACK
        board_mask[top : top + th, left : left + tw][template] = False

    _paint_collection(out, collector_bbox, collected)
    return out


def _feature_corners(
    board_rect: tuple[int, int, int, int], holes: list[tuple[int, int, int, int]]
) -> list[tuple[int, int]]:
    rects = [board_rect] + holes
    corners: list[tuple[int, int]] = []
    for r0, c0, h, w in rects:
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        corners.extend([(r0, c0), (r0, c1), (r1, c0), (r1, c1)])
    return corners


def _sample_unique_shape(
    rng: random.Random,
    occ: np.ndarray,
    board_rect: tuple[int, int, int, int],
    holes: list[tuple[int, int, int, int]],
    step_index: int,
) -> tuple[int, int, np.ndarray] | None:
    local_h, local_w = occ.shape
    corners = _feature_corners(board_rect, holes)
    for _ in range(MAX_SHAPE_RETRIES):
        anchor_r, anchor_c = corners[rng.randrange(len(corners))]
        h = rng.randint(2, 6)
        w = rng.randint(2, 6)
        r0 = anchor_r - rng.randint(0, h - 1)
        c0 = anchor_c - rng.randint(0, w - 1)
        r0 = max(0, min(r0, local_h - h))
        c0 = max(0, min(c0, local_w - w))

        template = occ[r0 : r0 + h, c0 : c0 + w].copy()
        cells = int(template.sum())
        if cells < 3 or cells > 14:
            continue
        if not _mask_is_connected(template):
            continue
        if step_index > 0 and cells == h * w:
            continue
        if step_index > 0 and not np.any(~template):
            continue

        hits = _all_template_matches(occ, template)
        if hits != [(r0, c0)]:
            continue
        return r0, c0, template
    return None


def _choose_collector_geometry(
    rng: random.Random, sprites: list[np.ndarray]
) -> tuple[int, int]:
    max_w = max(sprite.shape[1] for sprite in sprites)
    total_w = sum(sprite.shape[1] for sprite in sprites)
    candidates: list[tuple[int, int, int]] = []
    for inner_w in range(max_w, total_w + 1):
        try:
            _placements, used_h, row_count = _pack_sprites(sprites, 64, inner_w)
        except ValueError:
            continue
        candidates.append((inner_w, used_h, row_count))
    assert candidates, "collector must have at least one feasible width"

    want_wrap = rng.random() < 0.5
    filtered = [c for c in candidates if (c[2] > 1) == want_wrap]
    options = filtered if filtered else candidates
    inner_w, used_h, _row_count = options[rng.randrange(len(options))]
    extra_h = rng.randint(0, 2)
    return inner_w, used_h + extra_h


def _build_input_from_plan(
    rng: random.Random,
    board_h: int,
    board_w: int,
    board_color: int,
    shapes: list[dict[str, object]],
    border_colors: list[int],
) -> np.ndarray:
    molds: list[np.ndarray] = []
    sprites: list[np.ndarray] = []
    for shape, border_color in zip(shapes, border_colors):
        template = np.asarray(shape["template"], dtype=bool)
        inner = np.full(template.shape, BLACK, dtype=int)
        inner[template] = GREY
        mold = np.full((template.shape[0] + 2, template.shape[1] + 2), int(border_color), dtype=int)
        mold[1:-1, 1:-1] = inner
        molds.append(mold)

        piece_mask = _crop_true_mask(template)
        piece_mask = _rotate_mask(piece_mask, ROTATION_FROM_BORDER[int(border_color)])
        sprites.append(_sprite_from_mask(piece_mask, board_color))

    collector_inner_w, collector_inner_h = _choose_collector_geometry(rng, sprites)
    collector_outer_w = collector_inner_w + 2
    collector_outer_h = collector_inner_h + 2

    top_need = 0
    left_need = 0
    bottom_need = 0
    right_need = 0
    for shape in shapes:
        r0, c0 = shape["top_left"]  # type: ignore[misc]
        template = np.asarray(shape["template"], dtype=bool)
        top_need = max(top_need, LOCAL_PAD - int(r0))
        left_need = max(left_need, LOCAL_PAD - int(c0))
        bottom_need = max(bottom_need, int(r0) + template.shape[0] - (LOCAL_PAD + board_h))
        right_need = max(right_need, int(c0) + template.shape[1] - (LOCAL_PAD + board_w))

    board_top = rng.randint(top_need, top_need + 2)
    board_left = rng.randint(left_need, left_need + 2)
    board_bottom = board_top + board_h
    board_right = board_left + board_w

    mold_gap_x = rng.randint(3, 5)
    mold_col = board_right + mold_gap_x
    row = rng.randint(1, 2)
    max_mold_w = max(mold.shape[1] for mold in molds)
    mold_right = mold_col + max_mold_w
    mold_rows: list[int] = []
    for mold in molds:
        mold_rows.append(row)
        row += mold.shape[0] + rng.randint(1, 2)
    molds_total_bottom = max(r + mold.shape[0] for r, mold in zip(mold_rows, molds))

    collector_top = max(board_bottom, molds_total_bottom) + rng.randint(2, 4)
    collector_left = rng.randint(1, 3)

    grid_h = collector_top + collector_outer_h + rng.randint(1, 2)
    grid_w = max(
        mold_right + rng.randint(1, 2),
        collector_left + collector_outer_w + rng.randint(1, 2),
        board_right + right_need + rng.randint(2, 4),
    )
    grid_h = max(grid_h, board_bottom + bottom_need + rng.randint(1, 2))
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    grid[board_top:board_bottom, board_left:board_right] = int(board_color)

    for mold, row in zip(molds, mold_rows):
        h, w = mold.shape
        col = mold_col + rng.randint(0, max(0, max_mold_w - w))
        grid[row : row + h, col : col + w] = mold

    cr0 = collector_top
    cc0 = collector_left
    cr1 = collector_top + collector_outer_h - 1
    cc1 = collector_left + collector_outer_w - 1
    grid[cr0, cc0 : cc1 + 1] = MAROON
    grid[cr1, cc0 : cc1 + 1] = MAROON
    grid[cr0 : cr1 + 1, cc0] = MAROON
    grid[cr0 : cr1 + 1, cc1] = MAROON

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    board_h = rng.randint(8, 13)
    board_w = rng.randint(9, 15)
    board_color = BOARD_COLOR_CHOICES[rng.randrange(len(BOARD_COLOR_CHOICES))]
    mold_count = rng.choice([3, 4])

    local_h = board_h + 2 * LOCAL_PAD
    local_w = board_w + 2 * LOCAL_PAD
    occ = np.zeros((local_h, local_w), dtype=bool)
    occ[LOCAL_PAD : LOCAL_PAD + board_h, LOCAL_PAD : LOCAL_PAD + board_w] = True
    board_rect = (LOCAL_PAD, LOCAL_PAD, board_h, board_w)

    shapes: list[dict[str, object]] = []
    holes: list[tuple[int, int, int, int]] = []
    for step_index in range(mold_count):
        sampled = _sample_unique_shape(rng, occ, board_rect, holes, step_index)
        if sampled is None:
            return None
        r0, c0, template = sampled
        h, w = template.shape
        occ[r0 : r0 + h, c0 : c0 + w][template] = False
        holes.append((r0, c0, h, w))
        shapes.append({"top_left": (r0, c0), "template": template.copy()})

    border_colors = rng.sample(list(ROTATION_FROM_BORDER.keys()), mold_count)
    grid = _build_input_from_plan(rng, board_h, board_w, board_color, shapes, border_colors)

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid input")
