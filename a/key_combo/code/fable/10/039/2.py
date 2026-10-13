from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
WALL = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MOVE_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Piece:
    color: int
    side: str
    depth: int
    offset: int
    top: int
    left: int
    height: int
    width: int
    mask: np.ndarray
    tag: str = ""

    @property
    def lane(self) -> int:
        if self.side in ("top", "bottom"):
            return self.left + self.offset
        return self.top + self.offset

    def cells(self, top: int | None = None, left: int | None = None) -> list[tuple[int, int]]:
        rr0 = self.top if top is None else int(top)
        cc0 = self.left if left is None else int(left)
        coords = np.argwhere(self.mask)
        return [(rr0 + int(r), cc0 + int(c)) for r, c in coords]

    def bbox(self, top: int | None = None, left: int | None = None) -> tuple[int, int, int, int]:
        rr0 = self.top if top is None else int(top)
        cc0 = self.left if left is None else int(left)
        return (rr0, cc0, rr0 + self.height - 1, cc0 + self.width - 1)


def _shape_mask(height: int, width: int, side: str, depth: int, offset: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if side == "top":
        mask[:depth, offset] = False
    elif side == "bottom":
        mask[height - depth :, offset] = False
    elif side == "left":
        mask[offset, :depth] = False
    elif side == "right":
        mask[offset, width - depth :] = False
    else:
        raise ValueError(side)
    return mask


def _make_piece(
    color: int,
    side: str,
    depth: int,
    offset: int,
    top: int,
    left: int,
    height: int,
    width: int,
    tag: str = "",
) -> Piece:
    return Piece(
        color=int(color),
        side=side,
        depth=int(depth),
        offset=int(offset),
        top=int(top),
        left=int(left),
        height=int(height),
        width=int(width),
        mask=_shape_mask(int(height), int(width), side, int(depth), int(offset)),
        tag=tag,
    )


def _border_color(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    colors = np.unique(border)
    assert len(colors) == 1, "all border cells must belong to the same wall color"
    return int(colors[0])


def _background_color(grid: np.ndarray, wall: int) -> int:
    interior = grid[grid != wall]
    colors, counts = np.unique(interior, return_counts=True)
    assert len(colors) > 0, "grid must contain non-wall cells"
    return int(colors[np.argmax(counts)])


def _monochrome_components(grid: np.ndarray, skip: set[int]) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in skip:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, np.array(coords, dtype=int)))
    return comps


def _parse_piece(color: int, coords: np.ndarray) -> Piece:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    patch = np.zeros((height, width), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    candidates: list[tuple[str, int, int]] = []

    top_missing = np.flatnonzero(~patch[0])
    if len(top_missing) == 1:
        offset = int(top_missing[0])
        depth = 0
        while depth < height and not patch[depth, offset]:
            depth += 1
        if depth in (1, 2) and depth < height and patch[depth, offset]:
            expected = np.ones_like(patch, dtype=bool)
            expected[:depth, offset] = False
            if np.array_equal(patch, expected):
                candidates.append(("top", depth, offset))

    bottom_missing = np.flatnonzero(~patch[-1])
    if len(bottom_missing) == 1:
        offset = int(bottom_missing[0])
        depth = 0
        while depth < height and not patch[height - 1 - depth, offset]:
            depth += 1
        if depth in (1, 2) and depth < height and patch[height - 1 - depth, offset]:
            expected = np.ones_like(patch, dtype=bool)
            expected[height - depth :, offset] = False
            if np.array_equal(patch, expected):
                candidates.append(("bottom", depth, offset))

    left_missing = np.flatnonzero(~patch[:, 0])
    if len(left_missing) == 1:
        offset = int(left_missing[0])
        depth = 0
        while depth < width and not patch[offset, depth]:
            depth += 1
        if depth in (1, 2) and depth < width and patch[offset, depth]:
            expected = np.ones_like(patch, dtype=bool)
            expected[offset, :depth] = False
            if np.array_equal(patch, expected):
                candidates.append(("left", depth, offset))

    right_missing = np.flatnonzero(~patch[:, -1])
    if len(right_missing) == 1:
        offset = int(right_missing[0])
        depth = 0
        while depth < width and not patch[offset, width - 1 - depth]:
            depth += 1
        if depth in (1, 2) and depth < width and patch[offset, width - 1 - depth]:
            expected = np.ones_like(patch, dtype=bool)
            expected[offset, width - depth :] = False
            if np.array_equal(patch, expected):
                candidates.append(("right", depth, offset))

    assert len(candidates) == 1, "each object must be a solid rectangle with exactly one side notch"
    side, depth, offset = candidates[0]
    return _make_piece(int(color), side, depth, offset, int(r0), int(c0), height, width)


def _extract_pieces(grid: np.ndarray, wall: int, bg: int) -> list[Piece]:
    pieces = [_parse_piece(color, coords) for color, coords in _monochrome_components(grid, {wall, bg})]
    assert pieces, "expected at least one colored notched rectangle"
    return pieces


def _tab_maps(grid: np.ndarray, wall: int) -> dict[str, dict[int, int]]:
    h, w = grid.shape
    tabs = {side: {} for side in SIDES}
    for c in range(1, w - 1):
        depth = 0
        while 1 + depth < h - 1 and int(grid[1 + depth, c]) == wall:
            depth += 1
        if depth:
            tabs["top"][c] = depth
        depth = 0
        while h - 2 - depth > 0 and int(grid[h - 2 - depth, c]) == wall:
            depth += 1
        if depth:
            tabs["bottom"][c] = depth
    for r in range(1, h - 1):
        depth = 0
        while 1 + depth < w - 1 and int(grid[r, 1 + depth]) == wall:
            depth += 1
        if depth:
            tabs["left"][r] = depth
        depth = 0
        while w - 2 - depth > 0 and int(grid[r, w - 2 - depth]) == wall:
            depth += 1
        if depth:
            tabs["right"][r] = depth
    return tabs


def _sort_key(piece: Piece) -> int:
    if piece.side == "top":
        return piece.top
    if piece.side == "bottom":
        return -piece.top
    if piece.side == "left":
        return piece.left
    return -piece.left


def _simulate_origins(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    pieces: list[Piece],
    movable_ids: set[int],
) -> dict[int, tuple[int, int]]:
    h, w = shape
    occupied = {tuple(x) for x in np.argwhere(wall_mask)}
    cell_cache = {id(piece): piece.cells() for piece in pieces}
    for cells in cell_cache.values():
        occupied.update(cells)

    origins: dict[int, tuple[int, int]] = {}
    for side in SIDES:
        group = sorted((piece for piece in pieces if id(piece) in movable_ids and piece.side == side), key=_sort_key)
        for piece in group:
            cells = cell_cache[id(piece)]
            for cell in cells:
                occupied.remove(cell)
            dr, dc = MOVE_BY_SIDE[side]
            top, left = piece.top, piece.left
            while True:
                shifted = [(r + dr, c + dc) for r, c in cells]
                if any(not (0 <= r < h and 0 <= c < w) or (r, c) in occupied for r, c in shifted):
                    break
                cells = shifted
                top += dr
                left += dc
            origins[id(piece)] = (top, left)
            for cell in cells:
                occupied.add(cell)

    for piece in pieces:
        origins.setdefault(id(piece), (piece.top, piece.left))
    return origins


def _render_grid(shape: tuple[int, int], tabs: dict[str, dict[int, int]], pieces: list[Piece], origins: dict[int, tuple[int, int]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL
    for c, depth in tabs["top"].items():
        grid[1 : 1 + depth, c] = WALL
    for c, depth in tabs["bottom"].items():
        grid[h - 1 - depth : h - 1, c] = WALL
    for r, depth in tabs["left"].items():
        grid[r, 1 : 1 + depth] = WALL
    for r, depth in tabs["right"].items():
        grid[r, w - 1 - depth : w - 1] = WALL
    for piece in pieces:
        top, left = origins[id(piece)]
        region = grid[top : top + piece.height, left : left + piece.width]
        assert np.all(region[piece.mask] == BG), "pieces must not overlap walls or each other when rendered"
        region[piece.mask] = piece.color
    return grid


def _dock_origin(piece: Piece, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if piece.side == "top":
        return (1, piece.left)
    if piece.side == "bottom":
        return (h - 1 - piece.height, piece.left)
    if piece.side == "left":
        return (piece.top, 1)
    return (piece.top, w - 1 - piece.width)


def _adjacent_origin(front: Piece, back: Piece, front_origin: tuple[int, int]) -> tuple[int, int]:
    if front.side == "top":
        return (front_origin[0] + front.height, back.left)
    if front.side == "bottom":
        return (front_origin[0] - back.height, back.left)
    if front.side == "left":
        return (back.top, front_origin[1] + front.width)
    return (back.top, front_origin[1] - back.width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each notched rectangle toward its facing wall only when the aligned wall tab has the matching depth.

    Input structure:
    - The grid border is a one-cell-thick wall of a single color, and some border cells extend inward as one-cell-wide tabs of depth 1 or 2.
    - Inside the field are separated monochrome rectangles, each missing a one-cell-wide notch of depth 1 or 2 on exactly one side.
    - A rectangle's notch points toward one of the four walls; the notch row or column is the rectangle's alignment lane.

    Rule:
    - For each rectangle, inspect the wall that faces its notch and look for a wall tab on the same lane.
    - If the aligned tab exists and its depth equals the notch depth, slide the rectangle straight toward that wall until the next step would collide; a docked rectangle ends flush with the wall and the tab fills its notch.
    - If no aligned tab exists, or the aligned tab depth differs, the rectangle stays put; when multiple rectangles slide along the same lane toward the same wall, the nearer one docks first and later ones stop against it.

    Key insight:
    - Movement depends on two visible conditions at once: lane alignment chooses the candidate tab, and exact depth complement decides whether the rectangle is allowed to slide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    wall = _border_color(g)
    bg = _background_color(g, wall)
    pieces = _extract_pieces(g, wall, bg)
    tabs = _tab_maps(g, wall)
    movable_ids = {
        id(piece)
        for piece in pieces
        if tabs[piece.side].get(piece.lane, 0) == piece.depth
    }
    origins = _simulate_origins(g.shape, g == wall, pieces, movable_ids)
    out = np.full_like(g, bg)
    out[g == wall] = wall
    for piece in pieces:
        top, left = origins[id(piece)]
        region = out[top : top + piece.height, left : left + piece.width]
        region[piece.mask] = piece.color
    return out


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _sample_dims(side: str) -> tuple[int, int, int, int]:
    if side in ("left", "right"):
        height = random.randint(4, 6)
        width = random.randint(4, 7)
        offset = random.randint(1, height - 2)
    else:
        height = random.randint(4, 7)
        width = random.randint(4, 6)
        offset = random.randint(1, width - 2)
    depth = random.randint(1, 2)
    return height, width, depth, offset


def _sample_single(side: str, color: int, shape: tuple[int, int], tag: str) -> Piece | None:
    h, w = shape
    for _ in range(80):
        height, width, depth, offset = _sample_dims(side)
        if h - height - 5 < 4 or w - width - 5 < 4:
            continue
        top = random.randint(4, h - height - 5)
        left = random.randint(4, w - width - 5)
        return _make_piece(color, side, depth, offset, top, left, height, width, tag)
    return None


def _sample_pair(side: str, colors: tuple[int, int], shape: tuple[int, int], tag: str) -> list[Piece] | None:
    h, w = shape
    for _ in range(120):
        height, width, depth, offset = _sample_dims(side)
        gap = random.randint(2, 4)
        if side == "left":
            max_front_left = w - (2 * width + gap + 2)
            if max_front_left < 4:
                continue
            if h - height - 5 < 4:
                continue
            top = random.randint(4, h - height - 5)
            front_left = random.randint(4, max_front_left)
            back_left = front_left + width + gap
            front = _make_piece(colors[0], side, depth, offset, top, front_left, height, width, f"{tag}_front")
            back = _make_piece(colors[1], side, depth, offset, top, back_left, height, width, f"{tag}_back")
            return [front, back]
        if side == "right":
            min_front_left = width + gap + 2
            max_front_left = w - width - 4
            if min_front_left > max_front_left:
                continue
            if h - height - 5 < 4:
                continue
            top = random.randint(4, h - height - 5)
            front_left = random.randint(min_front_left, max_front_left)
            back_left = front_left - width - gap
            front = _make_piece(colors[0], side, depth, offset, top, front_left, height, width, f"{tag}_front")
            back = _make_piece(colors[1], side, depth, offset, top, back_left, height, width, f"{tag}_back")
            return [front, back]
        if side == "top":
            max_front_top = h - (2 * height + gap + 2)
            if max_front_top < 4:
                continue
            if w - width - 5 < 4:
                continue
            left = random.randint(4, w - width - 5)
            front_top = random.randint(4, max_front_top)
            back_top = front_top + height + gap
            front = _make_piece(colors[0], side, depth, offset, front_top, left, height, width, f"{tag}_front")
            back = _make_piece(colors[1], side, depth, offset, back_top, left, height, width, f"{tag}_back")
            return [front, back]
        min_front_top = height + gap + 2
        max_front_top = h - height - 4
        if min_front_top > max_front_top:
            continue
        if w - width - 5 < 4:
            continue
        left = random.randint(4, w - width - 5)
        front_top = random.randint(min_front_top, max_front_top)
        back_top = front_top - height - gap
        front = _make_piece(colors[0], side, depth, offset, front_top, left, height, width, f"{tag}_front")
        back = _make_piece(colors[1], side, depth, offset, back_top, left, height, width, f"{tag}_back")
        return [front, back]
    return None


def _sample_single_clear(
    side: str,
    color: int,
    shape: tuple[int, int],
    tag: str,
    existing_boxes: list[tuple[int, int, int, int]],
) -> Piece | None:
    for _ in range(120):
        piece = _sample_single(side, color, shape, tag)
        if piece is None:
            continue
        if all(_boxes_clear(piece.bbox(), box, pad=1) for box in existing_boxes):
            return piece
    return None


def _sample_pair_clear(
    side: str,
    colors: tuple[int, int],
    shape: tuple[int, int],
    tag: str,
    existing_boxes: list[tuple[int, int, int, int]],
) -> list[Piece] | None:
    for _ in range(120):
        pair = _sample_pair(side, colors, shape, tag)
        if pair is None:
            continue
        boxes = [piece.bbox() for piece in pair]
        if not _boxes_clear(boxes[0], boxes[1], pad=1):
            continue
        if all(all(_boxes_clear(box, other, pad=1) for other in existing_boxes) for box in boxes):
            return pair
    return None


def _build_tabs(shape: tuple[int, int], metas: list[dict]) -> dict[str, dict[int, int]] | None:
    h, w = shape
    usable = {
        "top": set(range(2, w - 2)),
        "bottom": set(range(2, w - 2)),
        "left": set(range(2, h - 2)),
        "right": set(range(2, h - 2)),
    }
    tabs = {side: {} for side in SIDES}
    for meta in metas:
        piece: Piece = meta["piece"]
        lane = piece.lane
        if lane not in usable[piece.side]:
            return None
        outcome = meta["outcome"]
        if outcome in ("match", "pair_front", "pair_back"):
            needed = piece.depth
            existing = tabs[piece.side].get(lane)
            if existing not in (None, needed):
                return None
            tabs[piece.side][lane] = needed
        elif outcome == "mismatch":
            wrong = 1 if piece.depth == 2 else 2
            if lane in tabs[piece.side]:
                return None
            tabs[piece.side][lane] = wrong

    free_lanes = {side: sorted(usable[side] - set(tabs[side])) for side in SIDES}
    extra_count = random.randint(1, 3)
    choices = [(side, lane) for side in SIDES for lane in free_lanes[side]]
    random.shuffle(choices)
    for side, lane in choices[:extra_count]:
        tabs[side][lane] = random.randint(1, 2)
    return tabs


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, pieces: list[Piece], metas: list[dict], tabs: dict[str, dict[int, int]]) -> bool:
    wall = _border_color(inp)
    bg = _background_color(inp, wall)
    if wall != WALL or bg != BG:
        return False
    parsed = _extract_pieces(inp, wall, bg)
    by_color = {piece.color: piece for piece in pieces}
    if len(parsed) != len(pieces):
        return False
    for piece in parsed:
        ref = by_color.get(piece.color)
        if ref is None:
            return False
        if (piece.side, piece.depth, piece.offset, piece.top, piece.left, piece.height, piece.width) != (
            ref.side,
            ref.depth,
            ref.offset,
            ref.top,
            ref.left,
            ref.height,
            ref.width,
        ):
            return False

    movable_ids = {id(meta["piece"]) for meta in metas if meta["outcome"] in ("match", "pair_front", "pair_back")}
    origins = _simulate_origins(inp.shape, inp == WALL, pieces, movable_ids)
    moved = 0
    stayed = 0
    pair_fronts: dict[str, Piece] = {}
    pair_backs: dict[str, Piece] = {}
    for meta in metas:
        piece: Piece = meta["piece"]
        if meta["outcome"] == "pair_front":
            pair_fronts[meta["pair_tag"]] = piece
        elif meta["outcome"] == "pair_back":
            pair_backs[meta["pair_tag"]] = piece

    for meta in metas:
        piece: Piece = meta["piece"]
        origin = origins[id(piece)]
        if origin == (piece.top, piece.left):
            stayed += 1
        else:
            moved += 1
        if meta["outcome"] == "match":
            if origin != _dock_origin(piece, inp.shape):
                return False
        elif meta["outcome"] == "mismatch" or meta["outcome"] == "absent":
            if origin != (piece.top, piece.left):
                return False
        elif meta["outcome"] == "pair_front":
            if origin != _dock_origin(piece, inp.shape):
                return False
        elif meta["outcome"] == "pair_back":
            front = pair_fronts[meta["pair_tag"]]
            target = _adjacent_origin(front, piece, origins[id(front)])
            if origin != target:
                return False

    if moved < 1 or stayed < 1:
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    tab_total = sum(len(tabs[side]) for side in SIDES)
    if tab_total < len(metas):
        return False
    return True


def _generate_case() -> np.ndarray | None:
    shape = (random.randint(16, 24), random.randint(16, 24))
    colors = list(OBJECT_COLORS)
    random.shuffle(colors)

    include_pair = random.random() < 0.65
    side_pool = list(SIDES)
    random.shuffle(side_pool)
    used_sides: list[str] = []
    metas: list[dict] = []
    pieces: list[Piece] = []
    existing_boxes: list[tuple[int, int, int, int]] = []

    if include_pair:
        pair_side = side_pool.pop()
        used_sides.append(pair_side)
        pair_pieces = _sample_pair_clear(pair_side, (colors.pop(), colors.pop()), shape, "pair0", existing_boxes)
        if pair_pieces is None:
            return None
        pieces.extend(pair_pieces)
        existing_boxes.extend(piece.bbox() for piece in pair_pieces)
        metas.append({"piece": pair_pieces[0], "outcome": "pair_front", "pair_tag": "pair0"})
        metas.append({"piece": pair_pieces[1], "outcome": "pair_back", "pair_tag": "pair0"})

    match_side = side_pool.pop() if side_pool else random.choice(SIDES)
    used_sides.append(match_side)
    match_piece = _sample_single_clear(match_side, colors.pop(), shape, "single_match", existing_boxes)
    if match_piece is None:
        return None
    pieces.append(match_piece)
    existing_boxes.append(match_piece.bbox())
    metas.append({"piece": match_piece, "outcome": "match"})

    stay_side = side_pool.pop() if side_pool else random.choice([side for side in SIDES if side not in used_sides] or SIDES)
    used_sides.append(stay_side)
    stay_piece = _sample_single_clear(stay_side, colors.pop(), shape, "single_stay", existing_boxes)
    if stay_piece is None:
        return None
    pieces.append(stay_piece)
    existing_boxes.append(stay_piece.bbox())
    metas.append({"piece": stay_piece, "outcome": random.choice(("mismatch", "absent"))})

    if side_pool and random.random() < 0.7:
        extra_side = side_pool.pop()
        extra_piece = _sample_single_clear(extra_side, colors.pop(), shape, "single_extra", existing_boxes)
        if extra_piece is None:
            return None
        pieces.append(extra_piece)
        existing_boxes.append(extra_piece.bbox())
        metas.append({"piece": extra_piece, "outcome": random.choice(("match", "mismatch", "absent"))})

    tabs = _build_tabs(shape, metas)
    if tabs is None:
        return None

    movable_ids = {id(meta["piece"]) for meta in metas if meta["outcome"] in ("match", "pair_front", "pair_back")}
    wall_only = _render_grid(shape, tabs, [], {})
    origins = _simulate_origins(shape, wall_only == WALL, pieces, movable_ids)
    try:
        expected = _render_grid(shape, tabs, pieces, origins)
        inp = _render_grid(shape, tabs, pieces, {id(piece): (piece.top, piece.left) for piece in pieces})
    except AssertionError:
        return None

    if not _semantic_checks(inp, expected, pieces, metas, tabs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid docking puzzle input")
