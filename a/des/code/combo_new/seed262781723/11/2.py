import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
GEN_RETRIES = 400
GRID_MIN = 25
GRID_MAX = 30


@dataclass
class _Container:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class _Piece:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    holes: int
    area: int
    crop: np.ndarray


@dataclass
class _GenPiece:
    holes: int
    mask: np.ndarray
    color: int


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _count_holes(mask: np.ndarray) -> int:
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _outside_background(mask: np.ndarray) -> np.ndarray:
    bg = ~mask
    h, w = bg.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        cr, cc = stack.pop()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return outside


def _is_rectangular_frame(submask: np.ndarray) -> bool:
    h, w = submask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(submask, border)


def _find_containers(grid: np.ndarray) -> tuple[int, list[_Container]]:
    colors = [int(v) for v in np.unique(grid) if int(v) != BLACK]
    by_color: dict[int, list[_Container]] = {}
    for color in colors:
        frames: list[_Container] = []
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            submask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            if _is_rectangular_frame(submask):
                frames.append(_Container(color=color, coords=coords, bbox=(r0, c0, r1, c1)))
        if frames:
            by_color[color] = frames
    assert by_color, "expected at least one container color"
    frame_color = max(
        by_color,
        key=lambda color: (
            len(by_color[color]),
            sum((box.bbox[2] - box.bbox[0] + 1) * (box.bbox[3] - box.bbox[1] + 1) for box in by_color[color]),
        ),
    )
    containers = sorted(by_color[frame_color], key=lambda item: (item.bbox[1], item.bbox[0]))
    return int(frame_color), containers


def _piece_from_component(grid: np.ndarray, color: int, coords: np.ndarray) -> _Piece:
    r0, c0, r1, c1 = _bbox(coords)
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    crop = np.zeros(mask.shape, dtype=int)
    crop[mask] = int(color)
    return _Piece(
        color=int(color),
        coords=coords,
        bbox=(r0, c0, r1, c1),
        holes=_count_holes(mask),
        area=int(mask.sum()),
        crop=crop,
    )


def _find_pieces(grid: np.ndarray, frame_color: int) -> list[_Piece]:
    pieces: list[_Piece] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, frame_color):
            continue
        for coords in _components(grid == color):
            pieces.append(_piece_from_component(grid, color, coords))
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the container indexed by its number of enclosed holes.

    Input structure:
    - Black background with several grey hollow rectangular frames.
    - The grey frames are non-overlapping containers, ordered from left to right.
    - Outside the containers are scattered 4-connected monochrome fragments in non-black, non-grey colors.
    - A fragment may contain zero or more enclosed black cavities inside its own bounding box.

    Rule:
    - Detect the grey rectangular containers and sort them by column from left to right.
    - For each colored fragment, count its enclosed 4-connected black hole regions.
    - Send each fragment to the container whose left-to-right index matches that hole count.
    - Replace the fragment by its tight bounding-box crop, keep its color, sort fragments in each container by area from small to large, and stack them top-down with no blank spacer rows while centering each crop horizontally.
    - Remove all original fragment pixels outside the containers and keep the grey container frames unchanged.

    Key insight:
    - The classification key is the fragment's topology, not its color or outline: pieces are grouped by how many black cavities they enclose, then a second independent rule orders each group by area before stacking.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, containers = _find_containers(g)
    pieces = _find_pieces(g, frame_color)

    out = np.zeros_like(g)
    for container in containers:
        out[container.coords[:, 0], container.coords[:, 1]] = int(frame_color)

    grouped: dict[int, list[_Piece]] = {idx: [] for idx in range(len(containers))}
    for piece in pieces:
        assert 0 <= piece.holes < len(containers), "piece hole count exceeds container range"
        grouped[piece.holes].append(piece)

    for hole_count, container in enumerate(containers):
        ordered = sorted(
            grouped[hole_count],
            key=lambda piece: (
                int(piece.area),
                int(piece.crop.shape[0] * piece.crop.shape[1]),
                int(piece.bbox[0]),
                int(piece.bbox[1]),
            ),
        )
        r0, c0, r1, c1 = container.bbox
        inner_left = c0 + 1
        inner_right = c1 - 1
        cursor = r0 + 1
        for piece in ordered:
            ph, pw = piece.crop.shape
            assert cursor + ph - 1 <= r1 - 1, "piece stack exceeds container height"
            start_c = inner_left + (inner_right - inner_left + 1 - pw) // 2
            assert inner_left <= start_c <= inner_right - pw + 1, "piece exceeds container width"
            target = out[cursor : cursor + ph, start_c : start_c + pw]
            piece_mask = piece.crop != BLACK
            target[piece_mask] = piece.crop[piece_mask]
            cursor += ph

    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    turns = random.randrange(4)
    if turns:
        out = np.rot90(out, turns)
    return np.array(out, dtype=bool)


def _add_outside_arm(mask: np.ndarray) -> bool:
    outside = _outside_background(mask)
    h, w = mask.shape
    candidates: list[tuple[int, int, int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and outside[nr, nc]:
                candidates.append((rr, cc, dr, dc))
    random.shuffle(candidates)
    for r, c, dr, dc in candidates:
        length = random.randint(1, 2)
        cells: list[tuple[int, int]] = []
        ok = True
        for step in range(1, length + 1):
            rr = r + dr * step
            cc = c + dc * step
            if not (0 <= rr < h and 0 <= cc < w) or mask[rr, cc] or not outside[rr, cc]:
                ok = False
                break
            cells.append((rr, cc))
        if ok:
            for rr, cc in cells:
                mask[rr, cc] = True
            return True
    return False


def _make_solid_piece() -> np.ndarray | None:
    for _ in range(200):
        canvas = np.zeros((7, 7), dtype=bool)
        start = (3, 3)
        canvas[start] = True
        cells = {start}
        target_area = random.randint(4, 9)
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 7 and 0 <= nc < 7 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            choice = random.choice(frontier)
            cells.add(choice)
            canvas[choice] = True
        mask = _crop_bool(canvas)
        if min(mask.shape) < 2 or max(mask.shape) > 5:
            continue
        if _count_holes(mask) != 0:
            continue
        return _random_transform(mask)
    return None


def _make_holey_piece(holes: int) -> np.ndarray | None:
    for _ in range(250):
        orient_vertical = random.random() < 0.5
        if holes == 1:
            chamber_sizes = [random.randint(2, 3)]
        else:
            chamber_sizes = [random.randint(1, 2) for _ in range(holes)]
        if orient_vertical:
            inner_h = random.randint(2, 3)
            height = inner_h + 2
            width = sum(chamber_sizes) + max(0, holes - 1) + 2
            base = np.zeros((height, width), dtype=bool)
            base[0, :] = True
            base[-1, :] = True
            base[:, 0] = True
            base[:, -1] = True
            wall_col = 1 + chamber_sizes[0]
            for idx in range(holes - 1):
                base[:, wall_col] = True
                if idx < holes - 2:
                    wall_col += 1 + chamber_sizes[idx + 1]
        else:
            inner_w = random.randint(2, 3)
            height = sum(chamber_sizes) + max(0, holes - 1) + 2
            width = inner_w + 2
            base = np.zeros((height, width), dtype=bool)
            base[0, :] = True
            base[-1, :] = True
            base[:, 0] = True
            base[:, -1] = True
            wall_row = 1 + chamber_sizes[0]
            for idx in range(holes - 1):
                base[wall_row, :] = True
                if idx < holes - 2:
                    wall_row += 1 + chamber_sizes[idx + 1]

        canvas = np.zeros((base.shape[0] + 6, base.shape[1] + 6), dtype=bool)
        canvas[3 : 3 + base.shape[0], 3 : 3 + base.shape[1]] = base
        for _ in range(random.randint(1, 3)):
            _add_outside_arm(canvas)
        mask = _random_transform(_crop_bool(canvas))
        if _count_holes(mask) != holes:
            continue
        if min(mask.shape) < 3 or max(mask.shape) > 8:
            continue
        return mask
    return None


def _make_piece_mask(holes: int) -> np.ndarray | None:
    if holes == 0:
        return _make_solid_piece()
    return _make_holey_piece(holes)


def _pick_group_counts(num_containers: int) -> list[int]:
    counts = [1] * num_containers
    extra_groups = random.sample(range(num_containers), random.randint(1, min(3, num_containers)))
    for idx in extra_groups:
        counts[idx] += 1
    return counts


def _reserve(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1):
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _draw_frame(grid: np.ndarray, r: int, c: int, h: int, w: int):
    grid[r, c : c + w] = GREY
    grid[r + h - 1, c : c + w] = GREY
    grid[r : r + h, c] = GREY
    grid[r : r + h, c + w - 1] = GREY


def _gen_attempt() -> np.ndarray | None:
    num_containers = 4 if random.random() < 0.4 else 3
    counts = _pick_group_counts(num_containers)
    total_pieces = sum(counts)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    if total_pieces > len(palette):
        return None
    random.shuffle(palette)

    masks_by_group: list[list[np.ndarray]] = [[] for _ in range(num_containers)]
    for holes in range(num_containers):
        used_areas: set[int] = set()
        for _ in range(counts[holes]):
            mask = None
            for _ in range(300):
                candidate = _make_piece_mask(holes)
                if candidate is None:
                    continue
                area = int(candidate.sum())
                if area in used_areas:
                    continue
                mask = candidate
                used_areas.add(area)
                break
            if mask is None:
                return None
            masks_by_group[holes].append(mask)

    containers: list[tuple[int, int]] = []
    for group_masks in masks_by_group:
        widths = [int(mask.shape[1]) for mask in group_masks]
        heights = [int(mask.shape[0]) for mask in group_masks]
        inner_w = max(widths) + random.randint(0, 2)
        inner_h = sum(heights) + random.randint(0, 2)
        if inner_w > 8 or inner_h > 12:
            return None
        containers.append((inner_h + 2, inner_w + 2))

    gaps = [random.randint(2, 3) for _ in range(num_containers - 1)]
    span_w = sum(w for _h, w in containers) + sum(gaps)
    if span_w > 28:
        return None

    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(max(GRID_MIN, span_w + 2), GRID_MAX)
    top = random.randint(1, 2)
    x = random.randint(1, grid_w - span_w - 1)
    positions: list[tuple[int, int, int, int]] = []
    for idx, (height, width) in enumerate(containers):
        positions.append((top, x, height, width))
        if idx < len(gaps):
            x += width + gaps[idx]

    container_bottom = max(r + h for r, _c, h, _w in positions)
    pieces_top = container_bottom + random.randint(2, 4)
    if pieces_top >= grid_h - 3:
        return None

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    for r, c, h, w in positions:
        _reserve(blocked, r, c, h, w, pad=1)

    all_pieces: list[_GenPiece] = []
    color_idx = 0
    for holes, group_masks in enumerate(masks_by_group):
        for mask in group_masks:
            all_pieces.append(_GenPiece(holes=holes, mask=mask, color=palette[color_idx]))
            color_idx += 1
    random.shuffle(all_pieces)
    all_pieces.sort(key=lambda piece: (piece.mask.shape[0] * piece.mask.shape[1], piece.mask.shape[0]), reverse=True)

    placements: list[tuple[_GenPiece, int, int]] = []
    for piece in all_pieces:
        ph, pw = piece.mask.shape
        candidates: list[tuple[int, int]] = []
        for r in range(pieces_top, grid_h - ph + 1):
            for c in range(0, grid_w - pw + 1):
                rr0 = max(0, r - 1)
                cc0 = max(0, c - 1)
                rr1 = min(grid_h, r + ph + 1)
                cc1 = min(grid_w, c + pw + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        _reserve(blocked, r, c, ph, pw, pad=1)
        placements.append((piece, r, c))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for r, c, h, w in positions:
        _draw_frame(grid, r, c, h, w)
    for piece, r, c in placements:
        sub = grid[r : r + piece.mask.shape[0], c : c + piece.mask.shape[1]]
        sub[piece.mask] = int(piece.color)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
