import random

import numpy as np


BLACK = 0
AZURE = 8
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

FRAME_COUNT_RANGE = (2, 3)
INNER_H_RANGE = (3, 5)
INNER_W_RANGE = (3, 6)
MAX_PIECES_PER_FRAME = 4
CASE_RETRIES = 400
FRAME_SPEC_RETRIES = 120
PARTITION_RETRIES = 120
OBJECT_PLACEMENT_RETRIES = 150
MIN_CANVAS_SIDE = 18
MAX_CANVAS_SIDE = 30
FRAME_PADDING = 2
PIECE_PADDING = 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    shape = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return shape, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    used_colors = set()
    for coords in _components(grid == AZURE):
        r0, c0, r1, c1 = _bbox(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 3 and width >= 3, "every frame must have a non-empty interior"
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _frame_border_mask(height, width)
        assert np.all(sub[border] == AZURE), "azure component must be a full rectangular border"
        assert int(np.count_nonzero(sub == AZURE)) == int(border.sum()), "frame border cannot contain extra azure cells"
        inner = sub[1:-1, 1:-1]
        non_bg = np.argwhere((inner != BLACK) & (inner != AZURE))
        assert len(non_bg) == 1, "each frame interior must contain exactly one colored anchor"
        assert int(np.count_nonzero(inner != BLACK)) == 1, "frame interior can only contain the anchor"
        ar, ac = map(int, non_bg[0])
        color = int(inner[ar, ac])
        assert color not in used_colors, "anchor colors must be unique per frame"
        used_colors.add(color)
        frames.append(
            {
                "color": color,
                "outer_bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
                "inner_shape": (int(height - 2), int(width - 2)),
                "anchor": (int(ar), int(ac)),
            }
        )
    assert frames, "expected at least one assembly frame"

    occ = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["outer_bbox"]
        assert not occ[r0 : r1 + 1, c0 : c1 + 1].any(), "frames cannot overlap"
        occ[r0 : r1 + 1, c0 : c1 + 1] = True
    return sorted(frames, key=lambda f: (f["outer_bbox"][0], f["outer_bbox"][1]))


def _outside_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["outer_bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _collect_pieces(grid: np.ndarray, frames: list[dict]) -> dict[int, list[dict]]:
    outside = _outside_mask(grid.shape, frames)
    frame_colors = {int(frame["color"]) for frame in frames}
    stray = {int(v) for v in np.unique(grid[outside]) if int(v) != BLACK}
    assert stray <= frame_colors, "every outside fragment color must match some anchor color"

    pieces_by_color = {color: [] for color in frame_colors}
    for color in frame_colors:
        mask = (grid == color) & outside
        for coords in _components(mask):
            shape, box_h, box_w = _normalize(coords)
            pieces_by_color[color].append(
                {
                    "shape": shape,
                    "bbox": (int(box_h), int(box_w)),
                    "area": int(len(coords)),
                    "coords_global": tuple(sorted(coords)),
                }
            )
    return pieces_by_color


def _candidate_placements(
    piece: dict,
    box_h: int,
    box_w: int,
    anchor: tuple[int, int],
) -> list[dict]:
    shape = piece["shape"]
    piece_h, piece_w = piece["bbox"]
    placements = []
    for top in range(box_h - piece_h + 1):
        for left in range(box_w - piece_w + 1):
            cells = []
            bitmask = 0
            ok = True
            for pr, pc in shape:
                rr = int(top + pr)
                cc = int(left + pc)
                if (rr, cc) == anchor:
                    ok = False
                    break
                idx = int(rr * box_w + cc)
                bitmask |= 1 << idx
                cells.append((rr, cc))
            if ok:
                placements.append(
                    {
                        "top": int(top),
                        "left": int(left),
                        "mask": bitmask,
                        "cells": tuple(cells),
                    }
                )
    return placements


def _solve_frame(box_h: int, box_w: int, anchor: tuple[int, int], pieces: list[dict]) -> list[dict]:
    needed_mask = 0
    for r in range(box_h):
        for c in range(box_w):
            if (r, c) != anchor:
                needed_mask |= 1 << int(r * box_w + c)

    need_area = int(box_h * box_w - 1)
    piece_area = sum(int(piece["area"]) for piece in pieces)
    assert piece_area == need_area, "fragments plus anchor must cover the full inner rectangle"
    if not pieces:
        assert need_area == 0, "missing fragments for a non-trivial frame"
        return []

    options = []
    for piece in pieces:
        placements = _candidate_placements(piece, box_h, box_w, anchor)
        assert placements, "every fragment must have at least one in-frame translation"
        options.append(placements)

    assigned = [None] * len(pieces)
    first_solution = None
    solution_signatures = set()

    def dfs(remaining_mask: int, remaining_ids: tuple[int, ...]) -> bool:
        nonlocal first_solution
        if not remaining_ids:
            if remaining_mask == 0:
                signature = tuple(
                    sorted((pieces[i]["shape"], assigned[i]["top"], assigned[i]["left"]) for i in range(len(pieces)))
                )
                if signature not in solution_signatures:
                    solution_signatures.add(signature)
                    if first_solution is None:
                        first_solution = tuple(assigned)
                return len(solution_signatures) >= 2
            return False

        best_id = None
        best_fits = None
        for piece_id in remaining_ids:
            fits = [placement for placement in options[piece_id] if placement["mask"] & remaining_mask == placement["mask"]]
            if not fits:
                return False
            if best_fits is None or len(fits) < len(best_fits):
                best_id = piece_id
                best_fits = fits
                if len(best_fits) == 1:
                    break

        next_ids = tuple(piece_id for piece_id in remaining_ids if piece_id != best_id)
        for placement in best_fits:
            assigned[best_id] = placement
            if dfs(remaining_mask ^ placement["mask"], next_ids):
                return True
            assigned[best_id] = None
        return False

    dfs(needed_mask, tuple(range(len(pieces))))
    assert len(solution_signatures) == 1 and first_solution is not None, "fragment assembly must be uniquely determined"
    return [dict(placement) for placement in first_solution]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move same-color fragments into their matching azure assembly frames to reconstruct solid rectangles.

    Input structure:
    - The background is black.
    - There are one or more azure rectangular frames with 1-cell-thick borders and black interiors.
    - Each frame interior contains exactly one colored anchor pixel, and all anchor colors are distinct.
    - Outside the frames, fragments are 4-connected monochrome shapes whose colors match one anchor each.

    Rule:
    - For each anchor color, collect every outside fragment of that color and translate those fragments without rotation or reflection.
    - Place the translated fragments inside the frame whose anchor has that color so that the anchor and the moved fragments exactly fill the frame interior as a solid rectangle.
    - Erase the outside fragments after moving them, while keeping every azure frame border unchanged.

    Key insight:
    - Color links each fragment group to one frame, and the fixed anchor location turns the reassembly into a unique exact-cover packing problem inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    frames = _find_frames(g)
    pieces_by_color = _collect_pieces(g, frames)

    out = g.copy()
    for frame in frames:
        color = int(frame["color"])
        box_h, box_w = frame["inner_shape"]
        anchor = frame["anchor"]
        pieces = pieces_by_color.get(color, [])
        placements = _solve_frame(box_h, box_w, anchor, pieces)

        for piece in pieces:
            for r, c in piece["coords_global"]:
                out[r, c] = BLACK

        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BLACK
        out[ir0 + anchor[0], ic0 + anchor[1]] = color
        for piece, placement in zip(pieces, placements):
            for pr, pc in piece["shape"]:
                rr = int(ir0 + placement["top"] + pr)
                cc = int(ic0 + placement["left"] + pc)
                out[rr, cc] = color

    return out


def _partition_sizes(area: int, piece_count: int) -> list[int]:
    if piece_count == 1:
        return [area]
    sizes = [2] * piece_count
    for _ in range(area - 2 * piece_count):
        sizes[random.randrange(piece_count)] += 1
    random.shuffle(sizes)
    return sizes


def _partition_rectangle(inner_h: int, inner_w: int, anchor: tuple[int, int], piece_count: int) -> list[set[tuple[int, int]]] | None:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) != anchor]
    if piece_count == 1:
        return [set(cells)]
    area = len(cells)
    sizes = _partition_sizes(area, piece_count)
    all_cells = set(cells)

    for _ in range(PARTITION_RETRIES):
        seeds = random.sample(cells, piece_count)
        pieces = [set([seed]) for seed in seeds]
        remaining = set(all_cells) - set(seeds)

        while remaining:
            expandable = []
            for i, piece in enumerate(pieces):
                if len(piece) >= sizes[i]:
                    continue
                frontier = set()
                for r, c in piece:
                    for dr, dc in NEIGHBORS4:
                        nbr = (r + dr, c + dc)
                        if nbr in remaining:
                            frontier.add(nbr)
                if frontier:
                    expandable.append((i, tuple(frontier)))
            if not expandable:
                break

            best_gap = max(sizes[i] - len(pieces[i]) for i, _ in expandable)
            candidates = [(i, frontier) for i, frontier in expandable if sizes[i] - len(pieces[i]) == best_gap]
            piece_id, frontier = random.choice(candidates)
            cell = random.choice(frontier)
            pieces[piece_id].add(cell)
            remaining.remove(cell)

        if remaining:
            continue
        if any(len(piece) != sizes[i] for i, piece in enumerate(pieces)):
            continue
        return pieces
    return None


def _make_frame_spec(color: int) -> dict | None:
    for _ in range(FRAME_SPEC_RETRIES):
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        anchor = (random.randrange(inner_h), random.randrange(inner_w))
        area = inner_h * inner_w - 1
        max_pieces = min(MAX_PIECES_PER_FRAME, max(1, area // 2))
        if area <= 4:
            piece_count = 1
        else:
            piece_count = random.randint(1, max_pieces)

        piece_sets = _partition_rectangle(inner_h, inner_w, anchor, piece_count)
        if piece_sets is None:
            continue

        pieces = []
        for cell_set in piece_sets:
            shape, box_h, box_w = _normalize(list(cell_set))
            pieces.append({"shape": shape, "bbox": (int(box_h), int(box_w)), "area": int(len(shape))})

        try:
            _solve_frame(inner_h, inner_w, anchor, pieces)
        except AssertionError:
            continue

        return {
            "color": int(color),
            "inner_shape": (int(inner_h), int(inner_w)),
            "anchor": anchor,
            "pieces": pieces,
        }
    return None


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    grid_h, grid_w = blocked.shape
    if top < 0 or left < 0 or top + height > grid_h or left + width > grid_w:
        return False
    return not bool(blocked[top : top + height, left : left + width].any())


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    grid_h, grid_w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(grid_h, top + height + pad)
    c1 = min(grid_w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _draw_frame(grid: np.ndarray, top: int, left: int, frame: dict, assembled: bool) -> None:
    inner_h, inner_w = frame["inner_shape"]
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top : top + outer_h, left] = AZURE
    grid[top : top + outer_h, left + outer_w - 1] = AZURE
    grid[top, left : left + outer_w] = AZURE
    grid[top + outer_h - 1, left : left + outer_w] = AZURE

    ir0 = top + 1
    ic0 = left + 1
    if assembled:
        grid[ir0 : ir0 + inner_h, ic0 : ic0 + inner_w] = frame["color"]
    else:
        grid[ir0 : ir0 + inner_h, ic0 : ic0 + inner_w] = BLACK
        ar, ac = frame["anchor"]
        grid[ir0 + ar, ic0 + ac] = frame["color"]


def _draw_piece(grid: np.ndarray, top: int, left: int, color: int, piece: dict) -> None:
    for dr, dc in piece["shape"]:
        grid[top + dr, left + dc] = color


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, frame_count)

    frames = []
    for color in colors:
        frame = _make_frame_spec(color)
        if frame is None:
            return None
        frames.append(frame)

    if not any(len(frame["pieces"]) >= 2 for frame in frames):
        return None

    total_bbox_area = 0
    tallest = 0
    widest = 0
    for frame in frames:
        inner_h, inner_w = frame["inner_shape"]
        total_bbox_area += (inner_h + 2) * (inner_w + 2)
        tallest = max(tallest, inner_h + 2)
        widest = max(widest, inner_w + 2)
        for piece in frame["pieces"]:
            piece_h, piece_w = piece["bbox"]
            total_bbox_area += piece_h * piece_w
            tallest = max(tallest, piece_h)
            widest = max(widest, piece_w)

    side_floor = max(MIN_CANVAS_SIDE, int(np.ceil(np.sqrt(total_bbox_area * 2.4))))
    min_h = max(side_floor, tallest + 4)
    min_w = max(side_floor, widest + 4)
    if min_h > MAX_CANVAS_SIDE or min_w > MAX_CANVAS_SIDE:
        return None

    canvas_h = random.randint(min_h, MAX_CANVAS_SIDE)
    canvas_w = random.randint(min_w, MAX_CANVAS_SIDE)
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)

    frame_order = sorted(range(len(frames)), key=lambda i: -((frames[i]["inner_shape"][0] + 2) * (frames[i]["inner_shape"][1] + 2)))
    placed_frames = []
    for idx in frame_order:
        frame = frames[idx]
        inner_h, inner_w = frame["inner_shape"]
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        spot = None
        for _ in range(OBJECT_PLACEMENT_RETRIES):
            top = random.randint(0, canvas_h - outer_h)
            left = random.randint(0, canvas_w - outer_w)
            if _can_place(blocked, top, left, outer_h, outer_w):
                spot = (top, left)
                _mark_blocked(blocked, top, left, outer_h, outer_w, FRAME_PADDING)
                break
        if spot is None:
            return None
        placed_frames.append((idx, spot[0], spot[1]))

    output = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    input_grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    frames_with_positions = [None] * len(frames)
    for idx, top, left in placed_frames:
        frame = dict(frames[idx])
        frame["outer_top"] = int(top)
        frame["outer_left"] = int(left)
        frames_with_positions[idx] = frame
        _draw_frame(output, top, left, frame, assembled=True)
        _draw_frame(input_grid, top, left, frame, assembled=False)

    piece_jobs = []
    for frame in frames_with_positions:
        for piece in frame["pieces"]:
            piece_jobs.append((piece["bbox"][0] * piece["bbox"][1], frame["color"], piece))
    piece_jobs = sorted(piece_jobs, key=lambda item: (-item[0], random.random()))

    for _area, color, piece in piece_jobs:
        piece_h, piece_w = piece["bbox"]
        spot = None
        for _ in range(OBJECT_PLACEMENT_RETRIES):
            top = random.randint(0, canvas_h - piece_h)
            left = random.randint(0, canvas_w - piece_w)
            if _can_place(blocked, top, left, piece_h, piece_w):
                spot = (top, left)
                _mark_blocked(blocked, top, left, piece_h, piece_w, PIECE_PADDING)
                break
        if spot is None:
            return None
        _draw_piece(input_grid, spot[0], spot[1], color, piece)

    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(CASE_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        try:
            got = main(inp)
        except AssertionError:
            continue
        if np.array_equal(got, expected):
            return inp
    raise RuntimeError("failed to generate a valid assembly-frame input")
