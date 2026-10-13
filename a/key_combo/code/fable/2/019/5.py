from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BLACK = int(Color.BLACK)
OBSTACLE = int(Color.GRAY)
PIECE_PALETTE = [int(c) for c in Color.ALL_COLORS if int(c) not in (BLACK, OBSTACLE)]
ORIENTATIONS = ("left", "right", "top", "bottom")
MAX_GENERATION_ATTEMPTS = 400


def _largest_empty_rectangles(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Return the maximum empty-rectangle area and every max-area bbox."""
    empty = grid == int(bg)
    height, width = map(int, empty.shape)
    best_area = 0
    best_rects: list[tuple[int, int, int, int]] = []

    for r0 in range(height):
        good = np.ones(width, dtype=bool)
        for r1 in range(r0, height):
            good &= empty[r1]
            rect_h = r1 - r0 + 1
            if rect_h < 2:
                continue

            c = 0
            while c < width:
                if not good[c]:
                    c += 1
                    continue
                start = c
                while c < width and good[c]:
                    c += 1
                end = c - 1
                rect_w = end - start + 1
                if rect_w < 2:
                    continue

                area = rect_h * rect_w
                rect = (r0, start, r1, end)
                if area > best_area:
                    best_area = area
                    best_rects = [rect]
                elif area == best_area:
                    best_rects.append(rect)

    return best_area, best_rects


def _extract_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    """Extract the scattered colored rectangles in source reading order."""
    colors = [int(c) for c in np.unique(grid) if int(c) not in (int(bg), OBSTACLE)]
    components = find_components(
        grid,
        bg=int(bg),
        colors=colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert components, "expected at least one colored rectangle"

    pieces = []
    for component in components:
        r0, c0, r1, c1 = map(int, component["bbox"])
        color = int(component["color"])
        patch = crop_bbox(grid, component["bbox"])
        assert np.all(patch == color), "every colored object must be a solid rectangle"
        pieces.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "shape": tuple(map(int, patch.shape)),
                "area": int(component["area"]),
            }
        )

    pieces.sort(key=lambda piece: (piece["bbox"][0], piece["bbox"][1], piece["shape"][0], piece["shape"][1]))
    order_keys = [(piece["bbox"][0], piece["bbox"][1]) for piece in pieces]
    assert len(set(order_keys)) == len(order_keys), "source rectangles need a unique reading order"
    return pieces


def _pack_by_source_order(container_shape: tuple[int, int], pieces: list[dict], bg: int) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """
    Fill the cavity in source reading order.

    At each step, the next source rectangle starts at the current top-leftmost
    empty cell of the cavity. This visible order anchor removes the otherwise
    unavoidable 180-degree ambiguity of pure rectangle tilings.
    """
    height, width = map(int, container_shape)
    patch = np.full((height, width), int(bg), dtype=int)
    filled = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int, int, int, int]] = []

    for piece in pieces:
        empty_coords = np.argwhere(~filled)
        assert empty_coords.size > 0, "container filled too early"
        r0, c0 = map(int, empty_coords[0])
        rect_h, rect_w = map(int, piece["shape"])
        assert r0 + rect_h <= height and c0 + rect_w <= width, "piece sequence exceeds the cavity bounds"
        assert not bool(filled[r0 : r0 + rect_h, c0 : c0 + rect_w].any()), "piece sequence overlaps inside the cavity"
        patch[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(piece["color"])
        filled[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        placements.append((r0, c0, rect_h, rect_w, int(piece["color"])))

    assert bool(filled.all()), "piece sequence must tile the cavity without holes"
    return patch, placements


def _solve(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == BLACK, "the task uses black background"

    pieces = _extract_pieces(g, bg)
    cavity_area, cavities = _largest_empty_rectangles(g, bg)
    assert len(cavities) == 1, "expected a unique largest empty rectangle"
    cavity = cavities[0]
    r0, c0, r1, c1 = cavity
    assert cavity_area == sum(piece["area"] for piece in pieces), "piece area must equal cavity area"

    packed, placements = _pack_by_source_order((r1 - r0 + 1, c1 - c0 + 1), pieces, bg)
    out = g.copy()
    for piece in pieces:
        pr0, pc0, pr1, pc1 = piece["bbox"]
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = int(bg)
    paste(out, packed, r0, c0)
    return out, cavity, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the scattered solid rectangles into the hidden largest empty rectangle, preserving source reading order.

    Input structure:
    - A black background contains gray obstacle pixels and several non-gray solid-colored rectangles.
    - Treating both the obstacles and the colored rectangles as occupied, exactly one all-black rectangle with height and width at least 2 has the maximum area.
    - The source rectangles are separated, and their top-left corners define a unique row-major order.

    Rule:
    - Find the unique largest all-black rectangle.
    - Erase every colored rectangle from its source position.
    - Sort the source rectangles by their top-left coordinates.
    - Fill the hidden rectangle in that order: each next rectangle is placed with pure translation so its top-left corner lands on the current top-leftmost empty cavity cell.
    - Leave every gray obstacle and every other background cell unchanged.

    Key insight:
    - The container is an implicit maximal black cavity, and the scattered rectangles themselves provide the visible ordering cue that resolves the otherwise symmetric rectangle tiling.
    """
    out, _cavity, _placements = _solve(grid)
    return out


def _composition(total: int, parts: int) -> list[int]:
    """Uniform composition sampler used for spreading source pieces with variable gaps."""
    total = int(total)
    parts = int(parts)
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    cuts = [-1] + cuts + [total + parts - 1]
    return [right - left - 1 for left, right in zip(cuts, cuts[1:])]


def _random_partition(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    """Randomly partition a rectangle into smaller rectangles sorted in row-major order."""
    rects = [(0, 0, int(height), int(width))]
    split_orients: list[str] = []

    while len(rects) < int(count):
        candidates = []
        for idx, (r0, c0, rect_h, rect_w) in enumerate(rects):
            for cut in range(1, rect_h):
                top = (r0, c0, cut, rect_w)
                bottom = (r0 + cut, c0, rect_h - cut, rect_w)
                if top[2] * top[3] >= 2 and bottom[2] * bottom[3] >= 2:
                    candidates.append((idx, "H", top, bottom))
            for cut in range(1, rect_w):
                left = (r0, c0, rect_h, cut)
                right = (r0, c0 + cut, rect_h, rect_w - cut)
                if left[2] * left[3] >= 2 and right[2] * right[3] >= 2:
                    candidates.append((idx, "V", left, right))

        if not candidates:
            return None
        idx, orient, first, second = random.choice(candidates)
        rects = rects[:idx] + [first, second] + rects[idx + 1 :]
        split_orients.append(orient)

    rects.sort(key=lambda rect: (rect[0], rect[1], rect[2], rect[3]))
    dims = [(rect_h, rect_w) for _r0, _c0, rect_h, rect_w in rects]
    if len(set(dims)) != len(dims):
        return None
    if len({r0 for r0, _c0, _h, _w in rects}) < 2 or len({c0 for _r0, c0, _h, _w in rects}) < 2:
        return None
    if len({rect_h for rect_h, _rect_w in dims}) < 2 or len({rect_w for _rect_h, rect_w in dims}) < 2:
        return None
    if not any(rect_h < height and rect_w < width for _r0, _c0, rect_h, rect_w in rects):
        return None
    if count >= 4 and ("H" not in split_orients or "V" not in split_orients):
        return None
    return rects


def _place_source_pieces(
    grid: np.ndarray,
    slot_rects: list[tuple[int, int, int, int]],
    colors: list[int],
    orientation: str,
) -> bool:
    """Paint the source rectangles in a clean row-major order outside the cavity."""
    height, width = map(int, grid.shape)
    count = len(slot_rects)

    if orientation in {"left", "right"}:
        max_piece_w = max(rect_w for _r0, _c0, _rect_h, rect_w in slot_rects)
        band_w = max_piece_w + 2
        used_h = sum(rect_h for _r0, _c0, rect_h, _rect_w in slot_rects) + (count - 1)
        extra = (height - 2) - used_h
        if extra < 0:
            return False
        gaps = _composition(extra, count + 1)
        divider_col = band_w if orientation == "left" else width - band_w - 1
        source_col = 1 if orientation == "left" else divider_col + 2
        grid[:, divider_col] = OBSTACLE

        row = 1 + gaps[0]
        for idx, ((_, _, rect_h, rect_w), color) in enumerate(zip(slot_rects, colors)):
            grid[row : row + rect_h, source_col : source_col + rect_w] = int(color)
            row += rect_h + 1 + gaps[idx + 1]
        return True

    max_piece_h = max(rect_h for _r0, _c0, rect_h, _rect_w in slot_rects)
    band_h = max_piece_h + 2
    used_w = sum(rect_w for _r0, _c0, _rect_h, rect_w in slot_rects) + (count - 1)
    extra = (width - 2) - used_w
    if extra < 0:
        return False
    gaps = _composition(extra, count + 1)
    divider_row = band_h if orientation == "top" else height - band_h - 1
    source_row = 1 if orientation == "top" else divider_row + 2
    grid[divider_row, :] = OBSTACLE

    col = 1 + gaps[0]
    for idx, ((_, _, rect_h, rect_w), color) in enumerate(zip(slot_rects, colors)):
        grid[source_row : source_row + rect_h, col : col + rect_w] = int(color)
        col += rect_w + 1 + gaps[idx + 1]
    return True


def _build_layout() -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int], tuple[int, int, int, int]] | None:
    """Sample one raw scene before obstacle refinement."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        piece_count = random.randint(3, 5)
        cavity_h = random.randint(4, 8)
        cavity_w = random.randint(4, 8)
        slot_rects = _random_partition(cavity_h, cavity_w, piece_count)
        if slot_rects is None:
            continue

        orientation = random.choice(ORIENTATIONS)
        max_piece_h = max(rect_h for _r0, _c0, rect_h, _rect_w in slot_rects)
        max_piece_w = max(rect_w for _r0, _c0, _rect_h, rect_w in slot_rects)

        if orientation in {"left", "right"}:
            band_w = max_piece_w + 2
            min_h = max(cavity_h + 2, sum(rect_h for _r0, _c0, rect_h, _rect_w in slot_rects) + piece_count + 1)
            grid_h = min_h + random.randint(0, 4)
            grid_w = band_w + 2 + cavity_w + random.randint(2, 5)
        else:
            band_h = max_piece_h + 2
            grid_h = band_h + 2 + cavity_h + random.randint(2, 5)
            min_w = max(cavity_w + 2, sum(rect_w for _r0, _c0, _rect_h, rect_w in slot_rects) + piece_count + 1)
            grid_w = min_w + random.randint(0, 4)

        if max(grid_h, grid_w) > 30:
            continue

        grid = np.full((grid_h, grid_w), BLACK, dtype=int)
        colors = random.sample(PIECE_PALETTE, piece_count)
        if not _place_source_pieces(grid, slot_rects, colors, orientation):
            continue

        if orientation == "left":
            container_col = max_piece_w + 4
            if container_col + cavity_w > grid_w - 1:
                continue
            container_row = random.randint(1, grid_h - cavity_h - 1)
        elif orientation == "right":
            divider_col = grid_w - (max_piece_w + 2) - 1
            if divider_col - cavity_w - 1 < 1:
                continue
            container_col = random.randint(1, divider_col - cavity_w - 1)
            container_row = random.randint(1, grid_h - cavity_h - 1)
        elif orientation == "top":
            container_row = max_piece_h + 4
            if container_row + cavity_h > grid_h - 1:
                continue
            container_col = random.randint(1, grid_w - cavity_w - 1)
        else:
            divider_row = grid_h - (max_piece_h + 2) - 1
            if divider_row - cavity_h - 1 < 1:
                continue
            container_row = random.randint(1, divider_row - cavity_h - 1)
            container_col = random.randint(1, grid_w - cavity_w - 1)

        cavity = (container_row, container_col, container_row + cavity_h - 1, container_col + cavity_w - 1)
        if np.any(grid[cavity[0] : cavity[2] + 1, cavity[1] : cavity[3] + 1] != BLACK):
            continue
        return grid, slot_rects, colors, cavity

    return None


def _ensure_unique_container(inp: np.ndarray, cavity: tuple[int, int, int, int], bg: int) -> bool:
    """Add gray blockers until the sampled cavity is the unique largest empty rectangle."""
    cr0, cc0, cr1, cc1 = cavity
    target_area = (cr1 - cr0 + 1) * (cc1 - cc0 + 1)
    protected = inp != int(bg)
    protected[cr0 : cr1 + 1, cc0 : cc1 + 1] = True
    height, width = map(int, inp.shape)

    for _ in range(250):
        best_area, rects = _largest_empty_rectangles(inp, bg)
        if best_area == target_area and len(rects) == 1 and rects[0] == cavity:
            return True
        if best_area < target_area:
            return False

        bad_rects = [rect for rect in rects if best_area > target_area or rect != cavity]
        if not bad_rects:
            return False

        r0, c0, r1, c1 = max(
            bad_rects,
            key=lambda rect: ((rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1), rect[2] - rect[0], rect[3] - rect[1]),
        )
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        candidates = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if inp[r, c] != int(bg):
                    continue
                if protected[r, c]:
                    continue
                if cr0 <= r <= cr1 and cc0 <= c <= cc1:
                    continue
                near_fg = bool(
                    np.any(
                        inp[max(0, r - 1) : min(height, r + 2), max(0, c - 1) : min(width, c + 2)] != int(bg)
                    )
                )
                dist = abs(r - center_r) + abs(c - center_c)
                candidates.append((near_fg, dist, r, c))

        if not candidates:
            return False
        candidates.sort()
        _near_fg, _dist, rr, cc = candidates[0]
        inp[rr, cc] = OBSTACLE
        protected[rr, cc] = True

    return False


def _expected_output_from_layout(
    inp: np.ndarray,
    slot_rects: list[tuple[int, int, int, int]],
    colors: list[int],
    cavity: tuple[int, int, int, int],
    bg: int,
) -> np.ndarray:
    """Construct the semantic target independently of the solver."""
    out = inp.copy()
    for piece in _extract_pieces(inp, bg):
        r0, c0, r1, c1 = piece["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    for (slot_r, slot_c, rect_h, rect_w), color in zip(slot_rects, colors):
        out[cavity[0] + slot_r : cavity[0] + slot_r + rect_h, cavity[1] + slot_c : cavity[1] + slot_c + rect_w] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    built = _build_layout()
    if built is None:
        return None
    inp, slot_rects, colors, cavity = built
    bg = BLACK

    candidate_cells = [
        (r, c)
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if inp[r, c] == bg and not (cavity[0] <= r <= cavity[2] and cavity[1] <= c <= cavity[3])
    ]
    random.shuffle(candidate_cells)
    for r, c in candidate_cells[: random.randint(1, 4)]:
        inp[r, c] = OBSTACLE

    if not _ensure_unique_container(inp, cavity, bg):
        return None
    if int(get_bg(inp)) != bg:
        return None

    expected = _expected_output_from_layout(inp, slot_rects, colors, cavity, bg)
    solved, found_cavity, _placements = _solve(inp)
    if found_cavity != cavity:
        return None
    if not np.array_equal(solved, expected):
        return None

    area, rects = _largest_empty_rectangles(inp, bg)
    if area != (cavity[2] - cavity[0] + 1) * (cavity[3] - cavity[1] + 1):
        return None
    if rects != [cavity]:
        return None

    pieces = _extract_pieces(inp, bg)
    if sum(piece["area"] for piece in pieces) != area:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-container puzzle")
