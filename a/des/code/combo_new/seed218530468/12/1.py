import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CANVAS_SIZE = 30
BG = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300
PLACE_ATTEMPTS = 200

OPENING_TO_CELLS = {
    "top_left": ((0, 1), (1, 0), (1, 1)),
    "top_right": ((0, 0), (1, 0), (1, 1)),
    "bottom_left": ((0, 0), (0, 1), (1, 1)),
    "bottom_right": ((0, 0), (0, 1), (1, 0)),
}
LOCAL_TO_OPENING = {
    frozenset(cells): opening for opening, cells in OPENING_TO_CELLS.items()
}
OPPOSITE_CORNER = {
    "top_left": "bottom_right",
    "top_right": "bottom_left",
    "bottom_left": "top_right",
    "bottom_right": "top_left",
}
FRAME_RING_RELS = tuple(
    (r, c)
    for r in range(4)
    for c in range(4)
    if r in (0, 3) or c in (0, 3)
)
MARKER_OFFSET = {
    "top_left": (-1, -1),
    "top_right": (-1, 4),
    "bottom_left": (4, -1),
    "bottom_right": (4, 4),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if seen[sr, sc] or color == bg:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = sr
            c0 = c1 = sc

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                r0 = min(r0, r)
                r1 = max(r1, r)
                c0 = min(c0, c)
                c1 = max(c1, c)
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if not _in_bounds(h, w, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return comps


def _is_frame_component(comp: dict) -> bool:
    if comp["area"] != 12:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (4, 4):
        return False

    pts = set(comp["coords"])
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            should_be_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in pts) != should_be_border:
                return False
    return True


def _l_opening(comp: dict) -> str | None:
    if comp["area"] != 3:
        return None
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None
    local = frozenset((r - r0, c - c0) for r, c in comp["coords"])
    return LOCAL_TO_OPENING.get(local)


def _find_frames(
    grid: np.ndarray, comps: list[dict], singleton_colors: dict[tuple[int, int], int]
) -> list[dict]:
    frames: list[dict] = []

    for comp in comps:
        if not _is_frame_component(comp):
            continue
        r0, c0, _, _ = comp["bbox"]
        candidates: list[tuple[str, int]] = []
        for corner, (dr, dc) in MARKER_OFFSET.items():
            pos = (r0 + dr, c0 + dc)
            if pos in singleton_colors and singleton_colors[pos] != int(comp["color"]):
                candidates.append((corner, int(singleton_colors[pos])))
        assert len(candidates) == 1, "each frame must have exactly one diagonal marker"
        marker_corner, marker_color = candidates[0]
        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "marker_corner": marker_corner,
                "marker_color": marker_color,
            }
        )

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored L fragments with marker-tagged frames and complete each frame's inner square.

    Input structure:
    - Black is the background.
    - Several monochrome 3-pixel L fragments are scattered on the grid; each color appears on exactly two complementary L shapes.
    - Several hollow 4x4 frame components appear elsewhere, each with one single-pixel colored marker on the diagonal outside one corner.

    Rule:
    - Group the scattered L fragments by color.
    - Match each frame to the L pair whose color equals the frame's diagonal marker color.
    - Use the marker corner to identify the complementary orientation pair, then fill the frame's 2x2 interior with that color.
    - Erase the original scattered L fragments to black while keeping the frame borders and diagonal marker pixels unchanged.

    Key insight:
    - The colored marker binds a frame to one color pair and its corner position identifies which complementary L orientation belongs to that frame, but the visible output is the completed 2x2 square inside the matched frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = _find_components(g, bg)
    singletons = {
        comp["coords"][0]: int(comp["color"])
        for comp in comps
        if comp["area"] == 1
    }
    frames = _find_frames(g, comps, singletons)
    assert frames, "at least one frame is required"

    l_by_color: dict[int, list[dict]] = defaultdict(list)
    l_pieces: list[dict] = []
    for comp in comps:
        opening = _l_opening(comp)
        if opening is None:
            continue
        piece = {
            "color": int(comp["color"]),
            "coords": comp["coords"],
            "opening": opening,
        }
        l_pieces.append(piece)
        l_by_color[piece["color"]].append(piece)

    assert l_pieces, "at least one L fragment pair is required"
    assert len(frames) == len(l_by_color), "frames and L-color groups must match"

    out = g.copy()
    for piece in l_pieces:
        for r, c in piece["coords"]:
            out[r, c] = bg

    seen_marker_colors: set[int] = set()
    for frame in frames:
        marker_color = int(frame["marker_color"])
        assert marker_color not in seen_marker_colors, "marker colors must be unique per frame"
        seen_marker_colors.add(marker_color)

        pair = l_by_color[marker_color]
        assert len(pair) == 2, "each marker color must match exactly two L fragments"
        openings = {piece["opening"] for piece in pair}
        needed = {frame["marker_corner"], OPPOSITE_CORNER[frame["marker_corner"]]}
        assert openings == needed, "same-color L pair must use the marker corner and its opposite"

        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = marker_color

    return out


def _shape_cells(top: int, left: int, rels: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in rels]


def _reserve(blocked: np.ndarray, cells: list[tuple[int, int]], margin: int = 1) -> None:
    h, w = blocked.shape
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = max(0, min(rs) - margin)
    r1 = min(h - 1, max(rs) + margin)
    c0 = max(0, min(cs) - margin)
    c1 = min(w - 1, max(cs) + margin)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _can_place(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not _in_bounds(h, w, r, c):
            return False
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = max(0, min(rs) - 1)
    r1 = min(h - 1, max(rs) + 1)
    c0 = max(0, min(cs) - 1)
    c1 = min(w - 1, max(cs) + 1)
    return not np.any(blocked[r0 : r1 + 1, c0 : c1 + 1])


def _paint(grid: np.ndarray, blocked: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color
    _reserve(blocked, cells)


def _try_place_frame(
    grid: np.ndarray, blocked: np.ndarray, color: int, corner: str, rng: random.Random
) -> bool:
    rels = FRAME_RING_RELS + (MARKER_OFFSET[corner],)
    min_r = min(dr for dr, _ in rels)
    max_r = max(dr for dr, _ in rels)
    min_c = min(dc for _, dc in rels)
    max_c = max(dc for _, dc in rels)

    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(-min_r, CANVAS_SIZE - 1 - max_r)
        left = rng.randint(-min_c, CANVAS_SIZE - 1 - max_c)
        all_cells = _shape_cells(top, left, rels)
        if not _can_place(blocked, all_cells):
            continue

        ring_cells = _shape_cells(top, left, FRAME_RING_RELS)
        marker_cell = [(top + MARKER_OFFSET[corner][0], left + MARKER_OFFSET[corner][1])]
        _paint(grid, blocked, ring_cells, FRAME_COLOR)
        for r, c in marker_cell:
            grid[r, c] = color
        _reserve(blocked, ring_cells + marker_cell)
        return True

    return False


def _try_place_l(
    grid: np.ndarray, blocked: np.ndarray, color: int, opening: str, rng: random.Random
) -> bool:
    rels = OPENING_TO_CELLS[opening]
    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(0, CANVAS_SIZE - 2)
        left = rng.randint(0, CANVAS_SIZE - 2)
        cells = _shape_cells(top, left, rels)
        if not _can_place(blocked, cells):
            continue
        _paint(grid, blocked, cells, color)
        return True
    return False


def _crop_to_content(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_extra = rng.randint(0, min(3, int(r0)))
    left_extra = rng.randint(0, min(3, int(c0)))
    bottom_extra = rng.randint(0, min(3, int(grid.shape[0] - 1 - r1)))
    right_extra = rng.randint(0, min(3, int(grid.shape[1] - 1 - c1)))

    return grid[
        int(r0) - top_extra : int(r1) + bottom_extra + 1,
        int(c0) - left_extra : int(c1) + right_extra + 1,
    ].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    color_count = rng.choice((2, 3, 4))
    colors = rng.sample(PALETTE, color_count)
    corners = list(OPENING_TO_CELLS)
    rng.shuffle(corners)
    chosen_corners = corners[:color_count]
    specs = list(zip(colors, chosen_corners))
    rng.shuffle(specs)

    for color, corner in specs:
        if not _try_place_frame(grid, blocked, color, corner, rng):
            return None

    l_specs = []
    for color, corner in specs:
        l_specs.append((color, corner))
        l_specs.append((color, OPPOSITE_CORNER[corner]))
    rng.shuffle(l_specs)

    for color, opening in l_specs:
        if not _try_place_l(grid, blocked, color, opening, rng):
            return None

    cropped = _crop_to_content(grid, rng)
    try:
        out = main(cropped)
    except AssertionError:
        return None

    assert cropped.ndim == 2 and out.ndim == 2
    assert max(cropped.shape) <= 30 and max(out.shape) <= 30
    return cropped


def generate_input() -> np.ndarray:
    rng = random
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
