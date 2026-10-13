import random
from collections import deque

import numpy as np


BG = 0
GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 4)
PLACEMENT_PAD = 1
GEN_ATTEMPTS = 500
PALETTE = tuple(range(1, 10))

PATTERN_LIBRARY = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangle_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or len(coords) != 2 * (h + w) - 4:
        return False
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    for rr in range(h):
        for cc in range(w):
            on_perimeter = rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1
            if ((rr, cc) in rel) != on_perimeter:
                return False
    return True


def _pattern_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    anchor = tuple(map(int, np.argwhere(mask)[0]))
    return mask, anchor


def _parse_scene(grid: np.ndarray):
    bg = _get_bg(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    frames: dict[int, dict] = {}
    marker_colors = set()

    for color in colors:
        components = _components_of_color(grid, color)
        frame_indices = [idx for idx, comp in enumerate(components) if _is_rectangle_frame(comp)]
        if not frame_indices:
            continue
        assert len(frame_indices) == 1, "each frame color must have exactly one rectangular border"
        frame_index = frame_indices[0]
        frame_coords = components[frame_index]
        pattern_components = [comp for idx, comp in enumerate(components) if idx != frame_index]
        assert len(pattern_components) == 1, "each frame color must have exactly one matching outside pattern"
        pattern_coords = pattern_components[0]

        fr0, fc0, fr1, fc1 = _bbox(frame_coords)
        assert fr1 - fr0 >= 2 and fc1 - fc0 >= 2, "frames must enclose a non-empty interior"
        interior = grid[fr0 + 1 : fr1, fc0 + 1 : fc1]
        marker_cells = np.argwhere(interior != bg)
        assert len(marker_cells) == 1, "each frame interior must contain exactly one marker pixel"
        mr, mc = map(int, marker_cells[0])
        marker_color = int(interior[mr, mc])
        assert marker_color != color, "marker color must differ from the frame color"
        marker = (fr0 + 1 + mr, fc0 + 1 + mc)

        mask, anchor = _pattern_mask(pattern_coords)
        frames[color] = {
            "frame_bbox": (fr0, fc0, fr1, fc1),
            "marker": marker,
            "pattern_mask": mask,
            "anchor": anchor,
        }
        marker_colors.add(marker_color)

    assert frames, "expected at least one frame"
    assert len(marker_colors) == 1, "all markers must share one color"
    return bg, frames


def _transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    bg, frames = _parse_scene(out)
    for color, info in frames.items():
        r0, c0, r1, c1 = info["frame_bbox"]
        marker_r, marker_c = info["marker"]
        pattern = info["pattern_mask"]
        anchor_r, anchor_c = info["anchor"]
        ph, pw = pattern.shape

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = bg

        rows = np.arange(r0 + 1, r1, dtype=int)
        cols = np.arange(c0 + 1, c1, dtype=int)
        tiled_rows = (rows - (marker_r - anchor_r)) % ph
        tiled_cols = (cols - (marker_c - anchor_c)) % pw
        tiled_mask = pattern[np.ix_(tiled_rows, tiled_cols)]
        interior[tiled_mask] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow colored stamp with a periodic tiling of its matching outside ink pattern, phase-aligned by the interior marker.

    Input structure:
    - The background is black.
    - Several one-pixel-thick hollow rectangular frames appear in distinct colors.
    - Each frame encloses exactly one single-pixel marker of a shared marker color.
    - Outside the frames, each frame color also appears once as a small connected pattern.

    Rule:
    - Match every rectangular frame with the unique outside pattern that has the same color.
    - Use the pattern's tight bounding box as the tiling period and repeat that mask across the frame interior.
    - Align the tiling phase so the frame's marker lands on the pattern's top-left occupied pixel within one period.
    - Draw only inside the frame interior, clip partial periods at the edges, keep the frame border unchanged, and overwrite the marker.

    Key insight:
    - The outside same-color shape is not copied once; it defines a periodic tile, and the lone interior marker fixes the tile phase rather than the placement center.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    return _transform(grid)


def _random_pattern(rng: random.Random, require_shifted_anchor: bool) -> np.ndarray:
    candidates = []
    for base in PATTERN_LIBRARY:
        for turns in range(4):
            mask = np.rot90(base, turns)
            if require_shifted_anchor and mask[0, 0]:
                continue
            candidates.append(mask.copy())
    assert candidates
    return rng.choice(candidates)


def _mark_occupied(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    occupied[r : r + h, c : c + w] = True


def _bbox_is_free(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = PLACEMENT_PAD) -> bool:
    oh, ow = occupied.shape
    if r < 0 or c < 0 or r + h > oh or c + w > ow:
        return False
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(oh, r + h + pad)
    cc1 = min(ow, c + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_bbox(occupied: np.ndarray, h: int, w: int, rng: random.Random, attempts: int = 300):
    oh, ow = occupied.shape
    if h > oh or w > ow:
        return None
    for _ in range(attempts):
        r = rng.randrange(0, oh - h + 1)
        c = rng.randrange(0, ow - w + 1)
        if _bbox_is_free(occupied, r, c, h, w):
            _mark_occupied(occupied, r, c, h, w)
            return r, c
    return None


def _sample_marker(
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    rng: random.Random,
    force_edge: bool,
) -> tuple[int, int]:
    rows = list(range(top + 1, top + 1 + inner_h))
    cols = list(range(left + 1, left + 1 + inner_w))
    if force_edge and (inner_h > 1 or inner_w > 1):
        if rng.random() < 0.5 and inner_h > 1:
            return rng.choice([rows[0], rows[-1]]), rng.choice(cols)
        if inner_w > 1:
            return rng.choice(rows), rng.choice([cols[0], cols[-1]])
    return rng.choice(rows), rng.choice(cols)


def _draw_frame(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int, color: int) -> None:
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _draw_pattern(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    if frame_count + 1 > len(PALETTE):
        return None

    chosen_colors = rng.sample(PALETTE, frame_count + 1)
    marker_color = chosen_colors[0]
    frame_colors = chosen_colors[1:]
    shifted_index = rng.randrange(frame_count)
    edge_marker_index = rng.randrange(frame_count)

    specs = []
    for index, color in enumerate(frame_colors):
        pattern = _random_pattern(rng, require_shifted_anchor=(index == shifted_index))
        ph, pw = pattern.shape

        min_inner_h = max(3, ph)
        min_inner_w = max(3, pw)
        max_inner_h = min(8, ph + 5)
        max_inner_w = min(8, pw + 5)
        inner_h = rng.randint(min_inner_h, max_inner_h)
        inner_w = rng.randint(min_inner_w, max_inner_w)
        specs.append(
            {
                "color": color,
                "pattern": pattern,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "force_edge_marker": index == edge_marker_index,
            }
        )

    occupied = np.zeros((height, width), dtype=bool)
    for spec in sorted(specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        placed = _place_bbox(occupied, spec["outer_h"], spec["outer_w"], rng)
        if placed is None:
            return None
        spec["frame_top"], spec["frame_left"] = placed

    for spec in specs:
        ph, pw = spec["pattern"].shape
        placed = _place_bbox(occupied, ph, pw, rng)
        if placed is None:
            return None
        spec["pattern_top"], spec["pattern_left"] = placed

    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        _draw_frame(grid, spec["frame_top"], spec["frame_left"], spec["outer_h"], spec["outer_w"], spec["color"])
        marker_r, marker_c = _sample_marker(
            spec["frame_top"],
            spec["frame_left"],
            spec["inner_h"],
            spec["inner_w"],
            rng,
            spec["force_edge_marker"],
        )
        grid[marker_r, marker_c] = marker_color
        _draw_pattern(grid, spec["pattern_top"], spec["pattern_left"], spec["pattern"], spec["color"])

    try:
        output = _transform(grid)
    except AssertionError:
        return None

    if np.array_equal(output, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-tiling input")
