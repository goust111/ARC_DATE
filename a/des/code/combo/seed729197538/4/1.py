from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _corner_majority_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, int]]]:
    best_color = None
    best_frames: list[dict[str, int]] = []
    best_score = (-1, -1)
    for color in sorted(map(int, np.unique(grid))):
        if color == bg:
            continue
        frames: list[dict[str, int]] = []
        for coords in _components_of_color(grid, color):
            if not _is_rect_frame(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            frames.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
        score = (len(frames), sum((f["r1"] - f["r0"] + 1) * (f["c1"] - f["c0"] + 1) for f in frames))
        if score > best_score:
            best_score = score
            best_color = color
            best_frames = frames
    assert best_color is not None and best_frames, "expected at least one rectangular frame"
    best_frames.sort(key=lambda f: (f["r0"], f["c0"], f["r1"], f["c1"]))
    return int(best_color), best_frames


def _mapping_block_corner_score(h: int, w: int, r: int, c: int) -> int:
    return min(
        r + c,
        r + (w - (c + 4)),
        (h - (r + 3)) + c,
        (h - (r + 3)) + (w - (c + 4)),
    )


def _find_mapping_for_color(grid: np.ndarray, bg: int, frame_color: int, color: int) -> np.ndarray:
    h, w = grid.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    for r in range(h - 2):
        for c in range(w - 3):
            patch = grid[r : r + 3, c : c + 4]
            if frame_color in patch:
                continue
            if not bool((patch[:, 0] == color).all()):
                continue
            pat = patch[:, 1:4]
            vals = set(map(int, np.unique(pat)))
            if not (vals <= {bg, color} and bg in vals and color in vals):
                continue
            candidates.append((r, c, pat.copy()))
    assert candidates, f"missing mapping block for color {color}"
    candidates.sort(key=lambda item: (_mapping_block_corner_score(h, w, item[0], item[1]), item[0], item[1]))
    return candidates[0][2]


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = (height + pattern.shape[0] - 1) // pattern.shape[0]
    cc = (width + pattern.shape[1] - 1) // pattern.shape[1]
    return np.tile(pattern, (rr, cc))[:height, :width]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray rectangular enclosure by tiling the 3x3 pattern keyed by its interior marker color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _corner_majority_bg(g)
    frame_color, frames = _find_frames(g, bg)

    marker_colors: set[int] = set()
    frame_infos: list[dict[str, int]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_pts = np.argwhere((interior != bg) & (interior != frame_color))
        assert len(marker_pts) == 1, "each frame interior must contain exactly one marker pixel"
        mr, mc = marker_pts[0]
        marker_color = int(interior[int(mr), int(mc)])
        marker_colors.add(marker_color)
        frame_infos.append(
            {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "marker_color": marker_color}
        )

    mapping = {color: _find_mapping_for_color(g, bg, frame_color, color) for color in marker_colors}

    out = g.copy()
    for frame in frame_infos:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        pattern = mapping[int(frame["marker_color"])]
        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(pattern, ih, iw)
    return out


# --------------- generator ---------------

_BG = 0
_FRAME = 5
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_PATTERN_LIBRARY = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=int),
]
_MAX_ATTEMPTS = 400


def _table_dims(n_entries: int, orientation: str, gap: int) -> tuple[int, int]:
    if orientation == "vertical":
        return n_entries * 3 + (n_entries - 1) * gap, 4
    return 3, n_entries * 4 + (n_entries - 1) * gap


def _table_origin(
    h: int,
    w: int,
    table_h: int,
    table_w: int,
    corner: str,
    margin_r: int,
    margin_c: int,
) -> tuple[int, int]:
    if corner.startswith("top"):
        r0 = margin_r
    else:
        r0 = h - margin_r - table_h
    if corner.endswith("left"):
        c0 = margin_c
    else:
        c0 = w - margin_c - table_w
    return r0, c0


def _mark_occupied(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = occ.shape
    occ[max(0, r0 - pad) : min(hh, r0 + h + pad), max(0, c0 - pad) : min(ww, c0 + w + pad)] = True


def _render_mapping_block(color: int, pattern_mask: np.ndarray) -> np.ndarray:
    block = np.zeros((3, 4), dtype=int)
    block[:, 0] = color
    block[:, 1:4] = np.where(pattern_mask == 1, color, _BG)
    return block


def _free_positions(occ: np.ndarray, box_h: int, box_w: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - box_h):
        for c0 in range(1, w - box_w):
            if occ[r0 - 1 : r0 + box_h + 1, c0 - 1 : c0 + box_w + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_frame(
    inp: np.ndarray,
    out: np.ndarray,
    occ: np.ndarray,
    box_h: int,
    box_w: int,
    marker_color: int,
    pattern: np.ndarray,
    rng: random.Random,
) -> bool:
    positions = _free_positions(occ, box_h, box_w)
    if not positions:
        return False
    r0, c0 = rng.choice(positions)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    inp[r0, c0 : c1 + 1] = _FRAME
    inp[r1, c0 : c1 + 1] = _FRAME
    inp[r0 : r1 + 1, c0] = _FRAME
    inp[r0 : r1 + 1, c1] = _FRAME
    out[r0, c0 : c1 + 1] = _FRAME
    out[r1, c0 : c1 + 1] = _FRAME
    out[r0 : r1 + 1, c0] = _FRAME
    out[r0 : r1 + 1, c1] = _FRAME

    marker_r = rng.randint(r0 + 1, r1 - 1)
    marker_c = rng.randint(c0 + 1, c1 - 1)
    inp[marker_r, marker_c] = marker_color

    tiled = _tile_pattern(pattern, box_h - 2, box_w - 2)
    out[r0 + 1 : r1, c0 + 1 : c1] = tiled

    _mark_occupied(occ, r0, c0, box_h, box_w, pad=1)
    return True


def _expected_output_spec(
    h: int,
    w: int,
    corner: str,
    orientation: str,
    margin_r: int,
    margin_c: int,
    entry_colors: list[int],
    entry_patterns: list[np.ndarray],
    frame_colors: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), _BG, dtype=int)
    out = np.full((h, w), _BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    gap = 1

    table_h, table_w = _table_dims(len(entry_colors), orientation, gap)
    r0, c0 = _table_origin(h, w, table_h, table_w, corner, margin_r, margin_c)
    blocks_by_color: dict[int, np.ndarray] = {}
    for idx, (color, mask) in enumerate(zip(entry_colors, entry_patterns)):
        if orientation == "vertical":
            br = r0 + idx * (3 + gap)
            bc = c0
        else:
            br = r0
            bc = c0 + idx * (4 + gap)
        block = _render_mapping_block(color, mask)
        inp[br : br + 3, bc : bc + 4] = block
        out[br : br + 3, bc : bc + 4] = block
        blocks_by_color[color] = block[:, 1:4].copy()
    _mark_occupied(occ, r0, c0, table_h, table_w, pad=1)

    size_choices = [(hh, ww) for hh in range(4, min(11, h - 1) + 1) for ww in range(4, min(11, w - 1) + 1)]
    rng.shuffle(size_choices)
    for marker_color in frame_colors:
        placed = False
        for box_h, box_w in size_choices:
            if _place_frame(inp, out, occ, box_h, box_w, marker_color, blocks_by_color[marker_color], rng):
                placed = True
                break
        if not placed:
            return None

    if int(np.count_nonzero(inp == _BG)) < (h * w) // 3:
        return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    n_entries = rng.randint(2, 5)
    orientation = rng.choice(["vertical", "horizontal"])
    corner = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    margin_r = rng.randint(0, 1)
    margin_c = rng.randint(0, 1)
    gap = 1
    table_h, table_w = _table_dims(n_entries, orientation, gap)

    min_h = max(20, table_h + 6)
    min_w = max(20, table_w + 6)
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)

    entry_colors = rng.sample(_COLOR_POOL, n_entries)
    pattern_ids = rng.sample(range(len(_PATTERN_LIBRARY)), n_entries)
    entry_patterns = [_PATTERN_LIBRARY[idx] for idx in pattern_ids]

    n_frames = rng.randint(2, 5)
    n_used_colors = rng.randint(2, min(n_entries, n_frames))
    used_colors = rng.sample(entry_colors, n_used_colors)
    frame_colors = used_colors[:]
    while len(frame_colors) < n_frames:
        frame_colors.append(rng.choice(used_colors))
    rng.shuffle(frame_colors)

    built = _expected_output_spec(
        h,
        w,
        corner,
        orientation,
        margin_r,
        margin_c,
        entry_colors,
        entry_patterns,
        frame_colors,
        rng,
    )
    if built is None:
        return None
    inp, expected = built
    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
