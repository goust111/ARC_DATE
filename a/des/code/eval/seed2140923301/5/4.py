from __future__ import annotations

import random
from collections import Counter

import numpy as np


MARKER_COLOR = 9
ANCHOR_COLOR = 3
SHAPE_ORDER = ("horizontal", "vertical", "l", "square")

# Generator configuration.
GEN_ATTEMPTS = 400
GEN_FRAME_COUNT = (4, 5)
GEN_GRID_H = (24, 30)
GEN_GRID_W = (24, 30)
GEN_FRAME_SIZES = (7, 9)
GEN_LIBRARY_SPAN = 15  # 4 * 3 + 3 separators
GEN_CLEAR_BAND = 4
GEN_PLACE_ATTEMPTS = 400

STAMP_LAYOUTS = [
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[2, 0, 2], [0, 1, 0], [2, 0, 2]], dtype=int),
    np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=int),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False
    target = set()
    for c in range(c0, c1 + 1):
        target.add((r0, c))
        target.add((r1, c))
    for r in range(r0 + 1, r1):
        target.add((r, c0))
        target.add((r, c1))
    return set(coords) == target


def _find_library(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    if h < 3 or w < GEN_LIBRARY_SPAN:
        raise AssertionError("grid too small for stamp library")

    candidates: list[tuple[int, int, int, list[np.ndarray]]] = []
    for r0 in sorted({0, h - 3}):
        band = grid[r0 : r0 + 3]
        for c0 in range(w - GEN_LIBRARY_SPAN + 1):
            if c0 > 0 and not np.all(band[:, :c0] == bg):
                continue
            if c0 + GEN_LIBRARY_SPAN < w and not np.all(band[:, c0 + GEN_LIBRARY_SPAN :] == bg):
                continue

            region = band[:, c0 : c0 + GEN_LIBRARY_SPAN]
            if not np.all(region[:, 3] == bg):
                continue
            if not np.all(region[:, 7] == bg):
                continue
            if not np.all(region[:, 11] == bg):
                continue

            patches = [
                region[:, 0:3].copy(),
                region[:, 4:7].copy(),
                region[:, 8:11].copy(),
                region[:, 12:15].copy(),
            ]
            if not all(np.any(patch != bg) for patch in patches):
                continue
            score = int(np.count_nonzero(region != bg))
            candidates.append((score, r0, c0, patches))

    assert candidates, "failed to locate stamp library"
    candidates.sort(key=lambda item: (item[0], -item[1], -item[2]))
    return candidates[-1][3]


def _find_anchor(grid: np.ndarray, bg: int, bbox_: tuple[int, int, int, int]) -> tuple[int, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox_
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2

    rays = [
        ((r0 - 1, mid_c), (-1, 0)),
        ((r1 + 1, mid_c), (1, 0)),
        ((mid_r, c0 - 1), (0, -1)),
        ((mid_r, c1 + 1), (0, 1)),
    ]
    hits: list[tuple[int, int]] = []

    for (r, c), (dr, dc) in rays:
        while 0 <= r < h and 0 <= c < w:
            val = int(grid[r, c])
            if val == ANCHOR_COLOR:
                hits.append((int(r), int(c)))
                break
            if val != int(bg):
                break
            r += dr
            c += dc

    assert len(hits) == 1, "each frame must have exactly one anchor"
    return hits[0]


def _classify_shape(coords: np.ndarray) -> str:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        return "horizontal"
    if np.all(cols == cols[0]):
        return "vertical"

    norm_rows = rows - int(rows.min())
    norm_cols = cols - int(cols.min())
    points = {(int(r), int(c)) for r, c in zip(norm_rows, norm_cols)}
    if points == {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return "square"

    assert len(points) == 3
    assert max(r for r, _ in points) == 1 and max(c for _, c in points) == 1
    return "l"


def _paste_patch(out: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    h, w = out.shape
    cr, cc = center
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = int(cr + dr)
            cc2 = int(cc + dc)
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = int(patch[dr + 1, dc + 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose a stamp from the edge library based on the maroon marker shape inside each frame,
    then stamp that 3x3 pattern around the frame's external green anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    stamps = _find_library(g, bg)
    out = g.copy()

    frames = []
    for color in map(int, np.unique(g)):
        if color in {bg, MARKER_COLOR, ANCHOR_COLOR}:
            continue
        for comp in _components_of_color(g, color):
            if not _is_frame_component(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            marker_coords = np.argwhere(inner == MARKER_COLOR)
            if 2 <= len(marker_coords) <= 4:
                frames.append((r0, c0, r1, c1, marker_coords + np.array([r0 + 1, c0 + 1])))

    assert frames, "no valid frames found"
    frames.sort(key=lambda item: (item[0], item[1]))

    for r0, c0, r1, c1, marker_coords in frames:
        shape = _classify_shape(marker_coords)
        stamp = stamps[SHAPE_ORDER.index(shape)]
        anchor = _find_anchor(g, bg, (r0, c0, r1, c1))
        _paste_patch(out, stamp, anchor)

    return out


def _shape_cells(shape: str, inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    if shape == "horizontal":
        length = random.randint(2, min(4, inner_w))
        row = random.randint(0, inner_h - 1)
        start = random.randint(0, inner_w - length)
        return [(row, start + dc) for dc in range(length)]

    if shape == "vertical":
        length = random.randint(2, min(4, inner_h))
        col = random.randint(0, inner_w - 1)
        start = random.randint(0, inner_h - length)
        return [(start + dr, col) for dr in range(length)]

    if shape == "square":
        row = random.randint(0, inner_h - 2)
        col = random.randint(0, inner_w - 2)
        return [(row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1)]

    row = random.randint(0, inner_h - 2)
    col = random.randint(0, inner_w - 2)
    variants = [
        [(row, col), (row, col + 1), (row + 1, col)],
        [(row, col), (row, col + 1), (row + 1, col + 1)],
        [(row, col), (row + 1, col), (row + 1, col + 1)],
        [(row, col + 1), (row + 1, col), (row + 1, col + 1)],
    ]
    return random.choice(variants)


def _make_stamps(colors: list[int]) -> list[np.ndarray]:
    stamps = []
    palette = [int(c) for c in colors]
    for layout in STAMP_LAYOUTS:
        stamp = np.zeros((3, 3), dtype=int)
        for idx in range(3):
            stamp[layout == idx] = palette[idx]
        stamps.append(stamp)
    return stamps


def _sample_stamp_colors(bg: int) -> list[int]:
    pool = [c for c in range(10) if c not in {bg, MARKER_COLOR, ANCHOR_COLOR}]
    return random.sample(pool, 3)


def _shape_sequence(n_frames: int) -> list[str]:
    shapes = list(SHAPE_ORDER)
    while len(shapes) < n_frames:
        shapes.append(random.choice(SHAPE_ORDER))
    random.shuffle(shapes)
    return shapes


def _stamp_bbox(frame_bbox: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "top":
        return r0 - 3, mid_c - 1, r0 - 1, mid_c + 1
    if side == "bottom":
        return r1 + 1, mid_c - 1, r1 + 3, mid_c + 1
    if side == "left":
        return mid_r - 1, c0 - 3, mid_r + 1, c0 - 1
    return mid_r - 1, c1 + 1, mid_r + 1, c1 + 3


def _anchor_for_side(frame_bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "top":
        return r0 - 2, mid_c
    if side == "bottom":
        return r1 + 2, mid_c
    if side == "left":
        return mid_r, c0 - 2
    return mid_r, c1 + 2


def _reserve_bbox(frame_bbox: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int]:
    sr0, sc0, sr1, sc1 = _stamp_bbox(frame_bbox, side)
    r0, c0, r1, c1 = frame_bbox
    ur0 = min(r0, sr0)
    uc0 = min(c0, sc0)
    ur1 = max(r1, sr1)
    uc1 = max(c1, sc1)
    return ur0, uc0, ur1, uc1


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _manual_output(grid: np.ndarray, stamps: list[np.ndarray], frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        stamp = stamps[SHAPE_ORDER.index(frame["shape"])]
        _paste_patch(out, stamp, frame["anchor"])
    return out


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((count for color, count in counts.items() if color != int(bg)), default=0)


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*GEN_FRAME_COUNT)
    min_h = GEN_GRID_H[0]
    min_w = GEN_GRID_W[0] + (2 if n_frames >= 5 else 0)
    h = random.randint(min_h, GEN_GRID_H[1])
    w = random.randint(min_w, GEN_GRID_W[1])

    bg = random.choice([c for c in range(10) if c not in {MARKER_COLOR, ANCHOR_COLOR}])
    library_side = random.choice(["top", "bottom"])
    library_row = 0 if library_side == "top" else h - 3
    library_col = random.randint(0, w - GEN_LIBRARY_SPAN)

    frame_colors = random.sample([c for c in range(10) if c not in {bg, MARKER_COLOR, ANCHOR_COLOR}], n_frames)
    stamps = _make_stamps(_sample_stamp_colors(bg))
    shapes = _shape_sequence(n_frames)

    specs = []
    for idx in range(n_frames):
        outer_h = random.choice(GEN_FRAME_SIZES)
        outer_w = random.choice(GEN_FRAME_SIZES)
        side = random.choice(["top", "bottom", "left", "right"])
        specs.append(
            {
                "shape": shapes[idx],
                "color": frame_colors[idx],
                "side": side,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )

    occupied = np.zeros((h, w), dtype=bool)
    if library_side == "top":
        occupied[:GEN_CLEAR_BAND, :] = True
    else:
        occupied[h - GEN_CLEAR_BAND :, :] = True

    frames: list[dict] = [None] * n_frames  # type: ignore[list-item]
    order = sorted(
        range(n_frames),
        key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"],
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        placed = False
        for _ in range(GEN_PLACE_ATTEMPTS):
            r0 = random.randint(1, h - spec["outer_h"] - 2)
            c0 = random.randint(1, w - spec["outer_w"] - 2)
            bbox_ = (r0, c0, r0 + spec["outer_h"] - 1, c0 + spec["outer_w"] - 1)
            rr0, cc0, rr1, cc1 = _reserve_bbox(bbox_, spec["side"])
            if rr0 < 0 or cc0 < 0 or rr1 >= h or cc1 >= w:
                continue
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            frames[idx] = {
                "bbox": bbox_,
                "shape": spec["shape"],
                "color": spec["color"],
                "side": spec["side"],
                "anchor": _anchor_for_side(bbox_, spec["side"]),
            }
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), bg, dtype=int)
    library = np.full((3, GEN_LIBRARY_SPAN), bg, dtype=int)
    library[:, 0:3] = stamps[0]
    library[:, 4:7] = stamps[1]
    library[:, 8:11] = stamps[2]
    library[:, 12:15] = stamps[3]
    grid[library_row : library_row + 3, library_col : library_col + GEN_LIBRARY_SPAN] = library

    for frame in frames:
        bbox_ = frame["bbox"]
        _draw_frame(grid, bbox_, frame["color"])
        inner_h = bbox_[2] - bbox_[0] - 1
        inner_w = bbox_[3] - bbox_[1] - 1
        for dr, dc in _shape_cells(frame["shape"], inner_h, inner_w):
            grid[bbox_[0] + 1 + dr, bbox_[1] + 1 + dc] = MARKER_COLOR
        ar, ac = frame["anchor"]
        grid[ar, ac] = ANCHOR_COLOR

    if not _bg_is_dominant(grid, bg):
        return None

    expected = _manual_output(grid, stamps, frames)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
