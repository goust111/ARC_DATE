import random
from collections import Counter, deque

import numpy as np


BG = 0
TRANSFORMS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
)
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1)
    if name == "rot180":
        return np.rot90(arr, 2)
    if name == "rot270":
        return np.rot90(arr, 3)
    if name == "flip_h":
        return np.fliplr(arr)
    if name == "flip_v":
        return np.flipud(arr)
    if name == "transpose":
        return arr.T.copy()
    if name == "anti_transpose":
        return np.fliplr(np.flipud(arr)).T.copy()
    raise ValueError(name)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _extract_frames(grid: np.ndarray):
    frames = []
    for color in sorted(set(map(int, grid.flat)) - {BG}):
        for pts in _components_of_color(grid, color):
            rows = [r for r, _ in pts]
            cols = [c for _, c in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            side_h = r1 - r0 + 1
            side_w = c1 - c0 + 1
            if side_h != side_w or side_h < 5:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros_like(patch, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue
            if len(pts) != int(border.sum()):
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "inside": patch[1:-1, 1:-1].copy(),
                }
            )
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1]))
    return frames


def _infer_transform(src: np.ndarray, dst: np.ndarray) -> str:
    matches = [name for name in TRANSFORMS if np.array_equal(_apply_transform(src, name), dst)]
    assert len(matches) == 1, "training pair must determine a unique transform"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    assert len(frames) == 4, "expected exactly four hollow square frames"

    nonempty = []
    empty = None
    for frame in frames:
        colors = frozenset(int(x) for x in np.unique(frame["inside"]) if int(x) != BG)
        frame["palette"] = colors
        if colors:
            nonempty.append(frame)
        else:
            assert empty is None, "expected exactly one blank frame"
            empty = frame
    assert empty is not None and len(nonempty) == 3

    pair = None
    for i in range(len(nonempty)):
        for j in range(i + 1, len(nonempty)):
            if nonempty[i]["palette"] == nonempty[j]["palette"]:
                assert pair is None, "training pair must be unique"
                pair = (nonempty[i], nonempty[j])
    assert pair is not None, "missing training pair"

    train_src, train_dst = pair
    test_frames = [f for f in nonempty if f not in pair]
    assert len(test_frames) == 1
    test_src = test_frames[0]

    shape = train_src["inside"].shape
    assert train_dst["inside"].shape == shape == test_src["inside"].shape == empty["inside"].shape
    transform = _infer_transform(train_src["inside"], train_dst["inside"])

    out = g.copy()
    r0, c0, r1, c1 = empty["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = _apply_transform(test_src["inside"], transform)
    return out


def _random_mask(rng: random.Random, n: int, area_lo: int, area_hi: int) -> np.ndarray:
    target = rng.randint(area_lo, area_hi)
    start = (rng.randrange(n), rng.randrange(n))
    cells = {start}
    frontier = {start}
    while len(cells) < target and frontier:
        rr, cc = rng.choice(tuple(frontier))
        neigh = [(nr, nc) for nr, nc in _neighbors4(rr, cc) if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in cells]
        if not neigh:
            frontier.remove((rr, cc))
            continue
        pick = rng.choice(neigh)
        cells.add(pick)
        frontier.add(pick)
    while len(cells) < target:
        rr, cc = rng.choice(tuple(cells))
        neigh = [(nr, nc) for nr, nc in _neighbors4(rr, cc) if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in cells]
        if not neigh:
            break
        cells.add(rng.choice(neigh))
    mask = np.zeros((n, n), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _color_sprite(rng: random.Random, mask: np.ndarray, palette: list[int]) -> np.ndarray:
    n = mask.shape[0]
    sprite = np.zeros((n, n), dtype=int)
    cells = list(map(tuple, np.argwhere(mask)))
    rng.shuffle(cells)
    for idx, (r, c) in enumerate(cells):
        if idx < len(palette):
            sprite[r, c] = palette[idx]
        else:
            sprite[r, c] = rng.choice(palette)
    return sprite


def _unique_transform_match(src: np.ndarray, dst: np.ndarray, wanted: str) -> bool:
    matches = [name for name in TRANSFORMS if np.array_equal(_apply_transform(src, name), dst)]
    return matches == [wanted]


def _build_sprite_for_transform(rng: random.Random, n: int, palette: list[int], transform: str) -> np.ndarray | None:
    for _ in range(120):
        mask = _random_mask(rng, n, max(len(palette), n), min(n * n - 2, n + 4 + rng.randint(0, n)))
        src = _color_sprite(rng, mask, palette)
        dst = _apply_transform(src, transform)
        if not _unique_transform_match(src, dst, transform):
            continue
        if np.array_equal(src, dst):
            continue
        return src
    return None


def _place_frame(grid: np.ndarray, top: int, left: int, border_color: int, inside: np.ndarray):
    n = inside.shape[0]
    side = n + 2
    grid[top : top + side, left : left + side] = BG
    grid[top, left : left + side] = border_color
    grid[top + side - 1, left : left + side] = border_color
    grid[top : top + side, left] = border_color
    grid[top : top + side, left + side - 1] = border_color
    grid[top + 1 : top + side - 1, left + 1 : left + side - 1] = inside


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    transform = rng.choice(TRANSFORMS)
    n = rng.randint(4, 7)
    train_palette_size, test_palette_size = rng.choice([(2, 2), (2, 3), (3, 2)])

    colors = rng.sample(COLOR_POOL, 4 + train_palette_size + test_palette_size)
    border_colors = colors[:4]
    train_palette = colors[4 : 4 + train_palette_size]
    test_palette = colors[4 + train_palette_size :]

    train_src = _build_sprite_for_transform(rng, n, train_palette, transform)
    test_src = _build_sprite_for_transform(rng, n, test_palette, transform)
    if train_src is None or test_src is None:
        return None
    train_dst = _apply_transform(train_src, transform)
    answer = _apply_transform(test_src, transform)
    blank = np.zeros((n, n), dtype=int)

    side = n + 2
    row_gap = rng.randint(1, 4)
    col_gap = rng.randint(1, 4)
    top_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)

    h = top_margin + side + row_gap + side + bottom_margin
    w = left_margin + side + col_gap + side + right_margin
    grid = np.zeros((h, w), dtype=int)

    slots = [
        (top_margin, left_margin),
        (top_margin, left_margin + side + col_gap),
        (top_margin + side + row_gap, left_margin),
        (top_margin + side + row_gap, left_margin + side + col_gap),
    ]
    rng.shuffle(slots)

    contents = [train_src, train_dst, test_src, blank]
    expected = grid.copy()
    for (top, left), border_color, inside in zip(slots, border_colors, contents):
        _place_frame(grid, top, left, border_color, inside)

    expected[:] = grid
    blank_slot = slots[3]
    _place_frame(expected, blank_slot[0], blank_slot[1], border_colors[3], answer)

    canvas_transform = rng.choice(TRANSFORMS)
    grid = _apply_transform(grid, canvas_transform)
    expected = _apply_transform(expected, canvas_transform)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
