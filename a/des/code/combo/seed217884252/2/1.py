from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
ALL_SIDES = ("up", "down", "left", "right")
ROT_K = {"up": 0, "left": 1, "down": 2, "right": 3}

SPRITE_FAMILIES_UP = [
    np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1],
        ],
        dtype=bool,
    ),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not arr[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in nbrs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_boundary(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((int(r0), int(c)))
        pts.append((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        pts.append((int(r), int(c0)))
        pts.append((int(r), int(c1)))
    return pts


def _marker_side(marker: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str:
    r, c = marker
    if r == r0:
        return "up"
    if r == r1:
        return "down"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("marker must lie on the frame boundary")


def _infer_orientation(mask: np.ndarray) -> str:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    assert bool(m.any())

    def edge_stat(side: str) -> tuple[int, int, int, float]:
        if side == "up":
            edge = np.flatnonzero(m[0])
            inward = int(m[1].sum()) if h > 1 else 0
            center = (w - 1) / 2.0
        elif side == "down":
            edge = np.flatnonzero(m[h - 1])
            inward = int(m[h - 2].sum()) if h > 1 else 0
            center = (w - 1) / 2.0
        elif side == "left":
            edge = np.flatnonzero(m[:, 0])
            inward = int(m[:, 1].sum()) if w > 1 else 0
            center = (h - 1) / 2.0
        else:
            edge = np.flatnonzero(m[:, w - 1])
            inward = int(m[:, w - 2].sum()) if w > 1 else 0
            center = (h - 1) / 2.0

        count = int(len(edge))
        spread = int(edge[-1] - edge[0]) if count > 0 else 10**9
        center_offset = float(abs(float(edge.mean()) - center)) if count > 0 else float("inf")
        return count, spread, -inward, center_offset

    stats = {side: edge_stat(side) for side in ALL_SIDES}
    counts = {side: stats[side][0] for side in ALL_SIDES}
    min_count = min(counts.values())
    candidates = [side for side in ALL_SIDES if counts[side] == min_count]
    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(key=lambda side: stats[side])
    if len(candidates) == 1 or stats[candidates[0]] != stats[candidates[1]]:
        return candidates[0]

    for family in SPRITE_FAMILIES_UP:
        for side, k in ROT_K.items():
            rotated = np.rot90(family, k)
            if rotated.shape == m.shape and np.array_equal(rotated, m):
                return side

    raise AssertionError("sprite orientation is ambiguous")


def _extract_frames(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    non_bg = [int(v) for v in g[g != bg]]
    assert non_bg, "expected foreground"
    frame_color = int(Counter(non_bg).most_common(1)[0][0])

    frames: list[dict[str, object]] = []
    for comp in _components(g == frame_color, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 5 or w < 5:
            continue

        boundary = _rect_boundary(r0, c0, r1, c1)
        boundary_set = set(boundary)
        comp_set = set(comp)
        if not comp_set.issubset(boundary_set):
            continue

        non_frame = [(r, c) for (r, c) in boundary if int(g[r, c]) != frame_color]
        if len(non_frame) != 1:
            continue
        marker = non_frame[0]
        if int(g[marker[0], marker[1]]) == bg:
            continue
        if marker in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        if comp_set != boundary_set - {marker}:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        sprite_mask = inner != bg
        comps = _components(sprite_mask, connectivity=4)
        if len(comps) != 1:
            continue
        sprite_pixels = comps[0]
        sr0, sc0, sr1, sc1 = _bbox(sprite_pixels)
        sprite_crop = sprite_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]
        orientation = _infer_orientation(sprite_crop)
        pixels = [(int(r), int(c), int(inner[r, c])) for r, c in sprite_pixels]

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "size": (h, w),
                "marker_side": _marker_side(marker, r0, c0, r1, c1),
                "orientation": orientation,
                "pixels": pixels,
            }
        )

    assert frames, "expected frames"
    sizes = {tuple(frame["size"]) for frame in frames}
    assert len(sizes) == 1, "all frames must share a size"
    return bg, frame_color, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite to the frame whose border marker lies on that sprite's facing side.
    """
    g = np.asarray(grid, dtype=int)
    bg, _frame_color, frames = _extract_frames(g)
    target_by_side = {str(frame["marker_side"]): frame for frame in frames}
    assert set(target_by_side) == set(ALL_SIDES), "expected one target frame per direction"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)] = int(bg)

    for frame in frames:
        target = target_by_side[str(frame["orientation"])]
        tr0, tc0, _tr1, _tc1 = target["bbox"]
        for rr, cc, color in frame["pixels"]:
            out[int(tr0) + 1 + int(rr), int(tc0) + 1 + int(cc)] = int(color)

    return out


def _random_derangement(labels: list[str]) -> list[str]:
    while True:
        perm = labels[:]
        random.shuffle(perm)
        if all(src != dst for src, dst in zip(labels, perm)):
            return perm


def _rotated_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), ROT_K[direction])


def _colorize_base(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    n_colors = random.randint(2, min(4, len(colors)))
    palette = random.sample(colors, n_colors)
    random.shuffle(coords)

    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    for idx, (r, c) in enumerate(coords):
        if idx < n_colors:
            patch[r, c] = int(palette[idx])
        else:
            patch[r, c] = int(random.choice(palette))
    return patch


def _pick_offset(free_space: int) -> int:
    if free_space <= 0:
        return 0
    inner = [x for x in range(1, free_space) if 0 < x < free_space]
    if inner and random.random() < 0.75:
        return int(random.choice(inner))
    return int(random.randint(0, free_space))


def _marker_coord(side: str, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    if side == "up":
        return int(r0), int(random.randint(c0 + 1, c1 - 1))
    if side == "down":
        return int(r1), int(random.randint(c0 + 1, c1 - 1))
    if side == "left":
        return int(random.randint(r0 + 1, r1 - 1)), int(c0)
    return int(random.randint(r0 + 1, r1 - 1)), int(c1)


def _draw_frame(out: np.ndarray, frame: dict[str, object]) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    out[int(r0), int(c0) : int(c1) + 1] = FRAME_COLOR
    out[int(r1), int(c0) : int(c1) + 1] = FRAME_COLOR
    out[int(r0) : int(r1) + 1, int(c0)] = FRAME_COLOR
    out[int(r0) : int(r1) + 1, int(c1)] = FRAME_COLOR
    mr, mc = frame["marker_coord"]
    out[int(mr), int(mc)] = int(frame["marker_color"])


def _render_input(height: int, width: int, frames: list[dict[str, object]]) -> np.ndarray:
    out = np.full((height, width), BG_COLOR, dtype=int)
    for frame in frames:
        _draw_frame(out, frame)
        r0, c0, _r1, _c1 = frame["bbox"]
        for rr, cc, color in frame["pixels"]:
            out[int(r0) + 1 + int(rr), int(c0) + 1 + int(cc)] = int(color)
    return out


def _render_expected(height: int, width: int, frames: list[dict[str, object]]) -> np.ndarray:
    out = np.full((height, width), BG_COLOR, dtype=int)
    for frame in frames:
        _draw_frame(out, frame)
    targets = {str(frame["marker_side"]): frame for frame in frames}
    for frame in frames:
        target = targets[str(frame["orientation"])]
        tr0, tc0, _tr1, _tc1 = target["bbox"]
        for rr, cc, color in frame["pixels"]:
            out[int(tr0) + 1 + int(rr), int(tc0) + 1 + int(cc)] = int(color)
    return out


def _build_frame_records() -> tuple[int, int, list[dict[str, object]]] | None:
    outer_h = random.randint(8, 11)
    outer_w = random.randint(8, 11)
    inner_h = outer_h - 2
    inner_w = outer_w - 2

    top_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    row_gap = random.randint(2, 6)
    col_gap = random.randint(2, 6)
    bottom_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    height = top_margin + outer_h + row_gap + outer_h + bottom_margin
    width = left_margin + outer_w + col_gap + outer_w + right_margin
    slots = [
        (top_margin, left_margin),
        (top_margin, left_margin + outer_w + col_gap),
        (top_margin + outer_h + row_gap, left_margin),
        (top_margin + outer_h + row_gap, left_margin + outer_w + col_gap),
    ]

    marker_sides = list(ALL_SIDES)
    random.shuffle(marker_sides)
    sprite_dirs = _random_derangement(marker_sides)

    free_colors = [c for c in range(10) if c not in {BG_COLOR, FRAME_COLOR}]
    marker_colors = random.sample(free_colors, 4)
    sprite_colors = [c for c in free_colors if c not in marker_colors]
    if len(sprite_colors) < 4:
        return None

    frames: list[dict[str, object]] = []
    for idx, (r0, c0) in enumerate(slots):
        marker_side = marker_sides[idx]
        sprite_dir = sprite_dirs[idx]
        marker_color = marker_colors[idx]

        family = random.choice(SPRITE_FAMILIES_UP)
        base_patch = _colorize_base(family, sprite_colors)
        patch = np.rot90(base_patch, ROT_K[sprite_dir])
        mask = np.rot90(family, ROT_K[sprite_dir])
        sh, sw = mask.shape
        if sh > inner_h or sw > inner_w:
            return None

        off_r = _pick_offset(inner_h - sh)
        off_c = _pick_offset(inner_w - sw)
        pixels = [
            (int(off_r + rr), int(off_c + cc), int(patch[rr, cc]))
            for rr, cc in np.argwhere(mask)
        ]

        r1 = int(r0 + outer_h - 1)
        c1 = int(c0 + outer_w - 1)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "marker_side": marker_side,
                "marker_color": int(marker_color),
                "marker_coord": _marker_coord(marker_side, r0, c0, r1, c1),
                "orientation": sprite_dir,
                "pixels": pixels,
            }
        )

    return height, width, frames


def _gen_attempt() -> np.ndarray | None:
    built = _build_frame_records()
    if built is None:
        return None
    height, width, frames = built

    inp = _render_input(height, width, frames)
    expected = _render_expected(height, width, frames)
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
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
