from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 0
GRAY_COLOR = 5
SIDE_NAMES = ("top", "bottom", "left", "right")
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
FRAME_COLOR_POOL = (1, 2, 3, 4)

GRID_H_RANGE = (24, 35)
GRID_W_RANGE = (24, 35)
FRAME_COUNT_CHOICES = (1, 2, 3, 4)
FRAME_COUNT_WEIGHTS = (1, 3, 3, 1)
INNER_H_RANGE = (5, 11)
INNER_W_RANGE = (5, 11)
GAP_WIDTH_RANGE = (2, 4)
PATCH_PADDING = 1
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _side_segments(r0: int, c0: int, r1: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }


def _parse_frame(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
        return None

    coord_set = set(coords)
    border = _border_cells(r0, c0, r1, c1)
    if not coord_set <= border:
        return None

    side_cells = _side_segments(r0, c0, r1, c1)
    gap_side = None
    gap_cells: set[tuple[int, int]] = set()
    side_colors: dict[str, int] = {}

    for side, cells in side_cells.items():
        vals = [int(grid[r, c]) for r, c in cells]
        black_idx = [i for i, v in enumerate(vals) if v == BG_COLOR]
        if black_idx:
            if gap_side is not None:
                return None
            if len(black_idx) < GAP_WIDTH_RANGE[0] or len(black_idx) > GAP_WIDTH_RANGE[1]:
                return None
            if black_idx != list(range(black_idx[0], black_idx[-1] + 1)):
                return None
            colors = {v for v in vals if v != BG_COLOR}
            if len(colors) != 1 or GRAY_COLOR in colors:
                return None
            gap_side = side
            gap_cells = {cells[i] for i in black_idx}
            side_colors[side] = int(next(iter(colors)))
        else:
            colors = set(vals)
            if len(colors) != 1:
                return None
            color = int(next(iter(colors)))
            if color in (BG_COLOR, GRAY_COLOR):
                return None
            side_colors[side] = color

    if gap_side is None:
        return None
    if border - coord_set != gap_cells:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    pattern_mask = interior == GRAY_COLOR
    if not pattern_mask.any():
        return None
    if len(_components(pattern_mask)) != 1:
        return None

    rel_coords = [(int(r), int(c)) for r, c in np.argwhere(pattern_mask)]
    rel_rows = [r for r, _ in rel_coords]
    rel_cols = [c for _, c in rel_coords]
    if (
        min(rel_rows) == 0
        or max(rel_rows) == interior.shape[0] - 1
        or min(rel_cols) == 0
        or max(rel_cols) == interior.shape[1] - 1
    ):
        return None

    abs_coords = [(int(r0 + 1 + r), int(c0 + 1 + c)) for r, c in rel_coords]
    return {
        "bbox": (r0, c0, r1, c1),
        "gap_side": gap_side,
        "side_colors": side_colors,
        "pattern": abs_coords,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG_COLOR) & (grid != GRAY_COLOR)
    frames = []
    for comp in _components(mask):
        frame = _parse_frame(grid, comp)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Duplicate each interior gray pattern out through the frame gap and recolor the
    external copy with the opposite side's border color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one gapped rectangular frame"

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_side = frame["gap_side"]
        copy_color = int(frame["side_colors"][OPPOSITE_SIDE[gap_side]])
        pattern = frame["pattern"]

        rows = [r for r, _ in pattern]
        cols = [c for _, c in pattern]
        pr0, pr1 = min(rows), max(rows)
        pc0, pc1 = min(cols), max(cols)

        if gap_side == "top":
            shift = (pr1 - pr0 + 1) + (pr0 - r0)
            dr, dc = -shift, 0
        elif gap_side == "bottom":
            shift = (pr1 - pr0 + 1) + (r1 - pr1)
            dr, dc = shift, 0
        elif gap_side == "left":
            shift = (pc1 - pc0 + 1) + (pc0 - c0)
            dr, dc = 0, -shift
        else:
            shift = (pc1 - pc0 + 1) + (c1 - pc1)
            dr, dc = 0, shift

        for r, c in pattern:
            rr = int(r + dr)
            cc = int(c + dc)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            out[rr, cc] = copy_color

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _build_shape_library() -> list[np.ndarray]:
    bases = [
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    ]
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in bases:
        for k in range(4):
            rot = np.rot90(base, k)
            for flip in (False, True):
                cur = np.fliplr(rot) if flip else rot
                cur = _crop_mask(cur)
                key = (tuple(map(int, cur.shape)), cur.tobytes())
                if key not in seen:
                    seen.add(key)
                    variants.append(cur.copy())
    return variants


SHAPE_LIBRARY = _build_shape_library()


def _choose_shape(direction: str, gap_width: int, rng: random.Random) -> np.ndarray | None:
    if direction in ("top", "bottom"):
        candidates = [
            shape
            for shape in SHAPE_LIBRARY
            if 2 <= shape.shape[1] <= gap_width and 2 <= shape.shape[0] <= 4
        ]
    else:
        candidates = [
            shape
            for shape in SHAPE_LIBRARY
            if 2 <= shape.shape[0] <= gap_width and 2 <= shape.shape[1] <= 4
        ]
    if not candidates:
        return None
    return rng.choice(candidates).copy()


def _draw_frame(
    patch: np.ndarray,
    frame_r: int,
    frame_c: int,
    inner_h: int,
    inner_w: int,
    side_colors: dict[str, int],
    gap_side: str,
    gap_start: int,
    gap_width: int,
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r0, c0 = frame_r, frame_c
    r1, c1 = frame_r + outer_h - 1, frame_c + outer_w - 1

    patch[r0:r1 + 1, c0] = side_colors["left"]
    patch[r0:r1 + 1, c1] = side_colors["right"]
    patch[r0, c0 + 1:c1] = side_colors["top"]
    patch[r1, c0 + 1:c1] = side_colors["bottom"]

    if gap_side == "top":
        patch[r0, c0 + 1 + gap_start : c0 + 1 + gap_start + gap_width] = BG_COLOR
    elif gap_side == "bottom":
        patch[r1, c0 + 1 + gap_start : c0 + 1 + gap_start + gap_width] = BG_COLOR
    elif gap_side == "left":
        patch[r0 + 1 + gap_start : r0 + 1 + gap_start + gap_width, c0] = BG_COLOR
    else:
        patch[r0 + 1 + gap_start : r0 + 1 + gap_start + gap_width, c1] = BG_COLOR


def _make_system(direction: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    gap_width = rng.randint(*GAP_WIDTH_RANGE)
    shape = _choose_shape(direction, gap_width, rng)
    if shape is None:
        return None
    ph, pw = map(int, shape.shape)

    if direction in ("top", "bottom"):
        inner_h_min = max(INNER_H_RANGE[0], ph + 3)
        inner_w_min = max(INNER_W_RANGE[0], gap_width + 2)
        if inner_h_min > INNER_H_RANGE[1] or inner_w_min > INNER_W_RANGE[1]:
            return None
        inner_h = rng.randint(inner_h_min, INNER_H_RANGE[1])
        inner_w = rng.randint(inner_w_min, INNER_W_RANGE[1])
        gap_start = rng.randint(1, inner_w - gap_width - 1)
        pattern_c = gap_start + rng.randint(0, gap_width - pw)
        pattern_r = rng.randint(1, inner_h - ph - 1)
    else:
        inner_h_min = max(INNER_H_RANGE[0], gap_width + 2)
        inner_w_min = max(INNER_W_RANGE[0], pw + 3)
        if inner_h_min > INNER_H_RANGE[1] or inner_w_min > INNER_W_RANGE[1]:
            return None
        inner_h = rng.randint(inner_h_min, INNER_H_RANGE[1])
        inner_w = rng.randint(inner_w_min, INNER_W_RANGE[1])
        gap_start = rng.randint(1, inner_h - gap_width - 1)
        pattern_r = gap_start + rng.randint(0, gap_width - ph)
        pattern_c = rng.randint(1, inner_w - pw - 1)

    side_colors = dict(zip(SIDE_NAMES, rng.sample(FRAME_COLOR_POOL, len(SIDE_NAMES))))

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    patch_h = outer_h + (ph if direction in ("top", "bottom") else 0)
    patch_w = outer_w + (pw if direction in ("left", "right") else 0)
    frame_r = ph if direction == "top" else 0
    frame_c = pw if direction == "left" else 0

    inp = np.full((patch_h, patch_w), BG_COLOR, dtype=int)
    _draw_frame(inp, frame_r, frame_c, inner_h, inner_w, side_colors, direction, gap_start, gap_width)

    src_r = frame_r + 1 + pattern_r
    src_c = frame_c + 1 + pattern_c
    inp[src_r : src_r + ph, src_c : src_c + pw][shape] = GRAY_COLOR

    out = inp.copy()
    copy_color = int(side_colors[OPPOSITE_SIDE[direction]])
    if direction == "top":
        dst_r, dst_c = frame_r - ph, src_c
    elif direction == "bottom":
        dst_r, dst_c = frame_r + outer_h, src_c
    elif direction == "left":
        dst_r, dst_c = src_r, frame_c - pw
    else:
        dst_r, dst_c = src_r, frame_c + outer_w
    out[dst_r : dst_r + ph, dst_c : dst_c + pw][shape] = copy_color
    return inp, out


def _place_patch(
    inp: np.ndarray,
    out: np.ndarray,
    reserved: np.ndarray,
    inp_patch: np.ndarray,
    out_patch: np.ndarray,
    rng: random.Random,
) -> bool:
    h, w = inp.shape
    ph, pw = inp_patch.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - PATCH_PADDING)
            rr1 = min(h, r0 + ph + PATCH_PADDING)
            cc0 = max(0, c0 - PATCH_PADDING)
            cc1 = min(w, c0 + pw + PATCH_PADDING)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((int(r0), int(c0)))
    if not candidates:
        return False

    r0, c0 = rng.choice(candidates)
    inp_mask = inp_patch != BG_COLOR
    out_mask = out_patch != BG_COLOR
    inp[r0 : r0 + ph, c0 : c0 + pw][inp_mask] = inp_patch[inp_mask]
    out[r0 : r0 + ph, c0 : c0 + pw][out_mask] = out_patch[out_mask]
    reserved[r0 : r0 + ph, c0 : c0 + pw] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    inp = np.full((h, w), BG_COLOR, dtype=int)
    out = inp.copy()
    reserved = np.zeros((h, w), dtype=bool)

    count = rng.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    directions = rng.sample(SIDE_NAMES, k=count)
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for direction in directions:
        system = _make_system(direction, rng)
        if system is None:
            return None
        patches.append(system)

    rng.shuffle(patches)
    for inp_patch, out_patch in patches:
        if not _place_patch(inp, out, reserved, inp_patch, out_patch, rng):
            return None

    try:
        parsed = _find_frames(inp)
        if len(parsed) != count:
            return None
        result = main(inp)
    except Exception:
        return None

    if np.array_equal(result, inp):
        return None
    if not np.array_equal(result, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
