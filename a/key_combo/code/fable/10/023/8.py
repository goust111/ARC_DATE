import random
from collections import deque

import numpy as np


BG = 0
FRAME = 1
RED = 2
GREEN = 3
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
SIDES = ("left", "right", "top", "bottom")


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_in_bbox(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[int(r) - r0, int(c) - c0] = True
    return out


def _is_hollow_frame(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    color = int(grid[coords[0, 0], coords[0, 1]])
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if int(grid[rr, cc]) != color:
            return False
        if rr not in (r0, r1) and cc not in (c0, c1):
            return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
        return False
    return len(coords) == int(border.sum())


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, GREY}):
        for coords in _components(grid == color, connectivity=4):
            if _is_hollow_frame(grid, coords):
                frames.append({"color": color, "bbox": _bbox(coords)})
    return frames


def _find_sources(grid: np.ndarray) -> list[np.ndarray]:
    return _components(grid == GREY, connectivity=8)


def _assignment_candidates(
    src_bbox: tuple[int, int, int, int], frames: list[dict]
) -> list[tuple[int, str, int]]:
    sr0, sc0, sr1, sc1 = src_bbox
    candidates: list[tuple[int, str, int]] = []
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        row_aligned = r0 + 1 <= sr0 and sr1 <= r1 - 1
        col_aligned = c0 + 1 <= sc0 and sc1 <= c1 - 1
        if row_aligned and sc1 < c0:
            candidates.append((idx, "left", c0 - sc1))
        if row_aligned and sc0 > c1:
            candidates.append((idx, "right", sc0 - c1))
        if col_aligned and sr1 < r0:
            candidates.append((idx, "top", r0 - sr1))
        if col_aligned and sr0 > r1:
            candidates.append((idx, "bottom", sr0 - r1))
    return candidates


def _reflect_coords(
    coords: np.ndarray, frame_bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    reflected: list[tuple[int, int]] = []
    if side == "left":
        for r, c in coords:
            reflected.append((int(r), int(2 * c0 - c)))
    elif side == "right":
        for r, c in coords:
            reflected.append((int(r), int(2 * c1 - c)))
    elif side == "top":
        for r, c in coords:
            reflected.append((int(2 * r0 - r), int(c)))
    else:
        for r, c in coords:
            reflected.append((int(2 * r1 - r), int(c)))
    return reflected


def _layer_depth(frame_bbox: tuple[int, int, int, int], r: int, c: int) -> int:
    r0, c0, r1, c1 = frame_bbox
    depth = min(r - r0, r1 - r, c - c0, c1 - c)
    assert depth >= 1
    return int(depth)


def _paint_reflection(
    out: np.ndarray, reflected: list[tuple[int, int]], frame_bbox: tuple[int, int, int, int]
) -> None:
    r0, c0, r1, c1 = frame_bbox
    for r, c in reflected:
        assert r0 < r < r1 and c0 < c < c1
        depth = _layer_depth(frame_bbox, r, c)
        out[r, c] = RED if depth % 2 == 1 else GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect each grey outside sprite through the facing wall of its aligned frame, then recolor by interior layer depth.

    Input structure:
    - The grid is black except for one-pixel-thick hollow rectangular frames and grey sprites outside them.
    - Every grey sprite is aligned with exactly one frame along a shared row band or column band.
    - Frames are empty inside, and every reflected sprite footprint fits strictly inside its matched frame.

    Rule:
    - Detect the hollow frames and the grey source sprites.
    - For each grey sprite, find the unique aligned frame and reflect the sprite across the frame wall that faces it:
      left/right alignment uses that vertical wall, top/bottom alignment uses that horizontal wall.
    - Erase the original grey sprite, keep the frame unchanged, and recolor every reflected pixel by its minimum
      distance to the frame border: odd depth becomes red, even depth becomes green.

    Key insight:
    - The frame supplies two pieces of geometry at once: the relevant reflection wall decides where the sprite enters,
      and the frame interior itself is a depth field that decides each reflected pixel's final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    frames = _find_frames(g)
    sources = _find_sources(g)
    assert frames and sources

    out = g.copy()
    out[out == GREY] = BG

    for src in sources:
        candidates = _assignment_candidates(_bbox(src), frames)
        assert len(candidates) == 1, "each grey object must align to exactly one frame"
        frame_idx, side, _gap = candidates[0]
        frame_bbox = frames[frame_idx]["bbox"]
        reflected = _reflect_coords(src, frame_bbox, side)
        _paint_reflection(out, reflected, frame_bbox)

    return out


def _random_connected_mask(height: int, width: int, side: str) -> np.ndarray | None:
    depth_is_horizontal = side in {"left", "right"}
    total = height * width
    min_area = max(3, total // 2)

    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        if depth_is_horizontal:
            seeds = [
                (random.randrange(height), 0),
                (random.randrange(height), width - 1),
            ]
        else:
            seeds = [
                (0, random.randrange(width)),
                (height - 1, random.randrange(width)),
            ]
        cells = {seeds[0], seeds[1]}
        target = random.randint(min_area, total)
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))
        for r, c in cells:
            mask[r, c] = True
        if depth_is_horizontal:
            if np.array_equal(mask, np.fliplr(mask)):
                continue
            if not mask[:, 0].any() or not mask[:, -1].any():
                continue
        else:
            if np.array_equal(mask, np.flipud(mask)):
                continue
            if not mask[0, :].any() or not mask[-1, :].any():
                continue
        return mask
    return None


def _build_unit(side: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(200):
        frame_h = random.randint(6, 11)
        frame_w = random.randint(6, 11)
        inner_h = frame_h - 2
        inner_w = frame_w - 2

        if side in {"left", "right"}:
            depth_span = random.randint(2, min(3, inner_w))
            span_h = random.randint(2, min(4, inner_h))
            mask = _random_connected_mask(span_h, depth_span, side)
            if mask is None:
                continue
            row0 = random.randint(1, inner_h - span_h + 1)
            col0 = 1 if side == "left" else inner_w - depth_span + 1
            patch_h = frame_h
            patch_w = frame_w + depth_span
            frame_r = 0
            frame_c = depth_span if side == "left" else 0
        else:
            depth_span = random.randint(2, min(3, inner_h))
            span_w = random.randint(2, min(4, inner_w))
            mask = _random_connected_mask(depth_span, span_w, side)
            if mask is None:
                continue
            row0 = 1 if side == "top" else inner_h - depth_span + 1
            col0 = random.randint(1, inner_w - span_w + 1)
            patch_h = frame_h + depth_span
            patch_w = frame_w
            frame_r = depth_span if side == "top" else 0
            frame_c = 0

        inp = np.full((patch_h, patch_w), BG, dtype=int)
        out = np.full((patch_h, patch_w), BG, dtype=int)

        r0, c0 = frame_r, frame_c
        r1, c1 = frame_r + frame_h - 1, frame_c + frame_w - 1
        inp[r0:r1 + 1, c0] = FRAME
        inp[r0:r1 + 1, c1] = FRAME
        inp[r0, c0:c1 + 1] = FRAME
        inp[r1, c0:c1 + 1] = FRAME
        out[:] = inp

        target_coords: list[tuple[int, int]] = []
        for rr, cc in np.argwhere(mask):
            tr = frame_r + row0 + int(rr)
            tc = frame_c + col0 + int(cc)
            target_coords.append((tr, tc))

        source_coords = _reflect_coords(np.array(target_coords, dtype=int), (r0, c0, r1, c1), side)
        if any(not (0 <= sr < patch_h and 0 <= sc < patch_w) for sr, sc in source_coords):
            continue
        if any(r0 <= sr <= r1 and c0 <= sc <= c1 for sr, sc in source_coords):
            continue

        for sr, sc in source_coords:
            inp[sr, sc] = GREY
        _paint_reflection(out, target_coords, (r0, c0, r1, c1))

        if np.array_equal(main(inp), out):
            return inp, out
    return None


def _rect_fits(
    rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int = 1
) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = patch != BG
    dst_view = dst[top : top + h, left : left + w]
    dst_view[view] = patch[view]


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(20, 28)
    grid_w = random.randint(20, 28)
    count = random.randint(2, 4)

    sides = [random.choice(("left", "right")), random.choice(("top", "bottom"))]
    while len(sides) < count:
        sides.append(random.choice(SIDES))
    random.shuffle(sides)

    units: list[tuple[np.ndarray, np.ndarray]] = []
    for side in sides:
        unit = _build_unit(side)
        if unit is None:
            return None
        units.append(unit)
    order = sorted(range(count), key=lambda i: units[i][0].shape[0] * units[i][0].shape[1], reverse=True)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    for idx in order:
        unit_in, unit_out = units[idx]
        uh, uw = unit_in.shape
        positions = []
        for r0 in range(grid_h - uh + 1):
            for c0 in range(grid_w - uw + 1):
                rect = (r0, c0, r0 + uh - 1, c0 + uw - 1)
                if _rect_fits(rect, placed, gap=1):
                    positions.append((r0, c0, rect))
        if not positions:
            return None
        r0, c0, rect = random.choice(positions)
        _paste(inp, unit_in, r0, c0)
        _paste(out, unit_out, r0, c0)
        placed.append(rect)

    return inp, out


def _validate_example(inp: np.ndarray, expected: np.ndarray) -> bool:
    frames = _find_frames(inp)
    sources = _find_sources(inp)
    if len(frames) != len(sources) or len(frames) < 2:
        return False

    orientation_kinds: set[str] = set()
    has_even_depth = False

    for src in sources:
        mask = _mask_in_bbox(src)
        candidates = _assignment_candidates(_bbox(src), frames)
        if len(candidates) != 1:
            return False
        frame_idx, side, _gap = candidates[0]
        if side in {"left", "right"}:
            orientation_kinds.add("horizontal")
            if np.array_equal(mask, np.fliplr(mask)):
                return False
        else:
            orientation_kinds.add("vertical")
            if np.array_equal(mask, np.flipud(mask)):
                return False

        reflected = _reflect_coords(src, frames[frame_idx]["bbox"], side)
        for r, c in reflected:
            r0, c0, r1, c1 = frames[frame_idx]["bbox"]
            if not (r0 < r < r1 and c0 < c < c1):
                return False
            if _layer_depth(frames[frame_idx]["bbox"], r, c) % 2 == 0:
                has_even_depth = True

    if orientation_kinds != {"horizontal", "vertical"}:
        return False
    if not has_even_depth:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_example()
        if built is None:
            continue
        inp, expected = built
        if _validate_example(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
