import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PALETTE = list(range(10))
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_CW = {"top": 0, "right": 1, "bottom": 2, "left": 3}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_from_component(
    grid: np.ndarray, bg: int, color: int, coords: np.ndarray
) -> Optional[Dict[str, object]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    size = int(r1 - r0 + 1)
    if size < 4 or int(c1 - c0 + 1) != size:
        return None
    if int(coords.shape[0]) != int(4 * size - 5):
        return None

    border_mask = np.zeros((size, size), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    comp_mask = np.zeros((size, size), dtype=bool)
    for r, c in coords:
        rr, cc = int(r - r0), int(c - c0)
        if not bool(border_mask[rr, cc]):
            return None
        comp_mask[rr, cc] = True

    missing = np.argwhere(border_mask & (~comp_mask))
    if int(missing.shape[0]) != 1:
        return None

    gr, gc = map(int, missing[0])
    if (gr in (0, size - 1)) and (gc in (0, size - 1)):
        return None
    gap_r, gap_c = int(r0 + gr), int(c0 + gc)
    if int(grid[gap_r, gap_c]) != int(bg):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None
    inner_colors = {int(v) for v in inner.ravel() if int(v) != int(bg)}
    if not (2 <= len(inner_colors) <= 3):
        return None

    if gr == 0:
        side = "top"
    elif gc == size - 1:
        side = "right"
    elif gr == size - 1:
        side = "bottom"
    else:
        side = "left"

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(color),
        "side": side,
    }


def _find_frames(grid: np.ndarray) -> Tuple[int, List[Dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: List[Dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for coords in _components(g == int(color)):
            frame = _frame_from_component(g, bg, int(color), coords)
            if frame is not None:
                frames.append(frame)

    frames.sort(key=lambda item: tuple(int(x) for x in item["bbox"]))
    used = np.zeros(g.shape, dtype=bool)
    unique_frames: List[Dict[str, object]] = []
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        if bool(used[r0 : r1 + 1, c0 : c1 + 1].any()):
            continue
        used[r0 : r1 + 1, c0 : c1 + 1] = True
        unique_frames.append(frame)
    if not unique_frames:
        raise AssertionError("no square frame with a single border gap found")
    return int(bg), unique_frames


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, frames = _find_frames(g)
    out = np.full(g.shape, int(bg), dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        turns = int(SIDE_TO_CW[str(frame["side"])])
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(inner, (-turns) % 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed interior pattern according to the side containing the frame's single-cell gap.

    Input structure:
    - The background is the most common color.
    - The grid contains several one-pixel-thick monochrome square frames.
    - Each frame has exactly one background-colored gap on one of its four sides.
    - Inside every frame is a non-symmetric multicolor pattern occupying part of the inner square.

    Rule:
    - Read the gap position as a rotation code: top=0 degrees, right=90 degrees clockwise, bottom=180 degrees, left=270 degrees clockwise.
    - Rotate the entire inner square region of each frame by the encoded amount.
    - Remove every frame from the output and keep only the rotated inner patterns.
    - Preserve each rotated pattern at the original inner-region location on a background-colored canvas.

    Key insight:
    - The border gap is not a passage; its side encodes how much to rotate the pattern enclosed by that frame.
    """
    return _transform(grid)


def _boxes_too_close(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], pad: int) -> bool:
    return not (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _grow_connected_cells(rng: random.Random, size: int, area: int) -> Optional[List[Tuple[int, int]]]:
    start = (rng.randrange(size), rng.randrange(size))
    cells = [start]
    cell_set = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cell_set:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nxt = rng.choice(list(frontier))
        cell_set.add(nxt)
        cells.append(nxt)
    return cells


def _random_pattern(
    rng: random.Random, size: int, bg: int, palette: Sequence[int]
) -> Optional[np.ndarray]:
    if size < 3:
        return None
    for _ in range(200):
        patch = np.full((size, size), int(bg), dtype=int)
        color_count = rng.randint(2, 3)
        colors = list(rng.sample(list(palette), color_count))
        area_hi = min(size * size - 1, color_count + size + 1)
        area = rng.randint(color_count + 1, area_hi)
        cells = _grow_connected_cells(rng, size, area)
        if cells is None:
            continue
        rng.shuffle(cells)
        for idx, color in enumerate(colors):
            r, c = cells[idx]
            patch[r, c] = int(color)
        for r, c in cells[color_count:]:
            patch[r, c] = int(rng.choice(colors))
        if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            continue
        return patch
    return None


def _gap_cell(side: str, bbox: Tuple[int, int, int, int], offset: int) -> Tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    if side == "top":
        return (int(r0), int(c0 + offset))
    if side == "right":
        return (int(r0 + offset), int(c1))
    if side == "bottom":
        return (int(r1), int(c0 + offset))
    return (int(r0 + offset), int(c0))


def _draw_frame(grid: np.ndarray, bbox: Tuple[int, int, int, int], color: int, side: str, offset: int, bg: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    gr, gc = _gap_cell(side, bbox, offset)
    grid[gr, gc] = int(bg)


def _strict_bg(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(vals, counts)}
    bg_count = int(count_map.get(int(bg), 0))
    other_counts = [count for color, count in count_map.items() if int(color) != int(bg)]
    return bool(other_counts) and all(int(bg_count) > int(count) for count in other_counts)


def _place_boxes(rng: random.Random, h: int, w: int, sizes: Sequence[int]) -> Optional[List[Tuple[int, int, int, int]]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for size in sizes:
        if h - size - 1 < 1 or w - size - 1 < 1:
            return None
        placed = False
        for _ in range(300):
            r0 = rng.randint(1, h - size - 1)
            c0 = rng.randint(1, w - size - 1)
            bbox = (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1))
            if any(_boxes_too_close(bbox, other, pad=1) for other in boxes):
                continue
            boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    frame_count = rng.choices([2, 3, 4], weights=[2, 3, 3], k=1)[0]
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    bg = rng.choice(PALETTE)

    frame_colors = rng.sample([c for c in PALETTE if c != bg], frame_count)
    pattern_palette = [c for c in PALETTE if c != bg and c not in frame_colors]
    items: List[Dict[str, object]] = []
    for frame_color, side in zip(frame_colors, rng.sample(list(SIDES), frame_count)):
        size = rng.randint(5, 9)
        pattern = _random_pattern(rng, size - 2, bg, pattern_palette)
        if pattern is None:
            return None
        items.append(
            {
                "size": int(size),
                "frame_color": int(frame_color),
                "side": side,
                "gap_offset": int(rng.randint(1, size - 2)),
                "output_patch": pattern,
            }
        )

    items.sort(key=lambda item: int(item["size"]), reverse=True)
    boxes = _place_boxes(rng, h, w, [int(item["size"]) for item in items])
    if boxes is None:
        return None

    inp = np.full((h, w), int(bg), dtype=int)
    expected = np.full((h, w), int(bg), dtype=int)

    for item, bbox in zip(items, boxes):
        item["bbox"] = bbox
        _draw_frame(
            inp,
            bbox,
            int(item["frame_color"]),
            str(item["side"]),
            int(item["gap_offset"]),
            int(bg),
        )
        r0, c0, r1, c1 = map(int, bbox)
        output_patch = np.asarray(item["output_patch"], dtype=int)
        turns = int(SIDE_TO_CW[str(item["side"])])
        input_patch = np.rot90(output_patch, turns)
        inp[r0 + 1 : r1, c0 + 1 : c1] = input_patch
        expected[r0 + 1 : r1, c0 + 1 : c1] = output_patch

    if not _strict_bg(inp, bg):
        return None
    try:
        found_bg, frames = _find_frames(inp)
    except AssertionError:
        return None
    if int(found_bg) != int(bg) or len(frames) != frame_count:
        return None
    if not np.array_equal(_transform(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to generate a valid sample")
