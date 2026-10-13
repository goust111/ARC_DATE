import random

import numpy as np


_BG = 0
_FRAME = 5
_CANVAS = 30
_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_RETRY_LIMIT = 400
_KEY_LIBRARY = (
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1], [0, 1, 1]], dtype=bool),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("mask must contain at least one pixel")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _filled_rectangle(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _bbox_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] <= inner[0] <= inner[2] <= outer[2]
        and outer[1] <= inner[1] <= inner[3] <= outer[3]
    )


def _bbox_intersects(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _lock_side(lock_coords: list[tuple[int, int]], height: int, width: int) -> str | None:
    if all(r == 0 for r, _c in lock_coords):
        values = sorted(c for _r, c in lock_coords)
        return "top" if values == list(range(values[0], values[-1] + 1)) else None
    if all(r == height - 1 for r, _c in lock_coords):
        values = sorted(c for _r, c in lock_coords)
        return "bottom" if values == list(range(values[0], values[-1] + 1)) else None
    if all(c == 0 for _r, c in lock_coords):
        values = sorted(r for r, _c in lock_coords)
        return "left" if values == list(range(values[0], values[-1] + 1)) else None
    if all(c == width - 1 for _r, c in lock_coords):
        values = sorted(r for r, _c in lock_coords)
        return "right" if values == list(range(values[0], values[-1] + 1)) else None
    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 5 or width < 5:
            continue
        border = _border_mask(height, width)
        comp_mask = np.zeros((height, width), dtype=bool)
        for r, c in comp:
            comp_mask[r - r0, c - c0] = True
        if not np.array_equal(comp_mask, border):
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(int(v) for v in np.unique(sub[border]) if int(v) != bg)
        if len(colors) != 2:
            continue
        color_counts = {color: int(np.sum(sub[border] == color)) for color in colors}
        frame_color = max(color_counts, key=color_counts.get)
        lock_color = next(color for color in colors if color != frame_color)
        lock_coords = [
            tuple(map(int, rc)) for rc in np.argwhere(border & (sub == lock_color))
        ]
        side = _lock_side(lock_coords, height, width)
        if side is None:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "frame_color": int(frame_color),
                "lock_color": int(lock_color),
                "lock_side": side,
            }
        )
    if not frames:
        raise AssertionError("expected at least one locked frame")
    return frames


def _key_mask_for_color(
    grid: np.ndarray, color: int, frame_boxes: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    outside_components: list[list[tuple[int, int]]] = []
    for comp in _components(grid == color):
        box = _bbox(comp)
        if any(_bbox_intersects(box, frame_box) for frame_box in frame_boxes):
            continue
        outside_components.append(comp)
    if len(outside_components) != 1:
        raise AssertionError(f"expected one outside key for color {color}")
    key = outside_components[0]
    kr0, kc0, kr1, kc1 = _bbox(key)
    mask = np.zeros((kr1 - kr0 + 1, kc1 - kc0 + 1), dtype=bool)
    abs_cells: list[tuple[int, int]] = []
    for r, c in key:
        mask[r - kr0, c - kc0] = True
        abs_cells.append((r, c))
    return mask, abs_cells


def _cover_box_for_frame(grid: np.ndarray, frame: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["lock_color"])
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    cover_components = _components(interior == color)
    if len(cover_components) != 1:
        raise AssertionError(f"expected one cover block for color {color}")
    cover = cover_components[0]
    cr0, cc0, cr1, cc1 = _bbox(cover)
    cover_mask = np.zeros((cr1 - cr0 + 1, cc1 - cc0 + 1), dtype=bool)
    for rr, cc in cover:
        cover_mask[rr - cr0, cc - cc0] = True
    if not _filled_rectangle(cover_mask):
        raise AssertionError(f"expected a solid rectangular cover for color {color}")
    return r0 + 1 + cr0, c0 + 1 + cc0, r0 + 1 + cr1, c0 + 1 + cc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each external key shape to cut matching-color openings in the covered block inside its locked frame.

    Input structure:
    - The background is the dominant color.
    - Several hollow rectangular frames are drawn in a neutral frame color.
    - Each frame has one contiguous colored lock segment replacing part of one border side.
    - Outside the frames there is exactly one small connected key object of each lock color.
    - Inside each frame there is one solid rectangular cover block of that same color, separated from the border by background.

    Rule:
    - Pair every frame with the outside key whose color matches the lock segment.
    - Erase the outside key from the grid.
    - Align the key's bounding box top-left corner to the cover block's top-left corner inside the frame.
    - Wherever the aligned key has a pixel, remove the matching cover pixel to background; leave the rest of the frame, lock, and cover unchanged.

    Key insight:
    - Color establishes the lock-key pairing, but the real payload is the key's shape used as a top-left-aligned mask on the interior cover block.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    frame_boxes = [frame["bbox"] for frame in frames]
    out = g.copy()

    for frame in frames:
        color = int(frame["lock_color"])
        cover_r0, cover_c0, cover_r1, cover_c1 = _cover_box_for_frame(g, frame)
        key_mask, key_cells = _key_mask_for_color(g, color, frame_boxes)
        key_h, key_w = key_mask.shape
        cover_h = cover_r1 - cover_r0 + 1
        cover_w = cover_c1 - cover_c0 + 1
        if key_h > cover_h or key_w > cover_w:
            raise AssertionError(f"key for color {color} must fit inside its cover")

        for r, c in key_cells:
            out[r, c] = bg

        for dr, dc in map(tuple, np.argwhere(key_mask)):
            rr = cover_r0 + int(dr)
            cc = cover_c0 + int(dc)
            if out[rr, cc] != color:
                raise AssertionError(f"aligned key cell must land on the cover for color {color}")
            out[rr, cc] = bg

    return out


def _random_key_mask() -> np.ndarray:
    mask = random.choice(_KEY_LIBRARY).copy()
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return _crop_mask(mask.astype(bool))


def _lock_cells(side: str, height: int, width: int) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        length = random.randint(2, min(4, width - 2))
        start = random.randint(1, width - length - 1)
        row = 0 if side == "top" else height - 1
        return [(row, c) for c in range(start, start + length)]
    length = random.randint(2, min(4, height - 2))
    start = random.randint(1, height - length - 1)
    col = 0 if side == "left" else width - 1
    return [(r, col) for r in range(start, start + length)]


def _build_frame_patch(color: int, key_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    key_h, key_w = key_mask.shape
    cover_h = key_h + random.randint(0, 2)
    cover_w = key_w + random.randint(0, 2)
    outer_h_min = max(7, cover_h + 4)
    outer_w_min = max(7, cover_w + 4)
    outer_h = random.randint(outer_h_min, min(outer_h_min + 3, 12))
    outer_w = random.randint(outer_w_min, min(outer_w_min + 3, 12))

    patch = np.full((outer_h, outer_w), _BG, dtype=int)
    patch[0, :] = _FRAME
    patch[-1, :] = _FRAME
    patch[:, 0] = _FRAME
    patch[:, -1] = _FRAME

    cover_r0 = random.randint(2, outer_h - cover_h - 2)
    cover_c0 = random.randint(2, outer_w - cover_w - 2)
    patch[cover_r0 : cover_r0 + cover_h, cover_c0 : cover_c0 + cover_w] = color

    side = random.choice(("top", "bottom", "left", "right"))
    for rr, cc in _lock_cells(side, outer_h, outer_w):
        patch[rr, cc] = color

    meta = {
        "cover_bbox": (
            int(cover_r0),
            int(cover_c0),
            int(cover_r0 + cover_h - 1),
            int(cover_c0 + cover_w - 1),
        ),
        "key_mask": key_mask.copy(),
        "color": int(color),
    }
    return patch, meta


def _shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, _BG, dtype=int)
    patch[mask] = color
    return patch


def _reserve_and_place(
    reserved: np.ndarray, height: int, width: int, padding: int
) -> tuple[int, int] | None:
    grid_h, grid_w = reserved.shape
    for _ in range(200):
        r0 = random.randint(0, grid_h - height)
        c0 = random.randint(0, grid_w - width)
        rr0 = max(0, r0 - padding)
        cc0 = max(0, c0 - padding)
        rr1 = min(grid_h, r0 + height + padding)
        cc1 = min(grid_w, c0 + width + padding)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        reserved[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != _BG
    dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _non_bg_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != _BG)
    if coords.size == 0:
        raise AssertionError("grid must contain foreground pixels")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _non_bg_bbox(inp)
    top = random.randint(0, min(4, r0))
    left = random.randint(0, min(4, c0))
    bottom = random.randint(0, min(4, inp.shape[0] - 1 - r1))
    right = random.randint(0, min(4, inp.shape[1] - 1 - c1))
    rr0 = r0 - top
    cc0 = c0 - left
    rr1 = r1 + bottom
    cc1 = c1 + right
    return (
        inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(),
        out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(),
    )


def _valid_background(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_color = int(values[order[0]])
    if top_color != _BG:
        return False
    if len(order) > 1 and int(counts[order[0]]) == int(counts[order[1]]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        canvas = np.full((_CANVAS, _CANVAS), _BG, dtype=int)
        reserved = np.zeros((_CANVAS, _CANVAS), dtype=bool)
        placements: list[dict] = []
        n_frames = random.choice((1, 2, 3))
        colors = random.sample(_PALETTE, n_frames)
        ok = True

        for color in colors:
            key_mask = _random_key_mask()
            frame_patch, frame_meta = _build_frame_patch(color, key_mask)
            frame_pos = _reserve_and_place(reserved, frame_patch.shape[0], frame_patch.shape[1], 1)
            if frame_pos is None:
                ok = False
                break
            key_patch = _shape_patch(key_mask, color)
            key_pos = _reserve_and_place(reserved, key_patch.shape[0], key_patch.shape[1], 1)
            if key_pos is None:
                ok = False
                break

            fr0, fc0 = frame_pos
            kr0, kc0 = key_pos
            _paste(canvas, frame_patch, fr0, fc0)
            _paste(canvas, key_patch, kr0, kc0)

            cover_r0, cover_c0, _cover_r1, _cover_c1 = frame_meta["cover_bbox"]
            placements.append(
                {
                    "color": int(color),
                    "key_mask": key_mask.copy(),
                    "key_cells": [
                        (kr0 + int(r), kc0 + int(c)) for r, c in map(tuple, np.argwhere(key_mask))
                    ],
                    "cover_top_left": (fr0 + int(cover_r0), fc0 + int(cover_c0)),
                }
            )

        if not ok:
            continue

        expected = canvas.copy()
        for item in placements:
            color = int(item["color"])
            for r, c in item["key_cells"]:
                expected[r, c] = _BG
            cover_r0, cover_c0 = item["cover_top_left"]
            for dr, dc in map(tuple, np.argwhere(item["key_mask"])):
                rr = cover_r0 + int(dr)
                cc = cover_c0 + int(dc)
                if expected[rr, cc] != color:
                    ok = False
                    break
                expected[rr, cc] = _BG
            if not ok:
                break

        if not ok:
            continue

        inp, out = _crop_pair(canvas, expected)
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        if not _valid_background(inp):
            continue

        try:
            solved = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(solved, out):
            continue
        return inp

    raise RuntimeError("failed to generate a valid ARC input")
