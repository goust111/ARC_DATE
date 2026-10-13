from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GEN_RETRIES = 800
WORK_SIZE = 30


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_true(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise AssertionError("mask must contain at least one true cell")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(), (int(r0), int(c0))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    packed = np.asarray(mask, dtype=np.uint8)
    return tuple(int(x) for x in mask.shape), packed.tobytes()


def _looks_like_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    if not (
        np.all(mask[0, :])
        and np.all(mask[-1, :])
        and np.all(mask[:, 0])
        and np.all(mask[:, -1])
    ):
        return False
    return bool((~mask[1:-1, 1:-1]).any())


def _frame_info(grid: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(grid == color)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not (
        np.all(box[0, :] == color)
        and np.all(box[-1, :] == color)
        and np.all(box[:, 0] == color)
        and np.all(box[:, -1] == color)
    ):
        return None
    if not np.all((box == BG) | (box == color)):
        return None

    inner = box[1:-1, 1:-1]
    if inner.size == 0:
        return None
    hole_full = inner == BG
    if not bool(hole_full.any()):
        return None
    if len(_components(hole_full)) != 1:
        return None

    hole_shape, (hole_r0, hole_c0) = _crop_true(hole_full)
    hole_abs = [(r0 + 1 + r, c0 + 1 + c) for r, c in np.argwhere(hole_full)]
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "inner_shape": tuple(int(x) for x in inner.shape),
        "hole_full": hole_full,
        "hole_shape": hole_shape,
        "hole_offset": (int(hole_r0), int(hole_c0)),
        "hole_abs": hole_abs,
    }


def _piece_info(grid: np.ndarray, color: int) -> dict:
    mask = grid == color
    comps = _components(mask)
    if len(comps) != 1:
        raise AssertionError("each piece color must form exactly one component")
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    shape = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "shape": shape,
        "abs": coords,
    }


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    frames: list[dict] = []
    frame_colors: set[int] = set()
    for color in colors:
        info = _frame_info(grid, color)
        if info is not None:
            frames.append(info)
            frame_colors.add(int(color))
    if not frames:
        raise AssertionError("scene must contain at least one frame")

    pieces: list[dict] = []
    for color in colors:
        if color in frame_colors:
            continue
        pieces.append(_piece_info(grid, color))

    if len(frames) != len(pieces):
        raise AssertionError("piece count must equal frame count")

    frame_keys = [_mask_key(frame["hole_shape"]) for frame in frames]
    if len(frame_keys) != len(set(frame_keys)):
        raise AssertionError("frame hole shapes must be unique")

    piece_keys = [_mask_key(piece["shape"]) for piece in pieces]
    if len(piece_keys) != len(set(piece_keys)):
        raise AssertionError("piece shapes must be unique")

    return frames, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered fragments to the black gaps inside colored frames and refill the gaps with frame colors.

    Input structure:
    - Black background with several one-pixel-thick rectangular frames, each drawn in its own color.
    - Inside every frame, some interior cells are already filled with the frame color while one 4-connected black gap remains.
    - Elsewhere in the grid, each remaining non-black color forms one free fragment whose shape matches exactly one frame gap.

    Rule:
    - Detect each frame from its rectangular border and extract the unique black component inside the frame interior as that frame's gap shape.
    - Extract each free fragment as a shape mask normalized to its own bounding box.
    - Match fragments to gaps by exact shape equality only; translation is allowed but rotation and reflection are not.
    - Erase every fragment from its original location and fill the matched gap with the corresponding frame color.

    Key insight:
    The important black cells are missing interior patches of the frames rather than ordinary background, and fragment colors are only locators: exact shape matching decides the destination, after which the fragment adopts the frame color.
    """
    g = np.asarray(grid, dtype=int)
    frames, pieces = _analyze_scene(g)

    hole_to_frame = {_mask_key(frame["hole_shape"]): frame for frame in frames}
    out = g.copy()
    used: set[tuple[tuple[int, int], bytes]] = set()

    for piece in pieces:
        key = _mask_key(piece["shape"])
        if key not in hole_to_frame or key in used:
            raise AssertionError("every piece must match a unique frame hole")
        frame = hole_to_frame[key]
        used.add(key)
        for r, c in piece["abs"]:
            out[r, c] = BG
        for r, c in frame["hole_abs"]:
            out[r, c] = int(frame["color"])

    return out


def _random_connected_hole(rng: random.Random, inner_h: int, inner_w: int) -> np.ndarray:
    total = inner_h * inner_w
    min_area = 2
    max_area = total - 2 if total >= 4 else total - 1
    if max_area < min_area:
        max_area = min_area
    target = rng.randint(min_area, max_area)

    start = (rng.randrange(inner_h), rng.randrange(inner_w))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < inner_h and 0 <= nc < inner_w:
            frontier.add((nr, nc))

    while len(cells) < target:
        if not frontier:
            break
        nxt = rng.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        for nr, nc in _neighbors4(*nxt):
            if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) < target:
        for r in range(inner_h):
            for c in range(inner_w):
                if (r, c) in cells:
                    continue
                if any((nr, nc) in cells for nr, nc in _neighbors4(r, c) if 0 <= nr < inner_h and 0 <= nc < inner_w):
                    cells.add((r, c))
                if len(cells) >= target:
                    break
            if len(cells) >= target:
                break

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _build_frame_patch(frame_color: int, hole_full: np.ndarray) -> np.ndarray:
    inner_h, inner_w = hole_full.shape
    patch = np.full((inner_h + 2, inner_w + 2), frame_color, dtype=int)
    patch[1:-1, 1:-1][hole_full] = BG
    return patch


def _crop_piece_patch(piece_color: int, hole_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cropped, _ = _crop_true(hole_full)
    patch = np.zeros(cropped.shape, dtype=int)
    patch[cropped] = piece_color
    return patch, cropped


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    H, W = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    H, W = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    reserved[r0:r1, c0:c1] = True


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = dst[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _crop_canvas(rng: random.Random, grid: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_pad = rng.randint(0, min(int(r0), 3))
    left_pad = rng.randint(0, min(int(c0), 3))
    bottom_pad = rng.randint(0, min(int(grid.shape[0] - 1 - r1), 3))
    right_pad = rng.randint(0, min(int(grid.shape[1] - 1 - c1), 3))
    rr0 = int(r0) - top_pad
    cc0 = int(c0) - left_pad
    rr1 = int(r1) + bottom_pad
    cc1 = int(c1) + right_pad
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), expected[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.choice((2, 3, 4))
    colors = rng.sample(list(range(1, 10)), 2 * num_frames)
    frame_colors = colors[:num_frames]
    piece_colors = colors[num_frames:]

    specs = []
    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    for idx in range(num_frames):
        frame_color = frame_colors[idx]
        piece_color = piece_colors[idx]
        built = False
        for _ in range(120):
            inner_h = rng.randint(3, 7)
            inner_w = rng.randint(3, 7)
            hole_full = _random_connected_hole(rng, inner_h, inner_w)
            if len(_components(hole_full)) != 1:
                continue
            if int(hole_full.sum()) <= 1 or int(hole_full.sum()) >= inner_h * inner_w:
                continue
            cropped_hole, _ = _crop_true(hole_full)
            if _looks_like_frame(cropped_hole):
                continue
            key = _mask_key(cropped_hole)
            if key in used_keys:
                continue
            used_keys.add(key)
            frame_patch = _build_frame_patch(frame_color, hole_full)
            piece_patch, piece_shape = _crop_piece_patch(piece_color, hole_full)
            specs.append(
                {
                    "frame_color": int(frame_color),
                    "piece_color": int(piece_color),
                    "hole_full": hole_full,
                    "hole_shape": piece_shape,
                    "frame_patch": frame_patch,
                    "piece_patch": piece_patch,
                }
            )
            built = True
            break
        if not built:
            return None

    canvas = np.zeros((WORK_SIZE, WORK_SIZE), dtype=int)
    expected = np.zeros((WORK_SIZE, WORK_SIZE), dtype=int)
    reserved = np.zeros((WORK_SIZE, WORK_SIZE), dtype=bool)

    objects = []
    for idx, spec in enumerate(specs):
        fh, fw = spec["frame_patch"].shape
        ph, pw = spec["piece_patch"].shape
        objects.append(("frame", idx, fh, fw))
        objects.append(("piece", idx, ph, pw))
    objects.sort(key=lambda item: (max(item[2], item[3]), item[2] * item[3]), reverse=True)

    placements: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, h, w in objects:
        placed = False
        for _ in range(250):
            top = rng.randint(0, WORK_SIZE - h)
            left = rng.randint(0, WORK_SIZE - w)
            if not _can_place(reserved, top, left, h, w):
                continue
            _reserve(reserved, top, left, h, w)
            placements[(kind, idx)] = (top, left)
            patch = specs[idx]["frame_patch"] if kind == "frame" else specs[idx]["piece_patch"]
            _paste(canvas, patch, top, left)
            _paste(expected, patch, top, left)
            placed = True
            break
        if not placed:
            return None

    for idx, spec in enumerate(specs):
        frame_top, frame_left = placements[("frame", idx)]
        piece_top, piece_left = placements[("piece", idx)]
        hole_full = spec["hole_full"]
        inner = expected[frame_top + 1 : frame_top + 1 + hole_full.shape[0], frame_left + 1 : frame_left + 1 + hole_full.shape[1]]
        inner[hole_full] = int(spec["frame_color"])
        piece_patch = spec["piece_patch"]
        piece_region = expected[piece_top : piece_top + piece_patch.shape[0], piece_left : piece_left + piece_patch.shape[1]]
        piece_region[piece_patch != BG] = BG

    inp, out = _crop_canvas(rng, canvas, expected)

    try:
        frames, pieces = _analyze_scene(inp)
    except AssertionError:
        return None
    if len(frames) != num_frames or len(pieces) != num_frames:
        return None
    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid puzzle after {GEN_RETRIES} attempts")
