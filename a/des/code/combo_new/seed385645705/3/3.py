from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for rr, cc in _neighbors4(r, c, h, w):
                if mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(cells: np.ndarray) -> tuple[int, int, int, int]:
    rows = cells[:, 0]
    cols = cells[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _trim(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "shape must be non-empty"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_from_cells(cells: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[cells[:, 0] - r0, cells[:, 1] - c0] = True
    return mask


def _key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    cropped = _trim(mask)
    pts = np.argwhere(cropped)
    return tuple((int(r), int(c)) for r, c in pts)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _trim(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for candidate in (rot, np.fliplr(rot)):
            q = _trim(candidate)
            key = _key(q)
            if key not in seen:
                seen.add(key)
                variants.append(q)
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_key(v) for v in _unique_variants(mask))


def _shape_key_from_cells(cells: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _canonical_key(_mask_from_cells(cells))


def _border_path(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    path: list[tuple[int, int]] = []
    path.extend((r0, c) for c in range(c0, c1 + 1))
    path.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    path.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    path.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return path


def _find_frames(g: np.ndarray) -> list[dict]:
    frames = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        assert np.all(g[r0, c0 : c1 + 1] == GREY), "frame top edge must be grey"
        assert np.all(g[r1, c0 : c1 + 1] == GREY), "frame bottom edge must be grey"
        assert np.all(g[r0 : r1 + 1, c0] == GREY), "frame left edge must be grey"
        assert np.all(g[r0 : r1 + 1, c1] == GREY), "frame right edge must be grey"

        crop_bg = g[r0 : r1 + 1, c0 : c1 + 1] == BG
        holes = []
        ch, cw = crop_bg.shape
        for local in _components(crop_bg):
            touches_crop_edge = any(
                int(r) == 0 or int(c) == 0 or int(r) == ch - 1 or int(c) == cw - 1
                for r, c in local
            )
            if touches_crop_edge:
                continue
            cells = local + np.array([r0, c0])
            hb = _bbox(cells)
            holes.append({"cells": cells, "bbox": hb, "key": _shape_key_from_cells(cells)})
        if holes:
            holes.sort(key=lambda h: (h["bbox"][0], h["bbox"][1]))
            frames.append({"bbox": (r0, c0, r1, c1), "holes": holes})
    assert frames, "expected at least one grey frame"
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1]))
    return frames


def _find_pieces(g: np.ndarray) -> list[dict]:
    pieces = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) not in (BG, GREY)):
        for cells in _components(g == color):
            pieces.append(
                {
                    "color": int(color),
                    "cells": cells,
                    "bbox": _bbox(cells),
                    "key": _shape_key_from_cells(cells),
                }
            )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match external colored fragments to black holes in grey frames and repaint the frame borders.

    Input structure:
    - The background is black, and grey pixels form one or more rectangular framework components with one-cell outer borders.
    - Each framework encloses one or more 4-connected black hole regions separated by grey material.
    - Outside the frameworks are monochrome 4-connected colored fragments; their colors are neither black nor grey.
    - The fragment count equals the total hole count, and every fragment shape matches exactly one hole up to rotation or reflection.

    Rule:
    - Canonicalize every hole and fragment under the eight rotate/flip variants and match equal shapes one-to-one.
    - Erase each fragment from its original position, then fill its matched hole with the fragment color.
    - For each frame, sort its holes from top to bottom and then left to right; repaint the frame's outer border clockwise from the top-left corner in equal consecutive segments using the matched fragment colors in that order.

    Key insight: Shape matching decides which fragments belong inside each frame; the same matches are then reused as labels for the otherwise grey border segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    pieces = _find_pieces(g)
    hole_count = sum(len(frame["holes"]) for frame in frames)
    assert len(pieces) == hole_count, "fragment count must equal hole count"

    piece_by_key: dict[tuple[tuple[int, int], ...], int] = {}
    for i, piece in enumerate(pieces):
        key = piece["key"]
        assert key not in piece_by_key, "fragment shapes must be unique"
        piece_by_key[key] = i

    used_pieces: set[int] = set()
    match: dict[tuple[int, int], dict] = {}
    for fi, frame in enumerate(frames):
        for hi, hole in enumerate(frame["holes"]):
            key = hole["key"]
            assert key in piece_by_key, "every hole must have a matching fragment"
            pi = piece_by_key[key]
            assert pi not in used_pieces, "matching must be one-to-one"
            used_pieces.add(pi)
            match[(fi, hi)] = pieces[pi]
    assert len(used_pieces) == len(pieces)

    out = g.copy()
    for piece in pieces:
        cells = piece["cells"]
        out[cells[:, 0], cells[:, 1]] = BG

    for fi, frame in enumerate(frames):
        for hi, hole in enumerate(frame["holes"]):
            color = int(match[(fi, hi)]["color"])
            cells = hole["cells"]
            out[cells[:, 0], cells[:, 1]] = color

    for fi, frame in enumerate(frames):
        colors = [int(match[(fi, hi)]["color"]) for hi in range(len(frame["holes"]))]
        path = _border_path(frame["bbox"])
        n = len(colors)
        for i, color in enumerate(colors):
            start = i * len(path) // n
            end = (i + 1) * len(path) // n
            for r, c in path[start:end]:
                out[r, c] = color

    return out


# --------------- generator ---------------

GRID_SIDE_RANGE = (18, 30)
TOTAL_HOLE_RANGE = (2, 6)
FRAME_COUNT_MAX = 3
HOLES_PER_FRAME_MAX = 3
FRAME_SIDE_RANGE = (5, 11)
SHAPE_SIDE_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 8)
GENERATION_ATTEMPTS = 600
SHAPE_ATTEMPTS = 300
FRAME_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200
OBJECT_PADDING = 1


def _random_connected_mask() -> np.ndarray | None:
    h = random.randint(*SHAPE_SIDE_RANGE)
    w = random.randint(*SHAPE_SIDE_RANGE)
    target = random.randint(SHAPE_AREA_RANGE[0], min(SHAPE_AREA_RANGE[1], h * w))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], h, w))
    while int(mask.sum()) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        for nb in _neighbors4(r, c, h, w):
            if not mask[nb]:
                frontier.add(nb)
    return _trim(mask)


def _sample_unique_shapes(n: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    keys: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(SHAPE_ATTEMPTS):
        if len(shapes) == n:
            return shapes
        mask = _random_connected_mask()
        if mask is None:
            continue
        key = _canonical_key(mask)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(mask)
    return None


def _hole_counts() -> list[int]:
    total = random.randint(*TOTAL_HOLE_RANGE)
    min_frames = (total + HOLES_PER_FRAME_MAX - 1) // HOLES_PER_FRAME_MAX
    n_frames = random.randint(min_frames, min(FRAME_COUNT_MAX, total))
    counts = [1] * n_frames
    remaining = total - n_frames
    while remaining > 0:
        choices = [i for i, c in enumerate(counts) if c < HOLES_PER_FRAME_MAX]
        i = random.choice(choices)
        counts[i] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _mark_padding(blocked: np.ndarray, cells: list[tuple[int, int]], padding: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - padding), min(h, r + padding + 1)):
            for cc in range(max(0, c - padding), min(w, c + padding + 1)):
                blocked[rr, cc] = True


def _build_frame_patch(shapes: list[np.ndarray]) -> dict | None:
    n = len(shapes)
    for _ in range(FRAME_ATTEMPTS):
        fh = random.randint(*FRAME_SIDE_RANGE)
        fw = random.randint(*FRAME_SIDE_RANGE)
        if (2 * fh + 2 * fw - 4) % n != 0:
            continue
        inner_h, inner_w = fh - 2, fw - 2
        order = list(range(n))
        random.shuffle(order)
        blocked = np.zeros((inner_h, inner_w), dtype=bool)
        patch = np.full((fh, fw), GREY, dtype=int)
        holes_by_shape: dict[int, np.ndarray] = {}

        ok = True
        for idx in order:
            mask = shapes[idx]
            mh, mw = mask.shape
            candidates = []
            for r0 in range(inner_h - mh + 1):
                for c0 in range(inner_w - mw + 1):
                    sub = blocked[r0 : r0 + mh, c0 : c0 + mw]
                    if not np.any(sub & mask):
                        candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            ir, ic = random.choice(candidates)
            true_cells = []
            for rr, cc in np.argwhere(mask):
                rr, cc = int(rr), int(cc)
                patch[1 + ir + rr, 1 + ic + cc] = BG
                true_cells.append((ir + rr, ic + cc))
            _mark_padding(blocked, true_cells, 1)
            local_cells = np.array(
                [(1 + r, 1 + c) for r, c in true_cells],
                dtype=int,
            )
            holes_by_shape[idx] = local_cells

        if not ok:
            continue
        holes = []
        for idx, cells in holes_by_shape.items():
            holes.append(
                {
                    "cells": cells,
                    "bbox": _bbox(cells),
                    "mask": shapes[idx],
                    "key": _canonical_key(shapes[idx]),
                }
            )
        holes.sort(key=lambda h: (h["bbox"][0], h["bbox"][1]))
        return {"patch": patch, "holes": holes}
    return None


def _rect_is_clear(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> bool:
    H, W = reserved.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve_rect(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> None:
    H, W = reserved.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    reserved[rr0:rr1, cc0:cc1] = True


def _place_frame(grid: np.ndarray, reserved: np.ndarray, frame: dict) -> dict | None:
    H, W = grid.shape
    patch = frame["patch"]
    fh, fw = patch.shape
    candidates = []
    for r0 in range(1, H - fh):
        for c0 in range(1, W - fw):
            if _rect_is_clear(reserved, r0, c0, fh, fw, OBJECT_PADDING):
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    crop = grid[r0 : r0 + fh, c0 : c0 + fw]
    crop[patch == GREY] = GREY
    _reserve_rect(reserved, r0, c0, fh, fw, OBJECT_PADDING)

    placed_holes = []
    for hole in frame["holes"]:
        cells = hole["cells"] + np.array([r0, c0])
        placed_holes.append(
            {
                "cells": cells,
                "bbox": _bbox(cells),
                "mask": hole["mask"],
                "key": hole["key"],
            }
        )
    placed_holes.sort(key=lambda h: (h["bbox"][0], h["bbox"][1]))
    return {"bbox": (r0, c0, r0 + fh - 1, c0 + fw - 1), "holes": placed_holes}


def _cells_clear_for_mask(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    r0: int,
    c0: int,
) -> bool:
    H, W = grid.shape
    for rr, cc in np.argwhere(mask):
        r = r0 + int(rr)
        c = c0 + int(cc)
        if not (0 <= r < H and 0 <= c < W):
            return False
        for pr in range(max(0, r - OBJECT_PADDING), min(H, r + OBJECT_PADDING + 1)):
            for pc in range(max(0, c - OBJECT_PADDING), min(W, c + OBJECT_PADDING + 1)):
                if reserved[pr, pc]:
                    return False
    return True


def _reserve_mask(reserved: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    H, W = reserved.shape
    cells = []
    for rr, cc in np.argwhere(mask):
        r = r0 + int(rr)
        c = c0 + int(cc)
        cells.append((r, c))
        for pr in range(max(0, r - OBJECT_PADDING), min(H, r + OBJECT_PADDING + 1)):
            for pc in range(max(0, c - OBJECT_PADDING), min(W, c + OBJECT_PADDING + 1)):
                reserved[pr, pc] = True
    return cells


def _place_piece(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> dict | None:
    H, W = grid.shape
    mh, mw = mask.shape
    candidates = []
    for r0 in range(H - mh + 1):
        for c0 in range(W - mw + 1):
            if _cells_clear_for_mask(grid, reserved, mask, r0, c0):
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    cells = _reserve_mask(reserved, mask, r0, c0)
    for r, c in cells:
        grid[r, c] = int(color)
    arr = np.array(cells, dtype=int)
    return {"cells": arr, "color": int(color), "mask": mask, "bbox": _bbox(arr)}


def _expected_output(inp: np.ndarray, frames: list[dict], pieces: list[dict]) -> np.ndarray:
    out = inp.copy()
    for piece in pieces:
        cells = piece["cells"]
        out[cells[:, 0], cells[:, 1]] = BG
    for frame in frames:
        for hole in frame["holes"]:
            color = int(hole["color"])
            cells = hole["cells"]
            out[cells[:, 0], cells[:, 1]] = color
        colors = [int(h["color"]) for h in frame["holes"]]
        path = _border_path(frame["bbox"])
        n = len(colors)
        for i, color in enumerate(colors):
            start = i * len(path) // n
            end = (i + 1) * len(path) // n
            for r, c in path[start:end]:
                out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    counts = _hole_counts()
    total_holes = sum(counts)
    shapes = _sample_unique_shapes(total_holes)
    if shapes is None:
        return None

    random.shuffle(shapes)
    frame_specs = []
    cursor = 0
    for count in counts:
        spec = _build_frame_patch(shapes[cursor : cursor + count])
        if spec is None:
            return None
        frame_specs.append(spec)
        cursor += count

    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((H, W), BG, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    placed_frames = []
    random.shuffle(frame_specs)
    for frame in frame_specs:
        placed = _place_frame(grid, reserved, frame)
        if placed is None:
            return None
        placed_frames.append(placed)
    placed_frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1]))

    all_holes = [hole for frame in placed_frames for hole in frame["holes"]]
    colors = random.sample(PIECE_COLORS, total_holes)
    for hole, color in zip(all_holes, colors):
        hole["color"] = int(color)

    piece_jobs = []
    for hole in all_holes:
        variants = _unique_variants(hole["mask"])
        mask = random.choice(variants)
        piece_jobs.append({"mask": mask, "color": int(hole["color"]), "key": hole["key"]})
    random.shuffle(piece_jobs)

    placed_pieces = []
    for job in piece_jobs:
        piece = _place_piece(grid, reserved, job["mask"], job["color"])
        if piece is None:
            return None
        placed_pieces.append(piece)

    expected = _expected_output(grid, placed_frames, placed_pieces)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
