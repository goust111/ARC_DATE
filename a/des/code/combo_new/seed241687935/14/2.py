import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
CANVAS_MIN = 20
CANVAS_MAX = 30
FRAME_MIN = 8
FRAME_MAX = 14
HOLE_COUNT_RANGE = (2, 5)
SHAPE_MAX_SIDE = 5
SHAPE_MAX_AREA = 7


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            mask, bbox = _coords_to_mask(arr)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "mask": mask,
                    "bbox": bbox,
                    "area": int(arr.shape[0]),
                    "signature": _shape_signature(mask),
                }
            )
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _hole_regions_in_component(grid: np.ndarray, bg: int, comp: dict) -> list[dict]:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((sub != bg) & (sub != comp["color"])):
        return []
    mask = comp["mask"]
    if not (mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
        return []

    holes = []
    for coords in _mask_components(sub == bg):
        if _touches_border(coords, sub.shape):
            continue
        hole_mask, hole_bbox_local = _coords_to_mask(coords)
        hr0, hc0, hr1, hc1 = hole_bbox_local
        global_coords = coords + np.asarray([r0, c0], dtype=int)
        holes.append(
            {
                "coords": global_coords,
                "bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
                "mask": hole_mask,
                "signature": _shape_signature(hole_mask),
                "area": int(global_coords.shape[0]),
            }
        )
    if not holes:
        return []
    if int(comp["area"]) + sum(h["area"] for h in holes) != sub.size:
        return []
    holes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return holes


def _extract_frame_and_holes(grid: np.ndarray, bg: int) -> tuple[dict, list[dict], list[dict]]:
    comps = _color_components(grid, bg)
    candidates = []
    for comp in comps:
        holes = _hole_regions_in_component(grid, bg, comp)
        if holes:
            candidates.append((comp, holes))
    assert candidates, "expected a frame component with enclosed background holes"
    frame, holes = max(candidates, key=lambda item: (item[0]["area"], len(item[1])))
    pieces = [comp for comp in comps if comp is not frame]
    return frame, holes, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered fragment into the uniquely matching hole of the holed frame.

    Input structure:
    - The background is the most common color.
    - One large monochrome rectangular component forms a filled frame whose interior contains several disjoint background holes.
    - Outside that frame are monochrome 4-connected fragments in colors different from the frame color.

    Rule:
    - Extract every enclosed background hole inside the large rectangle and normalize its exact shape.
    - Match each outside fragment to the unique hole with the same shape under translation only.
    - Fill each hole with the fragment's own color and erase the fragment from its original location.

    Key insight:
    - The meaningful objects to match are the frame's negative-space holes; each outside fragment is a translated copy of one such hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    _frame, holes, pieces = _extract_frame_and_holes(g, bg)
    assert len(holes) == len(pieces), "hole and fragment counts must match"

    by_signature: dict[tuple[tuple[int, int], bytes], dict] = {}
    for piece in pieces:
        sig = piece["signature"]
        assert sig not in by_signature, "fragment shapes must identify holes unambiguously"
        by_signature[sig] = piece

    out = g.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg

    for hole in holes:
        piece = by_signature[hole["signature"]]
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = piece["color"]

    return out


def _has_internal_hole(mask: np.ndarray) -> bool:
    for coords in _mask_components(~mask):
        if not _touches_border(coords, mask.shape):
            return True
    return False


def _random_connected_shape(max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        ph = random.randint(1, max_h)
        pw = random.randint(1, max_w)
        area_hi = min(SHAPE_MAX_AREA, ph * pw)
        if area_hi < 2:
            continue
        target = random.randint(2, area_hi)
        start = (random.randrange(ph), random.randrange(pw))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        arr = np.asarray(sorted(cells), dtype=int)
        mask, _ = _coords_to_mask(arr)
        if _has_internal_hole(mask):
            continue
        return mask
    raise RuntimeError("failed to build a connected shape")


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, padding: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        r0 = max(0, r - padding)
        r1 = min(h, r + padding + 1)
        c0 = max(0, c - padding)
        c1 = min(w, c + padding + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _all_positions(occupied: np.ndarray, mask: np.ndarray, padding: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    mh, mw = mask.shape
    positions = []
    for top in range(h - mh + 1):
        for left in range(w - mw + 1):
            if _can_place(occupied, mask, top, left, padding):
                positions.append((top, left))
    return positions


def _stamp_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, value: bool = True) -> None:
    rows, cols = np.argwhere(mask).T
    dst[top + rows, left + cols] = value


def _build_hole_layout(interior_h: int, interior_w: int, n_holes: int) -> list[dict] | None:
    shapes = []
    seen = set()
    max_h = min(SHAPE_MAX_SIDE, interior_h)
    max_w = min(SHAPE_MAX_SIDE, interior_w)
    for _ in range(400):
        if len(shapes) == n_holes:
            break
        mask = _random_connected_shape(max_h, max_w)
        sig = _shape_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(mask)
    if len(shapes) != n_holes:
        return None

    occupied = np.zeros((interior_h, interior_w), dtype=bool)
    holes = []
    for mask in sorted(shapes, key=lambda m: (-int(m.sum()), -m.shape[0], -m.shape[1])):
        positions = _all_positions(occupied, mask, padding=1)
        if not positions:
            return None
        top, left = random.choice(positions)
        _stamp_mask(occupied, mask, top, left)
        holes.append({"mask": mask, "top": int(top), "left": int(left), "area": int(mask.sum())})
    return holes


def _expected_output(inp: np.ndarray, bg: int, holes: list[dict], pieces: list[dict]) -> np.ndarray:
    out = inp.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg
    for hole in holes:
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = hole["piece_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(CANVAS_MIN, CANVAS_MAX)
    w = random.randint(CANVAS_MIN, CANVAS_MAX)
    fh = random.randint(FRAME_MIN, min(FRAME_MAX, h - 4))
    fw = random.randint(FRAME_MIN, min(FRAME_MAX, w - 4))
    interior_h, interior_w = fh - 2, fw - 2
    max_holes = min(HOLE_COUNT_RANGE[1], max(2, (interior_h * interior_w) // 9))
    if max_holes < HOLE_COUNT_RANGE[0]:
        return None
    n_holes = random.randint(HOLE_COUNT_RANGE[0], max_holes)

    holes_local = _build_hole_layout(interior_h, interior_w, n_holes)
    if holes_local is None:
        return None

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette.pop()
    frame_color = palette.pop()
    piece_colors = [palette.pop() for _ in range(n_holes)]
    random.shuffle(piece_colors)

    inp = np.full((h, w), bg, dtype=int)
    fr0 = random.randint(1, h - fh - 1)
    fc0 = random.randint(1, w - fw - 1)
    inp[fr0 : fr0 + fh, fc0 : fc0 + fw] = frame_color

    holes = []
    used_signatures = set()
    for spec, color in zip(holes_local, piece_colors):
        mask = spec["mask"]
        top = fr0 + 1 + spec["top"]
        left = fc0 + 1 + spec["left"]
        rows, cols = np.argwhere(mask).T
        inp[top + rows, left + cols] = bg
        coords = np.column_stack((top + rows, left + cols))
        sig = _shape_signature(mask)
        if sig in used_signatures:
            return None
        used_signatures.add(sig)
        holes.append(
            {
                "mask": mask,
                "coords": coords.astype(int),
                "piece_color": int(color),
                "signature": sig,
            }
        )

    occupied = np.zeros((h, w), dtype=bool)
    occupied[max(0, fr0 - 1) : min(h, fr0 + fh + 1), max(0, fc0 - 1) : min(w, fc0 + fw + 1)] = True

    piece_specs = []
    for hole in sorted(holes, key=lambda item: -int(item["mask"].sum())):
        mask = hole["mask"]
        positions = _all_positions(occupied, mask, padding=1)
        if not positions:
            return None
        top, left = random.choice(positions)
        rows, cols = np.argwhere(mask).T
        inp[top + rows, left + cols] = hole["piece_color"]
        piece_coords = np.column_stack((top + rows, left + cols)).astype(int)
        _stamp_mask(occupied, mask, top, left)
        piece_specs.append({"coords": piece_coords, "signature": hole["signature"]})

    if get_bg(inp) != bg:
        return None

    try:
        frame, solved_holes, pieces = _extract_frame_and_holes(inp, bg)
    except Exception:
        return None
    if frame["color"] != frame_color or len(solved_holes) != n_holes or len(pieces) != n_holes:
        return None

    out = _expected_output(inp, bg, holes, piece_specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
