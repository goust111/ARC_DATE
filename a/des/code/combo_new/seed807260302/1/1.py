from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
GENERATION_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0, "cannot trim an empty mask"
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == int(bg):
                continue

            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == int(bg):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "colors": set(int(grid[int(r), int(c)]) for r, c in coords),
                }
            )
    return comps


def _contiguous(values: np.ndarray) -> bool:
    vals = sorted(int(v) for v in values)
    return bool(vals) and vals[-1] - vals[0] + 1 == len(vals)


def _frame_from_component(g: np.ndarray, comp: dict, bg: int = BG) -> dict | None:
    if len(comp["colors"]) != 1:
        return None

    color = int(next(iter(comp["colors"])))
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    if not bool(np.all((patch == int(bg)) | (patch == color))):
        return None
    if not bool(np.all(patch[1:-1, 1:-1] == int(bg))):
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    comp_mask = np.zeros((h, w), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0])
    comp_mask[rel[:, 0], rel[:, 1]] = True
    if bool(np.any(comp_mask & ~border)):
        return None

    missing = border & ~comp_mask
    if not bool(missing.any()):
        return None
    if not bool(np.all(patch[missing] == int(bg))):
        return None
    if not bool(np.array_equal(comp_mask, border & ~missing)):
        return None

    miss = np.argwhere(missing)
    side = None
    if bool(np.all(miss[:, 0] == 0)):
        side = "top"
        span = miss[:, 1]
        if int(span.min()) == 0 or int(span.max()) == w - 1:
            return None
        gap_coords = [(r0, c0 + int(c)) for c in sorted(span)]
    elif bool(np.all(miss[:, 0] == h - 1)):
        side = "bottom"
        span = miss[:, 1]
        if int(span.min()) == 0 or int(span.max()) == w - 1:
            return None
        gap_coords = [(r1, c0 + int(c)) for c in sorted(span)]
    elif bool(np.all(miss[:, 1] == 0)):
        side = "left"
        span = miss[:, 0]
        if int(span.min()) == 0 or int(span.max()) == h - 1:
            return None
        gap_coords = [(r0 + int(r), c0) for r in sorted(span)]
    elif bool(np.all(miss[:, 1] == w - 1)):
        side = "right"
        span = miss[:, 0]
        if int(span.min()) == 0 or int(span.max()) == h - 1:
            return None
        gap_coords = [(r0 + int(r), c1) for r in sorted(span)]
    else:
        return None

    if side is None or len(gap_coords) < 2 or not _contiguous(span):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "color": color,
        "side": side,
        "gap_coords": gap_coords,
        "gap_mask": _trim_mask(missing),
    }


def _piece_from_component(comp: dict) -> dict:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return {"coords": comp["coords"], "mask": _trim_mask(mask)}


def _piece_matches_gap(piece_mask: np.ndarray, gap_mask: np.ndarray) -> bool:
    for k in range(4):
        rotated = _trim_mask(np.rot90(piece_mask, int(k)))
        if rotated.shape == gap_mask.shape and bool(np.array_equal(rotated, gap_mask)):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each external sprite into the matching open side of a hollow frame and recolor it as that frame.

    Input structure:
    - The background is black.
    - The grid contains several monochrome hollow rectangular frames; each frame border has exactly one contiguous gap on one side.
    - Outside the frames are separated monochrome sprites whose colors do not identify their destination.
    - Each frame gap has exactly one external sprite whose shape matches the missing border cells under a 0/90/180/270 rotation.

    Rule:
    - Detect every frame, its gap side, and the precise missing border-cell mask.
    - Match each frame to the only unused external sprite whose mask can be rotated to equal that gap mask.
    - Clear every sprite from its original position.
    - Fill each frame gap with the frame color, so the inserted sprite becomes part of that frame border.

    Key insight:
    The pairing is by rotated shape rather than by color; after placement the sprite takes the frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    frames: list[dict] = []
    pieces: list[dict] = []
    for comp in _components(g, BG):
        frame = _frame_from_component(g, comp, BG)
        if frame is None:
            pieces.append(_piece_from_component(comp))
        else:
            frames.append(frame)

    assert frames, "expected at least one frame"
    assert len(pieces) == len(frames), "each frame must have exactly one external sprite"

    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    used: set[int] = set()
    for frame in frames:
        matches = [
            idx
            for idx, piece in enumerate(pieces)
            if idx not in used and _piece_matches_gap(piece["mask"], frame["gap_mask"])
        ]
        assert len(matches) == 1, "each frame gap must have a unique matching sprite"
        used.add(int(matches[0]))
        for r, c in frame["gap_coords"]:
            out[int(r), int(c)] = int(frame["color"])

    assert len(used) == len(frames), "every sprite must be used"
    return out


# ------------------------- generator -------------------------


def _mark_rect(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rows, cols = reserved.shape
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(rows, int(r0 + h + pad))
    cc1 = min(cols, int(c0 + w + pad))
    reserved[rr0:rr1, cc0:cc1] = True


def _mark_cells(reserved: np.ndarray, cells: list[tuple[int, int]], pad: int) -> None:
    rows, cols = reserved.shape
    for r, c in cells:
        rr0 = max(0, int(r - pad))
        rr1 = min(rows, int(r + pad + 1))
        cc0 = max(0, int(c - pad))
        cc1 = min(cols, int(c + pad + 1))
        reserved[rr0:rr1, cc0:cc1] = True


def _can_place_rect(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rows, cols = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(rows, int(r0 + h + pad))
    cc1 = min(cols, int(c0 + w + pad))
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    side: str,
    gap_start: int,
    gap_len: int,
) -> list[tuple[int, int]]:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)

    if side == "top":
        gap = [(r0, c0 + offset) for offset in range(gap_start, gap_start + gap_len)]
    elif side == "bottom":
        gap = [(r1, c0 + offset) for offset in range(gap_start, gap_start + gap_len)]
    elif side == "left":
        gap = [(r0 + offset, c0) for offset in range(gap_start, gap_start + gap_len)]
    else:
        gap = [(r0 + offset, c1) for offset in range(gap_start, gap_start + gap_len)]

    for r, c in gap:
        grid[int(r), int(c)] = BG
    return gap


def _candidate_positions(rows: int, cols: int, h: int, w: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
    random.shuffle(positions)
    return positions


def _bar_mask(length: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        return np.ones((1, int(length)), dtype=bool)
    return np.ones((int(length), 1), dtype=bool)


def _can_place_sprite(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    r0: int,
    c0: int,
) -> bool:
    rows, cols = grid.shape
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if int(grid[rr, cc]) != BG or bool(reserved[rr, cc]):
            return False
    return True


def _place_sprite(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> list[tuple[int, int]] | None:
    rows, cols = grid.shape
    h, w = mask.shape
    for r0, c0 in _candidate_positions(rows, cols, h, w):
        if not _can_place_sprite(grid, reserved, mask, int(r0), int(c0)):
            continue
        cells: list[tuple[int, int]] = []
        for dr, dc in np.argwhere(mask):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            grid[rr, cc] = int(color)
            cells.append((rr, cc))
        _mark_cells(reserved, cells, pad=1)
        return cells
    return None


def _expected_output(inp: np.ndarray, frame_specs: list[dict], sprite_cells: list[list[tuple[int, int]]]) -> np.ndarray:
    out = inp.copy()
    for cells in sprite_cells:
        for r, c in cells:
            out[int(r), int(c)] = BG
    for spec in frame_specs:
        for r, c in spec["gap"]:
            out[int(r), int(c)] = int(spec["color"])
    return out


def _sample_frame_spec(length: int, color: int) -> dict:
    side = random.choice(SIDES)
    if side in ("top", "bottom"):
        w = random.randint(max(5, int(length) + 2), 9)
        h = random.randint(5, 9)
        gap_start = random.randint(1, w - int(length) - 1)
    else:
        h = random.randint(max(5, int(length) + 2), 9)
        w = random.randint(5, 9)
        gap_start = random.randint(1, h - int(length) - 1)
    return {
        "side": side,
        "h": int(h),
        "w": int(w),
        "length": int(length),
        "gap_start": int(gap_start),
        "color": int(color),
    }


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    min_side = 18 + 2 * n_frames
    rows = random.randint(min_side, 30)
    cols = random.randint(min_side, 30)
    grid = np.full((rows, cols), BG, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_colors = palette[:n_frames]
    sprite_palette = palette[n_frames:] + palette[:n_frames]
    lengths = random.sample(list(range(2, 7)), n_frames)

    frame_specs = [_sample_frame_spec(lengths[i], frame_colors[i]) for i in range(n_frames)]
    random.shuffle(frame_specs)

    for spec in frame_specs:
        placed = False
        for r0, c0 in _candidate_positions(rows, cols, spec["h"], spec["w"]):
            if not _can_place_rect(reserved, int(r0), int(c0), spec["h"], spec["w"], pad=1):
                continue
            spec["r0"] = int(r0)
            spec["c0"] = int(c0)
            spec["gap"] = _draw_frame(
                grid,
                int(r0),
                int(c0),
                spec["h"],
                spec["w"],
                spec["color"],
                spec["side"],
                spec["gap_start"],
                spec["length"],
            )
            _mark_rect(reserved, int(r0), int(c0), spec["h"], spec["w"], pad=1)
            placed = True
            break
        if not placed:
            return None

    sprite_cells: list[list[tuple[int, int]]] = []
    random.shuffle(frame_specs)
    for i, spec in enumerate(frame_specs):
        horizontal_gap = spec["side"] in ("top", "bottom")
        base = _bar_mask(spec["length"], horizontal=horizontal_gap)
        sprite_mask = _trim_mask(np.rot90(base, random.randint(0, 3)))

        choices = [c for c in sprite_palette if int(c) != int(spec["color"])]
        if not choices:
            return None
        sprite_color = int(random.choice(choices))
        cells = _place_sprite(grid, reserved, sprite_mask, sprite_color)
        if cells is None:
            return None
        sprite_cells.append(cells)

    expected = _expected_output(grid, frame_specs, sprite_cells)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
