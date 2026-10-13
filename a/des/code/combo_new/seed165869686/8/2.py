from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 4)
INNER_SIDE_RANGE = (3, 7)
OBJECT_PAD = 1


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        for pts in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(pts)
            comps.append(
                {
                    "color": int(color),
                    "coords": [(int(r), int(c)) for r, c in pts],
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(pts)),
                    "shape": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
                }
            )
    return comps


def _gap_local_coords(height: int, width: int, side: str, start: int, length: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, int(c)) for c in range(start, start + length)]
    if side == "bottom":
        return [(int(height - 1), int(c)) for c in range(start, start + length)]
    if side == "left":
        return [(int(r), 0) for r in range(start, start + length)]
    if side == "right":
        return [(int(r), int(width - 1)) for r in range(start, start + length)]
    raise ValueError(f"unknown side: {side}")


def _frame_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    coords = set(comp["coords"])
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != int(bg)):
        return None

    missing: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if not on_border:
                continue
            if (r, c) in coords:
                continue
            if int(grid[r, c]) != int(bg):
                return None
            missing.append((int(r), int(c)))

    if not missing:
        return None

    mr0, mc0, mr1, mc1 = _bbox(missing)
    if mr0 == mr1 and mr0 in (r0, r1):
        expected = {(int(mr0), int(c)) for c in range(mc0, mc1 + 1)}
    elif mc0 == mc1 and mc0 in (c0, c1):
        expected = {(int(r), int(mc0)) for r in range(mr0, mr1 + 1)}
    else:
        return None

    if set(missing) != expected:
        return None

    return {
        "color": int(comp["color"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_coords": sorted(expected),
        "gap_bbox": (int(mr0), int(mc0), int(mr1), int(mc1)),
        "gap_shape": (int(mr1 - mr0 + 1), int(mc1 - mc0 + 1)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each loose monochrome piece into the matching frame-colored gap on a rectangular border.

    Input structure:
    - The background color is the most common color.
    - Several monochrome rectangular frames appear on the grid; each frame is a 1-pixel-thick border around a background interior, but one contiguous rectangular segment of that border is missing.
    - Separate from the frames are solid monochrome pieces whose colors match frame colors; each such piece has exactly the same shape as the missing border segment of its matching frame.

    Rule:
    - Detect each frame by finding a monochrome border-only rectangle with a background interior and a single contiguous missing segment on one side.
    - For every frame color, find the separate piece of the same color whose solid rectangle matches the frame gap's size and orientation.
    - Remove that piece from its original location and translate it onto the missing border cells, completing the frame while leaving all frame interiors as background.

    Key insight:
    - The loose piece does not belong inside the frame's hollow center; it restores the border itself, so color matching must be combined with gap-shape detection on the frame perimeter.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    comps = _monochrome_components(g, bg)

    frames: list[dict] = []
    pieces: list[dict] = []
    for comp in comps:
        frame = _frame_info(g, comp, bg)
        if frame is not None:
            frames.append(frame)
        else:
            pieces.append(comp)

    assert frames, "expected at least one frame"

    out = g.copy()
    used_piece_ids: set[int] = set()

    for frame in frames:
        gap_h, gap_w = frame["gap_shape"]
        matches = []
        for idx, piece in enumerate(pieces):
            if idx in used_piece_ids or int(piece["color"]) != int(frame["color"]):
                continue
            if tuple(piece["shape"]) != (int(gap_h), int(gap_w)):
                continue
            if int(piece["area"]) != int(gap_h * gap_w):
                continue
            matches.append(idx)

        assert len(matches) == 1, "each frame must have exactly one matching piece"
        piece = pieces[matches[0]]
        used_piece_ids.add(matches[0])

        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)
        for r, c in frame["gap_coords"]:
            out[int(r), int(c)] = int(frame["color"])

    return out


def _reserve(occupied: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = OBJECT_PAD) -> None:
    h, w = occupied.shape
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(h, int(r0 + height + pad))
    cc1 = min(w, int(c0 + width + pad))
    occupied[rr0:rr1, cc0:cc1] = True


def _candidate_positions(
    occupied: np.ndarray,
    height: int,
    width: int,
    row_min: int = 0,
    row_max: int | None = None,
    col_min: int = 0,
    col_max: int | None = None,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    if row_max is None:
        row_max = h - height
    if col_max is None:
        col_max = w - width
    row_min = max(0, int(row_min))
    col_min = max(0, int(col_min))
    row_max = min(int(row_max), h - height)
    col_max = min(int(col_max), w - width)
    if row_min > row_max or col_min > col_max:
        return []

    spots: list[tuple[int, int]] = []
    for r0 in range(row_min, row_max + 1):
        for c0 in range(col_min, col_max + 1):
            if occupied[r0 : r0 + height, c0 : c0 + width].any():
                continue
            spots.append((int(r0), int(c0)))
    return spots


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _full_frame_patch(height: int, width: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _sample_frame_spec(color: int, bg: int) -> dict:
    inner_h = random.randint(*INNER_SIDE_RANGE)
    inner_w = random.randint(*INNER_SIDE_RANGE)
    height = int(inner_h + 2)
    width = int(inner_w + 2)
    side = random.choice(["top", "bottom", "left", "right"])
    side_len = int(width if side in {"top", "bottom"} else height)
    gap_len = random.randint(1, side_len - 1)
    start = random.randint(0, side_len - gap_len)
    gap_local = _gap_local_coords(height, width, side, start, gap_len)
    gap_rows = [r for r, _ in gap_local]
    gap_cols = [c for _, c in gap_local]
    gap_h = max(gap_rows) - min(gap_rows) + 1
    gap_w = max(gap_cols) - min(gap_cols) + 1

    full_patch = _full_frame_patch(height, width, color, bg)
    input_patch = full_patch.copy()
    for r, c in gap_local:
        input_patch[int(r), int(c)] = int(bg)

    piece_patch = np.full((int(gap_h), int(gap_w)), int(color), dtype=int)
    return {
        "color": int(color),
        "frame_patch_out": full_patch,
        "frame_patch_in": input_patch,
        "frame_shape": (int(height), int(width)),
        "piece_patch": piece_patch,
        "piece_shape": (int(gap_h), int(gap_w)),
    }


def _grid_choices(specs: list[dict]) -> list[tuple[int, int]]:
    max_h = max(max(int(s["frame_shape"][0]), int(s["piece_shape"][0])) for s in specs)
    max_w = max(max(int(s["frame_shape"][1]), int(s["piece_shape"][1])) for s in specs)
    reserve = 0
    for spec in specs:
        fh, fw = map(int, spec["frame_shape"])
        ph, pw = map(int, spec["piece_shape"])
        reserve += (fh + 2 * OBJECT_PAD) * (fw + 2 * OBJECT_PAD)
        reserve += (ph + 2 * OBJECT_PAD) * (pw + 2 * OBJECT_PAD)

    choices: list[tuple[int, int]] = []
    for h in range(max(GRID_MIN, max_h + 2), GRID_MAX + 1):
        for w in range(max(GRID_MIN, max_w + 2), GRID_MAX + 1):
            if h * w < reserve + 30:
                continue
            choices.append((int(h), int(w)))
    return choices


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flatten()))
    return int(counts[int(bg)]) == max(map(int, counts.values()))


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    bg = random.randrange(10)
    colors = random.sample([c for c in range(10) if c != bg], n_frames)
    specs = [_sample_frame_spec(int(color), bg) for color in colors]

    dims = _grid_choices(specs)
    if not dims:
        return None
    grid_h, grid_w = random.choice(dims)

    out = np.full((grid_h, grid_w), int(bg), dtype=int)
    inp = out.copy()
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    frame_order = sorted(
        range(n_frames),
        key=lambda idx: specs[idx]["frame_shape"][0] * specs[idx]["frame_shape"][1],
        reverse=True,
    )

    for idx in frame_order:
        fh, fw = map(int, specs[idx]["frame_shape"])
        spots = _candidate_positions(occupied, fh, fw)
        if not spots:
            return None
        r0, c0 = random.choice(spots)
        _paste(out, specs[idx]["frame_patch_out"], r0, c0)
        _paste(inp, specs[idx]["frame_patch_in"], r0, c0)
        _reserve(occupied, r0, c0, fh, fw)

    piece_order = sorted(
        range(n_frames),
        key=lambda idx: specs[idx]["piece_shape"][0] * specs[idx]["piece_shape"][1],
        reverse=True,
    )

    for idx in piece_order:
        ph, pw = map(int, specs[idx]["piece_shape"])
        spots = _candidate_positions(occupied, ph, pw)
        if not spots:
            return None
        r0, c0 = random.choice(spots)
        _paste(inp, specs[idx]["piece_patch"], r0, c0, transparent=bg)
        _reserve(occupied, r0, c0, ph, pw)

    if not _bg_is_dominant(inp, bg):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
