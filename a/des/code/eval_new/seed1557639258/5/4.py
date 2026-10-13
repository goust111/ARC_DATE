from __future__ import annotations

import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SIDES = ("top", "right", "bottom", "left")
_ROTATIONS = {"top": 0, "right": -1, "bottom": 2, "left": 1}
_BG = 0
_BASE_SIZE = 30
_MAX_GENERATION_ATTEMPTS = 400
_GENERATOR_PADDING = 1


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )

    return components


def _missing_run(bits: np.ndarray) -> tuple[int, int] | None:
    missing = np.flatnonzero(~bits)
    if missing.size == 0:
        return None
    if missing[0] == 0 or missing[-1] == bits.size - 1:
        return None
    if missing.size < 2 or missing.size > 4:
        return None
    if not np.all(np.diff(missing) == 1):
        return None
    return int(missing[0]), int(missing[-1])


def _is_centered(run: tuple[int, int], length: int) -> bool:
    start, end = run
    gap_center = (start + end) / 2.0
    side_center = (length - 1) / 2.0
    return abs(gap_center - side_center) <= 1.0


def _analyze_frame_component(component: dict) -> dict | None:
    mask = component["mask"]
    h, w = map(int, mask.shape)
    if h < 5 or w < 5:
        return None
    if bool(mask[1:-1, 1:-1].any()):
        return None

    sides = {
        "top": _missing_run(mask[0, :]),
        "bottom": _missing_run(mask[-1, :]),
        "left": _missing_run(mask[:, 0]),
        "right": _missing_run(mask[:, -1]),
    }
    gap_sides = [side for side, run in sides.items() if run is not None]
    if len(gap_sides) != 1:
        return None

    side = gap_sides[0]
    run = sides[side]
    assert run is not None
    if not _is_centered(run, w if side in ("top", "bottom") else h):
        return None

    for other_side in _SIDES:
        if other_side == side:
            continue
        edge = (
            mask[0, :]
            if other_side == "top"
            else mask[-1, :]
            if other_side == "bottom"
            else mask[:, 0]
            if other_side == "left"
            else mask[:, -1]
        )
        if not bool(edge.all()):
            return None

    gap_width = run[1] - run[0] + 1
    expected_area = 2 * (h + w) - 4 - gap_width
    if int(component["area"]) != expected_area:
        return None

    return {"side": side, "gap_width": gap_width}


def _project_patch(patch: np.ndarray, side: str) -> np.ndarray:
    rotated = np.rot90(patch, _ROTATIONS[side])
    if side in ("top", "bottom"):
        return np.flipud(rotated)
    return np.fliplr(rotated)


def _canvas_top_left(frame_bbox: tuple[int, int, int, int], side: str, patch_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    ph, pw = map(int, patch_shape)
    if side == "top":
        return r1 + 1, c0 + 1
    if side == "bottom":
        return r0 - ph, c0 + 1
    if side == "left":
        return r0 + 1, c1 + 1
    return r0 + 1, c0 - pw


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    target = out[top : top + h, left : left + w]
    mask = patch != bg
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each frame's interior pattern onto the blank canvas opposite its gap after an orientation-coded rotation.

    Input structure:
    - bg = most common color (typically black).
    - Several monochrome hollow rectangular frames appear in the grid; each frame has exactly one central gap of width 2-4 on one side.
    - Inside each frame is a smaller asymmetric pattern in another color.
    - Outside the side opposite the gap there is an unobstructed blank canvas region aligned with the frame interior.

    Rule:
    - Detect which side of each frame contains the gap.
    - Rotate the full interior patch according to that side: top -> 0 deg, right -> 90 deg clockwise, bottom -> 180 deg, left -> 270 deg clockwise.
    - Reflect the rotated patch along the gap-side normal and draw it onto the blank canvas on the opposite side, leaving background cells transparent.
    - Keep the original frames and original interior patterns unchanged.

    Key insight: The missing border segment has a double role: it simultaneously encodes the rotation to apply to the interior pattern and which exterior blank region receives the mirrored projection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    h, w = g.shape
    out = g.copy()

    for component in _find_monochrome_components(g, bg):
        frame_info = _analyze_frame_component(component)
        if frame_info is None:
            continue

        r0, c0, r1, c1 = map(int, component["bbox"])
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        if not bool(np.any(interior != bg)):
            continue

        projected = _project_patch(interior, frame_info["side"])
        top, left = _canvas_top_left(component["bbox"], frame_info["side"], projected.shape)
        ph, pw = projected.shape
        if not (0 <= top and 0 <= left and top + ph <= h and left + pw <= w):
            continue
        if bool(np.any(g[top : top + ph, left : left + pw] != bg)):
            continue

        _paste(out, projected, top, left, bg)

    return out


def _dihedral_key(arr: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    keys: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        keys.add((tuple(map(int, rot.shape)), rot.tobytes()))
        flip = np.fliplr(rot)
        keys.add((tuple(map(int, flip.shape)), flip.tobytes()))
    return keys


def _random_shape_patch(rng: np.random.Generator, size: int, color: int, bg: int) -> np.ndarray | None:
    max_area = min(size * size - 2, max(5, (size * size) // 2 + 1))
    if max_area < 3:
        return None
    area = int(rng.integers(3, max_area + 1))
    cells = {(int(rng.integers(size)), int(rng.integers(size)))}

    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in _DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(frontier[int(rng.integers(len(frontier)))])

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if len(set(rows)) < 2 or len(set(cols)) < 2:
        return None

    patch = np.full((size, size), bg, dtype=int)
    for r, c in cells:
        patch[r, c] = color

    if len(_dihedral_key(patch)) != 8:
        return None
    return patch


def _make_frame_patch(size: int, color: int, bg: int, side: str, gap_width: int) -> np.ndarray:
    side_len = size + 2
    patch = np.full((side_len, side_len), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    gap_start = (side_len - gap_width) // 2
    gap_slice = slice(gap_start, gap_start + gap_width)
    if side == "top":
        patch[0, gap_slice] = bg
    elif side == "bottom":
        patch[-1, gap_slice] = bg
    elif side == "left":
        patch[gap_slice, 0] = bg
    else:
        patch[gap_slice, -1] = bg
    return patch


def _footprint_dims(size: int, side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        return 2 * size + 2, size + 2
    return size + 2, 2 * size + 2


def _frame_offset(size: int, side: str) -> tuple[int, int]:
    if side == "bottom":
        return size, 0
    if side == "right":
        return 0, size
    return 0, 0


def _place_footprints(
    rng: np.random.Generator, specs: list[dict], base_shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = map(int, base_shape)
    occ = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["foot_h"] * specs[i]["foot_w"], reverse=True)
    placements = [(-1, -1)] * len(specs)

    for idx in order:
        fh = int(specs[idx]["foot_h"])
        fw = int(specs[idx]["foot_w"])
        found = False
        for _ in range(250):
            top = int(rng.integers(0, h - fh + 1))
            left = int(rng.integers(0, w - fw + 1))
            r0 = max(0, top - _GENERATOR_PADDING)
            c0 = max(0, left - _GENERATOR_PADDING)
            r1 = min(h, top + fh + _GENERATOR_PADDING)
            c1 = min(w, left + fw + _GENERATOR_PADDING)
            if bool(occ[r0:r1, c0:c1].any()):
                continue
            occ[top : top + fh, left : left + fw] = True
            placements[idx] = (top, left)
            found = True
            break
        if not found:
            return None

    return placements


def _sample_crop_bounds(
    rng: np.random.Generator, grid_shape: tuple[int, int], footprints: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    min_r = min(r0 for r0, _, _, _ in footprints)
    min_c = min(c0 for _, c0, _, _ in footprints)
    max_r = max(r1 for _, _, r1, _ in footprints)
    max_c = max(c1 for _, _, _, c1 in footprints)

    top_margin = int(rng.integers(0, min_r + 1))
    left_margin = int(rng.integers(0, min_c + 1))
    bottom_margin = int(rng.integers(0, grid_shape[0] - max_r + 1))
    right_margin = int(rng.integers(0, grid_shape[1] - max_c + 1))

    r0 = min_r - top_margin
    c0 = min_c - left_margin
    r1 = max_r + bottom_margin
    c1 = max_c + right_margin
    return r0, r1, c0, c1


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    num_objects = int(rng.integers(1, 5))
    size_hi = {1: 8, 2: 8, 3: 7, 4: 6}[num_objects]
    directions = list(rng.choice(_SIDES, size=num_objects, replace=False))
    colors = list(map(int, rng.choice(np.arange(1, 10), size=2 * num_objects, replace=False)))

    specs: list[dict] = []
    for i in range(num_objects):
        size = int(rng.integers(4, size_hi))
        side = directions[i]
        frame_color = colors[2 * i]
        pattern_color = colors[2 * i + 1]
        pattern = None
        for _ in range(80):
            pattern = _random_shape_patch(rng, size, pattern_color, _BG)
            if pattern is not None:
                break
        if pattern is None:
            return None
        gap_width = int(rng.integers(2, min(4, size) + 1))
        foot_h, foot_w = _footprint_dims(size, side)
        specs.append(
            {
                "size": size,
                "side": side,
                "frame_color": frame_color,
                "pattern_color": pattern_color,
                "pattern": pattern,
                "gap_width": gap_width,
                "foot_h": foot_h,
                "foot_w": foot_w,
            }
        )

    placements = _place_footprints(rng, specs, (_BASE_SIZE, _BASE_SIZE))
    if placements is None:
        return None

    inp = np.full((_BASE_SIZE, _BASE_SIZE), _BG, dtype=int)
    expected = inp.copy()
    footprints: list[tuple[int, int, int, int]] = []

    for spec, (ftop, fleft) in zip(specs, placements):
        size = int(spec["size"])
        side = spec["side"]
        frame_patch = _make_frame_patch(size, int(spec["frame_color"]), _BG, side, int(spec["gap_width"]))
        fdr, fdc = _frame_offset(size, side)
        frame_top = int(ftop + fdr)
        frame_left = int(fleft + fdc)
        frame_h, frame_w = frame_patch.shape
        frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

        inp[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_patch
        expected[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_patch

        pattern = spec["pattern"]
        _paste(inp, pattern, frame_top + 1, frame_left + 1, _BG)
        _paste(expected, pattern, frame_top + 1, frame_left + 1, _BG)

        projected = _project_patch(pattern, side)
        canvas_top, canvas_left = _canvas_top_left(frame_bbox, side, projected.shape)
        ph, pw = projected.shape
        if not (0 <= canvas_top and 0 <= canvas_left and canvas_top + ph <= _BASE_SIZE and canvas_left + pw <= _BASE_SIZE):
            return None
        if bool(np.any(inp[canvas_top : canvas_top + ph, canvas_left : canvas_left + pw] != _BG)):
            return None
        _paste(expected, projected, canvas_top, canvas_left, _BG)

        footprints.append((ftop, fleft, ftop + int(spec["foot_h"]), fleft + int(spec["foot_w"])))

    r0, r1, c0, c1 = _sample_crop_bounds(rng, inp.shape, footprints)
    inp = inp[r0:r1, c0:c1]
    expected = expected[r0:r1, c0:c1]
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
