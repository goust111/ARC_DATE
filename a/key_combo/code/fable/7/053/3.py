import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": int(color), "coords": coords, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _is_rectangular_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    actual = set(comp["coords"])
    if actual != expected:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1][0, :] == GREY))


def _center_axis(box: tuple[int, int, int, int], orientation: str) -> int:
    r0, c0, r1, c1 = box
    if orientation == "row":
        return (r0 + 1 + r1 - 1) // 2
    return (c0 + 1 + c1 - 1) // 2


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for comp in _find_color_components(grid, GREY):
        if _is_rectangular_frame(grid, comp):
            frames.append(tuple(map(int, comp["bbox"])))
    frames.sort()
    return frames


def _find_protrusions(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[dict]:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    protrusions = []
    if c0 - 1 >= 0:
        for r in range(r0 + 1, r1):
            color = int(grid[r, c0 - 1])
            if color not in (BLACK, GREY):
                protrusions.append(
                    {"coord": (r, c0 - 1), "side": "left", "orientation": "row", "axis": r, "color": color}
                )
    if c1 + 1 < w:
        for r in range(r0 + 1, r1):
            color = int(grid[r, c1 + 1])
            if color not in (BLACK, GREY):
                protrusions.append(
                    {"coord": (r, c1 + 1), "side": "right", "orientation": "row", "axis": r, "color": color}
                )
    if r0 - 1 >= 0:
        for c in range(c0 + 1, c1):
            color = int(grid[r0 - 1, c])
            if color not in (BLACK, GREY):
                protrusions.append(
                    {"coord": (r0 - 1, c), "side": "top", "orientation": "col", "axis": c, "color": color}
                )
    if r1 + 1 < h:
        for c in range(c0 + 1, c1):
            color = int(grid[r1 + 1, c])
            if color not in (BLACK, GREY):
                protrusions.append(
                    {"coord": (r1 + 1, c), "side": "bottom", "orientation": "col", "axis": c, "color": color}
                )
    protrusions.sort(key=lambda p: (p["coord"][0], p["coord"][1], p["color"]))
    colors = [p["color"] for p in protrusions]
    assert 1 <= len(protrusions) <= 2, "each frame must have one or two protrusions"
    assert len(colors) == len(set(colors)), "frame protrusion colors must be distinct"
    return protrusions


def _transform(
    grid: np.ndarray,
    *,
    ignore_color: bool = False,
    center_axes: bool = False,
    active: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    src = np.asarray(grid, dtype=int)
    out = src.copy()
    frames = _find_frames(src)
    assert frames, "expected at least one rectangular frame"

    parsed = [(box, _find_protrusions(src, box)) for box in frames]
    for frame_i, (box, protrusions) in enumerate(parsed):
        r0, c0, r1, c1 = box
        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        interior = src[ir0 : ir1 + 1, ic0 : ic1 + 1]
        for protrusion_i, protrusion in enumerate(protrusions):
            if active is not None and (frame_i, protrusion_i) not in active:
                continue
            axis = _center_axis(box, protrusion["orientation"]) if center_axes else int(protrusion["axis"])
            if ignore_color:
                coords = np.argwhere((interior != BLACK) & (interior != GREY))
            else:
                coords = np.argwhere(interior == int(protrusion["color"]))
            for rr, cc in coords:
                r = ir0 + int(rr)
                c = ic0 + int(cc)
                if protrusion["orientation"] == "row":
                    nr, nc = 2 * axis - r, c
                else:
                    nr, nc = r, 2 * axis - c
                if ir0 <= nr <= ir1 and ic0 <= nc <= ic1 and int(out[nr, nc]) == BLACK:
                    out[nr, nc] = int(src[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror same-color interior pixels inside each frame across the axis selected by each wall protrusion.

    Input structure:
    - Black background with one or more one-cell-thick grey rectangular frames.
    - Each frame interior contains colored pixels on black cells.
    - Each frame has one or two single-pixel colored protrusions attached to its outer wall, and each protrusion color also appears inside that frame.

    Rule:
    - A protrusion on the left or right wall selects its own row as the mirror axis; a protrusion on the top or bottom wall selects its own column.
    - For each protrusion, only the interior pixels of the same color are mirrored across that axis; original pixels stay in place.
    - A mirrored pixel is added only when its destination stays strictly inside the frame and is currently black; the frame, protrusions, and all other colors remain unchanged.

    Key insight:
    - Each protrusion is a double selector: its wall position picks the fold axis, and its color picks which interior pixels participate.
    """
    return _transform(np.asarray(grid, dtype=int))


def _frame_case() -> str:
    return random.choices(
        ["single_row", "single_col", "double_mixed", "double_row", "double_col"],
        weights=[2, 2, 3, 1, 1],
        k=1,
    )[0]


def _non_center_choices(lo: int, hi: int, center: int) -> list[int]:
    return [v for v in range(lo, hi + 1) if v != center]


def _pick_pairs(
    frame_h: int,
    frame_w: int,
    orientation: str,
    axis: int,
    occupied: np.ndarray,
    reserved: set[tuple[int, int]],
    max_count: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    sides = []
    if orientation == "row":
        groups = {"near_top": [], "near_bottom": []}
        for r in range(2, frame_h):
            if r == axis:
                continue
            nr = 2 * axis - r
            if not (2 <= nr <= frame_h - 1):
                continue
            group = "near_top" if r < axis else "near_bottom"
            for c in range(2, frame_w):
                src = (r, c)
                dst = (nr, c)
                if src == dst or occupied[src] or occupied[dst] or src in reserved or dst in reserved:
                    continue
                groups[group].append((src, dst))
        sides = [pairs for pairs in groups.values() if pairs]
    else:
        groups = {"near_left": [], "near_right": []}
        for c in range(2, frame_w):
            if c == axis:
                continue
            nc = 2 * axis - c
            if not (2 <= nc <= frame_w - 1):
                continue
            group = "near_left" if c < axis else "near_right"
            for r in range(2, frame_h):
                src = (r, c)
                dst = (r, nc)
                if src == dst or occupied[src] or occupied[dst] or src in reserved or dst in reserved:
                    continue
                groups[group].append((src, dst))
        sides = [pairs for pairs in groups.values() if pairs]
    if not sides:
        return []
    candidates = random.choice(sides)
    count = random.randint(1, min(max_count, len(candidates)))
    return random.sample(candidates, count)


def _build_frame_patch() -> tuple[np.ndarray, dict] | None:
    frame_h = random.randint(7, 11)
    frame_w = random.randint(7, 11)
    patch = np.zeros((frame_h + 2, frame_w + 2), dtype=int)
    patch[1, 1 : frame_w + 1] = GREY
    patch[frame_h, 1 : frame_w + 1] = GREY
    patch[1 : frame_h + 1, 1] = GREY
    patch[1 : frame_h + 1, frame_w] = GREY

    case = _frame_case()
    center_row = (2 + frame_h - 1) // 2
    center_col = (2 + frame_w - 1) // 2
    row_axes = _non_center_choices(2, frame_h - 1, center_row)
    col_axes = _non_center_choices(2, frame_w - 1, center_col)
    if not row_axes or not col_axes:
        return None

    if case == "single_row":
        specs = [(random.choice(["left", "right"]), random.choice(row_axes))]
    elif case == "single_col":
        specs = [(random.choice(["top", "bottom"]), random.choice(col_axes))]
    elif case == "double_mixed":
        specs = [
            (random.choice(["left", "right"]), random.choice(row_axes)),
            (random.choice(["top", "bottom"]), random.choice(col_axes)),
        ]
        random.shuffle(specs)
    elif case == "double_row":
        specs = [("left", random.choice(row_axes)), ("right", random.choice(row_axes))]
    else:
        specs = [("top", random.choice(col_axes)), ("bottom", random.choice(col_axes))]

    n_protrusions = len(specs)
    n_distractors = random.randint(1, 2)
    colors = random.sample(PALETTE, n_protrusions + n_distractors)
    protrusion_colors = colors[:n_protrusions]
    distractor_colors = colors[n_protrusions:]

    protrusions = []
    for (side, axis), color in zip(specs, protrusion_colors):
        if side == "left":
            coord = (axis, 0)
            orientation = "row"
        elif side == "right":
            coord = (axis, frame_w + 1)
            orientation = "row"
        elif side == "top":
            coord = (0, axis)
            orientation = "col"
        else:
            coord = (frame_h + 1, axis)
            orientation = "col"
        patch[coord] = int(color)
        protrusions.append({"side": side, "orientation": orientation, "axis": int(axis), "coord": coord, "color": int(color)})

    occupied = patch != BLACK
    reserved: set[tuple[int, int]] = set()
    effects = []
    for protrusion in protrusions:
        pairs = _pick_pairs(frame_h, frame_w, protrusion["orientation"], protrusion["axis"], occupied, reserved, 3)
        if not pairs:
            return None
        for src, dst in pairs:
            patch[src] = int(protrusion["color"])
            occupied[src] = True
            reserved.add(dst)
        effects.append({"color": protrusion["color"], "pairs": pairs})

    for color in distractor_colors:
        anchor = random.choice(protrusions)
        pairs = _pick_pairs(frame_h, frame_w, anchor["orientation"], anchor["axis"], occupied, reserved, 2)
        if not pairs:
            return None
        for src, dst in pairs[:1]:
            patch[src] = int(color)
            occupied[src] = True
            reserved.add(dst)

    interior_free = [(r, c) for r in range(2, frame_h) for c in range(2, frame_w) if not occupied[r, c] and (r, c) not in reserved]
    random.shuffle(interior_free)
    extra_count = min(len(interior_free), random.randint(1, 3))
    extra_palette = [c for c in PALETTE if c not in protrusion_colors]
    for idx in range(extra_count):
        patch[interior_free[idx]] = int(random.choice(extra_palette))
        occupied[interior_free[idx]] = True

    meta = {
        "patch_shape": patch.shape,
        "frame_box_local": (1, 1, frame_h, frame_w),
        "protrusions": protrusions,
        "effects": effects,
    }
    return patch, meta


def _candidate_positions(occupied: np.ndarray, patch_shape: tuple[int, int], border: int, padding: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch_shape
    out = []
    for r in range(border, h - ph - border + 1):
        for c in range(border, w - pw - border + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(h, r + ph + padding)
            cc1 = min(w, c + pw + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                out.append((r, c))
    return out


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    region[mask] = patch[mask]


def _attempt_grid() -> tuple[np.ndarray, list[dict]] | None:
    frame_count = random.choices([1, 2, 3], weights=[2, 4, 1], k=1)[0]
    patches = []
    metas = []
    for _ in range(frame_count):
        built = _build_frame_patch()
        if built is None:
            return None
        patch, meta = built
        patches.append(patch)
        metas.append(meta)

    min_h = max(16, max(p.shape[0] for p in patches) + 4)
    min_w = max(16, max(p.shape[1] for p in patches) + 4)
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed = []

    order = sorted(range(frame_count), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        patch = patches[idx]
        candidates = _candidate_positions(occupied, patch.shape, border=1, padding=2)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _paste_patch(grid, patch, top, left)
        ph, pw = patch.shape
        rr0 = max(0, top - 2)
        cc0 = max(0, left - 2)
        rr1 = min(grid_h, top + ph + 2)
        cc1 = min(grid_w, left + pw + 2)
        occupied[rr0:rr1, cc0:cc1] = True
        placed.append((idx, top, left))

    meta_by_idx = {idx: meta for idx, meta in enumerate(metas)}
    placed.sort()
    final_meta = []
    for idx, top, left in placed:
        meta = meta_by_idx[idx]
        r0, c0, r1, c1 = meta["frame_box_local"]
        abs_box = (top + r0, left + c0, top + r1, left + c1)
        abs_protrusions = []
        for protrusion in meta["protrusions"]:
            pr, pc = protrusion["coord"]
            abs_protrusions.append(
                {
                    "side": protrusion["side"],
                    "orientation": protrusion["orientation"],
                    "axis": top + pr if protrusion["orientation"] == "row" else left + pc,
                    "coord": (top + pr, left + pc),
                    "color": protrusion["color"],
                }
            )
        final_meta.append({"box": abs_box, "protrusions": abs_protrusions})
    final_meta.sort(key=lambda item: item["box"])
    return grid, final_meta


def _signature(frames: list[dict]) -> list[tuple]:
    sig = []
    for frame in frames:
        box = tuple(frame["box"])
        protrusions = tuple(
            sorted((p["side"], int(p["axis"]), int(p["color"]), tuple(p["coord"])) for p in frame["protrusions"])
        )
        sig.append((box, protrusions))
    return sig


def _parse_signature(grid: np.ndarray) -> list[tuple]:
    sig = []
    for box in _find_frames(grid):
        protrusions = _find_protrusions(grid, box)
        sig.append(
            (
                tuple(box),
                tuple(sorted((p["side"], int(p["axis"]), int(p["color"]), tuple(p["coord"])) for p in protrusions)),
            )
        )
    return sig


def _changed_only_inside_interiors(inp: np.ndarray, out: np.ndarray) -> bool:
    boxes = _find_frames(inp)
    interiors = [(r0 + 1, c0 + 1, r1 - 1, c1 - 1) for r0, c0, r1, c1 in boxes]
    for r, c in np.argwhere(inp != out):
        if int(inp[r, c]) != BLACK:
            return False
        if not any(ir0 <= r <= ir1 and ic0 <= c <= ic1 for ir0, ic0, ir1, ic1 in interiors):
            return False
    return True


def _validate_generated(grid: np.ndarray, meta: list[dict]) -> bool:
    if _signature(meta) != _parse_signature(grid):
        return False

    actual = main(grid)
    if np.array_equal(actual, grid):
        return False
    if not _changed_only_inside_interiors(grid, actual):
        return False

    all_ids = set()
    for frame_i, frame in enumerate(meta):
        for protrusion_i, _ in enumerate(frame["protrusions"]):
            all_ids.add((frame_i, protrusion_i))

    for ident in all_ids:
        without = _transform(grid, active=all_ids - {ident})
        if np.array_equal(actual, without):
            return False

    wrong_all_colors = _transform(grid, ignore_color=True)
    if np.array_equal(actual, wrong_all_colors):
        return False

    wrong_center = _transform(grid, center_axes=True)
    if np.array_equal(actual, wrong_center):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        attempted = _attempt_grid()
        if attempted is None:
            continue
        grid, meta = attempted
        try:
            if _validate_generated(grid, meta):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid protrusion-selector folding puzzle")
