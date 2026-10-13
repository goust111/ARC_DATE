from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(vals[counts.argmax()])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(pts)),
                }
            )
    return comps


def _contiguous_missing(inner_present: list[bool]) -> tuple[int, int] | None:
    missing = [i for i, present in enumerate(inner_present) if not present]
    if not missing:
        return None
    if missing != list(range(missing[0], missing[-1] + 1)):
        return None
    return missing[0], missing[-1]


def _frame_from_component(comp: dict) -> dict | None:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    cells = {tuple(map(int, p)) for p in coords.tolist()}
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    if not cells.issubset(expected):
        return None

    if (r0, c0) not in cells or (r0, c1) not in cells or (r1, c0) not in cells or (r1, c1) not in cells:
        return None

    top = [(r0, c) in cells for c in range(c0, c1 + 1)]
    bottom = [(r1, c) in cells for c in range(c0, c1 + 1)]
    left = [(r, c0) in cells for r in range(r0, r1 + 1)]
    right = [(r, c1) in cells for r in range(r0, r1 + 1)]

    side_runs: dict[str, tuple[int, int]] = {}
    for side, line in (
        ("top", top),
        ("bottom", bottom),
        ("left", left),
        ("right", right),
    ):
        run = _contiguous_missing(line[1:-1])
        if run is not None:
            side_runs[side] = run

    if len(side_runs) != 1:
        return None

    gap_side = next(iter(side_runs))
    for side, line in (
        ("top", top),
        ("bottom", bottom),
        ("left", left),
        ("right", right),
    ):
        inner = line[1:-1]
        if side == gap_side:
            if any(inner) is False:
                return None
        else:
            if not all(inner):
                return None

    missing_total = len(expected) - len(cells)
    gap_start_i, gap_end_i = side_runs[gap_side]
    gap_len = gap_end_i - gap_start_i + 1
    if missing_total != gap_len:
        return None

    if gap_side in ("top", "bottom"):
        gap_start = c0 + 1 + gap_start_i
        gap_end = c0 + 1 + gap_end_i
    else:
        gap_start = r0 + 1 + gap_start_i
        gap_end = r0 + 1 + gap_end_i

    return {
        "color": int(comp["color"]),
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "side": gap_side,
        "gap_start": int(gap_start),
        "gap_end": int(gap_end),
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    comps = _find_components(g, bg)

    frames: list[dict] = []
    frame_mask = np.zeros_like(g, dtype=bool)
    for comp in comps:
        frame = _frame_from_component(comp)
        if frame is None:
            continue
        frames.append(frame)
        frame_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    assert frames, "expected at least one gapped frame"

    sprite_grid = g.copy()
    sprite_grid[frame_mask] = bg
    sprites = _find_components(sprite_grid, bg)
    assert sprites, "expected sprites outside the frames"
    for sprite in sprites:
        assert 3 <= int(sprite["area"]) <= 6, "sprite area must be 3-6"

    return bg, frames, sprites, frame_mask


def _gap_exterior_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    if side == "top":
        return [(r0 - 1, c) for c in range(frame["gap_start"], frame["gap_end"] + 1)]
    if side == "bottom":
        return [(r1 + 1, c) for c in range(frame["gap_start"], frame["gap_end"] + 1)]
    if side == "left":
        return [(r, c0 - 1) for r in range(frame["gap_start"], frame["gap_end"] + 1)]
    return [(r, c1 + 1) for r in range(frame["gap_start"], frame["gap_end"] + 1)]


def _sprite_frame_distance(sprite: dict, frame: dict) -> int:
    pts = [tuple(map(int, p)) for p in sprite["coords"].tolist()]
    targets = _gap_exterior_cells(frame)
    return min(abs(sr - tr) + abs(sc - tc) for sr, sc in pts for tr, tc in targets)


def _pair_frames_and_sprites(frames: list[dict], sprites: list[dict]) -> list[tuple[dict, dict]]:
    assert len(frames) == len(sprites), "frame/sprite counts must match"
    assigned: dict[int, list[int]] = {i: [] for i in range(len(frames))}
    for si, sprite in enumerate(sprites):
        dists = [_sprite_frame_distance(sprite, frame) for frame in frames]
        order = sorted(range(len(frames)), key=lambda i: dists[i])
        if len(order) > 1:
            assert dists[order[0]] < dists[order[1]], "sprite-to-frame match must be unambiguous"
        assigned[order[0]].append(si)

    pairs: list[tuple[dict, dict]] = []
    for fi, frame in enumerate(frames):
        assert len(assigned[fi]) == 1, "each frame must have exactly one nearest sprite"
        pairs.append((frame, sprites[assigned[fi][0]]))
    return pairs


def _translated_coords(frame: dict, sprite: dict) -> np.ndarray:
    coords = sprite["coords"].astype(int, copy=False)
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    if side == "top":
        dr, dc = (r1 - 1) - int(coords[:, 0].max()), 0
    elif side == "bottom":
        dr, dc = (r0 + 1) - int(coords[:, 0].min()), 0
    elif side == "left":
        dr, dc = 0, (c1 - 1) - int(coords[:, 1].max())
    else:
        dr, dc = 0, (c0 + 1) - int(coords[:, 1].min())

    shifted = coords + np.array([dr, dc], dtype=int)
    assert np.all((r0 + 1 <= shifted[:, 0]) & (shifted[:, 0] <= r1 - 1))
    assert np.all((c0 + 1 <= shifted[:, 1]) & (shifted[:, 1] <= c1 - 1))
    return shifted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine each frame's gap side, then slide the nearest outside sprite inward and
    park it against the opposite inner wall while leaving the frame itself unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, sprites, frame_mask = _extract_scene(g)
    pairs = _pair_frames_and_sprites(frames, sprites)

    out = g.copy()
    sprite_mask = (~frame_mask) & (g != bg)
    out[sprite_mask] = bg

    for frame, sprite in pairs:
        shifted = _translated_coords(frame, sprite)
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == bg), "sprite destination must be empty"
        out[shifted[:, 0], shifted[:, 1]] = int(sprite["color"])

    return out


SHAPE_LIBRARY = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)},
    {(0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2)},
]


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rotate_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _, c in cells)
    return _normalize_shape({(c, max_c - r) for r, c in cells})


def _flip_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _, c in cells)
    return _normalize_shape({(r, max_c - c) for r, c in cells})


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _shape_variants() -> list[set[tuple[int, int]]]:
    variants: list[set[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for base in SHAPE_LIBRARY:
        cur = _normalize_shape(base)
        for _ in range(4):
            for candidate in (cur, _flip_shape(cur)):
                key = frozenset(candidate)
                if key not in seen:
                    seen.add(key)
                    variants.append(candidate)
            cur = _rotate_shape(cur)
    return variants


ALL_SHAPES = _shape_variants()


def _sample_shape(max_h: int, max_w: int) -> set[tuple[int, int]] | None:
    candidates = [shape for shape in ALL_SHAPES if _shape_dims(shape)[0] <= max_h and _shape_dims(shape)[1] <= max_w]
    if not candidates:
        return None
    return set(random.choice(candidates))


def _draw_frame(grid: np.ndarray, frame: dict, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = frame["color"]
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    if frame["side"] == "top":
        grid[r0, frame["gap_start"] : frame["gap_end"] + 1] = bg
    elif frame["side"] == "bottom":
        grid[r1, frame["gap_start"] : frame["gap_end"] + 1] = bg
    elif frame["side"] == "left":
        grid[frame["gap_start"] : frame["gap_end"] + 1, c0] = bg
    else:
        grid[frame["gap_start"] : frame["gap_end"] + 1, c1] = bg


def _paint_shape(grid: np.ndarray, origin: tuple[int, int], shape: set[tuple[int, int]], color: int) -> np.ndarray:
    pts = []
    orow, ocol = origin
    for dr, dc in shape:
        rr = orow + dr
        cc = ocol + dc
        grid[rr, cc] = color
        pts.append((rr, cc))
    return np.array(pts, dtype=int)


def _inner_fit_bounds(frame: dict, shape: set[tuple[int, int]]) -> tuple[int, int]:
    h, w = _shape_dims(shape)
    r0, c0, r1, c1 = frame["bbox"]
    assert h <= (r1 - r0 - 1) and w <= (c1 - c0 - 1)
    return h, w


def _sprite_origin_for_frame(frame: dict, shape: set[tuple[int, int]], clearance: int) -> tuple[int, int]:
    h, w = _inner_fit_bounds(frame, shape)
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    if side in ("top", "bottom"):
        start = frame["gap_start"]
        stop = frame["gap_end"] - w + 1
        scol = random.randint(start, stop)
        if side == "top":
            srow = r0 - clearance - h
        else:
            srow = r1 + 1 + clearance
        return int(srow), int(scol)

    start = frame["gap_start"]
    stop = frame["gap_end"] - h + 1
    srow = random.randint(start, stop)
    if side == "left":
        scol = c0 - clearance - w
    else:
        scol = c1 + 1 + clearance
    return int(srow), int(scol)


def _destination_bbox(frame: dict, shape: set[tuple[int, int]], origin: tuple[int, int]) -> tuple[int, int, int, int]:
    dummy = {"coords": _coords_from_origin(origin, shape)}
    shifted = _translated_coords(frame, dummy)
    return _bbox(shifted)


def _coords_from_origin(origin: tuple[int, int], shape: set[tuple[int, int]]) -> np.ndarray:
    orow, ocol = origin
    return np.array([(orow + dr, ocol + dc) for dr, dc in shape], dtype=int)


def _padded_rect(rect: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, rect[0] - pad),
        max(0, rect[1] - pad),
        min(h - 1, rect[2] + pad),
        min(w - 1, rect[3] + pad),
    )


def _reserve_clear(area: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return bool(np.all(~area[r0 : r1 + 1, c0 : c1 + 1]))


def _mark_reserved(area: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    area[r0 : r1 + 1, c0 : c1 + 1] = True


def _choose_colors(count: int, bg: int) -> list[int]:
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return palette[:count]


def _sample_frame_spec(grid_shape: tuple[int, int]) -> tuple[dict, set[tuple[int, int]], tuple[int, int], int] | None:
    gh, gw = grid_shape
    side = random.choice(SIDES)
    outer_h = random.randint(6, min(11, gh - 4))
    outer_w = random.randint(6, min(11, gw - 4))
    inner_h = outer_h - 2
    inner_w = outer_w - 2

    shape = _sample_shape(min(4, inner_h), min(4, inner_w))
    if shape is None:
        return None

    sprite_h, sprite_w = _shape_dims(shape)
    if side in ("top", "bottom"):
        side_len = inner_w
        min_gap = sprite_w
    else:
        side_len = inner_h
        min_gap = sprite_h
    max_gap = min(side_len, min_gap + 2)
    if max_gap < min_gap:
        return None
    gap_len = random.randint(min_gap, max_gap)
    gap_offset = random.randint(0, side_len - gap_len)

    clearance = random.randint(0, 3)
    margin = 1

    if side == "top":
        r0_min = sprite_h + clearance + margin
        r0_max = gh - outer_h - margin
        c0_min = margin
        c0_max = gw - outer_w - margin
    elif side == "bottom":
        r0_min = margin
        r0_max = gh - outer_h - (sprite_h + clearance + margin)
        c0_min = margin
        c0_max = gw - outer_w - margin
    elif side == "left":
        r0_min = margin
        r0_max = gh - outer_h - margin
        c0_min = sprite_w + clearance + margin
        c0_max = gw - outer_w - margin
    else:
        r0_min = margin
        r0_max = gh - outer_h - margin
        c0_min = margin
        c0_max = gw - outer_w - (sprite_w + clearance + margin)

    if r0_min > r0_max or c0_min > c0_max:
        return None

    r0 = random.randint(r0_min, r0_max)
    c0 = random.randint(c0_min, c0_max)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    if side in ("top", "bottom"):
        gap_start = c0 + 1 + gap_offset
        gap_end = gap_start + gap_len - 1
    else:
        gap_start = r0 + 1 + gap_offset
        gap_end = gap_start + gap_len - 1

    frame = {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_start": int(gap_start),
        "gap_end": int(gap_end),
    }
    sprite_origin = _sprite_origin_for_frame(frame, shape, clearance)
    return frame, shape, sprite_origin, clearance


def _render_expected(scene: list[dict], grid_shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    for item in scene:
        frame = item["frame"]
        _draw_frame(out, frame, bg)
        shifted = _translated_coords(frame, {"coords": item["sprite_coords"]})
        out[shifted[:, 0], shifted[:, 1]] = item["sprite_color"]
    return out


def _build_case() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    bg = 0
    n_objects = random.randint(1, 3)
    colors = _choose_colors(n_objects * 2, bg)
    reserved = np.zeros((h, w), dtype=bool)
    scene: list[dict] = []

    for i in range(n_objects):
        placed = False
        for _ in range(200):
            spec = _sample_frame_spec((h, w))
            if spec is None:
                continue
            frame, shape, sprite_origin, _clearance = spec
            frame["color"] = colors[2 * i]
            sprite_color = colors[2 * i + 1]
            if sprite_color == frame["color"]:
                continue

            sprite_coords = _coords_from_origin(sprite_origin, shape)
            if (
                np.any(sprite_coords[:, 0] < 0)
                or np.any(sprite_coords[:, 0] >= h)
                or np.any(sprite_coords[:, 1] < 0)
                or np.any(sprite_coords[:, 1] >= w)
            ):
                continue

            dest_bbox = _destination_bbox(frame, shape, sprite_origin)
            sprite_bbox = _bbox(sprite_coords)
            frame_bbox = frame["bbox"]
            union_bbox = (
                min(frame_bbox[0], sprite_bbox[0], dest_bbox[0]),
                min(frame_bbox[1], sprite_bbox[1], dest_bbox[1]),
                max(frame_bbox[2], sprite_bbox[2], dest_bbox[2]),
                max(frame_bbox[3], sprite_bbox[3], dest_bbox[3]),
            )
            padded = _padded_rect(union_bbox, 2, (h, w))
            if not _reserve_clear(reserved, padded):
                continue

            grid = np.full((h, w), bg, dtype=int)
            _draw_frame(grid, frame, bg)
            coords_now = _paint_shape(grid, sprite_origin, shape, sprite_color)

            try:
                out = main(grid)
            except AssertionError:
                continue
            expected = _render_expected(
                [
                    {
                        "frame": frame,
                        "sprite_coords": coords_now,
                        "sprite_color": sprite_color,
                    }
                ],
                (h, w),
                bg,
            )
            if not np.array_equal(out, expected):
                continue

            scene.append(
                {
                    "frame": frame,
                    "shape": shape,
                    "sprite_origin": sprite_origin,
                    "sprite_coords": coords_now,
                    "sprite_color": sprite_color,
                }
            )
            _mark_reserved(reserved, padded)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), bg, dtype=int)
    for item in scene:
        _draw_frame(grid, item["frame"], bg)
        _paint_shape(grid, item["sprite_origin"], item["shape"], item["sprite_color"])

    try:
        bg2, frames, sprites, _ = _extract_scene(grid)
        if bg2 != bg or len(frames) != n_objects or len(sprites) != n_objects:
            return None
        expected = _render_expected(scene, (h, w), bg)
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-frame sliding puzzle")
