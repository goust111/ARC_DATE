from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    m = np.asarray(mask, dtype=bool)
    return int(m.shape[0]), int(m.shape[1]), m.tobytes()


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _detect_frame(comp: dict) -> dict | None:
    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    border = _border_cells(r0, c0, r1, c1)
    if not coords.issubset(border):
        return None
    missing = sorted(border - coords)
    if not missing:
        return None

    side: str | None = None
    axis_vals: list[int] = []
    if all(r == r0 and c0 < c < c1 for r, c in missing):
        side = "up"
        axis_vals = sorted(c for _r, c in missing)
    elif all(r == r1 and c0 < c < c1 for r, c in missing):
        side = "down"
        axis_vals = sorted(c for _r, c in missing)
    elif all(c == c0 and r0 < r < r1 for r, c in missing):
        side = "left"
        axis_vals = sorted(r for r, _c in missing)
    elif all(c == c1 and r0 < r < r1 for r, c in missing):
        side = "right"
        axis_vals = sorted(r for r, _c in missing)
    else:
        return None

    if any(b != a + 1 for a, b in zip(axis_vals, axis_vals[1:])):
        return None

    gap_cells = sorted(
        missing,
        key=lambda rc: (int(rc[1]), int(rc[0])) if side in ("up", "down") else (int(rc[0]), int(rc[1])),
    )
    gap_cell = tuple(map(int, gap_cells[len(gap_cells) // 2]))
    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "dir": SIDE_TO_VEC[side],
        "gap_cell": gap_cell,
    }


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    dr, dc = map(int, direction)
    return [(int(r0 + step * dr), int(c0 + step * dc)) for step in range(int(length))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose each frame's matching outside sprite from the stitched fragment shape inside the frame,
    then shoot a colored ray outward through the frame gap.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _get_bg(g)

    frames = []
    for comp in _color_components(g, bg):
        frame = _detect_frame(comp)
        if frame is not None:
            frames.append(frame)
    assert frames, "expected at least one frame"

    out = g.copy()
    stripped = g.copy()

    for frame in frames:
        fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
        inside = stripped[fr0 + 1 : fr1, fc0 + 1 : fc1]
        frag_mask = (inside != int(bg)) & (inside != int(frame["color"]))
        frag_rel = np.argwhere(frag_mask)
        assert frag_rel.size > 0, "each frame must contain fragments"

        rr0, cc0 = np.min(frag_rel, axis=0)
        rr1, cc1 = np.max(frag_rel, axis=0)
        shape = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        shape[frag_rel[:, 0] - rr0, frag_rel[:, 1] - cc0] = True

        frame["shape_key"] = _shape_key(shape)
        frame["ray_len"] = int(shape.sum())
        frame["fragment_coords"] = frag_rel + np.array([fr0 + 1, fc0 + 1], dtype=int)

        stripped[frame["coords"][:, 0], frame["coords"][:, 1]] = bg
        stripped[fr0 + 1 : fr1, fc0 + 1 : fc1] = bg

    sprites = []
    for idx, comp in enumerate(_color_components(stripped, bg)):
        mask = _tight_mask(comp["coords"])
        sprites.append(
            {
                "id": int(idx),
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "shape_key": _shape_key(mask),
                "area": int(mask.sum()),
            }
        )
    assert sprites, "expected outside sprites"

    sprites_by_shape: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    for sprite in sprites:
        sprites_by_shape[sprite["shape_key"]].append(sprite)

    used_sprite_ids: set[int] = set()
    for frame in frames:
        matches = [sprite for sprite in sprites_by_shape[frame["shape_key"]] if int(sprite["id"]) not in used_sprite_ids]
        assert len(matches) == 1, "every frame must have one exact sprite match"
        sprite = matches[0]
        used_sprite_ids.add(int(sprite["id"]))

        frag_coords = frame["fragment_coords"]
        out[frag_coords[:, 0], frag_coords[:, 1]] = int(bg)
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = int(bg)

        for rr, cc in _ray_cells(frame["gap_cell"], frame["dir"], sprite["area"]):
            assert 0 <= rr < H and 0 <= cc < W, "ray must fit in the grid"
            out[rr, cc] = int(sprite["color"])

    return out


# --------------- generator ---------------

FRAME_COUNT_RANGE = (1, 3)
DISTRACTOR_COUNT_RANGE = (0, 2)
SHAPE_AREA_RANGE = (3, 6)
SHAPE_BBOX_MAX = 4
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
PADDING = 1
MAX_RETRIES = 400
PLACEMENT_ATTEMPTS = 250


def _mask_is_frame_like(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    fake = {
        "color": 1,
        "coords": coords.astype(int),
        "bbox": _bbox_from_coords(coords.astype(int)),
    }
    return _detect_frame(fake) is not None


def _random_connected_mask() -> np.ndarray:
    for _ in range(300):
        area = random.randint(*SHAPE_AREA_RANGE)
        canvas = np.zeros((7, 7), dtype=bool)
        start = (3, 3)
        canvas[start] = True
        cells = {start}
        frontier = set((start[0] + dr, start[1] + dc) for dr, dc in DIRS4)

        while len(cells) < area and frontier:
            candidates = [rc for rc in frontier if 1 <= rc[0] <= 5 and 1 <= rc[1] <= 5 and rc not in cells]
            if not candidates:
                break
            nr, nc = random.choice(candidates)
            cells.add((int(nr), int(nc)))
            canvas[nr, nc] = True
            frontier.discard((nr, nc))
            for dr, dc in DIRS4:
                frontier.add((int(nr + dr), int(nc + dc)))

        if len(cells) != area:
            continue

        mask = _tight_mask(np.argwhere(canvas))
        if int(mask.shape[0]) > SHAPE_BBOX_MAX or int(mask.shape[1]) > SHAPE_BBOX_MAX:
            continue
        if _mask_is_frame_like(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a connected shape")


def _full_rect_cells(height: int, width: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in range(int(height)) for c in range(int(width))]


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> np.ndarray:
    coords = np.argwhere(patch != int(bg))
    placed = coords + np.array([int(top), int(left)], dtype=int)
    grid[placed[:, 0], placed[:, 1]] = patch[coords[:, 0], coords[:, 1]]
    return placed


def _can_place(blocked: np.ndarray, cells: list[tuple[int, int]], top: int, left: int) -> bool:
    H, W = blocked.shape
    for rr, cc in cells:
        gr = int(top + rr)
        gc = int(left + cc)
        if not (0 <= gr < H and 0 <= gc < W):
            return False
        if bool(blocked[gr, gc]):
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], top: int, left: int, pad: int) -> None:
    H, W = blocked.shape
    for rr, cc in cells:
        gr = int(top + rr)
        gc = int(left + cc)
        r0 = max(0, int(gr - pad))
        r1 = min(H, int(gr + pad + 1))
        c0 = max(0, int(gc - pad))
        c1 = min(W, int(gc + pad + 1))
        blocked[r0:r1, c0:c1] = True


def _random_top_left(blocked: np.ndarray, cells: list[tuple[int, int]]) -> tuple[int, int] | None:
    H, W = blocked.shape
    min_r = min(int(r) for r, _c in cells)
    max_r = max(int(r) for r, _c in cells)
    min_c = min(int(c) for _r, c in cells)
    max_c = max(int(c) for _r, c in cells)

    lo_r = int(-min_r)
    hi_r = int(H - 1 - max_r)
    lo_c = int(-min_c)
    hi_c = int(W - 1 - max_c)
    if lo_r > hi_r or lo_c > hi_c:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(lo_r, hi_r)
        left = random.randint(lo_c, hi_c)
        if _can_place(blocked, cells, top, left):
            return int(top), int(left)
    return None


def _build_frame_spec(shape_mask: np.ndarray, frame_color: int, bg: int, side: str) -> dict:
    sh, sw = map(int, shape_mask.shape)
    inner_h = int(sh + random.randint(2, 4))
    inner_w = int(sw + random.randint(2, 4))
    patch = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)

    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)

    if side == "up":
        gap = (0, random.randint(1, inner_w))
    elif side == "down":
        gap = (inner_h + 1, random.randint(1, inner_w))
    elif side == "left":
        gap = (random.randint(1, inner_h), 0)
    else:
        gap = (random.randint(1, inner_h), inner_w + 1)
    patch[gap] = int(bg)

    row_off = random.randint(1, inner_h - sh - 1)
    col_off = random.randint(1, inner_w - sw - 1)
    frag_palette = [c for c in ALL_COLORS if c not in (int(bg), int(frame_color))]
    frag_colors = random.sample(frag_palette, k=min(len(frag_palette), random.randint(2, 4)))

    fragment_rel = []
    for rr, cc in np.argwhere(shape_mask):
        gr = int(1 + row_off + rr)
        gc = int(1 + col_off + cc)
        patch[gr, gc] = int(random.choice(frag_colors))
        fragment_rel.append((gr, gc))

    ray_rel = _ray_cells(tuple(map(int, gap)), SIDE_TO_VEC[side], int(shape_mask.sum()))
    return {
        "type": "frame",
        "shape_mask": np.asarray(shape_mask, dtype=bool),
        "shape_key": _shape_key(shape_mask),
        "patch": patch,
        "bg": int(bg),
        "frame_color": int(frame_color),
        "gap_rel": tuple(map(int, gap)),
        "side": side,
        "dir": SIDE_TO_VEC[side],
        "fragment_rel": [(int(r), int(c)) for r, c in fragment_rel],
        "occupy_rel": [tuple(map(int, rc)) for rc in np.argwhere(patch != int(bg))],
        "reserve_rel": _full_rect_cells(*patch.shape) + [(int(r), int(c)) for r, c in ray_rel],
    }


def _build_sprite_spec(shape_mask: np.ndarray, color: int, bg: int, matched: bool) -> dict:
    patch = np.full(shape_mask.shape, int(bg), dtype=int)
    patch[shape_mask] = int(color)
    return {
        "type": "matched_sprite" if matched else "distractor",
        "shape_mask": np.asarray(shape_mask, dtype=bool),
        "shape_key": _shape_key(shape_mask),
        "patch": patch,
        "bg": int(bg),
        "color": int(color),
        "occupy_rel": [tuple(map(int, rc)) for rc in np.argwhere(shape_mask)],
        "reserve_rel": _full_rect_cells(*shape_mask.shape),
    }


def _expected_output(inp: np.ndarray, frames: list[dict], sprites: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        frag = np.array(frame["fragment_coords"], dtype=int)
        out[frag[:, 0], frag[:, 1]] = int(bg)
    for sprite in sprites:
        if sprite["type"] != "matched_sprite":
            continue
        coords = np.array(sprite["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(bg)
    for frame, sprite in zip(frames, sprites):
        if sprite["type"] != "matched_sprite":
            continue
        for rr, cc in _ray_cells(frame["gap_cell"], frame["dir"], int(sprite["area"])):
            out[rr, cc] = int(sprite["color"])
    return out


def _generate_case() -> np.ndarray | None:
    bg = random.choice(ALL_COLORS)
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    n_distractors = random.randint(*DISTRACTOR_COUNT_RANGE)

    palette = [c for c in ALL_COLORS if c != int(bg)]
    random.shuffle(palette)
    needed_colors = n_frames + n_frames + n_distractors
    if needed_colors > len(palette):
        return None

    frame_colors = palette[:n_frames]
    sprite_colors = palette[n_frames : n_frames + n_frames + n_distractors]

    used_shapes: set[tuple[int, int, bytes]] = set()
    shapes = []
    while len(shapes) < n_frames + n_distractors:
        mask = _random_connected_mask()
        key = _shape_key(mask)
        if key in used_shapes:
            continue
        used_shapes.add(key)
        shapes.append(mask)

    directions = random.sample(list(SIDE_TO_VEC.keys()), k=n_frames)
    frame_specs = [
        _build_frame_spec(shapes[i], int(frame_colors[i]), int(bg), directions[i])
        for i in range(n_frames)
    ]
    sprite_specs = [
        _build_sprite_spec(shapes[i], int(sprite_colors[i]), int(bg), matched=True)
        for i in range(n_frames)
    ]
    distractor_specs = [
        _build_sprite_spec(shapes[n_frames + i], int(sprite_colors[n_frames + i]), int(bg), matched=False)
        for i in range(n_distractors)
    ]
    all_sprites = sprite_specs + distractor_specs

    reserve_budget = 0
    for frame in frame_specs:
        reserve_budget += len(frame["reserve_rel"])
    for sprite in all_sprites:
        reserve_budget += len(sprite["reserve_rel"])
    side_min = max(GRID_SIDE_MIN, int(np.ceil(np.sqrt(reserve_budget * 2.2))))
    if side_min > GRID_SIDE_MAX:
        return None
    H = random.randint(side_min, GRID_SIDE_MAX)
    W = random.randint(side_min, GRID_SIDE_MAX)

    grid = np.full((H, W), int(bg), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    random.shuffle(frame_specs)
    for frame in frame_specs:
        loc = _random_top_left(blocked, frame["reserve_rel"])
        if loc is None:
            return None
        top, left = loc
        frame["top_left"] = (int(top), int(left))
        frame["coords"] = _paint_patch(grid, frame["patch"], int(top), int(left), int(bg))
        frame["fragment_coords"] = [
            (int(top + rr), int(left + cc))
            for rr, cc in frame["fragment_rel"]
        ]
        frame["gap_cell"] = (int(top + frame["gap_rel"][0]), int(left + frame["gap_rel"][1]))
        _mark_blocked(blocked, frame["reserve_rel"], int(top), int(left), PADDING)

    random.shuffle(all_sprites)
    for sprite in all_sprites:
        loc = _random_top_left(blocked, sprite["reserve_rel"])
        if loc is None:
            return None
        top, left = loc
        sprite["top_left"] = (int(top), int(left))
        sprite["coords"] = _paint_patch(grid, sprite["patch"], int(top), int(left), int(bg))
        sprite["area"] = int(sprite["shape_mask"].sum())
        _mark_blocked(blocked, sprite["reserve_rel"], int(top), int(left), PADDING)

    frame_specs.sort(key=lambda d: d["shape_key"])
    sprite_specs.sort(key=lambda d: d["shape_key"])
    distractor_specs.sort(key=lambda d: d["shape_key"])
    all_sprites = sprite_specs + distractor_specs

    expected = grid.copy()
    for frame in frame_specs:
        frag = np.array(frame["fragment_coords"], dtype=int)
        expected[frag[:, 0], frag[:, 1]] = int(bg)
    for sprite in sprite_specs:
        coords = np.array(sprite["coords"], dtype=int)
        expected[coords[:, 0], coords[:, 1]] = int(bg)
    for frame, sprite in zip(frame_specs, sprite_specs):
        for rr, cc in _ray_cells(frame["gap_cell"], frame["dir"], int(sprite["area"])):
            expected[rr, cc] = int(sprite["color"])

    if _get_bg(grid) != int(bg):
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
