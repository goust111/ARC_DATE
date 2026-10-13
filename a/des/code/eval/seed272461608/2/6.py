from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
MARKER_TO_OP = {
    1: "vflip",    # blue
    2: "hflip",    # red
    3: "rot_cw",   # green
    4: "rot_ccw",  # yellow
}
FRAME_COLOR_POOL = (5, 6, 7, 8, 9)
NON_BG_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIDE = 30


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop_non_bg(arr: np.ndarray) -> np.ndarray:
    box = _bbox_from_mask(arr != BG_COLOR)
    assert box is not None, "expected a non-empty object"
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _apply_marker(arr: np.ndarray, marker_color: int) -> np.ndarray:
    op = MARKER_TO_OP[int(marker_color)]
    if op == "hflip":
        return np.fliplr(arr)
    if op == "vflip":
        return np.flipud(arr)
    if op == "rot_cw":
        return np.rot90(arr, -1)
    if op == "rot_ccw":
        return np.rot90(arr, 1)
    raise AssertionError(op)


def _frame_from_component(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    frame_color = int(grid[coords[0, 0], coords[0, 1]])
    rel = coords - np.array([r0, c0], dtype=int)
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    if not bool(np.all(on_border)):
        return None

    border_size = 2 * (h + w) - 4
    if int(coords.shape[0]) != border_size - 1:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    mismatch = np.argwhere(border_mask & (patch != frame_color))
    if mismatch.shape[0] != 1:
        return None

    mr, mc = map(int, mismatch[0])
    if mr != 0 or mc in (0, w - 1):
        return None
    if w % 2 == 0 or mc != w // 2:
        return None

    marker_color = int(patch[mr, mc])
    if marker_color not in MARKER_TO_OP:
        return None
    if marker_color == BG_COLOR:
        return None

    if not bool(np.all(patch[border_mask & ~((np.arange(h)[:, None] == mr) & (np.arange(w)[None, :] == mc))] == frame_color)):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "area": h * w,
        "frame_color": frame_color,
        "marker_color": marker_color,
    }


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _target_chain(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG_COLOR:
            continue
        for coords in _components(grid == color):
            frame = _frame_from_component(grid, coords)
            if frame is not None:
                frames.append(frame)

    assert frames, "expected at least one marked frame"

    for frame in frames:
        parents = [
            other
            for other in frames
            if other is not frame and _contains(other["bbox"], frame["bbox"])
        ]
        frame["parent"] = min(parents, key=lambda item: int(item["area"])) if parents else None
        frame["children"] = []

    for frame in frames:
        parent = frame["parent"]
        if parent is not None:
            parent["children"].append(frame)

    leaf_candidates = []
    for frame in frames:
        if frame["children"]:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(inner != BG_COLOR):
            leaf_candidates.append(frame)

    assert len(leaf_candidates) == 1, "expected one innermost framed object"

    chain: list[dict] = []
    cur = leaf_candidates[0]
    while cur is not None:
        chain.append(cur)
        cur = cur["parent"]
    chain.reverse()
    return chain


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the nested marked frames, compose their transforms from inner to outer,
    and return the transformed innermost object without the frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    chain = _target_chain(g)
    r0, c0, r1, c1 = chain[-1]["bbox"]
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    obj = _crop_non_bg(inner)

    for frame in reversed(chain):
        obj = _apply_marker(obj, int(frame["marker_color"]))
    return _crop_non_bg(obj)


def _signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _dihedral_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rot = np.rot90(arr, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    found: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if r > 0 and not mask[r - 1, c]:
            found.add((r - 1, c))
        if r + 1 < h and not mask[r + 1, c]:
            found.add((r + 1, c))
        if c > 0 and not mask[r, c - 1]:
            found.add((r, c - 1))
        if c + 1 < w and not mask[r, c + 1]:
            found.add((r, c + 1))
    return list(found)


def _monotone_corner_path(h: int, w: int) -> list[tuple[int, int]]:
    start, end = random.choice([
        ((0, 0), (h - 1, w - 1)),
        ((0, w - 1), (h - 1, 0)),
    ])
    r, c = start
    tr, tc = end
    dr = 1 if tr >= r else -1
    dc = 1 if tc >= c else -1

    moves = [(dr, 0)] * abs(tr - r) + [(0, dc)] * abs(tc - c)
    random.shuffle(moves)

    path = [(r, c)]
    for rr, cc in moves:
        r += rr
        c += cc
        path.append((r, c))
    return path


def _sample_object(colors: list[int]) -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        if h * w < 6:
            continue

        path = _monotone_corner_path(h, w)
        min_pixels = max(len(path), 4)
        max_pixels = h * w
        if min_pixels > max_pixels:
            continue

        target_pixels = random.randint(min_pixels, max_pixels)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in path:
            mask[r, c] = True
        while int(mask.sum()) < target_pixels:
            choices = _frontier(mask)
            if not choices:
                break
            rr, cc = random.choice(choices)
            mask[rr, cc] = True

        n_colors = random.randint(2, min(3, len(colors)))
        chosen_colors = random.sample(colors, n_colors)
        obj = np.full((h, w), BG_COLOR, dtype=int)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)

        for idx, color in enumerate(chosen_colors):
            r, c = coords[idx]
            obj[r, c] = int(color)
        for r, c in coords[n_colors:]:
            obj[r, c] = int(random.choice(chosen_colors))

        if _bbox_from_mask(obj != BG_COLOR) != (0, 0, h - 1, w - 1):
            continue
        if len({_signature(v) for v in _dihedral_variants(obj)}) != 8:
            continue
        return obj

    raise RuntimeError("failed to sample an asymmetric object")


def _odd_width_gaps(base_width: int, left: int, right: int) -> tuple[int, int]:
    if (base_width + left + right) % 2 == 0:
        if random.choice([False, True]):
            left += 1
        else:
            right += 1
    return left, right


def _sample_layers(obj_shape: tuple[int, int], n_frames: int) -> list[dict]:
    obj_h, obj_w = map(int, obj_shape)
    layers: list[dict] = []

    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    left, right = _odd_width_gaps(obj_w, left, right)
    cur_h = obj_h + top + bottom + 2
    cur_w = obj_w + left + right + 2
    layers.append({
        "height": cur_h,
        "width": cur_w,
        "content_offset": (1 + top, 1 + left),
    })

    for _ in range(n_frames - 1):
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        left, right = _odd_width_gaps(cur_w, left, right)
        cur_h = cur_h + top + bottom + 2
        cur_w = cur_w + left + right + 2
        layers.append({
            "height": cur_h,
            "width": cur_w,
            "content_offset": (1 + top, 1 + left),
        })

    return layers


def _draw_frame(grid: np.ndarray, top: int, left: int, h: int, w: int, frame_color: int, marker_color: int) -> None:
    grid[top, left : left + w] = frame_color
    grid[top + h - 1, left : left + w] = frame_color
    grid[top : top + h, left] = frame_color
    grid[top : top + h, left + w - 1] = frame_color
    grid[top, left + w // 2] = marker_color


def _expected_output(obj: np.ndarray, markers_inner_to_outer: list[int]) -> np.ndarray:
    out = obj.copy()
    for marker in markers_inner_to_outer:
        out = _apply_marker(out, marker)
    return _crop_non_bg(out)


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(1, 4)
    frame_colors = random.sample(FRAME_COLOR_POOL, n_frames)
    object_palette = [color for color in NON_BG_COLORS if color not in frame_colors]
    obj = _sample_object(object_palette)

    layers = _sample_layers(obj.shape, n_frames)
    outer_h = int(layers[-1]["height"])
    outer_w = int(layers[-1]["width"])

    margin_top = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_right = random.randint(1, 2)
    grid_h = outer_h + margin_top + margin_bottom
    grid_w = outer_w + margin_left + margin_right
    if grid_h > MAX_GRID_SIDE or grid_w > MAX_GRID_SIDE:
        return None

    markers = [random.choice(tuple(MARKER_TO_OP)) for _ in range(n_frames)]
    expected = _expected_output(obj, markers)

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    top = margin_top
    left = margin_left

    for layer, frame_color, marker in zip(
        reversed(layers),
        reversed(frame_colors),
        reversed(markers),
    ):
        _draw_frame(
            grid,
            top,
            left,
            int(layer["height"]),
            int(layer["width"]),
            int(frame_color),
            int(marker),
        )
        dr, dc = map(int, layer["content_offset"])
        top += dr
        left += dc

    oh, ow = obj.shape
    grid[top : top + oh, left : left + ow] = obj

    try:
        chain = _target_chain(grid)
    except AssertionError:
        return None

    if len(chain) != n_frames:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
