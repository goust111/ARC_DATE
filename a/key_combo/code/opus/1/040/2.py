from __future__ import annotations

import random
from typing import Callable

import numpy as np


BG = 0
OBJ_COLORS = (1, 3, 4)
MARKER = 5
FRAME = 7
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _gray_roles(grid: np.ndarray) -> dict[str, int]:
    comps = _components(grid == MARKER)
    assert len(comps) == 4, "expected four gray marker-tail components"

    infos = []
    for coords in comps:
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0 = int(rs.min())
        r1 = int(rs.max())
        c0 = int(cs.min())
        c1 = int(cs.max())
        vertical = c0 == c1 and r0 < r1
        horizontal = r0 == r1 and c0 < c1
        assert vertical or horizontal, "each gray component must be a nontrivial straight tail"
        infos.append(
            {
                "coords": coords,
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "vertical": vertical,
                "horizontal": horizontal,
                "size": int(len(coords)),
            }
        )

    top_i = min(range(4), key=lambda i: (infos[i]["r1"], infos[i]["r0"], infos[i]["c0"]))
    bottom_i = max(range(4), key=lambda i: (infos[i]["r0"], infos[i]["r1"], infos[i]["c0"]))
    left_i = min(range(4), key=lambda i: (infos[i]["c1"], infos[i]["c0"], infos[i]["r0"]))
    right_i = max(range(4), key=lambda i: (infos[i]["c0"], infos[i]["c1"], infos[i]["r0"]))
    assert len({top_i, bottom_i, left_i, right_i}) == 4, "marker roles must be unique"

    top = infos[top_i]
    bottom = infos[bottom_i]
    left = infos[left_i]
    right = infos[right_i]
    assert top["vertical"] and bottom["vertical"], "top/bottom tails must be vertical"
    assert left["horizontal"] and right["horizontal"], "left/right tails must be horizontal"

    top_marker_r = top["r1"]
    bottom_marker_r = bottom["r0"]
    left_marker_c = left["c1"]
    right_marker_c = right["c0"]
    assert top_marker_r < bottom_marker_r
    assert left_marker_c < right_marker_c

    up_len = top["size"] - 1
    down_len = bottom["size"] - 1
    left_len = left["size"] - 1
    right_len = right["size"] - 1
    assert min(up_len, down_len, left_len, right_len) >= 1

    top_edge = top_marker_r - up_len
    bottom_edge = bottom_marker_r + down_len
    left_edge = left_marker_c - left_len
    right_edge = right_marker_c + right_len
    assert top_edge == top["r0"]
    assert bottom_edge == bottom["r1"]
    assert left_edge == left["c0"]
    assert right_edge == right["c1"]

    return {
        "top": top_edge,
        "bottom": bottom_edge,
        "left": left_edge,
        "right": right_edge,
    }


def _objects(grid: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    out: list[dict[str, np.ndarray | int]] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BG, MARKER):
            continue
        for coords in _components(grid == color):
            out.append({"color": color, "coords": coords})
    return out


def _strictly_inside(coords: np.ndarray, top: int, bottom: int, left: int, right: int) -> bool:
    return bool(
        np.all(coords[:, 0] > top)
        and np.all(coords[:, 0] < bottom)
        and np.all(coords[:, 1] > left)
        and np.all(coords[:, 1] < right)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure four gray tail lengths to expand a rectangle, draw its frame, and recolor only fully enclosed objects.

    Input structure:
    - The background is black.
    - Exactly four disconnected gray components are straight one-pixel-wide tails with one marker pixel at the inward end.
    - The remaining foreground consists of small single-color 4-connected objects away from the gray tails.

    Rule:
    - Identify the topmost, bottommost, leftmost, and rightmost gray markers by their extreme positions, then measure each tail length excluding the marker pixel itself.
    - Expand the rectangle of marker extremes outward by those four lengths to obtain one axis-aligned frame.
    - Recolor every non-gray object to orange if and only if all of its pixels lie strictly inside that frame.
    - Draw a one-pixel-wide orange border on the frame and keep the original gray marker-tail pixels gray.

    Key insight:
    - The gray tails do not move objects; their lengths determine how far each side of the final rectangle extends, which in turn decides which small objects count as enclosed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bounds = _gray_roles(g)
    top = bounds["top"]
    bottom = bounds["bottom"]
    left = bounds["left"]
    right = bounds["right"]

    out = g.copy()
    for obj in _objects(g):
        coords = obj["coords"]
        assert isinstance(coords, np.ndarray)
        if _strictly_inside(coords, top, bottom, left, right):
            out[coords[:, 0], coords[:, 1]] = FRAME

    out[top, left : right + 1] = FRAME
    out[bottom, left : right + 1] = FRAME
    out[top : bottom + 1, left] = FRAME
    out[top : bottom + 1, right] = FRAME
    out[g == MARKER] = MARKER
    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.asarray([[ch == "#" for ch in row] for row in rows], dtype=bool)


_BASE_SHAPES = [
    _mask_from_rows(["###"]),
    _mask_from_rows(["##", "#."]),
    _mask_from_rows(["##", "##"]),
    _mask_from_rows(["###", ".#."]),
    _mask_from_rows(["##.", ".##"]),
    _mask_from_rows(["###", "#.."]),
    _mask_from_rows(["####"]),
    _mask_from_rows(["###", "#.#"]),
    _mask_from_rows(["###", "##."]),
    _mask_from_rows(["###", ".##"]),
    _mask_from_rows(["###", "###"]),
    _mask_from_rows(["###", "###", "#.."]),
    _mask_from_rows(["###", "###", ".#."]),
]


def _shape_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in _BASE_SHAPES:
        for k in range(4):
            rot = np.rot90(base, k)
            for variant in (rot, np.fliplr(rot)):
                key = variant.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                variants.append(variant.copy())
    return variants


_SHAPES = _shape_variants()


def _update_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _valid_rect(rect: tuple[int, int, int, int]) -> bool:
    r0, r1, c0, c1 = rect
    return r0 <= r1 and c0 <= c1


def _candidate_positions(mask: np.ndarray, rect: tuple[int, int, int, int], blocked: np.ndarray) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = rect
    h, w = mask.shape
    if r1 - r0 + 1 < h or c1 - c0 + 1 < w:
        return []
    cells = np.argwhere(mask)
    positions = []
    for br in range(r0, r1 - h + 2):
        for bc in range(c0, c1 - w + 2):
            rr = cells[:, 0] + br
            cc = cells[:, 1] + bc
            if not np.any(blocked[rr, cc]):
                positions.append((br, bc))
    random.shuffle(positions)
    return positions


def _place_in_rects(
    grid: np.ndarray,
    blocked: np.ndarray,
    rects: list[tuple[str, tuple[int, int, int, int]]],
    color: int,
) -> tuple[np.ndarray, str] | None:
    options: list[tuple[np.ndarray, tuple[int, int], str]] = []
    shape_order = _SHAPES[:]
    random.shuffle(shape_order)
    rect_order = rects[:]
    random.shuffle(rect_order)
    for name, rect in rect_order:
        if not _valid_rect(rect):
            continue
        for mask in shape_order:
            positions = _candidate_positions(mask, rect, blocked)
            if positions:
                options.append((mask, positions[0], name))
    if not options:
        return None

    mask, (br, bc), name = random.choice(options)
    coords = np.argwhere(mask) + np.asarray([br, bc], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = color
    _update_blocked(blocked, coords)
    return coords, name


def _sample_lengths() -> tuple[int, int, int, int]:
    while True:
        vals = tuple(random.randint(1, 4) for _ in range(4))
        if len(set(vals)) >= 2 and max(vals) >= 2:
            return vals


def _sample_scene_geometry() -> dict[str, int] | None:
    h = random.randint(14, 19)
    w = random.randint(16, 21)
    up_len, down_len, left_len, right_len = _sample_lengths()

    min_frame_h = up_len + down_len + 7
    min_frame_w = left_len + right_len + 8
    if min_frame_h > h - 2 or min_frame_w > w - 2:
        return None

    frame_h = random.randint(min_frame_h, h - 2)
    frame_w = random.randint(min_frame_w, w - 2)
    top = random.randint(1, h - frame_h - 1)
    left = random.randint(1, w - frame_w - 1)
    bottom = top + frame_h - 1
    right = left + frame_w - 1

    top_marker_r = top + up_len
    bottom_marker_r = bottom - down_len
    left_marker_c = left + left_len
    right_marker_c = right - right_len

    marker_cols = [
        c
        for c in range(left_marker_c + 1, right_marker_c)
        if c not in (left_marker_c, right_marker_c)
    ]
    marker_rows = [
        r
        for r in range(top_marker_r + 1, bottom_marker_r)
        if r not in (top_marker_r, bottom_marker_r)
    ]
    if len(marker_cols) < 2 or len(marker_rows) < 2:
        return None

    top_marker_c, bottom_marker_c = random.sample(marker_cols, 2)
    left_marker_r, right_marker_r = random.sample(marker_rows, 2)

    if len({top_marker_r, bottom_marker_r, left_marker_r, right_marker_r}) < 4:
        return None
    if len({top_marker_c, bottom_marker_c, left_marker_c, right_marker_c}) < 4:
        return None

    return {
        "h": h,
        "w": w,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "up_len": up_len,
        "down_len": down_len,
        "left_len": left_len,
        "right_len": right_len,
        "top_marker_r": top_marker_r,
        "bottom_marker_r": bottom_marker_r,
        "left_marker_c": left_marker_c,
        "right_marker_c": right_marker_c,
        "top_marker_c": top_marker_c,
        "bottom_marker_c": bottom_marker_c,
        "left_marker_r": left_marker_r,
        "right_marker_r": right_marker_r,
    }


def _build_gray_grid(scene: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    h = scene["h"]
    w = scene["w"]
    g = np.zeros((h, w), dtype=int)
    gray_mask = np.zeros((h, w), dtype=bool)

    parts = [
        (slice(scene["top"], scene["top_marker_r"] + 1), scene["top_marker_c"]),
        (slice(scene["bottom_marker_r"], scene["bottom"] + 1), scene["bottom_marker_c"]),
        (scene["left_marker_r"], slice(scene["left"], scene["left_marker_c"] + 1)),
        (scene["right_marker_r"], slice(scene["right_marker_c"], scene["right"] + 1)),
    ]
    for row_sel, col_sel in parts:
        g[row_sel, col_sel] = MARKER
        gray_mask[row_sel, col_sel] = True
    return g, gray_mask


def _sample_object_colors(count: int) -> list[int]:
    palette_size = random.randint(2, min(3, count))
    palette = random.sample(list(OBJ_COLORS), palette_size)
    colors = palette[:]
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _oracle_output(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> np.ndarray:
    out = grid.copy()
    for obj in _objects(grid):
        coords = obj["coords"]
        assert isinstance(coords, np.ndarray)
        if _strictly_inside(coords, top, bottom, left, right):
            out[coords[:, 0], coords[:, 1]] = FRAME
    out[top, left : right + 1] = FRAME
    out[bottom, left : right + 1] = FRAME
    out[top : bottom + 1, left] = FRAME
    out[top : bottom + 1, right] = FRAME
    out[grid == MARKER] = MARKER
    return out


def _inside_marker_bbox(coords: np.ndarray, scene: dict[str, int]) -> bool:
    return bool(
        np.all(coords[:, 0] > scene["top_marker_r"])
        and np.all(coords[:, 0] < scene["bottom_marker_r"])
        and np.all(coords[:, 1] > scene["left_marker_c"])
        and np.all(coords[:, 1] < scene["right_marker_c"])
    )


def _choose_extra_kind() -> str:
    return random.choices(["core", "ring", "outside"], weights=[2, 3, 3], k=1)[0]


def _gen_attempt() -> np.ndarray | None:
    scene = _sample_scene_geometry()
    if scene is None:
        return None

    grid, gray_mask = _build_gray_grid(scene)
    blocked = np.zeros_like(gray_mask)
    _update_blocked(blocked, np.argwhere(gray_mask))

    core_rect = (
        scene["top_marker_r"] + 1,
        scene["bottom_marker_r"] - 1,
        scene["left_marker_c"] + 1,
        scene["right_marker_c"] - 1,
    )
    ring_rects = [
        ("ring_top", (scene["top"] + 1, scene["top_marker_r"] - 1, scene["left"] + 1, scene["right"] - 1)),
        ("ring_bottom", (scene["bottom_marker_r"] + 1, scene["bottom"] - 1, scene["left"] + 1, scene["right"] - 1)),
        ("ring_left", (scene["top"] + 1, scene["bottom"] - 1, scene["left"] + 1, scene["left_marker_c"] - 1)),
        ("ring_right", (scene["top"] + 1, scene["bottom"] - 1, scene["right_marker_c"] + 1, scene["right"] - 1)),
    ]
    outside_rects = [
        ("outside_top", (0, scene["top"] - 1, 0, scene["w"] - 1)),
        ("outside_bottom", (scene["bottom"] + 1, scene["h"] - 1, 0, scene["w"] - 1)),
        ("outside_left", (scene["top"], scene["bottom"], 0, scene["left"] - 1)),
        ("outside_right", (scene["top"], scene["bottom"], scene["right"] + 1, scene["w"] - 1)),
    ]

    object_count = random.randint(4, 6)
    kinds = ["core", "ring", "outside"]
    while len(kinds) < object_count:
        kinds.append(_choose_extra_kind())
    random.shuffle(kinds)
    colors = _sample_object_colors(object_count)

    objects = []
    for kind, color in zip(kinds, colors):
        if kind == "core":
            placed = _place_in_rects(grid, blocked, [("core", core_rect)], color)
        elif kind == "ring":
            placed = _place_in_rects(grid, blocked, ring_rects, color)
        else:
            placed = _place_in_rects(grid, blocked, outside_rects, color)
        if placed is None:
            return None
        coords, region_name = placed
        objects.append({"kind": kind, "region": region_name, "coords": coords, "color": color})

    selected = [obj for obj in objects if _strictly_inside(obj["coords"], scene["top"], scene["bottom"], scene["left"], scene["right"])]
    ring_selected = [obj for obj in selected if not _inside_marker_bbox(obj["coords"], scene)]
    outside = [
        obj
        for obj in objects
        if not np.any(
            (obj["coords"][:, 0] >= scene["top"])
            & (obj["coords"][:, 0] <= scene["bottom"])
            & (obj["coords"][:, 1] >= scene["left"])
            & (obj["coords"][:, 1] <= scene["right"])
        )
    ]

    if not selected or not ring_selected or not outside:
        return None

    for obj in objects:
        coords = obj["coords"]
        on_border = np.any(
            ((coords[:, 0] == scene["top"]) | (coords[:, 0] == scene["bottom"]))
            & (coords[:, 1] >= scene["left"])
            & (coords[:, 1] <= scene["right"])
        ) or np.any(
            ((coords[:, 1] == scene["left"]) | (coords[:, 1] == scene["right"]))
            & (coords[:, 0] >= scene["top"])
            & (coords[:, 0] <= scene["bottom"])
        )
        if on_border:
            return None

    expected = _oracle_output(grid, scene["top"], scene["bottom"], scene["left"], scene["right"])
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    derived = _gray_roles(grid)
    if (
        derived["top"] != scene["top"]
        or derived["bottom"] != scene["bottom"]
        or derived["left"] != scene["left"]
        or derived["right"] != scene["right"]
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosing-frame input")
