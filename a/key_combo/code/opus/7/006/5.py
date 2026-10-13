from __future__ import annotations

import random
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader

import numpy as np

try:
    from solvers.arc_utils import Color, find_components, get_bg
except Exception:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_ref",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    Color = _ARC_UTILS.Color
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg


BAR_COLOR = Color.ORANGE
BG_COLOR = Color.BLACK
OBJECT_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.PINK)
GEN_ATTEMPTS = 500

GRID_HEIGHT_RANGE = (14, 18)
GRID_WIDTH_RANGE = (10, 15)
MIN_GROUP_GAP = 2

_CLOSED_MASK = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)

_OPEN_MASKS = [
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1], [1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]


@dataclass
class Obj:
    color: int
    patch: np.ndarray
    closed: bool
    bbox: tuple[int, int, int, int]

    @property
    def height(self) -> int:
        return int(self.patch.shape[0])

    @property
    def width(self) -> int:
        return int(self.patch.shape[1])

    @property
    def col(self) -> int:
        return int(self.bbox[1])


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any((~mask) & (~exterior)))


def _extract_objects(grid: np.ndarray) -> list[Obj]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3
    assert np.all(g[-1] == BAR_COLOR)
    assert not np.any(g[:-1] == BAR_COLOR)

    colors = [int(c) for c in np.unique(g[:-1]) if int(c) not in (BG_COLOR, BAR_COLOR)]
    comps = find_components(g[:-1], bg=BG_COLOR, connectivity=4, monochromatic=True, colors=colors)
    objects: list[Obj] = []
    for comp in comps:
        color = int(comp["color"])
        bbox = tuple(map(int, comp["bbox"]))
        coords = np.asarray(comp["coords"], dtype=int)
        mask = _mask_from_coords(coords, bbox)
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        objects.append(Obj(color=color, patch=patch, closed=_has_enclosed_hole(mask), bbox=bbox))
    objects.sort(key=lambda obj: (obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]))
    return objects


def _can_place(canvas: np.ndarray, patch: np.ndarray, row: int, col: int) -> bool:
    h, w = patch.shape
    if row < 0 or col < 0 or row + h > canvas.shape[0] or col + w > canvas.shape[1]:
        return False
    mask = patch != BG_COLOR
    return bool(np.all(canvas[row : row + h, col : col + w][mask] == BG_COLOR))


def _place(canvas: np.ndarray, patch: np.ndarray, row: int, col: int) -> None:
    h, w = patch.shape
    region = canvas[row : row + h, col : col + w]
    mask = patch != BG_COLOR
    region[mask] = patch[mask]


def _pack_group(objects: list[Obj], usable_height: int, width: int, upward: bool) -> tuple[np.ndarray, dict[int, int]]:
    canvas = np.zeros((usable_height, width), dtype=int)
    rows: dict[int, int] = {}
    for idx, obj in enumerate(objects):
        limit = usable_height - obj.height
        row_iter = range(0, limit + 1) if upward else range(limit, -1, -1)
        for row in row_iter:
            if _can_place(canvas, obj.patch, row, obj.col):
                _place(canvas, obj.patch, row, obj.col)
                rows[idx] = int(row)
                break
        else:
            raise AssertionError("group packing failed")
    return canvas, rows


def _merged_slid_state(grid: np.ndarray) -> tuple[np.ndarray, list[Obj], list[Obj]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    usable_height = h - 1
    objects = _extract_objects(g)
    closed = [obj for obj in objects if obj.closed]
    open_objects = [obj for obj in objects if not obj.closed]
    assert 1 <= len(closed) <= 2
    assert 1 <= len(open_objects) <= 4

    closed = sorted(closed, key=lambda obj: (obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]))
    open_objects = sorted(open_objects, key=lambda obj: (-obj.bbox[2], obj.bbox[1], -obj.bbox[0], obj.bbox[3]))

    closed_canvas, _ = _pack_group(closed, usable_height, w, upward=True)
    open_canvas, _ = _pack_group(open_objects, usable_height, w, upward=False)
    overlap = (closed_canvas != BG_COLOR) & (open_canvas != BG_COLOR)
    assert not np.any(overlap), "closed/open group overlap is invalid for this task"

    out = np.zeros_like(g)
    out[-1, :] = BAR_COLOR
    out[:-1, :] = closed_canvas
    fg_mask = open_canvas != BG_COLOR
    out_body = out[:-1, :]
    out_body[fg_mask] = open_canvas[fg_mask]
    return out, closed, open_objects


def _column_gap(slid: np.ndarray, c: int) -> int:
    bar_row = slid.shape[0] - 1
    for r in range(bar_row - 1, -1, -1):
        if int(slid[r, c]) not in (BG_COLOR, BAR_COLOR):
            return int((bar_row - 1) - r)
    return 0


def _gap_vector(slid: np.ndarray) -> list[int]:
    return [_column_gap(slid, c) for c in range(slid.shape[1])]


def _add_rays(slid: np.ndarray) -> np.ndarray:
    out = np.array(slid, copy=True)
    bar_row = out.shape[0] - 1
    for c in range(out.shape[1]):
        r_obj = None
        for r in range(bar_row - 1, -1, -1):
            if int(out[r, c]) not in (BG_COLOR, BAR_COLOR):
                r_obj = int(r)
                break
        if r_obj is None:
            continue
        for rr in range(bar_row - 1, r_obj, -1):
            if int(out[rr, c]) != BG_COLOR:
                break
            out[rr, c] = BAR_COLOR
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Slide closed objects to the top and open objects to the bottom, then encode each post-slide column gap as an orange upward ray.

    Input structure:
    - Black is the background and the bottom row is a full orange bar.
    - The remaining foreground consists of 3-5 monochromatic 4-connected objects in blue, red, green, or pink.
    - Some objects are closed 3x3 rings with a fully enclosed black hole; the others are open line-like sprites with no enclosed hole.

    Rule:
    - Classify each non-orange foreground component by whether its bounding box contains a black region enclosed by the object under 4-connectivity.
    - Redraw the grid on a blank canvas with the orange bar preserved, sliding closed objects upward in original top-to-bottom order and open objects downward in original bottom-to-top order, always keeping shape, color, and columns fixed.
    - For each column, find the first post-slide foreground pixel above the bar; the number of blank cells between that pixel and the bar determines the height of an orange ray grown upward from just above the bar, stopping before any foreground object.

    Key insight:
    - The ray heights are not read from the original layout: they visualize the global spacing created only after topological classification moves hole-objects upward and no-hole objects downward.
    """
    slid, _closed, _open = _merged_slid_state(np.asarray(input_grid, dtype=int))
    return _add_rays(slid)


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _sample_colors(n_objects: int) -> list[int]:
    colors = list(random.sample(list(OBJECT_COLORS), min(len(OBJECT_COLORS), n_objects)))
    while len(colors) < n_objects:
        colors.append(random.choice(OBJECT_COLORS))
    random.shuffle(colors)
    return colors


def _new_gen_obj(mask: np.ndarray, color: int, closed: bool) -> dict:
    patch = _make_patch(mask, color)
    return {
        "patch": patch,
        "closed": bool(closed),
        "color": int(color),
        "c": None,
        "input_row": None,
        "final_row": None,
    }


def _gen_objects() -> tuple[list[dict], list[dict], list[dict]]:
    n_closed = 1 if random.random() < 0.55 else 2
    n_total = random.choice((3, 4, 5))
    n_open = n_total - n_closed
    colors = _sample_colors(n_total)

    closed = [_new_gen_obj(_CLOSED_MASK.copy(), colors[i], True) for i in range(n_closed)]
    open_objects = [
        _new_gen_obj(random.choice(_OPEN_MASKS).copy(), colors[n_closed + i], False)
        for i in range(n_open)
    ]
    random.shuffle(closed)
    random.shuffle(open_objects)
    return closed, open_objects, closed + open_objects


def _assign_columns(objects: list[dict], width: int) -> bool:
    anchors: list[int] = []
    for obj in objects:
        obj_w = int(obj["patch"].shape[1])
        choices = list(range(width - obj_w + 1))
        if not choices:
            return False
        if anchors and random.random() < 0.65:
            anchor = random.choice(anchors)
            local = [c for c in choices if abs(c - anchor) <= 2]
            if local:
                choices = local
        obj["c"] = int(random.choice(choices))
        anchors.extend(range(int(obj["c"]), int(obj["c"]) + obj_w))
    return True


def _pack_gen_group(objects: list[dict], usable_height: int, width: int, upward: bool) -> np.ndarray | None:
    canvas = np.zeros((usable_height, width), dtype=int)
    for obj in objects:
        patch = obj["patch"]
        obj_h = int(patch.shape[0])
        limit = usable_height - obj_h
        rows = range(0, limit + 1) if upward else range(limit, -1, -1)
        for row in rows:
            if _can_place(canvas, patch, row, int(obj["c"])):
                _place(canvas, patch, row, int(obj["c"]))
                obj["final_row"] = int(row)
                break
        else:
            return None
    return canvas


def _touch_or_overlap_same_color(objects: list[dict], row_key: str) -> bool:
    placed: list[tuple[int, np.ndarray]] = []
    for obj in objects:
        if obj[row_key] is None:
            return True
        coords = np.argwhere(obj["patch"] != BG_COLOR)
        coords = coords + np.array([int(obj[row_key]), int(obj["c"])])
        placed.append((int(obj["color"]), coords))

    for i, (color_i, coords_i) in enumerate(placed):
        for color_j, coords_j in placed[i + 1 :]:
            if color_i != color_j:
                continue
            distances = np.abs(coords_i[:, None, :] - coords_j[None, :, :]).sum(axis=2)
            if np.any(distances <= 1):
                return True
    return False


def _render_from_rows(height: int, width: int, objects: list[dict], row_key: str) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=int)
    for obj in objects:
        _place(canvas, obj["patch"], int(obj[row_key]), int(obj["c"]))
    return canvas


def _row_candidates(usable_height: int, obj_h: int) -> list[int]:
    rows = list(range(1, usable_height - obj_h))
    interior = [r for r in rows if 2 <= r <= usable_height - obj_h - 2]
    edge = [r for r in rows if r not in interior]
    random.shuffle(interior)
    random.shuffle(edge)
    return interior + edge if random.random() < 0.75 else edge + interior


def _can_scatter(canvas: np.ndarray, obj: dict, row: int) -> bool:
    patch = obj["patch"]
    col = int(obj["c"])
    h, w = patch.shape
    usable_height, width = canvas.shape
    if row < 1 or row + h > usable_height or row + h - 1 >= usable_height - 1 or col < 0 or col + w > width:
        return False
    mask = patch != BG_COLOR
    region = canvas[row : row + h, col : col + w]
    if np.any(region[mask] != BG_COLOR):
        return False
    for rr, cc in np.argwhere(mask):
        ar = row + int(rr)
        ac = col + int(cc)
        for nr, nc in ((ar - 1, ac), (ar + 1, ac), (ar, ac - 1), (ar, ac + 1)):
            if 0 <= nr < usable_height and 0 <= nc < width and canvas[nr, nc] != BG_COLOR:
                return False
    return True


def _scatter_input(height: int, width: int, closed: list[dict], open_objects: list[dict]) -> np.ndarray | None:
    usable_height = height - 1
    canvas = np.zeros((usable_height, width), dtype=int)

    def clear_obj(obj: dict) -> None:
        patch = obj["patch"]
        row = int(obj["input_row"])
        col = int(obj["c"])
        h, w = patch.shape
        region = canvas[row : row + h, col : col + w]
        region[patch != BG_COLOR] = BG_COLOR
        obj["input_row"] = None

    def rec(ci: int, oi: int, prev_closed_top: int, prev_open_bottom: int) -> bool:
        if ci == len(closed) and oi == len(open_objects):
            return True

        choices: list[str] = []
        if ci < len(closed):
            choices.append("closed")
        if oi < len(open_objects):
            choices.append("open")
        random.shuffle(choices)

        for choice in choices:
            if choice == "closed":
                obj = closed[ci]
                for row in _row_candidates(usable_height, int(obj["patch"].shape[0])):
                    if row <= prev_closed_top or not _can_scatter(canvas, obj, row):
                        continue
                    obj["input_row"] = int(row)
                    _place(canvas, obj["patch"], row, int(obj["c"]))
                    if rec(ci + 1, oi, row, prev_open_bottom):
                        return True
                    clear_obj(obj)
            else:
                obj = open_objects[oi]
                obj_h = int(obj["patch"].shape[0])
                for row in _row_candidates(usable_height, obj_h):
                    bottom = row + obj_h - 1
                    if bottom >= prev_open_bottom or not _can_scatter(canvas, obj, row):
                        continue
                    obj["input_row"] = int(row)
                    _place(canvas, obj["patch"], row, int(obj["c"]))
                    if rec(ci, oi + 1, prev_closed_top, bottom):
                        return True
                    clear_obj(obj)
        return False

    return canvas if rec(0, 0, -1, usable_height - 1) else None


def _build_input_from_scatter(height: int, width: int, canvas: np.ndarray) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:-1, :] = canvas
    grid[-1, :] = BAR_COLOR
    return grid


def _gen_mid_and_output(height: int, width: int, closed: list[dict], open_objects: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    usable_height = height - 1
    top_canvas = _pack_gen_group(closed, usable_height, width, upward=True)
    if top_canvas is None:
        return None
    bottom_canvas = _pack_gen_group(open_objects, usable_height, width, upward=False)
    if bottom_canvas is None:
        return None
    if np.any((top_canvas != BG_COLOR) & (bottom_canvas != BG_COLOR)):
        return None
    if _touch_or_overlap_same_color(closed + open_objects, "final_row"):
        return None

    closed_bottom = max(int(obj["final_row"]) + int(obj["patch"].shape[0]) - 1 for obj in closed)
    open_top = min(int(obj["final_row"]) for obj in open_objects)
    if open_top - closed_bottom - 1 < MIN_GROUP_GAP:
        return None

    mid = np.zeros((height, width), dtype=int)
    mid[:-1, :] = top_canvas
    fg_mask = bottom_canvas != BG_COLOR
    mid_body = mid[:-1, :]
    mid_body[fg_mask] = bottom_canvas[fg_mask]
    mid[-1, :] = BAR_COLOR
    return mid, _add_rays(mid)


def _input_gap_vector(grid: np.ndarray) -> list[int]:
    return _gap_vector(grid)


def _final_kind_gap_stats(mid: np.ndarray, closed: list[dict], open_objects: list[dict]) -> tuple[list[int], list[int]]:
    kind_grid = np.full(mid.shape[:-1] + (mid.shape[1],), "", dtype=object)
    usable_height = mid.shape[0] - 1
    for obj in closed:
        for rr, cc in np.argwhere(obj["patch"] != BG_COLOR):
            kind_grid[int(obj["final_row"]) + int(rr), int(obj["c"]) + int(cc)] = "closed"
    for obj in open_objects:
        for rr, cc in np.argwhere(obj["patch"] != BG_COLOR):
            kind_grid[int(obj["final_row"]) + int(rr), int(obj["c"]) + int(cc)] = "open"

    closed_gaps: list[int] = []
    open_gaps: list[int] = []
    for c in range(mid.shape[1]):
        for r in range(usable_height - 1, -1, -1):
            if int(mid[r, c]) not in (BG_COLOR, BAR_COLOR):
                gap = int((usable_height - 1) - r)
                if kind_grid[r, c] == "closed":
                    closed_gaps.append(gap)
                elif kind_grid[r, c] == "open":
                    open_gaps.append(gap)
                break
    return closed_gaps, open_gaps


def _semantic_checks(inp: np.ndarray, mid: np.ndarray, out: np.ndarray, closed: list[dict], open_objects: list[dict]) -> bool:
    if np.array_equal(inp, out) or np.array_equal(mid, out):
        return False
    if not np.all(inp[-1] == BAR_COLOR) or np.any(inp[:-1] == BAR_COLOR):
        return False

    input_gaps = _input_gap_vector(inp)
    final_gaps = _gap_vector(mid)
    if input_gaps == final_gaps:
        return False
    if sum(a != b for a, b in zip(input_gaps, final_gaps)) < 2:
        return False
    if len(set(final_gaps)) < 3 or max(final_gaps) < 3:
        return False

    closed_moves = [int(obj["input_row"]) != int(obj["final_row"]) for obj in closed]
    open_moves = [int(obj["input_row"]) != int(obj["final_row"]) for obj in open_objects]
    if not any(closed_moves) or not any(open_moves):
        return False

    closed_gaps, open_gaps = _final_kind_gap_stats(mid, closed, open_objects)
    if not closed_gaps or not open_gaps:
        return False
    if max(closed_gaps) < min(open_gaps) + 2:
        return False
    if not np.any((out[:-1] == BAR_COLOR) & (mid[:-1] == BG_COLOR)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    closed, open_objects, all_objects = _gen_objects()

    if not _assign_columns(all_objects, width):
        return None

    mid_out = _gen_mid_and_output(height, width, closed, open_objects)
    if mid_out is None:
        return None
    mid, expected_out = mid_out

    scattered = _scatter_input(height, width, closed, open_objects)
    if scattered is None:
        return None
    if _touch_or_overlap_same_color(all_objects, "input_row"):
        return None

    inp = _build_input_from_scatter(height, width, scattered)
    if not _semantic_checks(inp, mid, expected_out, closed, open_objects):
        return None
    if not np.array_equal(main(inp), expected_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
