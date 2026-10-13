from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg  # type: ignore


BLUE = 1
BLACK = 0
PALETTE = tuple(range(2, 10))
GEN_ATTEMPTS = 400

_CORNER_ROLES = {
    ((0, 0), (0, 1), (1, 0)): "TL",
    ((0, 0), (0, 1), (1, 1)): "TR",
    ((0, 0), (1, 0), (1, 1)): "BL",
    ((0, 1), (1, 0), (1, 1)): "BR",
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_role(coords: np.ndarray) -> str | None:
    if int(coords.shape[0]) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0, c1 - c0) != (1, 1):
        return None
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return _CORNER_ROLES.get(rel)


def _corner_anchor(role: str, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if role == "TL":
        return r0, c0
    if role == "TR":
        return r0, c1
    if role == "BL":
        return r1, c0
    return r1, c1


def _find_frames(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, dict[str, tuple[int, int]]] = {}

    for comp in comps:
        color = int(comp["color"])
        if color == BLUE:
            continue
        role = _corner_role(comp["coords"])
        if role is None:
            continue
        by_color.setdefault(color, {})
        assert role not in by_color[color]
        by_color[color][role] = _corner_anchor(role, comp["bbox"])

    frames = []
    for color, anchors in by_color.items():
        if set(anchors) != {"TL", "TR", "BL", "BR"}:
            continue
        top, left = anchors["TL"]
        top2, right = anchors["TR"]
        bottom, left2 = anchors["BL"]
        bottom2, right2 = anchors["BR"]
        assert top == top2 and left == left2 and bottom == bottom2 and right == right2
        assert top < bottom and left < right
        frames.append({"marker": color, "bbox": (top, left, bottom, right)})

    assert frames
    return frames


def _legend_columns(grid: np.ndarray) -> list[int]:
    bg = int(get_bg(grid))
    h, w = grid.shape
    assert h >= 2
    mask = (grid[h - 2 : h, :] != bg) & (grid[h - 2 : h, :] != BLUE)
    cols = [c for c in range(w) if bool(mask[:, c].all())]
    assert cols
    assert cols == list(range(cols[0], cols[-1] + 1))
    outside = set(range(w)) - set(cols)
    assert all(not bool(mask[:, c].any()) for c in outside)
    return cols


def _mirrored_cells(
    bbox: tuple[int, int, int, int],
    cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    top, left, bottom, right = bbox
    height = bottom - top + 1
    width = right - left + 1
    full = set()
    for r, c in cells:
        full.add((int(r), int(c)))
        if width > height:
            full.add((int(r), int(left + right - c)))
        else:
            full.add((int(top + bottom - r), int(c)))
    return full


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each blue motif inside a corner-marked rectangle, recolor it from the legend row selected by the frame aspect ratio, and erase outside blue noise.

    Input structure:
    - A black background contains 2-3 rectangular frames, each defined only by four same-colored 2x2 L-corner triominoes.
    - Every frame encloses a small blue 4-connected motif on exactly one side of its future mirror axis, plus there may be isolated blue pixels outside all frames.
    - A 2-row legend sits on the bottom edge; its columns align left-to-right with the frames sorted by each frame bbox's left column.

    Rule:
    - Recover each frame bbox from its four L-corners.
    - If a frame is wide (width > height), mirror its interior blue cells left-right and read the fill color from the legend's top row for that frame's column.
    - Otherwise mirror its interior blue cells top-bottom and read the fill color from the legend's bottom row for that frame's column.
    - Recolor every blue cell inside the completed mirrored motif to that fill color, keep markers and legend unchanged, and erase all blue cells outside the frames.

    Key insight:
    - The same geometric signal, the frame's aspect ratio, simultaneously chooses both the mirror axis and which legend row to query, so geometry and color lookup must be solved together.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))
    frames = sorted(_find_frames(g), key=lambda frame: int(frame["bbox"][1]))
    legend_cols = _legend_columns(g)
    assert len(frames) == len(legend_cols)

    out = g.copy()
    out[g == BLUE] = bg

    rr, cc = np.indices(g.shape)
    for index, frame in enumerate(frames):
        top, left, bottom, right = frame["bbox"]
        height = bottom - top + 1
        width = right - left + 1
        fill = int(g[h - 2, legend_cols[index]]) if width > height else int(g[h - 1, legend_cols[index]])
        inside = (top < rr) & (rr < bottom) & (left < cc) & (cc < right)
        blue_coords = np.argwhere((g == BLUE) & inside)
        assert len(blue_coords) > 0
        for r, c in blue_coords:
            r = int(r)
            c = int(c)
            out[r, c] = fill
            if width > height:
                out[r, left + right - c] = fill
            else:
                out[top + bottom - r, c] = fill

    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    top, left, bottom, right = box
    grid[top, left] = color
    grid[top, left + 1] = color
    grid[top + 1, left] = color

    grid[top, right - 1] = color
    grid[top, right] = color
    grid[top + 1, right] = color

    grid[bottom - 1, left] = color
    grid[bottom, left] = color
    grid[bottom, left + 1] = color

    grid[bottom - 1, right] = color
    grid[bottom, right - 1] = color
    grid[bottom, right] = color


def _padded_clear(used: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> bool:
    top, left, bottom, right = box
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(used.shape[0] - 1, bottom + pad)
    c1 = min(used.shape[1] - 1, right + pad)
    return not bool(used[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_used(used: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    top, left, bottom, right = box
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(used.shape[0] - 1, bottom + pad)
    c1 = min(used.shape[1] - 1, right + pad)
    used[r0 : r1 + 1, c0 : c1 + 1] = True


def _frame_kind_list(count: int) -> list[str]:
    kinds = ["wide", "high"]
    if count == 3:
        kinds.append(random.choice(["wide", "high", "square"]))
    random.shuffle(kinds)
    return kinds


def _frame_dims(kind: str) -> tuple[int, int]:
    if kind == "wide":
        height = random.randint(5, 7)
        width = random.randint(height + 1, min(10, height + 4))
        return height, width
    if kind == "high":
        width = random.randint(5, 7)
        height = random.randint(width + 1, min(10, width + 4))
        return height, width
    side = random.randint(6, 8)
    return side, side


def _allowed_half_cells(
    box: tuple[int, int, int, int],
    kind: str,
    source_side: str,
) -> list[tuple[int, int]]:
    top, left, bottom, right = box
    allowed = []
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if kind == "wide":
                if source_side == "left" and 2 * c < left + right:
                    allowed.append((r, c))
                if source_side == "right" and 2 * c > left + right:
                    allowed.append((r, c))
            else:
                if source_side == "top" and 2 * r < top + bottom:
                    allowed.append((r, c))
                if source_side == "bottom" and 2 * r > top + bottom:
                    allowed.append((r, c))
    return allowed


def _grow_connected_shape(
    allowed: list[tuple[int, int]],
    size: int,
) -> set[tuple[int, int]] | None:
    allowed_set = set(allowed)
    for _ in range(80):
        chosen = {random.choice(allowed)}
        while len(chosen) < size:
            frontier = [
                cell
                for cell in allowed
                if cell not in chosen
                and any(abs(cell[0] - r) + abs(cell[1] - c) == 1 for r, c in chosen)
            ]
            if not frontier:
                break
            chosen.add(random.choice(frontier))
        if len(chosen) != size:
            continue
        rows = {r for r, _c in chosen}
        cols = {c for _r, c in chosen}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return chosen
    return None


def _place_frames(
    specs: list[dict],
    height: int,
    width: int,
) -> bool:
    used = np.zeros((height, width), dtype=bool)
    used_lefts: set[int] = set()

    for spec in sorted(specs, key=lambda item: item["dims"][0] * item["dims"][1], reverse=True):
        box_h, box_w = spec["dims"]
        candidates = []
        for top in range(1, height - 2 - box_h):
            bottom = top + box_h - 1
            if bottom >= height - 3:
                continue
            for left in range(1, width - box_w):
                if left in used_lefts:
                    continue
                right = left + box_w - 1
                box = (top, left, bottom, right)
                if _padded_clear(used, box, pad=1):
                    candidates.append(box)
        if not candidates:
            return False
        box = random.choice(candidates)
        spec["bbox"] = box
        used_lefts.add(int(box[1]))
        _mark_used(used, box, pad=1)
    return True


def _paint_noise(grid: np.ndarray, frames: list[dict]) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    inside_boxes = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        top, left, bottom, right = frame["bbox"]
        inside_boxes[top : bottom + 1, left : right + 1] = True

    candidates = []
    for r in range(1, h - 3):
        for c in range(1, w - 1):
            if inside_boxes[r, c] or int(grid[r, c]) != BLACK:
                continue
            if any(int(grid[r + dr, c + dc]) != BLACK for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    count = random.randint(1, 3)
    noise = []
    for r, c in candidates:
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in noise):
            continue
        noise.append((r, c))
        if len(noise) == count:
            break
    if len(noise) != count:
        return None
    for r, c in noise:
        grid[r, c] = BLUE
    return noise


def _manual_output(inp: np.ndarray, frames: list[dict], legend_cols: list[int]) -> np.ndarray:
    out = inp.copy()
    out[out == BLUE] = BLACK
    frame_order = sorted(frames, key=lambda frame: int(frame["bbox"][1]))
    for index, frame in enumerate(frame_order):
        top, left, bottom, right = frame["bbox"]
        height = bottom - top + 1
        width = right - left + 1
        fill = frame["wide_fill"] if width > height else frame["tall_fill"]
        assert int(inp[-2, legend_cols[index]]) == int(frame["wide_fill"])
        assert int(inp[-1, legend_cols[index]]) == int(frame["tall_fill"])
        for r, c in _mirrored_cells(frame["bbox"], frame["blue_cells"]):
            out[r, c] = int(fill)
    return out


def _attempt() -> np.ndarray | None:
    frame_count = random.choice([2, 3])
    kinds = _frame_kind_list(frame_count)
    marker_colors = random.sample(PALETTE, frame_count)
    fill_pool = [color for color in PALETTE if color not in marker_colors]
    height = random.randint(18, 24)
    width = random.randint(18, 24)

    specs = []
    for marker, kind in zip(marker_colors, kinds, strict=False):
        dims = _frame_dims(kind)
        wide_fill, tall_fill = random.sample(fill_pool, 2)
        source_side = random.choice(["left", "right"]) if kind == "wide" else random.choice(["top", "bottom"])
        specs.append(
            {
                "marker": int(marker),
                "kind": kind,
                "dims": dims,
                "wide_fill": int(wide_fill),
                "tall_fill": int(tall_fill),
                "source_side": source_side,
            }
        )

    if not _place_frames(specs, height, width):
        return None

    frame_order = sorted(specs, key=lambda frame: int(frame["bbox"][1]))
    legend_start_max = width - frame_count - 1
    if legend_start_max < 1:
        return None
    legend_start = random.randint(1, legend_start_max)
    legend_cols = list(range(legend_start, legend_start + frame_count))

    inp = np.zeros((height, width), dtype=int)
    for index, frame in enumerate(frame_order):
        inp[height - 2, legend_cols[index]] = frame["wide_fill"]
        inp[height - 1, legend_cols[index]] = frame["tall_fill"]

    for frame in specs:
        _draw_frame(inp, frame["bbox"], frame["marker"])
        allowed = _allowed_half_cells(frame["bbox"], frame["kind"], frame["source_side"])
        if len(allowed) < 3:
            return None
        max_size = min(6, len(allowed))
        size = random.randint(3, max_size)
        blue_cells = _grow_connected_shape(allowed, size)
        if blue_cells is None:
            return None
        full_cells = _mirrored_cells(frame["bbox"], blue_cells)
        if full_cells == blue_cells:
            return None
        frame["blue_cells"] = blue_cells
        for r, c in blue_cells:
            inp[r, c] = BLUE

    noise = _paint_noise(inp, specs)
    if noise is None:
        return None

    expected = _manual_output(inp, specs, legend_cols)

    if np.any(expected == BLUE):
        return None
    if not any((frame["bbox"][3] - frame["bbox"][1] + 1) > (frame["bbox"][2] - frame["bbox"][0] + 1) for frame in specs):
        return None
    if not any((frame["bbox"][3] - frame["bbox"][1] + 1) <= (frame["bbox"][2] - frame["bbox"][0] + 1) for frame in specs):
        return None

    for frame in specs:
        top, left, bottom, right = frame["bbox"]
        height_box = bottom - top + 1
        width_box = right - left + 1
        fill = frame["wide_fill"] if width_box > height_box else frame["tall_fill"]
        full_cells = _mirrored_cells(frame["bbox"], frame["blue_cells"])
        if not full_cells - set(frame["blue_cells"]):
            return None
        for r, c in full_cells:
            if int(expected[r, c]) != int(fill):
                return None
        inside = expected[top + 1 : bottom, left + 1 : right]
        if BLUE in inside:
            return None

    for r, c in noise:
        if int(expected[r, c]) != BLACK:
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/9/049/1")
