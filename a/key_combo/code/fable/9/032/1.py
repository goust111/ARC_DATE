from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
CANVAS_LIMIT = 30
GEN_RETRIES = 400
ROOT_GAP = 2
FRAME_SIZE_CHOICES = (7, 9, 11, 13)
NESTED_CHILD_SIZE_CHOICES = (7, 9)
CORNERS = ("TL", "TR", "BL", "BR")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            bbox = _bbox_from_coords(coords)
            comps.append({"color": int(color), "coords": coords, "bbox": bbox, "area": int(len(cells))})
    return comps


def _is_rectangular_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return False
    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return False
    patch = np.zeros((h, w), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0])
    patch[rel[:, 0], rel[:, 1]] = True
    if not patch[0, :].all() or not patch[-1, :].all():
        return False
    if not patch[:, 0].all() or not patch[:, -1].all():
        return False
    return not patch[1:-1, 1:-1].any()


def _infer_frame_color_and_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates = []
    for color in _colors_present(grid, exclude={bg}):
        comps = _components_for_color(grid, color)
        if comps and all(_is_rectangular_frame_component(comp) for comp in comps):
            total = sum(comp["area"] for comp in comps)
            candidates.append((int(color), int(total), comps))
    assert len(candidates) == 1, "expected exactly one frame color"
    frame_color, _total, frames = candidates[0]
    return int(frame_color), frames


def _complete_inside_frame(out: np.ndarray, bbox: tuple[int, int, int, int], bg: int, frame_color_only: int | None = None) -> None:
    r0, c0, r1, c1 = bbox
    coords = np.argwhere(out[r0 + 1 : r1, c0 + 1 : c1] != bg)
    snapshot = []
    for dr, dc in coords:
        r = int(r0 + 1 + dr)
        c = int(c0 + 1 + dc)
        color = int(out[r, c])
        if frame_color_only is not None and color == frame_color_only:
            continue
        snapshot.append((r, c, color))
    for r, c, color in snapshot:
        rr = int(r0 + r1 - r)
        cc = int(c0 + c1 - c)
        current = int(out[rr, cc])
        if current == bg:
            out[rr, cc] = color
        else:
            assert current == color, "reflections should not create color conflicts"


def _completed_grid(grid: np.ndarray, *, ignore_frame_color: bool = False, reverse_order: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frames = _infer_frame_color_and_frames(g, bg)
    ordered = sorted(frames, key=lambda comp: int(comp["area"]), reverse=bool(reverse_order))
    out = g.copy()
    for frame in ordered:
        _complete_inside_frame(out, frame["bbox"], bg, frame_color if ignore_frame_color else None)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every nested frame by cascading 180-degree point symmetry from inner frames to outer frames.

    Input structure:
    - A black background contains one or more same-colored hollow rectangular frames with odd heights and widths.
    - Frames may be nested, forming one or more containment trees, and inner frames lie strictly inside outer-frame interiors.
    - Each frame interior contains partial motif pixels, while nested frames contribute additional non-background content.

    Rule:
    - Treat the unique color whose connected components are hollow odd-sized rectangles as the frame color.
    - For each original frame, reflect every non-background pixel strictly inside that frame across the frame's geometric center, preserving color.
    - Process smaller nested frames before larger enclosing frames, so an outer frame reflects already-completed inner frames as whole objects.

    Key insight:
    - The outer completion must reflect the fully completed inner frame, including its border and interior, so a simple motif-only reflection or outer-first pass is insufficient.
    """
    return _completed_grid(grid)


def _draw_frame_border(h: int, w: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    if not np.any(mask):
        return
    h, w = patch.shape
    dst_slice = dst[top : top + h, left : left + w]
    dst_slice[mask] = patch[mask]


def _corner_box(h: int, w: int, corner: str) -> tuple[int, int, int, int]:
    cr = h // 2
    cc = w // 2
    if corner[0] == "T":
        r0, r1 = 1, cr - 1
    else:
        r0, r1 = cr + 1, h - 2
    if corner[1] == "L":
        c0, c1 = 1, cc - 1
    else:
        c0, c1 = cc + 1, w - 2
    return int(r0), int(c0), int(r1), int(c1)


def _random_connected_blob(box_h: int, box_w: int, n_cells: int) -> list[tuple[int, int]]:
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    while len(cells) < n_cells:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    return list(cells)


def _frame_completion_local(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    _complete_inside_frame(out, (0, 0, patch.shape[0] - 1, patch.shape[1] - 1), BG)
    return out


def _build_leaf(frame_color: int, motif_color: int, *, small: bool) -> dict:
    size_choices = NESTED_CHILD_SIZE_CHOICES if small else FRAME_SIZE_CHOICES
    h = random.choice(size_choices)
    w = random.choice(size_choices)
    patch = _draw_frame_border(h, w, frame_color)
    r0, c0, r1, c1 = _corner_box(h, w, random.choice(CORNERS))
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    max_cells = min(4, box_h * box_w)
    n_cells = random.randint(1, max_cells)
    for rr, cc in _random_connected_blob(box_h, box_w, n_cells):
        patch[r0 + rr, c0 + cc] = motif_color
    completed = _frame_completion_local(patch)
    assert np.count_nonzero(completed == motif_color) > np.count_nonzero(patch == motif_color)
    return {"input": patch, "output": completed, "nested": False}


def _free_parent_cells(src_h: int, src_w: int, child_top: int, child_left: int, child_h: int, child_w: int) -> list[tuple[int, int]]:
    free = []
    for r in range(src_h):
        for c in range(src_w):
            if child_top - 1 <= r <= child_top + child_h and child_left - 1 <= c <= child_left + child_w:
                continue
            free.append((r, c))
    return free


def _build_nested_root(frame_color: int, motif_color: int) -> dict | None:
    child = _build_leaf(frame_color, motif_color, small=True)
    child_in = child["input"]
    child_out = child["output"]
    child_h, child_w = child_in.shape
    src_h = child_h + random.randint(2, 4)
    src_w = child_w + random.randint(2, 4)
    h = 2 * src_h + 3
    w = 2 * src_w + 3
    if h > CANVAS_LIMIT - 1 or w > CANVAS_LIMIT - 1:
        return None

    patch_in = _draw_frame_border(h, w, frame_color)
    patch_base = _draw_frame_border(h, w, frame_color)

    sb_r0, sb_c0, _sb_r1, _sb_c1 = _corner_box(h, w, random.choice(CORNERS))
    if src_h - child_h < 2 or src_w - child_w < 2:
        return None
    child_top = random.randint(1, src_h - child_h - 1)
    child_left = random.randint(1, src_w - child_w - 1)
    child_r = sb_r0 + child_top
    child_c = sb_c0 + child_left
    _paste_non_bg(patch_in, child_in, child_r, child_c)
    _paste_non_bg(patch_base, child_out, child_r, child_c)

    free = _free_parent_cells(src_h, src_w, child_top, child_left, child_h, child_w)
    if not free:
        return None
    motif_cells = random.sample(free, k=random.randint(1, min(3, len(free))))
    for rr, cc in motif_cells:
        patch_in[sb_r0 + rr, sb_c0 + cc] = motif_color
        patch_base[sb_r0 + rr, sb_c0 + cc] = motif_color

    completed = _frame_completion_local(patch_base)
    if np.count_nonzero(completed == frame_color) <= np.count_nonzero(patch_in == frame_color):
        return None
    if np.array_equal(completed, patch_in):
        return None
    return {"input": patch_in, "output": completed, "nested": True}


def _rects_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_roots(root_patches: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    order = sorted(range(len(root_patches)), key=lambda i: root_patches[i]["input"].shape[0] * root_patches[i]["input"].shape[1], reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    boxes: list[tuple[int, int, int, int]] = []
    for idx in order:
        patch = root_patches[idx]["input"]
        h, w = patch.shape
        anchors = [(r, c) for r in range(CANVAS_LIMIT - h + 1) for c in range(CANVAS_LIMIT - w + 1)]
        random.shuffle(anchors)
        placed = False
        for r, c in anchors:
            box = (r, c, r + h - 1, c + w - 1)
            if all(_rects_far_enough(box, other, ROOT_GAP) for other in boxes):
                placements[idx] = (r, c)
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None

    input_canvas = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=int)
    output_canvas = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=int)
    for idx, root in enumerate(root_patches):
        r, c = placements[idx]
        _paste_non_bg(input_canvas, root["input"], r, c)
        _paste_non_bg(output_canvas, root["output"], r, c)

    non_bg = np.argwhere(input_canvas != BG)
    if non_bg.size == 0:
        return None
    r0, c0 = non_bg.min(axis=0)
    r1, c1 = non_bg.max(axis=0)
    top_pad = random.randint(0, min(4, int(r0)))
    left_pad = random.randint(0, min(4, int(c0)))
    bottom_pad = random.randint(0, min(4, CANVAS_LIMIT - 1 - int(r1)))
    right_pad = random.randint(0, min(4, CANVAS_LIMIT - 1 - int(c1)))
    rr0 = int(r0 - top_pad)
    cc0 = int(c0 - left_pad)
    rr1 = int(r1 + bottom_pad)
    cc1 = int(c1 + right_pad)
    cropped_in = input_canvas[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
    cropped_out = output_canvas[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
    if max(cropped_in.shape) > CANVAS_LIMIT:
        return None
    return cropped_in, cropped_out


def _generate_scene() -> tuple[np.ndarray, np.ndarray, int, int] | None:
    frame_color, motif_color = random.sample(range(1, 10), 2)
    nested_root = _build_nested_root(frame_color, motif_color)
    if nested_root is None:
        return None

    roots = [nested_root]
    if random.random() < 0.55:
        roots.append(_build_leaf(frame_color, motif_color, small=False))
    placed = _place_roots(roots)
    if placed is None:
        return None
    inp, expected = placed
    if np.count_nonzero(inp == BG) <= np.count_nonzero(inp != BG):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp, expected, frame_color, motif_color


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        scene = _generate_scene()
        if scene is None:
            continue
        inp, expected, frame_color, motif_color = scene
        got = main(inp)
        if not np.array_equal(got, expected):
            continue
        try:
            shortcut_ignore_frames = _completed_grid(inp, ignore_frame_color=True)
            shortcut_outer_first = _completed_grid(inp, reverse_order=True)
        except Exception:
            continue
        if np.array_equal(shortcut_ignore_frames, expected):
            continue
        if np.array_equal(shortcut_outer_first, expected):
            continue
        if np.count_nonzero(expected == frame_color) <= np.count_nonzero(inp == frame_color):
            continue
        if np.count_nonzero(expected == motif_color) <= np.count_nonzero(inp == motif_color):
            continue
        return inp
    raise RuntimeError("failed to generate a valid cascaded nested-frame sample")
