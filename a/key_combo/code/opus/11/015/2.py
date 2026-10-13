from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CTRL = 2
FG = 3
SMALL_FILL = 5
LARGE_FILL = 7

SMALL_DIMS = ((3, 3), (3, 4), (4, 3))
LARGE_DIMS = ((4, 4),)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_holes(coords: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    fg_mask = np.zeros((h, w), dtype=bool)
    fg_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    bg_mask = ~fg_mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if bg_mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if bg_mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = bg_mask & ~outside
    if shape is None:
        shape = (int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1)
    out = np.zeros(shape, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = holes
    return out


def _hole_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _whole_sprite_holes(sprite: np.ndarray) -> list[np.ndarray]:
    coords = np.argwhere(sprite)
    if coords.size == 0:
        return []
    return _hole_components(_component_holes(coords, sprite.shape))


def _control_edges(ctrl_points: list[tuple[int, int]], h: int, w: int) -> tuple[list[str], int | None, int | None]:
    edges: list[str] = []
    cut_col: int | None = None
    cut_row: int | None = None
    for r, c in ctrl_points:
        if r == 0:
            edges.append("top")
            cut_col = c
        elif r == h - 1:
            edges.append("bottom")
            cut_col = c
        elif c == 0:
            edges.append("left")
            cut_row = r
        elif c == w - 1:
            edges.append("right")
            cut_row = r
        else:
            raise AssertionError("control pixels must lie on the border")
    return edges, cut_col, cut_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the green sprite by border-controlled rays, keep only edge-adjacent fragments, then fill surviving holes by post-cut area.

    Input structure:
    - A black background contains one 4-connected green sprite with several enclosed black holes.
    - One or two red control pixels lie on the border; a top/bottom control defines a vertical cut column and a left/right control defines a horizontal cut row.
    - The cut lines pass through the sprite and partition it into left/right halves or four quadrants.

    Rule:
    - Remove the sprite pixels that lie on every control cut line, turning those ray traces into black background.
    - Split the remaining green pixels into 4-connected fragments and keep only the fragments whose region still touches at least one control edge.
    - For each kept fragment, recount its enclosed black pixels after the cut: fill all enclosed cells grey when the total enclosed area is 1 or 2, orange when it is 3 or more, and leave the fragment unchanged when the enclosed area is 0.

    Key insight:
    - Hole color depends on which holes remain closed after the ray cut, not on the hole sizes in the original uncut sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    ctrl_points = [(int(r), int(c)) for r, c in np.argwhere(g == CTRL)]
    assert 1 <= len(ctrl_points) <= 2
    edges, cut_col, cut_row = _control_edges(ctrl_points, h, w)

    sprite = g == FG
    assert np.any(sprite)

    cut_sprite = sprite.copy()
    if cut_col is not None:
        cut_sprite[:, cut_col] = False
    if cut_row is not None:
        cut_sprite[cut_row, :] = False

    out = np.full_like(g, BG)
    for comp in _components(cut_sprite):
        rows = comp[:, 0]
        cols = comp[:, 1]

        row_side = "both"
        if cut_row is not None:
            row_side = "top" if int(rows.max()) < cut_row else "bottom"
        col_side = "both"
        if cut_col is not None:
            col_side = "left" if int(cols.max()) < cut_col else "right"

        keep = False
        for edge in edges:
            if edge in ("top", "bottom") and row_side in (edge, "both"):
                keep = True
            if edge in ("left", "right") and col_side in (edge, "both"):
                keep = True
        if not keep:
            continue

        out[comp[:, 0], comp[:, 1]] = FG
        holes = _component_holes(comp, g.shape)
        hole_count = int(holes.sum())
        if 1 <= hole_count <= 2:
            out[holes] = SMALL_FILL
        elif hole_count >= 3:
            out[holes] = LARGE_FILL

    return out


def _draw_rect_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for cc in range(c0, c0 + width):
        cells.add((r0, cc))
        cells.add((r0 + height - 1, cc))
    for rr in range(r0 + 1, r0 + height - 1):
        cells.add((rr, c0))
        cells.add((rr, c0 + width - 1))
    for rr, cc in cells:
        grid[rr, cc] = FG
    return cells


def _status_to_dims(category: str) -> tuple[int, int]:
    if category == "small":
        return random.choice(SMALL_DIMS)
    return random.choice(LARGE_DIMS)


def _choose_cut_positions(h: int, w: int) -> tuple[int, int]:
    return random.randint(5, h - 6), random.randint(5, w - 6)


def _vertical_piece(
    grid: np.ndarray,
    *,
    top: int,
    side: str,
    cut_col: int,
    height: int,
    width: int,
    open_after_cut: bool,
) -> tuple[int, int]:
    if side == "left":
        right = cut_col if open_after_cut else cut_col - 1
        c0 = right - width + 1
    else:
        c0 = cut_col if open_after_cut else cut_col + 1
    _draw_rect_frame(grid, top, c0, height, width)
    mid_r = top + height // 2
    if not open_after_cut:
        grid[mid_r, cut_col] = FG
    return mid_r, c0


def _horizontal_piece(
    grid: np.ndarray,
    *,
    left: int,
    side: str,
    cut_row: int,
    height: int,
    width: int,
    open_after_cut: bool,
) -> tuple[int, int]:
    if side == "top":
        bottom = cut_row if open_after_cut else cut_row - 1
        r0 = bottom - height + 1
    else:
        r0 = cut_row if open_after_cut else cut_row + 1
    _draw_rect_frame(grid, r0, left, height, width)
    mid_c = left + width // 2
    if not open_after_cut:
        grid[cut_row, mid_c] = FG
    return r0, mid_c


def _build_two_control_case() -> tuple[np.ndarray, list[tuple[int, int]]]:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    cut_row, cut_col = _choose_cut_positions(h, w)
    grid = np.full((h, w), BG, dtype=int)

    v_edge = random.choice(("top", "bottom"))
    h_edge = random.choice(("left", "right"))
    ctrl_points = [
        (0 if v_edge == "top" else h - 1, cut_col),
        (cut_row, 0 if h_edge == "left" else w - 1),
    ]
    for r, c in ctrl_points:
        grid[r, c] = CTRL

    kept = ["TL", "TR", "BL", "BR"]
    vanish = {
        ("top", "left"): "BR",
        ("top", "right"): "BL",
        ("bottom", "left"): "TR",
        ("bottom", "right"): "TL",
    }[(v_edge, h_edge)]
    include_vanish = random.random() < 0.6
    labels = [lab for lab in kept if lab != vanish] + ([vanish] if include_vanish else [])

    statuses: dict[str, tuple[str, bool]] = {}
    kept_labels = [lab for lab in labels if lab != vanish]
    small_lab = random.choice(kept_labels)
    large_lab = random.choice([lab for lab in kept_labels if lab != small_lab])
    statuses[small_lab] = ("small", False)
    statuses[large_lab] = ("large", False)
    for lab in kept_labels:
        if lab in statuses:
            continue
        if random.random() < 0.7:
            statuses[lab] = (random.choice(("small", "large")), True)
        else:
            statuses[lab] = (random.choice(("small", "large")), False)
    if include_vanish:
        statuses[vanish] = (random.choice(("small", "large")), random.random() < 0.5)

    anchor_rows: list[int] = []
    top_slots = {
        "TL": cut_row - 1,
        "TR": cut_row - 1,
        "BL": cut_row + 1,
        "BR": cut_row + 1,
    }
    for lab in labels:
        category, open_after_cut = statuses[lab]
        height, width = _status_to_dims(category)
        if lab[0] == "T":
            top = top_slots[lab] - height + 1
        else:
            top = top_slots[lab]
        if top < 1 or top + height > h - 1:
            raise ValueError("piece height out of bounds")
        side = "left" if lab[1] == "L" else "right"
        mid_r, _ = _vertical_piece(
            grid,
            top=top,
            side=side,
            cut_col=cut_col,
            height=height,
            width=width,
            open_after_cut=open_after_cut,
        )
        anchor_rows.append(mid_r)

    for rr in range(min(anchor_rows), max(anchor_rows) + 1):
        grid[rr, cut_col] = FG
    return grid, ctrl_points


def _build_vertical_only_case() -> tuple[np.ndarray, list[tuple[int, int]]]:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    _, cut_col = _choose_cut_positions(h, w)
    grid = np.full((h, w), BG, dtype=int)

    edge = random.choice(("top", "bottom"))
    ctrl_points = [(0 if edge == "top" else h - 1, cut_col)]
    grid[ctrl_points[0]] = CTRL

    left_status = ("small", False)
    right_status = ("large", random.random() < 0.5)
    if random.random() < 0.5:
        left_status, right_status = right_status, left_status

    left_h, left_w = _status_to_dims(left_status[0])
    right_h, right_w = _status_to_dims(right_status[0])
    left_top = 2
    right_top = h - right_h - 2
    if right_top <= left_top + left_h:
        left_top = 1
        right_top = h - right_h - 1

    left_mid, _ = _vertical_piece(
        grid,
        top=left_top,
        side="left",
        cut_col=cut_col,
        height=left_h,
        width=left_w,
        open_after_cut=left_status[1],
    )
    right_mid, _ = _vertical_piece(
        grid,
        top=right_top,
        side="right",
        cut_col=cut_col,
        height=right_h,
        width=right_w,
        open_after_cut=right_status[1],
    )
    for rr in range(min(left_mid, right_mid), max(left_mid, right_mid) + 1):
        grid[rr, cut_col] = FG
    return grid, ctrl_points


def _build_horizontal_only_case() -> tuple[np.ndarray, list[tuple[int, int]]]:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    cut_row, _ = _choose_cut_positions(h, w)
    grid = np.full((h, w), BG, dtype=int)

    edge = random.choice(("left", "right"))
    ctrl_points = [(cut_row, 0 if edge == "left" else w - 1)]
    grid[ctrl_points[0]] = CTRL

    top_status = ("small", False)
    bottom_status = ("large", random.random() < 0.5)
    if random.random() < 0.5:
        top_status, bottom_status = bottom_status, top_status

    top_h, top_w = _status_to_dims(top_status[0])
    bottom_h, bottom_w = _status_to_dims(bottom_status[0])
    top_left = 2
    bottom_left = w - bottom_w - 2
    if bottom_left <= top_left + top_w:
        top_left = 1
        bottom_left = w - bottom_w - 1

    _, top_mid = _horizontal_piece(
        grid,
        left=top_left,
        side="top",
        cut_row=cut_row,
        height=top_h,
        width=top_w,
        open_after_cut=top_status[1],
    )
    _, bottom_mid = _horizontal_piece(
        grid,
        left=bottom_left,
        side="bottom",
        cut_row=cut_row,
        height=bottom_h,
        width=bottom_w,
        open_after_cut=bottom_status[1],
    )
    for cc in range(min(top_mid, bottom_mid), max(top_mid, bottom_mid) + 1):
        grid[cut_row, cc] = FG
    return grid, ctrl_points


def _valid_input(grid: np.ndarray, ctrl_points: list[tuple[int, int]]) -> bool:
    sprite = grid == FG
    comps = _components(sprite)
    if len(comps) != 1:
        return False
    area = int(sprite.sum())
    if not (30 <= area <= 50):
        return False

    holes = _whole_sprite_holes(sprite)
    if not (2 <= len(holes) <= 4):
        return False
    if not all(1 <= int(hole.shape[0]) <= 6 for hole in holes):
        return False

    edges, cut_col, cut_row = _control_edges(ctrl_points, *grid.shape)
    if cut_col is not None and int(np.count_nonzero(sprite[:, cut_col])) == 0:
        return False
    if cut_row is not None and int(np.count_nonzero(sprite[cut_row, :])) == 0:
        return False

    out = main(grid)
    out_comps = _components(out != BG)
    if len(ctrl_points) == 2 and len(out_comps) < 3:
        return False
    if not np.any(out == SMALL_FILL):
        return False
    if not np.any(out == LARGE_FILL):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    builder = random.choices(
        [_build_two_control_case, _build_vertical_only_case, _build_horizontal_only_case],
        weights=[6, 2, 2],
    )[0]
    try:
        grid, ctrl_points = builder()
    except ValueError:
        return None
    if _valid_input(grid, ctrl_points):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
