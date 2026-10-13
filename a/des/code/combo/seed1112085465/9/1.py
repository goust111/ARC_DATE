from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_RETRIES = 400
SIDES = ("top", "bottom", "left", "right")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _transform_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for _ in range(random.randint(0, 3)):
        out = {(c, -r) for r, c in out}
    if random.random() < 0.5:
        out = {(r, -c) for r, c in out}
    if random.random() < 0.5:
        out = {(-r, c) for r, c in out}
    return _normalize_cells(out)


def _random_shape(max_h: int, max_w: int) -> set[tuple[int, int]]:
    box_h = random.randint(2, max_h)
    box_w = random.randint(2, max_w)
    max_area = box_h * box_w
    target = random.randint(3, max(3, min(max_area, max_area - 1)))
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_shape(max_h, max_w)
        cells.add(random.choice(frontier))
    return _transform_cells(cells)


def _paint_shape(cells: set[tuple[int, int]], palette: list[int]) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    patch = np.full((h, w), BG, dtype=int)
    base = random.choice(palette)
    accent = None
    if len(palette) >= 2 and len(cells) >= 5 and random.random() < 0.45:
        accent_choices = [c for c in palette if c != base]
        accent = random.choice(accent_choices)
    cells_list = list(cells)
    accent_cells: set[tuple[int, int]] = set()
    if accent is not None:
        accent_count = random.randint(1, min(3, len(cells_list) - 1))
        accent_cells = set(random.sample(cells_list, accent_count))
    for cell in cells_list:
        patch[cell] = accent if cell in accent_cells else base
    return patch


def _separator_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    best: tuple[int, int, list[int], list[int]] | None = None
    for color in sorted(set(map(int, grid.ravel())) - {BG}):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) < 2 or len(cols) < 2:
            continue
        row_spans = [(rows[i] + 1, rows[i + 1] - 1) for i in range(len(rows) - 1) if rows[i] + 1 <= rows[i + 1] - 1]
        col_spans = [(cols[i] + 1, cols[i + 1] - 1) for i in range(len(cols) - 1) if cols[i] + 1 <= cols[i + 1] - 1]
        if not row_spans or not col_spans:
            continue
        equal_rows = len({r1 - r0 for r0, r1 in row_spans}) == 1
        equal_cols = len({c1 - c0 for c0, c1 in col_spans}) == 1
        score = (int(equal_rows) + int(equal_cols), len(rows) * len(cols))
        candidate = (score[0], score[1], rows, cols)
        if best is None or candidate > best:
            best = candidate
            best_color = color
    assert best is not None, "failed to detect separator lattice"
    return int(best_color), best[2], best[3]


def _extract_object(panel: np.ndarray, sep: int) -> np.ndarray:
    mask = (panel != BG) & (panel != sep)
    comps = _components(mask)
    assert len(comps) == 1, "each panel should contain one connected object"
    comp = comps[0]
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in comp:
        out[r - r0, c - c0] = int(panel[r, c])
    return out


def _gap_side(panel: np.ndarray, sep: int) -> str:
    frame_mask = panel == sep
    box = _bbox(frame_mask)
    assert box is not None, "panel frame not found"
    r0, c0, r1, c1 = box

    top_missing = int(np.sum(panel[r0, c0 : c1 + 1] != sep))
    bottom_missing = int(np.sum(panel[r1, c0 : c1 + 1] != sep))
    left_missing = int(np.sum(panel[r0 : r1 + 1, c0] != sep))
    right_missing = int(np.sum(panel[r0 : r1 + 1, c1] != sep))
    missing = {
        "top": top_missing,
        "bottom": bottom_missing,
        "left": left_missing,
        "right": right_missing,
    }
    side = max(missing, key=missing.get)
    assert missing[side] == 1, "expected a unique single-pixel frame gap"
    assert sum(v > 0 for v in missing.values()) == 1, "gap side must be unique"
    return side


def _pack_horiz(parts: list[np.ndarray], bg: int) -> np.ndarray:
    if not parts:
        return np.full((0, 0), bg, dtype=int)
    h = max(part.shape[0] for part in parts)
    w = sum(part.shape[1] for part in parts)
    out = np.full((h, w), bg, dtype=int)
    c = 0
    for part in parts:
        ph, pw = part.shape
        out[:ph, c : c + pw] = part
        c += pw
    return out


def _pack_vert(parts: list[np.ndarray], bg: int) -> np.ndarray:
    if not parts:
        return np.full((0, 0), bg, dtype=int)
    h = sum(part.shape[0] for part in parts)
    w = max(part.shape[1] for part in parts)
    out = np.full((h, w), bg, dtype=int)
    r = 0
    for part in parts:
        ph, pw = part.shape
        out[r : r + ph, :pw] = part
        r += ph
    return out


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    if patch.size == 0:
        return
    ph, pw = patch.shape
    out[top : top + ph, left : left + pw] = patch


def _compose(groups: dict[str, list[np.ndarray]], bg: int) -> np.ndarray:
    top_patch = _pack_horiz(groups["top"], bg)
    bottom_patch = _pack_horiz(groups["bottom"], bg)
    left_patch = _pack_vert(groups["left"], bg)
    right_patch = _pack_vert(groups["right"], bg)

    if left_patch.size == 0 and right_patch.size == 0:
        w = max(top_patch.shape[1], bottom_patch.shape[1])
        out = np.full((top_patch.shape[0] + bottom_patch.shape[0], w), bg, dtype=int)
        _paste(out, top_patch, 0, 0)
        _paste(out, bottom_patch, out.shape[0] - bottom_patch.shape[0], 0)
        return out
    if top_patch.size == 0 and bottom_patch.size == 0:
        h = max(left_patch.shape[0], right_patch.shape[0])
        out = np.full((h, left_patch.shape[1] + right_patch.shape[1]), bg, dtype=int)
        _paste(out, left_patch, h - left_patch.shape[0], 0)
        _paste(out, right_patch, 0, out.shape[1] - right_patch.shape[1])
        return out

    top_h = max(top_patch.shape[0], right_patch.shape[0])
    bottom_h = max(left_patch.shape[0], bottom_patch.shape[0])
    left_w = max(top_patch.shape[1], left_patch.shape[1])
    right_w = max(right_patch.shape[1], bottom_patch.shape[1])

    out = np.full((top_h + bottom_h, left_w + right_w), bg, dtype=int)
    _paste(out, top_patch, 0, 0)
    _paste(out, right_patch, 0, left_w + right_w - right_patch.shape[1])
    _paste(out, left_patch, top_h + bottom_h - left_patch.shape[0], 0)
    _paste(out, bottom_patch, top_h + bottom_h - bottom_patch.shape[0], left_w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract one object from each framed panel and group it by the panel-frame gap side.

    The full-grid grey separator rows/columns define equal panel slots. Inside each slot,
    a grey rectangular frame has exactly one one-pixel gap. The inner colored object is
    cropped to its tight bbox and moved into a canonical 2x2 output layout:
    top-gap -> upper-left block, right-gap -> upper-right block,
    left-gap -> lower-left block, bottom-gap -> lower-right block.

    Within a block, top/bottom groups keep panel column order and are packed
    left-to-right; left/right groups keep panel row order and are packed top-to-bottom.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep, full_rows, full_cols = _separator_color(g)
    row_spans = [(full_rows[i] + 1, full_rows[i + 1] - 1) for i in range(len(full_rows) - 1) if full_rows[i] + 1 <= full_rows[i + 1] - 1]
    col_spans = [(full_cols[i] + 1, full_cols[i + 1] - 1) for i in range(len(full_cols) - 1) if full_cols[i] + 1 <= full_cols[i + 1] - 1]
    assert row_spans and col_spans

    groups_with_keys: dict[str, list[tuple[tuple[int, int], np.ndarray]]] = {side: [] for side in SIDES}
    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            panel = g[r0 : r1 + 1, c0 : c1 + 1]
            side = _gap_side(panel, sep)
            obj = _extract_object(panel, sep)
            if side in {"top", "bottom"}:
                key = (pc, pr)
            else:
                key = (pr, pc)
            groups_with_keys[side].append((key, obj))

    groups = {side: [obj for _key, obj in sorted(items, key=lambda item: item[0])] for side, items in groups_with_keys.items()}
    return _compose(groups, BG)


def _panel_grid_shape() -> tuple[int, int]:
    return random.choice([(2, 2), (2, 3), (3, 2), (3, 3)])


def _direction_list(count: int) -> list[str]:
    dirs = list(SIDES)
    while len(dirs) < count:
        dirs.append(random.choice(SIDES))
    random.shuffle(dirs)
    return dirs


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols = _panel_grid_shape()
    slot_h = random.randint(8, 11)
    slot_w = random.randint(8, 11)
    sep = 5

    h = panel_rows * slot_h + (panel_rows + 1)
    w = panel_cols * slot_w + (panel_cols + 1)
    grid = np.full((h, w), BG, dtype=int)

    full_rows = [i * (slot_h + 1) for i in range(panel_rows + 1)]
    full_cols = [i * (slot_w + 1) for i in range(panel_cols + 1)]
    grid[full_rows, :] = sep
    grid[:, full_cols] = sep

    palette = [c for c in range(1, 10) if c != sep]
    directions = _direction_list(panel_rows * panel_cols)

    idx = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = full_rows[pr] + 1
            c0 = full_cols[pc] + 1
            panel = grid[r0 : r0 + slot_h, c0 : c0 + slot_w]

            frame_top = 1
            frame_left = 1
            frame_bottom = slot_h - 2
            frame_right = slot_w - 2
            panel[frame_top, frame_left : frame_right + 1] = sep
            panel[frame_bottom, frame_left : frame_right + 1] = sep
            panel[frame_top : frame_bottom + 1, frame_left] = sep
            panel[frame_top : frame_bottom + 1, frame_right] = sep

            side = directions[idx]
            idx += 1
            if side == "top":
                gap_c = random.randint(frame_left + 1, frame_right - 1)
                panel[frame_top, gap_c] = BG
            elif side == "bottom":
                gap_c = random.randint(frame_left + 1, frame_right - 1)
                panel[frame_bottom, gap_c] = BG
            elif side == "left":
                gap_r = random.randint(frame_top + 1, frame_bottom - 1)
                panel[gap_r, frame_left] = BG
            else:
                gap_r = random.randint(frame_top + 1, frame_bottom - 1)
                panel[gap_r, frame_right] = BG

            max_obj_h = slot_h - 4
            max_obj_w = slot_w - 4
            shape = _random_shape(max_obj_h, max_obj_w)
            patch = _paint_shape(shape, palette)
            ph, pw = patch.shape
            inner_top = frame_top + 1
            inner_left = frame_left + 1
            inner_h = frame_bottom - frame_top - 1
            inner_w = frame_right - frame_left - 1
            if ph > inner_h or pw > inner_w:
                return None
            off_r = random.randint(inner_top, inner_top + inner_h - ph)
            off_c = random.randint(inner_left, inner_left + inner_w - pw)
            for rr in range(ph):
                for cc in range(pw):
                    if patch[rr, cc] != BG:
                        panel[off_r + rr, off_c + cc] = int(patch[rr, cc])

    try:
        out = main(grid)
    except Exception:
        return None
    if not isinstance(out, np.ndarray) or out.ndim != 2 or out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
