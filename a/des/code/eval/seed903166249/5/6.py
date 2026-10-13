from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ROTATE_FROM_UP = {"top": 0, "right": 3, "bottom": 2, "left": 1}

STAMP_SHAPES = (
    (".Y.", ".C.", "CC."),
    (".Y.", "CCC", ".C."),
    (".Y.", "CC.", ".CC"),
    (".Y.", "CCC", ".C.", ".C."),
    (".Y.", ".C.", "CCC", ".C.", ".C."),
    (".Y.", "CCC", ".C.", ".C.", ".C."),
    (".Y.", "CC.", ".C.", ".CC", ".C."),
    (".Y.", "CCC", ".C.", "CC.", ".C."),
)
OVERLAP_SHAPE_IDS = (4, 5, 7)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _yellow_pos(patch: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(patch == YELLOW)
    assert pts.shape == (1, 2)
    return int(pts[0, 0]), int(pts[0, 1])


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_info(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None
    color = colors.pop()
    if color in (BLACK, YELLOW):
        return None

    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    if h < 3 or w < 3:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(patch[~border] != BLACK):
        return None

    wrong = np.argwhere(border & (patch != color))
    if wrong.shape != (1, 2):
        return None
    gr, gc = int(wrong[0, 0]), int(wrong[0, 1])
    if int(patch[gr, gc]) != BLACK:
        return None

    side = None
    if w % 2 == 1 and (gr, gc) == (0, w // 2):
        side = "top"
    elif w % 2 == 1 and (gr, gc) == (h - 1, w // 2):
        side = "bottom"
    elif h % 2 == 1 and (gr, gc) == (h // 2, 0):
        side = "left"
    elif h % 2 == 1 and (gr, gc) == (h // 2, w - 1):
        side = "right"
    if side is None:
        return None

    expected = np.full((h, w), BLACK, dtype=int)
    expected[border] = color
    expected[gr, gc] = BLACK
    if not np.array_equal(patch, expected):
        return None

    bbox = (r0, c0, r1, c1)
    return {
        "color": color,
        "bbox": bbox,
        "gap_side": side,
        "gap_abs": (r0 + gr, c0 + gc),
        "border_cells": _border_cells(bbox),
    }


def _stamp_info(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    values = [int(grid[r, c]) for r, c in coords]
    colors = set(values)
    if YELLOW not in colors or len(colors) != 2:
        return None
    if values.count(YELLOW) != 1:
        return None
    colors.remove(YELLOW)
    color = colors.pop()
    if color == BLACK:
        return None
    if not (3 <= len(values) <= 8):
        return None

    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[r, c])
    return {"color": color, "coords": coords, "patch": patch}


def _extract_order_strip(comps: list[np.ndarray], grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    h = grid.shape[0]
    dots = []
    for comp in comps:
        if int(comp.shape[0]) != 1:
            continue
        r, c = map(int, comp[0])
        color = int(grid[r, c])
        if r < h - 2 or color in (BLACK, YELLOW):
            continue
        dots.append((r, c, color))

    assert dots
    best_row = max({r for r, _c, _color in dots}, key=lambda r: sum(rr == r for rr, _cc, _vv in dots))
    chosen = [(c, color) for r, c, color in dots if r == best_row]
    chosen.sort()
    colors = [color for c, color in chosen]
    assert len(colors) == len(set(colors))
    return colors, {(best_row, c) for c, _color in chosen}


def _rotate_stamp_to_side(patch: np.ndarray, side: str) -> np.ndarray:
    matches = []
    for k in range(4):
        rotated = np.rot90(patch, k)
        yr, yc = _yellow_pos(rotated)
        h, w = rotated.shape
        ok = (
            (side == "top" and yr == 0)
            or (side == "right" and yc == w - 1)
            or (side == "bottom" and yr == h - 1)
            or (side == "left" and yc == 0)
        )
        if ok:
            interior_bonus = int(
                (side in ("top", "bottom") and 0 < yc < w - 1)
                or (side in ("left", "right") and 0 < yr < h - 1)
            )
            matches.append((interior_bonus, rotated))
    assert matches
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each marked stamp toward the matching frame gap, then stamp in bottom-strip order."""
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BLACK)
    order, order_cells = _extract_order_strip(comps, g)

    frames: dict[int, dict] = {}
    stamps: dict[int, dict] = {}
    used_order = set(order_cells)

    for comp in comps:
        if int(comp.shape[0]) == 1 and tuple(map(int, comp[0])) in used_order:
            continue

        frame = _frame_info(g, comp)
        if frame is not None:
            assert frame["color"] not in frames
            frames[frame["color"]] = frame
            continue

        stamp = _stamp_info(g, comp)
        if stamp is not None:
            assert stamp["color"] not in stamps
            stamps[stamp["color"]] = stamp
            continue

        raise AssertionError("unrecognized component")

    assert set(order) == set(frames) == set(stamps)

    out = g.copy()
    for r, c in used_order:
        out[r, c] = BLACK
    for stamp in stamps.values():
        coords = stamp["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for color in order:
        frame = frames[color]
        patch = _rotate_stamp_to_side(stamps[color]["patch"], frame["gap_side"])
        yr, yc = _yellow_pos(patch)
        top = int(frame["gap_abs"][0] - yr)
        left = int(frame["gap_abs"][1] - yc)
        for rr, cc in np.argwhere(patch != BLACK):
            value = int(patch[rr, cc])
            if value == YELLOW:
                value = color
            out[top + int(rr), left + int(cc)] = value

    return out


def _make_stamp_patch(shape_id: int, color: int) -> np.ndarray:
    rows = STAMP_SHAPES[shape_id]
    patch = np.full((len(rows), len(rows[0])), BLACK, dtype=int)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "Y":
                patch[r, c] = YELLOW
            elif ch == "C":
                patch[r, c] = color
    return patch


def _frame_spec(color: int, r0: int, c0: int, h: int, w: int, side: str) -> dict:
    gap = {
        "top": (r0, c0 + w // 2),
        "right": (r0 + h // 2, c0 + w - 1),
        "bottom": (r0 + h - 1, c0 + w // 2),
        "left": (r0 + h // 2, c0),
    }[side]
    return {
        "color": color,
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "gap_side": side,
        "gap_abs": gap,
        "border_cells": _border_cells((r0, c0, r0 + h - 1, c0 + w - 1)),
    }


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    gr, gc = frame["gap_abs"]
    grid[gr, gc] = BLACK


def _placed_output_cells(frame: dict, up_patch: np.ndarray) -> tuple[np.ndarray, int, int, set[tuple[int, int]]]:
    patch = np.rot90(up_patch, ROTATE_FROM_UP[frame["gap_side"]])
    yr, yc = _yellow_pos(patch)
    top = int(frame["gap_abs"][0] - yr)
    left = int(frame["gap_abs"][1] - yc)
    cells = {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(patch != BLACK)
    }
    return patch, top, left, cells


def _validate_layout(
    frames: list[dict],
    up_patches: list[np.ndarray],
    h: int,
    w: int,
    overlap_pair: tuple[int, int] | None,
) -> list[set[tuple[int, int]]] | None:
    footprints: list[set[tuple[int, int]]] = []
    for i, frame in enumerate(frames):
        _patch, _top, _left, cells = _placed_output_cells(frame, up_patches[i])
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            return None
        for j, other in enumerate(frames):
            allowed_gap = other["gap_abs"]
            blocked = other["border_cells"] - {allowed_gap}
            if i == j:
                blocked = other["border_cells"] - {frame["gap_abs"]}
            if cells & blocked:
                return None
        footprints.append(cells)

    overlaps = []
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            inter = footprints[i] & footprints[j]
            if inter:
                overlaps.append((i, j))

    if overlap_pair is None:
        if overlaps:
            return None
    else:
        target = tuple(sorted(overlap_pair))
        if target not in {tuple(sorted(pair)) for pair in overlaps}:
            return None
        if any(tuple(sorted(pair)) != target for pair in overlaps):
            return None

    return footprints


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _odd_sizes(max_size: int, min_size: int = 5) -> list[int]:
    return [size for size in range(min_size, max_size + 1) if size % 2 == 1]


def _random_free_frame(
    color: int,
    h: int,
    w: int,
    usable_h: int,
    existing: list[dict],
    up_patches: list[np.ndarray],
    overlap_pair: tuple[int, int] | None,
    all_h: int,
    all_w: int,
) -> tuple[dict, np.ndarray] | None:
    hs = _odd_sizes(min(11, usable_h - 2))
    ws = _odd_sizes(min(11, w - 2))
    if not hs or not ws:
        return None

    for _ in range(200):
        fh = random.choice(hs)
        fw = random.choice(ws)
        if usable_h - fh < 1 or w - fw < 2:
            continue
        r0 = random.randint(1, usable_h - fh)
        c0 = random.randint(1, w - fw - 1)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if any(_boxes_too_close(bbox, frame["bbox"], pad=2) for frame in existing):
            continue

        side = random.choice(SIDES)
        shape_id = random.randrange(len(STAMP_SHAPES))
        frame = _frame_spec(color, r0, c0, fh, fw, side)
        patch = _make_stamp_patch(shape_id, color)
        if _validate_layout(existing + [frame], up_patches + [patch], all_h, all_w, overlap_pair) is None:
            continue
        return frame, patch
    return None


def _build_disjoint_scene(h: int, w: int, colors: list[int]) -> tuple[list[dict], list[np.ndarray]] | None:
    usable_h = h - 3
    frames: list[dict] = []
    patches: list[np.ndarray] = []
    for color in colors:
        candidate = _random_free_frame(color, h, w, usable_h, frames, patches, None, h, w)
        if candidate is None:
            return None
        frame, patch = candidate
        frames.append(frame)
        patches.append(patch)
    return frames, patches


def _build_nested_scene(h: int, w: int, colors: list[int]) -> tuple[list[dict], list[np.ndarray]] | None:
    usable_h = h - 3
    frames: list[dict] = []
    patches: list[np.ndarray] = []

    for _ in range(300):
        inner_h = random.choice([5, 7])
        inner_w = random.choice([5, 7])
        outer_h = inner_h + random.choice([4, 6])
        outer_w = inner_w + random.choice([4, 6])
        if outer_h > usable_h - 1 or outer_w > w - 2:
            continue

        r0 = random.randint(1, usable_h - outer_h)
        c0 = random.randint(1, w - outer_w - 1)
        ir0 = r0 + (outer_h - inner_h) // 2
        ic0 = c0 + (outer_w - inner_w) // 2

        axis = random.choice(["vertical", "horizontal"])
        if axis == "vertical":
            outer_side, inner_side = random.choice([("bottom", "top"), ("top", "bottom")])
        else:
            outer_side, inner_side = random.choice([("right", "left"), ("left", "right")])

        outer = _frame_spec(colors[0], r0, c0, outer_h, outer_w, outer_side)
        inner = _frame_spec(colors[1], ir0, ic0, inner_h, inner_w, inner_side)
        outer_patch = _make_stamp_patch(random.choice(OVERLAP_SHAPE_IDS), colors[0])
        inner_patch = _make_stamp_patch(random.choice(OVERLAP_SHAPE_IDS), colors[1])
        if _validate_layout([outer, inner], [outer_patch, inner_patch], h, w, (0, 1)) is None:
            continue
        frames = [outer, inner]
        patches = [outer_patch, inner_patch]
        break
    else:
        return None

    for color in colors[2:]:
        candidate = _random_free_frame(color, h, w, usable_h, frames, patches, (0, 1), h, w)
        if candidate is None:
            return None
        frame, patch = candidate
        frames.append(frame)
        patches.append(patch)

    return frames, patches


def _update_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_input_stamp(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, usable_h: int) -> bool:
    ph, pw = patch.shape
    if ph > usable_h or pw > grid.shape[1]:
        return False

    candidates = [
        (r0, c0)
        for r0 in range(0, usable_h - ph + 1)
        for c0 in range(0, grid.shape[1] - pw + 1)
    ]
    random.shuffle(candidates)
    fg = [(int(r), int(c)) for r, c in np.argwhere(patch != BLACK)]

    for r0, c0 in candidates:
        abs_cells = [(r0 + r, c0 + c) for r, c in fg]
        if any(bool(blocked[r, c]) for r, c in abs_cells):
            continue
        for r, c in abs_cells:
            grid[r, c] = int(patch[r - r0, c - c0])
        _update_blocked(blocked, abs_cells)
        return True
    return False


def _sample_strip_columns(width: int, count: int) -> list[int] | None:
    slots = list(range(1, width - 1))
    random.shuffle(slots)
    cols: list[int] = []
    for c in slots:
        if all(abs(c - other) > 1 for other in cols):
            cols.append(c)
            if len(cols) == count:
                return sorted(cols)
    return None


def _build_input_grid(h: int, w: int, frames: list[dict], up_patches: list[np.ndarray]) -> np.ndarray | None:
    usable_h = h - 3
    grid = np.full((h, w), BLACK, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[usable_h:, :] = True
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    order = random.sample([frame["color"] for frame in frames], len(frames))
    shuffled = list(range(len(frames)))
    random.shuffle(shuffled)
    for i in shuffled:
        patch = np.rot90(up_patches[i], random.randrange(4))
        if not _place_input_stamp(grid, blocked, patch, usable_h):
            return None

    strip_row = h - 1 if random.random() < 0.5 else h - 2
    cols = _sample_strip_columns(w, len(order))
    if cols is None:
        return None
    for col, color in zip(cols, order):
        grid[strip_row, col] = int(color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        overlap_mode = random.random() < 0.5
        if overlap_mode:
            h = random.randint(22, 28)
            w = random.randint(22, 28)
            count = random.randint(2, 3)
            colors = random.sample(PALETTE, count)
            scene = _build_nested_scene(h, w, colors)
        else:
            h = random.randint(18, 28)
            w = random.randint(18, 28)
            count = random.randint(2, 4)
            colors = random.sample(PALETTE, count)
            scene = _build_disjoint_scene(h, w, colors)

        if scene is None:
            continue

        frames, patches = scene
        grid = _build_input_grid(h, w, frames, patches)
        if grid is None:
            continue

        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
