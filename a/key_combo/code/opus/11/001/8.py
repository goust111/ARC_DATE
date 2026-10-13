from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
SEP = 5
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_H_RANGE = (15, 22)
PANEL_W_RANGE = (9, 13)
SHAPE_COUNT_CHOICES = (2, 3)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, f"expected one full-height separator column, got {cols}"
    return int(cols[0])


def _slide_left_panel(left: np.ndarray, bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    out = left.copy()
    comps = _components(out == BLACK)
    out[out == BLACK] = bg
    target_col = left.shape[1] - 1
    shifted: list[np.ndarray] = []
    for comp in comps:
        max_c = int(comp[:, 1].max())
        delta = target_col - max_c
        moved = comp.copy()
        moved[:, 1] += delta
        out[moved[:, 0], moved[:, 1]] = BLACK
        shifted.append(moved)
    return out, shifted


def _enclosed_bg_components(left: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = left.shape
    enclosed: list[np.ndarray] = []
    for comp in _components(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 for r, c in comp):
            continue
        enclosed.append(comp)
    return enclosed


def _separator_holes(left: np.ndarray, bg: int) -> list[np.ndarray]:
    touch_col = left.shape[1] - 1
    return [comp for comp in _enclosed_bg_components(left, bg) if np.any(comp[:, 1] == touch_col)]


def _active_rows_from_left(left: np.ndarray, bg: int) -> list[int]:
    return sorted({int(r) for comp in _separator_holes(left, bg) for r in comp[:, 0]})


def _transform(grid: np.ndarray, *, with_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30
    assert np.all(np.isin(g, np.arange(10)))

    sep = _find_separator_col(g)
    assert w == 2 * sep + 1, "expected two equal-width panels"

    left_in = g[:, :sep]
    right = g[:, sep + 1 :]
    assert right.shape == left_in.shape

    left_slid, shifted_components = _slide_left_panel(left_in, BG)
    hole_components = _separator_holes(left_slid, BG)
    active_rows = _active_rows_from_left(left_slid, BG)

    out = g.copy()
    out[:, :sep] = left_slid

    marker_col = w - 1
    for r in active_rows:
        control = int(g[r, marker_col])
        assert control in (RED, GREEN), f"active row {r} must have a red/green control marker"
        row = out[r, sep + 1 : marker_col]
        row[row == BLUE] = control

    if with_details:
        return out, {
            "separator": sep,
            "active_rows": active_rows,
            "slid_left": left_slid,
            "shifted_components": shifted_components,
            "hole_components": hole_components,
            "all_enclosed_bg": _enclosed_bg_components(left_slid, BG),
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the left black components to the separator, use separator-touching holes to choose rows, and recolor only those rows on the right panel.

    Input structure:
    - The grid uses teal (8) background and a full-height grey (5) separator column that splits it into two equal-width side panels.
    - The left panel contains 2-3 black (0) connected shapes placed away from the separator.
    - The right panel contains a blue (1) foreground mask, and its last column is a row-marker column with teal background except for some red (2) or green (3) control pixels.

    Rule:
    - Move every black connected component in the left panel horizontally right until its rightmost black cell is flush with the separator.
    - In that slid left panel, find teal 4-connected enclosed regions that do not touch the outer left/top/bottom border and that include at least one cell in the column immediately left of the separator; collect every row occupied by such regions.
    - For each collected row, read the control color from the rightmost marker column and recolor every blue pixel in that same row of the right panel to that control color.
    - Preserve non-selected blue rows, the marker column, the separator, and the teal background; the output keeps the slid left-panel positions.

    Key insight:
    - The left panel does not paint the right panel directly: sliding changes which separator-backed holes exist, and those hole rows act as a row-selection mask for per-row recoloring on the right.
    """
    return _transform(grid)


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _emitter_patch() -> np.ndarray:
    h, w = random.choice([(3, 3), (4, 2), (5, 2), (3, 4)])
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, 0] = BLACK
    return patch


def _ring_patch() -> np.ndarray:
    h, w = random.choice([(3, 3), (3, 4), (4, 3)])
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    return patch


def _c_left_patch() -> np.ndarray:
    h, w = random.choice([(3, 3), (3, 4), (4, 3)])
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, -1] = BLACK
    return patch


def _solid_patch() -> np.ndarray:
    h, w = random.choice([(2, 3), (3, 2), (3, 3)])
    return np.full((h, w), BLACK, dtype=int)


def _control_colors(rows: list[int]) -> dict[int, int]:
    if len(rows) == 1:
        return {rows[0]: random.choice([RED, GREEN])}
    red_count = random.randint(1, len(rows) - 1)
    red_rows = set(random.sample(rows, red_count))
    return {r: (RED if r in red_rows else GREEN) for r in rows}


def _gen_blue_mask(height: int, width: int, required_rows: list[int]) -> np.ndarray | None:
    req_min = min(required_rows)
    req_max = max(required_rows)
    full_width = width + 1
    for _ in range(180):
        top = random.randint(max(0, req_min - 3), req_min)
        bottom = random.randint(req_max, min(height - 1, req_max + 3))
        span = bottom - top + 1
        if span < len(required_rows) + 1 and not any(required_rows[i] + 1 < required_rows[i + 1] for i in range(len(required_rows) - 1)):
            continue

        target_ratio = random.uniform(0.30, 0.46)
        target_area = int(round(target_ratio * height * full_width))
        base_width = max(2, min(width, target_area // max(1, span)))
        w_lo = max(2, base_width - 1)
        w_hi = min(width, base_width + 1)
        if w_lo > w_hi:
            continue

        mask = np.zeros((height, width), dtype=bool)
        seg_width = random.randint(w_lo, w_hi)
        left = random.randint(0, width - seg_width)
        for r in range(top, bottom + 1):
            if r > top:
                seg_width = random.randint(w_lo, w_hi)
                lo = max(0, left - seg_width + 1)
                hi = min(width - seg_width, left + prev_width - 1)
                if lo > hi:
                    break
                left = random.randint(lo, hi)
            mask[r, left : left + seg_width] = True
            prev_width = seg_width
        else:
            if not all(mask[r].any() for r in required_rows):
                continue
            blue_rows = {r for r in range(height) if mask[r].any()}
            if blue_rows <= set(required_rows):
                continue
            ratio = float(mask.sum()) / float(height * full_width)
            if 0.28 <= ratio <= 0.48:
                return mask
    return None


def _place_flush_components(height: int, panel_w: int, patches: list[np.ndarray]):
    min_h = sum(p.shape[0] for p in patches) + (len(patches) - 1) + 2
    if height < min_h:
        return None
    gaps = [1] * (len(patches) + 1)
    extras = _scatter(height - min_h, len(gaps))
    gaps = [g + e for g, e in zip(gaps, extras)]

    left = np.full((height, panel_w), BG, dtype=int)
    placements: list[dict] = []
    r = gaps[0]
    for idx, patch in enumerate(patches):
        h, w = patch.shape
        c = panel_w - w
        left[r : r + h, c : c + w][patch == BLACK] = BLACK
        placements.append({"row": r, "col": c, "patch": patch})
        r += h + gaps[idx + 1]
    return left, placements


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    height = random.randint(*GRID_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    width = 2 * panel_w + 1

    n_shapes = random.choice(SHAPE_COUNT_CHOICES)
    emitter_count = 1 if n_shapes == 2 else random.choice([1, 2])

    patches = []
    kinds = []
    for _ in range(emitter_count):
        patches.append(_emitter_patch())
        kinds.append("emitter")
    patches.append(_ring_patch())
    kinds.append("ring")
    while len(patches) < n_shapes:
        kind = random.choice(["c_left", "solid", "ring"])
        patch = {"c_left": _c_left_patch, "solid": _solid_patch, "ring": _ring_patch}[kind]()
        patches.append(patch)
        kinds.append(kind)

    order = list(range(n_shapes))
    random.shuffle(order)
    patches = [patches[i] for i in order]
    kinds = [kinds[i] for i in order]

    placed = _place_flush_components(height, panel_w, patches)
    if placed is None:
        return None
    flush_left, placements = placed

    active_rows = _active_rows_from_left(flush_left, BG)
    if not active_rows:
        return None

    blue_mask = _gen_blue_mask(height, panel_w - 1, active_rows)
    if blue_mask is None:
        return None

    controls = _control_colors(active_rows)
    right_in = np.full((height, panel_w), BG, dtype=int)
    right_in[:, : panel_w - 1][blue_mask] = BLUE
    for r, color in controls.items():
        right_in[r, panel_w - 1] = color

    max_shifts = [pos["col"] for pos in placements]
    deltas = [random.randint(0, shift) for shift in max_shifts]
    movable_emitters = [i for i, (kind, shift) in enumerate(zip(kinds, max_shifts)) if kind == "emitter" and shift > 0]
    if movable_emitters and all(deltas[i] == 0 for i in movable_emitters):
        pick = random.choice(movable_emitters)
        deltas[pick] = random.randint(1, max_shifts[pick])
    if all(delta == 0 for delta in deltas):
        movable = [i for i, shift in enumerate(max_shifts) if shift > 0]
        if not movable:
            return None
        pick = random.choice(movable)
        deltas[pick] = random.randint(1, max_shifts[pick])

    input_left = np.full((height, panel_w), BG, dtype=int)
    for pos, delta in zip(placements, deltas):
        patch = pos["patch"]
        r0 = pos["row"]
        c0 = pos["col"] - delta
        input_left[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch == BLACK] = BLACK

    inp = np.full((height, width), BG, dtype=int)
    inp[:, :panel_w] = input_left
    inp[:, panel_w] = SEP
    inp[:, panel_w + 1 :] = right_in

    expected = np.full((height, width), BG, dtype=int)
    expected[:, :panel_w] = flush_left
    expected[:, panel_w] = SEP
    expected[:, panel_w + 1 :] = right_in
    for r, color in controls.items():
        row = expected[r, panel_w + 1 : width - 1]
        row[row == BLUE] = color

    meta = {
        "active_rows": active_rows,
        "kinds": kinds,
        "controls": controls,
        "has_ring": "ring" in kinds,
    }
    return inp, expected, meta


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        out, details = _transform(inp, with_details=True)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    sep = details["separator"]
    active_rows = details["active_rows"]
    if active_rows != meta["active_rows"]:
        return False
    if set(active_rows) != set(meta["controls"]):
        return False
    if not any(np.any(inp[r, sep + 1 : -1] == BLUE) for r in active_rows):
        return False

    inactive_blue_rows = [r for r in range(inp.shape[0]) if r not in active_rows and np.any(inp[r, sep + 1 : -1] == BLUE)]
    if not inactive_blue_rows:
        return False
    if not any(np.array_equal(inp[r, sep + 1 : -1], out[r, sep + 1 : -1]) for r in inactive_blue_rows):
        return False
    if not any(np.any(inp[r, sep + 1 : -1] != out[r, sep + 1 : -1]) for r in active_rows):
        return False
    if not np.any(inp[:, :sep] != out[:, :sep]):
        return False

    black_rows = {r for r in range(inp.shape[0]) if np.any(out[r, :sep] == BLACK)}
    if not set(active_rows) < black_rows:
        return False

    left_comps = _components(inp[:, :sep] == BLACK)
    if len(left_comps) not in (2, 3):
        return False

    if len(active_rows) >= 2 and set(meta["controls"].values()) != {RED, GREEN}:
        return False

    if meta["has_ring"]:
        enclosed = details["all_enclosed_bg"]
        if not any(np.all(comp[:, 1] < sep - 1) for comp in enclosed):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected, meta = built
        if _validate_case(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 8")
