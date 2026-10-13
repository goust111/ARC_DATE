from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FILL_COLOR = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 600

_GEN_CALL_COUNT = 0


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _foreground_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append({"coords": arr, "bbox": _bbox_from_coords(arr), "area": int(arr.shape[0])})
    return comps


def _hole_components(panel: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(panel, dtype=int)
    h, w = g.shape
    zero_mask = g == BG
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(zero_mask[r, c]) and not bool(outside[r, c]):
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(zero_mask[r, c]) and not bool(outside[r, c]):
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(zero_mask[nr, nc]) and not bool(outside[nr, nc]):
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = zero_mask & ~outside
    return _components_from_mask(holes)


def _hole_count(panel: np.ndarray) -> int:
    return len(_hole_components(panel))


def _fill_holes(panel: np.ndarray) -> np.ndarray:
    out = np.asarray(panel, dtype=int).copy()
    for coords in _hole_components(out):
        out[coords[:, 0], coords[:, 1]] = FILL_COLOR
    return out


def _repair_panel(panel: np.ndarray) -> np.ndarray:
    repaired = np.asarray(panel, dtype=int).copy()
    h, w = repaired.shape
    while True:
        changed = False
        for r in range(h):
            for c in range(w):
                if int(repaired[r, c]) != BG:
                    continue
                for rr, cc in ((r, w - 1 - c), (h - 1 - r, c), (h - 1 - r, w - 1 - c)):
                    value = int(repaired[rr, cc])
                    if value != BG:
                        repaired[r, c] = value
                        changed = True
                        break
        if not changed:
            break
    return repaired


def _panel_solution(panel: np.ndarray) -> np.ndarray:
    repaired = _repair_panel(panel)
    filled = _fill_holes(repaired)
    h, w = filled.shape
    return filled[: h // 2, : w // 2].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the foreground panel with the most pre-repair holes, restore its bilateral symmetry, fill all repaired holes red, and return its upper-left quadrant.

    Input structure:
    - The grid is a black background containing several separated rectangular foreground panels.
    - Each panel has even height and width and is intended to satisfy both left-right and up-down reflection symmetry.
    - Exactly one panel has a few foreground pixels replaced by black, while panels may also contain genuine black holes enclosed by foreground structure.

    Rule:
    - Split the scene into foreground panels and count, for each panel, the number of 4-connected black regions inside its bounding box that do not touch that box boundary.
    - Choose the unique panel with the largest such pre-repair hole count.
    - Repair black missing pixels in that panel by copying the first available non-black value from its horizontal mirror, otherwise its vertical mirror, otherwise its 180-degree counterpart.
    - After repairing, fill every enclosed black hole in the panel with red and output the panel's upper-left quadrant.

    Key insight:
    - The selected panel is determined by hole topology before repair, but the symmetry repair changes that topology by closing regions that were previously open, so hole filling must happen only after restoration.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    comps = _foreground_components(g)
    assert len(comps) >= 1, "expected at least one foreground panel"

    scored = []
    for comp in comps:
        panel = _crop_bbox(g, comp["bbox"])
        h, w = panel.shape
        assert h % 2 == 0 and w % 2 == 0, "panel dimensions must be even"
        scored.append((_hole_count(panel), panel))

    best = max(count for count, _panel in scored)
    winners = [panel for count, panel in scored if count == best]
    assert len(winners) == 1, "expected a unique panel with the most holes"
    return _panel_solution(winners[0])


def _outer_frame(h: int, w: int, color: int) -> np.ndarray:
    panel = np.zeros((h, w), dtype=int)
    panel[0, :] = color
    panel[-1, :] = color
    panel[:, 0] = color
    panel[:, -1] = color
    return panel


def _add_rect_frame(panel: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    panel[r0, c0 : c1 + 1] = color
    panel[r1, c0 : c1 + 1] = color
    panel[r0 : r1 + 1, c0] = color
    panel[r0 : r1 + 1, c1] = color


def _make_panel(kind: str, h: int, w: int, color: int) -> np.ndarray:
    panel = _outer_frame(h, w, color)
    mh = h // 2
    mw = w // 2

    if kind in {"split_v", "cross"}:
        panel[:, mw - 1 : mw + 1] = color
    if kind in {"split_h", "cross"}:
        panel[mh - 1 : mh + 1, :] = color
    if kind == "rings":
        for r0, c0 in ((1, 1), (1, w - 4), (h - 4, 1), (h - 4, w - 4)):
            _add_rect_frame(panel, r0, c0, 3, color)

    expected_holes = {"outer": 1, "split_v": 2, "split_h": 2, "cross": 4, "rings": 5}[kind]
    assert _hole_count(panel) == expected_holes, (kind, h, w, _hole_count(panel))
    return panel


def _cross_missing_positions(h: int, w: int, count: int) -> list[tuple[int, int]]:
    mw = w // 2
    safe = [(0, c) for c in range(1, mw - 1)]
    assert len(safe) >= count, (h, w, count)
    return random.sample(safe, count)


def _ring_missing_positions(h: int, w: int, mode: str) -> list[tuple[int, int]]:
    left = w // 2 - 1
    right = w // 2
    top_left = (0, left)
    top_right = (0, right)
    bottom_left = (h - 1, left)
    bottom_right = (h - 1, right)
    if mode == "h":
        return [top_left, bottom_left]
    if mode == "v":
        return [top_left, top_right]
    if mode == "rot":
        return [top_left, top_right, bottom_left]
    raise ValueError(f"unknown ring missing mode: {mode}")


def _break_target_panel(base: np.ndarray, template: str, missing_mode: str) -> np.ndarray:
    h, w = base.shape
    broken = base.copy()
    if template == "cross":
        max_safe = max(2, (w // 2) - 2)
        missing_count = random.randint(2, min(4, max_safe))
        missing = _cross_missing_positions(h, w, missing_count)
    else:
        missing = _ring_missing_positions(h, w, missing_mode)
    for r, c in missing:
        broken[r, c] = BG
    return broken


def _layout_panels(panels: list[np.ndarray]) -> np.ndarray | None:
    slot_ids = [0, 1, 2, 3]
    empty_slot = random.choice(slot_ids)
    used_slots = [slot for slot in slot_ids if slot != empty_slot]
    random.shuffle(used_slots)

    assignments = {}
    for panel, slot in zip(panels, used_slots):
        assignments[slot] = panel

    top_h = max(assignments[slot].shape[0] for slot in used_slots if slot in (0, 1))
    bottom_h = max(assignments[slot].shape[0] for slot in used_slots if slot in (2, 3))
    left_w = max(assignments[slot].shape[1] for slot in used_slots if slot in (0, 2))
    right_w = max(assignments[slot].shape[1] for slot in used_slots if slot in (1, 3))

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gap_r = random.randint(2, 3)
    gap_c = random.randint(2, 3)

    grid_h = top_margin + top_h + gap_r + bottom_h + bottom_margin
    grid_w = left_margin + left_w + gap_c + right_w + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    row_base = {0: top_margin, 1: top_margin, 2: top_margin + top_h + gap_r, 3: top_margin + top_h + gap_r}
    col_base = {0: left_margin, 1: left_margin + left_w + gap_c, 2: left_margin, 3: left_margin + left_w + gap_c}
    row_cap = {0: top_h, 1: top_h, 2: bottom_h, 3: bottom_h}
    col_cap = {0: left_w, 1: right_w, 2: left_w, 3: right_w}

    for slot, panel in assignments.items():
        ph, pw = panel.shape
        r0 = row_base[slot] + random.randint(0, row_cap[slot] - ph)
        c0 = col_base[slot] + random.randint(0, col_cap[slot] - pw)
        grid[r0 : r0 + ph, c0 : c0 + pw] = panel

    return grid


def _generation_plan() -> tuple[str, str]:
    global _GEN_CALL_COUNT
    idx = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    if idx % 6 == 0:
        return "rings", "rot"
    if idx % 6 == 1:
        return "rings", "v"
    if idx % 6 == 2:
        return "cross", "h"
    return random.choice([("rings", "h"), ("rings", "v"), ("rings", "rot"), ("cross", "h")])


def _build_instance() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    target_template, missing_mode = _generation_plan()

    target_kind = target_template
    target_dim = random.choice([10, 12] if target_kind == "rings" else [8, 10, 12])
    other_kinds = ["outer", random.choice(["split_v", "split_h"])]
    random.shuffle(other_kinds)

    colors = random.sample(INPUT_COLORS, 3)
    base_target = _make_panel(target_kind, target_dim, target_dim, colors[0])
    broken_target = _break_target_panel(base_target, target_template, missing_mode)
    if len(_foreground_components(broken_target)) != 1:
        return None
    if not np.array_equal(_repair_panel(broken_target), base_target):
        return None

    pre_target = _hole_count(broken_target)
    post_target = _hole_count(base_target)
    if not (pre_target < post_target):
        return None

    panels = [broken_target]
    pre_counts = [pre_target]
    for kind, color in zip(other_kinds, colors[1:]):
        dim = random.choice([8, 10, 12])
        panel = _make_panel(kind, dim, dim, color)
        panels.append(panel)
        pre_counts.append(_hole_count(panel))

    if pre_target != max(pre_counts) or pre_counts.count(pre_target) != 1:
        return None

    random.shuffle(panels)
    grid = _layout_panels(panels)
    if grid is None:
        return None
    return grid, broken_target, base_target, pre_target, post_target


def _validate_instance(
    grid: np.ndarray, broken_target: np.ndarray, base_target: np.ndarray, pre_target: int, post_target: int
) -> bool:
    comps = _foreground_components(grid)
    if len(comps) != 3:
        return False

    panels = [_crop_bbox(grid, comp["bbox"]) for comp in comps]
    counts = [_hole_count(panel) for panel in panels]
    if counts.count(max(counts)) != 1 or max(counts) != pre_target:
        return False

    idx = counts.index(max(counts))
    chosen = panels[idx]
    repaired = _repair_panel(chosen)
    if not np.array_equal(repaired, base_target):
        return False
    if _hole_count(chosen) != pre_target or _hole_count(repaired) != post_target:
        return False
    if np.array_equal(chosen, repaired):
        return False

    expected = _fill_holes(base_target)[: base_target.shape[0] // 2, : base_target.shape[1] // 2]
    if np.array_equal(expected, chosen[: chosen.shape[0] // 2, : chosen.shape[1] // 2]):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, broken_target, base_target, pre_target, post_target = built
        if _validate_instance(grid, broken_target, base_target, pre_target, post_target):
            return grid
    raise RuntimeError("failed to generate a valid panel-selection symmetry-hole input")
