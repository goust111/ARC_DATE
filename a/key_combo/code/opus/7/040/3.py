from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GENERATION_RETRIES = 500
PAIR_RETRIES = 300
EXTRA_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _frame_area(rect: dict) -> int:
    return int((rect["h"] - 2) * (rect["w"] - 2))


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _draw_frame(panel: np.ndarray, rect: dict, color: int) -> None:
    r0 = int(rect["r"])
    c0 = int(rect["c"])
    h = int(rect["h"])
    w = int(rect["w"])
    panel[r0, c0 : c0 + w] = color
    panel[r0 + h - 1, c0 : c0 + w] = color
    panel[r0 : r0 + h, c0] = color
    panel[r0 : r0 + h, c0 + w - 1] = color


def _frame_mask(shape: tuple[int, int], rect: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _draw_frame(mask, rect, True)
    return mask


def _rects_too_close(a: dict, b: dict, pad: int = 1) -> bool:
    ar0, ac0 = int(a["r"]), int(a["c"])
    ar1 = ar0 + int(a["h"]) - 1
    ac1 = ac0 + int(a["w"]) - 1
    br0, bc0 = int(b["r"]), int(b["c"])
    br1 = br0 + int(b["h"]) - 1
    bc1 = bc0 + int(b["w"]) - 1
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _overlap_count(a: dict, b: dict, shape: tuple[int, int]) -> int:
    return int(np.count_nonzero(_frame_mask(shape, a) & _frame_mask(shape, b)))


def _copy_rect(rect: dict, *, r: int | None = None, c: int | None = None) -> dict:
    out = dict(rect)
    if r is not None:
        out["r"] = int(r)
    if c is not None:
        out["c"] = int(c)
    out["area"] = _frame_area(out)
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_frames(panel: np.ndarray, color: int) -> tuple[list[dict], np.ndarray]:
    comps = _components(panel == int(color))
    frames: list[dict] = []
    owner = np.full(panel.shape, -1, dtype=int)
    seen_areas: set[int] = set()
    for comp in comps:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "each component must be a hollow rectangle frame"
        sub = panel[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        assert np.all(sub[border] == int(color)), "expected a complete rectangular border"
        assert np.all(sub[~border] == BLACK), "frame interiors must be background"
        assert len(comp) == int(border.sum()), "expected a one-pixel-thick frame"
        rect = {"r": int(r0), "c": int(c0), "h": int(h), "w": int(w), "area": int((h - 2) * (w - 2))}
        assert rect["area"] not in seen_areas, "frame interior areas must be unique within a panel"
        seen_areas.add(int(rect["area"]))
        idx = len(frames)
        for rr, cc in comp:
            owner[rr, cc] = idx
        frames.append(rect)
    return frames, owner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse the separator-stacked blue and red frame panels into one green arbitration mask.

    Input structure:
    - A single solid grey row splits the input into equal-height upper and lower panels.
    - The upper panel contains 2-3 blue one-cell-thick hollow rectangular frames on black background.
    - The lower panel contains 2-3 red one-cell-thick hollow rectangular frames on black background.

    Rule:
    - Remove the grey separator and align the two panels cell by cell.
    - A cell becomes green when exactly one panel contributes a frame pixel there.
    - When both panels contribute frame pixels at the same cell, identify the owning blue and red frames, compare their enclosed interior areas, and keep the larger frame's pixel; the kept pixel is rendered green.

    Key insight:
    - The puzzle combines separator-row panel collapse with per-overlap reasoning about which hollow rectangle encloses the larger empty region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert len(separator_rows) == 1, "expected exactly one solid grey separator row"
    sep = int(separator_rows[0])
    assert int(np.count_nonzero(g == GREY)) == g.shape[1], "grey may only appear in the separator row"
    assert 2 * sep + 1 == g.shape[0], "separator must split two equal-height panels"

    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape and top.shape[0] > 0 and top.shape[1] > 0

    top_frames, top_owner = _extract_frames(top, BLUE)
    bottom_frames, bottom_owner = _extract_frames(bottom, RED)
    assert 2 <= len(top_frames) <= 3, "upper panel must contain 2-3 frames"
    assert 2 <= len(bottom_frames) <= 3, "lower panel must contain 2-3 frames"

    all_areas = [int(frame["area"]) for frame in top_frames + bottom_frames]
    assert len(all_areas) == len(set(all_areas)), "all frame interior areas must be unique"

    top_mask = top == BLUE
    bottom_mask = bottom == RED
    overlap = top_mask & bottom_mask
    for r, c in np.argwhere(overlap):
        top_frame = top_frames[int(top_owner[int(r), int(c)])]
        bottom_frame = bottom_frames[int(bottom_owner[int(r), int(c)])]
        assert int(top_frame["area"]) != int(bottom_frame["area"]), "overlap arbitration must be unambiguous"

    out = np.zeros_like(top, dtype=int)
    out[top_mask | bottom_mask] = GREEN
    return out


def _dimension_candidates(panel_h: int, panel_w: int) -> list[dict]:
    max_h = min(panel_h, 7)
    max_w = min(panel_w, 8)
    return [
        {"r": 0, "c": 0, "h": h, "w": w, "area": int((h - 2) * (w - 2))}
        for h in range(3, max_h + 1)
        for w in range(3, max_w + 1)
    ]


def _sample_pair(panel_h: int, panel_w: int, used_areas: set[int], winner: str) -> dict | None:
    dims = _dimension_candidates(panel_h, panel_w)
    for _ in range(PAIR_RETRIES):
        axis = random.choice(("h", "v"))
        shifted = random.choice(("top", "bottom"))
        if winner == "top":
            top_template = random.choice(dims)
            smaller = [
                d
                for d in dims
                if int(d["area"]) not in used_areas | {int(top_template["area"])}
                and int(d["area"]) < int(top_template["area"])
            ]
            if not smaller:
                continue
            bottom_template = random.choice(smaller)
        else:
            bottom_template = random.choice(dims)
            smaller = [
                d
                for d in dims
                if int(d["area"]) not in used_areas | {int(bottom_template["area"])}
                and int(d["area"]) < int(bottom_template["area"])
            ]
            if not smaller:
                continue
            top_template = random.choice(smaller)

        top_r = 1 if axis == "v" and shifted == "top" else 0
        top_c = 1 if axis == "h" and shifted == "top" else 0
        bottom_r = 1 if axis == "v" and shifted == "bottom" else 0
        bottom_c = 1 if axis == "h" and shifted == "bottom" else 0

        top_rel = _copy_rect(top_template, r=top_r, c=top_c)
        bottom_rel = _copy_rect(bottom_template, r=bottom_r, c=bottom_c)
        group_h = max(top_rel["r"] + top_rel["h"], bottom_rel["r"] + bottom_rel["h"])
        group_w = max(top_rel["c"] + top_rel["w"], bottom_rel["c"] + bottom_rel["w"])
        if group_h > panel_h or group_w > min(panel_w, 8):
            continue

        overlap = _overlap_count(top_rel, bottom_rel, (group_h, group_w))
        if overlap <= 0:
            continue
        total = 2 * (top_rel["h"] + top_rel["w"] - 2) + 2 * (bottom_rel["h"] + bottom_rel["w"] - 2) - overlap
        if overlap * 10 > total * 4:
            continue

        return {
            "winner": winner,
            "top": top_rel,
            "bottom": bottom_rel,
            "group_h": int(group_h),
            "group_w": int(group_w),
        }
    return None


def _place_extra_frame(
    panel_h: int,
    panel_w: int,
    same_panel: list[dict],
    other_panel: list[dict],
    used_areas: set[int],
) -> dict | None:
    dims = _dimension_candidates(panel_h, panel_w)
    random.shuffle(dims)
    for template in dims:
        if int(template["area"]) in used_areas:
            continue
        if int(template["h"]) > 5 and int(template["w"]) > 5:
            continue
        for _ in range(EXTRA_RETRIES):
            r = random.randint(0, panel_h - int(template["h"]))
            c = random.randint(0, panel_w - int(template["w"]))
            rect = _copy_rect(template, r=r, c=c)
            if any(_rects_too_close(rect, other, pad=1) for other in same_panel):
                continue
            if any(_overlap_count(rect, other, (panel_h, panel_w)) > 0 for other in other_panel):
                continue
            return rect
    return None


def _render_input(panel_h: int, panel_w: int, top_frames: list[dict], bottom_frames: list[dict]) -> np.ndarray:
    top = np.zeros((panel_h, panel_w), dtype=int)
    bottom = np.zeros((panel_h, panel_w), dtype=int)
    for rect in top_frames:
        _draw_frame(top, rect, BLUE)
    for rect in bottom_frames:
        _draw_frame(bottom, rect, RED)
    separator = np.full((1, panel_w), GREY, dtype=int)
    return np.vstack([top, separator, bottom])


def _expected_output(grid: np.ndarray) -> np.ndarray:
    sep = grid.shape[0] // 2
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    out = np.zeros_like(top, dtype=int)
    out[(top == BLUE) | (bottom == RED)] = GREEN
    return out


def _validate_generated(
    grid: np.ndarray,
    top_target: int,
    bottom_target: int,
    paired_rects: list[tuple[dict, dict, str]],
) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False

    expected = _expected_output(grid)
    if not np.array_equal(solved, expected):
        return False

    sep = grid.shape[0] // 2
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    try:
        top_frames, _ = _extract_frames(top, BLUE)
        bottom_frames, _ = _extract_frames(bottom, RED)
    except AssertionError:
        return False

    if len(top_frames) != int(top_target) or len(bottom_frames) != int(bottom_target):
        return False

    areas = [int(rect["area"]) for rect in top_frames + bottom_frames]
    if len(areas) != len(set(areas)):
        return False

    overlap = (top == BLUE) & (bottom == RED)
    union = (top == BLUE) | (bottom == RED)
    if int(np.count_nonzero(overlap)) <= 0:
        return False
    if int(np.count_nonzero((top == BLUE) & (bottom != RED))) <= 0:
        return False
    if int(np.count_nonzero((bottom == RED) & (top != BLUE))) <= 0:
        return False
    if int(np.count_nonzero(overlap)) * 10 > int(np.count_nonzero(union)) * 4:
        return False

    saw_top_win = False
    saw_bottom_win = False
    for top_rect, bottom_rect, winner in paired_rects:
        count = _overlap_count(top_rect, bottom_rect, top.shape)
        if count <= 0:
            return False
        if winner == "top":
            if int(top_rect["area"]) <= int(bottom_rect["area"]):
                return False
            saw_top_win = True
        else:
            if int(bottom_rect["area"]) <= int(top_rect["area"]):
                return False
            saw_bottom_win = True
    return saw_top_win and saw_bottom_win


def _attempt_generate() -> np.ndarray | None:
    panel_h = random.randint(7, 9)
    top_target = random.choice((2, 3))
    bottom_target = random.choice((2, 3))
    extras = (top_target - 2) + (bottom_target - 2)
    panel_w = random.randint(16, 20) if extras else random.randint(14, 18)

    used_areas: set[int] = set()
    pair_top = _sample_pair(panel_h, panel_w, used_areas, winner="top")
    if pair_top is None:
        return None
    used_areas.update({int(pair_top["top"]["area"]), int(pair_top["bottom"]["area"])})

    pair_bottom = _sample_pair(panel_h, panel_w, used_areas, winner="bottom")
    if pair_bottom is None:
        return None
    used_areas.update({int(pair_bottom["top"]["area"]), int(pair_bottom["bottom"]["area"])})

    gap = random.randint(2, 4)
    needed_w = int(pair_top["group_w"]) + gap + int(pair_bottom["group_w"])
    if needed_w > panel_w:
        return None

    left_c = random.randint(0, panel_w - needed_w)
    right_c_min = left_c + int(pair_top["group_w"]) + gap
    right_c = random.randint(right_c_min, panel_w - int(pair_bottom["group_w"]))
    left_r = random.randint(0, panel_h - int(pair_top["group_h"]))
    right_r = random.randint(0, panel_h - int(pair_bottom["group_h"]))

    top_frames = [
        _copy_rect(pair_top["top"], r=left_r + int(pair_top["top"]["r"]), c=left_c + int(pair_top["top"]["c"])),
        _copy_rect(
            pair_bottom["top"],
            r=right_r + int(pair_bottom["top"]["r"]),
            c=right_c + int(pair_bottom["top"]["c"]),
        ),
    ]
    bottom_frames = [
        _copy_rect(
            pair_top["bottom"],
            r=left_r + int(pair_top["bottom"]["r"]),
            c=left_c + int(pair_top["bottom"]["c"]),
        ),
        _copy_rect(
            pair_bottom["bottom"],
            r=right_r + int(pair_bottom["bottom"]["r"]),
            c=right_c + int(pair_bottom["bottom"]["c"]),
        ),
    ]

    if any(_rects_too_close(top_frames[i], top_frames[j], pad=1) for i in range(2) for j in range(i + 1, 2)):
        return None
    if any(_rects_too_close(bottom_frames[i], bottom_frames[j], pad=1) for i in range(2) for j in range(i + 1, 2)):
        return None

    if top_target == 3:
        extra_top = _place_extra_frame(panel_h, panel_w, top_frames, bottom_frames, used_areas)
        if extra_top is None:
            return None
        used_areas.add(int(extra_top["area"]))
        top_frames.append(extra_top)

    if bottom_target == 3:
        extra_bottom = _place_extra_frame(panel_h, panel_w, bottom_frames, top_frames, used_areas)
        if extra_bottom is None:
            return None
        used_areas.add(int(extra_bottom["area"]))
        bottom_frames.append(extra_bottom)

    grid = _render_input(panel_h, panel_w, top_frames, bottom_frames)
    paired_rects = [
        (top_frames[0], bottom_frames[0], "top"),
        (top_frames[1], bottom_frames[1], "bottom"),
    ]
    if not _validate_generated(grid, top_target, bottom_target, paired_rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
