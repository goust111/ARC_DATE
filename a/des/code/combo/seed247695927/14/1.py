from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


YELLOW = 4
BLUE = 1
RED = 2
GREEN = 3

GRID_H_MIN = 16
GRID_H_MAX = 24
GRID_W_MIN = 16
GRID_W_MAX = 26

OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 5
OBJECT_H_MIN = 3
OBJECT_H_MAX = 6
OBJECT_W_MIN = 3
OBJECT_W_MAX = 9
MIN_ROW_GAP = 1
MAX_FOREGROUND_RATIO = 0.42
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_contiguous(mask: np.ndarray) -> bool:
    comps = _find_components(mask)
    return len(comps) == 1 and int(comps[0].shape[0]) == int(mask.sum())


def _has_flanked_boundary_gap(mask: np.ndarray) -> bool:
    h, w = mask.shape

    for c in range(w):
        if not mask[0, c] and mask[0, :c].any() and mask[0, c + 1 :].any():
            return True
        if not mask[h - 1, c] and mask[h - 1, :c].any() and mask[h - 1, c + 1 :].any():
            return True

    for r in range(h):
        if not mask[r, 0] and mask[:r, 0].any() and mask[r + 1 :, 0].any():
            return True
        if not mask[r, w - 1] and mask[:r, w - 1].any() and mask[r + 1 :, w - 1].any():
            return True

    return False


def _place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    components = _find_components(g == fg)
    assert components, "expected at least one foreground object"

    out = np.full_like(g, bg)
    width = g.shape[1]

    for coords in components:
        bbox = _bbox_from_coords(coords)
        r0, _c0, r1, _c1 = bbox
        mask = _coords_to_mask(coords, bbox)
        target_c = width - mask.shape[1] if _has_flanked_boundary_gap(mask) else 0
        color = GREEN if target_c > 0 else RED

        dst = out[r0 : r1 + 1, target_c : target_c + mask.shape[1]]
        assert np.all(dst[mask] == bg), "moved objects must not overlap"
        dst[mask] = color

    return out


def _random_heights(count: int, grid_h: int) -> list[int] | None:
    for _ in range(80):
        hs = [random.randint(OBJECT_H_MIN, OBJECT_H_MAX) for _ in range(count)]
        if sum(hs) + MIN_ROW_GAP * (count - 1) <= grid_h:
            return hs
    return None


def _row_starts(heights: list[int], grid_h: int) -> list[int]:
    extra = grid_h - sum(heights) - MIN_ROW_GAP * (len(heights) - 1)
    gaps = [0] * (len(heights) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    starts = []
    r = gaps[0]
    for i, h in enumerate(heights):
        starts.append(r)
        r += h
        if i + 1 < len(heights):
            r += MIN_ROW_GAP + gaps[i + 1]
    return starts


def _make_notch_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge in {"top", "bottom"} and w >= 3:
            run = random.randint(1, w - 2)
            start = random.randint(1, w - run - 1)
            depth = random.randint(1, h - 1)
            if edge == "top":
                mask[:depth, start : start + run] = False
            else:
                mask[h - depth :, start : start + run] = False
        elif edge in {"left", "right"} and h >= 3:
            run = random.randint(1, h - 2)
            start = random.randint(1, h - run - 1)
            depth = random.randint(1, w - 1)
            if edge == "left":
                mask[start : start + run, :depth] = False
            else:
                mask[start : start + run, w - depth :] = False
        else:
            continue

        if random.random() < 0.3 and h >= 5 and w >= 5:
            rr = random.randint(1, h - 3)
            cc = random.randint(1, w - 3)
            mask[rr : rr + 2, cc : cc + 2] = False

        if _has_flanked_boundary_gap(mask) and _is_contiguous(mask):
            return mask
    return None


def _make_closed_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        style = random.choice(["solid", "hole", "barbell"])
        if style == "hole" and h >= 5 and w >= 5:
            ih = random.randint(1, h - 4)
            iw = random.randint(1, w - 4)
            r0 = random.randint(1, h - ih - 1)
            c0 = random.randint(1, w - iw - 1)
            mask[r0 : r0 + ih, c0 : c0 + iw] = False
        elif style == "barbell" and h >= 4 and w >= 4:
            if random.random() < 0.5:
                mask[1 : h - 1, 1 : w - 1] = False
                col = random.randint(1, w - 2)
                mask[:, col] = True
            else:
                mask[1 : h - 1, 1 : w - 1] = False
                row = random.randint(1, h - 2)
                mask[row, :] = True

        if not _has_flanked_boundary_gap(mask) and _is_contiguous(mask):
            return mask
    return None


def _expected_output(objects: list[dict], h: int, w: int) -> np.ndarray:
    out = np.full((h, w), YELLOW, dtype=int)
    for obj in objects:
        dst_c = w - obj["mask"].shape[1] if obj["kind"] == "notch" else 0
        color = GREEN if obj["kind"] == "notch" else RED
        _place_mask(out, obj["mask"], obj["r0"], dst_c, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_H_MIN, GRID_H_MAX)
    w = random.randint(GRID_W_MIN, GRID_W_MAX)
    count = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    heights = _random_heights(count, h)
    if heights is None:
        return None
    starts = _row_starts(heights, h)

    right_count = random.randint(1, count - 1)
    kinds = ["notch"] * right_count + ["closed"] * (count - right_count)
    random.shuffle(kinds)

    max_obj_w = min(OBJECT_W_MAX, w - 2)
    if max_obj_w < OBJECT_W_MIN:
        return None

    objects = []
    total_fg = 0
    for kind, r0, oh in zip(kinds, starts, heights):
        ow = random.randint(OBJECT_W_MIN, max_obj_w)
        if kind == "notch":
            if oh < 3 or ow < 3:
                return None
            mask = _make_notch_mask(oh, ow)
        else:
            mask = _make_closed_mask(oh, ow)
        if mask is None:
            return None

        total_fg += int(mask.sum())
        c0 = random.randint(1, w - ow - 1)
        objects.append({"kind": kind, "mask": mask, "r0": r0, "c0": c0})

    if total_fg > int(h * w * MAX_FOREGROUND_RATIO):
        return None

    inp = np.full((h, w), YELLOW, dtype=int)
    for obj in objects:
        _place_mask(inp, obj["mask"], obj["r0"], obj["c0"], BLUE)

    if Counter(kinds)["notch"] == 0 or Counter(kinds)["closed"] == 0:
        return None

    expected = _expected_output(objects, h, w)
    try:
        got = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
