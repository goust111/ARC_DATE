from __future__ import annotations

import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals = np.bincount(grid.ravel())
    return int(np.argmax(vals))


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            box = _bbox(cells)
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": box,
                    "area": len(cells),
                }
            )
    return comps


def _is_horizontal_bar(comp: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    return (
        r0 == r1
        and comp["area"] == c1 - c0 + 1
        and (r0 == 0 or r0 == h - 1)
    )


def _is_vertical_bar(comp: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    return (
        c0 == c1
        and comp["area"] == r1 - r0 + 1
        and (c0 == 0 or c0 == w - 1)
    )


def _scale_axis(mask: np.ndarray, target: int, axis: int) -> np.ndarray:
    cur = mask.shape[axis]
    if cur == target:
        return mask.copy()
    if target > cur:
        if target % cur != 0:
            raise ValueError("non-integer enlargement")
        factor = target // cur
        return np.repeat(mask, factor, axis=axis)

    if cur % target != 0:
        raise ValueError("non-integer shrink")
    factor = cur // target
    pieces = np.split(mask, target, axis=axis)
    reduced = []
    for piece in pieces:
        ref = np.take(piece, 0, axis=axis)
        if not np.all(piece == np.expand_dims(ref, axis=axis)):
            raise ValueError("cannot shrink losslessly")
        reduced.append(ref)
    return np.stack(reduced, axis=axis)


def _scale_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    out = _scale_axis(mask, target_w, axis=1)
    out = _scale_axis(out, target_h, axis=0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resize each interior monochrome object along the axes indicated by same-colored edge ruler bars.

    Input structure:
    - Black background with one-pixel-thick colored bars attached to the top/bottom or left/right edge.
    - Interior foreground components are monochrome objects, with distinct colors from each other.
    - A color may have a horizontal bar, a vertical bar, both, or neither; bars stay disconnected from objects.

    Rule:
    - A horizontal bar sets the target width of the object with the same color; a vertical bar sets its target height.
    - If the target along an axis is larger, enlarge by repeating whole columns or rows by an integer factor.
    - If the target is smaller, shrink by merging equal-width repeated column or row blocks, keeping the object's top-left anchor fixed.
    - Objects with no same-colored bar on an axis keep that axis unchanged, and all edge bars remain in place.

    Key insight:
    - Each bar is not a multiplier token but a ruler giving the final size on its own axis, so the same mechanism handles both upscaling and downscaling.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    hbars: dict[int, int] = {}
    vbars: dict[int, int] = {}
    objects: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        if _is_horizontal_bar(comp, g.shape):
            hbars[color] = c1 - c0 + 1
        elif _is_vertical_bar(comp, g.shape):
            vbars[color] = r1 - r0 + 1
        else:
            objects.append(comp)

    out = g.copy()
    for obj in objects:
        for r, c in obj["cells"]:
            out[r, c] = bg

    for obj in objects:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        mask = g[r0 : r1 + 1, c0 : c1 + 1] == color
        target_w = hbars.get(color, mask.shape[1])
        target_h = vbars.get(color, mask.shape[0])
        scaled = _scale_mask(mask, target_h=target_h, target_w=target_w)
        patch = out[r0 : r0 + scaled.shape[0], c0 : c0 + scaled.shape[1]]
        patch[scaled] = color

    return out


GRID_SIDE_RANGE = (20, 29)
OBJECT_COUNT_CHOICES = (3, 4)
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_DIM = 9
PLACEMENT_GAP = 1
GEN_RETRIES = 500


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    q = deque([tuple(map(int, pts[0]))])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _full_bbox(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return (r0, c0, r1, c1) == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _random_core(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    attempts = random.randint(0, max(1, h * w // 3))
    for _ in range(attempts):
        filled = np.argwhere(mask)
        if len(filled) <= 2:
            break
        r, c = map(int, random.choice(filled.tolist()))
        trial = mask.copy()
        trial[r, c] = False
        if trial.sum() >= max(2, (h * w + 1) // 2) and _mask_connected(trial) and _full_bbox(trial):
            mask = trial
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.rot90(mask, random.choice((0, 2)))
    return mask.astype(bool)


def _choose_factor(base: int) -> int:
    options = [k for k in (2, 3) if base * k <= MAX_DIM]
    if not options:
        raise ValueError("no factor")
    return random.choice(options)


def _build_object_spec(hmode: str, vmode: str) -> dict | None:
    for _ in range(80):
        core_h = random.randint(2, 4)
        core_w = random.randint(2, 4)
        try:
            hfactor = _choose_factor(core_w) if hmode != "none" else 1
            vfactor = _choose_factor(core_h) if vmode != "none" else 1
        except ValueError:
            continue

        in_h = core_h * vfactor if vmode == "shrink" else core_h
        in_w = core_w * hfactor if hmode == "shrink" else core_w
        out_h = core_h * vfactor if vmode == "enlarge" else core_h
        out_w = core_w * hfactor if hmode == "enlarge" else core_w
        if max(in_h, in_w, out_h, out_w) > MAX_DIM:
            continue

        core = _random_core(core_h, core_w)
        inp = _scale_mask(core, in_h, in_w)
        out = _scale_mask(core, out_h, out_w)
        return {
            "hmode": hmode,
            "vmode": vmode,
            "core": core,
            "input_mask": inp,
            "output_mask": out,
            "bar_w": out_w if hmode != "none" else None,
            "bar_h": out_h if vmode != "none" else None,
            "room_h": max(in_h, out_h),
            "room_w": max(in_w, out_w),
        }
    return None


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_objects(specs: list[dict], h: int, w: int) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    for spec in specs:
        ok = False
        max_r = h - spec["room_h"] - 2
        max_c = w - spec["room_w"] - 2
        if max_r < 2 or max_c < 2:
            return False
        for _ in range(200):
            r0 = random.randint(2, max_r)
            c0 = random.randint(2, max_c)
            rect = (r0, c0, r0 + spec["room_h"] - 1, c0 + spec["room_w"] - 1)
            if any(_rect_overlaps(rect, prev, gap=PLACEMENT_GAP) for prev in placed):
                continue
            spec["pos"] = (r0, c0)
            placed.append(rect)
            ok = True
            break
        if not ok:
            return False
    return True


def _free_segment(used: np.ndarray, length: int) -> int | None:
    starts = list(range(1, len(used) - length))
    random.shuffle(starts)
    for start in starts:
        end = start + length
        lo = max(0, start - 1)
        hi = min(len(used), end + 1)
        if not used[lo:hi].any():
            used[start:end] = True
            return start
    return None


def _place_bars(specs: list[dict], h: int, w: int) -> bool:
    top_used = np.zeros(w, dtype=bool)
    bot_used = np.zeros(w, dtype=bool)
    left_used = np.zeros(h, dtype=bool)
    right_used = np.zeros(h, dtype=bool)

    for spec in specs:
        if spec["bar_w"] is not None:
            sides = [("top", top_used, 0), ("bottom", bot_used, h - 1)]
            random.shuffle(sides)
            placed = False
            for name, used, row in sides:
                start = _free_segment(used, spec["bar_w"])
                if start is not None:
                    spec["hbar"] = (row, start, spec["bar_w"])
                    placed = True
                    break
            if not placed:
                return False
        if spec["bar_h"] is not None:
            sides = [("left", left_used, 0), ("right", right_used, w - 1)]
            random.shuffle(sides)
            placed = False
            for name, used, col in sides:
                start = _free_segment(used, spec["bar_h"])
                if start is not None:
                    spec["vbar"] = (start, col, spec["bar_h"])
                    placed = True
                    break
            if not placed:
                return False
    return True


def _render(specs: list[dict], h: int, w: int, which: str) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        color = spec["color"]
        if "hbar" in spec:
            row, start, length = spec["hbar"]
            grid[row, start : start + length] = color
        if "vbar" in spec:
            start, col, length = spec["vbar"]
            grid[start : start + length, col] = color

    key = "input_mask" if which == "input" else "output_mask"
    for spec in specs:
        r0, c0 = spec["pos"]
        mask = spec[key]
        grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = spec["color"]
    return grid


def _make_specs(count: int) -> list[dict] | None:
    base_modes = [
        (random.choice(("enlarge", "shrink")), "none"),
        ("none", random.choice(("enlarge", "shrink"))),
        (random.choice(("enlarge", "shrink")), random.choice(("enlarge", "shrink"))),
    ]
    if len({base_modes[0][0], base_modes[1][1]}) == 1:
        first = base_modes[0][0]
        other = "shrink" if first == "enlarge" else "enlarge"
        base_modes[1] = ("none", other)

    extra_choices = [
        ("none", "none"),
        ("enlarge", "none"),
        ("shrink", "none"),
        ("none", "enlarge"),
        ("none", "shrink"),
        ("enlarge", "shrink"),
        ("shrink", "enlarge"),
    ]
    while len(base_modes) < count:
        base_modes.append(random.choice(extra_choices))

    specs: list[dict] = []
    for hmode, vmode in base_modes:
        spec = _build_object_spec(hmode, vmode)
        if spec is None:
            return None
        specs.append(spec)
    colors = random.sample(COLORS, count)
    for spec, color in zip(specs, colors):
        spec["color"] = color
    return specs


def _attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    count = random.choice(OBJECT_COUNT_CHOICES)
    specs = _make_specs(count)
    if specs is None:
        return None
    if not _place_objects(specs, h, w):
        return None
    if not _place_bars(specs, h, w):
        return None

    inp = _render(specs, h, w, "input")
    expected = _render(specs, h, w, "output")
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
