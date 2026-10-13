from __future__ import annotations

import random

import numpy as np


BLACK = 0
SIDES = ("top", "bottom", "left", "right")
DELTA_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GREY = 5


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _most_common_non_black(grid: np.ndarray) -> int | None:
    vals, counts = np.unique(grid, return_counts=True)
    pairs = [(int(count), int(val)) for val, count in zip(vals, counts) if int(val) != BLACK]
    if not pairs:
        return None
    pairs.sort(reverse=True)
    return pairs[0][1]


def _side_cells(r0: int, c0: int, r1: int, c1: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _contiguous(indices: list[int]) -> bool:
    return all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1))


def _gap_cells(spec: dict) -> list[tuple[int, int]]:
    r0 = spec["r0"]
    c0 = spec["c0"]
    h = spec["h"]
    w = spec["w"]
    start = spec["gap_start"]
    width = spec["gap_width"]
    side = spec["gap_side"]
    if side == "top":
        return [(r0, c0 + start + i) for i in range(width)]
    if side == "bottom":
        return [(r0 + h - 1, c0 + start + i) for i in range(width)]
    if side == "left":
        return [(r0 + start + i, c0) for i in range(width)]
    return [(r0 + start + i, c0 + w - 1) for i in range(width)]


def _extract_frames(grid: np.ndarray) -> list[dict]:
    grey = _most_common_non_black(grid)
    assert grey is not None, "expected at least one non-black color"

    frames: list[dict] = []
    for comp in _components4(grid == grey):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior == grey):
            continue

        gap_side = None
        gap_cells = None
        valid = True
        for side in SIDES:
            cells = _side_cells(r0, c0, r1, c1, side)
            vals = [int(grid[r, c]) for r, c in cells]
            missing = [i for i, v in enumerate(vals) if v != grey]
            if not missing:
                continue
            if (
                gap_side is not None
                or len(missing) > 3
                or 0 in missing
                or len(cells) - 1 in missing
                or not _contiguous(missing)
                or any(vals[i] != BLACK for i in missing)
                or any(vals[i] != grey for i in range(len(vals)) if i not in missing)
            ):
                valid = False
                break
            gap_side = side
            gap_cells = [cells[i] for i in missing]

        if not valid or gap_side is None or gap_cells is None:
            continue

        perimeter = set(_side_cells(r0, c0, r1, c1, "top"))
        perimeter |= set(_side_cells(r0, c0, r1, c1, "bottom"))
        perimeter |= set(_side_cells(r0, c0, r1, c1, "left"))
        perimeter |= set(_side_cells(r0, c0, r1, c1, "right"))
        if set(comp) != perimeter.difference(gap_cells):
            continue

        seed_mask = (interior != BLACK) & (interior != grey)
        if not np.any(seed_mask):
            continue
        seed_colors = np.unique(interior[seed_mask])
        if seed_colors.size != 1:
            continue
        seed_color = int(seed_colors[0])
        seed_local = np.argwhere(seed_mask)
        sr0, sc0 = seed_local.min(axis=0)
        sr1, sc1 = seed_local.max(axis=0)
        sh = int(sr1 - sr0 + 1)
        sw = int(sc1 - sc0 + 1)
        area = int(seed_local.shape[0])
        if sh != sw or sh not in (1, 2) or area != sh * sw:
            continue

        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "gap_side": gap_side,
                "gap_cells": gap_cells,
                "seed_color": seed_color,
                "seed_area": area,
            }
        )

    frames.sort(key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))
    return frames


def _paint_frame_branch(out: np.ndarray, frame: dict) -> None:
    dr, dc = DELTA_BY_SIDE[frame["gap_side"]]
    length = frame["seed_area"]
    color = frame["seed_color"]
    h, w = out.shape
    for start_r, start_c in frame["gap_cells"]:
        r = start_r
        c = start_c
        for _ in range(length):
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(out[r, c]) != BLACK:
                break
            out[r, c] = color
            r += dr
            c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored outward branches from each fence opening using the enclosed seed block as the branch template.

    Input structure:
    - Black is the background.
    - Several grey rectangular fences appear as 1-pixel-thick borders; each fence has one contiguous gap on exactly one side.
    - Inside each fence there is exactly one solid non-grey, non-black seed square of size 1x1 or 2x2, and the remaining interior cells are black.

    Rule:
    - Detect each grey fence, its bounding rectangle, and the side where the border gap appears.
    - For every gap cell, draw a straight ray starting at that black gap cell and extending outward, away from the fence interior.
    - Use the seed color for the ray, and use the seed area as the ray length.
    - Stop a ray early if it would enter a non-black cell; keep all original fences and seed squares unchanged.

    Key insight:
    - The gap side gives the direction, the gap width gives the number of parallel rays, and the seed square supplies both the color and the branch length.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    assert frames, "expected at least one valid gapped fence"

    out = g.copy()
    for frame in frames:
        _paint_frame_branch(out, frame)
    return out


def _make_spec(rows: int, cols: int, color: int, *, forced: dict | None = None) -> dict | None:
    if forced is None:
        h = random.randint(5, min(9, rows - 2))
        w = random.randint(5, min(9, cols - 2))
        side = random.choice(SIDES)
        gap_limit = h - 2 if side in ("left", "right") else w - 2
        gap_width = random.randint(1, min(3, gap_limit))
        gap_start = random.randint(1, gap_limit - gap_width + 1)
        seed_size = random.choice((1, 2))
        r0 = random.randint(1, rows - h - 1)
        c0 = random.randint(1, cols - w - 1)
    else:
        h = forced["h"]
        w = forced["w"]
        side = forced["gap_side"]
        gap_width = forced["gap_width"]
        gap_start = forced["gap_start"]
        seed_size = forced["seed_size"]
        r0 = forced["r0"]
        c0 = forced["c0"]

    inner_h = h - 2
    inner_w = w - 2
    if seed_size > inner_h or seed_size > inner_w:
        return None
    seed_r = random.randint(r0 + 1, r0 + h - 1 - seed_size)
    seed_c = random.randint(c0 + 1, c0 + w - 1 - seed_size)
    return {
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "gap_side": side,
        "gap_width": gap_width,
        "gap_start": gap_start,
        "seed_color": color,
        "seed_size": seed_size,
        "seed_r": seed_r,
        "seed_c": seed_c,
    }


def _bbox(spec: dict) -> tuple[int, int, int, int]:
    return spec["r0"], spec["c0"], spec["r0"] + spec["h"] - 1, spec["c0"] + spec["w"] - 1


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _render_spec(grid: np.ndarray, spec: dict) -> None:
    r0 = spec["r0"]
    c0 = spec["c0"]
    h = spec["h"]
    w = spec["w"]
    seed_size = spec["seed_size"]
    seed_color = spec["seed_color"]

    grid[r0, c0 : c0 + w] = GREY
    grid[r0 + h - 1, c0 : c0 + w] = GREY
    grid[r0 : r0 + h, c0] = GREY
    grid[r0 : r0 + h, c0 + w - 1] = GREY
    for r, c in _gap_cells(spec):
        grid[r, c] = BLACK
    grid[spec["seed_r"] : spec["seed_r"] + seed_size, spec["seed_c"] : spec["seed_c"] + seed_size] = seed_color


def _spec_signature(spec: dict) -> tuple:
    return (
        spec["r0"],
        spec["c0"],
        spec["h"],
        spec["w"],
        spec["gap_side"],
        tuple(_gap_cells(spec)),
        spec["seed_color"],
        spec["seed_size"] * spec["seed_size"],
    )


def _apply_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in sorted(specs, key=lambda item: (item["r0"], item["c0"], item["h"], item["w"])):
        frame = {
            "gap_side": spec["gap_side"],
            "gap_cells": _gap_cells(spec),
            "seed_color": spec["seed_color"],
            "seed_area": spec["seed_size"] * spec["seed_size"],
        }
        _paint_frame_branch(out, frame)
    return out


def _is_nonblack_blocked(grid: np.ndarray, spec: dict) -> bool:
    dr, dc = DELTA_BY_SIDE[spec["gap_side"]]
    full_len = spec["seed_size"] * spec["seed_size"]
    for r0, c0 in _gap_cells(spec):
        r = r0
        c = c0
        used = 0
        while used < full_len and 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) == BLACK:
            used += 1
            r += dr
            c += dc
        if used < full_len and 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) != BLACK:
            return True
    return False


def _place_blocked_pair(rows: int, cols: int, color_a: int, color_b: int) -> tuple[dict, dict] | None:
    for _ in range(400):
        side = random.choice(SIDES)
        sh = random.randint(5, 8)
        sw = random.randint(5, 8)
        bh = random.randint(5, 8)
        bw = random.randint(5, 8)
        gap_width = random.randint(1, min(3, (sh - 2) if side in ("left", "right") else (sw - 2)))
        gap_start = random.randint(1, ((sh - 2) if side in ("left", "right") else (sw - 2)) - gap_width + 1)
        dist = random.randint(1, 3)

        if side == "right":
            max_c0 = cols - sw - bw - dist
            if max_c0 < 1 or rows - sh - 1 < 1 or rows - bh - 1 < 1:
                continue
            sr0 = random.randint(1, rows - sh - 1)
            sc0 = random.randint(1, max_c0)
            gap_r0 = sr0 + gap_start
            gap_r1 = gap_r0 + gap_width - 1
            br0_low = max(1, gap_r1 - bh + 1)
            br0_high = min(rows - bh - 1, gap_r0)
            if br0_low > br0_high:
                continue
            br0 = random.randint(br0_low, br0_high)
            bc0 = sc0 + sw - 1 + dist
        elif side == "left":
            min_c0 = bw + dist
            if cols - sw - 1 < min_c0 or rows - sh - 1 < 1 or rows - bh - 1 < 1:
                continue
            sr0 = random.randint(1, rows - sh - 1)
            sc0 = random.randint(min_c0, cols - sw - 1)
            gap_r0 = sr0 + gap_start
            gap_r1 = gap_r0 + gap_width - 1
            br0_low = max(1, gap_r1 - bh + 1)
            br0_high = min(rows - bh - 1, gap_r0)
            if br0_low > br0_high:
                continue
            br0 = random.randint(br0_low, br0_high)
            bc0 = sc0 - dist - bw + 1
        elif side == "bottom":
            max_r0 = rows - sh - bh - dist
            if max_r0 < 1 or cols - sw - 1 < 1 or cols - bw - 1 < 1:
                continue
            sr0 = random.randint(1, max_r0)
            sc0 = random.randint(1, cols - sw - 1)
            gap_c0 = sc0 + gap_start
            gap_c1 = gap_c0 + gap_width - 1
            bc0_low = max(1, gap_c1 - bw + 1)
            bc0_high = min(cols - bw - 1, gap_c0)
            if bc0_low > bc0_high:
                continue
            bc0 = random.randint(bc0_low, bc0_high)
            br0 = sr0 + sh - 1 + dist
        else:
            min_r0 = bh + dist
            if rows - sh - 1 < min_r0 or cols - sw - 1 < 1 or cols - bw - 1 < 1:
                continue
            sr0 = random.randint(min_r0, rows - sh - 1)
            sc0 = random.randint(1, cols - sw - 1)
            gap_c0 = sc0 + gap_start
            gap_c1 = gap_c0 + gap_width - 1
            bc0_low = max(1, gap_c1 - bw + 1)
            bc0_high = min(cols - bw - 1, gap_c0)
            if bc0_low > bc0_high:
                continue
            bc0 = random.randint(bc0_low, bc0_high)
            br0 = sr0 - dist - bh + 1

        source = _make_spec(
            rows,
            cols,
            color_a,
            forced={
                "r0": sr0,
                "c0": sc0,
                "h": sh,
                "w": sw,
                "gap_side": side,
                "gap_width": gap_width,
                "gap_start": gap_start,
                "seed_size": 2,
            },
        )
        blocker_side_choices = [name for name in SIDES if name != OPPOSITE[side]]
        blocker = _make_spec(
            rows,
            cols,
            color_b,
            forced={
                "r0": br0,
                "c0": bc0,
                "h": bh,
                "w": bw,
                "gap_side": random.choice(blocker_side_choices),
                "gap_width": random.randint(1, min(3, (bh - 2) if blocker_side_choices[0] in ("left", "right") else (bw - 2))),
                "gap_start": 1,
                "seed_size": random.choice((1, 2)),
            },
        )
        if source is None or blocker is None:
            continue

        blocker_limit = bh - 2 if blocker["gap_side"] in ("left", "right") else bw - 2
        blocker["gap_width"] = random.randint(1, min(3, blocker_limit))
        blocker["gap_start"] = random.randint(1, blocker_limit - blocker["gap_width"] + 1)

        if not _boxes_separate(_bbox(source), _bbox(blocker), 0):
            continue

        test = np.zeros((rows, cols), dtype=int)
        _render_spec(test, source)
        _render_spec(test, blocker)
        if _is_nonblack_blocked(test, source):
            return source, blocker
    return None


def _place_random_spec(rows: int, cols: int, specs: list[dict], color: int) -> dict | None:
    used_boxes = [_bbox(spec) for spec in specs]
    for _ in range(400):
        spec = _make_spec(rows, cols, color)
        if spec is None:
            continue
        box = _bbox(spec)
        if all(_boxes_separate(box, other, 1) for other in used_boxes):
            return spec
    return None


def _valid_generated(grid: np.ndarray, specs: list[dict], need_block: bool, blocked_source: dict | None) -> bool:
    parsed = _extract_frames(grid)
    if len(parsed) != len(specs):
        return False

    want = sorted(_spec_signature(spec) for spec in specs)
    got = sorted(
        (
            frame["r0"],
            frame["c0"],
            frame["r1"] - frame["r0"] + 1,
            frame["c1"] - frame["c0"] + 1,
            frame["gap_side"],
            tuple(frame["gap_cells"]),
            frame["seed_color"],
            frame["seed_area"],
        )
        for frame in parsed
    )
    if want != got:
        return False

    expected = _apply_specs(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if need_block and (blocked_source is None or not _is_nonblack_blocked(grid, blocked_source)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        rows = random.randint(16, 30)
        cols = random.randint(16, 30)
        num_fences = random.choice((1, 2, 2, 3, 3, 4))
        need_block = num_fences >= 2 and random.random() < 0.5

        colors = random.sample(SEED_COLORS, num_fences)
        specs: list[dict] = []
        blocked_source = None
        color_index = 0

        if need_block:
            pair = _place_blocked_pair(rows, cols, colors[0], colors[1])
            if pair is None:
                continue
            blocked_source, blocker = pair
            specs.extend([blocked_source, blocker])
            color_index = 2

        ok = True
        while color_index < num_fences:
            spec = _place_random_spec(rows, cols, specs, colors[color_index])
            if spec is None:
                ok = False
                break
            specs.append(spec)
            color_index += 1
        if not ok:
            continue

        grid = np.zeros((rows, cols), dtype=int)
        for spec in specs:
            _render_spec(grid, spec)
        if _valid_generated(grid, specs, need_block, blocked_source):
            return grid

    raise RuntimeError("failed to generate a valid ARC input")
