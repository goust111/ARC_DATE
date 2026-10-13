from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
DIRECTIONS = ("up", "down", "left", "right")


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _mirror_stamp(stamp: np.ndarray, direction: str, copy_index: int) -> np.ndarray:
    if copy_index % 2 == 0:
        return stamp
    return np.fliplr(stamp) if direction in ("up", "down") else np.flipud(stamp)


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = out.shape
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = int(color)


def _extract_container(grid: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _perimeter_mask(h, w)
    missing = np.argwhere(border & (local != int(color)))
    if missing.shape[0] != 1:
        return None
    gap_r, gap_c = map(int, missing[0])
    if local[gap_r, gap_c] != BG:
        return None
    if (gap_r in (0, h - 1)) and (gap_c in (0, w - 1)):
        return None
    if int(np.count_nonzero(border & (local == int(color)))) != int(border.sum()) - 1:
        return None

    if gap_r == 0:
        direction = "up"
    elif gap_r == h - 1:
        direction = "down"
    elif gap_c == 0:
        direction = "left"
    elif gap_c == w - 1:
        direction = "right"
    else:
        return None

    inner = local[1:-1, 1:-1]
    if inner.size == 0:
        return None
    inner_colors = set(map(int, np.unique(inner)))
    if not inner_colors.issubset({BG, YELLOW, int(color)}):
        return None

    stamp_full = inner == int(color)
    if not stamp_full.any() or not _is_connected(stamp_full):
        return None
    stamp = _tight_mask(stamp_full)
    markers = int(np.count_nonzero(inner == YELLOW))

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "gap": (r0 + gap_r, c0 + gap_c),
        "stamp": stamp,
        "markers": markers,
    }


def _apply_container_copies(out: np.ndarray, container: dict) -> None:
    color = int(container["color"])
    r0, c0, r1, c1 = map(int, container["bbox"])
    gap_r, gap_c = map(int, container["gap"])
    stamp = np.asarray(container["stamp"], dtype=bool)
    count = int(container["markers"])
    direction = str(container["direction"])
    sh, sw = stamp.shape

    for idx in range(count):
        cur = _mirror_stamp(stamp, direction, idx)
        if direction == "up":
            top = r0 - sh - idx * (sh + 1)
            left = gap_c - sw // 2
        elif direction == "down":
            top = r1 + 1 + idx * (sh + 1)
            left = gap_c - sw // 2
        elif direction == "left":
            top = gap_r - sh // 2
            left = c0 - sw - idx * (sw + 1)
        else:
            top = gap_r - sh // 2
            left = c1 + 1 + idx * (sw + 1)
        _paint_mask(out, cur, top, left, color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each inner stamp outside its gapped frame once per yellow marker.

    Input structure:
    - bg = black.
    - The grid contains several one-pixel-thick colored rectangular frames, each with exactly one single-cell gap on one side.
    - Inside each frame there is one same-color connected stamp and several yellow single-pixel markers.

    Rule:
    - For each frame, find the side containing the gap to determine the outward copy direction.
    - Count the yellow markers inside that frame to determine how many copies to place.
    - Extract the same-color interior stamp, place the first copy directly outside the gap, leave one blank cell between later copies, and mirror every even-numbered copy across the axis perpendicular to the travel direction.
    - Keep the original frames, stamps, and markers unchanged, and clip copied pixels that leave the grid.

    Key insight:
    - The border color also identifies the stamp, while the gap side and the yellow-marker count independently encode direction and repetition count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for color in sorted(set(map(int, np.unique(g))) - {BG, YELLOW}):
        container = _extract_container(g, color)
        if container is not None:
            _apply_container_copies(out, container)
    return out


def _make_stamp(direction: str) -> np.ndarray | None:
    for _ in range(200):
        if direction in ("up", "down"):
            h = random.choice([2, 3, 4])
            w = random.choice([3, 5])
            mirror_axis = "lr"
        else:
            h = random.choice([3, 5])
            w = random.choice([2, 3, 4])
            mirror_axis = "ud"

        area = h * w
        target = random.randint(max(3, area // 2), area - 1)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _is_connected(mask):
            continue
        if mirror_axis == "lr" and np.array_equal(mask, np.fliplr(mask)):
            continue
        if mirror_axis == "ud" and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _make_container_spec(color: int) -> dict | None:
    direction = random.choice(DIRECTIONS)
    stamp = _make_stamp(direction)
    if stamp is None:
        return None

    sh, sw = stamp.shape
    inner_h = random.randint(sh + 2, sh + 4)
    inner_w = random.randint(sw + 2, sw + 4)
    stamp_top = random.randint(1, inner_h - sh - 1)
    stamp_left = random.randint(1, inner_w - sw - 1)

    free_cells = inner_h * inner_w - int(stamp.sum())
    if free_cells < 1:
        return None
    markers = random.choice([1, 2, 3])
    if markers > free_cells:
        markers = free_cells

    frame_h = inner_h + 2
    frame_w = inner_w + 2

    if direction == "up":
        gap_rel = (0, 1 + stamp_left + sw // 2)
        copy_len = markers * sh + (markers - 1)
        used_r0, used_r1 = -copy_len, frame_h - 1
        used_c0, used_c1 = 0, frame_w - 1
    elif direction == "down":
        gap_rel = (frame_h - 1, 1 + stamp_left + sw // 2)
        copy_len = markers * sh + (markers - 1)
        used_r0, used_r1 = 0, frame_h + copy_len - 1
        used_c0, used_c1 = 0, frame_w - 1
    elif direction == "left":
        gap_rel = (1 + stamp_top + sh // 2, 0)
        copy_len = markers * sw + (markers - 1)
        used_r0, used_r1 = 0, frame_h - 1
        used_c0, used_c1 = -copy_len, frame_w - 1
    else:
        gap_rel = (1 + stamp_top + sh // 2, frame_w - 1)
        copy_len = markers * sw + (markers - 1)
        used_r0, used_r1 = 0, frame_h - 1
        used_c0, used_c1 = 0, frame_w + copy_len - 1

    marker_candidates = [
        (1 + r, 1 + c)
        for r in range(inner_h)
        for c in range(inner_w)
        if not (
            stamp_top <= r < stamp_top + sh
            and stamp_left <= c < stamp_left + sw
            and stamp[r - stamp_top, c - stamp_left]
        )
    ]
    if len(marker_candidates) < markers:
        return None

    return {
        "color": int(color),
        "direction": direction,
        "stamp": stamp,
        "markers": int(markers),
        "frame_h": int(frame_h),
        "frame_w": int(frame_w),
        "stamp_top": int(stamp_top),
        "stamp_left": int(stamp_left),
        "gap_rel": tuple(map(int, gap_rel)),
        "used_r0": int(used_r0),
        "used_r1": int(used_r1),
        "used_c0": int(used_c0),
        "used_c1": int(used_c1),
        "marker_rel": random.sample(marker_candidates, markers),
    }


def _place_specs(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    reserved = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for spec in specs:
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        used_r0 = int(spec["used_r0"])
        used_r1 = int(spec["used_r1"])
        used_c0 = int(spec["used_c0"])
        used_c1 = int(spec["used_c1"])

        min_top = max(0, -used_r0)
        max_top = min(h - frame_h, h - 1 - used_r1)
        min_left = max(0, -used_c0)
        max_left = min(w - frame_w, w - 1 - used_c1)
        if min_top > max_top or min_left > max_left:
            return None

        success = False
        for _ in range(120):
            top = random.randint(min_top, max_top)
            left = random.randint(min_left, max_left)
            rr0 = max(0, top + used_r0 - 1)
            rr1 = min(h - 1, top + used_r1 + 1)
            cc0 = max(0, left + used_c0 - 1)
            cc1 = min(w - 1, left + used_c1 + 1)
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed.append(
                {
                    **spec,
                    "top": int(top),
                    "left": int(left),
                    "bbox": (int(top), int(left), int(top + frame_h - 1), int(left + frame_w - 1)),
                    "gap": (int(top + spec["gap_rel"][0]), int(left + spec["gap_rel"][1])),
                }
            )
            success = True
            break
        if not success:
            return None

    return placed


def _render_input(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        stamp = np.asarray(spec["stamp"], dtype=bool)

        grid[top, left : left + frame_w] = color
        grid[top + frame_h - 1, left : left + frame_w] = color
        grid[top : top + frame_h, left] = color
        grid[top : top + frame_h, left + frame_w - 1] = color

        gap_r, gap_c = map(int, spec["gap"])
        grid[gap_r, gap_c] = BG

        _paint_mask(
            grid,
            stamp,
            top + 1 + int(spec["stamp_top"]),
            left + 1 + int(spec["stamp_left"]),
            color,
        )
        for rr, cc in spec["marker_rel"]:
            grid[top + int(rr), left + int(cc)] = YELLOW
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        _apply_container_copies(out, spec)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)

    n_objects = random.choice([1, 2, 2, 3])
    if min(h, w) < 20:
        n_objects = min(n_objects, 2)

    colors = random.sample(PALETTE, n_objects)
    specs = []
    for color in colors:
        spec = _make_container_spec(color)
        if spec is None:
            return None
        specs.append(spec)

    if all(int(spec["markers"]) == 1 for spec in specs):
        return None

    placed = _place_specs(h, w, specs)
    if placed is None:
        return None

    inp = _render_input(h, w, placed)
    expected = _expected_output(inp, placed)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
