from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400
RESERVE_MARGIN = 1
_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _gap_side(
    gap_coord: tuple[int, int], bbox: tuple[int, int, int, int]
) -> str | None:
    r, c = gap_coord
    r0, c0, r1, c1 = bbox
    matches: list[str] = []
    if r == r0:
        matches.append("up")
    if r == r1:
        matches.append("down")
    if c == c0:
        matches.append("left")
    if c == c1:
        matches.append("right")
    return matches[0] if len(matches) == 1 else None


def _find_frame_objects(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects: list[dict] = []

    for comp in _components(g != int(bg)):
        r0, c0, r1, c1 = _bbox(comp)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 3 or width < 3:
            continue

        occ = np.zeros((height, width), dtype=bool)
        vals: list[int] = []
        for r, c in comp:
            occ[r - r0, c - c0] = True
            vals.append(int(g[r, c]))

        border = _border_mask(height, width)
        if not np.array_equal(occ, border):
            continue

        counts = Counter(vals)
        if len(counts) != 2:
            continue
        frame_color, frame_count = counts.most_common(1)[0]
        if int(frame_count) != int(border.sum()) - 1:
            continue

        gap_colors = [color for color, count in counts.items() if int(color) != int(frame_color) and int(count) == 1]
        if len(gap_colors) != 1:
            continue
        gap_color = int(gap_colors[0])

        gap_coord = next((coord for coord in comp if int(g[coord]) == gap_color), None)
        if gap_coord is None:
            continue
        side = _gap_side(gap_coord, (r0, c0, r1, c1))
        if side is None:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern_mask = interior != int(bg)
        if not pattern_mask.any():
            continue
        if not np.all(interior[pattern_mask] == int(gap_color)):
            continue

        pattern_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr, cc in np.argwhere(pattern_mask)
        ]
        objects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "frame_color": int(frame_color),
                "gap_color": int(gap_color),
                "gap_coord": (int(gap_coord[0]), int(gap_coord[1])),
                "side": side,
                "pattern_coords": pattern_coords,
            }
        )

    objects.sort(key=lambda obj: obj["bbox"])
    return objects, int(bg)


def _translate_pattern(
    bbox: tuple[int, int, int, int],
    side: str,
    pattern_coords: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    pr0, pc0, pr1, pc1 = _bbox(pattern_coords)
    r0, c0, r1, c1 = bbox
    if side == "up":
        dr, dc = int(r0 - 1 - pr1), 0
    elif side == "down":
        dr, dc = int(r1 + 1 - pr0), 0
    elif side == "left":
        dr, dc = 0, int(c0 - 1 - pc1)
    else:
        dr, dc = 0, int(c1 + 1 - pc0)
    return [(int(r + dr), int(c + dc)) for r, c in pattern_coords]


def _apply_rule(grid: np.ndarray, objects: list[dict], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    height, width = out.shape
    moved: list[tuple[int, list[tuple[int, int]]]] = []

    for obj in objects:
        gap_r, gap_c = obj["gap_coord"]
        out[gap_r, gap_c] = int(obj["frame_color"])
        for r, c in obj["pattern_coords"]:
            out[r, c] = int(bg)
        moved.append(
            (
                int(obj["gap_color"]),
                _translate_pattern(obj["bbox"], obj["side"], obj["pattern_coords"]),
            )
        )

    for color, coords in moved:
        for r, c in coords:
            assert 0 <= r < height and 0 <= c < width, "translated pattern left the grid"
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each inner monochrome pattern out of its frame through the matching colored notch.

    Input structure:
    - Background is black.
    - Several axis-aligned rectangular frames are drawn with a single border color.
    - Each frame has exactly one border pixel recolored to the interior pattern color, creating a one-cell notch marker.
    - Inside each frame there is a separate monochrome pattern using that same notch color.

    Rule:
    - Detect every rectangular frame and identify which side contains the unique recolored border pixel.
    - Treat that recolored border pixel as both the direction marker and the color match for the interior pattern.
    - Remove the interior pattern, restore the notch cell to the frame color, and translate the whole pattern straight out through that side.
    - Stop the translation when the moved pattern sits completely outside the frame and its nearest edge is flush with the frame exterior.

    Key insight:
    The single off-color border pixel simultaneously says which interior pattern belongs to the frame and which direction that pattern must be pushed.
    """
    objects, bg = _find_frame_objects(np.asarray(grid, dtype=int))
    return _apply_rule(np.asarray(grid, dtype=int), objects, bg)


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _path_between(
    start: tuple[int, int], end: tuple[int, int], row_first: bool
) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(int(r), int(c))]
    if row_first:
        while r != er:
            r += 1 if er > r else -1
            path.append((int(r), int(c)))
        while c != ec:
            c += 1 if ec > c else -1
            path.append((int(r), int(c)))
    else:
        while c != ec:
            c += 1 if ec > c else -1
            path.append((int(r), int(c)))
        while r != er:
            r += 1 if er > r else -1
            path.append((int(r), int(c)))
    return path


def _random_pattern_mask(height: int, width: int) -> np.ndarray:
    if height == 1 and width == 1:
        return np.ones((1, 1), dtype=bool)

    anchors = [
        (0, _randint(0, width - 1)),
        (height - 1, _randint(0, width - 1)),
        (_randint(0, height - 1), 0),
        (_randint(0, height - 1), width - 1),
    ]
    ordered: list[tuple[int, int]] = []
    for idx in _RNG.permutation(len(anchors)):
        cell = anchors[int(idx)]
        if cell not in ordered:
            ordered.append(cell)

    mask = np.zeros((height, width), dtype=bool)
    cursor = ordered[0]
    mask[cursor] = True
    for anchor in ordered[1:]:
        for r, c in _path_between(cursor, anchor, bool(_randint(0, 1))):
            mask[r, c] = True
        cursor = anchor

    extra_steps = _randint(0, max(1, (height * width) // 2))
    for _ in range(extra_steps):
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier or _RNG.random() < 0.35:
            continue
        nr, nc = frontier[_randint(0, len(frontier) - 1)]
        mask[nr, nc] = True

    return mask


def _make_object_spec(direction: str) -> dict:
    frame_h = _randint(5, 9)
    frame_w = _randint(5, 9)
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    pattern_h = _randint(1, min(4, inner_h - 2))
    pattern_w = _randint(1, min(4, inner_w - 2))
    off_r = _randint(1, inner_h - pattern_h - 1)
    off_c = _randint(1, inner_w - pattern_w - 1)

    frame_color = _choice(COLORS)
    pattern_color = _choice([color for color in COLORS if color != frame_color])
    pattern_mask = _random_pattern_mask(pattern_h, pattern_w)

    if direction in ("up", "down"):
        gap_pos = _randint(1, frame_w - 2)
    else:
        gap_pos = _randint(1, frame_h - 2)

    pattern_rel_bbox = (
        int(1 + off_r),
        int(1 + off_c),
        int(1 + off_r + pattern_h - 1),
        int(1 + off_c + pattern_w - 1),
    )
    if direction == "up":
        dest_rel_bbox = (-pattern_h, int(pattern_rel_bbox[1]), -1, int(pattern_rel_bbox[3]))
    elif direction == "down":
        dest_rel_bbox = (frame_h, int(pattern_rel_bbox[1]), frame_h + pattern_h - 1, int(pattern_rel_bbox[3]))
    elif direction == "left":
        dest_rel_bbox = (int(pattern_rel_bbox[0]), -pattern_w, int(pattern_rel_bbox[2]), -1)
    else:
        dest_rel_bbox = (int(pattern_rel_bbox[0]), frame_w, int(pattern_rel_bbox[2]), frame_w + pattern_w - 1)

    union_rel_bbox = (
        min(0, int(dest_rel_bbox[0])),
        min(0, int(dest_rel_bbox[1])),
        max(frame_h - 1, int(dest_rel_bbox[2])),
        max(frame_w - 1, int(dest_rel_bbox[3])),
    )
    return {
        "direction": direction,
        "frame_h": int(frame_h),
        "frame_w": int(frame_w),
        "frame_color": int(frame_color),
        "pattern_color": int(pattern_color),
        "pattern_mask": pattern_mask,
        "pattern_h": int(pattern_h),
        "pattern_w": int(pattern_w),
        "off_r": int(off_r),
        "off_c": int(off_c),
        "gap_pos": int(gap_pos),
        "union_rel_bbox": union_rel_bbox,
    }


def _place_and_draw(specs: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    for spec in specs:
        ur0, uc0, ur1, uc1 = spec["union_rel_bbox"]
        lo_r = int(RESERVE_MARGIN - ur0)
        hi_r = int(height - 1 - RESERVE_MARGIN - ur1)
        lo_c = int(RESERVE_MARGIN - uc0)
        hi_c = int(width - 1 - RESERVE_MARGIN - uc1)
        if lo_r > hi_r or lo_c > hi_c:
            return None

        candidates = [
            (int(r0), int(c0))
            for r0 in range(lo_r, hi_r + 1)
            for c0 in range(lo_c, hi_c + 1)
        ]
        order = _RNG.permutation(len(candidates))
        chosen: tuple[int, int] | None = None
        for idx in order:
            r0, c0 = candidates[int(idx)]
            rr0 = int(r0 + ur0 - RESERVE_MARGIN)
            cc0 = int(c0 + uc0 - RESERVE_MARGIN)
            rr1 = int(r0 + ur1 + RESERVE_MARGIN)
            cc1 = int(c0 + uc1 + RESERVE_MARGIN)
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            chosen = (int(r0), int(c0))
            break
        if chosen is None:
            return None

        r0, c0 = chosen
        fh = int(spec["frame_h"])
        fw = int(spec["frame_w"])
        frame_color = int(spec["frame_color"])
        pattern_color = int(spec["pattern_color"])
        pr0 = int(r0 + 1 + spec["off_r"])
        pc0 = int(c0 + 1 + spec["off_c"])

        grid[r0, c0 : c0 + fw] = frame_color
        grid[r0 + fh - 1, c0 : c0 + fw] = frame_color
        grid[r0 : r0 + fh, c0] = frame_color
        grid[r0 : r0 + fh, c0 + fw - 1] = frame_color

        direction = spec["direction"]
        gap_pos = int(spec["gap_pos"])
        if direction == "up":
            gap = (int(r0), int(c0 + gap_pos))
        elif direction == "down":
            gap = (int(r0 + fh - 1), int(c0 + gap_pos))
        elif direction == "left":
            gap = (int(r0 + gap_pos), int(c0))
        else:
            gap = (int(r0 + gap_pos), int(c0 + fw - 1))
        grid[gap] = pattern_color

        pattern_mask = spec["pattern_mask"]
        ph = int(spec["pattern_h"])
        pw = int(spec["pattern_w"])
        patch = grid[pr0 : pr0 + ph, pc0 : pc0 + pw]
        patch[pattern_mask] = pattern_color

        placed.append(
            {
                "bbox": (int(r0), int(c0), int(r0 + fh - 1), int(c0 + fw - 1)),
                "side": direction,
                "frame_color": frame_color,
                "gap_color": pattern_color,
                "pattern_pixels": int(pattern_mask.sum()),
            }
        )

    found, bg = _find_frame_objects(grid)
    if bg != BG or len(found) != len(placed):
        return None

    observed = sorted(
        (
            obj["bbox"],
            obj["side"],
            int(obj["frame_color"]),
            int(obj["gap_color"]),
            len(obj["pattern_coords"]),
        )
        for obj in found
    )
    expected = sorted(
        (
            obj["bbox"],
            obj["side"],
            int(obj["frame_color"]),
            int(obj["gap_color"]),
            int(obj["pattern_pixels"]),
        )
        for obj in placed
    )
    if observed != expected:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_objects = int(_choice((1, 2, 3)))
        if n_objects == 1:
            height = _randint(12, 22)
            width = _randint(12, 22)
        elif n_objects == 2:
            height = _randint(18, 28)
            width = _randint(18, 28)
        else:
            height = _randint(22, 30)
            width = _randint(22, 30)

        perm = [DIRS[int(i)] for i in _RNG.permutation(len(DIRS))]
        specs = [_make_object_spec(perm[i]) for i in range(n_objects)]
        grid = _place_and_draw(specs, height, width)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
