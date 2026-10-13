from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GRAY = 5
ANCHOR_COLORS = (1, 2, 3, 4)
PATTERN_COLORS = (6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _tight_bbox_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_rect_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _extract_pattern_from_box(grid: np.ndarray, gray_box: list[tuple[int, int]], bg: int, gray: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(gray_box)
    assert r1 - r0 >= 2 and c1 - c0 >= 2
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != bg) & (inner != gray)
    assert np.any(mask), "source box must contain a non-gray pattern"
    pr0, pc0, pr1, pc1 = _tight_bbox_mask(mask)
    return inner[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _border_anchors(grid: np.ndarray, bg: int, gray: int) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    anchors: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for c in range(w):
        if grid[0, c] != bg and grid[0, c] != gray:
            seen.add((0, c))
            anchors.append((0, c, 1, 0, int(grid[0, c])))
        if grid[h - 1, c] != bg and grid[h - 1, c] != gray:
            seen.add((h - 1, c))
            anchors.append((h - 1, c, -1, 0, int(grid[h - 1, c])))
    for r in range(h):
        if (r, 0) not in seen and grid[r, 0] != bg and grid[r, 0] != gray:
            anchors.append((r, 0, 0, 1, int(grid[r, 0])))
        if (r, w - 1) not in seen and grid[r, w - 1] != bg and grid[r, w - 1] != gray:
            anchors.append((r, w - 1, 0, -1, int(grid[r, w - 1])))
    return anchors


def _slot_map(crop: np.ndarray, bg: int, gray: int) -> dict[int, tuple[tuple[int, int, int, int], list[tuple[int, int]]]]:
    h, w = crop.shape
    slots: dict[int, tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = {}
    for comp in _components(crop == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        colors = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(crop[nr, nc])
                    if v != bg and v != gray:
                        colors.add(v)
        assert len(colors) == 1, "each output slot must have exactly one colored marker"
        color = next(iter(colors))
        assert color not in slots, "slot marker colors must be unique"
        slots[color] = (_bbox(comp), comp)
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each border-linked source motif into the matching marked slot of the central tray.

    Input structure:
    - The background is dominant.
    - A large gray rectangular tray sits near the center; gray divider lines split it into several closed slots, and each slot boundary contains one colored marker pixel.
    - Away from the tray, each source consists of a gray rectangular box containing a small non-gray motif, plus a same-colored one-cell-wide line running straight to a single colored anchor on the outer border.

    Rule:
    - Follow every border anchor inward along its same-colored straight line to identify the attached gray source box.
    - Extract the tight bounding box of the non-gray motif inside that source box.
    - Inside the central tray, find the closed slot whose boundary marker has the same color, and place the extracted motif centered inside that slot without rotating it.
    - Remove everything outside the tray by returning only the tray area with its gray walls, colored slot markers, and the newly placed motifs.

    Key insight:
    - Color is a routing label, not a paint rule: one chain maps anchor color to a source motif, and a second chain maps the same color to a destination slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)

    counts = Counter(int(v) for v in g.ravel() if int(v) != bg)
    assert counts, "expected foreground structure"
    gray = max(counts, key=counts.get)

    gray_mask = g == gray
    gray_comps = _components(gray_mask)
    assert gray_comps, "expected gray rectangles"

    label = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(gray_comps):
        for r, c in comp:
            label[r, c] = idx

    central_idx = max(range(len(gray_comps)), key=lambda i: len(gray_comps[i]))
    central_box = _bbox(gray_comps[central_idx])
    cr0, cc0, cr1, cc1 = central_box

    patterns: dict[int, np.ndarray] = {}
    for ar, ac, dr, dc, color in _border_anchors(g, bg, gray):
        r, c = ar, ac
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != color:
                break
            r, c = nr, nc
        br, bc = r + dr, c + dc
        assert 0 <= br < h and 0 <= bc < w
        source_idx = int(label[br, bc])
        assert source_idx >= 0 and source_idx != central_idx, "each anchor line must end at a non-central gray box"
        assert color not in patterns, "anchor colors must be unique"
        patterns[color] = _extract_pattern_from_box(g, gray_comps[source_idx], bg, gray)

    crop = g[cr0 : cr1 + 1, cc0 : cc1 + 1].copy()
    slots = _slot_map(crop, bg, gray)
    assert patterns and patterns.keys() == slots.keys(), "anchor colors and slot colors must match exactly"

    for color, patch in patterns.items():
        (sr0, sc0, sr1, sc1), slot_cells = slots[color]
        for r, c in slot_cells:
            crop[r, c] = bg
        slot_h = sr1 - sr0 + 1
        slot_w = sc1 - sc0 + 1
        ph, pw = patch.shape
        assert ph <= slot_h and pw <= slot_w
        pr0 = sr0 + (slot_h - ph) // 2
        pc0 = sc0 + (slot_w - pw) // 2
        mask = patch != bg
        crop[pr0 : pr0 + ph, pc0 : pc0 + pw][mask] = patch[mask]

    return crop


def _connected_shape(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    max_cells = h * w
    target = 1 if max_cells == 1 else rng.randint(2, max_cells)
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    if len(cells) < target:
        return None
    patch = np.full((h, w), BG, dtype=int)
    palette = rng.sample(PATTERN_COLORS, k=rng.randint(1, 2))
    used_colors = set()
    for r, c in cells:
        color = rng.choice(palette)
        patch[r, c] = color
        used_colors.add(color)
    if len(cells) >= 3 and len(palette) == 2 and len(used_colors) == 1:
        rr, cc = rng.choice(list(cells))
        other = palette[0] if patch[rr, cc] == palette[1] else palette[1]
        patch[rr, cc] = other
    r0, c0, r1, c1 = _tight_bbox_mask(patch != BG)
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _sample_unique_patterns(inner_h: int, inner_w: int, n: int, rng: random.Random) -> list[np.ndarray] | None:
    patterns: list[np.ndarray] = []
    signatures: set[bytes] = set()
    for _ in range(n):
        for _attempt in range(100):
            ph = rng.randint(1, inner_h)
            pw = rng.randint(1, inner_w)
            patch = _connected_shape(ph, pw, rng)
            if patch is None:
                continue
            sig = patch.tobytes() + bytes(patch.shape)
            if sig in signatures:
                continue
            signatures.add(sig)
            patterns.append(patch)
            break
        else:
            return None
    return patterns


def _pick_non_overlapping_starts(start: int, span: int, size: int, count: int, rng: random.Random) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(start, start + span - size + 1))
    rng.shuffle(candidates)
    picked: list[int] = []
    for pos in candidates:
        if all(pos + size < prev or prev + size < pos for prev in picked):
            picked.append(pos)
            if len(picked) == count:
                return sorted(picked)
    return None


def _slot_bboxes(top: int, left: int, rows: int, cols: int, box_h: int, box_w: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for rr in range(rows):
        for cc in range(cols):
            r0 = top + rr * (box_h - 1) + 1
            c0 = left + cc * (box_w - 1) + 1
            boxes.append((r0, c0, r0 + box_h - 3, c0 + box_w - 3))
    return boxes


def _draw_tray(
    grid: np.ndarray,
    top: int,
    left: int,
    rows: int,
    cols: int,
    box_h: int,
    box_w: int,
    colors: list[int],
    marker_side: str,
) -> list[tuple[int, int, int, int]]:
    out_h = rows * (box_h - 1) + 1
    out_w = cols * (box_w - 1) + 1
    for rr in range(rows + 1):
        grid[top + rr * (box_h - 1), left : left + out_w] = GRAY
    for cc in range(cols + 1):
        grid[top : top + out_h, left + cc * (box_w - 1)] = GRAY

    slot_boxes = _slot_bboxes(top, left, rows, cols, box_h, box_w)
    for color, (r0, c0, r1, c1) in zip(colors, slot_boxes):
        if marker_side == "top":
            grid[r0 - 1, c0 + (c1 - c0) // 2] = color
        elif marker_side == "bottom":
            grid[r1 + 1, c0 + (c1 - c0) // 2] = color
        elif marker_side == "left":
            grid[r0 + (r1 - r0) // 2, c0 - 1] = color
        elif marker_side == "right":
            grid[r0 + (r1 - r0) // 2, c1 + 1] = color
        else:
            raise ValueError(marker_side)
    return slot_boxes


def _render_source_box(grid: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, patch: np.ndarray, rng: random.Random) -> None:
    _draw_rect_border(grid, r0, c0, box_h, box_w, GRAY)
    inner_h = box_h - 2
    inner_w = box_w - 2
    ph, pw = patch.shape
    off_r = rng.randint(0, inner_h - ph)
    off_c = rng.randint(0, inner_w - pw)
    region = grid[r0 + 1 + off_r : r0 + 1 + off_r + ph, c0 + 1 + off_c : c0 + 1 + off_c + pw]
    mask = patch != BG
    region[mask] = patch[mask]


def _side_assignment(n: int, out_h: int, out_w: int, box_h: int, box_w: int, rng: random.Random) -> list[str] | None:
    top_cap = bottom_cap = max(1, (out_w + 1) // (box_w + 1))
    left_cap = right_cap = max(1, (out_h + 1) // (box_h + 1))
    sides = ("top", "bottom", "left", "right")
    for _ in range(200):
        chosen = [rng.choice(sides) for _ in range(n)]
        counts = Counter(chosen)
        if len(counts) < 2:
            continue
        if counts["top"] > top_cap or counts["bottom"] > bottom_cap:
            continue
        if counts["left"] > left_cap or counts["right"] > right_cap:
            continue
        return chosen
    return None


def _make_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n = rng.randint(2, 4)
    box_h = rng.randint(4, 5)
    box_w = rng.randint(4, 5)
    inner_h = box_h - 2
    inner_w = box_w - 2

    patterns = _sample_unique_patterns(inner_h, inner_w, n, rng)
    if patterns is None:
        return None

    orientation = rng.choice(("horizontal", "vertical"))
    if orientation == "horizontal":
        rows, cols = 1, n
        marker_side = rng.choice(("top", "bottom"))
    else:
        rows, cols = n, 1
        marker_side = rng.choice(("left", "right"))

    out_h = rows * (box_h - 1) + 1
    out_w = cols * (box_w - 1) + 1
    sides = _side_assignment(n, out_h, out_w, box_h, box_w, rng)
    if sides is None:
        return None

    side_counts = Counter(sides)
    top_gap = rng.randint(1, 3) if side_counts["top"] else 0
    bottom_gap = rng.randint(1, 3) if side_counts["bottom"] else 0
    left_gap = rng.randint(1, 3) if side_counts["left"] else 0
    right_gap = rng.randint(1, 3) if side_counts["right"] else 0

    top_band = box_h + top_gap + 2 if side_counts["top"] else 1
    bottom_band = box_h + bottom_gap + 2 if side_counts["bottom"] else 1
    left_band = box_w + left_gap + 2 if side_counts["left"] else 1
    right_band = box_w + right_gap + 2 if side_counts["right"] else 1

    h = top_band + out_h + bottom_band
    w = left_band + out_w + right_band
    if h > 30 or w > 30:
        return None

    top = top_band
    left = left_band
    colors = rng.sample(ANCHOR_COLORS, k=n)
    slot_boxes = _slot_bboxes(top, left, rows, cols, box_h, box_w)

    top_starts = _pick_non_overlapping_starts(left, out_w, box_w, side_counts["top"], rng)
    bottom_starts = _pick_non_overlapping_starts(left, out_w, box_w, side_counts["bottom"], rng)
    left_starts = _pick_non_overlapping_starts(top, out_h, box_h, side_counts["left"], rng)
    right_starts = _pick_non_overlapping_starts(top, out_h, box_h, side_counts["right"], rng)
    if top_starts is None or bottom_starts is None or left_starts is None or right_starts is None:
        return None

    top_q = deque(top_starts)
    bottom_q = deque(bottom_starts)
    left_q = deque(left_starts)
    right_q = deque(right_starts)

    grid = np.full((h, w), BG, dtype=int)
    _draw_tray(grid, top, left, rows, cols, box_h, box_w, colors, marker_side)
    expected = grid[top : top + out_h, left : left + out_w].copy()

    slot_by_color = {color: box for color, box in zip(colors, slot_boxes)}
    for color, patch in zip(colors, patterns):
        sr0, sc0, sr1, sc1 = slot_by_color[color]
        slot_h = sr1 - sr0 + 1
        slot_w = sc1 - sc0 + 1
        ph, pw = patch.shape
        pr0 = sr0 - top + (slot_h - ph) // 2
        pc0 = sc0 - left + (slot_w - pw) // 2
        region = expected[pr0 : pr0 + ph, pc0 : pc0 + pw]
        mask = patch != BG
        region[mask] = patch[mask]

    for color, patch, side in zip(colors, patterns, sides):
        if side == "top":
            r0 = 1 + top_gap
            c0 = top_q.popleft()
            _render_source_box(grid, r0, c0, box_h, box_w, patch, rng)
            touch_c = c0 + rng.randint(1, box_w - 2)
            grid[0:r0, touch_c] = color
        elif side == "bottom":
            r0 = h - box_h - bottom_gap - 1
            c0 = bottom_q.popleft()
            _render_source_box(grid, r0, c0, box_h, box_w, patch, rng)
            touch_c = c0 + rng.randint(1, box_w - 2)
            grid[r0 + box_h : h, touch_c] = color
        elif side == "left":
            r0 = left_q.popleft()
            c0 = 1 + left_gap
            _render_source_box(grid, r0, c0, box_h, box_w, patch, rng)
            touch_r = r0 + rng.randint(1, box_h - 2)
            grid[touch_r, 0:c0] = color
        elif side == "right":
            r0 = right_q.popleft()
            c0 = w - box_w - right_gap - 1
            _render_source_box(grid, r0, c0, box_h, box_w, patch, rng)
            touch_r = r0 + rng.randint(1, box_h - 2)
            grid[touch_r, c0 + box_w : w] = color
        else:
            raise ValueError(side)

    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        candidate = _make_case(rng)
        if candidate is None:
            continue
        inp, expected = candidate
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
