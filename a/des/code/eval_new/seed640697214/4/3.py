from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
MARKER_COLOR = 2
TEMPLATE_COLORS = (1, 3, 4, 7)

GRID_HEIGHT_RANGE = (24, 30)
GRID_WIDTH_RANGE = (24, 30)
FRAME_COUNT_CHOICES = (4, 5)
INNER_HEIGHT_RANGE = (4, 8)
INNER_WIDTH_RANGE = (4, 8)
TEMPLATE_DIM_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4))
TEMPLATE_AREA_RANGE = (3, 8)
PLACEMENT_RETRIES = 160
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
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
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for k in range(4):
        rotated = np.rot90(mask, k)
        for flipped in (False, True):
            cur = np.fliplr(rotated) if flipped else rotated
            key = (tuple(map(int, cur.shape)), cur.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _is_rectangular_frame(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 6 or w < 6:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False

    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            present = (r, c) in coord_set
            if on_border != present:
                return False
            if present and int(grid[r, c]) != int(color):
                return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            if _is_rectangular_frame(g, color, comp):
                frames.append(
                    {
                        "color": color,
                        "coords": comp,
                        "bbox": _bbox(comp),
                    }
                )
    assert frames, "expected at least one closed rectangular frame"
    frames.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    return frames


def _find_templates(grid: np.ndarray, frames: list[dict[str, object]], bg: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    blocked = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    templates = []
    for comp in _components((g != bg) & (~blocked)):
        r0, c0, r1, c1 = _bbox(comp)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h > 4 or w > 4:
            continue
        patch = np.full((h, w), int(bg), dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = int(g[r, c])
        templates.append({"bbox": (r0, c0, r1, c1), "patch": patch})

    assert len(templates) == 4, "expected exactly four small templates outside the frames"
    templates.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    return [np.asarray(item["patch"], dtype=int) for item in templates]


def _stamp_center(out: np.ndarray, patch: np.ndarray, inner_bbox: tuple[int, int, int, int], bg: int) -> None:
    r0, c0, r1, c1 = inner_bbox
    ih = int(r1 - r0 + 1)
    iw = int(c1 - c0 + 1)
    ph, pw = map(int, patch.shape)
    assert ph <= ih and pw <= iw, "template must fit inside the frame interior"

    sr = int(r0 + (ih - ph) // 2)
    sc = int(c0 + (iw - pw) // 2)
    sub = out[sr : sr + ph, sc : sc + pw]
    mask = patch != int(bg)
    sub[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the marker pixels inside each rectangular frame and stamp the matching bottom-right template into the frame.

    Input structure:
    - The background is the most common color.
    - Several one-cell-thick monochrome rectangular frames enclose empty interiors; each interior contains a small number of marker pixels.
    - A separate group of four small template sprites is stacked vertically near the bottom-right corner, with background gaps between consecutive templates.

    Rule:
    - Detect every closed rectangular frame and count how many non-background marker pixels lie inside its interior.
    - Read the template group from top to bottom as templates 1 through 4, choose template N for a frame with N markers, and clear that frame interior back to background.
    - Stamp the chosen template centered inside the cleared interior; if exact centering is impossible, use the upper-left of the two central alignments.
    - Keep the frame borders and the template group itself unchanged.

    Key insight:
    - Marker positions are irrelevant: only the count matters, and the template index is encoded implicitly by the templates' vertical order rather than by their colors or shapes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = _find_frames(g, bg)
    templates = _find_templates(g, frames, bg)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(inner != int(bg)))
        assert 1 <= marker_count <= 4, "frame marker count must map to one of the four templates"

        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)
        _stamp_center(out, templates[marker_count - 1], (r0 + 1, c0 + 1, r1 - 1, c1 - 1), bg)

    return out


def _random_connected_mask() -> np.ndarray:
    h, w = random.choice(TEMPLATE_DIM_CHOICES)
    area = random.randint(TEMPLATE_AREA_RANGE[0], min(TEMPLATE_AREA_RANGE[1], h * w))
    start = (random.randrange(h), random.randrange(w))
    cells = {start}

    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_connected_mask()
        cells.add(random.choice(frontier))

    return _tight_mask([(int(r), int(c)) for r, c in cells])


def _build_templates() -> list[np.ndarray] | None:
    patches: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for color in TEMPLATE_COLORS:
        patch = None
        for _ in range(80):
            mask = _random_connected_mask()
            key = _canonical_mask(mask)
            if key in seen:
                continue
            seen.add(key)
            patch = np.full(mask.shape, int(BG), dtype=int)
            patch[mask] = int(color)
            break
        if patch is None:
            return None
        patches.append(patch)
    return patches


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int = BG) -> None:
    ph, pw = map(int, patch.shape)
    sub = grid[top : top + ph, left : left + pw]
    mask = patch != int(bg)
    sub[mask] = patch[mask]


def _place_template_group(grid: np.ndarray, templates: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    group_h = int(sum(int(t.shape[0]) for t in templates) + (len(templates) - 1))
    group_w = int(max(int(t.shape[1]) for t in templates))
    bottom_margin = random.randint(0, 1)
    right_margin = random.randint(0, 1)
    r0 = int(h - bottom_margin - group_h)
    c0 = int(w - right_margin - group_w)
    if r0 < 1 or c0 < 1:
        return None

    rr = r0
    for patch in templates:
        ph, pw = map(int, patch.shape)
        cc = int(c0 + (group_w - pw) // 2)
        _paste_patch(grid, patch, rr, cc)
        rr += ph + 1
    return r0, c0, r0 + group_h - 1, c0 + group_w - 1


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(FRAME_COLOR)
    grid[r1, c0 : c1 + 1] = int(FRAME_COLOR)
    grid[r0 : r1 + 1, c0] = int(FRAME_COLOR)
    grid[r0 : r1 + 1, c1] = int(FRAME_COLOR)

    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    markers = random.sample(cells, count)
    for r, c in markers:
        grid[r, c] = int(MARKER_COLOR)
    return markers


def _render_expected(grid: np.ndarray, frames: list[dict[str, object]], templates: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        count = int(frame["count"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(BG)
        _stamp_center(out, templates[count - 1], (r0 + 1, c0 + 1, r1 - 1, c1 - 1), BG)
    return out


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), int(BG), dtype=int)

    templates = _build_templates()
    if templates is None:
        return None
    group_bbox = _place_template_group(grid, templates)
    if group_bbox is None:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    _mark(occupied, *group_bbox, pad=1)

    frame_count = random.choice(FRAME_COUNT_CHOICES)
    counts = [1, 2, 3, 4] + [random.randint(1, 4) for _ in range(frame_count - 4)]
    random.shuffle(counts)

    frames: list[dict[str, object]] = []
    max_inner_h = INNER_HEIGHT_RANGE[1] - (1 if frame_count > 4 else 0)
    max_inner_w = INNER_WIDTH_RANGE[1] - (1 if frame_count > 4 else 0)

    for count in counts:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            inner_h = random.randint(INNER_HEIGHT_RANGE[0], max_inner_h)
            inner_w = random.randint(INNER_WIDTH_RANGE[0], max_inner_w)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if h - outer_h - 2 < 1 or w - outer_w - 2 < 1:
                continue
            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
            r1 = int(r0 + outer_h - 1)
            c1 = int(c0 + outer_w - 1)
            if not _can_place(occupied, r0, c0, r1, c1, pad=1):
                continue

            _mark(occupied, r0, c0, r1, c1, pad=1)
            _draw_frame(grid, (r0, c0, r1, c1), count)
            frames.append({"bbox": (r0, c0, r1, c1), "count": count})
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(grid, frames, templates)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 3")
