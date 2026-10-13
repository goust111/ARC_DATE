from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return ((r0 + r1) // 2, (c0 + c1) // 2)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_hollow_rect_component(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    need = 2 * (h + w) - 4
    if len(comp) != need:
        return None
    cells = set(comp)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border != ((rr, cc) in cells):
                return None
    return (r0, c0, r1, c1)


def _find_frames(grid: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != 0):
        for comp in _color_components(grid, color):
            bbox = _is_hollow_rect_component(comp)
            if bbox is not None:
                frames.append(Frame(color=color, bbox=bbox))
    assert len(frames) >= 3, "expected several hollow rectangular frames"
    colors = [frame.color for frame in frames]
    assert len(colors) == len(set(colors)), "frame border colors must be distinct"
    return frames


def _interior(grid: np.ndarray, frame: Frame) -> np.ndarray:
    r0, c0, r1, c1 = frame.bbox
    return grid[r0 + 1 : r1, c0 + 1 : c1].copy()


def _isolated_roster_pixels(inner: np.ndarray) -> list[tuple[int, int, int]] | None:
    coords = np.argwhere(inner != 0)
    if len(coords) == 0:
        return None
    pixels: list[tuple[int, int, int]] = []
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(inner.shape[0], rr + 2)):
            for nc in range(max(0, cc - 1), min(inner.shape[1], cc + 2)):
                if (nr != rr or nc != cc) and inner[nr, nc] != 0:
                    return None
        pixels.append((rr, cc, int(inner[rr, cc])))
    return pixels


def _matches_relative_layout(candidate: Frame, roster_pixels: list[tuple[int, int, int]], by_color: dict[int, Frame]) -> bool:
    pos_by_color = {color: (r, c) for r, c, color in roster_pixels}
    colors = list(pos_by_color)
    for i, color_a in enumerate(colors):
        ra, ca = pos_by_color[color_a]
        fra, fca = by_color[color_a].center
        for color_b in colors[i + 1 :]:
            rb, cb = pos_by_color[color_b]
            frb, fcb = by_color[color_b].center
            if _sign(ra - rb) != _sign(fra - frb):
                return False
            if _sign(ca - cb) != _sign(fca - fcb):
                return False
    return True


def _find_roster_frame(grid: np.ndarray, frames: list[Frame]) -> tuple[Frame, int]:
    by_color = {frame.color: frame for frame in frames}
    frame_colors = set(by_color)
    candidates: list[tuple[Frame, int]] = []
    for frame in frames:
        inner = _interior(grid, frame)
        roster_pixels = _isolated_roster_pixels(inner)
        if roster_pixels is None:
            continue
        colors = [color for _r, _c, color in roster_pixels]
        color_set = set(colors)
        if len(colors) != len(color_set):
            continue
        if not color_set <= frame_colors:
            continue
        if frame.color not in color_set:
            continue
        missing = list(frame_colors - color_set)
        if len(missing) != 1:
            continue
        if missing[0] == frame.color:
            continue
        if not _matches_relative_layout(frame, roster_pixels, by_color):
            continue
        candidates.append((frame, missing[0]))
    assert len(candidates) == 1, "expected one roster frame"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the roster frame whose isolated color map omits one frame color, then return that omitted frame's interior.

    Input structure:
    - Several distinct one-cell-thick hollow rectangular frames are scattered on a black background, and each frame has its own interior pattern.
    - Exactly one frame contains only isolated single pixels whose colors are frame-border colors, laid out with the same relative directions as the full-frame arrangement.
    - That roster includes its own border color but omits exactly one frame color.

    Rule:
    - Detect all hollow rectangular frames and collect their border colors.
    - Identify the unique roster frame: its interior consists entirely of isolated pixels, each using a frame-border color, and the roster's pixel layout matches the frames' relative positions.
    - Subtract the roster's color set from the full set of frame colors; the single missing color names the target frame.
    - Output the target frame's interior only, excluding the border.

    Key insight:
    - The target frame is designated indirectly: another frame contains a miniature color roster of the whole scene, and the only absent color points to the answer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _find_frames(g)
    roster, missing_color = _find_roster_frame(g, frames)
    _ = roster
    target = next(frame for frame in frames if frame.color == missing_color)
    return _interior(g, target)


_PALETTE = tuple(range(1, 10))
_SIZE_OPTIONS = ((5, 5), (5, 7), (7, 5), (7, 7))
_SLOT_CENTERS = (4, 13, 22)
_BASE_SHAPE = (27, 27)
_MAX_TRIES = 400


def _random_connected_patch(h: int, w: int, colors: list[int]) -> np.ndarray:
    area = h * w
    count = random.randint(max(3, area // 3), max(3, area - 1))
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = {start}
    while len(cells) < count:
        base_r, base_c = random.choice(tuple(frontier))
        nbrs = []
        for nr, nc in ((base_r - 1, base_c), (base_r + 1, base_c), (base_r, base_c - 1), (base_r, base_c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if nbrs:
            nxt = random.choice(nbrs)
            cells.add(nxt)
            frontier.add(nxt)
        else:
            frontier.remove((base_r, base_c))
            if not frontier:
                frontier = set(cells)

    patch = np.zeros((h, w), dtype=int)
    used_colors = random.sample(colors, random.randint(1, min(3, len(colors))))
    order = list(cells)
    random.shuffle(order)
    assigned = used_colors[:]
    while len(assigned) < len(order):
        assigned.append(random.choice(used_colors))
    random.shuffle(assigned)
    for (r, c), color in zip(order, assigned):
        patch[r, c] = color
    assert any(
        patch[r, c] != 0 and patch[nr, nc] != 0
        for r in range(h)
        for c in range(w)
        for nr, nc in ((r + 1, c), (r, c + 1))
        if nr < h and nc < w
    )
    return patch


def _frame_top_left(slot_r: int, slot_c: int, outer_h: int, outer_w: int) -> tuple[int, int]:
    return _SLOT_CENTERS[slot_r] - outer_h // 2, _SLOT_CENTERS[slot_c] - outer_w // 2


def _draw_frame(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_patch(grid: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, c0, _r1, _c1 = bbox
    grid[r0 + 1 : r0 + 1 + patch.shape[0], c0 + 1 : c0 + 1 + patch.shape[1]] = patch


def _crop_scene(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(2, int(r0)))
    left = random.randint(0, min(2, int(c0)))
    bottom = random.randint(0, min(2, grid.shape[0] - 1 - int(r1)))
    right = random.randint(0, min(2, grid.shape[1] - 1 - int(c1)))
    return grid[int(r0) - top : int(r1) + bottom + 1, int(c0) - left : int(c1) + right + 1].copy()


def _choose_slots(n_frames: int) -> list[tuple[int, int]]:
    slots = [(r, c) for r in range(3) for c in range(3)]
    while True:
        chosen = random.sample(slots, n_frames)
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if len(rows) >= 2 and len(cols) >= 2:
            return chosen


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.randint(4, 6)
    border_colors = random.sample(_PALETTE, n_frames)
    inside_palette = [color for color in _PALETTE if color not in border_colors]
    if len(inside_palette) < 3:
        return None

    slots = _choose_slots(n_frames)
    roster_idx, target_idx = random.sample(range(n_frames), 2)

    sizes = [random.choice(_SIZE_OPTIONS) for _ in range(n_frames)]
    sizes[roster_idx] = (7, 7)
    if not any(i != target_idx and sizes[i] == sizes[target_idx] for i in range(n_frames)):
        other_choices = [i for i in range(n_frames) if i != target_idx and i != roster_idx]
        if other_choices:
            sizes[random.choice(other_choices)] = sizes[target_idx]

    grid = np.zeros(_BASE_SHAPE, dtype=int)
    frames: list[Frame] = []
    patches: dict[int, np.ndarray] = {}

    for idx, ((slot_r, slot_c), color, (outer_h, outer_w)) in enumerate(zip(slots, border_colors, sizes)):
        top, left = _frame_top_left(slot_r, slot_c, outer_h, outer_w)
        bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
        frame = Frame(color=color, bbox=bbox)
        frames.append(frame)
        _draw_frame(grid, color, bbox)

        if idx == roster_idx:
            continue
        patch = _random_connected_patch(outer_h - 2, outer_w - 2, inside_palette)
        patches[color] = patch
        _place_patch(grid, bbox, patch)

    roster_frame = frames[roster_idx]
    target_frame = frames[target_idx]
    expected = patches[target_frame.color].copy()

    roster_inner = np.zeros((5, 5), dtype=int)
    for idx, ((slot_r, slot_c), color) in enumerate(zip(slots, border_colors)):
        if idx == target_idx:
            continue
        roster_inner[slot_r * 2, slot_c * 2] = color
    _place_patch(grid, roster_frame.bbox, roster_inner)

    cropped = _crop_scene(grid)

    try:
        solved = main(cropped)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    frames_found = _find_frames(cropped)
    roster_found, missing_color = _find_roster_frame(cropped, frames_found)
    if missing_color != target_frame.color:
        return None
    if roster_found.color != roster_frame.color:
        return None
    if np.count_nonzero(expected) < 3:
        return None
    if np.array_equal(expected, roster_inner):
        return None
    return cropped, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_TRIES):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid self-referential roster frame task")
