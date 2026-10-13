from __future__ import annotations

import random
from collections import Counter, deque, defaultdict
from dataclasses import dataclass

import numpy as np

Coord = tuple[int, int]
ShapeKey = tuple[Coord, ...]
BBox = tuple[int, int, int, int]

BACKGROUND = 0
FLOOD_COLOR = 8
GEN_CANVAS = 30
GEN_ATTEMPTS = 400
GEN_FRAME_COUNT = (2, 4)
GEN_FRAME_SIZE = (6, 10)
GEN_MARGIN = (1, 3)
GEN_FRAME_PADDING = 2
GEN_PLUG_PADDING = 1
GEN_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 9)

GAP_LIBRARY = (
    {"name": "line1", "kind": "line", "length": 1},
    {"name": "line2", "kind": "line", "length": 2},
    {"name": "line3", "kind": "line", "length": 3},
    {"name": "line4", "kind": "line", "length": 4},
    {"name": "L22", "kind": "corner", "arms": (2, 2)},
    {"name": "L32", "kind": "corner", "arms": (3, 2)},
    {"name": "L33", "kind": "corner", "arms": (3, 3)},
)

DISTRACTOR_SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
)


@dataclass(frozen=True)
class Frame:
    bbox: BBox
    coords: frozenset[Coord]
    gap_coords: frozenset[Coord]
    gap_key: ShapeKey


@dataclass(frozen=True)
class Plug:
    color: int
    coords: frozenset[Coord]
    key: ShapeKey


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def bbox_of(coords: set[Coord] | frozenset[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def connected_components(mask: np.ndarray) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: set[Coord] = set()
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def normalize_coords(coords: set[Coord] | frozenset[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def rotate_once(coords: tuple[Coord, ...]) -> tuple[Coord, ...]:
    height = max(r for r, _ in coords) + 1
    return normalize_coords(tuple((c, height - 1 - r) for r, c in coords))


def shape_key(coords: set[Coord] | frozenset[Coord] | tuple[Coord, ...]) -> ShapeKey:
    cur = normalize_coords(coords)
    variants: list[ShapeKey] = []
    for _ in range(4):
        variants.append(cur)
        cur = rotate_once(cur)
    return min(variants)


def border_coords(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    border: set[Coord] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def is_connected(coords: set[Coord]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def infer_frames(grid: np.ndarray, frame_color: int) -> list[Frame]:
    frames: list[Frame] = []
    for comp in connected_components(grid == frame_color):
        r0, c0, r1, c1 = bbox_of(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = border_coords((r0, c0, r1, c1))
        if not comp.issubset(border):
            continue
        gap = border - comp
        if not gap or not is_connected(gap):
            continue
        frames.append(
            Frame(
                bbox=(r0, c0, r1, c1),
                coords=frozenset(comp),
                gap_coords=frozenset(gap),
                gap_key=shape_key(gap),
            )
        )
    return frames


def infer_plugs(grid: np.ndarray, bg: int, frame_color: int) -> list[Plug]:
    plugs: list[Plug] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (bg, frame_color)):
        for comp in connected_components(grid == color):
            plugs.append(Plug(color=color, coords=frozenset(comp), key=shape_key(comp)))
    return plugs


def flood_from_boundary(grid: np.ndarray, bg: int, fill: int) -> None:
    h, w = grid.shape
    q = deque()
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == bg and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == bg and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        grid[r, c] = fill
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == bg:
                seen[nr, nc] = True
                q.append((nr, nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered plugs into matching frame gaps, then flood every boundary-reachable open region teal.

    Input structure:
    - The background contains several one-cell-thick hollow rectangular frames that share one structural color.
    - Each frame is missing one connected border patch that forms its gap.
    - Scattered monochrome plug pieces may match a frame gap up to rotation; some plugs may be distractors and some frames may stay unplugged.

    Rule:
    - For every gap shape that has exactly one plug with the same shape up to rotation, erase that plug from its source location and place it into the corresponding frame gap using the plug's color.
    - Leave unmatched plugs and unmatched frames unchanged.
    - After all moves, flood-fill from the grid boundary through 4-connected background cells and recolor every reachable background cell teal.

    Key insight:
    - Dry versus flooded rooms depends on the post-move topology, so the plug transport must be completed before computing boundary reachability.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    counts = Counter(map(int, g[g != bg].flatten()))
    assert counts, "expected at least one foreground color"
    frame_color = counts.most_common(1)[0][0]

    frames = infer_frames(g, frame_color)
    assert frames, "expected at least one valid frame"
    plugs = infer_plugs(g, bg, frame_color)

    frames_by_key: dict[ShapeKey, list[Frame]] = defaultdict(list)
    for frame in frames:
        frames_by_key[frame.gap_key].append(frame)

    plugs_by_key: dict[ShapeKey, list[Plug]] = defaultdict(list)
    for plug in plugs:
        plugs_by_key[plug.key].append(plug)

    out = g.copy()
    for key, matching_frames in frames_by_key.items():
        matching_plugs = plugs_by_key.get(key, [])
        if len(matching_frames) != 1 or len(matching_plugs) != 1:
            continue
        frame = matching_frames[0]
        plug = matching_plugs[0]
        for r, c in plug.coords:
            out[r, c] = bg
        for r, c in frame.gap_coords:
            out[r, c] = plug.color

    flood_from_boundary(out, bg, FLOOD_COLOR)
    return out


def patch_from_coords(coords: tuple[Coord, ...] | list[Coord], color: int) -> np.ndarray:
    norm = normalize_coords(tuple(coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.full((h, w), BACKGROUND, dtype=int)
    for r, c in norm:
        patch[r, c] = color
    return patch


def rotate_patch(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, k=turns).copy()


def choose_frame_size(template: dict) -> tuple[int, int]:
    min_h = GEN_FRAME_SIZE[0]
    min_w = GEN_FRAME_SIZE[0]
    if template["kind"] == "line":
        length = int(template["length"])
        min_w = max(min_w, length + 2)
    else:
        arm_h, arm_w = sorted(tuple(int(x) for x in template["arms"]), reverse=True)
        min_h = max(min_h, arm_h + 2)
        min_w = max(min_w, arm_w + 2)
    h = random.randint(min_h, GEN_FRAME_SIZE[1])
    w = random.randint(min_w, GEN_FRAME_SIZE[1])
    return h, w


def build_frame_patch(frame_color: int, template: dict) -> tuple[np.ndarray, list[Coord]]:
    h, w = choose_frame_size(template)
    patch = np.full((h, w), BACKGROUND, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color

    gap_mask = np.zeros((h, w), dtype=bool)
    if template["kind"] == "line":
        length = int(template["length"])
        start = random.randint(1, w - length - 1)
        gap_mask[0, start : start + length] = True
    else:
        arm_a, arm_b = (int(x) for x in template["arms"])
        gap_mask[0, :arm_a] = True
        gap_mask[:arm_b, 0] = True

    turns = random.randint(0, 3)
    patch = rotate_patch(patch, turns)
    gap_mask = np.rot90(gap_mask, k=turns)
    gap_coords = [tuple(map(int, rc)) for rc in np.argwhere(gap_mask)]
    for r, c in gap_coords:
        patch[r, c] = BACKGROUND
    return patch, gap_coords


def can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int, padding: int, border: int = 1) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < border or left < border or top + ph > h - border or left + pw > w - border:
        return False
    coords = np.argwhere(patch != BACKGROUND)
    for r_raw, c_raw in coords:
        r = top + int(r_raw)
        c = left + int(c_raw)
        if occupied[r, c]:
            return False
        r0 = max(0, r - padding)
        r1 = min(h, r + padding + 1)
        c0 = max(0, c - padding)
        c1 = min(w, c + padding + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    mask = patch != BACKGROUND
    window = grid[top : top + ph, left : left + pw]
    window[mask] = patch[mask]


def place_random(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, padding: int) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    max_r = h - ph - 1
    max_c = w - pw - 1
    if max_r < 1 or max_c < 1:
        return None
    candidates = [(r, c) for r in range(1, max_r + 1) for c in range(1, max_c + 1)]
    random.shuffle(candidates)
    for top, left in candidates:
        if can_place(occupied, patch, top, left, padding):
            paste_patch(grid, patch, top, left)
            occupied[top : top + ph, left : left + pw] |= patch != BACKGROUND
            return top, left
    return None


def crop_canvas(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BACKGROUND)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    def margin(limit: int) -> int:
        if limit <= 0:
            return 0
        return random.randint(GEN_MARGIN[0], min(GEN_MARGIN[1], limit))

    top = max(0, int(r0) - margin(int(r0)))
    left = max(0, int(c0) - margin(int(c0)))
    bottom_room = grid.shape[0] - 1 - int(r1)
    right_room = grid.shape[1] - 1 - int(c1)
    bottom = min(grid.shape[0], int(r1) + margin(int(bottom_room)) + 1)
    right = min(grid.shape[1], int(c1) + margin(int(right_room)) + 1)
    return grid[top:bottom, left:right].copy()


def frame_gap_key(gap_coords: list[Coord]) -> ShapeKey:
    return shape_key(tuple(gap_coords))


def distractor_shape_keys() -> set[ShapeKey]:
    return {shape_key(shape) for shape in DISTRACTOR_SHAPES}


def build_attempt() -> np.ndarray | None:
    grid = np.full((GEN_CANVAS, GEN_CANVAS), BACKGROUND, dtype=int)
    occupied = np.zeros((GEN_CANVAS, GEN_CANVAS), dtype=bool)

    frame_color = random.choice(GEN_COLOR_POOL)
    plug_colors = [c for c in GEN_COLOR_POOL if c != frame_color]
    random.shuffle(plug_colors)

    n_frames = random.randint(*GEN_FRAME_COUNT)
    templates = random.sample(list(GAP_LIBRARY), n_frames)
    if n_frames > 1 and random.random() < 0.7:
        n_matched = random.randint(1, n_frames - 1)
    else:
        n_matched = random.randint(1, n_frames)
    matched_indices = set(random.sample(range(n_frames), n_matched))

    n_distractors = 0
    if random.random() < 0.65:
        n_distractors = random.randint(1, 2)
    if n_matched + n_distractors > len(plug_colors):
        n_distractors = max(0, len(plug_colors) - n_matched)

    frames_info: list[tuple[list[Coord], ShapeKey]] = []
    for template in templates:
        patch, gap_coords = build_frame_patch(frame_color, template)
        placed = place_random(grid, occupied, patch, GEN_FRAME_PADDING)
        if placed is None:
            return None
        top, left = placed
        frames_info.append(
            (
                [(top + r, left + c) for r, c in gap_coords],
                frame_gap_key(gap_coords),
            )
        )

    color_index = 0
    for idx, (gap_coords, _gap_key) in enumerate(frames_info):
        if idx not in matched_indices:
            continue
        color = plug_colors[color_index]
        color_index += 1
        patch = patch_from_coords(gap_coords, color)
        patch = rotate_patch(patch, random.randint(0, 3))
        if place_random(grid, occupied, patch, GEN_PLUG_PADDING) is None:
            return None

    gap_keys = {gap_key for _, gap_key in frames_info}
    safe_distractors = [shape for shape in DISTRACTOR_SHAPES if shape_key(shape) not in gap_keys]
    if len(safe_distractors) < n_distractors:
        return None
    for _ in range(n_distractors):
        color = plug_colors[color_index]
        color_index += 1
        shape = random.choice(safe_distractors)
        patch = patch_from_coords(shape, color)
        patch = rotate_patch(patch, random.randint(0, 3))
        if place_random(grid, occupied, patch, GEN_PLUG_PADDING) is None:
            return None

    cropped = crop_canvas(grid)
    if FLOOD_COLOR in cropped:
        return None

    out = main(cropped)
    if cropped.shape[0] > 30 or cropped.shape[1] > 30:
        return None
    if out.shape[0] > 30 or out.shape[1] > 30:
        return None
    if not np.any(out == FLOOD_COLOR):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flood-and-plug input")
