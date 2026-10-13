import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "right", "down", "left")
ROT_K_BY_DIRECTION = {
    "down": 0,
    "right": 1,
    "up": 2,
    "left": 3,
}

SIZE_CHOICES = [3, 5, 7, 9]
PAIR_COUNT_CHOICES = [1, 2, 2, 3]
GRID_MIN = 18
GRID_MAX = 30
PLACEMENT_TRIES = 250
PATTERN_TRIES = 300
GEN_TRIES = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _direction_from_gap(
    gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int
) -> str | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1 and gc == c0 + (c1 - c0) // 2 and (c1 - c0 + 1) % 2 == 1:
        return "up"
    if gr == r1 and c0 < gc < c1 and gc == c0 + (c1 - c0) // 2 and (c1 - c0 + 1) % 2 == 1:
        return "down"
    if gc == c0 and r0 < gr < r1 and gr == r0 + (r1 - r0) // 2 and (r1 - r0 + 1) % 2 == 1:
        return "left"
    if gc == c1 and r0 < gr < r1 and gr == r0 + (r1 - r0) // 2 and (r1 - r0 + 1) % 2 == 1:
        return "right"
    return None


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in (bg, GREY):
            continue
        for comp in _components(grid == color):
            if len(comp) < 8:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            comp_set = set(comp)
            border = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            if any((r, c) not in border for r, c in comp):
                continue
            missing = list(border - comp_set)
            if len(missing) != 1:
                continue
            direction = _direction_from_gap(missing[0], r0, c0, r1, c1)
            if direction is None:
                continue
            if h <= 2 or w <= 2:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or not np.all(inner == bg):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "direction": direction,
                    "inner_top": r0 + 1,
                    "inner_left": c0 + 1,
                    "inner_shape": (h - 2, w - 2),
                }
            )
    return frames


def _component_from(start: tuple[int, int], mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    stack = [(sr, sc)]
    seen = {start}
    out: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        out.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return out


def _detect_templates(grid: np.ndarray, bg: int) -> list[dict]:
    non_bg = grid != bg
    templates: list[dict] = []
    used_grey: set[tuple[int, int]] = set()
    for gr, gc in map(tuple, np.argwhere(grid == GREY)):
        gr = int(gr)
        gc = int(gc)
        if (gr, gc) in used_grey:
            continue
        comp = _component_from((gr, gc), non_bg)
        grey_pixels = [(r, c) for r, c in comp if int(grid[r, c]) == GREY]
        if len(grey_pixels) != 1:
            continue
        used_grey.update(grey_pixels)
        body = [(r, c) for r, c in comp if int(grid[r, c]) != GREY]
        if not body:
            continue
        r0, c0, r1, c1 = _bbox(body)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if w % 2 != 1:
            continue
        if (gr, gc) != (r1 + 1, c0 + w // 2):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg) or np.any(patch == GREY):
            continue
        if len(body) != h * w:
            continue
        templates.append(
            {
                "patch": patch.copy(),
                "shape": (h, w),
                "marker": (gr, gc),
            }
        )
    return templates


def _match_frames_to_templates(frames: list[dict], templates: list[dict]) -> dict[int, int]:
    candidates: dict[int, list[int]] = {}
    for fi, frame in enumerate(frames):
        rot_k = ROT_K_BY_DIRECTION[frame["direction"]]
        matches = []
        for ti, template in enumerate(templates):
            if np.rot90(template["patch"], rot_k).shape == frame["inner_shape"]:
                matches.append(ti)
        assert matches, "each frame must have a size-compatible template"
        candidates[fi] = matches

    order = sorted(range(len(frames)), key=lambda fi: len(candidates[fi]))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        fi = order[idx]
        for ti in candidates[fi]:
            if ti in used:
                continue
            used.add(ti)
            assignment[fi] = ti
            if dfs(idx + 1):
                return True
            used.remove(ti)
            del assignment[fi]
        return False

    assert dfs(0), "frame/template matching must be unambiguous"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each notched frame with the size-matched template rotated so its default
    downward grey-marker orientation aligns with the frame's gap direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    frames = _detect_frames(g, bg)
    templates = _detect_templates(g, bg)
    assert frames and len(frames) == len(templates)

    assignment = _match_frames_to_templates(frames, templates)
    out = g.copy()
    for fi, ti in assignment.items():
        frame = frames[fi]
        patch = np.rot90(templates[ti]["patch"], ROT_K_BY_DIRECTION[frame["direction"]])
        h, w = patch.shape
        r0 = frame["inner_top"]
        c0 = frame["inner_left"]
        out[r0 : r0 + h, c0 : c0 + w] = patch
    return out


def _render_frame(size: int, color: int, direction: str) -> np.ndarray:
    arr = np.full((size + 2, size + 2), BG, dtype=int)
    arr[0, :] = color
    arr[-1, :] = color
    arr[:, 0] = color
    arr[:, -1] = color
    mid = (size + 2) // 2
    if direction == "up":
        arr[0, mid] = BG
    elif direction == "down":
        arr[-1, mid] = BG
    elif direction == "left":
        arr[mid, 0] = BG
    else:
        arr[mid, -1] = BG
    return arr


def _random_patch(size: int) -> np.ndarray:
    for _ in range(PATTERN_TRIES):
        n_colors = random.choice([2, 3])
        colors = random.sample(NON_SPECIAL_COLORS, n_colors)
        patch = np.full((size, size), colors[0], dtype=int)

        rect_h = random.randint(1, size - 1)
        rect_w = random.randint(1, size - 1)
        rect_r = random.randint(0, size - rect_h)
        rect_c = random.randint(0, size - rect_w)
        patch[rect_r : rect_r + rect_h, rect_c : rect_c + rect_w] = colors[1]

        extra_color = colors[2] if n_colors == 3 else colors[1]
        for _ in range(random.randint(size, size + 2)):
            rr = random.randrange(size)
            cc = random.randrange(size)
            patch[rr, cc] = extra_color

        if not all(np.any(patch == color) for color in colors):
            continue
        if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            continue
        return patch
    raise RuntimeError("failed to build an asymmetric template")


def _render_template(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    arr = np.full((h + 1, w), BG, dtype=int)
    arr[:h, :w] = patch
    arr[h, w // 2] = GREY
    return arr


def _place_box(
    occupied: np.ndarray, obj_h: int, obj_w: int, pad: int = 1
) -> tuple[int, int] | None:
    h, w = occupied.shape
    if obj_h > h or obj_w > w:
        return None
    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(0, h - obj_h)
        c0 = random.randint(0, w - obj_w)
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(h, r0 + obj_h + pad)
        cc1 = min(w, c0 + obj_w + pad)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        occupied[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    sizes = random.sample(SIZE_CHOICES, n_pairs)
    frame_colors = random.sample(NON_SPECIAL_COLORS, n_pairs)
    directions = [random.choice(DIRECTIONS) for _ in range(n_pairs)]
    patches = [_random_patch(size) for size in sizes]

    total_footprint = 0
    for size in sizes:
        total_footprint += (size + 4) * (size + 4)
        total_footprint += (size + 3) * (size + 4)
    base_dim = max(GRID_MIN, int(np.ceil(np.sqrt(total_footprint * 1.8))))
    if base_dim > GRID_MAX:
        return None

    height = random.randint(base_dim, GRID_MAX)
    width = random.randint(base_dim, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)
    out = grid.copy()
    occupied = np.zeros((height, width), dtype=bool)

    specs = []
    order = list(range(n_pairs))
    random.shuffle(order)
    for idx in order:
        size = sizes[idx]
        direction = directions[idx]
        frame = _render_frame(size, frame_colors[idx], direction)
        template = _render_template(patches[idx])

        frame_pos = _place_box(occupied, frame.shape[0], frame.shape[1], pad=1)
        if frame_pos is None:
            return None
        template_pos = _place_box(occupied, template.shape[0], template.shape[1], pad=1)
        if template_pos is None:
            return None

        fr, fc = frame_pos
        tr, tc = template_pos
        grid[fr : fr + frame.shape[0], fc : fc + frame.shape[1]] = frame
        grid[tr : tr + template.shape[0], tc : tc + template.shape[1]] = template
        specs.append(
            {
                "size": size,
                "direction": direction,
                "patch": patches[idx],
                "frame_pos": frame_pos,
            }
        )

    out[:, :] = grid
    for spec in specs:
        fr, fc = spec["frame_pos"]
        rotated = np.rot90(spec["patch"], ROT_K_BY_DIRECTION[spec["direction"]])
        s = rotated.shape[0]
        out[fr + 1 : fr + 1 + s, fc + 1 : fc + 1 + s] = rotated

    if _most_common_color(grid) != BG:
        return None
    if int(np.sum(grid == GREY)) != n_pairs:
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        case = _build_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC input")
