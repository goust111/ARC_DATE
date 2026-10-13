from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_shared_arc_utils", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    crop_bbox = _ARC_UTILS.crop_bbox
    bbox_h = _ARC_UTILS.bbox_h
    bbox_w = _ARC_UTILS.bbox_w
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        ORANGE = 7

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()

    def bbox_h(box: tuple[int, int, int, int]) -> int:
        return int(box[2] - box[0] + 1)

    def bbox_w(box: tuple[int, int, int, int]) -> int:
        return int(box[3] - box[1] + 1)

    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid), return_counts=True)
        return int(values[np.argmax(counts)])

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | tuple[int, ...] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        bg = int(bg)
        allowed = None if colors is None else {int(c) for c in colors if int(c) != bg}
        if return_format != "dict":
            raise ValueError("fallback find_components only supports return_format='dict'")
        if connectivity == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        h, w = g.shape
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c]:
                    continue
                color = int(g[r, c])
                if color == bg:
                    continue
                if allowed is not None and color not in allowed:
                    continue
                seen[r, c] = True
                stack = [(r, c)]
                coords: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                            continue
                        if monochromatic:
                            ok = int(g[nr, nc]) == color
                        else:
                            ok = int(g[nr, nc]) != bg and (allowed is None or int(g[nr, nc]) in allowed)
                        if ok:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                if len(coords) < int(min_area):
                    continue
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": color if monochromatic else None,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(len(coords)),
                    }
                )
        return out


BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
GREY = int(Color.GREY)
ORANGE = int(Color.ORANGE)
SWATCH_COLORS = [RED, BLUE, GREEN]

TEMPLATE_POOL = [
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=bool),
]

GEN_MAX_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_templates(grid: np.ndarray, stripe_row: int) -> list[dict]:
    upper = np.asarray(grid[:stripe_row], dtype=int)
    grey_components = find_components(
        upper,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )
    templates = []
    grey_count = int((upper == GREY).sum())
    used_grey = 0
    for comp in grey_components:
        box = comp["bbox"]
        if bbox_h(box) != 3 or bbox_w(box) != 3:
            continue
        mask = crop_bbox(upper, box) == GREY
        templates.append({"top": int(box[0]), "left": int(box[1]), "mask": mask})
        used_grey += int(mask.sum())
    assert len(templates) == 4
    assert used_grey == grey_count
    return templates


def _extract_obstacles(grid: np.ndarray, stripe_row: int) -> set[tuple[int, int]]:
    coords = np.argwhere(np.asarray(grid[:stripe_row], dtype=int) == ORANGE)
    return {tuple(map(int, coord)) for coord in coords}


def _extract_swatches_and_pattern(grid: np.ndarray, stripe_row: int) -> tuple[list[int], np.ndarray]:
    lower = np.asarray(grid[stripe_row + 1 :], dtype=int)
    colored = np.argwhere(lower != BG)
    assert colored.size > 0
    min_col = int(colored[:, 1].min())

    swatch_row = None
    swatch_colors = None
    for r in range(lower.shape[0] - 1):
        if min_col + 8 > lower.shape[1]:
            continue
        colors = []
        ok = True
        for c in (min_col, min_col + 3, min_col + 6):
            block = lower[r : r + 2, c : c + 2]
            if block.shape != (2, 2):
                ok = False
                break
            if np.any(block == BG):
                ok = False
                break
            if not np.all(block == block[0, 0]):
                ok = False
                break
            colors.append(int(block[0, 0]))
        if ok and np.all(lower[r : r + 2, min_col + 2] == BG) and np.all(lower[r : r + 2, min_col + 5] == BG):
            swatch_row = r
            swatch_colors = colors
            break
    assert swatch_row is not None and swatch_colors is not None

    swatch_mask = np.zeros_like(lower, dtype=bool)
    for c in (min_col, min_col + 3, min_col + 6):
        swatch_mask[swatch_row : swatch_row + 2, c : c + 2] = True

    pattern_mask = (lower != BG) & ~swatch_mask
    pattern_box = _bbox_from_mask(pattern_mask)
    assert pattern_box is not None
    r0, c0, r1, c1 = pattern_box
    pattern = lower[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all(pattern != BG)
    assert int(pattern_mask.sum()) == int(pattern.size)
    assert set(map(int, np.unique(pattern))).issubset(set(swatch_colors))
    return swatch_colors, pattern


def _filled_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in coords]


def _settle_templates(
    templates: list[dict],
    stripe_row: int,
    obstacles: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[dict]:
    h, w = map(int, shape)
    occupied = {(stripe_row, c) for c in range(w)}
    occupied.update({(int(r), int(c)) for r, c in obstacles})
    settled = []

    for template in sorted(templates, key=lambda t: (int(t["top"]) + 2, int(t["left"])), reverse=True):
        top = int(template["top"])
        left = int(template["left"])
        mask = np.asarray(template["mask"], dtype=bool)
        rel = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
        while True:
            can_drop = True
            for rr, cc in rel:
                nr = top + rr + 1
                nc = left + cc
                if nr >= h or (nr, nc) in occupied:
                    can_drop = False
                    break
            if not can_drop:
                break
            top += 1
        for cell in _filled_coords(mask, top, left):
            occupied.add(cell)
        settled.append({"top": top, "left": left, "mask": mask})
    return settled


def _expand_pattern(pattern: np.ndarray, mapping: dict[int, np.ndarray]) -> np.ndarray:
    pattern = np.asarray(pattern, dtype=int)
    h, w = pattern.shape
    out = np.zeros((h * 3, w * 3), dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(pattern[r, c])
            mask = np.asarray(mapping[color], dtype=bool)
            block = out[r * 3 : (r + 1) * 3, c * 3 : (c + 1) * 3]
            block[mask] = color
    return out


def _solve(grid: np.ndarray) -> tuple[list[int], np.ndarray, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    stripe_rows = [r for r in range(g.shape[0]) if np.all(g[r] == YELLOW)]
    assert len(stripe_rows) == 1
    stripe_row = int(stripe_rows[0])

    swatch_colors, pattern = _extract_swatches_and_pattern(g, stripe_row)
    templates = _extract_templates(g, stripe_row)
    obstacles = _extract_obstacles(g, stripe_row)
    settled = _settle_templates(templates, stripe_row, obstacles, g.shape)
    valid = sorted((t for t in settled if int(t["top"]) == stripe_row - 3), key=lambda t: int(t["left"]))
    assert len(valid) == len(swatch_colors) == 3
    return swatch_colors, pattern, templates, valid


def main(grid: np.ndarray) -> np.ndarray:
    """Drop floating 3x3 grey templates onto a stripe, keep only the grounded ones, and use them to expand a lower color index grid.

    Input structure:
    - The grid has a black background, one full-width yellow baseline row, four floating grey 3x3 template sprites above it, and one or two orange obstacle pixels.
    - Below the stripe, three 2x2 color swatches appear from left to right, with a dense color index grid on the right that uses only those swatch colors.
    - Each grey sprite is a connected 3x3 mask with internal black holes.

    Rule:
    - Treat every grey sprite as one rigid template and let it fall straight down, processing lower sprites first.
    - A template stops when one of its filled cells would hit the yellow stripe, an orange obstacle, or a previously settled template.
    - Keep only the templates whose 3x3 bounding box ends directly above the stripe; sort those grounded templates left to right and pair them with the swatches left to right.
    - Replace every pixel of the lower index grid with the paired 3x3 template, recoloring filled template cells to that pixel color while leaving template holes black.

    Key insight:
    - The pairing is determined only after simulating gravity and discarding the blocked template, so the extra grey sprite is a distractor rather than part of the legend.
    """
    swatch_colors, pattern, _templates, valid = _solve(grid)
    mapping = {int(swatch_colors[i]): np.asarray(valid[i]["mask"], dtype=bool) for i in range(3)}
    return _expand_pattern(pattern, mapping)


def _split_slack(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _draw_template(grid: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    patch = np.zeros((3, 3), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = GREY
    grid[top : top + 3, left : left + 3] = patch


def _make_pattern(height: int, width: int) -> np.ndarray:
    colors = SWATCH_COLORS
    for _ in range(100):
        mode = random.choice(["iid", "cycle", "bands"])
        pattern = np.zeros((height, width), dtype=int)
        if mode == "iid":
            pattern = np.array([[random.choice(colors) for _ in range(width)] for _ in range(height)], dtype=int)
        elif mode == "cycle":
            base = random.sample(colors, len(colors))
            phase = random.randrange(len(colors))
            for r in range(height):
                for c in range(width):
                    pattern[r, c] = base[(phase + r + 2 * c) % len(colors)]
        else:
            row_offsets = [random.randrange(len(colors)) for _ in range(height)]
            perm = random.sample(colors, len(colors))
            for r in range(height):
                for c in range(width):
                    pattern[r, c] = perm[(row_offsets[r] + c) % len(colors)]

        if set(map(int, np.unique(pattern))) == set(colors):
            return pattern
    raise RuntimeError("failed to sample a pattern using all three colors")


def _naive_output_without_filter(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    stripe_row = int([r for r in range(g.shape[0]) if np.all(g[r] == YELLOW)][0])
    swatch_colors, pattern = _extract_swatches_and_pattern(g, stripe_row)
    templates = sorted(_extract_templates(g, stripe_row), key=lambda t: int(t["left"]))
    mapping = {int(swatch_colors[i]): np.asarray(templates[i]["mask"], dtype=bool) for i in range(3)}
    return _expand_pattern(pattern, mapping)


def _instance_is_semantic(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    stripe_row = int([r for r in range(g.shape[0]) if np.all(g[r] == YELLOW)][0])
    swatch_colors, pattern, templates, valid = _solve(g)
    if set(map(int, np.unique(pattern))) != set(swatch_colors):
        return False
    if len({int(t["top"]) for t in templates}) < 3:
        return False
    if np.array_equal(main(g), _naive_output_without_filter(g)):
        return False
    grounded_without_obstacles = [
        t
        for t in _settle_templates(templates, stripe_row, set(), g.shape)
        if int(t["top"]) == stripe_row - 3
    ]
    if len(grounded_without_obstacles) != 4:
        return False
    return len(valid) == 3


def _gen_attempt() -> np.ndarray | None:
    pattern_h = random.randint(2, 4)
    pattern_w = random.randint(2, 4)

    swatch_left = random.randint(1, 2)
    gap_swatches_to_pattern = random.randint(2, 4)
    right_margin = random.randint(1, 2)
    min_width = max(18, swatch_left + 8 + gap_swatches_to_pattern + pattern_w + right_margin)
    if min_width > 24:
        return None
    width = random.randint(min_width, 24)

    stripe_row = random.randint(8, 11)
    swatch_top = stripe_row + random.randint(1, 2)
    pattern_top = swatch_top + random.choice([0, 1])
    bottom_margin = random.randint(1, 2)
    height = max(swatch_top + 2, pattern_top + pattern_h) + bottom_margin
    if height > 24:
        return None

    pattern_left = width - right_margin - pattern_w
    if pattern_left - (swatch_left + 8) < gap_swatches_to_pattern:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[stripe_row, :] = YELLOW

    slack = _split_slack(width - 15, 5)
    lefts = [slack[0]]
    for i in range(1, 4):
        lefts.append(lefts[-1] + 3 + 1 + slack[i])

    masks = [mask.copy() for mask in random.sample(TEMPLATE_POOL, 4)]
    blocked_index = random.choices([0, 1, 2], weights=[1, 3, 3], k=1)[0]
    blocked_final_top = random.randint(2, stripe_row - 5)
    final_tops = [stripe_row - 3] * 4
    final_tops[blocked_index] = blocked_final_top

    init_tops = None
    for _ in range(80):
        candidate = [random.randint(0, ft - 1) for ft in final_tops]
        if len(set(candidate)) >= 3:
            init_tops = candidate
            break
    if init_tops is None:
        return None

    for i in range(4):
        _draw_template(grid, init_tops[i], lefts[i], masks[i])

    support_options = [(2, c) for c in range(3) if masks[blocked_index][2, c]]
    if not support_options:
        support_options = [tuple(map(int, coord)) for coord in np.argwhere(masks[blocked_index])]
    rr, cc = random.choice(support_options)
    active_obstacle = (blocked_final_top + rr + 1, lefts[blocked_index] + cc)
    if active_obstacle[0] >= stripe_row or grid[active_obstacle] != BG:
        return None
    grid[active_obstacle] = ORANGE

    if random.random() < 0.6:
        free_cols = [c for c in range(width) if all(not (left <= c <= left + 2) for left in lefts)]
        random.shuffle(free_cols)
        for col in free_cols:
            row = random.randint(0, stripe_row - 2)
            if grid[row, col] == BG:
                grid[row, col] = ORANGE
                break

    for i, color in enumerate(SWATCH_COLORS):
        c0 = swatch_left + 3 * i
        grid[swatch_top : swatch_top + 2, c0 : c0 + 2] = int(color)

    pattern = _make_pattern(pattern_h, pattern_w)
    grid[pattern_top : pattern_top + pattern_h, pattern_left : pattern_left + pattern_w] = pattern

    try:
        if not _instance_is_semantic(grid):
            return None
        out = main(grid)
        if out.ndim != 2 or max(out.shape) > 30:
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
