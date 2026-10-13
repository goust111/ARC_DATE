from __future__ import annotations

import numpy as np


BG = 0
FRAME = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_hollow_rect_frame(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int], area: int) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or area != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return False
    return not np.any(patch[1:-1, 1:-1] == color)


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components_of_color(grid, color):
            box = _bbox(coords)
            if _is_hollow_rect_frame(grid, color, box, len(coords)):
                frames.append(
                    {
                        "color": color,
                        "bbox": box,
                    }
                )
    return frames


def _find_template_stack(grid: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    tiny = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        center = int(grid[r0 + 1, c0 + 1])
        if center in (bg, frame["color"]):
            continue
        tiny.append({**frame, "center": center})

    best_run: list[dict] = []
    best_score: tuple[int, int, int] | None = None
    groups: dict[tuple[int, int], list[dict]] = {}
    for frame in tiny:
        key = (int(frame["color"]), int(frame["bbox"][1]))
        groups.setdefault(key, []).append(frame)

    for items in groups.values():
        items.sort(key=lambda item: int(item["bbox"][0]))
        run: list[dict] = []
        for frame in items:
            if not run:
                run = [frame]
                continue
            prev = run[-1]
            pr0, pc0, pr1, _pc1 = prev["bbox"]
            r0, c0, _r1, _c1 = frame["bbox"]
            gap_ok = r0 == pr0 + 4 and c0 == pc0 and np.all(grid[pr1 + 1, pc0 : pc0 + 3] == bg)
            if gap_ok:
                run.append(frame)
            else:
                score = (len(run), run[-1]["bbox"][2], run[-1]["bbox"][3])
                if best_score is None or score > best_score:
                    best_score = score
                    best_run = run[:]
                run = [frame]
        if run:
            score = (len(run), run[-1]["bbox"][2], run[-1]["bbox"][3])
            if best_score is None or score > best_score:
                best_score = score
                best_run = run[:]

    assert best_run, "expected at least one 3x3 template frame"
    return sorted(best_run, key=lambda item: int(item["bbox"][0]))


def _paint_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed rectangle border by looking up the template for its number of interior seed colors.

    Input structure:
    - The background is black.
    - Several grey one-pixel-thick hollow rectangles enclose black interiors that contain scattered colored seed pixels.
    - Near the lower-right corner, a vertical list of 3x3 grey frame templates appears, with one black row between templates and a distinct colored center in each template.

    Rule:
    - Read the template list from top to bottom so template K provides the replacement border color for regions containing K distinct seed colors.
    - For every non-template grey rectangle, count the distinct interior colors while ignoring black and grey.
    - Repaint that rectangle's full border with the center color of template K, leaving the interior and the template list unchanged.

    Key insight:
    - The relevant hidden property is the number of distinct seed colors inside each enclosure, and the bottom-right template stack is a visual lookup table from that count to the new border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    template_stack = _find_template_stack(g, bg, frames)
    border_color = int(template_stack[0]["color"])
    template_boxes = {tuple(frame["bbox"]) for frame in template_stack}
    template_colors = [int(frame["center"]) for frame in template_stack]

    out = g.copy()
    for frame in frames:
        if int(frame["color"]) != border_color or tuple(frame["bbox"]) in template_boxes:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = {
            int(color)
            for color in np.unique(interior)
            if int(color) not in (bg, border_color)
        }
        k = len(seeds)
        assert 1 <= k <= len(template_colors), "region seed-count must match a template"
        _paint_frame(out, frame["bbox"], template_colors[k - 1])
    return out


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(mask.shape[0] - 1, r1)
    cc1 = min(mask.shape[1] - 1, c1)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _area_free(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(mask.shape[0] - 1, r1)
    cc1 = min(mask.shape[1] - 1, c1)
    return not np.any(mask[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _template_layout(rng: np.random.Generator, height: int, width: int, count: int) -> tuple[int, int]:
    stack_h = 4 * count - 1
    bottom_margin = int(rng.integers(0, min(3, height - stack_h + 1)))
    right_margin = int(rng.integers(0, min(3, width - 2)))
    r0 = height - stack_h - bottom_margin
    c0 = width - 3 - right_margin
    return r0, c0


def _region_size_options(height: int, width: int, k: int) -> list[tuple[int, int]]:
    opts = []
    for h in range(4, min(10, height - 1) + 1):
        for w in range(4, min(11, width - 1) + 1):
            if (h - 2) * (w - 2) >= k + 1:
                opts.append((h, w))
    return opts


def _place_seeds(
    rng: np.random.Generator,
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    k: int,
    force_duplicate: bool,
) -> None:
    r0, c0, r1, c1 = bbox
    slots = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    rng.shuffle(slots)

    colors = [int(x) for x in rng.choice(PALETTE, size=k, replace=False)]
    for color, (r, c) in zip(colors, slots[:k]):
        grid[r, c] = color

    remaining = slots[k:]
    if not remaining:
        return

    max_extra = min(len(remaining), k + 1)
    if force_duplicate:
        extra_count = int(rng.integers(1, max_extra + 1))
    else:
        extra_count = int(rng.integers(0, max_extra + 1))
    for idx in range(extra_count):
        r, c = remaining[idx]
        grid[r, c] = int(rng.choice(colors))


def _draw_template_stack(grid: np.ndarray, r0: int, c0: int, template_colors: list[int]) -> None:
    for idx, color in enumerate(template_colors):
        top = r0 + 4 * idx
        grid[top, c0 : c0 + 3] = FRAME
        grid[top + 2, c0 : c0 + 3] = FRAME
        grid[top : top + 3, c0] = FRAME
        grid[top : top + 3, c0 + 2] = FRAME
        grid[top + 1, c0 + 1] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    template_count = int(rng.integers(2, 5))
    region_ks = list(range(1, template_count + 1))
    extra_regions = int(rng.integers(0, 3))
    region_ks.extend(int(rng.integers(1, template_count + 1)) for _ in range(extra_regions))
    rng.shuffle(region_ks)

    stack_h = 4 * template_count - 1
    min_height = max(14, stack_h + 2, 9 + len(region_ks))
    min_width = max(16, 10 + 2 * len(region_ks))
    if min_height > 30 or min_width > 30:
        return None
    height = int(rng.integers(min_height, 31))
    width = int(rng.integers(min_width, 31))

    inp = np.full((height, width), BG, dtype=int)
    template_colors = [int(x) for x in rng.choice(PALETTE, size=template_count, replace=False)]
    tpl_r0, tpl_c0 = _template_layout(rng, height, width, template_count)
    _draw_template_stack(inp, tpl_r0, tpl_c0, template_colors)

    used = np.zeros((height, width), dtype=bool)
    _reserve(used, tpl_r0 - 1, tpl_c0 - 1, tpl_r0 + stack_h, tpl_c0 + 3)

    duplicate_region = int(rng.integers(0, len(region_ks)))
    region_boxes: list[tuple[tuple[int, int, int, int], int]] = []

    for idx, k in enumerate(region_ks):
        size_options = _region_size_options(height, width, k)
        if not size_options:
            return None
        placed = False
        for _ in range(250):
            rh, rw = size_options[int(rng.integers(0, len(size_options)))]
            if height - rh <= 1 or width - rw <= 1:
                continue
            r0 = int(rng.integers(1, height - rh))
            c0 = int(rng.integers(1, width - rw))
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            if not _area_free(used, r0 - 1, c0 - 1, r1 + 1, c1 + 1):
                continue
            _reserve(used, r0 - 1, c0 - 1, r1 + 1, c1 + 1)
            _paint_frame(inp, (r0, c0, r1, c1), FRAME)
            placed = True
            region_boxes.append(((r0, c0, r1, c1), k))
            _place_seeds(rng, inp, (r0, c0, r1, c1), k, force_duplicate=idx == duplicate_region)
            break
        if not placed:
            return None

    out = inp.copy()
    for bbox, k in region_boxes:
        _paint_frame(out, bbox, template_colors[k - 1])

    if _most_common_color(inp) != BG:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
