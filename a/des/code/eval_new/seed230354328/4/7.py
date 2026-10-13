from __future__ import annotations

from collections import deque
import random

import numpy as np


STAMP_SIZES = ((3, 3), (3, 5), (5, 3), (5, 5))
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _enclosed_holes(template_mask: np.ndarray) -> np.ndarray:
    points = [(int(r), int(c)) for r, c in np.argwhere(template_mask)]
    r0, c0, r1, c1 = _bbox(points)
    sub = template_mask[r0 : r1 + 1, c0 : c1 + 1]
    bg_sub = ~sub
    seen = np.zeros_like(bg_sub, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    sh, sw = sub.shape

    for r in range(sh):
        for c in (0, sw - 1):
            if bg_sub[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(sw):
        for r in (0, sh - 1):
            if bg_sub[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, sh, sw):
            if bg_sub[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = bg_sub & ~seen
    out = np.zeros_like(template_mask, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = holes
    return out


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps = _components(g != bg)
    assert len(comps) >= 2, "expected one template and at least one stamp"

    areas = [len(comp) for comp in comps]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "template must be the unique largest object"
    template_idx = areas.index(max_area)

    template_mask = np.zeros((h, w), dtype=bool)
    for r, c in comps[template_idx]:
        template_mask[r, c] = True

    holes = _enclosed_holes(template_mask)
    out = np.full_like(g, bg)
    out[template_mask] = g[template_mask]

    color_positions: dict[int, list[tuple[int, int]]] = {}
    for r, c in comps[template_idx]:
        color = int(g[r, c])
        color_positions.setdefault(color, []).append((r, c))
    anchors = {color: pts[0] for color, pts in color_positions.items() if len(pts) == 1}

    for idx, comp in enumerate(comps):
        if idx == template_idx:
            continue

        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        ph, pw = patch.shape
        assert ph % 2 == 1 and pw % 2 == 1, "stamp bbox must have a unique center"
        assert 3 <= ph <= 5 and 3 <= pw <= 5, "stamp bbox must be between 3 and 5 cells per side"

        center_r, center_c = ph // 2, pw // 2
        anchor_color = int(patch[center_r, center_c])
        assert anchor_color != bg, "stamp center must be non-background"
        assert anchor_color in anchors, "stamp center color must map to a unique template anchor"
        anchor_r, anchor_c = anchors[anchor_color]

        for pr in range(ph):
            for pc in range(pw):
                color = int(patch[pr, pc])
                if color == bg:
                    continue
                tr = anchor_r + pr - center_r
                tc = anchor_c + pc - center_c
                if 0 <= tr < h and 0 <= tc < w and holes[tr, tc] and out[tr, tc] == bg:
                    out[tr, tc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each small motif onto the large multicolor template at the uniquely color-matched anchor.

    Input structure:
    - bg = most common color.
    - One large connected multicolor template object occupies most of the grid and contains enclosed bg holes.
    - Several smaller stamp objects lie outside the template; each stamp has an odd-sized 3..5 cell bbox, a non-bg center color, and other cells in one additional color or bg.

    Rule:
    - Treat the unique largest non-bg connected component as the template and the remaining components as stamps.
    - For each stamp, use its center pixel color to find the unique template pixel of the same color, then align the stamp center to that anchor.
    - Copy stamp non-bg pixels only into enclosed template-hole cells that are currently bg; keep existing template pixels unchanged and erase the original outside stamps.

    Key insight: The center color is only a placement key for locating the anchor, while the rest of the stamp provides the paint that fills the template's missing cells.
    """
    return _solve_grid(grid)


def _path_cells(
    start: tuple[int, int],
    target: tuple[int, int],
    row_first: bool,
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = target
    pts = [(r, c)]
    if row_first:
        step = 1 if tr > r else -1
        while r != tr:
            r += step
            pts.append((r, c))
        if tc != c:
            step = 1 if tc > c else -1
            while c != tc:
                c += step
                pts.append((r, c))
    else:
        if tc != c:
            step = 1 if tc > c else -1
            while c != tc:
                c += step
                pts.append((r, c))
        if tr != r:
            step = 1 if tr > r else -1
            while r != tr:
                r += step
                pts.append((r, c))
    return pts


def _make_stamp(
    h: int,
    w: int,
    anchor_color: int,
    fill_color: int,
    bg: int,
    rng: random.Random,
) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    center = (h // 2, w // 2)

    targets = [
        (0, rng.randrange(w)),
        (h - 1, rng.randrange(w)),
        (rng.randrange(h), 0),
        (rng.randrange(h), w - 1),
    ]
    fill_cells: set[tuple[int, int]] = set()
    all_cells: set[tuple[int, int]] = {center}
    for target in targets:
        for cell in _path_cells(center, target, row_first=bool(rng.randrange(2))):
            all_cells.add(cell)
            if cell != center:
                fill_cells.add(cell)

    extra = rng.randint(0, max(2, (h * w) // 3))
    while extra > 0:
        frontier = set()
        for r, c in all_cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in all_cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cell = rng.choice(tuple(frontier))
        all_cells.add(cell)
        fill_cells.add(cell)
        extra -= 1

    for r, c in fill_cells:
        patch[r, c] = fill_color
    patch[center] = anchor_color
    return patch


def _make_template_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    bite_count = rng.randint(0, 2)
    for _ in range(bite_count):
        edge = rng.choice(["top", "bottom", "left", "right"])
        if edge in ("top", "bottom"):
            bite_h = rng.randint(1, min(3, h - 4))
            bite_w = rng.randint(2, min(6, w - 3))
            c0 = rng.randint(1, w - bite_w - 1)
            if edge == "top":
                mask[:bite_h, c0 : c0 + bite_w] = False
            else:
                mask[h - bite_h :, c0 : c0 + bite_w] = False
        else:
            bite_h = rng.randint(2, min(6, h - 3))
            bite_w = rng.randint(1, min(3, w - 4))
            r0 = rng.randint(1, h - bite_h - 1)
            if edge == "left":
                mask[r0 : r0 + bite_h, :bite_w] = False
            else:
                mask[r0 : r0 + bite_h, w - bite_w :] = False
    return mask


def _paint_template(
    mask: np.ndarray,
    bg: int,
    base_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h, w), bg, dtype=int)
    mode = rng.choice(["rows", "cols", "checker", "diag", "blocks"])

    if mode == "rows":
        band = rng.randint(1, 3)
        for r in range(h):
            color = base_colors[(r // band) % len(base_colors)]
            patch[r, mask[r]] = color
    elif mode == "cols":
        band = rng.randint(1, 3)
        for c in range(w):
            color = base_colors[(c // band) % len(base_colors)]
            patch[mask[:, c], c] = color
    elif mode == "checker":
        stride_r = rng.randint(1, 2)
        stride_c = rng.randint(1, 2)
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    idx = ((r // stride_r) + (c // stride_c)) % len(base_colors)
                    patch[r, c] = base_colors[idx]
    elif mode == "diag":
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    patch[r, c] = base_colors[(r - c) % len(base_colors)]
    else:
        split_r = rng.randint(1, h - 2)
        split_c = rng.randint(1, w - 2)
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                idx = (r >= split_r) + 2 * (c >= split_c)
                patch[r, c] = base_colors[idx % len(base_colors)]

    return patch


def _candidate_centers(mask: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    sh, sw = shape
    hh, hw = sh // 2, sw // 2
    centers: list[tuple[int, int]] = []
    for r in range(hh + 1, h - hh - 1):
        for c in range(hw + 1, w - hw - 1):
            shell = mask[r - hh - 1 : r + hh + 2, c - hw - 1 : c + hw + 2]
            if np.all(shell):
                centers.append((r, c))
    return centers


def _expected_output(
    grid_shape: tuple[int, int],
    bg: int,
    template_patch: np.ndarray,
    template_pos: tuple[int, int],
    stamp_specs: list[dict],
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    tr0, tc0 = template_pos
    th, tw = template_patch.shape
    mask = template_patch != bg
    out[tr0 : tr0 + th, tc0 : tc0 + tw][mask] = template_patch[mask]

    for spec in stamp_specs:
        patch = spec["patch"]
        anchor_r, anchor_c = spec["anchor_local"]
        center_r, center_c = patch.shape[0] // 2, patch.shape[1] // 2
        global_anchor = (tr0 + anchor_r, tc0 + anchor_c)
        for pr in range(patch.shape[0]):
            for pc in range(patch.shape[1]):
                color = int(patch[pr, pc])
                if color == bg:
                    continue
                rr = global_anchor[0] + pr - center_r
                cc = global_anchor[1] + pc - center_c
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == bg:
                    out[rr, cc] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)

    grid_h = rng.randint(20, 28)
    grid_w = rng.randint(20, 28)
    stamp_count = rng.randint(2, 4)
    base_count = rng.randint(2, 3)
    if base_count + stamp_count >= len(colors):
        return None

    template_h = rng.randint(10, min(15, grid_h - 6))
    template_w = rng.randint(10, min(17, grid_w - 6))

    base_colors = colors[:base_count]
    anchor_colors = colors[base_count : base_count + stamp_count]
    paint_pool = colors[base_count + stamp_count :]
    if not paint_pool:
        paint_pool = base_colors[:]

    template_mask = _make_template_mask(template_h, template_w, rng)
    if len(_components(template_mask)) != 1:
        return None

    template_patch = _paint_template(template_mask, bg, base_colors, rng)

    stamp_shapes = [rng.choice(STAMP_SIZES) for _ in range(stamp_count)]
    order = sorted(range(stamp_count), key=lambda i: stamp_shapes[i][0] * stamp_shapes[i][1], reverse=True)
    chosen_centers: list[tuple[int, int] | None] = [None] * stamp_count
    used_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        sh, sw = stamp_shapes[idx]
        candidates = _candidate_centers(template_mask, (sh, sw))
        rng.shuffle(candidates)
        found = None
        hh, hw = sh // 2, sw // 2
        for r, c in candidates:
            rect = (r - hh - 1, c - hw - 1, r + hh + 1, c + hw + 1)
            if any(_rect_overlap(rect, other) for other in used_rects):
                continue
            found = (r, c)
            used_rects.append(rect)
            break
        if found is None:
            return None
        chosen_centers[idx] = found

    stamp_specs: list[dict] = []
    for idx in range(stamp_count):
        sh, sw = stamp_shapes[idx]
        fill_color_choices = [c for c in (paint_pool + base_colors) if c != anchor_colors[idx]]
        fill_color = rng.choice(fill_color_choices)
        patch = _make_stamp(sh, sw, anchor_colors[idx], fill_color, bg, rng)
        anchor_r, anchor_c = chosen_centers[idx]
        center_r, center_c = patch.shape[0] // 2, patch.shape[1] // 2
        template_patch[anchor_r, anchor_c] = anchor_colors[idx]
        fill_pixels = 0
        for pr in range(sh):
            for pc in range(sw):
                if (pr, pc) == (center_r, center_c):
                    continue
                if int(patch[pr, pc]) == bg:
                    continue
                rr = anchor_r + pr - center_r
                cc = anchor_c + pc - center_c
                template_patch[rr, cc] = bg
                fill_pixels += 1
        if fill_pixels < 3:
            return None
        stamp_specs.append(
            {
                "patch": patch,
                "anchor_local": (anchor_r, anchor_c),
            }
        )

    template_fg = template_patch != bg
    if len(_components(template_fg)) != 1:
        return None

    hole_mask = _enclosed_holes(template_fg)
    expected_holes = sum(int(np.count_nonzero(spec["patch"] != bg)) - 1 for spec in stamp_specs)
    if int(hole_mask.sum()) != expected_holes:
        return None

    template_r0 = rng.randint(1, grid_h - template_h - 1)
    template_c0 = rng.randint(1, grid_w - template_w - 1)
    template_rect = (template_r0, template_c0, template_r0 + template_h - 1, template_c0 + template_w - 1)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[
        template_r0 : template_r0 + template_h,
        template_c0 : template_c0 + template_w,
    ][template_fg] = template_patch[template_fg]

    placed_rects: list[tuple[int, int, int, int]] = []
    for spec in stamp_specs:
        patch = spec["patch"]
        ph, pw = patch.shape
        pos = None
        for _ in range(250):
            r0 = rng.randint(1, grid_h - ph - 1)
            c0 = rng.randint(1, grid_w - pw - 1)
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if _rect_overlap(rect, template_rect):
                continue
            if any(_rect_overlap(rect, other) for other in placed_rects):
                continue
            padded = (r0 - 1, c0 - 1, r0 + ph, c0 + pw)
            tpadded = (template_rect[0] - 1, template_rect[1] - 1, template_rect[2] + 1, template_rect[3] + 1)
            if _rect_overlap(padded, tpadded):
                continue
            if any(
                _rect_overlap(
                    padded,
                    (other[0] - 1, other[1] - 1, other[2] + 1, other[3] + 1),
                )
                for other in placed_rects
            ):
                continue
            pos = (r0, c0)
            placed_rects.append(rect)
            break
        if pos is None:
            return None
        spec["placement"] = pos
        stamp_mask = patch != bg
        r0, c0 = pos
        grid[r0 : r0 + ph, c0 : c0 + pw][stamp_mask] = patch[stamp_mask]

    expected = _expected_output(
        grid.shape,
        bg,
        template_patch,
        (template_r0, template_c0),
        stamp_specs,
    )
    solved = _solve_grid(grid)
    if not np.array_equal(solved, expected):
        return None
    if int(np.count_nonzero(expected != grid)) < stamp_count * 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid template-and-stamps input")
