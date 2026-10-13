import itertools
import random

import numpy as np


GREY = 5
GEN_RETRIES = 300


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def bbox_of(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = bbox_of(coords)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return rel, int(r1 - r0 + 1), int(c1 - c0 + 1)


def coords_to_mask(coords: tuple[tuple[int, int], ...], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def perimeter_coords(h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r in range(h):
        for c in range(w):
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                out.add((r, c))
    return out


def is_rect_frame_component(coords: list[tuple[int, int]]) -> bool:
    rel, h, w = normalize_coords(coords)
    if h < 3 or w < 3:
        return False
    return set(rel) == perimeter_coords(h, w)


def find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in components_mask(grid == color):
            if is_rect_frame_component(comp):
                candidates.setdefault(color, []).append(bbox_of(comp))
    assert candidates, "no rectangular frames found"
    frame_color = max(
        candidates,
        key=lambda color: (
            len(candidates[color]),
            sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in candidates[color]),
        ),
    )
    frames = sorted(candidates[frame_color])
    return int(frame_color), frames


def extract_templates(
    grid: np.ndarray, bg: int, frame_color: int, frames: list[tuple[int, int, int, int]]
) -> dict[int, dict[str, object]]:
    blocked = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in frames:
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    templates: dict[int, dict[str, object]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame_color):
            continue
        comps = components_mask((grid == color) & (~blocked))
        if not comps:
            continue
        assert len(comps) == 1, f"expected exactly one template component for color {color}"
        rel, h, w = normalize_coords(comps[0])
        templates[color] = {"coords": rel, "h": h, "w": w}
    assert templates, "no templates found"
    return templates


def matching_offsets(
    shape: tuple[tuple[int, int], ...],
    shape_h: int,
    shape_w: int,
    interior_h: int,
    interior_w: int,
    fragments: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    shape_set = set(shape)
    hits: list[tuple[int, int]] = []
    for off_r in range(interior_h - shape_h + 1):
        for off_c in range(interior_w - shape_w + 1):
            ok = True
            for fr, fc in fragments:
                if (fr - off_r, fc - off_c) not in shape_set:
                    ok = False
                    break
            if ok:
                hits.append((off_r, off_c))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each framed region's template from colored fragments, then stamp the full template there.

    Input structure:
    - bg is the dominant color.
    - Several grey rectangular frames contain 2-3 colored fragment pixels on bg.
    - Along the bottom is a row of monochrome template sprites, one unique color per template.

    Rule:
    - Detect the grey rectangular frames and extract the bottom template sprites.
    - For each frame, use fragment color to choose the template.
    - Find the unique translation where the observed fragment positions are a subset of the template.
    - Paint the entire translated template inside the frame. Keep borders and the template row unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color, frames = find_frames(g, bg)
    templates = extract_templates(g, bg, frame_color, frames)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        pts = np.argwhere((inner != bg) & (inner != frame_color))
        assert pts.size > 0, "frame interior must contain colored fragments"
        colors = {int(inner[r, c]) for r, c in pts}
        assert len(colors) == 1, "all fragments in a frame must use one template color"
        color = colors.pop()
        assert color in templates, f"missing template for color {color}"

        fragments = {(int(r), int(c)) for r, c in pts}
        tpl = templates[color]
        shape = tpl["coords"]
        shape_h = int(tpl["h"])
        shape_w = int(tpl["w"])
        offsets = matching_offsets(shape, shape_h, shape_w, inner.shape[0], inner.shape[1], fragments)
        assert len(offsets) == 1, f"expected a unique placement for color {color}, found {offsets}"

        off_r, off_c = offsets[0]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        for dr, dc in shape:
            out[r0 + 1 + off_r + dr, c0 + 1 + off_c + dc] = color

    return out


def random_shape(rng: random.Random) -> tuple[tuple[tuple[int, int], ...], int, int]:
    for _ in range(200):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(2, 5)
        max_area = min(box_h * box_w - 1, 8)
        min_area = max(4, min(max_area, box_h + box_w - 2))
        if min_area > max_area:
            continue
        target = rng.randint(min_area, max_area)
        cells = {(rng.randrange(box_h), rng.randrange(box_w))}
        while len(cells) < target:
            frontier = []
            used = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells and (nr, nc) not in used:
                        frontier.append((nr, nc))
                        used.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue

        rel, h, w = normalize_coords(tuple(sorted(cells)))
        if h < 2 or w < 2 or len(rel) < 4:
            continue
        if len(rel) == h * w:
            continue
        if set(rel) == perimeter_coords(h, w):
            continue

        mask = coords_to_mask(rel, h, w)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, rng.randrange(4))
        rel2, h2, w2 = normalize_coords(tuple((int(r), int(c)) for r, c in np.argwhere(mask)))
        if len(rel2) == h2 * w2:
            continue
        if set(rel2) == perimeter_coords(h2, w2):
            continue
        return rel2, h2, w2
    raise RuntimeError("failed to create a non-degenerate template shape")


def spread_score(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    pairwise = 0
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            pairwise += abs(cells[i][0] - cells[j][0]) + abs(cells[i][1] - cells[j][1])
    return (
        pairwise,
        max(rows) - min(rows),
        max(cols) - min(cols),
    )


def choose_fragments(
    rng: random.Random,
    shape: tuple[tuple[int, int], ...],
    shape_h: int,
    shape_w: int,
    interior_h: int,
    interior_w: int,
    off_r: int,
    off_c: int,
) -> tuple[tuple[int, int], ...] | None:
    shifted = tuple((off_r + r, off_c + c) for r, c in shape)
    valid: dict[int, list[tuple[tuple[int, int], ...]]] = {2: [], 3: []}
    for k in (2, 3):
        if len(shape) < k:
            continue
        for combo in itertools.combinations(shifted, k):
            offsets = matching_offsets(shape, shape_h, shape_w, interior_h, interior_w, set(combo))
            if len(offsets) == 1 and offsets[0] == (off_r, off_c):
                valid[k].append(tuple(combo))

    ks = [k for k in (2, 3) if valid[k]]
    if not ks:
        return None
    if len(ks) == 2:
        want_k = 2 if rng.random() < 0.5 else 3
    else:
        want_k = ks[0]
    combos = sorted(valid[want_k], key=spread_score, reverse=True)
    keep = max(1, min(len(combos), 4))
    return rng.choice(combos[:keep])


def make_region(
    rng: random.Random, template: dict[str, object]
) -> dict[str, object] | None:
    shape = template["coords"]
    shape_h = int(template["h"])
    shape_w = int(template["w"])
    color = int(template["color"])

    for _ in range(80):
        extra_h = 0 if rng.random() < 0.35 else rng.randint(1, 2)
        extra_w = 0 if rng.random() < 0.35 else rng.randint(1, 2)
        if extra_h == 0 and extra_w == 0 and rng.random() < 0.8:
            if rng.random() < 0.5:
                extra_h = rng.randint(1, 2)
            else:
                extra_w = rng.randint(1, 2)

        inner_h = shape_h + extra_h
        inner_w = shape_w + extra_w
        off_r = rng.randint(0, extra_h)
        off_c = rng.randint(0, extra_w)
        fragments = choose_fragments(rng, shape, shape_h, shape_w, inner_h, inner_w, off_r, off_c)
        if fragments is None:
            continue
        return {
            "color": color,
            "shape": shape,
            "shape_h": shape_h,
            "shape_w": shape_w,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "off_r": off_r,
            "off_c": off_c,
            "fragments": fragments,
            "frame_h": inner_h + 2,
            "frame_w": inner_w + 2,
        }
    return None


def layout_row(
    rng: random.Random, sizes: list[tuple[int, int]], gap_lo: int, gap_hi: int
) -> tuple[list[tuple[int, int]], int, int]:
    x = 0
    positions: list[tuple[int, int]] = []
    max_h = max(h for h, _ in sizes)
    for i, (h, w) in enumerate(sizes):
        y = rng.randint(0, max_h - h)
        positions.append((y, x))
        x += w
        if i != len(sizes) - 1:
            x += rng.randint(gap_lo, gap_hi)
    return positions, max_h, x


def layout_frames(
    rng: random.Random, sizes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int, int]:
    n = len(sizes)
    order = list(range(n))
    rng.shuffle(order)
    if n <= 2:
        row_count = 1
    elif n <= 4:
        row_count = 2
    else:
        row_count = 3 if rng.random() < 0.7 else 2

    counts = [n // row_count] * row_count
    for i in range(n % row_count):
        counts[i] += 1
    rows = []
    start = 0
    for count in counts:
        rows.append(order[start : start + count])
        start += count

    positions = [(-1, -1)] * n
    y = 0
    content_w = 0
    for row_idx, row in enumerate(rows):
        row_sizes = [sizes[i] for i in row]
        row_pos, row_h, row_w = layout_row(rng, row_sizes, 1, 3)
        for i, idx in enumerate(row):
            ry, rx = row_pos[i]
            positions[idx] = (y + ry, rx)
        content_w = max(content_w, row_w)
        y += row_h
        if row_idx != len(rows) - 1:
            y += rng.randint(1, 3)
    return positions, y, content_w


def paint_shape(grid: np.ndarray, top: int, left: int, color: int, shape: tuple[tuple[int, int], ...]) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def build_task(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    bg_choices = [c for c in range(10) if c != GREY]
    bg = rng.choice(bg_choices)

    num_templates = rng.randint(2, 4)
    colors = rng.sample([c for c in range(10) if c not in (bg, GREY)], num_templates)
    templates = []
    for color in colors:
        shape, h, w = random_shape(rng)
        templates.append({"color": color, "coords": shape, "h": h, "w": w})

    num_regions = rng.randint(max(2, num_templates), min(num_templates + 1, 5))
    template_ids = list(range(num_templates))
    region_template_ids = template_ids[:]
    while len(region_template_ids) < num_regions:
        region_template_ids.append(rng.randrange(num_templates))
    rng.shuffle(region_template_ids)

    regions = []
    for tpl_idx in region_template_ids:
        region = make_region(rng, templates[tpl_idx])
        if region is None:
            return None
        regions.append(region)

    frame_sizes = [(int(r["frame_h"]), int(r["frame_w"])) for r in regions]
    frame_pos_rel, top_h, top_w = layout_frames(rng, frame_sizes)

    template_order = list(range(num_templates))
    rng.shuffle(template_order)
    template_sizes = [(int(templates[i]["h"]), int(templates[i]["w"])) for i in template_order]
    template_pos_rel, tpl_h, tpl_w = layout_row(rng, template_sizes, 2, 3)

    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    top_margin = rng.randint(1, 3)
    between_gap = rng.randint(2, 3)
    bottom_margin = rng.randint(1, 2)
    extra_w = rng.randint(0, 2)

    canvas_w = max(top_w, tpl_w) + left_margin + right_margin + extra_w
    canvas_h = top_margin + top_h + between_gap + tpl_h + bottom_margin
    if canvas_h > 30 or canvas_w > 30 or canvas_h * canvas_w > 900:
        return None

    tpl_y = canvas_h - bottom_margin - tpl_h
    max_top_x = canvas_w - right_margin - top_w
    max_tpl_x = canvas_w - right_margin - tpl_w
    if max_top_x < left_margin or max_tpl_x < left_margin:
        return None
    top_x = rng.randint(left_margin, max_top_x)
    tpl_x = rng.randint(left_margin, max_tpl_x)

    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    expected = grid.copy()

    for idx, region in enumerate(regions):
        y0 = top_margin + frame_pos_rel[idx][0]
        x0 = top_x + frame_pos_rel[idx][1]
        fh = int(region["frame_h"])
        fw = int(region["frame_w"])
        y1 = y0 + fh - 1
        x1 = x0 + fw - 1
        grid[y0 : y1 + 1, x0] = GREY
        grid[y0 : y1 + 1, x1] = GREY
        grid[y0, x0 : x1 + 1] = GREY
        grid[y1, x0 : x1 + 1] = GREY
        expected[y0 : y1 + 1, x0] = GREY
        expected[y0 : y1 + 1, x1] = GREY
        expected[y0, x0 : x1 + 1] = GREY
        expected[y1, x0 : x1 + 1] = GREY

        for fr, fc in region["fragments"]:
            grid[y0 + 1 + fr, x0 + 1 + fc] = int(region["color"])
        paint_shape(
            expected,
            y0 + 1 + int(region["off_r"]),
            x0 + 1 + int(region["off_c"]),
            int(region["color"]),
            region["shape"],
        )

    for idx, tpl_idx in enumerate(template_order):
        tpl = templates[tpl_idx]
        py = tpl_y + template_pos_rel[idx][0]
        px = tpl_x + template_pos_rel[idx][1]
        paint_shape(grid, py, px, int(tpl["color"]), tpl["coords"])
        paint_shape(expected, py, px, int(tpl["color"]), tpl["coords"])

    if get_bg(grid) != bg:
        return None
    if len(find_frames(grid, bg)[1]) != len(regions):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = build_task(rng)
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid input")
