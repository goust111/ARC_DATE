from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
MARKER_COLORS = (3, 4)
TEMPLATE_COLORS = (1, 2, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _region_classes(vertices: np.ndarray) -> list[list[str]]:
    rows = vertices.shape[0] - 1
    cols = vertices.shape[1] - 1
    out: list[list[str]] = []
    for i in range(rows):
        row: list[str] = []
        for j in range(cols):
            corners = [
                int(vertices[i, j]),
                int(vertices[i, j + 1]),
                int(vertices[i + 1, j]),
                int(vertices[i + 1, j + 1]),
            ]
            if len(set(corners)) == 1 and corners[0] not in (BG, SEP):
                row.append("valid")
            elif any(color in (BG, SEP) for color in corners):
                row.append("missing")
            else:
                row.append("mixed")
        out.append(row)
    return out


def _parse_objects(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], np.ndarray]:
    comps = _components(grid != BG)
    assert len(comps) >= 2, "expected a separator grid and a detached template"

    boxes = [(_bbox(comp), len(comp)) for comp in comps]
    boxes.sort(key=lambda item: item[1], reverse=True)
    grid_box = boxes[0][0]

    template_candidates: list[tuple[int, int, int, int]] = []
    for box, _area in boxes[1:]:
        patch = _crop(grid, box)
        if patch.size == 0:
            continue
        if np.all(patch != BG) and not np.any(patch == SEP):
            template_candidates.append(box)

    assert len(template_candidates) == 1, "expected exactly one detached multicolor template"
    template_box = template_candidates[0]
    template = _crop(grid, template_box)
    return grid_box, template_box, template


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grid_box, template_box, template = _parse_objects(g)
    gr0, gc0, gr1, gc1 = grid_box
    grid_crop = g[gr0 : gr1 + 1, gc0 : gc1 + 1]

    line_rows = [r for r in range(grid_crop.shape[0]) if np.all(grid_crop[r, :] != BG)]
    line_cols = [c for c in range(grid_crop.shape[1]) if np.all(grid_crop[:, c] != BG)]
    assert len(line_rows) >= 2 and len(line_cols) >= 2, "expected divider lines in both directions"
    assert all(b - a >= 2 for a, b in zip(line_rows, line_rows[1:])), "regions must have positive height"
    assert all(b - a >= 2 for a, b in zip(line_cols, line_cols[1:])), "regions must have positive width"

    region_rows = len(line_rows) - 1
    region_cols = len(line_cols) - 1
    rotated = np.rot90(template, k=-1)
    assert rotated.shape == (region_rows, region_cols), "rotated template must match region layout"

    valid_mask = np.zeros((region_rows, region_cols), dtype=bool)
    out = g.copy()
    classes: list[list[str]] = []
    for i in range(region_rows):
        class_row: list[str] = []
        for j in range(region_cols):
            corners = [
                int(grid_crop[line_rows[i], line_cols[j]]),
                int(grid_crop[line_rows[i], line_cols[j + 1]]),
                int(grid_crop[line_rows[i + 1], line_cols[j]]),
                int(grid_crop[line_rows[i + 1], line_cols[j + 1]]),
            ]
            valid = len(set(corners)) == 1 and corners[0] not in (BG, SEP)
            valid_mask[i, j] = valid
            class_row.append(
                "valid" if valid else ("missing" if any(v in (BG, SEP) for v in corners) else "mixed")
            )
            fill = int(rotated[i, j]) if valid else BG
            rr0 = gr0 + line_rows[i] + 1
            rr1 = gr0 + line_rows[i + 1]
            cc0 = gc0 + line_cols[j] + 1
            cc1 = gc0 + line_cols[j + 1]
            out[rr0:rr1, cc0:cc1] = fill
        classes.append(class_row)

    return {
        "output": out,
        "grid_box": grid_box,
        "template_box": template_box,
        "template": template,
        "rotated": rotated,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "valid_mask": valid_mask,
        "classes": classes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each divider-grid region with the 90-degree-clockwise-rotated template color only when its four corner markers agree.

    Input structure:
    - A black background contains one large separator grid made of one-pixel grey divider bars and colored corner markers at divider intersections.
    - The separator grid encloses an M-by-N array of black rectangular regions with one-pixel grey borders.
    - Away from that grid, a detached N-by-M multicolor template sits on the background with at least one pixel of padding.

    Rule:
    - Detect the detached template and rotate it 90 degrees clockwise so it becomes M-by-N.
    - Match each rotated template pixel to the corresponding region of the separator grid.
    - For a region, inspect the four intersection pixels at its corners: if all four are the same non-black, non-grey marker color, fill that region's interior with the matched template color; otherwise fill the region interior with black.
    - Keep the divider bars, the corner-marker pixels, and the original detached template unchanged.

    Key insight:
    - The template says what color each region wants after a geometric rotation, while the corner markers independently decide whether that proposal is allowed; the output requires both information sources.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _build_template(region_rows: int, region_cols: int) -> np.ndarray | None:
    template_h = region_cols
    template_w = region_rows
    palette_size = random.randint(3, min(5, len(TEMPLATE_COLORS)))
    palette = random.sample(list(TEMPLATE_COLORS), palette_size)
    template = np.array(
        [[random.choice(palette) for _ in range(template_w)] for _ in range(template_h)],
        dtype=int,
    )
    if len(set(map(int, template.flat))) < 3:
        return None

    rotated = np.rot90(template, k=-1)
    if len(set(map(int, rotated.flat))) < 3:
        return None

    if template_h == template_w:
        if np.array_equal(template, rotated):
            return None
        if np.array_equal(template, np.rot90(template, k=1)):
            return None
    return template


def _choose_valid_positions(region_rows: int, region_cols: int) -> list[tuple[int, int]] | None:
    classes = []
    for rr_mod in (0, 1):
        for cc_mod in (0, 1):
            cells = [
                (i, j)
                for i in range(region_rows)
                for j in range(region_cols)
                if i % 2 == rr_mod and j % 2 == cc_mod
            ]
            if len(cells) >= 3:
                classes.append(cells)
    if not classes:
        return None
    cells = random.choice(classes)
    max_pick = min(len(cells), max(3, region_rows * region_cols - 2), 5)
    count = random.randint(3, max_pick)
    random.shuffle(cells)
    return cells[:count]


def _build_vertices(region_rows: int, region_cols: int) -> np.ndarray | None:
    vertices = np.full((region_rows + 1, region_cols + 1), SEP, dtype=int)
    locked = np.zeros_like(vertices, dtype=bool)

    valid_regions = _choose_valid_positions(region_rows, region_cols)
    if valid_regions is None:
        return None

    random.shuffle(valid_regions)
    for idx, (i, j) in enumerate(valid_regions):
        color = MARKER_COLORS[idx % len(MARKER_COLORS)]
        if random.random() < 0.5:
            color = random.choice(MARKER_COLORS)
        vertices[i : i + 2, j : j + 2] = color
        locked[i : i + 2, j : j + 2] = True

    mixed_candidates = [
        (i, j)
        for i in range(region_rows)
        for j in range(region_cols)
        if not locked[i : i + 2, j : j + 2].any()
    ]
    if not mixed_candidates:
        return None

    random.shuffle(mixed_candidates)
    mixed_count = random.randint(1, min(3, len(mixed_candidates)))
    patterns = [
        np.array([[3, 4], [4, 3]], dtype=int),
        np.array([[4, 3], [3, 4]], dtype=int),
        np.array([[3, 3], [4, 4]], dtype=int),
        np.array([[4, 4], [3, 3]], dtype=int),
    ]
    for i, j in mixed_candidates[:mixed_count]:
        vertices[i : i + 2, j : j + 2] = random.choice(patterns)

    noise_pool = [(r, c) for r in range(vertices.shape[0]) for c in range(vertices.shape[1]) if not locked[r, c]]
    random.shuffle(noise_pool)
    for r, c in noise_pool[: random.randint(0, max(1, len(noise_pool) // 4))]:
        if random.random() < 0.65:
            vertices[r, c] = random.choice(MARKER_COLORS)

    classes = _region_classes(vertices)
    flat = [cls for row in classes for cls in row]
    if flat.count("valid") < 3 or len(flat) - flat.count("valid") < 2:
        return None
    if "mixed" not in flat or "missing" not in flat:
        return None
    return vertices


def _render_grid(
    region_rows: int,
    region_cols: int,
    row_heights: list[int],
    col_widths: list[int],
    vertices: np.ndarray,
) -> np.ndarray:
    h = sum(row_heights) + region_rows + 1
    w = sum(col_widths) + region_cols + 1
    grid = np.full((h, w), BG, dtype=int)

    row_lines = [0]
    cur = 0
    for size in row_heights:
        cur += 1 + size
        row_lines.append(cur)
    col_lines = [0]
    cur = 0
    for size in col_widths:
        cur += 1 + size
        col_lines.append(cur)

    grid[row_lines, :] = SEP
    grid[:, col_lines] = SEP
    for i in range(region_rows + 1):
        for j in range(region_cols + 1):
            grid[row_lines[i], col_lines[j]] = int(vertices[i, j])
    return grid


def _place_objects(grid_part: np.ndarray, template: np.ndarray) -> np.ndarray | None:
    gh, gw = grid_part.shape
    th, tw = template.shape
    gap_r = random.randint(2, 3)
    gap_c = random.randint(2, 3)
    pad_top = random.randint(1, 2)
    pad_bottom = random.randint(1, 2)
    pad_left = random.randint(1, 2)
    pad_right = random.randint(1, 2)

    canvas_h = gh + th + gap_r + pad_top + pad_bottom
    canvas_w = gw + tw + gap_c + pad_left + pad_right
    if canvas_h > 30 or canvas_w > 30:
        return None

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    quadrant = random.choice(("tl", "tr", "bl", "br"))
    if quadrant == "tl":
        tr, tc = pad_top, pad_left
        gr, gc = pad_top + th + gap_r, pad_left + tw + gap_c
    elif quadrant == "tr":
        tr, tc = pad_top, pad_left + gw + gap_c
        gr, gc = pad_top + th + gap_r, pad_left
    elif quadrant == "bl":
        tr, tc = pad_top + gh + gap_r, pad_left
        gr, gc = pad_top, pad_left + tw + gap_c
    else:
        tr, tc = pad_top + gh + gap_r, pad_left + gw + gap_c
        gr, gc = pad_top, pad_left

    canvas[gr : gr + gh, gc : gc + gw] = grid_part
    canvas[tr : tr + th, tc : tc + tw] = template
    return canvas


def _generate_case() -> np.ndarray | None:
    region_rows = random.randint(3, 4)
    region_cols = random.randint(3, 4)
    row_heights = [random.randint(2, 3) for _ in range(region_rows)]
    col_widths = [random.randint(2, 3) for _ in range(region_cols)]

    vertices = _build_vertices(region_rows, region_cols)
    if vertices is None:
        return None

    template = _build_template(region_rows, region_cols)
    if template is None:
        return None

    grid_part = _render_grid(region_rows, region_cols, row_heights, col_widths, vertices)
    canvas = _place_objects(grid_part, template)
    if canvas is None:
        return None

    try:
        details = _solve_details(canvas)
    except Exception:
        return None

    valid_mask = details["valid_mask"]
    rotated = details["rotated"]
    classes = [cls for row in details["classes"] for cls in row]
    valid_colors = rotated[valid_mask]
    if valid_colors.size < 3 or len(set(map(int, valid_colors.flat))) < 2:
        return None
    if not np.any(~valid_mask):
        return None

    if template.shape[0] == template.shape[1]:
        if np.array_equal(rotated[valid_mask], template[valid_mask]):
            return None

    out = details["output"]
    if np.array_equal(out, canvas):
        return None

    gr0, gc0, gr1, gc1 = details["grid_box"]
    grid_out = out[gr0 : gr1 + 1, gc0 : gc1 + 1]
    if np.count_nonzero(grid_out == BG) == 0:
        return None

    if classes.count("valid") < 3 or classes.count("mixed") < 1 or classes.count("missing") < 1:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _generate_case()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
